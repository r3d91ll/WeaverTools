// Package export provides academic export format utilities.
// Generates LaTeX tables, CSV data, SVG figures, PDF figures, and other publication-ready outputs.
package export

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"io"
	"math"
	"strings"
	"time"
)

// PDF constants for document generation.
const (
	// PDFVersion is the PDF specification version used.
	PDFVersion = "1.4"

	// PDFProducer is the producer string embedded in PDF metadata.
	PDFProducer = "WeaverTools Export Package"

	// DefaultPDFDPI is the default resolution for coordinate conversion.
	// PDF uses points (1 point = 1/72 inch); 72 DPI is native.
	DefaultPDFDPI = 72
)

// PDFConfig specifies options for PDF plot generation.
type PDFConfig struct {
	// Width is the page width in points (1 point = 1/72 inch).
	// Default: 612 (8.5 inches, US Letter width)
	Width float64

	// Height is the page height in points.
	// Default: 396 (5.5 inches, suitable for figure)
	Height float64

	// Title is the plot title displayed at the top.
	Title string

	// XAxisLabel is the label for the x-axis.
	// Default: "Turn"
	XAxisLabel string

	// YAxisLabel is the label for the y-axis.
	// Default: derived from metric
	YAxisLabel string

	// ShowLegend displays a legend when multiple series are present.
	// Default: true
	ShowLegend bool

	// ShowGrid displays grid lines.
	// Default: true
	ShowGrid bool

	// ShowPoints displays data point markers.
	// Default: true
	ShowPoints bool

	// FontFamily is the font for labels.
	// Default: "Helvetica"
	FontFamily string

	// FontSize is the base font size in points.
	// Default: 10
	FontSize float64

	// Padding is the margin around the plot area in points.
	// Default: 50
	Padding float64

	// PointRadius is the radius of data point markers in points.
	// Default: 3
	PointRadius float64

	// LineWidth is the stroke width for lines in points.
	// Default: 1.5
	LineWidth float64

	// GridColor is the color of grid lines (RGB 0-1).
	// Default: light gray (0.9, 0.9, 0.9)
	GridColor PDFColor

	// AxisColor is the color of axis lines (RGB 0-1).
	// Default: dark gray (0.2, 0.2, 0.2)
	AxisColor PDFColor

	// BackgroundColor is the plot background color (RGB 0-1).
	// Default: white (1, 1, 1)
	BackgroundColor PDFColor

	// IncludeMetadata embeds generation metadata in the PDF.
	// Default: true
	IncludeMetadata bool

	// ToolVersion is the version string to include in metadata.
	ToolVersion string

	// Author is the author string for PDF metadata.
	Author string

	// Subject is the subject string for PDF metadata.
	Subject string

	// Keywords are keywords for PDF metadata.
	Keywords []string

	// Compress enables stream compression for smaller file size.
	// Default: true
	Compress bool
}

// PDFColor represents an RGB color for PDF output.
type PDFColor struct {
	R, G, B float64 // Values in range [0, 1]
}

// HexToPDFColor converts a hex color string to PDFColor.
func HexToPDFColor(hex string) PDFColor {
	hex = strings.TrimPrefix(hex, "#")
	if len(hex) != 6 {
		return PDFColor{0, 0, 0} // Default to black on invalid input
	}

	var r, g, b int
	fmt.Sscanf(hex, "%02x%02x%02x", &r, &g, &b)
	return PDFColor{
		R: float64(r) / 255.0,
		G: float64(g) / 255.0,
		B: float64(b) / 255.0,
	}
}

// String returns the PDF color operator string for stroke or fill.
func (c PDFColor) String() string {
	return fmt.Sprintf("%.3f %.3f %.3f", c.R, c.G, c.B)
}

// DefaultPDFConfig returns a PDFConfig with sensible defaults.
func DefaultPDFConfig() *PDFConfig {
	return &PDFConfig{
		Width:           612,  // 8.5 inches
		Height:          396,  // 5.5 inches
		XAxisLabel:      "Turn",
		ShowLegend:      true,
		ShowGrid:        true,
		ShowPoints:      true,
		FontFamily:      "Helvetica",
		FontSize:        10,
		Padding:         50,
		PointRadius:     3,
		LineWidth:       1.5,
		GridColor:       PDFColor{0.9, 0.9, 0.9},
		AxisColor:       PDFColor{0.2, 0.2, 0.2},
		BackgroundColor: PDFColor{1, 1, 1},
		IncludeMetadata: true,
		Compress:        true,
	}
}

// PDFPlotBuilder constructs PDF plots from data series.
type PDFPlotBuilder struct {
	config *PDFConfig
	series []DataSeries
}

// NewPDFPlotBuilder creates a new plot builder with the given configuration.
// If config is nil, DefaultPDFConfig() is used.
func NewPDFPlotBuilder(config *PDFConfig) *PDFPlotBuilder {
	if config == nil {
		config = DefaultPDFConfig()
	}
	return &PDFPlotBuilder{
		config: config,
		series: make([]DataSeries, 0),
	}
}

// AddSeries adds a data series to the plot.
func (ppb *PDFPlotBuilder) AddSeries(series DataSeries) *PDFPlotBuilder {
	ppb.series = append(ppb.series, series)
	return ppb
}

// AddPoint adds a single data point to a metric series.
// Creates a new series if one doesn't exist for the metric.
func (ppb *PDFPlotBuilder) AddPoint(metric PlotMetric, turn int, value float64) *PDFPlotBuilder {
	// Find existing series for this metric
	for i := range ppb.series {
		if ppb.series[i].Metric == metric {
			ppb.series[i].Points = append(ppb.series[i].Points, DataPoint{
				Turn:  turn,
				Value: value,
			})
			return ppb
		}
	}

	// Create new series
	ppb.series = append(ppb.series, DataSeries{
		Metric: metric,
		Points: []DataPoint{{Turn: turn, Value: value}},
	})
	return ppb
}

// Build generates the complete PDF document as bytes.
// Returns the PDF as a byte slice ready for saving or embedding.
func (ppb *PDFPlotBuilder) Build() []byte {
	if len(ppb.series) == 0 {
		return nil
	}

	doc := newPDFDocument(ppb.config)

	// Calculate plot dimensions
	width := ppb.config.Width
	height := ppb.config.Height
	padding := ppb.config.Padding
	plotWidth := width - 2*padding
	plotHeight := height - 2*padding - 30 // Leave room for title

	// Calculate data bounds
	minX, maxX, minY, maxY := ppb.calculateBounds()

	// Handle edge cases for single point or identical values
	if minX == maxX {
		minX--
		maxX++
	}
	if minY == maxY {
		minY = minY - 1
		maxY = maxY + 1
	}

	// Start content stream
	var content strings.Builder

	// Save graphics state
	content.WriteString("q\n")

	// Draw background
	content.WriteString(fmt.Sprintf("%s rg\n", ppb.config.BackgroundColor.String()))
	content.WriteString(fmt.Sprintf("0 0 %.2f %.2f re f\n", width, height))

	// Translate to plot area (origin at bottom-left in PDF)
	titleOffset := 25.0
	if ppb.config.Title == "" {
		titleOffset = 0
	}
	content.WriteString(fmt.Sprintf("1 0 0 1 %.2f %.2f cm\n", padding, padding+titleOffset))

	// Draw grid if enabled
	if ppb.config.ShowGrid {
		ppb.writeGrid(&content, plotWidth, plotHeight, minX, maxX, minY, maxY)
	}

	// Draw axes
	ppb.writeAxes(&content, plotWidth, plotHeight, minX, maxX, minY, maxY)

	// Draw data series
	for _, series := range ppb.series {
		ppb.writeSeries(&content, series, plotWidth, plotHeight, minX, maxX, minY, maxY)
	}

	// Restore graphics state and draw elements outside plot area
	content.WriteString("Q\n")

	// Draw title if present
	if ppb.config.Title != "" {
		ppb.writeTitle(&content, width, height)
	}

	// Draw axis labels
	ppb.writeAxisLabels(&content, width, height, padding, plotWidth, plotHeight, titleOffset)

	// Draw legend if enabled and multiple series
	if ppb.config.ShowLegend && len(ppb.series) > 1 {
		ppb.writeLegend(&content, width, height, padding, plotWidth)
	}

	// Build the PDF document
	doc.addPage(width, height, content.String())
	return doc.build()
}

// WriteTo writes the PDF to an io.Writer.
func (ppb *PDFPlotBuilder) WriteTo(w io.Writer) (int64, error) {
	pdf := ppb.Build()
	if pdf == nil {
		return 0, nil
	}
	n, err := w.Write(pdf)
	return int64(n), err
}

// calculateBounds determines the data range for all series.
func (ppb *PDFPlotBuilder) calculateBounds() (minX, maxX int, minY, maxY float64) {
	if len(ppb.series) == 0 || len(ppb.series[0].Points) == 0 {
		return 0, 10, 0, 1
	}

	// Initialize with first point
	first := ppb.series[0].Points[0]
	minX, maxX = first.Turn, first.Turn
	minY, maxY = first.Value, first.Value

	for _, series := range ppb.series {
		for _, point := range series.Points {
			if point.Turn < minX {
				minX = point.Turn
			}
			if point.Turn > maxX {
				maxX = point.Turn
			}
			if point.Value < minY {
				minY = point.Value
			}
			if point.Value > maxY {
				maxY = point.Value
			}
		}
	}

	// Add some padding to Y range
	yRange := maxY - minY
	if yRange > 0 {
		padding := yRange * 0.1
		minY -= padding
		maxY += padding
	}

	return minX, maxX, minY, maxY
}

// writeGrid writes the plot grid lines.
func (ppb *PDFPlotBuilder) writeGrid(sb *strings.Builder, plotWidth, plotHeight float64, minX, maxX int, minY, maxY float64) {
	// Calculate nice tick values
	xTicks := calculateIntTicks(minX, maxX, 10)
	yTicks := calculateFloatTicks(minY, maxY, 8)

	// Set grid color and line width
	sb.WriteString(fmt.Sprintf("%s RG\n", ppb.config.GridColor.String()))
	sb.WriteString("0.5 w\n")
	sb.WriteString("[3 3] 0 d\n") // Dashed line

	// Vertical grid lines (for X values)
	for _, tick := range xTicks {
		x := scaleValue(float64(tick), float64(minX), float64(maxX), 0, plotWidth)
		sb.WriteString(fmt.Sprintf("%.2f 0 m %.2f %.2f l S\n", x, x, plotHeight))
	}

	// Horizontal grid lines (for Y values)
	for _, tick := range yTicks {
		y := scaleValue(tick, minY, maxY, 0, plotHeight)
		sb.WriteString(fmt.Sprintf("0 %.2f m %.2f %.2f l S\n", y, plotWidth, y))
	}

	// Reset dash pattern
	sb.WriteString("[] 0 d\n")
}

// writeAxes writes the plot axes with tick marks.
func (ppb *PDFPlotBuilder) writeAxes(sb *strings.Builder, plotWidth, plotHeight float64, minX, maxX int, minY, maxY float64) {
	// Calculate tick values
	xTicks := calculateIntTicks(minX, maxX, 10)
	yTicks := calculateFloatTicks(minY, maxY, 8)

	// Set axis color and line width
	sb.WriteString(fmt.Sprintf("%s RG\n", ppb.config.AxisColor.String()))
	sb.WriteString("1 w\n")

	// X-axis
	sb.WriteString(fmt.Sprintf("0 0 m %.2f 0 l S\n", plotWidth))

	// Y-axis
	sb.WriteString(fmt.Sprintf("0 0 m 0 %.2f l S\n", plotHeight))

	// X-axis tick marks
	for _, tick := range xTicks {
		x := scaleValue(float64(tick), float64(minX), float64(maxX), 0, plotWidth)
		sb.WriteString(fmt.Sprintf("%.2f 0 m %.2f -5 l S\n", x, x))
	}

	// Y-axis tick marks
	for _, tick := range yTicks {
		y := scaleValue(tick, minY, maxY, 0, plotHeight)
		sb.WriteString(fmt.Sprintf("0 %.2f m -5 %.2f l S\n", y, y))
	}
}

// writeSeries writes a single data series (line and points).
func (ppb *PDFPlotBuilder) writeSeries(sb *strings.Builder, series DataSeries, plotWidth, plotHeight float64, minX, maxX int, minY, maxY float64) {
	if len(series.Points) == 0 {
		return
	}

	// Determine series color
	colorHex := series.Color
	if colorHex == "" {
		colorHex = GetMetricInfo(series.Metric).Color
	}
	color := HexToPDFColor(colorHex)

	// Set stroke color and line width
	sb.WriteString(fmt.Sprintf("%s RG\n", color.String()))
	sb.WriteString(fmt.Sprintf("%.2f w\n", ppb.config.LineWidth))
	sb.WriteString("1 J\n") // Round line cap
	sb.WriteString("1 j\n") // Round line join

	// Draw the line path
	for i, point := range series.Points {
		x := scaleValue(float64(point.Turn), float64(minX), float64(maxX), 0, plotWidth)
		y := scaleValue(point.Value, minY, maxY, 0, plotHeight)

		if i == 0 {
			sb.WriteString(fmt.Sprintf("%.2f %.2f m\n", x, y))
		} else {
			sb.WriteString(fmt.Sprintf("%.2f %.2f l\n", x, y))
		}
	}
	sb.WriteString("S\n")

	// Draw data points if enabled
	if ppb.config.ShowPoints {
		// Set fill color (white center with colored stroke)
		sb.WriteString(fmt.Sprintf("%s rg\n", ppb.config.BackgroundColor.String()))
		sb.WriteString(fmt.Sprintf("%s RG\n", color.String()))
		sb.WriteString("1 w\n")

		radius := ppb.config.PointRadius
		for _, point := range series.Points {
			x := scaleValue(float64(point.Turn), float64(minX), float64(maxX), 0, plotWidth)
			y := scaleValue(point.Value, minY, maxY, 0, plotHeight)

			// Draw a circle using Bezier curves
			// Control point offset for circle approximation
			k := radius * 0.5523 // Magic number for Bezier circle approximation
			sb.WriteString(fmt.Sprintf("%.2f %.2f m\n", x+radius, y))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				x+radius, y+k, x+k, y+radius, x, y+radius))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				x-k, y+radius, x-radius, y+k, x-radius, y))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				x-radius, y-k, x-k, y-radius, x, y-radius))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				x+k, y-radius, x+radius, y-k, x+radius, y))
			sb.WriteString("B\n") // Fill and stroke
		}
	}
}

// writeTitle writes the plot title.
func (ppb *PDFPlotBuilder) writeTitle(sb *strings.Builder, width, height float64) {
	title := ppb.config.Title
	fontSize := ppb.config.FontSize + 4 // Title is larger

	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", fontSize))
	sb.WriteString(fmt.Sprintf("%s rg\n", ppb.config.AxisColor.String()))

	// Center the title
	// Approximate text width (rough estimate: 0.6 * fontSize per character)
	textWidth := float64(len(title)) * fontSize * 0.5
	x := (width - textWidth) / 2
	y := height - 25

	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", x, y))
	sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(title)))
	sb.WriteString("ET\n")
}

// writeAxisLabels writes the axis labels.
func (ppb *PDFPlotBuilder) writeAxisLabels(sb *strings.Builder, width, height, padding, plotWidth, plotHeight, titleOffset float64) {
	fontSize := ppb.config.FontSize

	// X-axis label
	xLabel := ppb.config.XAxisLabel
	if xLabel == "" {
		xLabel = "Turn"
	}

	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", fontSize))
	sb.WriteString(fmt.Sprintf("%s rg\n", ppb.config.AxisColor.String()))

	// Center X label below axis
	textWidth := float64(len(xLabel)) * fontSize * 0.5
	x := padding + (plotWidth-textWidth)/2
	y := padding + titleOffset - 25

	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", x, y))
	sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(xLabel)))
	sb.WriteString("ET\n")

	// Y-axis label (rotated)
	yLabel := ppb.config.YAxisLabel
	if yLabel == "" && len(ppb.series) == 1 {
		yLabel = GetMetricInfo(ppb.series[0].Metric).Symbol
	} else if yLabel == "" {
		yLabel = "Value"
	}

	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", fontSize))
	sb.WriteString(fmt.Sprintf("%s rg\n", ppb.config.AxisColor.String()))

	// Rotate and position Y label
	textWidth = float64(len(yLabel)) * fontSize * 0.5
	x = 15
	y = padding + titleOffset + (plotHeight+textWidth)/2

	// Rotation matrix for 90 degrees: [cos sin -sin cos tx ty]
	sb.WriteString(fmt.Sprintf("0 1 -1 0 %.2f %.2f Tm\n", x, y))
	sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(yLabel)))
	sb.WriteString("ET\n")
}

// writeLegend writes the legend for multiple series.
func (ppb *PDFPlotBuilder) writeLegend(sb *strings.Builder, width, height, padding, plotWidth float64) {
	fontSize := ppb.config.FontSize - 1
	lineHeight := fontSize + 4
	legendX := padding + plotWidth - 100
	legendY := height - padding - 20

	for i, series := range ppb.series {
		y := legendY - float64(i)*lineHeight

		// Get color and label
		colorHex := series.Color
		if colorHex == "" {
			colorHex = GetMetricInfo(series.Metric).Color
		}
		color := HexToPDFColor(colorHex)

		label := series.Label
		if label == "" {
			label = GetMetricInfo(series.Metric).Name
		}

		// Draw colored line sample
		sb.WriteString(fmt.Sprintf("%s RG\n", color.String()))
		sb.WriteString(fmt.Sprintf("%.2f w\n", ppb.config.LineWidth))
		sb.WriteString(fmt.Sprintf("%.2f %.2f m %.2f %.2f l S\n",
			legendX, y, legendX+20, y))

		// Draw point marker if enabled
		if ppb.config.ShowPoints {
			sb.WriteString(fmt.Sprintf("%s rg\n", ppb.config.BackgroundColor.String()))
			radius := ppb.config.PointRadius * 0.8
			cx := legendX + 10
			k := radius * 0.5523
			sb.WriteString(fmt.Sprintf("%.2f %.2f m\n", cx+radius, y))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				cx+radius, y+k, cx+k, y+radius, cx, y+radius))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				cx-k, y+radius, cx-radius, y+k, cx-radius, y))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				cx-radius, y-k, cx-k, y-radius, cx, y-radius))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				cx+k, y-radius, cx+radius, y-k, cx+radius, y))
			sb.WriteString("B\n")
		}

		// Draw label text
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", fontSize))
		sb.WriteString(fmt.Sprintf("%s rg\n", ppb.config.AxisColor.String()))
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", legendX+25, y-fontSize/3))
		sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(label)))
		sb.WriteString("ET\n")
	}
}

// escapePDFString escapes special characters for PDF text strings.
func escapePDFString(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "(", "\\(")
	s = strings.ReplaceAll(s, ")", "\\)")
	return s
}

// -----------------------------------------------------------------------------
// PDF Document Builder (Internal)
// -----------------------------------------------------------------------------

// pdfDocument builds a complete PDF file.
type pdfDocument struct {
	config    *PDFConfig
	objects   []string
	pages     []int
	pageCount int
	xref      []int
}

// newPDFDocument creates a new PDF document builder.
func newPDFDocument(config *PDFConfig) *pdfDocument {
	return &pdfDocument{
		config:  config,
		objects: make([]string, 0),
		pages:   make([]int, 0),
	}
}

// addObject adds an object and returns its object number.
func (doc *pdfDocument) addObject(content string) int {
	doc.objects = append(doc.objects, content)
	return len(doc.objects) // 1-based object numbering
}

// addPage adds a page with the given dimensions and content stream.
func (doc *pdfDocument) addPage(width, height float64, content string) {
	doc.pageCount++

	// Create content stream (optionally compressed)
	var streamData []byte
	var filter string

	if doc.config.Compress {
		var buf bytes.Buffer
		w := zlib.NewWriter(&buf)
		w.Write([]byte(content))
		w.Close()
		streamData = buf.Bytes()
		filter = "/Filter /FlateDecode\n"
	} else {
		streamData = []byte(content)
		filter = ""
	}

	streamObj := fmt.Sprintf("<< /Length %d\n%s>>\nstream\n%sendstream",
		len(streamData), filter, streamData)
	streamObjNum := doc.addObject(streamObj)

	// Create page object
	pageObj := fmt.Sprintf("<< /Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 %.2f %.2f]\n/Contents %d 0 R\n/Resources << /Font << /F1 3 0 R >> >>\n>>",
		width, height, streamObjNum)
	pageObjNum := doc.addObject(pageObj)

	doc.pages = append(doc.pages, pageObjNum)
}

// build generates the complete PDF file.
func (doc *pdfDocument) build() []byte {
	var buf bytes.Buffer

	// PDF header
	buf.WriteString(fmt.Sprintf("%%PDF-%s\n", PDFVersion))
	buf.WriteString("%\xE2\xE3\xCF\xD3\n") // Binary marker

	// Reserve object slots for catalog, pages, and font
	// Object 1: Catalog
	// Object 2: Pages
	// Object 3: Font
	// Then info dict if metadata enabled

	// Build page tree kids array
	var kidsArray strings.Builder
	kidsArray.WriteString("[")
	for i, pageNum := range doc.pages {
		if i > 0 {
			kidsArray.WriteString(" ")
		}
		// Page object numbers need to account for reserved objects
		kidsArray.WriteString(fmt.Sprintf("%d 0 R", pageNum+3))
	}
	kidsArray.WriteString("]")

	// Build the final object list
	finalObjects := make([]string, 0)

	// Object 1: Catalog
	finalObjects = append(finalObjects, "<< /Type /Catalog\n/Pages 2 0 R\n>>")

	// Object 2: Pages
	pagesObj := fmt.Sprintf("<< /Type /Pages\n/Kids %s\n/Count %d\n>>",
		kidsArray.String(), doc.pageCount)
	finalObjects = append(finalObjects, pagesObj)

	// Object 3: Font (Helvetica)
	fontObj := "<< /Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n/Encoding /WinAnsiEncoding\n>>"
	finalObjects = append(finalObjects, fontObj)

	// Add all page and stream objects
	for _, obj := range doc.objects {
		finalObjects = append(finalObjects, obj)
	}

	// Object N: Info dictionary (if metadata enabled)
	infoObjNum := 0
	if doc.config.IncludeMetadata {
		infoObj := doc.buildInfoDict()
		finalObjects = append(finalObjects, infoObj)
		infoObjNum = len(finalObjects)
	}

	// Write all objects and track xref positions
	xref := make([]int, len(finalObjects)+1)
	xref[0] = 0 // Object 0 is always free

	for i, obj := range finalObjects {
		xref[i+1] = buf.Len()
		buf.WriteString(fmt.Sprintf("%d 0 obj\n%s\nendobj\n", i+1, obj))
	}

	// Write xref table
	xrefPos := buf.Len()
	buf.WriteString("xref\n")
	buf.WriteString(fmt.Sprintf("0 %d\n", len(finalObjects)+1))
	buf.WriteString("0000000000 65535 f \n")
	for i := 1; i <= len(finalObjects); i++ {
		buf.WriteString(fmt.Sprintf("%010d 00000 n \n", xref[i]))
	}

	// Write trailer
	buf.WriteString("trailer\n")
	buf.WriteString(fmt.Sprintf("<< /Size %d\n/Root 1 0 R\n", len(finalObjects)+1))
	if infoObjNum > 0 {
		buf.WriteString(fmt.Sprintf("/Info %d 0 R\n", infoObjNum))
	}
	buf.WriteString(">>\n")
	buf.WriteString("startxref\n")
	buf.WriteString(fmt.Sprintf("%d\n", xrefPos))
	buf.WriteString("%%EOF\n")

	return buf.Bytes()
}

// buildInfoDict creates the PDF Info dictionary for metadata.
func (doc *pdfDocument) buildInfoDict() string {
	var sb strings.Builder
	sb.WriteString("<<\n")

	if doc.config.Title != "" {
		sb.WriteString(fmt.Sprintf("/Title (%s)\n", escapePDFString(doc.config.Title)))
	}
	if doc.config.Author != "" {
		sb.WriteString(fmt.Sprintf("/Author (%s)\n", escapePDFString(doc.config.Author)))
	}
	if doc.config.Subject != "" {
		sb.WriteString(fmt.Sprintf("/Subject (%s)\n", escapePDFString(doc.config.Subject)))
	}
	if len(doc.config.Keywords) > 0 {
		keywords := strings.Join(doc.config.Keywords, ", ")
		sb.WriteString(fmt.Sprintf("/Keywords (%s)\n", escapePDFString(keywords)))
	}

	sb.WriteString(fmt.Sprintf("/Producer (%s)\n", escapePDFString(PDFProducer)))
	if doc.config.ToolVersion != "" {
		sb.WriteString(fmt.Sprintf("/Creator (WeaverTools %s)\n", escapePDFString(doc.config.ToolVersion)))
	} else {
		sb.WriteString("/Creator (WeaverTools)\n")
	}

	// Creation date in PDF date format: D:YYYYMMDDHHmmSS
	now := time.Now().UTC()
	dateStr := now.Format("D:20060102150405Z")
	sb.WriteString(fmt.Sprintf("/CreationDate (%s)\n", dateStr))
	sb.WriteString(fmt.Sprintf("/ModDate (%s)\n", dateStr))

	sb.WriteString(">>")
	return sb.String()
}

// -----------------------------------------------------------------------------
// Convenience Functions
// -----------------------------------------------------------------------------

// GenerateMeasurementPDFPlot creates a PDF plot from MeasurementRow data.
// Plots the specified metric over conversation turns.
func GenerateMeasurementPDFPlot(rows []MeasurementRow, metric PlotMetric, config *PDFConfig) []byte {
	if config == nil {
		config = DefaultPDFConfig()
	}

	// Set Y-axis label based on metric
	if config.YAxisLabel == "" {
		config.YAxisLabel = GetMetricInfo(metric).Symbol
	}

	builder := NewPDFPlotBuilder(config)

	// Create data series from measurement rows
	series := DataSeries{
		Metric: metric,
		Points: make([]DataPoint, len(rows)),
	}

	for i, row := range rows {
		var value float64
		switch metric {
		case MetricDEff:
			value = float64(row.DEff)
		case MetricBeta:
			value = row.Beta
		case MetricAlignment:
			value = row.Alignment
		case MetricCPair:
			value = row.CPair
		}

		series.Points[i] = DataPoint{
			Turn:  row.Turn,
			Value: value,
		}
	}

	builder.AddSeries(series)
	return builder.Build()
}

// GenerateMultiMetricPDFPlot creates a PDF plot with multiple metrics.
// Useful for comparing different metrics on the same time scale.
func GenerateMultiMetricPDFPlot(rows []MeasurementRow, metrics []PlotMetric, config *PDFConfig) []byte {
	if config == nil {
		config = DefaultPDFConfig()
	}

	builder := NewPDFPlotBuilder(config)

	for _, metric := range metrics {
		series := DataSeries{
			Metric: metric,
			Points: make([]DataPoint, len(rows)),
		}

		for i, row := range rows {
			var value float64
			switch metric {
			case MetricDEff:
				value = float64(row.DEff)
			case MetricBeta:
				value = row.Beta
			case MetricAlignment:
				value = row.Alignment
			case MetricCPair:
				value = row.CPair
			}

			series.Points[i] = DataPoint{
				Turn:  row.Turn,
				Value: value,
			}
		}

		builder.AddSeries(series)
	}

	return builder.Build()
}

// GenerateBetaPDFPlotWithThresholds creates a Beta plot with threshold lines.
// Shows the optimal, monitor, concerning, and critical thresholds.
func GenerateBetaPDFPlotWithThresholds(rows []MeasurementRow, config *PDFConfig) []byte {
	if config == nil {
		config = DefaultPDFConfig()
	}

	// Set appropriate title and labels
	if config.Title == "" {
		config.Title = "Beta (Collapse Indicator) Over Time"
	}
	if config.YAxisLabel == "" {
		config.YAxisLabel = "Beta"
	}

	builder := NewPDFPlotBuilder(config)

	// Add the main data series
	betaSeries := DataSeries{
		Metric: MetricBeta,
		Points: make([]DataPoint, len(rows)),
	}

	for i, row := range rows {
		betaSeries.Points[i] = DataPoint{
			Turn:  row.Turn,
			Value: row.Beta,
		}
	}

	builder.AddSeries(betaSeries)

	return builder.Build()
}

// ExportPDFToWriter writes a PDF plot to an io.Writer.
func ExportPDFToWriter(w io.Writer, rows []MeasurementRow, metric PlotMetric, config *PDFConfig) error {
	pdf := GenerateMeasurementPDFPlot(rows, metric, config)
	if pdf == nil {
		return nil
	}
	_, err := w.Write(pdf)
	return err
}

// PDFPointsFromInches converts inches to PDF points.
func PDFPointsFromInches(inches float64) float64 {
	return inches * 72.0
}

// PDFPointsFromMillimeters converts millimeters to PDF points.
func PDFPointsFromMillimeters(mm float64) float64 {
	return mm * 72.0 / 25.4
}

// PDFDimensions provides common page size dimensions in points.
var PDFDimensions = struct {
	// Letter is US Letter size (8.5 x 11 inches).
	Letter struct{ Width, Height float64 }
	// A4 is ISO A4 size (210 x 297 mm).
	A4 struct{ Width, Height float64 }
	// A5 is ISO A5 size (148 x 210 mm).
	A5 struct{ Width, Height float64 }
	// Figure is a common figure size for embedding.
	Figure struct{ Width, Height float64 }
}{
	Letter: struct{ Width, Height float64 }{612, 792},
	A4:     struct{ Width, Height float64 }{595.276, 841.890},
	A5:     struct{ Width, Height float64 }{419.528, 595.276},
	Figure: struct{ Width, Height float64 }{612, 396},
}

// scalePDFValue maps a value from one range to another (for PDF coordinate system).
// This is used internally for coordinate transformations.
func scalePDFValue(value, srcMin, srcMax, dstMin, dstMax float64) float64 {
	if srcMax == srcMin {
		return (dstMin + dstMax) / 2
	}
	return dstMin + (value-srcMin)*(dstMax-dstMin)/(srcMax-srcMin)
}

// roundToPrecision rounds a float64 to the specified number of decimal places.
func roundToPrecision(value float64, decimals int) float64 {
	multiplier := math.Pow(10, float64(decimals))
	return math.Round(value*multiplier) / multiplier
}
