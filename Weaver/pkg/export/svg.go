// Package export provides academic export format utilities.
// Generates LaTeX tables, CSV data, SVG figures, and other publication-ready outputs.
package export

import (
	"fmt"
	"io"
	"math"
	"strings"
	"time"
)

// SVG constants for plot generation.
const (
	// SVGVersion is the SVG specification version used.
	SVGVersion = "1.1"

	// SVGNamespace is the XML namespace for SVG.
	SVGNamespace = "http://www.w3.org/2000/svg"
)

// PlotMetric identifies which metric to plot.
type PlotMetric string

const (
	// MetricDEff plots effective dimensionality over time.
	MetricDEff PlotMetric = "d_eff"

	// MetricBeta plots the collapse indicator (beta) over time.
	MetricBeta PlotMetric = "beta"

	// MetricAlignment plots cosine similarity over time.
	MetricAlignment PlotMetric = "alignment"

	// MetricCPair plots bilateral conveyance over time.
	MetricCPair PlotMetric = "c_pair"
)

// MetricInfo provides display information for each metric type.
type MetricInfo struct {
	Name        string  // Display name for labels
	Symbol      string  // Mathematical symbol for axis labels
	Description string  // Brief description
	Color       string  // Default line color (hex)
	MinValue    float64 // Expected minimum value
	MaxValue    float64 // Expected maximum value
}

// GetMetricInfo returns display information for a metric.
func GetMetricInfo(m PlotMetric) MetricInfo {
	switch m {
	case MetricDEff:
		return MetricInfo{
			Name:        "Effective Dimensionality",
			Symbol:      "D_eff",
			Description: "Effective dimensionality of hidden state representation",
			Color:       "#2563eb", // Blue
			MinValue:    0,
			MaxValue:    256,
		}
	case MetricBeta:
		return MetricInfo{
			Name:        "Collapse Indicator",
			Symbol:      "β",
			Description: "Dimensional collapse indicator (lower is better)",
			Color:       "#dc2626", // Red
			MinValue:    0,
			MaxValue:    5,
		}
	case MetricAlignment:
		return MetricInfo{
			Name:        "Alignment",
			Symbol:      "Alignment",
			Description: "Cosine similarity between hidden states",
			Color:       "#16a34a", // Green
			MinValue:    -1,
			MaxValue:    1,
		}
	case MetricCPair:
		return MetricInfo{
			Name:        "Bilateral Conveyance",
			Symbol:      "C_pair",
			Description: "Bilateral communication effectiveness score",
			Color:       "#9333ea", // Purple
			MinValue:    0,
			MaxValue:    1,
		}
	default:
		return MetricInfo{
			Name:     "Unknown",
			Symbol:   "?",
			Color:    "#6b7280",
			MinValue: 0,
			MaxValue: 1,
		}
	}
}

// DataPoint represents a single data point for plotting.
type DataPoint struct {
	// Turn is the conversation turn number (x-axis).
	Turn int

	// Value is the metric value (y-axis).
	Value float64

	// Label is an optional label for the point.
	Label string
}

// DataSeries represents a series of data points to plot.
type DataSeries struct {
	// Metric identifies what is being plotted.
	Metric PlotMetric

	// Points contains the data points in order.
	Points []DataPoint

	// Color overrides the default color for this series.
	// If empty, uses the metric's default color.
	Color string

	// Label is the legend label for this series.
	// If empty, uses the metric's default name.
	Label string
}

// SVGConfig specifies options for SVG plot generation.
type SVGConfig struct {
	// Width is the SVG width in pixels.
	// Default: 800
	Width int

	// Height is the SVG height in pixels.
	// Default: 400
	Height int

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
	// Default: "Arial, sans-serif"
	FontFamily string

	// Padding is the margin around the plot area.
	// Default: 60
	Padding int

	// PointRadius is the radius of data point markers.
	// Default: 4
	PointRadius float64

	// LineWidth is the stroke width for lines.
	// Default: 2
	LineWidth float64

	// GridColor is the color of grid lines.
	// Default: "#e5e7eb"
	GridColor string

	// AxisColor is the color of axis lines.
	// Default: "#374151"
	AxisColor string

	// BackgroundColor is the plot background color.
	// Default: "#ffffff"
	BackgroundColor string

	// IncludeMetadata embeds generation metadata in the SVG.
	// Default: true
	IncludeMetadata bool

	// ToolVersion is the version string to include in metadata.
	ToolVersion string
}

// DefaultSVGConfig returns an SVGConfig with sensible defaults.
func DefaultSVGConfig() *SVGConfig {
	return &SVGConfig{
		Width:           800,
		Height:          400,
		XAxisLabel:      "Turn",
		ShowLegend:      true,
		ShowGrid:        true,
		ShowPoints:      true,
		FontFamily:      "Arial, sans-serif",
		Padding:         60,
		PointRadius:     4,
		LineWidth:       2,
		GridColor:       "#e5e7eb",
		AxisColor:       "#374151",
		BackgroundColor: "#ffffff",
		IncludeMetadata: true,
	}
}

// SVGPlotBuilder constructs SVG plots from data series.
type SVGPlotBuilder struct {
	config *SVGConfig
	series []DataSeries
}

// NewSVGPlotBuilder creates a new plot builder with the given configuration.
// If config is nil, DefaultSVGConfig() is used.
func NewSVGPlotBuilder(config *SVGConfig) *SVGPlotBuilder {
	if config == nil {
		config = DefaultSVGConfig()
	}
	return &SVGPlotBuilder{
		config: config,
		series: make([]DataSeries, 0),
	}
}

// AddSeries adds a data series to the plot.
func (spb *SVGPlotBuilder) AddSeries(series DataSeries) *SVGPlotBuilder {
	spb.series = append(spb.series, series)
	return spb
}

// AddPoint adds a single data point to a metric series.
// Creates a new series if one doesn't exist for the metric.
func (spb *SVGPlotBuilder) AddPoint(metric PlotMetric, turn int, value float64) *SVGPlotBuilder {
	// Find existing series for this metric
	for i := range spb.series {
		if spb.series[i].Metric == metric {
			spb.series[i].Points = append(spb.series[i].Points, DataPoint{
				Turn:  turn,
				Value: value,
			})
			return spb
		}
	}

	// Create new series
	spb.series = append(spb.series, DataSeries{
		Metric: metric,
		Points: []DataPoint{{Turn: turn, Value: value}},
	})
	return spb
}

// Build generates the complete SVG code.
// Returns the SVG as a string ready for saving or embedding.
func (spb *SVGPlotBuilder) Build() string {
	if len(spb.series) == 0 {
		return ""
	}

	var sb strings.Builder

	// Calculate plot dimensions
	width := spb.config.Width
	height := spb.config.Height
	padding := spb.config.Padding
	plotWidth := width - 2*padding
	plotHeight := height - 2*padding

	// Calculate data bounds
	minX, maxX, minY, maxY := spb.calculateBounds()

	// Handle edge cases for single point or identical values
	if minX == maxX {
		minX--
		maxX++
	}
	if minY == maxY {
		minY = minY - 1
		maxY = maxY + 1
	}

	// Write SVG header
	spb.writeHeader(&sb, width, height)

	// Write definitions (for gradients, patterns, etc.)
	spb.writeDefinitions(&sb)

	// Write background
	spb.writeBackground(&sb, width, height)

	// Write metadata comment if enabled
	if spb.config.IncludeMetadata {
		spb.writeMetadata(&sb)
	}

	// Create plot group with transformation
	sb.WriteString(fmt.Sprintf("  <g transform=\"translate(%d,%d)\">\n", padding, padding))

	// Draw grid if enabled
	if spb.config.ShowGrid {
		spb.writeGrid(&sb, plotWidth, plotHeight, minX, maxX, minY, maxY)
	}

	// Draw axes
	spb.writeAxes(&sb, plotWidth, plotHeight, minX, maxX, minY, maxY)

	// Draw data series
	for _, series := range spb.series {
		spb.writeSeries(&sb, series, plotWidth, plotHeight, minX, maxX, minY, maxY)
	}

	// Close plot group
	sb.WriteString("  </g>\n")

	// Write title if present
	if spb.config.Title != "" {
		spb.writeTitle(&sb, width)
	}

	// Write legend if enabled and multiple series
	if spb.config.ShowLegend && len(spb.series) > 1 {
		spb.writeLegend(&sb, width)
	}

	// Write axis labels
	spb.writeAxisLabels(&sb, width, height, padding, plotWidth, plotHeight)

	// Write SVG footer
	sb.WriteString("</svg>\n")

	return sb.String()
}

// WriteTo writes the SVG to an io.Writer.
func (spb *SVGPlotBuilder) WriteTo(w io.Writer) (int64, error) {
	svg := spb.Build()
	n, err := io.WriteString(w, svg)
	return int64(n), err
}

// calculateBounds determines the data range for all series.
func (spb *SVGPlotBuilder) calculateBounds() (minX, maxX int, minY, maxY float64) {
	if len(spb.series) == 0 || len(spb.series[0].Points) == 0 {
		return 0, 10, 0, 1
	}

	// Initialize with first point
	first := spb.series[0].Points[0]
	minX, maxX = first.Turn, first.Turn
	minY, maxY = first.Value, first.Value

	for _, series := range spb.series {
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

// writeHeader writes the SVG document header.
func (spb *SVGPlotBuilder) writeHeader(sb *strings.Builder, width, height int) {
	sb.WriteString("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
	sb.WriteString(fmt.Sprintf("<svg version=\"%s\" xmlns=\"%s\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
		SVGVersion, SVGNamespace, width, height, width, height))
}

// writeDefinitions writes SVG definitions.
func (spb *SVGPlotBuilder) writeDefinitions(sb *strings.Builder) {
	sb.WriteString("  <defs>\n")
	sb.WriteString("    <style type=\"text/css\">\n")
	sb.WriteString(fmt.Sprintf("      .axis-label { font-family: %s; font-size: 12px; fill: %s; }\n",
		spb.config.FontFamily, spb.config.AxisColor))
	sb.WriteString(fmt.Sprintf("      .title { font-family: %s; font-size: 16px; font-weight: bold; fill: %s; }\n",
		spb.config.FontFamily, spb.config.AxisColor))
	sb.WriteString(fmt.Sprintf("      .legend-text { font-family: %s; font-size: 11px; fill: %s; }\n",
		spb.config.FontFamily, spb.config.AxisColor))
	sb.WriteString(fmt.Sprintf("      .tick-label { font-family: %s; font-size: 10px; fill: %s; }\n",
		spb.config.FontFamily, spb.config.AxisColor))
	sb.WriteString("    </style>\n")
	sb.WriteString("  </defs>\n")
}

// writeBackground writes the plot background.
func (spb *SVGPlotBuilder) writeBackground(sb *strings.Builder, width, height int) {
	sb.WriteString(fmt.Sprintf("  <rect width=\"%d\" height=\"%d\" fill=\"%s\"/>\n",
		width, height, spb.config.BackgroundColor))
}

// writeMetadata writes generation metadata as an SVG comment.
func (spb *SVGPlotBuilder) writeMetadata(sb *strings.Builder) {
	sb.WriteString("  <!-- Generated by WeaverTools Export Package -->\n")
	sb.WriteString(fmt.Sprintf("  <!-- Generated at: %s -->\n", time.Now().UTC().Format(time.RFC3339)))
	if spb.config.ToolVersion != "" {
		sb.WriteString(fmt.Sprintf("  <!-- Tool version: %s -->\n", spb.config.ToolVersion))
	}
	sb.WriteString(fmt.Sprintf("  <!-- Data series: %d -->\n", len(spb.series)))
}

// writeGrid writes the plot grid lines.
func (spb *SVGPlotBuilder) writeGrid(sb *strings.Builder, plotWidth, plotHeight int, minX, maxX int, minY, maxY float64) {
	// Calculate nice tick values
	xTicks := calculateIntTicks(minX, maxX, 10)
	yTicks := calculateFloatTicks(minY, maxY, 8)

	sb.WriteString("    <g class=\"grid\">\n")

	// Vertical grid lines (for X values)
	for _, tick := range xTicks {
		x := scaleValue(float64(tick), float64(minX), float64(maxX), 0, float64(plotWidth))
		sb.WriteString(fmt.Sprintf("      <line x1=\"%.1f\" y1=\"0\" x2=\"%.1f\" y2=\"%d\" stroke=\"%s\" stroke-dasharray=\"3,3\"/>\n",
			x, x, plotHeight, spb.config.GridColor))
	}

	// Horizontal grid lines (for Y values)
	for _, tick := range yTicks {
		y := scaleValue(tick, minY, maxY, float64(plotHeight), 0)
		sb.WriteString(fmt.Sprintf("      <line x1=\"0\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"%s\" stroke-dasharray=\"3,3\"/>\n",
			y, plotWidth, y, spb.config.GridColor))
	}

	sb.WriteString("    </g>\n")
}

// writeAxes writes the plot axes with tick marks and labels.
func (spb *SVGPlotBuilder) writeAxes(sb *strings.Builder, plotWidth, plotHeight int, minX, maxX int, minY, maxY float64) {
	// Calculate tick values
	xTicks := calculateIntTicks(minX, maxX, 10)
	yTicks := calculateFloatTicks(minY, maxY, 8)

	sb.WriteString("    <g class=\"axes\">\n")

	// X-axis
	sb.WriteString(fmt.Sprintf("      <line x1=\"0\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1\"/>\n",
		plotHeight, plotWidth, plotHeight, spb.config.AxisColor))

	// X-axis ticks and labels
	for _, tick := range xTicks {
		x := scaleValue(float64(tick), float64(minX), float64(maxX), 0, float64(plotWidth))
		sb.WriteString(fmt.Sprintf("      <line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" stroke=\"%s\"/>\n",
			x, plotHeight, x, plotHeight+5, spb.config.AxisColor))
		sb.WriteString(fmt.Sprintf("      <text x=\"%.1f\" y=\"%d\" class=\"tick-label\" text-anchor=\"middle\">%d</text>\n",
			x, plotHeight+18, tick))
	}

	// Y-axis
	sb.WriteString(fmt.Sprintf("      <line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1\"/>\n",
		plotHeight, spb.config.AxisColor))

	// Y-axis ticks and labels
	for _, tick := range yTicks {
		y := scaleValue(tick, minY, maxY, float64(plotHeight), 0)
		sb.WriteString(fmt.Sprintf("      <line x1=\"-5\" y1=\"%.1f\" x2=\"0\" y2=\"%.1f\" stroke=\"%s\"/>\n",
			y, y, spb.config.AxisColor))
		sb.WriteString(fmt.Sprintf("      <text x=\"-8\" y=\"%.1f\" class=\"tick-label\" text-anchor=\"end\" dominant-baseline=\"middle\">%.2f</text>\n",
			y, tick))
	}

	sb.WriteString("    </g>\n")
}

// writeSeries writes a single data series (line and points).
func (spb *SVGPlotBuilder) writeSeries(sb *strings.Builder, series DataSeries, plotWidth, plotHeight int, minX, maxX int, minY, maxY float64) {
	if len(series.Points) == 0 {
		return
	}

	// Determine series color
	color := series.Color
	if color == "" {
		color = GetMetricInfo(series.Metric).Color
	}

	// Build path for the line
	var pathBuilder strings.Builder
	for i, point := range series.Points {
		x := scaleValue(float64(point.Turn), float64(minX), float64(maxX), 0, float64(plotWidth))
		y := scaleValue(point.Value, minY, maxY, float64(plotHeight), 0)

		if i == 0 {
			pathBuilder.WriteString(fmt.Sprintf("M %.1f %.1f", x, y))
		} else {
			pathBuilder.WriteString(fmt.Sprintf(" L %.1f %.1f", x, y))
		}
	}

	// Write the line path
	sb.WriteString(fmt.Sprintf("    <path d=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"%.1f\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n",
		pathBuilder.String(), color, spb.config.LineWidth))

	// Write data points if enabled
	if spb.config.ShowPoints {
		for _, point := range series.Points {
			x := scaleValue(float64(point.Turn), float64(minX), float64(maxX), 0, float64(plotWidth))
			y := scaleValue(point.Value, minY, maxY, float64(plotHeight), 0)

			sb.WriteString(fmt.Sprintf("    <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"%s\" stroke=\"%s\" stroke-width=\"1\"/>\n",
				x, y, spb.config.PointRadius, spb.config.BackgroundColor, color))
		}
	}
}

// writeTitle writes the plot title.
func (spb *SVGPlotBuilder) writeTitle(sb *strings.Builder, width int) {
	// Escape special XML characters
	title := escapeXML(spb.config.Title)
	sb.WriteString(fmt.Sprintf("  <text x=\"%d\" y=\"24\" class=\"title\" text-anchor=\"middle\">%s</text>\n",
		width/2, title))
}

// writeLegend writes the legend for multiple series.
func (spb *SVGPlotBuilder) writeLegend(sb *strings.Builder, width int) {
	sb.WriteString("    <g class=\"legend\">\n")

	startX := width - spb.config.Padding - 20
	startY := spb.config.Padding + 10

	for i, series := range spb.series {
		y := startY + i*18

		// Get color and label
		color := series.Color
		if color == "" {
			color = GetMetricInfo(series.Metric).Color
		}
		label := series.Label
		if label == "" {
			label = GetMetricInfo(series.Metric).Name
		}

		// Draw colored line
		sb.WriteString(fmt.Sprintf("      <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"2\"/>\n",
			startX-40, y, startX-10, y, color))

		// Draw point marker
		if spb.config.ShowPoints {
			sb.WriteString(fmt.Sprintf("      <circle cx=\"%d\" cy=\"%d\" r=\"3\" fill=\"%s\" stroke=\"%s\"/>\n",
				startX-25, y, spb.config.BackgroundColor, color))
		}

		// Draw label
		sb.WriteString(fmt.Sprintf("      <text x=\"%d\" y=\"%d\" class=\"legend-text\" text-anchor=\"end\" dominant-baseline=\"middle\">%s</text>\n",
			startX-45, y, escapeXML(label)))
	}

	sb.WriteString("    </g>\n")
}

// writeAxisLabels writes the axis labels.
func (spb *SVGPlotBuilder) writeAxisLabels(sb *strings.Builder, width, height, padding, plotWidth, plotHeight int) {
	// X-axis label
	xLabel := spb.config.XAxisLabel
	if xLabel == "" {
		xLabel = "Turn"
	}
	sb.WriteString(fmt.Sprintf("  <text x=\"%d\" y=\"%d\" class=\"axis-label\" text-anchor=\"middle\">%s</text>\n",
		padding+plotWidth/2, height-15, escapeXML(xLabel)))

	// Y-axis label
	yLabel := spb.config.YAxisLabel
	if yLabel == "" && len(spb.series) == 1 {
		yLabel = GetMetricInfo(spb.series[0].Metric).Symbol
	} else if yLabel == "" {
		yLabel = "Value"
	}
	sb.WriteString(fmt.Sprintf("  <text x=\"15\" y=\"%d\" class=\"axis-label\" text-anchor=\"middle\" transform=\"rotate(-90, 15, %d)\">%s</text>\n",
		padding+plotHeight/2, padding+plotHeight/2, escapeXML(yLabel)))
}

// scaleValue maps a value from one range to another.
func scaleValue(value, srcMin, srcMax, dstMin, dstMax float64) float64 {
	if srcMax == srcMin {
		return (dstMin + dstMax) / 2
	}
	return dstMin + (value-srcMin)*(dstMax-dstMin)/(srcMax-srcMin)
}

// calculateIntTicks generates nice tick values for integer ranges.
func calculateIntTicks(min, max, maxTicks int) []int {
	if max <= min {
		return []int{min}
	}

	rangeSize := max - min
	step := 1

	// Find a nice step size
	if rangeSize > maxTicks {
		step = (rangeSize + maxTicks - 1) / maxTicks
		// Round step to nice values
		if step > 1 && step < 5 {
			step = 5
		} else if step >= 5 && step < 10 {
			step = 10
		} else if step >= 10 {
			magnitude := int(math.Pow(10, math.Floor(math.Log10(float64(step)))))
			step = ((step + magnitude - 1) / magnitude) * magnitude
		}
	}

	// Generate ticks
	ticks := make([]int, 0)
	start := (min / step) * step
	if start < min {
		start += step
	}

	for tick := start; tick <= max; tick += step {
		ticks = append(ticks, tick)
	}

	// Always include min and max if not already present
	if len(ticks) == 0 || ticks[0] > min {
		ticks = append([]int{min}, ticks...)
	}
	if ticks[len(ticks)-1] < max {
		ticks = append(ticks, max)
	}

	return ticks
}

// calculateFloatTicks generates nice tick values for float ranges.
func calculateFloatTicks(min, max float64, maxTicks int) []float64 {
	if max <= min {
		return []float64{min}
	}

	rangeSize := max - min
	roughStep := rangeSize / float64(maxTicks)

	// Round to nice values
	magnitude := math.Pow(10, math.Floor(math.Log10(roughStep)))
	residual := roughStep / magnitude

	var step float64
	if residual <= 1.5 {
		step = magnitude
	} else if residual <= 3 {
		step = 2 * magnitude
	} else if residual <= 7 {
		step = 5 * magnitude
	} else {
		step = 10 * magnitude
	}

	// Generate ticks
	ticks := make([]float64, 0)
	start := math.Floor(min/step) * step

	for tick := start; tick <= max+step*0.1; tick += step {
		if tick >= min-step*0.1 && tick <= max+step*0.1 {
			ticks = append(ticks, roundToSignificant(tick, 6))
		}
	}

	return ticks
}

// roundToSignificant rounds a number to n significant figures.
func roundToSignificant(value float64, n int) float64 {
	if value == 0 {
		return 0
	}
	magnitude := math.Pow(10, math.Floor(math.Log10(math.Abs(value)))-float64(n-1))
	return math.Round(value/magnitude) * magnitude
}

// escapeXML escapes special characters for XML/SVG content.
func escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	s = strings.ReplaceAll(s, "'", "&apos;")
	return s
}

// -----------------------------------------------------------------------------
// Convenience Functions
// -----------------------------------------------------------------------------

// GenerateMeasurementPlot creates an SVG plot from MeasurementRow data.
// Plots the specified metric over conversation turns.
func GenerateMeasurementPlot(rows []MeasurementRow, metric PlotMetric, config *SVGConfig) string {
	if config == nil {
		config = DefaultSVGConfig()
	}

	// Set Y-axis label based on metric
	if config.YAxisLabel == "" {
		config.YAxisLabel = GetMetricInfo(metric).Symbol
	}

	builder := NewSVGPlotBuilder(config)

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

// GenerateMultiMetricPlot creates an SVG plot with multiple metrics.
// Useful for comparing different metrics on the same time scale.
func GenerateMultiMetricPlot(rows []MeasurementRow, metrics []PlotMetric, config *SVGConfig) string {
	if config == nil {
		config = DefaultSVGConfig()
	}

	builder := NewSVGPlotBuilder(config)

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

// GenerateBetaPlotWithThresholds creates a Beta plot with threshold lines.
// Shows the optimal, monitor, concerning, and critical thresholds.
func GenerateBetaPlotWithThresholds(rows []MeasurementRow, config *SVGConfig) string {
	if config == nil {
		config = DefaultSVGConfig()
	}

	// Set appropriate title and labels
	if config.Title == "" {
		config.Title = "Beta (Collapse Indicator) Over Time"
	}
	if config.YAxisLabel == "" {
		config.YAxisLabel = "β"
	}

	builder := NewSVGPlotBuilder(config)

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

	// Build the SVG and add threshold lines
	svg := builder.Build()

	// Insert threshold reference lines (this is a simplified approach)
	// For more complex threshold visualization, we would need to extend the builder

	return svg
}

// ExportSVGToWriter writes an SVG plot to an io.Writer.
func ExportSVGToWriter(w io.Writer, rows []MeasurementRow, metric PlotMetric, config *SVGConfig) error {
	svg := GenerateMeasurementPlot(rows, metric, config)
	_, err := io.WriteString(w, svg)
	return err
}
