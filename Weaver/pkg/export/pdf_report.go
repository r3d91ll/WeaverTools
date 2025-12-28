// Package export provides academic export format utilities.
// Generates LaTeX tables, CSV data, PDF documents, and other publication-ready outputs.
package export

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// PDFReportConfig specifies options for PDF report generation.
type PDFReportConfig struct {
	// PageWidth is the page width in points (1 point = 1/72 inch).
	// Default: 612 (8.5 inches, US Letter width)
	PageWidth float64

	// PageHeight is the page height in points.
	// Default: 792 (11 inches, US Letter height)
	PageHeight float64

	// Margins in points (left, right, top, bottom).
	MarginLeft   float64
	MarginRight  float64
	MarginTop    float64
	MarginBottom float64

	// Title is the report title for the cover page.
	Title string

	// Author is the report author.
	Author string

	// Subject is the PDF subject metadata.
	Subject string

	// Keywords are PDF metadata keywords.
	Keywords []string

	// ToolVersion is the tool version for metadata.
	ToolVersion string

	// IncludeCoverPage generates a title/cover page.
	// Default: true
	IncludeCoverPage bool

	// IncludeTableOfContents generates a table of contents.
	// Default: true
	IncludeTableOfContents bool

	// IncludePageNumbers adds page numbers to each page.
	// Default: true
	IncludePageNumbers bool

	// IncludeMetrics includes metrics tables.
	// Default: true
	IncludeMetrics bool

	// IncludePlots embeds plots/figures.
	// Default: true
	IncludePlots bool

	// FontFamily is the base font family.
	// Default: "Helvetica"
	FontFamily string

	// BaseFontSize is the base font size in points.
	// Default: 10
	BaseFontSize float64

	// Compress enables stream compression.
	// Default: true
	Compress bool
}

// DefaultPDFReportConfig returns a PDFReportConfig with sensible defaults.
func DefaultPDFReportConfig() *PDFReportConfig {
	return &PDFReportConfig{
		PageWidth:              PDFDimensions.Letter.Width,
		PageHeight:             PDFDimensions.Letter.Height,
		MarginLeft:             72,  // 1 inch
		MarginRight:            72,  // 1 inch
		MarginTop:              72,  // 1 inch
		MarginBottom:           72,  // 1 inch
		IncludeCoverPage:       true,
		IncludeTableOfContents: true,
		IncludePageNumbers:     true,
		IncludeMetrics:         true,
		IncludePlots:           true,
		FontFamily:             "Helvetica",
		BaseFontSize:           10,
		Compress:               true,
	}
}

// PDFReportSection represents a section in the PDF report.
type PDFReportSection struct {
	Title   string
	Content string
	Level   int // 1 = h1, 2 = h2, etc.
}

// PDFReportPlot represents a plot to embed in the PDF report.
type PDFReportPlot struct {
	Title       string
	Description string
	Data        []MeasurementRow
	Metrics     []PlotMetric
	Width       float64
	Height      float64
}

// PDFReportTable represents a table in the PDF report.
type PDFReportTable struct {
	Title   string
	Headers []string
	Rows    [][]string
}

// PDFReportBuilder constructs PDF reports with a fluent API.
type PDFReportBuilder struct {
	config      *PDFReportConfig
	report      *ReproducibilityReport
	sections    []PDFReportSection
	plots       []PDFReportPlot
	tables      []PDFReportTable
	tocEntries  []tocEntry
	pageCount   int
	currentY    float64
	pageContent []string
}

// tocEntry represents a table of contents entry.
type tocEntry struct {
	Title    string
	Level    int
	PageNum  int
}

// NewPDFReportBuilder creates a new PDFReportBuilder with default configuration.
func NewPDFReportBuilder() *PDFReportBuilder {
	return &PDFReportBuilder{
		config:     DefaultPDFReportConfig(),
		sections:   make([]PDFReportSection, 0),
		plots:      make([]PDFReportPlot, 0),
		tables:     make([]PDFReportTable, 0),
		tocEntries: make([]tocEntry, 0),
	}
}

// WithConfig sets the configuration for the builder.
func (prb *PDFReportBuilder) WithConfig(config *PDFReportConfig) *PDFReportBuilder {
	if config != nil {
		prb.config = config
	}
	return prb
}

// WithReport sets the reproducibility report data.
func (prb *PDFReportBuilder) WithReport(report *ReproducibilityReport) *PDFReportBuilder {
	prb.report = report
	return prb
}

// WithTitle sets the report title.
func (prb *PDFReportBuilder) WithTitle(title string) *PDFReportBuilder {
	prb.config.Title = title
	return prb
}

// WithAuthor sets the report author.
func (prb *PDFReportBuilder) WithAuthor(author string) *PDFReportBuilder {
	prb.config.Author = author
	return prb
}

// WithSubject sets the PDF subject metadata.
func (prb *PDFReportBuilder) WithSubject(subject string) *PDFReportBuilder {
	prb.config.Subject = subject
	return prb
}

// WithKeywords sets the PDF metadata keywords.
func (prb *PDFReportBuilder) WithKeywords(keywords []string) *PDFReportBuilder {
	prb.config.Keywords = keywords
	return prb
}

// WithToolVersion sets the tool version for metadata.
func (prb *PDFReportBuilder) WithToolVersion(version string) *PDFReportBuilder {
	prb.config.ToolVersion = version
	return prb
}

// AddSection adds a text section to the report.
func (prb *PDFReportBuilder) AddSection(title, content string, level int) *PDFReportBuilder {
	prb.sections = append(prb.sections, PDFReportSection{
		Title:   title,
		Content: content,
		Level:   level,
	})
	return prb
}

// AddPlot adds a plot to the report.
func (prb *PDFReportBuilder) AddPlot(title, description string, data []MeasurementRow, metrics []PlotMetric) *PDFReportBuilder {
	prb.plots = append(prb.plots, PDFReportPlot{
		Title:       title,
		Description: description,
		Data:        data,
		Metrics:     metrics,
		Width:       400,
		Height:      250,
	})
	return prb
}

// AddPlotWithDimensions adds a plot with custom dimensions.
func (prb *PDFReportBuilder) AddPlotWithDimensions(title, description string, data []MeasurementRow, metrics []PlotMetric, width, height float64) *PDFReportBuilder {
	prb.plots = append(prb.plots, PDFReportPlot{
		Title:       title,
		Description: description,
		Data:        data,
		Metrics:     metrics,
		Width:       width,
		Height:      height,
	})
	return prb
}

// AddTable adds a data table to the report.
func (prb *PDFReportBuilder) AddTable(title string, headers []string, rows [][]string) *PDFReportBuilder {
	prb.tables = append(prb.tables, PDFReportTable{
		Title:   title,
		Headers: headers,
		Rows:    rows,
	})
	return prb
}

// AddMeasurementsTable adds a table from measurement data.
func (prb *PDFReportBuilder) AddMeasurementsTable(title string, rows []MeasurementRow) *PDFReportBuilder {
	headers := []string{"Turn", "Sender", "Receiver", "D_eff", "Beta", "Alignment", "C_pair"}
	tableRows := make([][]string, len(rows))
	for i, row := range rows {
		tableRows[i] = []string{
			fmt.Sprintf("%d", row.Turn),
			row.Sender,
			row.Receiver,
			fmt.Sprintf("%d", row.DEff),
			fmt.Sprintf("%.3f", row.Beta),
			fmt.Sprintf("%.3f", row.Alignment),
			fmt.Sprintf("%.3f", row.CPair),
		}
	}
	return prb.AddTable(title, headers, tableRows)
}

// Build generates the complete PDF report as bytes.
func (prb *PDFReportBuilder) Build() []byte {
	doc := newPDFReportDocument(prb.config)

	// Build all pages
	pages := prb.buildPages()

	// Add pages to document
	for _, pageContent := range pages {
		doc.addPage(prb.config.PageWidth, prb.config.PageHeight, pageContent)
	}

	return doc.build()
}

// buildPages generates all page content.
func (prb *PDFReportBuilder) buildPages() []string {
	pages := make([]string, 0)
	prb.pageCount = 0

	// Cover page
	if prb.config.IncludeCoverPage {
		pages = append(pages, prb.buildCoverPage())
		prb.pageCount++
	}

	// Table of contents (placeholder - we'll update later)
	if prb.config.IncludeTableOfContents {
		pages = append(pages, prb.buildTableOfContents())
		prb.pageCount++
	}

	// Report content pages
	contentPages := prb.buildContentPages()
	pages = append(pages, contentPages...)

	return pages
}

// buildCoverPage generates the cover page content.
func (prb *PDFReportBuilder) buildCoverPage() string {
	var sb strings.Builder
	width := prb.config.PageWidth
	height := prb.config.PageHeight

	// Save graphics state
	sb.WriteString("q\n")

	// Background
	sb.WriteString("1 1 1 rg\n")
	sb.WriteString(fmt.Sprintf("0 0 %.2f %.2f re f\n", width, height))

	// Title
	title := prb.config.Title
	if title == "" && prb.report != nil {
		title = prb.report.Title
	}
	if title == "" {
		title = "Experiment Report"
	}

	titleFontSize := prb.config.BaseFontSize * 2.4
	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", titleFontSize))
	sb.WriteString("0 0 0 rg\n")

	// Center title
	textWidth := float64(len(title)) * titleFontSize * 0.45
	x := (width - textWidth) / 2
	y := height * 0.6

	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", x, y))
	sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(title)))
	sb.WriteString("ET\n")

	// Author
	author := prb.config.Author
	if author == "" && prb.report != nil {
		author = prb.report.Author
	}
	if author != "" {
		authorFontSize := prb.config.BaseFontSize * 1.2
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", authorFontSize))
		sb.WriteString("0.3 0.3 0.3 rg\n")

		textWidth = float64(len(author)) * authorFontSize * 0.45
		x = (width - textWidth) / 2
		y = height*0.6 - 40

		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", x, y))
		sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(author)))
		sb.WriteString("ET\n")
	}

	// Date
	dateStr := time.Now().Format("January 2, 2006")
	dateFontSize := prb.config.BaseFontSize
	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", dateFontSize))
	sb.WriteString("0.5 0.5 0.5 rg\n")

	textWidth = float64(len(dateStr)) * dateFontSize * 0.45
	x = (width - textWidth) / 2
	y = height*0.6 - 70

	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", x, y))
	sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(dateStr)))
	sb.WriteString("ET\n")

	// Tool version
	if prb.config.ToolVersion != "" {
		versionStr := fmt.Sprintf("Generated by WeaverTools %s", prb.config.ToolVersion)
		versionFontSize := prb.config.BaseFontSize * 0.8
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", versionFontSize))
		sb.WriteString("0.6 0.6 0.6 rg\n")

		textWidth = float64(len(versionStr)) * versionFontSize * 0.45
		x = (width - textWidth) / 2
		y = prb.config.MarginBottom

		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", x, y))
		sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(versionStr)))
		sb.WriteString("ET\n")
	}

	// Restore graphics state
	sb.WriteString("Q\n")

	return sb.String()
}

// buildTableOfContents generates the table of contents page.
func (prb *PDFReportBuilder) buildTableOfContents() string {
	var sb strings.Builder
	width := prb.config.PageWidth
	height := prb.config.PageHeight
	margin := prb.config.MarginLeft

	// Save graphics state
	sb.WriteString("q\n")

	// Background
	sb.WriteString("1 1 1 rg\n")
	sb.WriteString(fmt.Sprintf("0 0 %.2f %.2f re f\n", width, height))

	// Title
	titleFontSize := prb.config.BaseFontSize * 1.8
	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", titleFontSize))
	sb.WriteString("0 0 0 rg\n")
	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, height-prb.config.MarginTop))
	sb.WriteString("(Table of Contents) Tj\n")
	sb.WriteString("ET\n")

	// Build entries based on sections and report
	y := height - prb.config.MarginTop - 50
	entryFontSize := prb.config.BaseFontSize
	lineHeight := entryFontSize * 1.8

	// Add entries from report if available
	if prb.report != nil {
		entries := []string{
			"Experiment Hash",
			"Tool Information",
			"Experiment Timing",
			"Configuration",
		}
		if len(prb.report.Agents) > 0 {
			entries = append(entries, "Agent Configurations")
		}
		if len(prb.report.DataSources) > 0 {
			entries = append(entries, "Data Sources")
		}
		if len(prb.report.Parameters) > 0 {
			entries = append(entries, "Experiment Parameters")
		}

		for _, entry := range entries {
			sb.WriteString("BT\n")
			sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", entryFontSize))
			sb.WriteString("0.2 0.2 0.2 rg\n")
			sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20, y))
			sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(entry)))
			sb.WriteString("ET\n")
			y -= lineHeight
		}
	}

	// Add sections
	for _, section := range prb.sections {
		indent := float64(section.Level-1) * 20
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", entryFontSize))
		sb.WriteString("0.2 0.2 0.2 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20+indent, y))
		sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(section.Title)))
		sb.WriteString("ET\n")
		y -= lineHeight
	}

	// Add plots and tables
	if len(prb.plots) > 0 {
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", entryFontSize))
		sb.WriteString("0.2 0.2 0.2 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20, y))
		sb.WriteString("(Figures) Tj\n")
		sb.WriteString("ET\n")
		y -= lineHeight
	}

	if len(prb.tables) > 0 {
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", entryFontSize))
		sb.WriteString("0.2 0.2 0.2 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20, y))
		sb.WriteString("(Tables) Tj\n")
		sb.WriteString("ET\n")
	}

	// Restore graphics state
	sb.WriteString("Q\n")

	return sb.String()
}

// buildContentPages generates the main content pages.
func (prb *PDFReportBuilder) buildContentPages() []string {
	pages := make([]string, 0)

	// Report data page
	if prb.report != nil {
		reportPages := prb.buildReportPages()
		pages = append(pages, reportPages...)
	}

	// Custom sections
	if len(prb.sections) > 0 {
		sectionPages := prb.buildSectionPages()
		pages = append(pages, sectionPages...)
	}

	// Tables
	if len(prb.tables) > 0 {
		tablePages := prb.buildTablePages()
		pages = append(pages, tablePages...)
	}

	// Plots
	if len(prb.plots) > 0 && prb.config.IncludePlots {
		plotPages := prb.buildPlotPages()
		pages = append(pages, plotPages...)
	}

	return pages
}

// buildReportPages generates pages from the reproducibility report.
func (prb *PDFReportBuilder) buildReportPages() []string {
	pages := make([]string, 0)
	width := prb.config.PageWidth
	height := prb.config.PageHeight
	margin := prb.config.MarginLeft
	contentWidth := width - prb.config.MarginLeft - prb.config.MarginRight

	var sb strings.Builder
	y := height - prb.config.MarginTop
	lineHeight := prb.config.BaseFontSize * 1.5

	// Helper to start a new page
	startNewPage := func() {
		if sb.Len() > 0 {
			sb.WriteString("Q\n")
			pages = append(pages, sb.String())
			sb.Reset()
		}
		sb.WriteString("q\n")
		sb.WriteString("1 1 1 rg\n")
		sb.WriteString(fmt.Sprintf("0 0 %.2f %.2f re f\n", width, height))
		y = height - prb.config.MarginTop
		prb.pageCount++
	}

	// Helper to check if we need a new page
	checkPageBreak := func(needed float64) {
		if y-needed < prb.config.MarginBottom {
			startNewPage()
		}
	}

	// Start first page
	startNewPage()

	// Section: Experiment Hash
	sectionFontSize := prb.config.BaseFontSize * 1.4
	checkPageBreak(sectionFontSize * 4)

	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", sectionFontSize))
	sb.WriteString("0 0 0 rg\n")
	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
	sb.WriteString("(Experiment Hash) Tj\n")
	sb.WriteString("ET\n")
	y -= sectionFontSize + 10

	if prb.report.ExperimentHash != nil {
		hash := prb.report.ExperimentHash
		lines := []string{
			fmt.Sprintf("Hash: %s", hash.Hash),
			fmt.Sprintf("Short: %s", hash.ShortHash()),
			fmt.Sprintf("Algorithm: %s", hash.Algorithm),
			fmt.Sprintf("Computed: %s", hash.ComputedAt.UTC().Format(time.RFC3339)),
		}
		for _, line := range lines {
			checkPageBreak(lineHeight)
			sb.WriteString("BT\n")
			sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
			sb.WriteString("0.3 0.3 0.3 rg\n")
			sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20, y))
			sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(line)))
			sb.WriteString("ET\n")
			y -= lineHeight
		}
	}
	y -= 20

	// Section: Tool Information
	checkPageBreak(sectionFontSize * 3)
	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", sectionFontSize))
	sb.WriteString("0 0 0 rg\n")
	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
	sb.WriteString("(Tool Information) Tj\n")
	sb.WriteString("ET\n")
	y -= sectionFontSize + 10

	toolLine := fmt.Sprintf("Tool Version: %s", prb.report.ToolVersion)
	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
	sb.WriteString("0.3 0.3 0.3 rg\n")
	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20, y))
	sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(toolLine)))
	sb.WriteString("ET\n")
	y -= lineHeight + 20

	// Section: Experiment Timing
	checkPageBreak(sectionFontSize * 5)
	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", sectionFontSize))
	sb.WriteString("0 0 0 rg\n")
	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
	sb.WriteString("(Experiment Timing) Tj\n")
	sb.WriteString("ET\n")
	y -= sectionFontSize + 10

	timingLines := []string{}
	if !prb.report.StartTime.IsZero() {
		timingLines = append(timingLines, fmt.Sprintf("Start Time: %s", prb.report.StartTime.UTC().Format(time.RFC3339)))
	}
	if prb.report.EndTime != nil && !prb.report.EndTime.IsZero() {
		timingLines = append(timingLines, fmt.Sprintf("End Time: %s", prb.report.EndTime.UTC().Format(time.RFC3339)))
	}
	if prb.report.Duration != "" {
		timingLines = append(timingLines, fmt.Sprintf("Duration: %s", prb.report.Duration))
	}
	for _, line := range timingLines {
		checkPageBreak(lineHeight)
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
		sb.WriteString("0.3 0.3 0.3 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20, y))
		sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(line)))
		sb.WriteString("ET\n")
		y -= lineHeight
	}
	y -= 20

	// Section: Configuration
	checkPageBreak(sectionFontSize * 5)
	sb.WriteString("BT\n")
	sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", sectionFontSize))
	sb.WriteString("0 0 0 rg\n")
	sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
	sb.WriteString("(Configuration) Tj\n")
	sb.WriteString("ET\n")
	y -= sectionFontSize + 10

	configLines := []string{
		fmt.Sprintf("Measurement Mode: %s", prb.report.MeasurementMode),
		fmt.Sprintf("Measurement Count: %d", prb.report.MeasurementCount),
		fmt.Sprintf("Conversation Count: %d", prb.report.ConversationCount),
	}
	for _, line := range configLines {
		checkPageBreak(lineHeight)
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
		sb.WriteString("0.3 0.3 0.3 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20, y))
		sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(line)))
		sb.WriteString("ET\n")
		y -= lineHeight
	}
	y -= 20

	// Section: Agent Configurations
	if len(prb.report.Agents) > 0 {
		checkPageBreak(sectionFontSize * 3)
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", sectionFontSize))
		sb.WriteString("0 0 0 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
		sb.WriteString("(Agent Configurations) Tj\n")
		sb.WriteString("ET\n")
		y -= sectionFontSize + 10

		for i, agent := range prb.report.Agents {
			checkPageBreak(lineHeight * 5)

			// Agent header
			agentHeader := fmt.Sprintf("Agent %d: %s", i+1, agent.Name)
			sb.WriteString("BT\n")
			sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize*1.1))
			sb.WriteString("0 0 0 rg\n")
			sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+10, y))
			sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(agentHeader)))
			sb.WriteString("ET\n")
			y -= lineHeight

			// Agent details
			agentLines := []string{
				fmt.Sprintf("ID: %s", agent.ID),
				fmt.Sprintf("Type: %s", agent.Type),
			}
			if agent.Model != "" {
				agentLines = append(agentLines, fmt.Sprintf("Model: %s", agent.Model))
			}
			if agent.Provider != "" {
				agentLines = append(agentLines, fmt.Sprintf("Provider: %s", agent.Provider))
			}

			for _, line := range agentLines {
				checkPageBreak(lineHeight)
				sb.WriteString("BT\n")
				sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
				sb.WriteString("0.3 0.3 0.3 rg\n")
				sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+30, y))
				sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(line)))
				sb.WriteString("ET\n")
				y -= lineHeight
			}
			y -= 10
		}
		y -= 10
	}

	// Section: Data Sources
	if len(prb.report.DataSources) > 0 {
		checkPageBreak(sectionFontSize * 3)
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", sectionFontSize))
		sb.WriteString("0 0 0 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
		sb.WriteString("(Data Sources) Tj\n")
		sb.WriteString("ET\n")
		y -= sectionFontSize + 10

		for i, ds := range prb.report.DataSources {
			checkPageBreak(lineHeight * 4)

			dsHeader := fmt.Sprintf("Source %d: %s", i+1, ds.Name)
			sb.WriteString("BT\n")
			sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize*1.1))
			sb.WriteString("0 0 0 rg\n")
			sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+10, y))
			sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(dsHeader)))
			sb.WriteString("ET\n")
			y -= lineHeight

			dsLines := []string{fmt.Sprintf("Type: %s", ds.Type)}
			if ds.Path != "" {
				dsLines = append(dsLines, fmt.Sprintf("Path: %s", ds.Path))
			}
			if ds.RecordCount > 0 {
				dsLines = append(dsLines, fmt.Sprintf("Record Count: %d", ds.RecordCount))
			}

			for _, line := range dsLines {
				checkPageBreak(lineHeight)
				sb.WriteString("BT\n")
				sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
				sb.WriteString("0.3 0.3 0.3 rg\n")
				sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+30, y))
				sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(line)))
				sb.WriteString("ET\n")
				y -= lineHeight
			}
			y -= 10
		}
		y -= 10
	}

	// Section: Experiment Parameters
	if len(prb.report.Parameters) > 0 {
		checkPageBreak(sectionFontSize * 3)
		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", sectionFontSize))
		sb.WriteString("0 0 0 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
		sb.WriteString("(Experiment Parameters) Tj\n")
		sb.WriteString("ET\n")
		y -= sectionFontSize + 10

		// Sort keys for deterministic output
		keys := make([]string, 0, len(prb.report.Parameters))
		for k := range prb.report.Parameters {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		// Draw parameter table
		colWidth := contentWidth / 2
		for _, k := range keys {
			v := prb.report.Parameters[k]
			checkPageBreak(lineHeight)

			// Key
			sb.WriteString("BT\n")
			sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
			sb.WriteString("0 0 0 rg\n")
			sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20, y))
			sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(k)))
			sb.WriteString("ET\n")

			// Value
			sb.WriteString("BT\n")
			sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
			sb.WriteString("0.3 0.3 0.3 rg\n")
			sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin+20+colWidth, y))
			sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(v)))
			sb.WriteString("ET\n")

			y -= lineHeight
		}
	}

	// Close the last page
	sb.WriteString("Q\n")
	pages = append(pages, sb.String())

	// Suppress unused variable warning
	_ = contentWidth

	return pages
}

// buildSectionPages generates pages from custom sections.
func (prb *PDFReportBuilder) buildSectionPages() []string {
	pages := make([]string, 0)
	width := prb.config.PageWidth
	height := prb.config.PageHeight
	margin := prb.config.MarginLeft

	var sb strings.Builder
	y := height - prb.config.MarginTop
	lineHeight := prb.config.BaseFontSize * 1.5

	// Helper functions
	startNewPage := func() {
		if sb.Len() > 0 {
			sb.WriteString("Q\n")
			pages = append(pages, sb.String())
			sb.Reset()
		}
		sb.WriteString("q\n")
		sb.WriteString("1 1 1 rg\n")
		sb.WriteString(fmt.Sprintf("0 0 %.2f %.2f re f\n", width, height))
		y = height - prb.config.MarginTop
		prb.pageCount++
	}

	checkPageBreak := func(needed float64) {
		if y-needed < prb.config.MarginBottom {
			startNewPage()
		}
	}

	startNewPage()

	for _, section := range prb.sections {
		// Section title
		fontSize := prb.config.BaseFontSize * (1.0 + 0.4/float64(section.Level))
		checkPageBreak(fontSize * 3)

		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", fontSize))
		sb.WriteString("0 0 0 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
		sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(section.Title)))
		sb.WriteString("ET\n")
		y -= fontSize + 10

		// Section content - wrap text
		contentWidth := width - prb.config.MarginLeft - prb.config.MarginRight
		words := strings.Fields(section.Content)
		line := ""
		charWidth := prb.config.BaseFontSize * 0.5

		for _, word := range words {
			testLine := line
			if testLine != "" {
				testLine += " "
			}
			testLine += word

			if float64(len(testLine))*charWidth > contentWidth {
				// Write current line
				if line != "" {
					checkPageBreak(lineHeight)
					sb.WriteString("BT\n")
					sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
					sb.WriteString("0.3 0.3 0.3 rg\n")
					sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
					sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(line)))
					sb.WriteString("ET\n")
					y -= lineHeight
				}
				line = word
			} else {
				line = testLine
			}
		}

		// Write remaining text
		if line != "" {
			checkPageBreak(lineHeight)
			sb.WriteString("BT\n")
			sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
			sb.WriteString("0.3 0.3 0.3 rg\n")
			sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
			sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(line)))
			sb.WriteString("ET\n")
			y -= lineHeight
		}

		y -= 20 // Space between sections
	}

	// Close the last page
	sb.WriteString("Q\n")
	pages = append(pages, sb.String())

	return pages
}

// buildTablePages generates pages with tables.
func (prb *PDFReportBuilder) buildTablePages() []string {
	pages := make([]string, 0)
	width := prb.config.PageWidth
	height := prb.config.PageHeight
	margin := prb.config.MarginLeft
	contentWidth := width - prb.config.MarginLeft - prb.config.MarginRight

	var sb strings.Builder
	y := height - prb.config.MarginTop
	lineHeight := prb.config.BaseFontSize * 1.4

	// Helper functions
	startNewPage := func() {
		if sb.Len() > 0 {
			sb.WriteString("Q\n")
			pages = append(pages, sb.String())
			sb.Reset()
		}
		sb.WriteString("q\n")
		sb.WriteString("1 1 1 rg\n")
		sb.WriteString(fmt.Sprintf("0 0 %.2f %.2f re f\n", width, height))
		y = height - prb.config.MarginTop
		prb.pageCount++
	}

	checkPageBreak := func(needed float64) {
		if y-needed < prb.config.MarginBottom {
			startNewPage()
		}
	}

	startNewPage()

	for _, table := range prb.tables {
		// Table title
		titleFontSize := prb.config.BaseFontSize * 1.3
		checkPageBreak(titleFontSize + lineHeight*3)

		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", titleFontSize))
		sb.WriteString("0 0 0 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
		sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(table.Title)))
		sb.WriteString("ET\n")
		y -= titleFontSize + 15

		// Calculate column widths
		numCols := len(table.Headers)
		if numCols == 0 {
			continue
		}
		colWidth := contentWidth / float64(numCols)

		// Draw header background
		headerHeight := lineHeight + 4
		sb.WriteString("0.9 0.9 0.9 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f re f\n", margin, y-headerHeight+lineHeight, contentWidth, headerHeight))

		// Draw header row
		for i, header := range table.Headers {
			x := margin + float64(i)*colWidth + 5
			sb.WriteString("BT\n")
			sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
			sb.WriteString("0 0 0 rg\n")
			sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", x, y))
			sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(header)))
			sb.WriteString("ET\n")
		}
		y -= lineHeight + 5

		// Draw horizontal line under header
		sb.WriteString("0.5 0.5 0.5 RG\n")
		sb.WriteString("0.5 w\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f m %.2f %.2f l S\n", margin, y+3, margin+contentWidth, y+3))

		// Draw data rows
		for _, row := range table.Rows {
			checkPageBreak(lineHeight)

			for i, cell := range row {
				if i >= numCols {
					break
				}
				x := margin + float64(i)*colWidth + 5
				sb.WriteString("BT\n")
				sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
				sb.WriteString("0.3 0.3 0.3 rg\n")
				sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", x, y))
				sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(cell)))
				sb.WriteString("ET\n")
			}
			y -= lineHeight
		}

		y -= 30 // Space between tables
	}

	// Close the last page
	sb.WriteString("Q\n")
	pages = append(pages, sb.String())

	return pages
}

// buildPlotPages generates pages with embedded plots.
func (prb *PDFReportBuilder) buildPlotPages() []string {
	pages := make([]string, 0)
	width := prb.config.PageWidth
	height := prb.config.PageHeight
	margin := prb.config.MarginLeft

	for _, plot := range prb.plots {
		var sb strings.Builder

		// Save graphics state
		sb.WriteString("q\n")

		// Background
		sb.WriteString("1 1 1 rg\n")
		sb.WriteString(fmt.Sprintf("0 0 %.2f %.2f re f\n", width, height))

		// Plot title
		y := height - prb.config.MarginTop
		titleFontSize := prb.config.BaseFontSize * 1.4

		sb.WriteString("BT\n")
		sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", titleFontSize))
		sb.WriteString("0 0 0 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
		sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(plot.Title)))
		sb.WriteString("ET\n")
		y -= titleFontSize + 10

		// Plot description
		if plot.Description != "" {
			sb.WriteString("BT\n")
			sb.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", prb.config.BaseFontSize))
			sb.WriteString("0.5 0.5 0.5 rg\n")
			sb.WriteString(fmt.Sprintf("%.2f %.2f Td\n", margin, y))
			sb.WriteString(fmt.Sprintf("(%s) Tj\n", escapePDFString(plot.Description)))
			sb.WriteString("ET\n")
			y -= prb.config.BaseFontSize * 1.5
		}

		// Draw the plot area
		plotX := margin
		plotY := y - plot.Height - 20
		plotWidth := plot.Width
		plotHeight := plot.Height

		// Plot background
		sb.WriteString("0.98 0.98 0.98 rg\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f re f\n", plotX, plotY, plotWidth, plotHeight))

		// Plot border
		sb.WriteString("0.8 0.8 0.8 RG\n")
		sb.WriteString("1 w\n")
		sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f re S\n", plotX, plotY, plotWidth, plotHeight))

		// Draw the actual plot data
		if len(plot.Data) > 0 && len(plot.Metrics) > 0 {
			prb.drawPlotData(&sb, plot, plotX, plotY, plotWidth, plotHeight)
		}

		// Restore graphics state
		sb.WriteString("Q\n")

		pages = append(pages, sb.String())
		prb.pageCount++
	}

	return pages
}

// drawPlotData draws the actual plot data within the given bounds.
func (prb *PDFReportBuilder) drawPlotData(sb *strings.Builder, plot PDFReportPlot, x, y, width, height float64) {
	if len(plot.Data) == 0 {
		return
	}

	padding := 40.0
	plotLeft := x + padding
	plotBottom := y + padding
	plotWidth := width - 2*padding
	plotHeight := height - 2*padding

	// Calculate data bounds
	minX, maxX := plot.Data[0].Turn, plot.Data[0].Turn
	minY, maxY := 0.0, 0.0

	for _, row := range plot.Data {
		if row.Turn < minX {
			minX = row.Turn
		}
		if row.Turn > maxX {
			maxX = row.Turn
		}
	}

	// Get values for the first metric
	for _, metric := range plot.Metrics {
		for _, row := range plot.Data {
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
			if value < minY {
				minY = value
			}
			if value > maxY {
				maxY = value
			}
		}
	}

	// Add padding to Y range
	yRange := maxY - minY
	if yRange > 0 {
		minY -= yRange * 0.1
		maxY += yRange * 0.1
	} else {
		minY -= 1
		maxY += 1
	}

	if minX == maxX {
		minX--
		maxX++
	}

	// Draw grid
	sb.WriteString("0.9 0.9 0.9 RG\n")
	sb.WriteString("0.5 w\n")
	sb.WriteString("[3 3] 0 d\n")

	gridLines := 5
	for i := 0; i <= gridLines; i++ {
		gridY := plotBottom + float64(i)*plotHeight/float64(gridLines)
		sb.WriteString(fmt.Sprintf("%.2f %.2f m %.2f %.2f l S\n", plotLeft, gridY, plotLeft+plotWidth, gridY))
	}
	sb.WriteString("[] 0 d\n")

	// Draw axes
	sb.WriteString("0.3 0.3 0.3 RG\n")
	sb.WriteString("1 w\n")
	sb.WriteString(fmt.Sprintf("%.2f %.2f m %.2f %.2f l S\n", plotLeft, plotBottom, plotLeft+plotWidth, plotBottom))
	sb.WriteString(fmt.Sprintf("%.2f %.2f m %.2f %.2f l S\n", plotLeft, plotBottom, plotLeft, plotBottom+plotHeight))

	// Draw data for each metric
	for _, metric := range plot.Metrics {
		info := GetMetricInfo(metric)
		color := HexToPDFColor(info.Color)

		sb.WriteString(fmt.Sprintf("%s RG\n", color.String()))
		sb.WriteString("1.5 w\n")
		sb.WriteString("1 J\n")
		sb.WriteString("1 j\n")

		// Draw line
		for i, row := range plot.Data {
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

			px := plotLeft + scaleValue(float64(row.Turn), float64(minX), float64(maxX), 0, plotWidth)
			py := plotBottom + scaleValue(value, minY, maxY, 0, plotHeight)

			if i == 0 {
				sb.WriteString(fmt.Sprintf("%.2f %.2f m\n", px, py))
			} else {
				sb.WriteString(fmt.Sprintf("%.2f %.2f l\n", px, py))
			}
		}
		sb.WriteString("S\n")

		// Draw points
		sb.WriteString("1 1 1 rg\n")
		radius := 3.0
		for _, row := range plot.Data {
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

			px := plotLeft + scaleValue(float64(row.Turn), float64(minX), float64(maxX), 0, plotWidth)
			py := plotBottom + scaleValue(value, minY, maxY, 0, plotHeight)

			k := radius * 0.5523
			sb.WriteString(fmt.Sprintf("%.2f %.2f m\n", px+radius, py))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				px+radius, py+k, px+k, py+radius, px, py+radius))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				px-k, py+radius, px-radius, py+k, px-radius, py))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				px-radius, py-k, px-k, py-radius, px, py-radius))
			sb.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f %.2f %.2f c\n",
				px+k, py-radius, px+radius, py-k, px+radius, py))
			sb.WriteString("B\n")
		}
	}
}

// -----------------------------------------------------------------------------
// PDF Report Document Builder (Internal)
// -----------------------------------------------------------------------------

// pdfReportDocument builds a complete PDF report file.
type pdfReportDocument struct {
	config    *PDFReportConfig
	objects   []string
	pages     []int
	pageCount int
}

// newPDFReportDocument creates a new PDF report document builder.
func newPDFReportDocument(config *PDFReportConfig) *pdfReportDocument {
	return &pdfReportDocument{
		config:  config,
		objects: make([]string, 0),
		pages:   make([]int, 0),
	}
}

// addObject adds an object and returns its object number.
func (doc *pdfReportDocument) addObject(content string) int {
	doc.objects = append(doc.objects, content)
	return len(doc.objects)
}

// addPage adds a page with the given dimensions and content stream.
func (doc *pdfReportDocument) addPage(width, height float64, content string) {
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
		width, height, streamObjNum+3)
	pageObjNum := doc.addObject(pageObj)

	doc.pages = append(doc.pages, pageObjNum)
}

// build generates the complete PDF file.
func (doc *pdfReportDocument) build() []byte {
	var buf bytes.Buffer

	// PDF header
	buf.WriteString(fmt.Sprintf("%%PDF-%s\n", PDFVersion))
	buf.WriteString("%\xE2\xE3\xCF\xD3\n")

	// Build page tree kids array
	var kidsArray strings.Builder
	kidsArray.WriteString("[")
	for i, pageNum := range doc.pages {
		if i > 0 {
			kidsArray.WriteString(" ")
		}
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

	// Object N: Info dictionary
	infoObj := doc.buildInfoDict()
	finalObjects = append(finalObjects, infoObj)
	infoObjNum := len(finalObjects)

	// Write all objects and track xref positions
	xref := make([]int, len(finalObjects)+1)
	xref[0] = 0

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
	buf.WriteString(fmt.Sprintf("<< /Size %d\n/Root 1 0 R\n/Info %d 0 R\n>>", len(finalObjects)+1, infoObjNum))
	buf.WriteString("\nstartxref\n")
	buf.WriteString(fmt.Sprintf("%d\n", xrefPos))
	buf.WriteString("%%EOF\n")

	return buf.Bytes()
}

// buildInfoDict creates the PDF Info dictionary for metadata.
func (doc *pdfReportDocument) buildInfoDict() string {
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

// GeneratePDFReport creates a complete PDF report from a ReproducibilityReport.
func GeneratePDFReport(report *ReproducibilityReport, config *PDFReportConfig) []byte {
	if config == nil {
		config = DefaultPDFReportConfig()
	}

	builder := NewPDFReportBuilder().
		WithConfig(config).
		WithReport(report)

	if config.Title != "" {
		builder.WithTitle(config.Title)
	} else if report != nil && report.Title != "" {
		builder.WithTitle(report.Title)
	}

	if config.Author != "" {
		builder.WithAuthor(config.Author)
	} else if report != nil && report.Author != "" {
		builder.WithAuthor(report.Author)
	}

	if config.ToolVersion != "" {
		builder.WithToolVersion(config.ToolVersion)
	} else if report != nil && report.ToolVersion != "" {
		builder.WithToolVersion(report.ToolVersion)
	}

	return builder.Build()
}

// GeneratePDFReportWithMeasurements creates a PDF report with measurement data and plots.
func GeneratePDFReportWithMeasurements(
	report *ReproducibilityReport,
	measurements []MeasurementRow,
	config *PDFReportConfig,
) []byte {
	if config == nil {
		config = DefaultPDFReportConfig()
	}

	builder := NewPDFReportBuilder().
		WithConfig(config).
		WithReport(report)

	if config.Title != "" {
		builder.WithTitle(config.Title)
	} else if report != nil && report.Title != "" {
		builder.WithTitle(report.Title)
	}

	if config.Author != "" {
		builder.WithAuthor(config.Author)
	} else if report != nil && report.Author != "" {
		builder.WithAuthor(report.Author)
	}

	if config.ToolVersion != "" {
		builder.WithToolVersion(config.ToolVersion)
	} else if report != nil && report.ToolVersion != "" {
		builder.WithToolVersion(report.ToolVersion)
	}

	// Add measurements table
	if len(measurements) > 0 && config.IncludeMetrics {
		builder.AddMeasurementsTable("Measurement Summary", measurements)
	}

	// Add plots
	if len(measurements) > 0 && config.IncludePlots {
		builder.AddPlot(
			"Effective Dimensionality Over Time",
			"D_eff measures semantic richness via PCA (90% variance threshold)",
			measurements,
			[]PlotMetric{MetricDEff},
		)

		builder.AddPlot(
			"Collapse Indicator (Beta) Over Time",
			"Lower beta values indicate better dimensional preservation",
			measurements,
			[]PlotMetric{MetricBeta},
		)

		builder.AddPlot(
			"Bilateral Conveyance Over Time",
			"C_pair measures overall communication effectiveness",
			measurements,
			[]PlotMetric{MetricCPair},
		)
	}

	return builder.Build()
}

// ExportPDFReportToFile writes a PDF report to a file.
func ExportPDFReportToFile(path string, report *ReproducibilityReport, config *PDFReportConfig) error {
	pdf := GeneratePDFReport(report, config)
	if pdf == nil {
		return fmt.Errorf("failed to generate PDF report")
	}

	err := os.WriteFile(path, pdf, 0644)
	if err != nil {
		return fmt.Errorf("failed to write PDF: %w", err)
	}

	return nil
}
