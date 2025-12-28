package export

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestDefaultPDFReportConfig(t *testing.T) {
	config := DefaultPDFReportConfig()

	if config == nil {
		t.Fatal("DefaultPDFReportConfig returned nil")
	}

	// Check default values
	if config.PageWidth != PDFDimensions.Letter.Width {
		t.Errorf("expected PageWidth %f, got %f", PDFDimensions.Letter.Width, config.PageWidth)
	}

	if config.PageHeight != PDFDimensions.Letter.Height {
		t.Errorf("expected PageHeight %f, got %f", PDFDimensions.Letter.Height, config.PageHeight)
	}

	if config.MarginLeft != 72 {
		t.Errorf("expected MarginLeft 72, got %f", config.MarginLeft)
	}

	if !config.IncludeCoverPage {
		t.Error("expected IncludeCoverPage to be true")
	}

	if !config.IncludeTableOfContents {
		t.Error("expected IncludeTableOfContents to be true")
	}

	if !config.IncludePageNumbers {
		t.Error("expected IncludePageNumbers to be true")
	}

	if !config.IncludeMetrics {
		t.Error("expected IncludeMetrics to be true")
	}

	if !config.IncludePlots {
		t.Error("expected IncludePlots to be true")
	}

	if config.FontFamily != "Helvetica" {
		t.Errorf("expected FontFamily 'Helvetica', got '%s'", config.FontFamily)
	}

	if config.BaseFontSize != 10 {
		t.Errorf("expected BaseFontSize 10, got %f", config.BaseFontSize)
	}

	if !config.Compress {
		t.Error("expected Compress to be true")
	}
}

func TestNewPDFReportBuilder(t *testing.T) {
	builder := NewPDFReportBuilder()

	if builder == nil {
		t.Fatal("NewPDFReportBuilder returned nil")
	}

	if builder.config == nil {
		t.Error("builder config is nil")
	}

	if builder.sections == nil {
		t.Error("builder sections is nil")
	}

	if builder.plots == nil {
		t.Error("builder plots is nil")
	}

	if builder.tables == nil {
		t.Error("builder tables is nil")
	}
}

func TestPDFReportBuilderFluentAPI(t *testing.T) {
	builder := NewPDFReportBuilder()

	// Test fluent API chaining
	result := builder.
		WithTitle("Test Report").
		WithAuthor("Test Author").
		WithSubject("Test Subject").
		WithKeywords([]string{"test", "pdf"}).
		WithToolVersion("1.0.0")

	if result != builder {
		t.Error("fluent API should return same builder")
	}

	if builder.config.Title != "Test Report" {
		t.Errorf("expected title 'Test Report', got '%s'", builder.config.Title)
	}

	if builder.config.Author != "Test Author" {
		t.Errorf("expected author 'Test Author', got '%s'", builder.config.Author)
	}

	if builder.config.Subject != "Test Subject" {
		t.Errorf("expected subject 'Test Subject', got '%s'", builder.config.Subject)
	}

	if len(builder.config.Keywords) != 2 {
		t.Errorf("expected 2 keywords, got %d", len(builder.config.Keywords))
	}

	if builder.config.ToolVersion != "1.0.0" {
		t.Errorf("expected tool version '1.0.0', got '%s'", builder.config.ToolVersion)
	}
}

func TestPDFReportBuilderWithConfig(t *testing.T) {
	customConfig := &PDFReportConfig{
		PageWidth:  595.276, // A4
		PageHeight: 841.890,
		MarginLeft: 50,
		BaseFontSize: 12,
	}

	builder := NewPDFReportBuilder().WithConfig(customConfig)

	if builder.config.PageWidth != 595.276 {
		t.Errorf("expected PageWidth 595.276, got %f", builder.config.PageWidth)
	}

	// Test nil config doesn't override
	builder2 := NewPDFReportBuilder().WithConfig(nil)
	if builder2.config.PageWidth != PDFDimensions.Letter.Width {
		t.Error("nil config should not change default config")
	}
}

func TestPDFReportBuilderAddSection(t *testing.T) {
	builder := NewPDFReportBuilder()

	builder.AddSection("Introduction", "This is the introduction content.", 1)
	builder.AddSection("Methods", "Description of methods used.", 2)

	if len(builder.sections) != 2 {
		t.Errorf("expected 2 sections, got %d", len(builder.sections))
	}

	if builder.sections[0].Title != "Introduction" {
		t.Errorf("expected section title 'Introduction', got '%s'", builder.sections[0].Title)
	}

	if builder.sections[0].Level != 1 {
		t.Errorf("expected section level 1, got %d", builder.sections[0].Level)
	}

	if builder.sections[1].Title != "Methods" {
		t.Errorf("expected section title 'Methods', got '%s'", builder.sections[1].Title)
	}
}

func TestPDFReportBuilderAddPlot(t *testing.T) {
	builder := NewPDFReportBuilder()

	data := []MeasurementRow{
		{Turn: 1, DEff: 128, Beta: 1.5, Alignment: 0.8, CPair: 0.7},
		{Turn: 2, DEff: 140, Beta: 1.3, Alignment: 0.85, CPair: 0.75},
	}

	builder.AddPlot("Test Plot", "Description", data, []PlotMetric{MetricDEff, MetricBeta})

	if len(builder.plots) != 1 {
		t.Errorf("expected 1 plot, got %d", len(builder.plots))
	}

	if builder.plots[0].Title != "Test Plot" {
		t.Errorf("expected plot title 'Test Plot', got '%s'", builder.plots[0].Title)
	}

	if len(builder.plots[0].Data) != 2 {
		t.Errorf("expected 2 data points, got %d", len(builder.plots[0].Data))
	}
}

func TestPDFReportBuilderAddPlotWithDimensions(t *testing.T) {
	builder := NewPDFReportBuilder()

	data := []MeasurementRow{
		{Turn: 1, DEff: 128, Beta: 1.5},
	}

	builder.AddPlotWithDimensions("Custom Plot", "Desc", data, []PlotMetric{MetricDEff}, 500, 300)

	if builder.plots[0].Width != 500 {
		t.Errorf("expected width 500, got %f", builder.plots[0].Width)
	}

	if builder.plots[0].Height != 300 {
		t.Errorf("expected height 300, got %f", builder.plots[0].Height)
	}
}

func TestPDFReportBuilderAddTable(t *testing.T) {
	builder := NewPDFReportBuilder()

	headers := []string{"Col1", "Col2", "Col3"}
	rows := [][]string{
		{"A", "B", "C"},
		{"D", "E", "F"},
	}

	builder.AddTable("Test Table", headers, rows)

	if len(builder.tables) != 1 {
		t.Errorf("expected 1 table, got %d", len(builder.tables))
	}

	if builder.tables[0].Title != "Test Table" {
		t.Errorf("expected table title 'Test Table', got '%s'", builder.tables[0].Title)
	}

	if len(builder.tables[0].Headers) != 3 {
		t.Errorf("expected 3 headers, got %d", len(builder.tables[0].Headers))
	}

	if len(builder.tables[0].Rows) != 2 {
		t.Errorf("expected 2 rows, got %d", len(builder.tables[0].Rows))
	}
}

func TestPDFReportBuilderAddMeasurementsTable(t *testing.T) {
	builder := NewPDFReportBuilder()

	data := []MeasurementRow{
		{Turn: 1, Sender: "Alice", Receiver: "Bob", DEff: 128, Beta: 1.5, Alignment: 0.8, CPair: 0.7},
		{Turn: 2, Sender: "Bob", Receiver: "Alice", DEff: 140, Beta: 1.3, Alignment: 0.85, CPair: 0.75},
	}

	builder.AddMeasurementsTable("Measurements", data)

	if len(builder.tables) != 1 {
		t.Errorf("expected 1 table, got %d", len(builder.tables))
	}

	table := builder.tables[0]
	if len(table.Headers) != 7 {
		t.Errorf("expected 7 headers (Turn, Sender, Receiver, D_eff, Beta, Alignment, C_pair), got %d", len(table.Headers))
	}

	if len(table.Rows) != 2 {
		t.Errorf("expected 2 rows, got %d", len(table.Rows))
	}

	// Check first row values
	if table.Rows[0][0] != "1" {
		t.Errorf("expected first row turn '1', got '%s'", table.Rows[0][0])
	}
	if table.Rows[0][1] != "Alice" {
		t.Errorf("expected first row sender 'Alice', got '%s'", table.Rows[0][1])
	}
}

func TestPDFReportBuilderWithReport(t *testing.T) {
	report := &ReproducibilityReport{
		Title:             "Experiment Report",
		Author:            "Test Author",
		GeneratedAt:       time.Now(),
		ToolVersion:       "1.0.0",
		MeasurementMode:   "passive",
		MeasurementCount:  100,
		ConversationCount: 10,
		StartTime:         time.Now().Add(-time.Hour),
	}

	builder := NewPDFReportBuilder().WithReport(report)

	if builder.report != report {
		t.Error("report not set correctly")
	}
}

func TestPDFReportBuilderBuild(t *testing.T) {
	builder := NewPDFReportBuilder().
		WithTitle("Test Report").
		WithAuthor("Test Author")

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build returned nil")
	}

	if len(pdf) == 0 {
		t.Fatal("Build returned empty PDF")
	}

	// Check PDF header
	if !bytes.HasPrefix(pdf, []byte("%PDF-")) {
		t.Error("PDF does not start with %PDF- header")
	}

	// Check PDF footer
	if !bytes.HasSuffix(pdf, []byte("%%EOF\n")) {
		t.Error("PDF does not end with EOF marker")
	}
}

func TestPDFReportBuilderBuildWithReport(t *testing.T) {
	startTime := time.Now().Add(-time.Hour)
	endTime := time.Now()

	report := &ReproducibilityReport{
		Title:             "Full Report",
		Author:            "Test Author",
		GeneratedAt:       time.Now(),
		ToolVersion:       "2.0.0",
		MeasurementMode:   "active",
		MeasurementCount:  50,
		ConversationCount: 5,
		StartTime:         startTime,
		EndTime:           &endTime,
		Duration:          "1h 0m",
		Agents: []AgentConfig{
			{ID: "agent1", Name: "Alice", Type: "llm", Model: "gpt-4"},
			{ID: "agent2", Name: "Bob", Type: "llm", Model: "claude-3"},
		},
		DataSources: []DataSource{
			{Name: "dataset1", Type: "file", Path: "/data/test.csv", RecordCount: 1000},
		},
		Parameters: map[string]string{
			"temperature": "0.7",
			"max_tokens":  "1000",
		},
		ExperimentHash: &ExperimentHash{
			Hash:       "abc123def456",
			Algorithm:  "SHA-256",
			ComputedAt: time.Now(),
		},
	}

	builder := NewPDFReportBuilder().
		WithReport(report).
		WithTitle("Full Report")

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build with report returned nil")
	}

	if len(pdf) == 0 {
		t.Fatal("Build with report returned empty PDF")
	}

	// Verify PDF structure
	if !bytes.HasPrefix(pdf, []byte("%PDF-")) {
		t.Error("PDF does not start with %PDF- header")
	}
}

func TestPDFReportBuilderBuildWithSections(t *testing.T) {
	builder := NewPDFReportBuilder().
		WithTitle("Section Test").
		AddSection("Introduction", "This is the introduction.", 1).
		AddSection("Background", "Background information.", 2).
		AddSection("Methods", "Methods description.", 2)

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build with sections returned nil")
	}

	if len(pdf) == 0 {
		t.Fatal("Build with sections returned empty PDF")
	}
}

func TestPDFReportBuilderBuildWithTables(t *testing.T) {
	builder := NewPDFReportBuilder().
		WithTitle("Table Test").
		AddTable("Simple Table", []string{"A", "B"}, [][]string{{"1", "2"}, {"3", "4"}})

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build with tables returned nil")
	}

	if len(pdf) == 0 {
		t.Fatal("Build with tables returned empty PDF")
	}
}

func TestPDFReportBuilderBuildWithPlots(t *testing.T) {
	data := []MeasurementRow{
		{Turn: 1, DEff: 100, Beta: 1.5, Alignment: 0.8, CPair: 0.7},
		{Turn: 2, DEff: 110, Beta: 1.4, Alignment: 0.82, CPair: 0.72},
		{Turn: 3, DEff: 120, Beta: 1.3, Alignment: 0.85, CPair: 0.75},
	}

	builder := NewPDFReportBuilder().
		WithTitle("Plot Test").
		AddPlot("D_eff Over Time", "Effective dimensionality", data, []PlotMetric{MetricDEff}).
		AddPlot("Beta Over Time", "Collapse indicator", data, []PlotMetric{MetricBeta})

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build with plots returned nil")
	}

	if len(pdf) == 0 {
		t.Fatal("Build with plots returned empty PDF")
	}
}

func TestGeneratePDFReport(t *testing.T) {
	report := &ReproducibilityReport{
		Title:             "Generated Report",
		Author:            "Test Author",
		GeneratedAt:       time.Now(),
		ToolVersion:       "1.0.0",
		MeasurementMode:   "passive",
		MeasurementCount:  10,
		ConversationCount: 2,
		StartTime:         time.Now(),
	}

	pdf := GeneratePDFReport(report, nil)

	if pdf == nil {
		t.Fatal("GeneratePDFReport returned nil")
	}

	if !bytes.HasPrefix(pdf, []byte("%PDF-")) {
		t.Error("PDF does not have proper header")
	}
}

func TestGeneratePDFReportWithMeasurements(t *testing.T) {
	report := &ReproducibilityReport{
		Title:             "Measurements Report",
		ToolVersion:       "1.0.0",
		MeasurementMode:   "active",
		MeasurementCount:  3,
		ConversationCount: 1,
		StartTime:         time.Now(),
	}

	measurements := []MeasurementRow{
		{Turn: 1, Sender: "Alice", Receiver: "Bob", DEff: 128, Beta: 1.5, Alignment: 0.8, CPair: 0.7},
		{Turn: 2, Sender: "Bob", Receiver: "Alice", DEff: 140, Beta: 1.3, Alignment: 0.85, CPair: 0.75},
		{Turn: 3, Sender: "Alice", Receiver: "Bob", DEff: 150, Beta: 1.2, Alignment: 0.9, CPair: 0.8},
	}

	pdf := GeneratePDFReportWithMeasurements(report, measurements, nil)

	if pdf == nil {
		t.Fatal("GeneratePDFReportWithMeasurements returned nil")
	}

	if len(pdf) == 0 {
		t.Fatal("GeneratePDFReportWithMeasurements returned empty PDF")
	}
}

func TestGeneratePDFReportNilReport(t *testing.T) {
	pdf := GeneratePDFReport(nil, nil)

	// Should still generate a valid PDF even with nil report
	if pdf == nil {
		t.Fatal("GeneratePDFReport with nil report returned nil")
	}
}

func TestPDFReportConfigOptions(t *testing.T) {
	config := &PDFReportConfig{
		PageWidth:              595.276, // A4
		PageHeight:             841.890,
		MarginLeft:             50,
		MarginRight:            50,
		MarginTop:              60,
		MarginBottom:           60,
		Title:                  "Custom Report",
		Author:                 "Custom Author",
		Subject:                "Custom Subject",
		Keywords:               []string{"test", "pdf", "report"},
		ToolVersion:            "2.0.0",
		IncludeCoverPage:       true,
		IncludeTableOfContents: true,
		IncludePageNumbers:     true,
		IncludeMetrics:         true,
		IncludePlots:           true,
		FontFamily:             "Helvetica",
		BaseFontSize:           11,
		Compress:               true,
	}

	report := &ReproducibilityReport{
		Title:       "Test",
		ToolVersion: "2.0.0",
	}

	pdf := GeneratePDFReport(report, config)

	if pdf == nil {
		t.Fatal("GeneratePDFReport with custom config returned nil")
	}

	// The PDF should contain the custom title in metadata
	pdfStr := string(pdf)
	if !strings.Contains(pdfStr, "Custom Report") {
		t.Error("PDF should contain custom title")
	}
}

func TestPDFReportNoCoverPage(t *testing.T) {
	config := DefaultPDFReportConfig()
	config.IncludeCoverPage = false

	builder := NewPDFReportBuilder().
		WithConfig(config).
		WithTitle("No Cover Page")

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build without cover page returned nil")
	}
}

func TestPDFReportNoTOC(t *testing.T) {
	config := DefaultPDFReportConfig()
	config.IncludeTableOfContents = false

	builder := NewPDFReportBuilder().
		WithConfig(config).
		WithTitle("No TOC")

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build without TOC returned nil")
	}
}

func TestPDFReportNoCompression(t *testing.T) {
	config := DefaultPDFReportConfig()
	config.Compress = false

	builder := NewPDFReportBuilder().
		WithConfig(config).
		WithTitle("Uncompressed")

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build without compression returned nil")
	}

	// Uncompressed PDF should be larger and contain readable content
	if len(pdf) == 0 {
		t.Error("Uncompressed PDF is empty")
	}
}

func TestPDFReportMultipleMetricPlots(t *testing.T) {
	data := []MeasurementRow{
		{Turn: 1, DEff: 100, Beta: 1.5, Alignment: 0.8, CPair: 0.7},
		{Turn: 2, DEff: 110, Beta: 1.4, Alignment: 0.82, CPair: 0.72},
	}

	builder := NewPDFReportBuilder().
		WithTitle("Multi-Metric Plot").
		AddPlot("All Metrics", "Multiple metrics", data, []PlotMetric{MetricDEff, MetricBeta, MetricAlignment, MetricCPair})

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build with multiple metrics returned nil")
	}
}

func TestPDFReportEmptyData(t *testing.T) {
	builder := NewPDFReportBuilder().
		WithTitle("Empty Report")

	// No sections, tables, or plots
	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build with empty data returned nil")
	}

	if len(pdf) == 0 {
		t.Fatal("Build with empty data returned empty PDF")
	}
}

func TestPDFReportLongContent(t *testing.T) {
	builder := NewPDFReportBuilder().
		WithTitle("Long Content Report")

	// Add long section content that should cause page breaks
	longContent := strings.Repeat("This is a test sentence that should wrap. ", 100)
	builder.AddSection("Long Section", longContent, 1)

	// Add multiple tables
	for i := 0; i < 5; i++ {
		headers := []string{"Col1", "Col2", "Col3"}
		rows := make([][]string, 20)
		for j := 0; j < 20; j++ {
			rows[j] = []string{"A", "B", "C"}
		}
		builder.AddTable("Table "+string(rune('A'+i)), headers, rows)
	}

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build with long content returned nil")
	}

	// PDF should be larger due to multiple pages
	if len(pdf) < 1000 {
		t.Error("PDF with long content seems too small")
	}
}

func TestPDFReportSpecialCharacters(t *testing.T) {
	builder := NewPDFReportBuilder().
		WithTitle("Special (Characters) Test").
		WithAuthor("Author with \\backslash").
		AddSection("Section (with) parentheses", "Content with (special) \\characters\\.", 1)

	pdf := builder.Build()

	if pdf == nil {
		t.Fatal("Build with special characters returned nil")
	}
}

func TestPDFReportBuilderChaining(t *testing.T) {
	// Test that all methods can be chained properly
	report := &ReproducibilityReport{
		Title: "Chaining Test",
	}

	data := []MeasurementRow{{Turn: 1, DEff: 100}}

	pdf := NewPDFReportBuilder().
		WithConfig(nil).
		WithReport(report).
		WithTitle("Title").
		WithAuthor("Author").
		WithSubject("Subject").
		WithKeywords([]string{"test"}).
		WithToolVersion("1.0").
		AddSection("Section", "Content", 1).
		AddPlot("Plot", "Description", data, []PlotMetric{MetricDEff}).
		AddPlotWithDimensions("Plot2", "Desc", data, []PlotMetric{MetricBeta}, 400, 200).
		AddTable("Table", []string{"A"}, [][]string{{"1"}}).
		AddMeasurementsTable("Measurements", data).
		Build()

	if pdf == nil {
		t.Fatal("Chained build returned nil")
	}
}
