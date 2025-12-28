// Package export provides academic export format utilities.
package export

import (
	"bytes"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// PDFConfig Tests
// -----------------------------------------------------------------------------

func TestDefaultPDFConfig(t *testing.T) {
	config := DefaultPDFConfig()

	if config == nil {
		t.Fatal("DefaultPDFConfig returned nil")
	}

	// Check default dimensions
	if config.Width != 612 {
		t.Errorf("expected Width=612, got %.2f", config.Width)
	}
	if config.Height != 396 {
		t.Errorf("expected Height=396, got %.2f", config.Height)
	}

	// Check default font settings
	if config.FontFamily != "Helvetica" {
		t.Errorf("expected FontFamily='Helvetica', got %q", config.FontFamily)
	}
	if config.FontSize != 10 {
		t.Errorf("expected FontSize=10, got %.2f", config.FontSize)
	}

	// Check default flags
	if !config.ShowLegend {
		t.Error("expected ShowLegend=true by default")
	}
	if !config.ShowGrid {
		t.Error("expected ShowGrid=true by default")
	}
	if !config.ShowPoints {
		t.Error("expected ShowPoints=true by default")
	}
	if !config.IncludeMetadata {
		t.Error("expected IncludeMetadata=true by default")
	}
	if !config.Compress {
		t.Error("expected Compress=true by default")
	}
}

func TestPDFConfigColors(t *testing.T) {
	config := DefaultPDFConfig()

	// Check default colors
	if config.BackgroundColor.R != 1 || config.BackgroundColor.G != 1 || config.BackgroundColor.B != 1 {
		t.Errorf("expected white background, got RGB(%.2f, %.2f, %.2f)",
			config.BackgroundColor.R, config.BackgroundColor.G, config.BackgroundColor.B)
	}

	if config.GridColor.R != 0.9 || config.GridColor.G != 0.9 || config.GridColor.B != 0.9 {
		t.Errorf("expected light gray grid color, got RGB(%.2f, %.2f, %.2f)",
			config.GridColor.R, config.GridColor.G, config.GridColor.B)
	}
}

// -----------------------------------------------------------------------------
// HexToPDFColor Tests
// -----------------------------------------------------------------------------

func TestHexToPDFColor(t *testing.T) {
	tests := []struct {
		name     string
		hex      string
		expected PDFColor
	}{
		{
			name:     "black",
			hex:      "#000000",
			expected: PDFColor{0, 0, 0},
		},
		{
			name:     "white",
			hex:      "#ffffff",
			expected: PDFColor{1, 1, 1},
		},
		{
			name:     "red",
			hex:      "#ff0000",
			expected: PDFColor{1, 0, 0},
		},
		{
			name:     "green",
			hex:      "#00ff00",
			expected: PDFColor{0, 1, 0},
		},
		{
			name:     "blue",
			hex:      "#0000ff",
			expected: PDFColor{0, 0, 1},
		},
		{
			name:     "metric blue",
			hex:      "#2563eb",
			expected: PDFColor{0.145, 0.388, 0.922},
		},
		{
			name:     "without hash",
			hex:      "ff0000",
			expected: PDFColor{1, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HexToPDFColor(tt.hex)

			// Allow small tolerance for floating point comparison
			tolerance := 0.01
			if diff := result.R - tt.expected.R; diff < -tolerance || diff > tolerance {
				t.Errorf("R: expected %.3f, got %.3f", tt.expected.R, result.R)
			}
			if diff := result.G - tt.expected.G; diff < -tolerance || diff > tolerance {
				t.Errorf("G: expected %.3f, got %.3f", tt.expected.G, result.G)
			}
			if diff := result.B - tt.expected.B; diff < -tolerance || diff > tolerance {
				t.Errorf("B: expected %.3f, got %.3f", tt.expected.B, result.B)
			}
		})
	}
}

func TestHexToPDFColorInvalid(t *testing.T) {
	// Invalid inputs should return black
	result := HexToPDFColor("invalid")
	if result.R != 0 || result.G != 0 || result.B != 0 {
		t.Errorf("expected black for invalid input, got RGB(%.2f, %.2f, %.2f)", result.R, result.G, result.B)
	}

	result = HexToPDFColor("#fff") // Too short
	if result.R != 0 || result.G != 0 || result.B != 0 {
		t.Errorf("expected black for short input, got RGB(%.2f, %.2f, %.2f)", result.R, result.G, result.B)
	}
}

func TestPDFColorString(t *testing.T) {
	color := PDFColor{0.5, 0.25, 0.75}
	result := color.String()

	if !strings.Contains(result, "0.500") {
		t.Errorf("expected R=0.500 in output, got %q", result)
	}
	if !strings.Contains(result, "0.250") {
		t.Errorf("expected G=0.250 in output, got %q", result)
	}
	if !strings.Contains(result, "0.750") {
		t.Errorf("expected B=0.750 in output, got %q", result)
	}
}

// -----------------------------------------------------------------------------
// PDFPlotBuilder Tests
// -----------------------------------------------------------------------------

func TestNewPDFPlotBuilder(t *testing.T) {
	// With nil config, should use defaults
	builder := NewPDFPlotBuilder(nil)
	if builder == nil {
		t.Fatal("NewPDFPlotBuilder returned nil")
	}

	// With custom config
	config := &PDFConfig{
		Width:  800,
		Height: 600,
	}
	builder = NewPDFPlotBuilder(config)
	if builder == nil {
		t.Fatal("NewPDFPlotBuilder with config returned nil")
	}
}

func TestPDFPlotBuilderAddSeries(t *testing.T) {
	builder := NewPDFPlotBuilder(nil)

	series := DataSeries{
		Metric: MetricBeta,
		Points: []DataPoint{
			{Turn: 1, Value: 0.5},
			{Turn: 2, Value: 0.7},
		},
	}

	result := builder.AddSeries(series)

	// Should return the builder for chaining
	if result != builder {
		t.Error("AddSeries should return the builder for chaining")
	}
}

func TestPDFPlotBuilderAddPoint(t *testing.T) {
	builder := NewPDFPlotBuilder(nil)

	// Add points to the same metric
	builder.AddPoint(MetricBeta, 1, 0.5)
	builder.AddPoint(MetricBeta, 2, 0.7)
	builder.AddPoint(MetricBeta, 3, 0.6)

	// Add points to a different metric
	builder.AddPoint(MetricDEff, 1, 128)
	builder.AddPoint(MetricDEff, 2, 135)

	// Build should produce output
	pdf := builder.Build()
	if pdf == nil {
		t.Fatal("Build returned nil after adding points")
	}
	if len(pdf) == 0 {
		t.Fatal("Build returned empty PDF")
	}
}

func TestPDFPlotBuilderEmptySeries(t *testing.T) {
	builder := NewPDFPlotBuilder(nil)

	// Build with no series should return nil
	pdf := builder.Build()
	if pdf != nil {
		t.Error("expected nil for empty builder")
	}
}

// -----------------------------------------------------------------------------
// PDF Output Structure Tests
// -----------------------------------------------------------------------------

func TestPDFOutputStructure(t *testing.T) {
	builder := NewPDFPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 0.5)
	builder.AddPoint(MetricBeta, 2, 0.7)

	pdf := builder.Build()
	pdfStr := string(pdf)

	// Check PDF header
	if !strings.HasPrefix(pdfStr, "%PDF-"+PDFVersion) {
		t.Errorf("expected PDF header, got prefix: %q", pdfStr[:min(20, len(pdfStr))])
	}

	// Check PDF trailer
	if !strings.HasSuffix(pdfStr, "%%EOF\n") {
		t.Errorf("expected %%EOF at end, got suffix: %q", pdfStr[max(0, len(pdfStr)-20):])
	}

	// Check for catalog
	if !strings.Contains(pdfStr, "/Type /Catalog") {
		t.Error("expected /Catalog in PDF")
	}

	// Check for pages
	if !strings.Contains(pdfStr, "/Type /Pages") {
		t.Error("expected /Pages in PDF")
	}

	// Check for page
	if !strings.Contains(pdfStr, "/Type /Page") {
		t.Error("expected /Page in PDF")
	}

	// Check for font
	if !strings.Contains(pdfStr, "/Type /Font") {
		t.Error("expected /Font in PDF")
	}
	if !strings.Contains(pdfStr, "/BaseFont /Helvetica") {
		t.Error("expected Helvetica font in PDF")
	}
}

func TestPDFOutputWithMetadata(t *testing.T) {
	config := DefaultPDFConfig()
	config.Title = "Test Plot"
	config.Author = "Test Author"
	config.Subject = "Test Subject"
	config.Keywords = []string{"test", "plot", "metrics"}
	config.ToolVersion = "1.0.0"
	config.IncludeMetadata = true

	builder := NewPDFPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 0.5)

	pdf := builder.Build()
	pdfStr := string(pdf)

	// Check metadata presence
	if !strings.Contains(pdfStr, "/Title (Test Plot)") {
		t.Error("expected title in metadata")
	}
	if !strings.Contains(pdfStr, "/Author (Test Author)") {
		t.Error("expected author in metadata")
	}
	if !strings.Contains(pdfStr, "/Subject (Test Subject)") {
		t.Error("expected subject in metadata")
	}
	if !strings.Contains(pdfStr, "/Keywords (test, plot, metrics)") {
		t.Error("expected keywords in metadata")
	}
	if !strings.Contains(pdfStr, "/Producer (WeaverTools Export Package)") {
		t.Error("expected producer in metadata")
	}
	if !strings.Contains(pdfStr, "/Creator (WeaverTools 1.0.0)") {
		t.Error("expected creator with version in metadata")
	}
	if !strings.Contains(pdfStr, "/CreationDate") {
		t.Error("expected creation date in metadata")
	}
}

func TestPDFOutputWithoutMetadata(t *testing.T) {
	config := DefaultPDFConfig()
	config.IncludeMetadata = false

	builder := NewPDFPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 0.5)

	pdf := builder.Build()
	pdfStr := string(pdf)

	// Should not contain info dictionary entries
	if strings.Contains(pdfStr, "/Info") {
		t.Error("expected no /Info reference when metadata disabled")
	}
}

func TestPDFOutputWithCompression(t *testing.T) {
	config := DefaultPDFConfig()
	config.Compress = true

	builder := NewPDFPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 0.5)

	pdf := builder.Build()
	pdfStr := string(pdf)

	if !strings.Contains(pdfStr, "/Filter /FlateDecode") {
		t.Error("expected FlateDecode filter when compression enabled")
	}
}

func TestPDFOutputWithoutCompression(t *testing.T) {
	config := DefaultPDFConfig()
	config.Compress = false

	builder := NewPDFPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 0.5)

	pdf := builder.Build()
	pdfStr := string(pdf)

	if strings.Contains(pdfStr, "/Filter /FlateDecode") {
		t.Error("expected no FlateDecode filter when compression disabled")
	}
}

// -----------------------------------------------------------------------------
// WriteTo Tests
// -----------------------------------------------------------------------------

func TestPDFWriteTo(t *testing.T) {
	builder := NewPDFPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 0.5)
	builder.AddPoint(MetricBeta, 2, 0.7)

	var buf bytes.Buffer
	n, err := builder.WriteTo(&buf)

	if err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}
	if n == 0 {
		t.Error("WriteTo wrote 0 bytes")
	}
	if int64(buf.Len()) != n {
		t.Errorf("WriteTo reported %d bytes but buffer has %d", n, buf.Len())
	}

	// Verify content
	if !strings.HasPrefix(buf.String(), "%PDF-") {
		t.Error("WriteTo output doesn't start with PDF header")
	}
}

func TestPDFWriteToEmpty(t *testing.T) {
	builder := NewPDFPlotBuilder(nil)

	var buf bytes.Buffer
	n, err := builder.WriteTo(&buf)

	if err != nil {
		t.Fatalf("WriteTo on empty builder failed: %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes for empty builder, got %d", n)
	}
}

// -----------------------------------------------------------------------------
// Convenience Function Tests
// -----------------------------------------------------------------------------

func TestGenerateMeasurementPDFPlot(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, DEff: 128, Beta: 1.5, Alignment: 0.8, CPair: 0.75},
		{Turn: 2, DEff: 135, Beta: 1.7, Alignment: 0.85, CPair: 0.78},
		{Turn: 3, DEff: 142, Beta: 1.6, Alignment: 0.82, CPair: 0.80},
	}

	// Test each metric type
	metrics := []PlotMetric{MetricDEff, MetricBeta, MetricAlignment, MetricCPair}

	for _, metric := range metrics {
		t.Run(string(metric), func(t *testing.T) {
			pdf := GenerateMeasurementPDFPlot(rows, metric, nil)
			if pdf == nil {
				t.Fatal("GenerateMeasurementPDFPlot returned nil")
			}
			if !strings.HasPrefix(string(pdf), "%PDF-") {
				t.Error("output is not valid PDF")
			}
		})
	}
}

func TestGenerateMeasurementPDFPlotWithConfig(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, DEff: 128, Beta: 1.5, Alignment: 0.8, CPair: 0.75},
	}

	config := &PDFConfig{
		Width:       400,
		Height:      300,
		Title:       "Custom Title",
		ToolVersion: "2.0.0",
	}

	pdf := GenerateMeasurementPDFPlot(rows, MetricBeta, config)
	if pdf == nil {
		t.Fatal("GenerateMeasurementPDFPlot with config returned nil")
	}

	pdfStr := string(pdf)
	if !strings.Contains(pdfStr, "Custom Title") {
		t.Error("expected custom title in output")
	}
}

func TestGenerateMultiMetricPDFPlot(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, DEff: 128, Beta: 1.5, Alignment: 0.8, CPair: 0.75},
		{Turn: 2, DEff: 135, Beta: 1.7, Alignment: 0.85, CPair: 0.78},
	}

	metrics := []PlotMetric{MetricBeta, MetricAlignment}

	pdf := GenerateMultiMetricPDFPlot(rows, metrics, nil)
	if pdf == nil {
		t.Fatal("GenerateMultiMetricPDFPlot returned nil")
	}
	if !strings.HasPrefix(string(pdf), "%PDF-") {
		t.Error("output is not valid PDF")
	}
}

func TestGenerateBetaPDFPlotWithThresholds(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Beta: 1.5},
		{Turn: 2, Beta: 2.1},
		{Turn: 3, Beta: 2.8},
	}

	pdf := GenerateBetaPDFPlotWithThresholds(rows, nil)
	if pdf == nil {
		t.Fatal("GenerateBetaPDFPlotWithThresholds returned nil")
	}

	pdfStr := string(pdf)
	if !strings.Contains(pdfStr, "Beta (Collapse Indicator) Over Time") {
		t.Error("expected title in output")
	}
}

func TestExportPDFToWriter(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Beta: 1.5},
		{Turn: 2, Beta: 1.7},
	}

	var buf bytes.Buffer
	err := ExportPDFToWriter(&buf, rows, MetricBeta, nil)

	if err != nil {
		t.Fatalf("ExportPDFToWriter failed: %v", err)
	}
	if buf.Len() == 0 {
		t.Error("ExportPDFToWriter wrote empty output")
	}
}

// -----------------------------------------------------------------------------
// PDF Dimension Helpers Tests
// -----------------------------------------------------------------------------

func TestPDFPointsFromInches(t *testing.T) {
	tests := []struct {
		inches   float64
		expected float64
	}{
		{1.0, 72.0},
		{8.5, 612.0},
		{11.0, 792.0},
		{0.5, 36.0},
	}

	for _, tt := range tests {
		result := PDFPointsFromInches(tt.inches)
		if result != tt.expected {
			t.Errorf("PDFPointsFromInches(%.2f) = %.2f, want %.2f", tt.inches, result, tt.expected)
		}
	}
}

func TestPDFPointsFromMillimeters(t *testing.T) {
	// 25.4 mm = 1 inch = 72 points
	result := PDFPointsFromMillimeters(25.4)
	if result != 72.0 {
		t.Errorf("PDFPointsFromMillimeters(25.4) = %.2f, want 72.0", result)
	}

	// A4 width = 210mm = ~595.28 points
	result = PDFPointsFromMillimeters(210)
	expected := 595.276
	tolerance := 0.01
	if diff := result - expected; diff < -tolerance || diff > tolerance {
		t.Errorf("PDFPointsFromMillimeters(210) = %.3f, want ~%.3f", result, expected)
	}
}

func TestPDFDimensions(t *testing.T) {
	// Test Letter size
	if PDFDimensions.Letter.Width != 612 {
		t.Errorf("Letter width = %.2f, want 612", PDFDimensions.Letter.Width)
	}
	if PDFDimensions.Letter.Height != 792 {
		t.Errorf("Letter height = %.2f, want 792", PDFDimensions.Letter.Height)
	}

	// Test A4 size (approximately)
	expectedA4Width := 595.276
	tolerance := 0.01
	if diff := PDFDimensions.A4.Width - expectedA4Width; diff < -tolerance || diff > tolerance {
		t.Errorf("A4 width = %.3f, want ~%.3f", PDFDimensions.A4.Width, expectedA4Width)
	}

	// Test Figure size
	if PDFDimensions.Figure.Width != 612 {
		t.Errorf("Figure width = %.2f, want 612", PDFDimensions.Figure.Width)
	}
	if PDFDimensions.Figure.Height != 396 {
		t.Errorf("Figure height = %.2f, want 396", PDFDimensions.Figure.Height)
	}
}

// -----------------------------------------------------------------------------
// Edge Case Tests
// -----------------------------------------------------------------------------

func TestPDFSinglePoint(t *testing.T) {
	builder := NewPDFPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 0.5)

	pdf := builder.Build()
	if pdf == nil {
		t.Fatal("Build returned nil for single point")
	}
	if !strings.HasPrefix(string(pdf), "%PDF-") {
		t.Error("output is not valid PDF")
	}
}

func TestPDFIdenticalValues(t *testing.T) {
	// All points with the same Y value
	builder := NewPDFPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 1.0)
	builder.AddPoint(MetricBeta, 2, 1.0)
	builder.AddPoint(MetricBeta, 3, 1.0)

	pdf := builder.Build()
	if pdf == nil {
		t.Fatal("Build returned nil for identical values")
	}
}

func TestPDFIdenticalTurns(t *testing.T) {
	// All points with the same X value
	builder := NewPDFPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 5, 0.5)
	builder.AddPoint(MetricBeta, 5, 0.7)
	builder.AddPoint(MetricBeta, 5, 0.6)

	pdf := builder.Build()
	if pdf == nil {
		t.Fatal("Build returned nil for identical turns")
	}
}

func TestPDFNegativeValues(t *testing.T) {
	builder := NewPDFPlotBuilder(nil)
	builder.AddPoint(MetricAlignment, 1, -0.5)
	builder.AddPoint(MetricAlignment, 2, 0.0)
	builder.AddPoint(MetricAlignment, 3, 0.5)

	pdf := builder.Build()
	if pdf == nil {
		t.Fatal("Build returned nil for negative values")
	}
}

func TestPDFLargeDataset(t *testing.T) {
	builder := NewPDFPlotBuilder(nil)

	// Add 100 data points
	for i := 0; i < 100; i++ {
		value := 1.5 + float64(i%10)*0.1
		builder.AddPoint(MetricBeta, i, value)
	}

	pdf := builder.Build()
	if pdf == nil {
		t.Fatal("Build returned nil for large dataset")
	}

	// PDF should be reasonable size (not exploded)
	if len(pdf) > 100000 {
		t.Errorf("PDF seems too large: %d bytes", len(pdf))
	}
}

func TestPDFSpecialCharactersInTitle(t *testing.T) {
	config := DefaultPDFConfig()
	config.Title = "Test (with) special \\ characters"

	builder := NewPDFPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 0.5)

	pdf := builder.Build()
	if pdf == nil {
		t.Fatal("Build returned nil with special characters")
	}

	// Verify special characters are escaped
	pdfStr := string(pdf)
	if strings.Contains(pdfStr, "(Test (with)") {
		t.Error("parentheses not properly escaped")
	}
}

func TestPDFEmptyTitle(t *testing.T) {
	config := DefaultPDFConfig()
	config.Title = ""

	builder := NewPDFPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 0.5)

	pdf := builder.Build()
	if pdf == nil {
		t.Fatal("Build returned nil with empty title")
	}
}

// -----------------------------------------------------------------------------
// escapePDFString Tests
// -----------------------------------------------------------------------------

func TestEscapePDFString(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"simple", "simple"},
		{"with (parens)", "with \\(parens\\)"},
		{"back\\slash", "back\\\\slash"},
		{"multiple (()), \\", "multiple \\(\\(\\)\\), \\\\"},
	}

	for _, tt := range tests {
		result := escapePDFString(tt.input)
		if result != tt.expected {
			t.Errorf("escapePDFString(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

// -----------------------------------------------------------------------------
// Integration Tests
// -----------------------------------------------------------------------------

func TestPDFFullWorkflow(t *testing.T) {
	// Create sample data
	rows := []MeasurementRow{
		{Turn: 1, Sender: "Alice", Receiver: "Bob", DEff: 128, Beta: 1.5, Alignment: 0.8, CPair: 0.75, BetaStatus: "optimal"},
		{Turn: 2, Sender: "Bob", Receiver: "Alice", DEff: 135, Beta: 1.7, Alignment: 0.85, CPair: 0.78, BetaStatus: "optimal"},
		{Turn: 3, Sender: "Alice", Receiver: "Bob", DEff: 142, Beta: 1.6, Alignment: 0.82, CPair: 0.80, BetaStatus: "optimal"},
		{Turn: 4, Sender: "Bob", Receiver: "Alice", DEff: 130, Beta: 2.1, Alignment: 0.75, CPair: 0.70, BetaStatus: "monitor"},
	}

	// Configure PDF for academic publication
	config := &PDFConfig{
		Width:           PDFDimensions.Figure.Width,
		Height:          PDFDimensions.Figure.Height,
		Title:           "Collapse Indicator Over Time",
		XAxisLabel:      "Conversation Turn",
		YAxisLabel:      "Beta",
		ShowLegend:      true,
		ShowGrid:        true,
		ShowPoints:      true,
		FontFamily:      "Helvetica",
		FontSize:        10,
		Padding:         50,
		PointRadius:     3,
		LineWidth:       1.5,
		IncludeMetadata: true,
		ToolVersion:     "1.0.0",
		Author:          "WeaverTools Research Team",
		Subject:         "Conveyance Metrics Analysis",
		Keywords:        []string{"conveyance", "metrics", "LLM", "collapse"},
		Compress:        true,
	}

	// Generate PDF
	pdf := GenerateMeasurementPDFPlot(rows, MetricBeta, config)
	if pdf == nil {
		t.Fatal("GenerateMeasurementPDFPlot returned nil")
	}

	// Verify structure
	pdfStr := string(pdf)
	if !strings.HasPrefix(pdfStr, "%PDF-"+PDFVersion) {
		t.Error("invalid PDF header")
	}
	if !strings.HasSuffix(pdfStr, "%%EOF\n") {
		t.Error("invalid PDF trailer")
	}

	// Verify metadata
	if !strings.Contains(pdfStr, "Collapse Indicator Over Time") {
		t.Error("expected title in PDF")
	}
	if !strings.Contains(pdfStr, "/Author (WeaverTools Research Team)") {
		t.Error("expected author in PDF")
	}

	// Write to buffer to verify WriteTo works
	var buf bytes.Buffer
	builder := NewPDFPlotBuilder(config)
	for _, row := range rows {
		builder.AddPoint(MetricBeta, row.Turn, row.Beta)
	}
	_, err := builder.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}
	if buf.Len() == 0 {
		t.Error("WriteTo wrote empty content")
	}
}

func TestPDFMultiSeriesWorkflow(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, DEff: 128, Beta: 1.5, Alignment: 0.8, CPair: 0.75},
		{Turn: 2, DEff: 135, Beta: 1.7, Alignment: 0.85, CPair: 0.78},
		{Turn: 3, DEff: 142, Beta: 1.6, Alignment: 0.82, CPair: 0.80},
	}

	config := DefaultPDFConfig()
	config.Title = "Multi-Metric Comparison"
	config.YAxisLabel = "Normalized Value"

	metrics := []PlotMetric{MetricBeta, MetricAlignment, MetricCPair}
	pdf := GenerateMultiMetricPDFPlot(rows, metrics, config)

	if pdf == nil {
		t.Fatal("GenerateMultiMetricPDFPlot returned nil")
	}
	if len(pdf) == 0 {
		t.Fatal("GenerateMultiMetricPDFPlot returned empty PDF")
	}
}

// min returns the smaller of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the larger of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
