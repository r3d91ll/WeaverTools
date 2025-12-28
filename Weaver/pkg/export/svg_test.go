// Package export tests for SVG plot generation.
package export

import (
	"bytes"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// SVGConfig Tests
// -----------------------------------------------------------------------------

func TestDefaultSVGConfig(t *testing.T) {
	config := DefaultSVGConfig()

	if config == nil {
		t.Fatal("DefaultSVGConfig() returned nil")
	}
	if config.Width != 800 {
		t.Errorf("expected Width 800, got %d", config.Width)
	}
	if config.Height != 400 {
		t.Errorf("expected Height 400, got %d", config.Height)
	}
	if config.XAxisLabel != "Turn" {
		t.Errorf("expected XAxisLabel 'Turn', got %q", config.XAxisLabel)
	}
	if !config.ShowLegend {
		t.Error("expected ShowLegend to be true by default")
	}
	if !config.ShowGrid {
		t.Error("expected ShowGrid to be true by default")
	}
	if !config.ShowPoints {
		t.Error("expected ShowPoints to be true by default")
	}
	if config.FontFamily != "Arial, sans-serif" {
		t.Errorf("expected FontFamily 'Arial, sans-serif', got %q", config.FontFamily)
	}
	if config.Padding != 60 {
		t.Errorf("expected Padding 60, got %d", config.Padding)
	}
	if config.PointRadius != 4 {
		t.Errorf("expected PointRadius 4, got %f", config.PointRadius)
	}
	if config.LineWidth != 2 {
		t.Errorf("expected LineWidth 2, got %f", config.LineWidth)
	}
	if !config.IncludeMetadata {
		t.Error("expected IncludeMetadata to be true by default")
	}
}

// -----------------------------------------------------------------------------
// MetricInfo Tests
// -----------------------------------------------------------------------------

func TestGetMetricInfo_DEff(t *testing.T) {
	info := GetMetricInfo(MetricDEff)

	if info.Name != "Effective Dimensionality" {
		t.Errorf("expected Name 'Effective Dimensionality', got %q", info.Name)
	}
	if info.Symbol != "D_eff" {
		t.Errorf("expected Symbol 'D_eff', got %q", info.Symbol)
	}
	if info.Color == "" {
		t.Error("expected Color to be set")
	}
	if info.MinValue != 0 {
		t.Errorf("expected MinValue 0, got %f", info.MinValue)
	}
}

func TestGetMetricInfo_Beta(t *testing.T) {
	info := GetMetricInfo(MetricBeta)

	if info.Name != "Collapse Indicator" {
		t.Errorf("expected Name 'Collapse Indicator', got %q", info.Name)
	}
	if info.Symbol != "β" {
		t.Errorf("expected Symbol 'β', got %q", info.Symbol)
	}
}

func TestGetMetricInfo_Alignment(t *testing.T) {
	info := GetMetricInfo(MetricAlignment)

	if info.Name != "Alignment" {
		t.Errorf("expected Name 'Alignment', got %q", info.Name)
	}
	if info.MinValue != -1 {
		t.Errorf("expected MinValue -1, got %f", info.MinValue)
	}
	if info.MaxValue != 1 {
		t.Errorf("expected MaxValue 1, got %f", info.MaxValue)
	}
}

func TestGetMetricInfo_CPair(t *testing.T) {
	info := GetMetricInfo(MetricCPair)

	if info.Name != "Bilateral Conveyance" {
		t.Errorf("expected Name 'Bilateral Conveyance', got %q", info.Name)
	}
	if info.Symbol != "C_pair" {
		t.Errorf("expected Symbol 'C_pair', got %q", info.Symbol)
	}
}

func TestGetMetricInfo_Unknown(t *testing.T) {
	info := GetMetricInfo(PlotMetric("unknown"))

	if info.Name != "Unknown" {
		t.Errorf("expected Name 'Unknown', got %q", info.Name)
	}
}

// -----------------------------------------------------------------------------
// SVGPlotBuilder Tests
// -----------------------------------------------------------------------------

func TestNewSVGPlotBuilder_NilConfig(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)

	if builder == nil {
		t.Fatal("NewSVGPlotBuilder(nil) returned nil")
	}
	if builder.config == nil {
		t.Error("expected default config to be set")
	}
}

func TestSVGPlotBuilder_EmptySeries(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)
	result := builder.Build()

	if result != "" {
		t.Errorf("expected empty string for empty series, got %q", result)
	}
}

func TestSVGPlotBuilder_SinglePoint(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 1.5)

	result := builder.Build()

	if result == "" {
		t.Error("expected non-empty SVG")
	}

	// Verify SVG structure
	if !strings.Contains(result, "<?xml version") {
		t.Error("expected XML declaration")
	}
	if !strings.Contains(result, "<svg") {
		t.Error("expected <svg> element")
	}
	if !strings.Contains(result, "</svg>") {
		t.Error("expected </svg> closing tag")
	}
	if !strings.Contains(result, `xmlns="http://www.w3.org/2000/svg"`) {
		t.Error("expected SVG namespace")
	}
}

func TestSVGPlotBuilder_MultiplePoints(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)

	// Add multiple points
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricBeta, 2, 1.7)
	builder.AddPoint(MetricBeta, 3, 1.9)
	builder.AddPoint(MetricBeta, 4, 2.1)
	builder.AddPoint(MetricBeta, 5, 1.8)

	result := builder.Build()

	if result == "" {
		t.Error("expected non-empty SVG")
	}

	// Verify line path exists
	if !strings.Contains(result, "<path") {
		t.Error("expected <path> element for line")
	}

	// Verify data points exist
	if !strings.Contains(result, "<circle") {
		t.Error("expected <circle> elements for data points")
	}
}

func TestSVGPlotBuilder_AddSeries(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)

	series := DataSeries{
		Metric: MetricDEff,
		Points: []DataPoint{
			{Turn: 1, Value: 100},
			{Turn: 2, Value: 110},
			{Turn: 3, Value: 105},
		},
	}

	builder.AddSeries(series)
	result := builder.Build()

	if result == "" {
		t.Error("expected non-empty SVG")
	}
}

func TestSVGPlotBuilder_MultipleSeries(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)

	// Add Beta series
	builder.AddSeries(DataSeries{
		Metric: MetricBeta,
		Points: []DataPoint{
			{Turn: 1, Value: 1.5},
			{Turn: 2, Value: 1.7},
		},
	})

	// Add Alignment series
	builder.AddSeries(DataSeries{
		Metric: MetricAlignment,
		Points: []DataPoint{
			{Turn: 1, Value: 0.9},
			{Turn: 2, Value: 0.85},
		},
	})

	result := builder.Build()

	// Should have two paths (one per series)
	pathCount := strings.Count(result, "<path")
	if pathCount < 2 {
		t.Errorf("expected at least 2 paths, got %d", pathCount)
	}
}

func TestSVGPlotBuilder_AddPointCreatesNewSeries(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)

	// Add points for two different metrics
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricDEff, 1, 100)
	builder.AddPoint(MetricBeta, 2, 1.7)
	builder.AddPoint(MetricDEff, 2, 110)

	result := builder.Build()

	// Should have two series (two paths)
	pathCount := strings.Count(result, "<path")
	if pathCount < 2 {
		t.Errorf("expected at least 2 paths, got %d", pathCount)
	}
}

func TestSVGPlotBuilder_Chaining(t *testing.T) {
	result := NewSVGPlotBuilder(nil).
		AddPoint(MetricBeta, 1, 1.5).
		AddPoint(MetricBeta, 2, 1.7).
		AddPoint(MetricBeta, 3, 1.9).
		Build()

	if result == "" {
		t.Error("expected non-empty result from chained builder")
	}
}

// -----------------------------------------------------------------------------
// SVG Output Structure Tests
// -----------------------------------------------------------------------------

func TestSVGOutput_ValidXML(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricBeta, 2, 1.7)

	result := builder.Build()

	// Check for XML declaration
	if !strings.HasPrefix(result, "<?xml") {
		t.Error("expected XML declaration at start")
	}

	// Check for proper opening and closing tags
	if !strings.Contains(result, "<svg") {
		t.Error("expected <svg> opening tag")
	}
	if !strings.Contains(result, "</svg>") {
		t.Error("expected </svg> closing tag")
	}
}

func TestSVGOutput_Dimensions(t *testing.T) {
	config := DefaultSVGConfig()
	config.Width = 1000
	config.Height = 600

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)

	result := builder.Build()

	if !strings.Contains(result, `width="1000"`) {
		t.Error("expected width attribute to be 1000")
	}
	if !strings.Contains(result, `height="600"`) {
		t.Error("expected height attribute to be 600")
	}
	if !strings.Contains(result, `viewBox="0 0 1000 600"`) {
		t.Error("expected viewBox to match dimensions")
	}
}

func TestSVGOutput_Title(t *testing.T) {
	config := DefaultSVGConfig()
	config.Title = "Test Plot Title"

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)

	result := builder.Build()

	if !strings.Contains(result, "Test Plot Title") {
		t.Error("expected title in output")
	}
}

func TestSVGOutput_Grid(t *testing.T) {
	config := DefaultSVGConfig()
	config.ShowGrid = true

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricBeta, 5, 2.0)

	result := builder.Build()

	// Grid should have dashed lines
	if !strings.Contains(result, "stroke-dasharray") {
		t.Error("expected dashed grid lines")
	}
}

func TestSVGOutput_NoGrid(t *testing.T) {
	config := DefaultSVGConfig()
	config.ShowGrid = false

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricBeta, 5, 2.0)

	result := builder.Build()

	// Should not have grid class
	if strings.Contains(result, `class="grid"`) {
		t.Error("expected no grid when ShowGrid is false")
	}
}

func TestSVGOutput_NoPoints(t *testing.T) {
	config := DefaultSVGConfig()
	config.ShowPoints = false

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricBeta, 2, 1.7)

	result := builder.Build()

	// Should still have the line path
	if !strings.Contains(result, "<path") {
		t.Error("expected line path")
	}

	// But should have fewer circles (only legend markers if any)
	circleCount := strings.Count(result, "<circle")
	if circleCount >= 2 {
		t.Errorf("expected fewer data point circles when ShowPoints is false, got %d", circleCount)
	}
}

func TestSVGOutput_Legend(t *testing.T) {
	config := DefaultSVGConfig()
	config.ShowLegend = true

	builder := NewSVGPlotBuilder(config)

	// Add two series to trigger legend
	builder.AddSeries(DataSeries{
		Metric: MetricBeta,
		Label:  "Beta Series",
		Points: []DataPoint{{Turn: 1, Value: 1.5}},
	})
	builder.AddSeries(DataSeries{
		Metric: MetricAlignment,
		Label:  "Alignment Series",
		Points: []DataPoint{{Turn: 1, Value: 0.9}},
	})

	result := builder.Build()

	if !strings.Contains(result, `class="legend"`) {
		t.Error("expected legend group")
	}
}

func TestSVGOutput_NoLegendSingleSeries(t *testing.T) {
	config := DefaultSVGConfig()
	config.ShowLegend = true

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)

	result := builder.Build()

	// Legend should not appear for single series
	if strings.Contains(result, `class="legend"`) {
		t.Error("expected no legend for single series")
	}
}

func TestSVGOutput_Axes(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricBeta, 5, 2.0)

	result := builder.Build()

	// Should have axis class
	if !strings.Contains(result, `class="axes"`) {
		t.Error("expected axes group")
	}

	// Should have tick labels
	if !strings.Contains(result, `class="tick-label"`) {
		t.Error("expected tick labels")
	}
}

func TestSVGOutput_AxisLabels(t *testing.T) {
	config := DefaultSVGConfig()
	config.XAxisLabel = "Conversation Turn"
	config.YAxisLabel = "Beta Value"

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)

	result := builder.Build()

	if !strings.Contains(result, "Conversation Turn") {
		t.Error("expected X-axis label")
	}
	if !strings.Contains(result, "Beta Value") {
		t.Error("expected Y-axis label")
	}
}

func TestSVGOutput_Metadata(t *testing.T) {
	config := DefaultSVGConfig()
	config.IncludeMetadata = true
	config.ToolVersion = "1.0.0"

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)

	result := builder.Build()

	if !strings.Contains(result, "Generated by WeaverTools") {
		t.Error("expected generation metadata")
	}
	if !strings.Contains(result, "Tool version: 1.0.0") {
		t.Error("expected tool version in metadata")
	}
}

func TestSVGOutput_NoMetadata(t *testing.T) {
	config := DefaultSVGConfig()
	config.IncludeMetadata = false

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)

	result := builder.Build()

	if strings.Contains(result, "Generated by WeaverTools") {
		t.Error("expected no metadata when IncludeMetadata is false")
	}
}

// -----------------------------------------------------------------------------
// XML Escaping Tests
// -----------------------------------------------------------------------------

func TestSVGOutput_EscapesSpecialChars(t *testing.T) {
	config := DefaultSVGConfig()
	config.Title = "Test <script> & \"special\" 'chars'"

	builder := NewSVGPlotBuilder(config)
	builder.AddPoint(MetricBeta, 1, 1.5)

	result := builder.Build()

	// Special characters should be escaped
	if strings.Contains(result, "<script>") {
		t.Error("expected < to be escaped")
	}
	if !strings.Contains(result, "&lt;script&gt;") {
		t.Error("expected &lt; and &gt; in output")
	}
	if !strings.Contains(result, "&amp;") {
		t.Error("expected &amp; in output")
	}
}

func TestEscapeXML(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Hello", "Hello"},
		{"<script>", "&lt;script&gt;"},
		{"A & B", "A &amp; B"},
		{`"quoted"`, "&quot;quoted&quot;"},
		{"it's", "it&apos;s"},
		{"<>&\"'", "&lt;&gt;&amp;&quot;&apos;"},
	}

	for _, tc := range tests {
		result := escapeXML(tc.input)
		if result != tc.expected {
			t.Errorf("escapeXML(%q) = %q, expected %q", tc.input, result, tc.expected)
		}
	}
}

// -----------------------------------------------------------------------------
// Color Tests
// -----------------------------------------------------------------------------

func TestSVGOutput_SeriesColors(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)

	// Add series with default colors
	builder.AddSeries(DataSeries{
		Metric: MetricBeta,
		Points: []DataPoint{{Turn: 1, Value: 1.5}},
	})
	builder.AddSeries(DataSeries{
		Metric: MetricAlignment,
		Points: []DataPoint{{Turn: 1, Value: 0.9}},
	})

	result := builder.Build()

	// Should use metric default colors
	betaInfo := GetMetricInfo(MetricBeta)
	alignInfo := GetMetricInfo(MetricAlignment)

	if !strings.Contains(result, betaInfo.Color) {
		t.Errorf("expected beta color %s in output", betaInfo.Color)
	}
	if !strings.Contains(result, alignInfo.Color) {
		t.Errorf("expected alignment color %s in output", alignInfo.Color)
	}
}

func TestSVGOutput_CustomSeriesColor(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)

	builder.AddSeries(DataSeries{
		Metric: MetricBeta,
		Color:  "#ff5500",
		Points: []DataPoint{{Turn: 1, Value: 1.5}},
	})

	result := builder.Build()

	if !strings.Contains(result, "#ff5500") {
		t.Error("expected custom color #ff5500 in output")
	}
}

// -----------------------------------------------------------------------------
// Convenience Function Tests
// -----------------------------------------------------------------------------

func TestGenerateMeasurementPlot_Empty(t *testing.T) {
	rows := []MeasurementRow{}
	result := GenerateMeasurementPlot(rows, MetricBeta, nil)

	if result != "" {
		t.Error("expected empty result for empty rows")
	}
}

func TestGenerateMeasurementPlot_SingleRow(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85},
	}

	result := GenerateMeasurementPlot(rows, MetricBeta, nil)

	if result == "" {
		t.Error("expected non-empty SVG")
	}
	if !strings.Contains(result, "<svg") {
		t.Error("expected SVG output")
	}
}

func TestGenerateMeasurementPlot_MultipleRows(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85},
		{Turn: 2, Sender: "B", Receiver: "A", DEff: 110, Beta: 1.7, Alignment: 0.85, CPair: 0.82},
		{Turn: 3, Sender: "A", Receiver: "B", DEff: 105, Beta: 1.9, Alignment: 0.88, CPair: 0.80},
	}

	// Test each metric type
	metrics := []PlotMetric{MetricDEff, MetricBeta, MetricAlignment, MetricCPair}

	for _, metric := range metrics {
		result := GenerateMeasurementPlot(rows, metric, nil)
		if result == "" {
			t.Errorf("expected non-empty SVG for metric %s", metric)
		}
	}
}

func TestGenerateMeasurementPlot_WithConfig(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Beta: 1.5},
		{Turn: 2, Beta: 1.7},
	}

	config := &SVGConfig{
		Width:  1200,
		Height: 600,
		Title:  "Custom Beta Plot",
	}

	result := GenerateMeasurementPlot(rows, MetricBeta, config)

	if !strings.Contains(result, `width="1200"`) {
		t.Error("expected custom width")
	}
	if !strings.Contains(result, "Custom Beta Plot") {
		t.Error("expected custom title")
	}
}

func TestGenerateMultiMetricPlot(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85},
		{Turn: 2, DEff: 110, Beta: 1.7, Alignment: 0.85, CPair: 0.82},
	}

	metrics := []PlotMetric{MetricBeta, MetricAlignment}
	result := GenerateMultiMetricPlot(rows, metrics, nil)

	if result == "" {
		t.Error("expected non-empty SVG")
	}

	// Should have multiple paths
	pathCount := strings.Count(result, "<path")
	if pathCount < 2 {
		t.Errorf("expected at least 2 paths for 2 metrics, got %d", pathCount)
	}
}

func TestGenerateBetaPlotWithThresholds(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Beta: 1.5},
		{Turn: 2, Beta: 2.0},
		{Turn: 3, Beta: 2.5},
	}

	result := GenerateBetaPlotWithThresholds(rows, nil)

	if result == "" {
		t.Error("expected non-empty SVG")
	}
	if !strings.Contains(result, "Beta") {
		t.Error("expected Beta in title or labels")
	}
}

// -----------------------------------------------------------------------------
// WriteTo Tests
// -----------------------------------------------------------------------------

func TestSVGPlotBuilder_WriteTo(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricBeta, 2, 1.7)

	var buf bytes.Buffer
	n, err := builder.WriteTo(&buf)

	if err != nil {
		t.Errorf("WriteTo failed: %v", err)
	}
	if n == 0 {
		t.Error("expected non-zero bytes written")
	}
	if buf.Len() == 0 {
		t.Error("expected non-empty buffer")
	}

	content := buf.String()
	if !strings.Contains(content, "<svg") {
		t.Error("expected SVG content in buffer")
	}
}

func TestExportSVGToWriter(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Beta: 1.5},
		{Turn: 2, Beta: 1.7},
	}

	var buf bytes.Buffer
	err := ExportSVGToWriter(&buf, rows, MetricBeta, nil)

	if err != nil {
		t.Errorf("ExportSVGToWriter failed: %v", err)
	}
	if buf.Len() == 0 {
		t.Error("expected non-empty buffer")
	}
}

// -----------------------------------------------------------------------------
// Edge Case Tests
// -----------------------------------------------------------------------------

func TestSVGPlotBuilder_SingleValueRange(t *testing.T) {
	// All points have same Y value
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricBeta, 2, 1.5)
	builder.AddPoint(MetricBeta, 3, 1.5)

	result := builder.Build()

	// Should not panic and should produce valid SVG
	if result == "" {
		t.Error("expected non-empty SVG even with constant Y values")
	}
	if !strings.Contains(result, "<svg") {
		t.Error("expected valid SVG output")
	}
}

func TestSVGPlotBuilder_SingleXValue(t *testing.T) {
	// All points at same X position
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 5, 1.5)
	builder.AddPoint(MetricBeta, 5, 1.7)
	builder.AddPoint(MetricBeta, 5, 1.9)

	result := builder.Build()

	// Should not panic and should produce valid SVG
	if result == "" {
		t.Error("expected non-empty SVG even with same X values")
	}
}

func TestSVGPlotBuilder_NegativeValues(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricAlignment, 1, -0.5)
	builder.AddPoint(MetricAlignment, 2, 0.0)
	builder.AddPoint(MetricAlignment, 3, 0.5)

	result := builder.Build()

	if result == "" {
		t.Error("expected non-empty SVG with negative values")
	}
}

func TestSVGPlotBuilder_LargeDataset(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)

	// Add 100 points
	for i := 1; i <= 100; i++ {
		value := 1.5 + float64(i)*0.01
		builder.AddPoint(MetricBeta, i, value)
	}

	result := builder.Build()

	if result == "" {
		t.Error("expected non-empty SVG for large dataset")
	}

	// Should have reasonable size (not exploding)
	if len(result) > 100000 {
		t.Error("SVG output seems too large")
	}
}

func TestSVGPlotBuilder_ZeroTurn(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 0, 1.5)
	builder.AddPoint(MetricBeta, 1, 1.7)

	result := builder.Build()

	if result == "" {
		t.Error("expected non-empty SVG with turn 0")
	}
}

// -----------------------------------------------------------------------------
// Tick Calculation Tests
// -----------------------------------------------------------------------------

func TestCalculateIntTicks(t *testing.T) {
	tests := []struct {
		min, max, maxTicks int
		wantMin, wantMax   int
	}{
		{1, 10, 10, 1, 10},
		{0, 5, 10, 0, 5},
		{1, 100, 10, 1, 100},
	}

	for _, tc := range tests {
		ticks := calculateIntTicks(tc.min, tc.max, tc.maxTicks)

		if len(ticks) == 0 {
			t.Errorf("calculateIntTicks(%d, %d, %d) returned empty slice", tc.min, tc.max, tc.maxTicks)
			continue
		}

		if ticks[0] > tc.wantMin {
			t.Errorf("calculateIntTicks(%d, %d, %d): first tick %d > min %d", tc.min, tc.max, tc.maxTicks, ticks[0], tc.wantMin)
		}

		if ticks[len(ticks)-1] < tc.wantMax {
			t.Errorf("calculateIntTicks(%d, %d, %d): last tick %d < max %d", tc.min, tc.max, tc.maxTicks, ticks[len(ticks)-1], tc.wantMax)
		}
	}
}

func TestCalculateFloatTicks(t *testing.T) {
	tests := []struct {
		min, max float64
		maxTicks int
	}{
		{0, 1, 8},
		{0, 5, 8},
		{-1, 1, 8},
		{1.5, 2.5, 8},
	}

	for _, tc := range tests {
		ticks := calculateFloatTicks(tc.min, tc.max, tc.maxTicks)

		if len(ticks) == 0 {
			t.Errorf("calculateFloatTicks(%f, %f, %d) returned empty slice", tc.min, tc.max, tc.maxTicks)
			continue
		}

		if len(ticks) > tc.maxTicks+2 {
			t.Errorf("calculateFloatTicks(%f, %f, %d) returned too many ticks: %d", tc.min, tc.max, tc.maxTicks, len(ticks))
		}
	}
}

func TestScaleValue(t *testing.T) {
	tests := []struct {
		value, srcMin, srcMax, dstMin, dstMax, expected float64
	}{
		{0, 0, 100, 0, 200, 0},
		{50, 0, 100, 0, 200, 100},
		{100, 0, 100, 0, 200, 200},
		{0, 0, 1, 400, 0, 400},   // Inverted Y-axis
		{1, 0, 1, 400, 0, 0},     // Inverted Y-axis
		{0.5, 0, 1, 400, 0, 200}, // Inverted Y-axis
	}

	for _, tc := range tests {
		result := scaleValue(tc.value, tc.srcMin, tc.srcMax, tc.dstMin, tc.dstMax)
		if result != tc.expected {
			t.Errorf("scaleValue(%f, %f, %f, %f, %f) = %f, expected %f",
				tc.value, tc.srcMin, tc.srcMax, tc.dstMin, tc.dstMax, result, tc.expected)
		}
	}
}

// -----------------------------------------------------------------------------
// SVG Validation Tests
// -----------------------------------------------------------------------------

func TestSVGOutput_ValidSVG11Structure(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 1.5)
	builder.AddPoint(MetricBeta, 2, 1.7)

	result := builder.Build()

	// Check SVG 1.1 version
	if !strings.Contains(result, `version="1.1"`) {
		t.Error("expected SVG version 1.1")
	}

	// Check required namespace
	if !strings.Contains(result, SVGNamespace) {
		t.Error("expected SVG namespace")
	}

	// Check encoding
	if !strings.Contains(result, `encoding="UTF-8"`) {
		t.Error("expected UTF-8 encoding")
	}
}

func TestSVGOutput_ProperNesting(t *testing.T) {
	builder := NewSVGPlotBuilder(nil)
	builder.AddPoint(MetricBeta, 1, 1.5)

	result := builder.Build()

	// Check that tags are properly nested (basic check)
	opens := strings.Count(result, "<g")
	closes := strings.Count(result, "</g>")
	if opens != closes {
		t.Errorf("mismatched <g> tags: %d opens, %d closes", opens, closes)
	}
}

// -----------------------------------------------------------------------------
// Integration Tests
// -----------------------------------------------------------------------------

func TestIntegration_FullMeasurementWorkflow(t *testing.T) {
	// Create measurement data as would come from a session
	rows := []MeasurementRow{
		{Turn: 1, Sender: "Claude", Receiver: "User", DEff: 128, Beta: 1.65, Alignment: 0.92, CPair: 0.88, BetaStatus: "optimal"},
		{Turn: 2, Sender: "User", Receiver: "Claude", DEff: 135, Beta: 1.72, Alignment: 0.89, CPair: 0.85, BetaStatus: "optimal"},
		{Turn: 3, Sender: "Claude", Receiver: "User", DEff: 142, Beta: 1.95, Alignment: 0.78, CPair: 0.82, BetaStatus: "monitor"},
		{Turn: 4, Sender: "User", Receiver: "Claude", DEff: 130, Beta: 2.15, Alignment: 0.75, CPair: 0.79, BetaStatus: "monitor"},
		{Turn: 5, Sender: "Claude", Receiver: "User", DEff: 125, Beta: 1.85, Alignment: 0.88, CPair: 0.86, BetaStatus: "optimal"},
	}

	// Generate plot for each metric
	config := &SVGConfig{
		Width:           1000,
		Height:          500,
		Title:           "Conveyance Metrics Over Session",
		ShowLegend:      true,
		ShowGrid:        true,
		ShowPoints:      true,
		IncludeMetadata: true,
		ToolVersion:     "1.0.0",
	}

	// Generate multi-metric plot
	metrics := []PlotMetric{MetricBeta, MetricAlignment}
	result := GenerateMultiMetricPlot(rows, metrics, config)

	// Verify output
	if result == "" {
		t.Error("expected non-empty SVG")
	}

	// Check all required elements
	mustContain := []string{
		"<?xml version",
		"<svg",
		`version="1.1"`,
		"xmlns",
		"Conveyance Metrics Over Session",
		"<path",
		"<circle",
		"</svg>",
	}

	for _, s := range mustContain {
		if !strings.Contains(result, s) {
			t.Errorf("expected output to contain %q", s)
		}
	}
}
