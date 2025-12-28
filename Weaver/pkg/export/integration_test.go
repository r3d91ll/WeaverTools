// Package export provides academic export format utilities.
// This file contains integration tests for the full export workflow.
package export

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"
)

// -----------------------------------------------------------------------------
// Test Data Fixtures
// -----------------------------------------------------------------------------

// createTestMeasurementRows creates a set of test measurement data.
// This simulates a conversation with multiple turns between agents.
func createTestMeasurementRows() []MeasurementRow {
	return []MeasurementRow{
		{
			Turn:       1,
			Sender:     "Alice",
			Receiver:   "Bob",
			DEff:       128,
			Beta:       1.75,
			Alignment:  0.923,
			CPair:      0.856,
			BetaStatus: "optimal",
		},
		{
			Turn:       2,
			Sender:     "Bob",
			Receiver:   "Alice",
			DEff:       124,
			Beta:       1.92,
			Alignment:  0.891,
			CPair:      0.834,
			BetaStatus: "optimal",
		},
		{
			Turn:       3,
			Sender:     "Alice",
			Receiver:   "Bob",
			DEff:       119,
			Beta:       2.15,
			Alignment:  0.876,
			CPair:      0.812,
			BetaStatus: "monitor",
		},
		{
			Turn:       4,
			Sender:     "Bob",
			Receiver:   "Alice",
			DEff:       115,
			Beta:       2.38,
			Alignment:  0.854,
			CPair:      0.789,
			BetaStatus: "monitor",
		},
		{
			Turn:       5,
			Sender:     "Alice",
			Receiver:   "Bob",
			DEff:       110,
			Beta:       2.67,
			Alignment:  0.832,
			CPair:      0.765,
			BetaStatus: "concerning",
		},
	}
}

// createTestCSVMeasurements creates CSV measurement data for testing.
func createTestCSVMeasurements() []*CSVMeasurement {
	now := time.Now()
	return []*CSVMeasurement{
		{
			ID:           "meas-001",
			Timestamp:    now,
			SessionID:    "session-test-001",
			TurnNumber:   1,
			SenderID:     "agent-alice",
			SenderName:   "Alice",
			SenderRole:   "user",
			ReceiverID:   "agent-bob",
			ReceiverName: "Bob",
			ReceiverRole: "assistant",
			DEff:         128,
			Beta:         1.75,
			Alignment:    0.923,
			CPair:        0.856,
			BetaStatus:   "optimal",
			IsUnilateral: false,
		},
		{
			ID:           "meas-002",
			Timestamp:    now.Add(time.Second * 30),
			SessionID:    "session-test-001",
			TurnNumber:   2,
			SenderID:     "agent-bob",
			SenderName:   "Bob",
			SenderRole:   "assistant",
			ReceiverID:   "agent-alice",
			ReceiverName: "Alice",
			ReceiverRole: "user",
			DEff:         124,
			Beta:         1.92,
			Alignment:    0.891,
			CPair:        0.834,
			BetaStatus:   "optimal",
			IsUnilateral: false,
		},
		{
			ID:           "meas-003",
			Timestamp:    now.Add(time.Minute),
			SessionID:    "session-test-001",
			TurnNumber:   3,
			SenderID:     "agent-alice",
			SenderName:   "Alice",
			SenderRole:   "user",
			ReceiverID:   "agent-bob",
			ReceiverName: "Bob",
			ReceiverRole: "assistant",
			DEff:         119,
			Beta:         2.15,
			Alignment:    0.876,
			CPair:        0.812,
			BetaStatus:   "monitor",
			IsUnilateral: false,
		},
	}
}

// createTestAgentConfigs creates agent configurations for testing.
func createTestAgentConfigs() []AgentConfig {
	return []AgentConfig{
		{
			ID:       "agent-alice",
			Name:     "Alice",
			Type:     "human",
			Provider: "local",
		},
		{
			ID:       "agent-bob",
			Name:     "Bob",
			Type:     "llm",
			Model:    "claude-3-opus-20240229",
			Provider: "anthropic",
			Parameters: map[string]string{
				"temperature": "0.7",
				"max_tokens":  "1024",
			},
		},
	}
}

// -----------------------------------------------------------------------------
// Full Workflow Integration Tests
// -----------------------------------------------------------------------------

// TestFullExportWorkflow tests the complete export workflow from measurements to all formats.
func TestFullExportWorkflow(t *testing.T) {
	// Create test data
	rows := createTestMeasurementRows()
	csvMeasurements := createTestCSVMeasurements()
	agents := createTestAgentConfigs()

	// Test that we have valid test data
	if len(rows) == 0 {
		t.Fatal("No test measurement rows created")
	}
	if len(csvMeasurements) == 0 {
		t.Fatal("No test CSV measurements created")
	}

	// Export to all formats and verify
	t.Run("LaTeX table export", func(t *testing.T) {
		latex := GenerateMeasurementTable(rows, nil)
		if latex == "" {
			t.Error("LaTeX table generation returned empty string")
		}

		// Verify LaTeX structure
		if !strings.Contains(latex, "\\begin{tabular}") {
			t.Error("LaTeX output missing \\begin{tabular}")
		}
		if !strings.Contains(latex, "\\end{tabular}") {
			t.Error("LaTeX output missing \\end{tabular}")
		}
		if !strings.Contains(latex, "\\toprule") {
			t.Error("LaTeX output missing \\toprule (booktabs style)")
		}
		if !strings.Contains(latex, "\\midrule") {
			t.Error("LaTeX output missing \\midrule (booktabs style)")
		}
		if !strings.Contains(latex, "\\bottomrule") {
			t.Error("LaTeX output missing \\bottomrule (booktabs style)")
		}

		// Verify data is present
		if !strings.Contains(latex, "Alice") {
			t.Error("LaTeX output missing sender name 'Alice'")
		}
		if !strings.Contains(latex, "Bob") {
			t.Error("LaTeX output missing receiver name 'Bob'")
		}
	})

	t.Run("LaTeX summary table export", func(t *testing.T) {
		stats := ComputeSummaryStats(rows)
		if stats.MeasurementCount != len(rows) {
			t.Errorf("SummaryStats.MeasurementCount = %d, want %d", stats.MeasurementCount, len(rows))
		}

		latex := GenerateSummaryTable(stats, nil)
		if latex == "" {
			t.Error("Summary table generation returned empty string")
		}
		if !strings.Contains(latex, "Measurements") {
			t.Error("Summary table missing 'Measurements' label")
		}
	})

	t.Run("CSV export", func(t *testing.T) {
		var buf bytes.Buffer
		err := ExportMeasurementsToCSV(&buf, csvMeasurements, nil)
		if err != nil {
			t.Fatalf("CSV export failed: %v", err)
		}

		csvContent := buf.String()
		if csvContent == "" {
			t.Error("CSV export returned empty string")
		}

		// Verify CSV can be parsed
		reader := csv.NewReader(strings.NewReader(csvContent))
		records, err := reader.ReadAll()
		if err != nil {
			t.Fatalf("CSV parsing failed: %v", err)
		}

		// Verify header row
		if len(records) < 2 {
			t.Fatal("CSV should have at least header and one data row")
		}

		headers := records[0]
		expectedHeaders := []string{"id", "timestamp", "session_id", "d_eff", "beta"}
		for _, expected := range expectedHeaders {
			found := false
			for _, h := range headers {
				if h == expected {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("CSV missing expected header: %s", expected)
			}
		}

		// Verify data row count (header + data)
		if len(records) != len(csvMeasurements)+1 {
			t.Errorf("CSV row count = %d, want %d (header + %d data rows)",
				len(records), len(csvMeasurements)+1, len(csvMeasurements))
		}
	})

	t.Run("CSV metadata export", func(t *testing.T) {
		var buf bytes.Buffer
		err := ExportMetadataToJSON(&buf, nil)
		if err != nil {
			t.Fatalf("Metadata export failed: %v", err)
		}

		var metadata DataDictionaryMetadata
		if err := json.Unmarshal(buf.Bytes(), &metadata); err != nil {
			t.Fatalf("Metadata JSON parsing failed: %v", err)
		}

		if metadata.Version == "" {
			t.Error("Metadata missing version")
		}
		if len(metadata.Columns) == 0 {
			t.Error("Metadata has no column definitions")
		}
	})

	t.Run("SVG plot export", func(t *testing.T) {
		svg := GenerateMeasurementPlot(rows, MetricBeta, nil)
		if svg == "" {
			t.Error("SVG generation returned empty string")
		}

		// Verify SVG structure
		if !strings.Contains(svg, "<?xml") {
			t.Error("SVG missing XML declaration")
		}
		if !strings.Contains(svg, "<svg") {
			t.Error("SVG missing <svg> element")
		}
		if !strings.Contains(svg, "</svg>") {
			t.Error("SVG missing </svg> closing tag")
		}
		if !strings.Contains(svg, "xmlns=\"http://www.w3.org/2000/svg\"") {
			t.Error("SVG missing namespace declaration")
		}
	})

	t.Run("SVG multi-metric plot export", func(t *testing.T) {
		metrics := []PlotMetric{MetricBeta, MetricAlignment, MetricCPair}
		svg := GenerateMultiMetricPlot(rows, metrics, nil)
		if svg == "" {
			t.Error("Multi-metric SVG generation returned empty string")
		}

		// Should contain multiple data paths
		pathCount := strings.Count(svg, "<path d=")
		if pathCount < len(metrics) {
			t.Errorf("Multi-metric SVG should have at least %d paths, got %d", len(metrics), pathCount)
		}
	})

	t.Run("PDF plot export", func(t *testing.T) {
		pdf := GenerateMeasurementPDFPlot(rows, MetricBeta, nil)
		if len(pdf) == 0 {
			t.Error("PDF generation returned empty bytes")
		}

		// Verify PDF header
		if len(pdf) < 8 || string(pdf[:5]) != "%PDF-" {
			t.Error("PDF missing PDF header signature")
		}

		// Verify PDF has expected structure
		pdfStr := string(pdf)
		if !strings.Contains(pdfStr, "/Type /Page") {
			t.Error("PDF missing page definition")
		}
		if !strings.Contains(pdfStr, "%%EOF") {
			t.Error("PDF missing EOF marker")
		}
	})

	t.Run("BibTeX citation export", func(t *testing.T) {
		entry := GenerateWeaverCitation("1.0.0", nil)
		if entry == nil {
			t.Fatal("BibTeX generation returned nil")
		}

		bibtex := entry.String()
		if bibtex == "" {
			t.Error("BibTeX entry string is empty")
		}

		// Verify required fields
		if !strings.Contains(bibtex, "@software") && !strings.Contains(bibtex, "@misc") {
			t.Error("BibTeX missing entry type")
		}
		if !strings.Contains(bibtex, "author =") {
			t.Error("BibTeX missing author field")
		}
		if !strings.Contains(bibtex, "title =") {
			t.Error("BibTeX missing title field")
		}
		if !strings.Contains(bibtex, "year =") {
			t.Error("BibTeX missing year field")
		}
	})

	t.Run("BibTeX with experiment hash", func(t *testing.T) {
		now := time.Now()
		hash := ComputeExperimentHash("1.0.0", "passive", 5, 1, now, nil, nil)

		entry := GenerateWeaverCitation("1.0.0", hash)
		bibtex := entry.String()

		// Verify hash is included in note
		if !strings.Contains(bibtex, "note =") || !strings.Contains(bibtex, "Experiment hash:") {
			t.Error("BibTeX with experiment hash should include hash in note field")
		}
	})

	t.Run("Reproducibility report export", func(t *testing.T) {
		now := time.Now()
		endTime := now.Add(time.Hour)

		report := NewReportBuilder().
			WithToolVersion("1.0.0").
			WithSessionID("session-test-001").
			WithSessionName("Integration Test Session").
			WithMeasurementMode("passive").
			WithMeasurementCount(len(csvMeasurements)).
			WithConversationCount(1).
			WithTimeRange(now, &endTime).
			WithAgents(agents).
			WithParameter("embedding_model", "text-embedding-3-small").
			Build()

		if report == nil {
			t.Fatal("Report builder returned nil")
		}

		// Verify hash is computed
		if report.ExperimentHash == nil {
			t.Error("Report should have computed experiment hash")
		}

		// Test Markdown output
		markdown := report.ToMarkdown()
		if markdown == "" {
			t.Error("Markdown report is empty")
		}
		if !strings.Contains(markdown, "# ") {
			t.Error("Markdown report missing header")
		}
		if !strings.Contains(markdown, "Experiment Hash") {
			t.Error("Markdown report missing experiment hash section")
		}

		// Test LaTeX output
		latex := report.ToLaTeX()
		if latex == "" {
			t.Error("LaTeX report is empty")
		}
		if !strings.Contains(latex, "\\section{") {
			t.Error("LaTeX report missing section command")
		}

		// Test JSON output
		jsonStr := report.ToJSON()
		if jsonStr == "" {
			t.Error("JSON report is empty")
		}
		var parsed ReproducibilityReport
		if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
			t.Fatalf("JSON report parsing failed: %v", err)
		}
	})

	t.Run("Experiment hash verification", func(t *testing.T) {
		now := time.Now()
		endTime := now.Add(time.Hour)
		params := map[string]string{
			"param1": "value1",
			"param2": "value2",
		}

		hash := ComputeExperimentHash("1.0.0", "passive", 5, 1, now, &endTime, params)
		if hash == nil {
			t.Fatal("Hash computation returned nil")
		}

		// Verify hash is deterministic
		hash2 := ComputeExperimentHash("1.0.0", "passive", 5, 1, now, &endTime, params)
		if hash.Hash != hash2.Hash {
			t.Error("Hash should be deterministic for identical inputs")
		}

		// Verify hash changes with different inputs
		hash3 := ComputeExperimentHash("1.0.1", "passive", 5, 1, now, &endTime, params)
		if hash.Hash == hash3.Hash {
			t.Error("Hash should differ for different version")
		}

		// Verify verification works
		if !hash.Verify() {
			t.Error("Hash verification should pass for valid hash")
		}

		// Verify short hash
		shortHash := hash.ShortHash()
		if len(shortHash) != 8 {
			t.Errorf("Short hash length = %d, want 8", len(shortHash))
		}
		if !strings.HasPrefix(hash.Hash, shortHash) {
			t.Error("Short hash should be prefix of full hash")
		}
	})
}

// TestExportFormatConsistency verifies that different export formats contain consistent data.
func TestExportFormatConsistency(t *testing.T) {
	rows := createTestMeasurementRows()
	csvMeasurements := MeasurementRowsToCSV(rows)

	// Export to CSV
	var csvBuf bytes.Buffer
	if err := ExportMeasurementsToCSV(&csvBuf, csvMeasurements, nil); err != nil {
		t.Fatalf("CSV export failed: %v", err)
	}

	// Export to LaTeX
	latex := GenerateMeasurementTable(rows, nil)

	// Verify that both contain the same measurement count
	csvReader := csv.NewReader(strings.NewReader(csvBuf.String()))
	csvRecords, _ := csvReader.ReadAll()
	csvDataRows := len(csvRecords) - 1 // Subtract header

	latexRows := strings.Count(latex, "\\\\") - 1 // Subtract header row count (one for header)

	if csvDataRows != latexRows {
		t.Errorf("Data row count mismatch: CSV=%d, LaTeX=%d", csvDataRows, latexRows)
	}

	// Verify measurement data appears in both
	for _, row := range rows {
		senderEscaped := Escape(row.Sender)
		if !strings.Contains(latex, senderEscaped) {
			t.Errorf("LaTeX missing sender: %s", row.Sender)
		}
	}
}

// TestExportWithSpecialCharacters verifies that special characters are handled correctly.
func TestExportWithSpecialCharacters(t *testing.T) {
	// Create rows with special characters
	specialRows := []MeasurementRow{
		{
			Turn:       1,
			Sender:     "User_A & Co.",
			Receiver:   "Agent $Beta$",
			DEff:       100,
			Beta:       2.0,
			Alignment:  0.9,
			CPair:      0.8,
			BetaStatus: "optimal",
		},
		{
			Turn:       2,
			Sender:     "Test % Progress",
			Receiver:   "Agent #1",
			DEff:       95,
			Beta:       2.1,
			Alignment:  0.85,
			CPair:      0.75,
			BetaStatus: "monitor",
		},
	}

	t.Run("LaTeX escaping", func(t *testing.T) {
		latex := GenerateMeasurementTable(specialRows, nil)

		// Verify special characters are escaped
		if strings.Contains(latex, "User_A & Co.") {
			t.Error("LaTeX should escape unescaped ampersand")
		}
		if strings.Contains(latex, "$Beta$") {
			t.Error("LaTeX should escape dollar signs in cell content")
		}
		if !strings.Contains(latex, `\&`) {
			t.Error("LaTeX should contain escaped ampersand")
		}
		if !strings.Contains(latex, `\_`) {
			t.Error("LaTeX should contain escaped underscore")
		}
	})

	t.Run("CSV with special chars", func(t *testing.T) {
		csvMeasurements := MeasurementRowsToCSV(specialRows)
		var buf bytes.Buffer
		err := ExportMeasurementsToCSV(&buf, csvMeasurements, nil)
		if err != nil {
			t.Fatalf("CSV export failed: %v", err)
		}

		// Verify CSV can still be parsed
		reader := csv.NewReader(strings.NewReader(buf.String()))
		_, err = reader.ReadAll()
		if err != nil {
			t.Fatalf("CSV with special characters cannot be parsed: %v", err)
		}
	})

	t.Run("SVG XML escaping", func(t *testing.T) {
		config := DefaultSVGConfig()
		config.Title = "Test <Plot> & Data"

		svg := GenerateMeasurementPlot(specialRows, MetricBeta, config)

		// Title should be XML-escaped
		if strings.Contains(svg, "Test <Plot>") {
			t.Error("SVG should escape angle brackets in title")
		}
		if !strings.Contains(svg, "&amp;") || !strings.Contains(svg, "&lt;") || !strings.Contains(svg, "&gt;") {
			t.Error("SVG should contain XML-escaped special characters")
		}
	})
}

// TestExportToWriters verifies all export functions work with io.Writer interface.
func TestExportToWriters(t *testing.T) {
	rows := createTestMeasurementRows()
	csvMeasurements := createTestCSVMeasurements()

	t.Run("CSV to writer", func(t *testing.T) {
		var buf bytes.Buffer
		err := ExportMeasurementsToCSV(&buf, csvMeasurements, nil)
		if err != nil {
			t.Errorf("ExportMeasurementsToCSV failed: %v", err)
		}
		if buf.Len() == 0 {
			t.Error("CSV buffer is empty")
		}
	})

	t.Run("SVG to writer", func(t *testing.T) {
		var buf bytes.Buffer
		err := ExportSVGToWriter(&buf, rows, MetricBeta, nil)
		if err != nil {
			t.Errorf("ExportSVGToWriter failed: %v", err)
		}
		if buf.Len() == 0 {
			t.Error("SVG buffer is empty")
		}
	})

	t.Run("PDF to writer", func(t *testing.T) {
		var buf bytes.Buffer
		err := ExportPDFToWriter(&buf, rows, MetricBeta, nil)
		if err != nil {
			t.Errorf("ExportPDFToWriter failed: %v", err)
		}
		if buf.Len() == 0 {
			t.Error("PDF buffer is empty")
		}
	})

	t.Run("BibTeX to writer", func(t *testing.T) {
		entry := GenerateWeaverCitation("1.0.0", nil)
		var buf bytes.Buffer
		err := ExportBibTeXToWriter(&buf, entry, nil)
		if err != nil {
			t.Errorf("ExportBibTeXToWriter failed: %v", err)
		}
		if buf.Len() == 0 {
			t.Error("BibTeX buffer is empty")
		}
	})

	t.Run("Report to writer", func(t *testing.T) {
		report := NewReportBuilder().
			WithToolVersion("1.0.0").
			WithMeasurementCount(5).
			Build()

		var buf bytes.Buffer
		err := ExportReportToWriter(&buf, report, nil)
		if err != nil {
			t.Errorf("ExportReportToWriter failed: %v", err)
		}
		if buf.Len() == 0 {
			t.Error("Report buffer is empty")
		}
	})

	t.Run("Metadata to writer", func(t *testing.T) {
		var buf bytes.Buffer
		err := ExportMetadataToJSON(&buf, nil)
		if err != nil {
			t.Errorf("ExportMetadataToJSON failed: %v", err)
		}
		if buf.Len() == 0 {
			t.Error("Metadata buffer is empty")
		}
	})
}

// TestExportEmptyData verifies behavior with empty input data.
func TestExportEmptyData(t *testing.T) {
	t.Run("LaTeX with empty rows", func(t *testing.T) {
		latex := GenerateMeasurementTable([]MeasurementRow{}, nil)
		if latex != "" {
			t.Error("LaTeX table with no data should return empty string")
		}
	})

	t.Run("CSV with empty measurements", func(t *testing.T) {
		var buf bytes.Buffer
		err := ExportMeasurementsToCSV(&buf, []*CSVMeasurement{}, nil)
		if err != nil {
			t.Errorf("CSV export with empty data should not error: %v", err)
		}
	})

	t.Run("SVG with empty rows", func(t *testing.T) {
		svg := GenerateMeasurementPlot([]MeasurementRow{}, MetricBeta, nil)
		if svg != "" {
			t.Error("SVG plot with no data should return empty string")
		}
	})

	t.Run("PDF with empty rows", func(t *testing.T) {
		pdf := GenerateMeasurementPDFPlot([]MeasurementRow{}, MetricBeta, nil)
		if pdf != nil {
			t.Error("PDF plot with no data should return nil")
		}
	})
}

// TestExportSingleDataPoint verifies handling of single data point edge case.
func TestExportSingleDataPoint(t *testing.T) {
	singleRow := []MeasurementRow{
		{
			Turn:       1,
			Sender:     "Alice",
			Receiver:   "Bob",
			DEff:       128,
			Beta:       1.75,
			Alignment:  0.923,
			CPair:      0.856,
			BetaStatus: "optimal",
		},
	}

	t.Run("LaTeX single row", func(t *testing.T) {
		latex := GenerateMeasurementTable(singleRow, nil)
		if latex == "" {
			t.Error("LaTeX table with single row should not be empty")
		}
		if !strings.Contains(latex, "Alice") {
			t.Error("LaTeX should contain the single data row")
		}
	})

	t.Run("SVG single point", func(t *testing.T) {
		svg := GenerateMeasurementPlot(singleRow, MetricBeta, nil)
		if svg == "" {
			t.Error("SVG plot with single point should not be empty")
		}
		// Should still produce valid SVG
		if !strings.Contains(svg, "<svg") {
			t.Error("Single-point SVG should still be valid SVG")
		}
	})

	t.Run("PDF single point", func(t *testing.T) {
		pdf := GenerateMeasurementPDFPlot(singleRow, MetricBeta, nil)
		if pdf == nil || len(pdf) == 0 {
			t.Error("PDF plot with single point should not be empty")
		}
	})
}

// TestExportLargeDataset verifies handling of larger datasets.
func TestExportLargeDataset(t *testing.T) {
	// Create a larger dataset (100 rows)
	largeRows := make([]MeasurementRow, 100)
	for i := 0; i < 100; i++ {
		largeRows[i] = MeasurementRow{
			Turn:       i + 1,
			Sender:     "Agent_A",
			Receiver:   "Agent_B",
			DEff:       100 - (i % 50),
			Beta:       1.5 + float64(i%50)*0.05,
			Alignment:  0.95 - float64(i%50)*0.01,
			CPair:      0.9 - float64(i%50)*0.01,
			BetaStatus: "optimal",
		}
	}

	t.Run("LaTeX large dataset", func(t *testing.T) {
		latex := GenerateMeasurementTable(largeRows, nil)
		if latex == "" {
			t.Error("LaTeX table with large dataset should not be empty")
		}
		// Count data rows
		rowCount := strings.Count(latex, "\\\\") - 1 // Subtract header
		if rowCount != 100 {
			t.Errorf("LaTeX row count = %d, want 100", rowCount)
		}
	})

	t.Run("CSV large dataset", func(t *testing.T) {
		csvMeasurements := MeasurementRowsToCSV(largeRows)
		var buf bytes.Buffer
		err := ExportMeasurementsToCSV(&buf, csvMeasurements, nil)
		if err != nil {
			t.Fatalf("CSV export failed: %v", err)
		}

		// Verify row count
		reader := csv.NewReader(strings.NewReader(buf.String()))
		records, _ := reader.ReadAll()
		if len(records) != 101 { // 100 data + 1 header
			t.Errorf("CSV record count = %d, want 101", len(records))
		}
	})

	t.Run("SVG large dataset", func(t *testing.T) {
		svg := GenerateMeasurementPlot(largeRows, MetricBeta, nil)
		if svg == "" {
			t.Error("SVG with large dataset should not be empty")
		}
		// Should have path elements with data points
		if !strings.Contains(svg, "<path") {
			t.Error("SVG should contain path elements")
		}
	})

	t.Run("PDF large dataset", func(t *testing.T) {
		pdf := GenerateMeasurementPDFPlot(largeRows, MetricBeta, nil)
		if pdf == nil || len(pdf) == 0 {
			t.Error("PDF with large dataset should not be empty")
		}
	})
}

// TestReproducibilityReportVerification tests the full verification workflow.
func TestReproducibilityReportVerification(t *testing.T) {
	now := time.Now()
	endTime := now.Add(time.Hour)
	params := map[string]string{
		"embedding_dim": "1536",
		"model_name":    "claude-3-opus",
	}

	// Create first report
	report1 := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithSessionID("session-001").
		WithMeasurementMode("passive").
		WithMeasurementCount(10).
		WithConversationCount(1).
		WithTimeRange(now, &endTime).
		WithParameters(params).
		Build()

	// Create second report with same parameters
	report2 := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithSessionID("session-001").
		WithMeasurementMode("passive").
		WithMeasurementCount(10).
		WithConversationCount(1).
		WithTimeRange(now, &endTime).
		WithParameters(params).
		Build()

	t.Run("identical configs produce identical hashes", func(t *testing.T) {
		if report1.ExperimentHash == nil || report2.ExperimentHash == nil {
			t.Fatal("Reports should have experiment hashes")
		}
		if report1.ExperimentHash.Hash != report2.ExperimentHash.Hash {
			t.Error("Identical configurations should produce identical hashes")
		}
	})

	t.Run("verification passes for valid report", func(t *testing.T) {
		if !report1.Verify() {
			t.Error("Report verification should pass")
		}
	})

	t.Run("different configs produce different hashes", func(t *testing.T) {
		report3 := NewReportBuilder().
			WithToolVersion("1.0.1"). // Different version
			WithSessionID("session-001").
			WithMeasurementMode("passive").
			WithMeasurementCount(10).
			WithConversationCount(1).
			WithTimeRange(now, &endTime).
			WithParameters(params).
			Build()

		if report1.ExperimentHash.Hash == report3.ExperimentHash.Hash {
			t.Error("Different configurations should produce different hashes")
		}
	})

	t.Run("report contains verification instructions", func(t *testing.T) {
		markdown := report1.ToMarkdown()
		if !strings.Contains(markdown, "Verification") {
			t.Error("Report should contain verification section")
		}
		if !strings.Contains(markdown, "SHA-256") {
			t.Error("Report should mention hash algorithm")
		}
	})
}

// TestBibTeXFileGeneration tests generating a complete .bib file.
func TestBibTeXFileGeneration(t *testing.T) {
	now := time.Now()
	hash := ComputeExperimentHash("1.0.0", "passive", 5, 1, now, nil, nil)

	// Create multiple entries
	entries := []*BibTeXEntry{
		GenerateWeaverCitation("1.0.0", hash),
		NewBibTeXBuilder().
			WithKey("weaver_paper_2024").
			WithTitle("Weaver: Multi-Agent Orchestration").
			WithYear(2024).
			WithAuthor("Smith, John and Doe, Jane").
			WithEntryType(EntryTypeMisc).
			Build(),
	}

	bibFile := GenerateBibTeXFile(entries, nil)

	t.Run("file has header comments", func(t *testing.T) {
		if !strings.Contains(bibFile, "% BibTeX entries generated by Weaver") {
			t.Error("BibTeX file should have header comment")
		}
		if !strings.Contains(bibFile, "% Generated:") {
			t.Error("BibTeX file should have generation timestamp")
		}
	})

	t.Run("file contains all entries", func(t *testing.T) {
		for _, entry := range entries {
			if !strings.Contains(bibFile, entry.Key) {
				t.Errorf("BibTeX file missing entry with key: %s", entry.Key)
			}
		}
	})

	t.Run("entries are sorted by key", func(t *testing.T) {
		idx1 := strings.Index(bibFile, "weaver_")
		idx2 := strings.Index(bibFile, "weaver_paper_2024")
		if idx1 > idx2 {
			t.Error("BibTeX entries should be sorted by key")
		}
	})
}

// TestMeasurementRowConversions tests conversions between export formats.
func TestMeasurementRowConversions(t *testing.T) {
	rows := createTestMeasurementRows()

	t.Run("MeasurementRow to CSV conversion", func(t *testing.T) {
		csvMeasurements := MeasurementRowsToCSV(rows)
		if len(csvMeasurements) != len(rows) {
			t.Errorf("Conversion should preserve row count: got %d, want %d",
				len(csvMeasurements), len(rows))
		}

		for i, csv := range csvMeasurements {
			if csv.TurnNumber != rows[i].Turn {
				t.Errorf("Row %d: Turn mismatch: got %d, want %d",
					i, csv.TurnNumber, rows[i].Turn)
			}
			if csv.SenderName != rows[i].Sender {
				t.Errorf("Row %d: Sender mismatch: got %s, want %s",
					i, csv.SenderName, rows[i].Sender)
			}
			if csv.Beta != rows[i].Beta {
				t.Errorf("Row %d: Beta mismatch: got %f, want %f",
					i, csv.Beta, rows[i].Beta)
			}
		}
	})

	t.Run("single row conversion", func(t *testing.T) {
		csv := MeasurementRowToCSV(rows[0])
		if csv == nil {
			t.Fatal("MeasurementRowToCSV should not return nil")
		}
		if csv.DEff != rows[0].DEff {
			t.Errorf("DEff mismatch: got %d, want %d", csv.DEff, rows[0].DEff)
		}
	})
}

// TestSVGPlotBuilderAPI tests the SVG plot builder fluent API.
func TestSVGPlotBuilderAPI(t *testing.T) {
	t.Run("fluent API builds valid SVG", func(t *testing.T) {
		config := DefaultSVGConfig()
		config.Title = "Test Plot"
		config.ShowGrid = true
		config.ShowPoints = true

		builder := NewSVGPlotBuilder(config)
		builder.AddPoint(MetricBeta, 1, 1.5)
		builder.AddPoint(MetricBeta, 2, 1.8)
		builder.AddPoint(MetricBeta, 3, 2.1)

		svg := builder.Build()
		if svg == "" {
			t.Error("Builder should produce non-empty SVG")
		}
		if !strings.Contains(svg, "Test Plot") {
			t.Error("SVG should contain title")
		}
	})

	t.Run("WriteTo interface", func(t *testing.T) {
		builder := NewSVGPlotBuilder(nil)
		builder.AddPoint(MetricBeta, 1, 1.5)

		var buf bytes.Buffer
		n, err := builder.WriteTo(&buf)
		if err != nil {
			t.Errorf("WriteTo failed: %v", err)
		}
		if n == 0 {
			t.Error("WriteTo should write bytes")
		}
		if buf.Len() == 0 {
			t.Error("Buffer should not be empty")
		}
	})
}

// TestPDFPlotBuilderAPI tests the PDF plot builder fluent API.
func TestPDFPlotBuilderAPI(t *testing.T) {
	t.Run("fluent API builds valid PDF", func(t *testing.T) {
		config := DefaultPDFConfig()
		config.Title = "Test PDF Plot"
		config.Author = "Test Author"

		builder := NewPDFPlotBuilder(config)
		builder.AddPoint(MetricBeta, 1, 1.5)
		builder.AddPoint(MetricBeta, 2, 1.8)
		builder.AddPoint(MetricBeta, 3, 2.1)

		pdf := builder.Build()
		if pdf == nil || len(pdf) == 0 {
			t.Error("Builder should produce non-empty PDF")
		}
		if string(pdf[:5]) != "%PDF-" {
			t.Error("PDF should start with PDF header")
		}
	})

	t.Run("WriteTo interface", func(t *testing.T) {
		builder := NewPDFPlotBuilder(nil)
		builder.AddPoint(MetricBeta, 1, 1.5)

		var buf bytes.Buffer
		n, err := builder.WriteTo(&buf)
		if err != nil {
			t.Errorf("WriteTo failed: %v", err)
		}
		if n == 0 {
			t.Error("WriteTo should write bytes")
		}
		if buf.Len() == 0 {
			t.Error("Buffer should not be empty")
		}
	})
}

// TestCSVWriterAPI tests the CSV writer fluent API.
func TestCSVWriterAPI(t *testing.T) {
	measurements := createTestCSVMeasurements()

	t.Run("writer tracks row count", func(t *testing.T) {
		var buf bytes.Buffer
		writer := NewCSVWriter(&buf, nil)

		for _, m := range measurements {
			if err := writer.Write(m); err != nil {
				t.Fatalf("Write failed: %v", err)
			}
		}
		if err := writer.Flush(); err != nil {
			t.Fatalf("Flush failed: %v", err)
		}

		if writer.RowsWritten() != len(measurements) {
			t.Errorf("RowsWritten = %d, want %d", writer.RowsWritten(), len(measurements))
		}
	})

	t.Run("writer with custom config", func(t *testing.T) {
		config := &CSVConfig{
			Dialect:           DialectTSV,
			IncludeHeader:     true,
			TimestampFormat:   time.RFC3339,
			Precision:         4,
			NAString:          "NULL",
			IncludeBetaStatus: true,
		}

		var buf bytes.Buffer
		writer := NewCSVWriter(&buf, config)

		for _, m := range measurements {
			if err := writer.Write(m); err != nil {
				t.Fatalf("Write failed: %v", err)
			}
		}
		if err := writer.Flush(); err != nil {
			t.Fatalf("Flush failed: %v", err)
		}

		// Verify TSV format (tabs)
		content := buf.String()
		if !strings.Contains(content, "\t") {
			t.Error("TSV output should contain tabs")
		}
	})
}

// TestHashBuilderAPI tests the hash builder fluent API.
func TestHashBuilderAPI(t *testing.T) {
	now := time.Now()
	endTime := now.Add(time.Hour)

	t.Run("fluent API builds valid hash", func(t *testing.T) {
		hash := NewHashBuilder().
			WithToolVersion("1.0.0").
			WithMeasurementMode("passive").
			WithMeasurementCount(10).
			WithConversationCount(2).
			WithTimeRange(now, &endTime).
			WithParameter("key1", "value1").
			WithParameter("key2", "value2").
			Build()

		if hash == nil {
			t.Fatal("Builder should produce non-nil hash")
		}
		if hash.Hash == "" {
			t.Error("Hash should not be empty")
		}
		if hash.Algorithm != HashAlgorithm {
			t.Errorf("Algorithm = %s, want %s", hash.Algorithm, HashAlgorithm)
		}
	})

	t.Run("WithParameters bulk add", func(t *testing.T) {
		params := map[string]string{
			"param1": "value1",
			"param2": "value2",
			"param3": "value3",
		}

		hash := NewHashBuilder().
			WithToolVersion("1.0.0").
			WithParameters(params).
			Build()

		if hash.Config.Parameters == nil {
			t.Fatal("Parameters should not be nil")
		}
		if len(hash.Config.Parameters) != 3 {
			t.Errorf("Parameter count = %d, want 3", len(hash.Config.Parameters))
		}
	})
}

// TestBibTeXBuilderAPI tests the BibTeX builder fluent API.
func TestBibTeXBuilderAPI(t *testing.T) {
	t.Run("fluent API builds valid entry", func(t *testing.T) {
		entry := NewBibTeXBuilder().
			WithKey("test_2024").
			WithTitle("Test Entry").
			WithAuthor("Author, Test").
			WithYear(2024).
			WithVersion("1.0.0").
			WithURL("https://example.com").
			WithNote("Test note").
			Build()

		if entry == nil {
			t.Fatal("Builder should produce non-nil entry")
		}
		if entry.Key != "test_2024" {
			t.Errorf("Key = %s, want test_2024", entry.Key)
		}

		// Validate entry
		errors := ValidateBibTeXEntry(entry)
		if errors != nil {
			t.Errorf("Entry should be valid, got errors: %v", errors)
		}
	})

	t.Run("auto-generated key", func(t *testing.T) {
		entry := NewBibTeXBuilder().
			WithVersion("2.0.0").
			Build()

		if entry.Key == "" {
			t.Error("Key should be auto-generated")
		}
		if !strings.Contains(entry.Key, "weaver") {
			t.Error("Auto-generated key should contain 'weaver' prefix")
		}
	})
}

// TestReportBuilderAPI tests the report builder fluent API.
func TestReportBuilderAPI(t *testing.T) {
	now := time.Now()
	endTime := now.Add(2 * time.Hour)

	t.Run("fluent API builds complete report", func(t *testing.T) {
		report := NewReportBuilder().
			WithTitle("Integration Test Report").
			WithAuthor("Test Framework").
			WithToolVersion("1.0.0").
			WithSessionID("session-001").
			WithSessionName("Test Session").
			WithSessionDescription("A test session for integration testing").
			WithMeasurementMode("passive").
			WithMeasurementCount(50).
			WithConversationCount(5).
			WithTimeRange(now, &endTime).
			WithAgent(AgentConfig{ID: "agent-1", Name: "Agent 1", Type: "llm"}).
			WithDataSource(DataSource{Name: "source-1", Type: "file", Path: "/data/test.csv"}).
			WithParameter("embedding_model", "text-embedding-3-large").
			WithEnvironmentVar("WEAVER_ENV", "test").
			Build()

		if report == nil {
			t.Fatal("Builder should produce non-nil report")
		}
		if report.Title != "Integration Test Report" {
			t.Errorf("Title = %s, want 'Integration Test Report'", report.Title)
		}
		if len(report.Agents) != 1 {
			t.Errorf("Agent count = %d, want 1", len(report.Agents))
		}
		if len(report.DataSources) != 1 {
			t.Errorf("DataSource count = %d, want 1", len(report.DataSources))
		}
		if len(report.Parameters) != 1 {
			t.Errorf("Parameter count = %d, want 1", len(report.Parameters))
		}
		if report.Duration == "" {
			t.Error("Duration should be computed when both start and end times are set")
		}
	})

	t.Run("format methods produce different outputs", func(t *testing.T) {
		report := NewReportBuilder().
			WithToolVersion("1.0.0").
			WithMeasurementCount(10).
			Build()

		markdown := report.ToMarkdown()
		latex := report.ToLaTeX()
		jsonStr := report.ToJSON()

		if markdown == latex {
			t.Error("Markdown and LaTeX outputs should differ")
		}
		if markdown == jsonStr {
			t.Error("Markdown and JSON outputs should differ")
		}
		if latex == jsonStr {
			t.Error("LaTeX and JSON outputs should differ")
		}
	})
}

// TestAllMetricTypes tests that all metric types work correctly.
func TestAllMetricTypes(t *testing.T) {
	rows := createTestMeasurementRows()
	metrics := []PlotMetric{MetricDEff, MetricBeta, MetricAlignment, MetricCPair}

	for _, metric := range metrics {
		t.Run(string(metric), func(t *testing.T) {
			// Test MetricInfo
			info := GetMetricInfo(metric)
			if info.Name == "" {
				t.Errorf("Metric %s should have a name", metric)
			}
			if info.Color == "" {
				t.Errorf("Metric %s should have a color", metric)
			}

			// Test SVG generation
			svg := GenerateMeasurementPlot(rows, metric, nil)
			if svg == "" {
				t.Errorf("SVG for metric %s should not be empty", metric)
			}

			// Test PDF generation
			pdf := GenerateMeasurementPDFPlot(rows, metric, nil)
			if pdf == nil || len(pdf) == 0 {
				t.Errorf("PDF for metric %s should not be empty", metric)
			}
		})
	}
}

// BenchmarkExportWorkflow benchmarks the export workflow.
func BenchmarkExportWorkflow(b *testing.B) {
	rows := createTestMeasurementRows()
	csvMeasurements := createTestCSVMeasurements()

	b.Run("LaTeX table", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = GenerateMeasurementTable(rows, nil)
		}
	})

	b.Run("CSV export", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var buf bytes.Buffer
			_ = ExportMeasurementsToCSV(&buf, csvMeasurements, nil)
		}
	})

	b.Run("SVG plot", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = GenerateMeasurementPlot(rows, MetricBeta, nil)
		}
	})

	b.Run("PDF plot", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = GenerateMeasurementPDFPlot(rows, MetricBeta, nil)
		}
	})

	b.Run("BibTeX", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			entry := GenerateWeaverCitation("1.0.0", nil)
			_ = entry.String()
		}
	})

	b.Run("Hash computation", func(b *testing.B) {
		now := time.Now()
		for i := 0; i < b.N; i++ {
			_ = ComputeExperimentHash("1.0.0", "passive", 10, 1, now, nil, nil)
		}
	})

	b.Run("Report generation", func(b *testing.B) {
		now := time.Now()
		for i := 0; i < b.N; i++ {
			report := NewReportBuilder().
				WithToolVersion("1.0.0").
				WithMeasurementCount(10).
				WithTimeRange(now, nil).
				Build()
			_ = report.ToMarkdown()
		}
	})
}

// Ensure io.Writer is implemented by buffers used in tests.
var _ io.Writer = (*bytes.Buffer)(nil)
