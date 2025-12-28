// Package export tests for CSV export functionality.
package export

import (
	"bytes"
	"encoding/csv"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// -----------------------------------------------------------------------------
// CSVConfig Tests
// -----------------------------------------------------------------------------

func TestDefaultCSVConfig(t *testing.T) {
	config := DefaultCSVConfig()

	if config == nil {
		t.Fatal("DefaultCSVConfig() returned nil")
	}
	if config.Dialect != DialectStandard {
		t.Errorf("expected Dialect %q, got %q", DialectStandard, config.Dialect)
	}
	if !config.IncludeHeader {
		t.Error("expected IncludeHeader to be true by default")
	}
	if config.TimestampFormat != time.RFC3339 {
		t.Errorf("expected TimestampFormat %q, got %q", time.RFC3339, config.TimestampFormat)
	}
	if config.Precision != 6 {
		t.Errorf("expected Precision 6, got %d", config.Precision)
	}
	if config.NAString != "NA" {
		t.Errorf("expected NAString %q, got %q", "NA", config.NAString)
	}
	if !config.IncludeBetaStatus {
		t.Error("expected IncludeBetaStatus to be true by default")
	}
	if config.IncludeMessageContent {
		t.Error("expected IncludeMessageContent to be false by default")
	}
	if config.IncludeTokenCount {
		t.Error("expected IncludeTokenCount to be false by default")
	}
}

// -----------------------------------------------------------------------------
// CSVDialect Tests
// -----------------------------------------------------------------------------

func TestCSVDialect_Constants(t *testing.T) {
	if DialectStandard != "standard" {
		t.Errorf("DialectStandard should be 'standard', got %q", DialectStandard)
	}
	if DialectExcel != "excel" {
		t.Errorf("DialectExcel should be 'excel', got %q", DialectExcel)
	}
	if DialectTSV != "tsv" {
		t.Errorf("DialectTSV should be 'tsv', got %q", DialectTSV)
	}
}

// -----------------------------------------------------------------------------
// CSVWriter Tests
// -----------------------------------------------------------------------------

func TestNewCSVWriter_NilConfig(t *testing.T) {
	var buf bytes.Buffer
	cw := NewCSVWriter(&buf, nil)

	if cw == nil {
		t.Fatal("NewCSVWriter() returned nil")
	}
	if cw.config == nil {
		t.Error("expected default config to be set")
	}
}

func TestCSVWriter_WriteHeader(t *testing.T) {
	var buf bytes.Buffer
	cw := NewCSVWriter(&buf, nil)

	err := cw.WriteHeader()
	if err != nil {
		t.Fatalf("WriteHeader() failed: %v", err)
	}
	cw.Flush()

	result := buf.String()

	// Check required headers are present
	requiredHeaders := []string{
		"id", "timestamp", "session_id", "conversation_id", "turn_number",
		"sender_id", "sender_name", "sender_role",
		"receiver_id", "receiver_name", "receiver_role",
		"d_eff", "beta", "alignment", "c_pair", "is_unilateral", "beta_status",
	}

	for _, h := range requiredHeaders {
		if !strings.Contains(result, h) {
			t.Errorf("expected header %q in output: %s", h, result)
		}
	}
}

func TestCSVWriter_WriteHeader_OnlyOnce(t *testing.T) {
	var buf bytes.Buffer
	cw := NewCSVWriter(&buf, nil)

	// Write header twice
	cw.WriteHeader()
	cw.WriteHeader()
	cw.Flush()

	// Count number of "id" occurrences (should be 1)
	result := buf.String()
	count := strings.Count(result, "id,")
	if count != 1 {
		t.Errorf("expected header written once, got %d occurrences", count)
	}
}

func TestCSVWriter_Write_SingleMeasurement(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	timestamp := time.Date(2024, 1, 15, 10, 30, 0, 0, time.UTC)
	m := &CSVMeasurement{
		ID:             "test-id-123",
		Timestamp:      timestamp,
		SessionID:      "session-1",
		ConversationID: "conv-1",
		TurnNumber:     1,
		SenderID:       "sender-1",
		SenderName:     "Alice",
		SenderRole:     "user",
		ReceiverID:     "receiver-1",
		ReceiverName:   "Bob",
		ReceiverRole:   "assistant",
		DEff:           128,
		Beta:           1.75,
		Alignment:      0.85,
		CPair:          0.92,
		BetaStatus:     "optimal",
		IsUnilateral:   false,
	}

	err := cw.Write(m)
	if err != nil {
		t.Fatalf("Write() failed: %v", err)
	}
	cw.Flush()

	result := buf.String()

	// Verify values are present
	mustContain := []string{
		"test-id-123",
		"2024-01-15T10:30:00Z",
		"session-1",
		"conv-1",
		"Alice",
		"Bob",
		"128",
		"1.750000",
		"0.850000",
		"0.920000",
		"optimal",
		"FALSE",
	}

	for _, v := range mustContain {
		if !strings.Contains(result, v) {
			t.Errorf("expected %q in output: %s", v, result)
		}
	}
}

func TestCSVWriter_Write_WithHeader(t *testing.T) {
	var buf bytes.Buffer
	cw := NewCSVWriter(&buf, nil)

	m := &CSVMeasurement{
		ID:        "test-1",
		Timestamp: time.Now(),
		TurnNumber: 1,
		DEff:      100,
		Beta:      1.5,
		Alignment: 0.9,
		CPair:     0.8,
	}

	err := cw.Write(m)
	if err != nil {
		t.Fatalf("Write() failed: %v", err)
	}
	cw.Flush()

	result := buf.String()
	lines := strings.Split(strings.TrimSpace(result), "\n")

	if len(lines) < 2 {
		t.Fatalf("expected at least 2 lines (header + data), got %d", len(lines))
	}

	// First line should be header
	if !strings.Contains(lines[0], "id,timestamp") {
		t.Errorf("first line should be header: %s", lines[0])
	}

	// Second line should be data
	if !strings.Contains(lines[1], "test-1") {
		t.Errorf("second line should contain data: %s", lines[1])
	}
}

func TestCSVWriter_WriteAll(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	cw := NewCSVWriter(&buf, config)

	measurements := []*CSVMeasurement{
		{ID: "m1", TurnNumber: 1, DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.8},
		{ID: "m2", TurnNumber: 2, DEff: 110, Beta: 1.7, Alignment: 0.85, CPair: 0.75},
		{ID: "m3", TurnNumber: 3, DEff: 120, Beta: 1.9, Alignment: 0.8, CPair: 0.7},
	}

	err := cw.WriteAll(measurements)
	if err != nil {
		t.Fatalf("WriteAll() failed: %v", err)
	}
	cw.Flush()

	// Should have 4 lines (header + 3 data rows)
	result := buf.String()
	lines := strings.Split(strings.TrimSpace(result), "\n")
	if len(lines) != 4 {
		t.Errorf("expected 4 lines, got %d", len(lines))
	}

	if cw.RowsWritten() != 3 {
		t.Errorf("expected RowsWritten() = 3, got %d", cw.RowsWritten())
	}
}

func TestCSVWriter_TSVDialect(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.Dialect = DialectTSV
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID:        "test-1",
		TurnNumber: 1,
		DEff:      100,
		Beta:      1.5,
		Alignment: 0.9,
		CPair:     0.8,
	}

	cw.Write(m)
	cw.Flush()

	result := buf.String()

	// TSV should use tabs not commas
	if strings.Contains(result, ",") && !strings.Contains(result, "\t") {
		t.Error("TSV dialect should use tabs as separators")
	}
	if !strings.Contains(result, "\t") {
		t.Error("expected tab characters in TSV output")
	}
}

func TestCSVWriter_NAValues(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	// Measurement with missing values
	m := &CSVMeasurement{
		ID:         "", // Empty
		SessionID:  "", // Empty
		TurnNumber: 1,
		DEff:       100,
		Beta:       1.5,
		Alignment:  0.9,
		CPair:      0.8,
	}

	cw.Write(m)
	cw.Flush()

	result := buf.String()

	// Empty strings should be replaced with NA
	if !strings.Contains(result, "NA") {
		t.Errorf("expected NA for missing values: %s", result)
	}
}

func TestCSVWriter_CustomNAString(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.NAString = "" // Empty string for missing values
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID:         "", // Empty
		TurnNumber: 1,
		DEff:       100,
		Beta:       1.5,
		Alignment:  0.9,
		CPair:      0.8,
	}

	cw.Write(m)
	cw.Flush()

	// Parse the CSV
	reader := csv.NewReader(strings.NewReader(buf.String()))
	record, err := reader.Read()
	if err != nil {
		t.Fatalf("failed to parse CSV: %v", err)
	}

	// First field (id) should be empty
	if record[0] != "" {
		t.Errorf("expected empty string for ID, got %q", record[0])
	}
}

func TestCSVWriter_BooleanFormatting(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	// Test TRUE
	m1 := &CSVMeasurement{ID: "1", IsUnilateral: true}
	cw.Write(m1)

	// Test FALSE
	m2 := &CSVMeasurement{ID: "2", IsUnilateral: false}
	cw.Write(m2)

	cw.Flush()

	result := buf.String()

	if !strings.Contains(result, "TRUE") {
		t.Error("expected TRUE for IsUnilateral=true")
	}
	if !strings.Contains(result, "FALSE") {
		t.Error("expected FALSE for IsUnilateral=false")
	}
}

func TestCSVWriter_FloatPrecision(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.Precision = 3
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID:        "1",
		DEff:      100,
		Beta:      1.123456789,
		Alignment: 0.987654321,
		CPair:     0.555555555,
	}

	cw.Write(m)
	cw.Flush()

	result := buf.String()

	// With precision 3, should have exactly 3 decimal places
	if !strings.Contains(result, "1.123") {
		t.Error("expected Beta formatted with 3 decimal places")
	}
	if !strings.Contains(result, "0.988") { // Rounded up
		t.Error("expected Alignment formatted with 3 decimal places")
	}
	if !strings.Contains(result, "0.556") { // Rounded up
		t.Error("expected CPair formatted with 3 decimal places")
	}
}

func TestCSVWriter_TimestampFormatting(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	timestamp := time.Date(2024, 6, 15, 14, 30, 45, 0, time.UTC)
	m := &CSVMeasurement{
		ID:        "1",
		Timestamp: timestamp,
	}

	cw.Write(m)
	cw.Flush()

	result := buf.String()

	// Default RFC3339 format
	if !strings.Contains(result, "2024-06-15T14:30:45Z") {
		t.Errorf("expected RFC3339 timestamp format: %s", result)
	}
}

func TestCSVWriter_ZeroTimestamp(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID:        "1",
		Timestamp: time.Time{}, // Zero time
	}

	cw.Write(m)
	cw.Flush()

	result := buf.String()

	// Zero timestamp should be NA
	if !strings.Contains(result, "NA") {
		t.Errorf("expected NA for zero timestamp: %s", result)
	}
}

func TestCSVWriter_IncludeMessageContent(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeMessageContent = true
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID:             "1",
		MessageContent: "Hello, world!",
	}

	cw.Write(m)
	cw.Flush()

	result := buf.String()

	// Check header includes message_content
	if !strings.Contains(result, "message_content") {
		t.Error("expected message_content header when IncludeMessageContent is true")
	}

	// Check data includes content
	if !strings.Contains(result, "Hello, world!") {
		t.Error("expected message content in output")
	}
}

func TestCSVWriter_IncludeTokenCount(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeTokenCount = true
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID:         "1",
		TokenCount: 150,
	}

	cw.Write(m)
	cw.Flush()

	result := buf.String()

	// Check header includes token_count
	if !strings.Contains(result, "token_count") {
		t.Error("expected token_count header when IncludeTokenCount is true")
	}

	// Check data includes count
	if !strings.Contains(result, "150") {
		t.Error("expected token count in output")
	}
}

func TestCSVWriter_NilMeasurement(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	err := cw.Write(nil)
	if err != nil {
		t.Fatalf("Write(nil) should not error: %v", err)
	}
	cw.Flush()

	result := buf.String()

	// Should produce a row of NA values
	if !strings.Contains(result, "NA") {
		t.Errorf("expected NA values for nil measurement: %s", result)
	}
}

// -----------------------------------------------------------------------------
// ExportMeasurementsToCSV Tests
// -----------------------------------------------------------------------------

func TestCSVExport_ExportMeasurementsToCSV(t *testing.T) {
	var buf bytes.Buffer

	measurements := []*CSVMeasurement{
		{ID: "m1", TurnNumber: 1, DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.8, BetaStatus: "optimal"},
		{ID: "m2", TurnNumber: 2, DEff: 110, Beta: 1.7, Alignment: 0.85, CPair: 0.75, BetaStatus: "optimal"},
	}

	err := ExportMeasurementsToCSV(&buf, measurements, nil)
	if err != nil {
		t.Fatalf("ExportMeasurementsToCSV() failed: %v", err)
	}

	result := buf.String()

	// Should have header + 2 data rows
	lines := strings.Split(strings.TrimSpace(result), "\n")
	if len(lines) != 3 {
		t.Errorf("expected 3 lines, got %d", len(lines))
	}
}

func TestCSVExport_EmptyMeasurements(t *testing.T) {
	var buf bytes.Buffer

	err := ExportMeasurementsToCSV(&buf, []*CSVMeasurement{}, nil)
	if err != nil {
		t.Fatalf("ExportMeasurementsToCSV() failed: %v", err)
	}

	result := buf.String()

	// Should have only header (if IncludeHeader is true by default)
	// With no writes, no header is written either
	if result != "" {
		t.Errorf("expected empty output for empty measurements, got: %s", result)
	}
}

// -----------------------------------------------------------------------------
// Conversion Function Tests
// -----------------------------------------------------------------------------

func TestMeasurementRowToCSV(t *testing.T) {
	row := MeasurementRow{
		Turn:       5,
		Sender:     "Alice",
		Receiver:   "Bob",
		DEff:       128,
		Beta:       1.75,
		Alignment:  0.85,
		CPair:      0.92,
		BetaStatus: "optimal",
	}

	csvM := MeasurementRowToCSV(row)

	if csvM.TurnNumber != 5 {
		t.Errorf("expected TurnNumber 5, got %d", csvM.TurnNumber)
	}
	if csvM.SenderName != "Alice" {
		t.Errorf("expected SenderName 'Alice', got %q", csvM.SenderName)
	}
	if csvM.ReceiverName != "Bob" {
		t.Errorf("expected ReceiverName 'Bob', got %q", csvM.ReceiverName)
	}
	if csvM.DEff != 128 {
		t.Errorf("expected DEff 128, got %d", csvM.DEff)
	}
	if csvM.Beta != 1.75 {
		t.Errorf("expected Beta 1.75, got %f", csvM.Beta)
	}
	if csvM.Alignment != 0.85 {
		t.Errorf("expected Alignment 0.85, got %f", csvM.Alignment)
	}
	if csvM.CPair != 0.92 {
		t.Errorf("expected CPair 0.92, got %f", csvM.CPair)
	}
	if csvM.BetaStatus != "optimal" {
		t.Errorf("expected BetaStatus 'optimal', got %q", csvM.BetaStatus)
	}
}

func TestMeasurementRowsToCSV(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Sender: "A", Receiver: "B", DEff: 100},
		{Turn: 2, Sender: "B", Receiver: "A", DEff: 110},
		{Turn: 3, Sender: "A", Receiver: "B", DEff: 120},
	}

	csvMs := MeasurementRowsToCSV(rows)

	if len(csvMs) != 3 {
		t.Fatalf("expected 3 measurements, got %d", len(csvMs))
	}

	for i, m := range csvMs {
		if m.TurnNumber != i+1 {
			t.Errorf("measurement %d: expected TurnNumber %d, got %d", i, i+1, m.TurnNumber)
		}
	}
}

// -----------------------------------------------------------------------------
// R Compatibility Tests
// -----------------------------------------------------------------------------

func TestCSVExport_RCompatibility_ColumnNames(t *testing.T) {
	var buf bytes.Buffer
	cw := NewCSVWriter(&buf, nil)
	cw.WriteHeader()
	cw.Flush()

	result := buf.String()

	// R requires column names to be valid identifiers:
	// - Start with letter or dot (not followed by number)
	// - Contain only letters, numbers, dots, underscores
	// - No spaces or special characters

	invalidChars := []string{" ", "-", "@", "#", "$", "%", "&", "*"}
	for _, ch := range invalidChars {
		if strings.Contains(result, ch) {
			t.Errorf("header contains invalid character for R: %q", ch)
		}
	}

	// All our column names use snake_case which R handles well
	expectedCols := []string{
		"id", "timestamp", "session_id", "conversation_id", "turn_number",
		"sender_id", "sender_name", "sender_role",
		"receiver_id", "receiver_name", "receiver_role",
		"d_eff", "beta", "alignment", "c_pair", "is_unilateral", "beta_status",
	}

	for _, col := range expectedCols {
		if !strings.Contains(result, col) {
			t.Errorf("expected column %q for R compatibility", col)
		}
	}
}

func TestCSVExport_RCompatibility_NAValues(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID: "", // Will become NA
	}
	cw.Write(m)
	cw.Flush()

	result := buf.String()

	// R's read.csv recognizes "NA" as missing value by default
	if !strings.Contains(result, "NA") {
		t.Error("expected NA for missing values (R compatibility)")
	}
}

func TestCSVExport_RCompatibility_BooleanValues(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	m1 := &CSVMeasurement{ID: "1", IsUnilateral: true}
	m2 := &CSVMeasurement{ID: "2", IsUnilateral: false}

	cw.Write(m1)
	cw.Write(m2)
	cw.Flush()

	result := buf.String()

	// R's read.csv recognizes TRUE/FALSE as logical values
	if !strings.Contains(result, "TRUE") {
		t.Error("expected TRUE for boolean (R compatibility)")
	}
	if !strings.Contains(result, "FALSE") {
		t.Error("expected FALSE for boolean (R compatibility)")
	}
}

// -----------------------------------------------------------------------------
// Python/Pandas Compatibility Tests
// -----------------------------------------------------------------------------

func TestCSVExport_PandasCompatibility_DatetimeFormat(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	timestamp := time.Date(2024, 3, 15, 10, 30, 45, 0, time.UTC)
	m := &CSVMeasurement{
		ID:        "1",
		Timestamp: timestamp,
	}
	cw.Write(m)
	cw.Flush()

	result := buf.String()

	// pandas.read_csv with parse_dates works well with ISO 8601 / RFC3339
	if !strings.Contains(result, "2024-03-15T10:30:45Z") {
		t.Errorf("expected ISO 8601 datetime format for pandas: %s", result)
	}
}

func TestCSVExport_PandasCompatibility_NumericPrecision(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.Precision = 6
	config.IncludeHeader = false
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID:        "1",
		Beta:      1.234567890123,
		Alignment: 0.999999999999,
		CPair:     0.000001234567,
	}
	cw.Write(m)
	cw.Flush()

	// Parse the result to verify precision
	reader := csv.NewReader(strings.NewReader(buf.String()))
	record, err := reader.Read()
	if err != nil {
		t.Fatalf("failed to parse CSV: %v", err)
	}

	// Beta is column index 12 (after d_eff at 11)
	// Just verify the output contains our values at proper precision
	result := buf.String()
	if !strings.Contains(result, "1.234568") { // Rounded to 6 places
		t.Errorf("expected Beta with 6 decimal precision: %s (record: %v)", result, record)
	}
}

// -----------------------------------------------------------------------------
// File Writing Tests
// -----------------------------------------------------------------------------

func TestCSVExport_WriteToFile(t *testing.T) {
	// Create temp directory
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test_export.csv")

	// Open file for writing
	f, err := os.Create(filePath)
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer f.Close()

	measurements := []*CSVMeasurement{
		{
			ID:             "m1",
			Timestamp:      time.Date(2024, 1, 15, 10, 0, 0, 0, time.UTC),
			SessionID:      "session-1",
			ConversationID: "conv-1",
			TurnNumber:     1,
			SenderName:     "Alice",
			ReceiverName:   "Bob",
			DEff:           128,
			Beta:           1.75,
			Alignment:      0.85,
			CPair:          0.92,
			BetaStatus:     "optimal",
		},
		{
			ID:             "m2",
			Timestamp:      time.Date(2024, 1, 15, 10, 1, 0, 0, time.UTC),
			SessionID:      "session-1",
			ConversationID: "conv-1",
			TurnNumber:     2,
			SenderName:     "Bob",
			ReceiverName:   "Alice",
			DEff:           135,
			Beta:           1.82,
			Alignment:      0.88,
			CPair:          0.90,
			BetaStatus:     "optimal",
		},
	}

	err = ExportMeasurementsToCSV(f, measurements, nil)
	if err != nil {
		t.Fatalf("ExportMeasurementsToCSV() failed: %v", err)
	}

	// Read back and verify
	data, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("failed to read temp file: %v", err)
	}

	result := string(data)

	// Verify content
	if !strings.Contains(result, "id,timestamp") {
		t.Error("expected header row")
	}
	if !strings.Contains(result, "m1") {
		t.Error("expected first measurement ID")
	}
	if !strings.Contains(result, "m2") {
		t.Error("expected second measurement ID")
	}
}

// TestCSVExport_WriteTempFileForValidation writes a CSV file to /tmp for
// external validation with R and Python. This is used by the verification command.
func TestCSVExport_WriteTempFileForValidation(t *testing.T) {
	filePath := "/tmp/test_export.csv"

	// Create file
	f, err := os.Create(filePath)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	defer f.Close()

	// Create sample data
	measurements := []*CSVMeasurement{
		{
			ID:             "measure-001",
			Timestamp:      time.Date(2024, 6, 15, 14, 30, 0, 0, time.UTC),
			SessionID:      "session-abc123",
			ConversationID: "conv-def456",
			TurnNumber:     1,
			SenderID:       "agent-1",
			SenderName:     "Claude",
			SenderRole:     "assistant",
			ReceiverID:     "user-1",
			ReceiverName:   "Alice",
			ReceiverRole:   "user",
			DEff:           128,
			Beta:           1.75,
			Alignment:      0.85,
			CPair:          0.92,
			BetaStatus:     "optimal",
			IsUnilateral:   false,
		},
		{
			ID:             "measure-002",
			Timestamp:      time.Date(2024, 6, 15, 14, 31, 0, 0, time.UTC),
			SessionID:      "session-abc123",
			ConversationID: "conv-def456",
			TurnNumber:     2,
			SenderID:       "user-1",
			SenderName:     "Alice",
			SenderRole:     "user",
			ReceiverID:     "agent-1",
			ReceiverName:   "Claude",
			ReceiverRole:   "assistant",
			DEff:           135,
			Beta:           1.82,
			Alignment:      0.88,
			CPair:          0.90,
			BetaStatus:     "optimal",
			IsUnilateral:   false,
		},
		{
			ID:             "measure-003",
			Timestamp:      time.Date(2024, 6, 15, 14, 32, 0, 0, time.UTC),
			SessionID:      "session-abc123",
			ConversationID: "conv-def456",
			TurnNumber:     3,
			SenderID:       "agent-1",
			SenderName:     "Claude",
			SenderRole:     "assistant",
			ReceiverID:     "user-1",
			ReceiverName:   "Alice",
			ReceiverRole:   "user",
			DEff:           142,
			Beta:           2.10,
			Alignment:      0.75,
			CPair:          0.85,
			BetaStatus:     "monitor",
			IsUnilateral:   false,
		},
	}

	err = ExportMeasurementsToCSV(f, measurements, nil)
	if err != nil {
		t.Fatalf("ExportMeasurementsToCSV() failed: %v", err)
	}

	// Verify file was created and has content
	info, err := os.Stat(filePath)
	if err != nil {
		t.Fatalf("failed to stat file: %v", err)
	}
	if info.Size() == 0 {
		t.Error("expected non-empty file")
	}

	// Read and verify basic structure
	data, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}

	content := string(data)
	lines := strings.Split(strings.TrimSpace(content), "\n")

	// Should have header + 3 data rows
	if len(lines) != 4 {
		t.Errorf("expected 4 lines (header + 3 rows), got %d", len(lines))
	}

	t.Logf("Created validation file at %s with %d bytes", filePath, info.Size())
}

// -----------------------------------------------------------------------------
// CSV Parsing Verification Tests
// -----------------------------------------------------------------------------

func TestCSVExport_ParseableOutput(t *testing.T) {
	var buf bytes.Buffer

	measurements := []*CSVMeasurement{
		{
			ID:           "m1",
			Timestamp:    time.Date(2024, 1, 15, 10, 0, 0, 0, time.UTC),
			TurnNumber:   1,
			SenderName:   "Alice",
			ReceiverName: "Bob",
			DEff:         128,
			Beta:         1.75,
			Alignment:    0.85,
			CPair:        0.92,
			BetaStatus:   "optimal",
		},
	}

	err := ExportMeasurementsToCSV(&buf, measurements, nil)
	if err != nil {
		t.Fatalf("ExportMeasurementsToCSV() failed: %v", err)
	}

	// Parse with Go's CSV reader
	reader := csv.NewReader(strings.NewReader(buf.String()))
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("CSV should be parseable: %v", err)
	}

	if len(records) != 2 {
		t.Errorf("expected 2 records (header + 1 row), got %d", len(records))
	}

	// Verify header count matches data count
	if len(records[0]) != len(records[1]) {
		t.Errorf("header column count (%d) should match data column count (%d)",
			len(records[0]), len(records[1]))
	}
}

func TestCSVExport_QuotedStringsWithCommas(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeMessageContent = true
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID:             "1",
		MessageContent: "Hello, world! This contains a comma, and more.",
	}
	cw.Write(m)
	cw.Flush()

	// Parse should still work
	reader := csv.NewReader(strings.NewReader(buf.String()))
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("CSV with commas in values should be parseable: %v", err)
	}

	// Find the message_content column
	headerRow := records[0]
	msgColIdx := -1
	for i, h := range headerRow {
		if h == "message_content" {
			msgColIdx = i
			break
		}
	}

	if msgColIdx < 0 {
		t.Fatal("message_content column not found")
	}

	// Verify the message is correctly parsed
	dataRow := records[1]
	if dataRow[msgColIdx] != "Hello, world! This contains a comma, and more." {
		t.Errorf("message content not correctly parsed: %q", dataRow[msgColIdx])
	}
}

func TestCSVExport_QuotedStringsWithNewlines(t *testing.T) {
	var buf bytes.Buffer
	config := DefaultCSVConfig()
	config.IncludeMessageContent = true
	cw := NewCSVWriter(&buf, config)

	m := &CSVMeasurement{
		ID:             "1",
		MessageContent: "Line 1\nLine 2\nLine 3",
	}
	cw.Write(m)
	cw.Flush()

	// Parse should still work
	reader := csv.NewReader(strings.NewReader(buf.String()))
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("CSV with newlines in values should be parseable: %v", err)
	}

	// Should have 2 records (header + 1 data row), not 4
	if len(records) != 2 {
		t.Errorf("expected 2 records despite newlines in content, got %d", len(records))
	}
}

// -----------------------------------------------------------------------------
// Integration Tests
// -----------------------------------------------------------------------------

func TestCSVExport_FullWorkflow(t *testing.T) {
	// Create sample measurement rows (like from LaTeX export)
	rows := []MeasurementRow{
		{Turn: 1, Sender: "Claude", Receiver: "User", DEff: 128, Beta: 1.65, Alignment: 0.92, CPair: 0.88, BetaStatus: "optimal"},
		{Turn: 2, Sender: "User", Receiver: "Claude", DEff: 135, Beta: 1.72, Alignment: 0.89, CPair: 0.85, BetaStatus: "optimal"},
		{Turn: 3, Sender: "Claude", Receiver: "User", DEff: 142, Beta: 1.95, Alignment: 0.78, CPair: 0.82, BetaStatus: "monitor"},
	}

	// Convert to CSV format
	csvMeasurements := MeasurementRowsToCSV(rows)

	// Export to CSV
	var buf bytes.Buffer
	err := ExportMeasurementsToCSV(&buf, csvMeasurements, nil)
	if err != nil {
		t.Fatalf("export failed: %v", err)
	}

	// Parse and verify
	reader := csv.NewReader(strings.NewReader(buf.String()))
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	// Should have header + 3 data rows
	if len(records) != 4 {
		t.Errorf("expected 4 records, got %d", len(records))
	}

	// Verify header structure
	header := records[0]
	if header[0] != "id" {
		t.Errorf("first header should be 'id', got %q", header[0])
	}

	// Verify data rows have correct turn numbers
	for i := 1; i <= 3; i++ {
		turnColIdx := -1
		for j, h := range header {
			if h == "turn_number" {
				turnColIdx = j
				break
			}
		}
		if turnColIdx >= 0 {
			expectedTurn := string(rune('0' + i))
			if records[i][turnColIdx] != expectedTurn {
				t.Errorf("row %d: expected turn %s, got %s", i, expectedTurn, records[i][turnColIdx])
			}
		}
	}
}

func TestCSVExport_LargeDataset(t *testing.T) {
	// Generate 1000 measurements
	measurements := make([]*CSVMeasurement, 1000)
	for i := 0; i < 1000; i++ {
		measurements[i] = &CSVMeasurement{
			ID:           "m" + string(rune('0'+i%10)) + string(rune('0'+i/10%10)) + string(rune('0'+i/100%10)),
			TurnNumber:   i + 1,
			DEff:         100 + i%50,
			Beta:         1.5 + float64(i%100)/100.0,
			Alignment:    0.7 + float64(i%30)/100.0,
			CPair:        0.6 + float64(i%40)/100.0,
			BetaStatus:   []string{"optimal", "monitor", "concerning"}[i%3],
			IsUnilateral: i%5 == 0,
		}
	}

	var buf bytes.Buffer
	err := ExportMeasurementsToCSV(&buf, measurements, nil)
	if err != nil {
		t.Fatalf("large dataset export failed: %v", err)
	}

	// Parse and verify row count
	reader := csv.NewReader(strings.NewReader(buf.String()))
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	// Should have header + 1000 data rows
	if len(records) != 1001 {
		t.Errorf("expected 1001 records, got %d", len(records))
	}
}
