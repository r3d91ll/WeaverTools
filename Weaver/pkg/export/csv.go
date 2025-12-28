// Package export provides academic export format utilities.
// Generates LaTeX tables, CSV data, and other publication-ready outputs.
package export

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
	"time"
)

// CSVDialect specifies the CSV format variant.
type CSVDialect string

const (
	// DialectStandard uses RFC 4180 compliant CSV (comma-separated, quoted strings).
	DialectStandard CSVDialect = "standard"

	// DialectExcel uses Excel-compatible format with potential BOM for UTF-8.
	DialectExcel CSVDialect = "excel"

	// DialectTSV uses tab-separated values instead of comma.
	DialectTSV CSVDialect = "tsv"
)

// CSVConfig specifies options for CSV export.
type CSVConfig struct {
	// Dialect specifies the CSV format variant.
	// Default: DialectStandard
	Dialect CSVDialect

	// IncludeHeader writes column headers as the first row.
	// Default: true
	IncludeHeader bool

	// TimestampFormat specifies the format for timestamp columns.
	// Default: time.RFC3339 (ISO 8601 format, compatible with R and Python).
	TimestampFormat string

	// Precision is the number of decimal places for floating-point values.
	// Default: 6 (sufficient for statistical analysis)
	Precision int

	// NAString is the representation for missing/NA values.
	// Default: "NA" (compatible with R and Python pandas)
	NAString string

	// IncludeBetaStatus includes the beta_status column.
	// Default: true
	IncludeBetaStatus bool

	// IncludeMessageContent includes the message_content column.
	// Default: false (can be large and may not be needed for analysis)
	IncludeMessageContent bool

	// IncludeTokenCount includes the token_count column.
	// Default: false
	IncludeTokenCount bool
}

// DefaultCSVConfig returns a CSVConfig with sensible defaults.
// Uses RFC 4180 standard format with ISO 8601 timestamps.
func DefaultCSVConfig() *CSVConfig {
	return &CSVConfig{
		Dialect:               DialectStandard,
		IncludeHeader:         true,
		TimestampFormat:       time.RFC3339,
		Precision:             6,
		NAString:              "NA",
		IncludeBetaStatus:     true,
		IncludeMessageContent: false,
		IncludeTokenCount:     false,
	}
}

// CSVMeasurement represents a measurement row for CSV export.
// This flattens the nested Measurement structure for tabular output.
type CSVMeasurement struct {
	// ID is the unique measurement identifier.
	ID string

	// Timestamp is when the measurement was taken.
	Timestamp time.Time

	// Session context
	SessionID      string
	ConversationID string
	TurnNumber     int

	// Participants
	SenderID     string
	SenderName   string
	SenderRole   string
	ReceiverID   string
	ReceiverName string
	ReceiverRole string

	// Core conveyance metrics
	DEff      int
	Beta      float64
	Alignment float64
	CPair     float64

	// Quality indicators
	BetaStatus   string
	IsUnilateral bool

	// Message context (optional)
	MessageContent string
	TokenCount     int
}

// CSVWriter writes measurements to CSV format.
type CSVWriter struct {
	config      *CSVConfig
	writer      *csv.Writer
	headerDone  bool
	rowsWritten int
}

// NewCSVWriter creates a new CSVWriter that writes to the given io.Writer.
// If config is nil, DefaultCSVConfig() is used.
func NewCSVWriter(w io.Writer, config *CSVConfig) *CSVWriter {
	if config == nil {
		config = DefaultCSVConfig()
	}

	csvWriter := csv.NewWriter(w)

	// Set delimiter based on dialect
	if config.Dialect == DialectTSV {
		csvWriter.Comma = '\t'
	}

	return &CSVWriter{
		config:      config,
		writer:      csvWriter,
		headerDone:  false,
		rowsWritten: 0,
	}
}

// WriteHeader writes the CSV header row.
// This is called automatically on first Write if IncludeHeader is true.
func (cw *CSVWriter) WriteHeader() error {
	if cw.headerDone {
		return nil
	}

	headers := cw.buildHeaders()
	if err := cw.writer.Write(headers); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	cw.headerDone = true
	return nil
}

// Write writes a single measurement row to the CSV.
// If IncludeHeader is true and this is the first write, the header is written first.
func (cw *CSVWriter) Write(m *CSVMeasurement) error {
	// Write header if needed
	if cw.config.IncludeHeader && !cw.headerDone {
		if err := cw.WriteHeader(); err != nil {
			return err
		}
	}

	row := cw.formatMeasurement(m)
	if err := cw.writer.Write(row); err != nil {
		return fmt.Errorf("failed to write CSV row: %w", err)
	}

	cw.rowsWritten++
	return nil
}

// WriteAll writes multiple measurement rows to the CSV.
func (cw *CSVWriter) WriteAll(measurements []*CSVMeasurement) error {
	for _, m := range measurements {
		if err := cw.Write(m); err != nil {
			return err
		}
	}
	return nil
}

// Flush flushes any buffered data to the underlying writer.
func (cw *CSVWriter) Flush() error {
	cw.writer.Flush()
	if err := cw.writer.Error(); err != nil {
		return fmt.Errorf("failed to flush CSV writer: %w", err)
	}
	return nil
}

// RowsWritten returns the number of data rows written (excluding header).
func (cw *CSVWriter) RowsWritten() int {
	return cw.rowsWritten
}

// buildHeaders constructs the column headers based on configuration.
func (cw *CSVWriter) buildHeaders() []string {
	// Build headers in a fixed order that's compatible with R and Python.
	// Column names use snake_case for R compatibility and are valid identifiers.
	headers := []string{
		"id",
		"timestamp",
		"session_id",
		"conversation_id",
		"turn_number",
		"sender_id",
		"sender_name",
		"sender_role",
		"receiver_id",
		"receiver_name",
		"receiver_role",
		"d_eff",
		"beta",
		"alignment",
		"c_pair",
		"is_unilateral",
	}

	if cw.config.IncludeBetaStatus {
		headers = append(headers, "beta_status")
	}

	if cw.config.IncludeMessageContent {
		headers = append(headers, "message_content")
	}

	if cw.config.IncludeTokenCount {
		headers = append(headers, "token_count")
	}

	return headers
}

// formatMeasurement converts a measurement to a slice of strings for CSV output.
func (cw *CSVWriter) formatMeasurement(m *CSVMeasurement) []string {
	if m == nil {
		return cw.buildEmptyRow()
	}

	precision := cw.config.Precision
	na := cw.config.NAString

	// Format timestamp
	timestamp := m.Timestamp.UTC().Format(cw.config.TimestampFormat)
	if m.Timestamp.IsZero() {
		timestamp = na
	}

	// Build row in the same order as headers
	row := []string{
		cw.formatString(m.ID, na),
		timestamp,
		cw.formatString(m.SessionID, na),
		cw.formatString(m.ConversationID, na),
		strconv.Itoa(m.TurnNumber),
		cw.formatString(m.SenderID, na),
		cw.formatString(m.SenderName, na),
		cw.formatString(m.SenderRole, na),
		cw.formatString(m.ReceiverID, na),
		cw.formatString(m.ReceiverName, na),
		cw.formatString(m.ReceiverRole, na),
		strconv.Itoa(m.DEff),
		cw.formatFloat(m.Beta, precision),
		cw.formatFloat(m.Alignment, precision),
		cw.formatFloat(m.CPair, precision),
		cw.formatBool(m.IsUnilateral),
	}

	if cw.config.IncludeBetaStatus {
		row = append(row, cw.formatString(m.BetaStatus, na))
	}

	if cw.config.IncludeMessageContent {
		row = append(row, cw.formatString(m.MessageContent, na))
	}

	if cw.config.IncludeTokenCount {
		row = append(row, strconv.Itoa(m.TokenCount))
	}

	return row
}

// buildEmptyRow creates a row with NA values for all columns.
func (cw *CSVWriter) buildEmptyRow() []string {
	headers := cw.buildHeaders()
	row := make([]string, len(headers))
	for i := range row {
		row[i] = cw.config.NAString
	}
	return row
}

// formatString returns the string value or NA if empty.
func (cw *CSVWriter) formatString(s, na string) string {
	if s == "" {
		return na
	}
	return s
}

// formatFloat formats a float64 with the specified precision.
func (cw *CSVWriter) formatFloat(f float64, precision int) string {
	return strconv.FormatFloat(f, 'f', precision, 64)
}

// formatBool formats a boolean as "TRUE" or "FALSE" for R/Python compatibility.
func (cw *CSVWriter) formatBool(b bool) string {
	if b {
		return "TRUE"
	}
	return "FALSE"
}

// ExportMeasurementsToCSV is a convenience function to export measurements to CSV.
// If config is nil, DefaultCSVConfig() is used.
func ExportMeasurementsToCSV(w io.Writer, measurements []*CSVMeasurement, config *CSVConfig) error {
	writer := NewCSVWriter(w, config)

	if err := writer.WriteAll(measurements); err != nil {
		return err
	}

	return writer.Flush()
}

// MeasurementRowToCSV converts a MeasurementRow (from LaTeX export) to CSVMeasurement.
// This provides interoperability between export formats.
func MeasurementRowToCSV(row MeasurementRow) *CSVMeasurement {
	return &CSVMeasurement{
		TurnNumber: row.Turn,
		SenderName: row.Sender,
		ReceiverName: row.Receiver,
		DEff:       row.DEff,
		Beta:       row.Beta,
		Alignment:  row.Alignment,
		CPair:      row.CPair,
		BetaStatus: row.BetaStatus,
	}
}

// MeasurementRowsToCSV converts multiple MeasurementRows to CSVMeasurements.
func MeasurementRowsToCSV(rows []MeasurementRow) []*CSVMeasurement {
	result := make([]*CSVMeasurement, len(rows))
	for i, row := range rows {
		result[i] = MeasurementRowToCSV(row)
	}
	return result
}
