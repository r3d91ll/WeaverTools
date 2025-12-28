// Package export provides academic export format utilities.
// Generates LaTeX tables, CSV data, and other publication-ready outputs.
package export

import (
	"encoding/csv"
	"encoding/json"
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

// -----------------------------------------------------------------------------
// Metadata Export for Data Dictionary
// -----------------------------------------------------------------------------

// ColumnType represents the data type of a column.
type ColumnType string

const (
	// TypeString is a text/string column.
	TypeString ColumnType = "string"

	// TypeInteger is an integer numeric column.
	TypeInteger ColumnType = "integer"

	// TypeFloat is a floating-point numeric column.
	TypeFloat ColumnType = "float"

	// TypeBoolean is a true/false column.
	TypeBoolean ColumnType = "boolean"

	// TypeDatetime is a timestamp/datetime column.
	TypeDatetime ColumnType = "datetime"

	// TypeCategorical is a categorical/factor column with defined levels.
	TypeCategorical ColumnType = "categorical"
)

// ColumnMetadata describes a single column in the data dictionary.
type ColumnMetadata struct {
	// Name is the column name as it appears in the CSV header.
	Name string `json:"name"`

	// Type is the data type of the column.
	Type ColumnType `json:"type"`

	// Description is a human-readable description of the column.
	Description string `json:"description"`

	// Unit is the unit of measurement (if applicable).
	// Empty for dimensionless values.
	Unit string `json:"unit,omitempty"`

	// RType is the corresponding R type (for R users).
	RType string `json:"r_type"`

	// PandasType is the corresponding pandas dtype (for Python users).
	PandasType string `json:"pandas_type"`

	// ValidRange specifies the valid range for numeric columns.
	// Format: "[min, max]" or "(min, max)" or "≥ 0", etc.
	ValidRange string `json:"valid_range,omitempty"`

	// ValidValues lists valid values for categorical columns.
	ValidValues []string `json:"valid_values,omitempty"`

	// NAHandling describes how NA/missing values are represented.
	NAHandling string `json:"na_handling,omitempty"`
}

// DataDictionaryMetadata is the complete metadata for a CSV export.
// This provides a data dictionary describing all columns, their types,
// and semantics for use in R and Python analysis workflows.
type DataDictionaryMetadata struct {
	// Version is the metadata format version.
	Version string `json:"version"`

	// GeneratedAt is when this metadata was generated.
	GeneratedAt time.Time `json:"generated_at"`

	// Description is an overview of the dataset.
	Description string `json:"description"`

	// Columns is the ordered list of column metadata.
	Columns []ColumnMetadata `json:"columns"`

	// NAString is the string used to represent missing values.
	NAString string `json:"na_string"`

	// TimestampFormat is the format used for datetime columns.
	TimestampFormat string `json:"timestamp_format"`

	// FloatPrecision is the number of decimal places for floats.
	FloatPrecision int `json:"float_precision"`

	// BooleanFormat describes the boolean representation.
	BooleanFormat string `json:"boolean_format"`

	// Notes contains additional notes for data consumers.
	Notes []string `json:"notes,omitempty"`
}

// GetColumnMetadata returns the standard column metadata for CSVMeasurement exports.
// This includes core columns that are always present in the export.
func GetColumnMetadata() []ColumnMetadata {
	return []ColumnMetadata{
		{
			Name:        "id",
			Type:        TypeString,
			Description: "Unique identifier for the measurement (UUID format)",
			RType:       "character",
			PandasType:  "object",
			NAHandling:  "Represented as 'NA'",
		},
		{
			Name:            "timestamp",
			Type:            TypeDatetime,
			Description:     "ISO 8601 timestamp when the measurement was recorded",
			RType:           "POSIXct",
			PandasType:      "datetime64[ns]",
			NAHandling:      "Represented as 'NA'",
		},
		{
			Name:        "session_id",
			Type:        TypeString,
			Description: "Identifier for the session containing this measurement",
			RType:       "character",
			PandasType:  "object",
			NAHandling:  "Represented as 'NA'",
		},
		{
			Name:        "conversation_id",
			Type:        TypeString,
			Description: "Identifier for the conversation within the session",
			RType:       "character",
			PandasType:  "object",
			NAHandling:  "Represented as 'NA'",
		},
		{
			Name:        "turn_number",
			Type:        TypeInteger,
			Description: "Sequential turn number within the conversation (0-indexed)",
			RType:       "integer",
			PandasType:  "int64",
			ValidRange:  "≥ 0",
		},
		{
			Name:        "sender_id",
			Type:        TypeString,
			Description: "Identifier for the message sender",
			RType:       "character",
			PandasType:  "object",
			NAHandling:  "Represented as 'NA'",
		},
		{
			Name:        "sender_name",
			Type:        TypeString,
			Description: "Display name of the message sender",
			RType:       "character",
			PandasType:  "object",
			NAHandling:  "Represented as 'NA'",
		},
		{
			Name:            "sender_role",
			Type:            TypeCategorical,
			Description:     "Role of the message sender in the conversation",
			RType:           "factor",
			PandasType:      "category",
			ValidValues:     []string{"user", "assistant", "system"},
			NAHandling:      "Represented as 'NA'",
		},
		{
			Name:        "receiver_id",
			Type:        TypeString,
			Description: "Identifier for the message receiver",
			RType:       "character",
			PandasType:  "object",
			NAHandling:  "Represented as 'NA'",
		},
		{
			Name:        "receiver_name",
			Type:        TypeString,
			Description: "Display name of the message receiver",
			RType:       "character",
			PandasType:  "object",
			NAHandling:  "Represented as 'NA'",
		},
		{
			Name:            "receiver_role",
			Type:            TypeCategorical,
			Description:     "Role of the message receiver in the conversation",
			RType:           "factor",
			PandasType:      "category",
			ValidValues:     []string{"user", "assistant", "system"},
			NAHandling:      "Represented as 'NA'",
		},
		{
			Name:        "d_eff",
			Type:        TypeInteger,
			Description: "Effective dimensionality of the hidden state representation",
			Unit:        "dimensions",
			RType:       "integer",
			PandasType:  "int64",
			ValidRange:  "≥ 0",
		},
		{
			Name:        "beta",
			Type:        TypeFloat,
			Description: "Collapse indicator (β) from the Conveyance Framework. Lower values indicate better dimensional preservation.",
			RType:       "numeric",
			PandasType:  "float64",
			ValidRange:  "≥ 0",
		},
		{
			Name:        "alignment",
			Type:        TypeFloat,
			Description: "Cosine similarity between sender and receiver hidden states",
			RType:       "numeric",
			PandasType:  "float64",
			ValidRange:  "[-1, 1]",
		},
		{
			Name:        "c_pair",
			Type:        TypeFloat,
			Description: "Bilateral conveyance score measuring communication effectiveness",
			RType:       "numeric",
			PandasType:  "float64",
			ValidRange:  "[0, 1]",
		},
		{
			Name:        "is_unilateral",
			Type:        TypeBoolean,
			Description: "Whether the measurement is unilateral (only one hidden state available)",
			RType:       "logical",
			PandasType:  "bool",
			NAHandling:  "FALSE if missing",
		},
	}
}

// GetBetaStatusMetadata returns metadata for the optional beta_status column.
func GetBetaStatusMetadata() ColumnMetadata {
	return ColumnMetadata{
		Name:        "beta_status",
		Type:        TypeCategorical,
		Description: "Quality classification of the β value indicating dimensional preservation status",
		RType:       "factor",
		PandasType:  "category",
		ValidValues: []string{"optimal", "monitor", "concerning", "critical", "unknown"},
		NAHandling:  "Represented as 'NA'",
	}
}

// GetMessageContentMetadata returns metadata for the optional message_content column.
func GetMessageContentMetadata() ColumnMetadata {
	return ColumnMetadata{
		Name:        "message_content",
		Type:        TypeString,
		Description: "Full text content of the message (may be large)",
		RType:       "character",
		PandasType:  "object",
		NAHandling:  "Represented as 'NA'",
	}
}

// GetTokenCountMetadata returns metadata for the optional token_count column.
func GetTokenCountMetadata() ColumnMetadata {
	return ColumnMetadata{
		Name:        "token_count",
		Type:        TypeInteger,
		Description: "Number of tokens in the message",
		Unit:        "tokens",
		RType:       "integer",
		PandasType:  "int64",
		ValidRange:  "≥ 0",
	}
}

// GenerateDataDictionaryMetadata creates complete metadata for a CSV export.
// The config parameter determines which optional columns are included.
func GenerateDataDictionaryMetadata(config *CSVConfig) *DataDictionaryMetadata {
	if config == nil {
		config = DefaultCSVConfig()
	}

	// Start with core columns
	columns := GetColumnMetadata()

	// Add optional columns based on config
	if config.IncludeBetaStatus {
		columns = append(columns, GetBetaStatusMetadata())
	}
	if config.IncludeMessageContent {
		columns = append(columns, GetMessageContentMetadata())
	}
	if config.IncludeTokenCount {
		columns = append(columns, GetTokenCountMetadata())
	}

	metadata := &DataDictionaryMetadata{
		Version:         "1.0",
		GeneratedAt:     time.Now().UTC(),
		Description:     "Conveyance measurement data from WeaverTools. Contains metrics capturing information transfer quality between agents in multi-agent conversations.",
		Columns:         columns,
		NAString:        config.NAString,
		TimestampFormat: config.TimestampFormat,
		FloatPrecision:  config.Precision,
		BooleanFormat:   "TRUE/FALSE",
		Notes: []string{
			"R users: Use read.csv() with default settings. NA values are recognized automatically.",
			"Python users: Use pandas.read_csv() with parse_dates=['timestamp'] for datetime parsing.",
			"Beta status thresholds: optimal=[1.5,2.0), monitor=[2.0,2.5), concerning=[2.5,3.0), critical=≥3.0",
			"Alignment values near 1.0 indicate high similarity; values near -1.0 indicate opposition.",
		},
	}

	return metadata
}

// ExportMetadataToJSON writes the data dictionary metadata to JSON format.
func ExportMetadataToJSON(w io.Writer, config *CSVConfig) error {
	metadata := GenerateDataDictionaryMetadata(config)

	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")

	if err := encoder.Encode(metadata); err != nil {
		return fmt.Errorf("failed to encode metadata to JSON: %w", err)
	}

	return nil
}

// ExportMetadataToJSONBytes returns the data dictionary metadata as a JSON byte slice.
func ExportMetadataToJSONBytes(config *CSVConfig) ([]byte, error) {
	metadata := GenerateDataDictionaryMetadata(config)

	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal metadata to JSON: %w", err)
	}

	return data, nil
}
