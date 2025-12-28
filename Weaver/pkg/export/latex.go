// Package export provides academic export format utilities.
// Generates LaTeX tables, CSV data, and other publication-ready outputs.
package export

import (
	"fmt"
	"strings"
)

// LaTeX special characters that require escaping.
// These characters have special meaning in LaTeX and must be escaped
// to prevent compilation errors or unintended formatting.
var latexSpecialChars = map[rune]string{
	'\\': `\textbackslash{}`,
	'{':  `\{`,
	'}':  `\}`,
	'$':  `\$`,
	'&':  `\&`,
	'#':  `\#`,
	'%':  `\%`,
	'_':  `\_`,
	'^':  `\textasciicircum{}`,
	'~':  `\textasciitilde{}`,
}

// LaTeX table column alignments.
const (
	AlignLeft   = "l"
	AlignCenter = "c"
	AlignRight  = "r"
)

// LaTeX table style options.
const (
	StylePlain    = "plain"    // Standard LaTeX tabular
	StyleBooktabs = "booktabs" // Professional booktabs style
)

// LaTeXEscaper handles text escaping for safe LaTeX output.
type LaTeXEscaper struct {
	// PreserveNewlines converts newlines to LaTeX line breaks (\\).
	// When false, newlines are converted to spaces.
	PreserveNewlines bool

	// PreserveMath leaves text inside $...$ unescaped.
	// This allows inline math expressions to pass through.
	PreserveMath bool
}

// DefaultLaTeXEscaper returns an escaper with sensible defaults.
// Newlines are converted to spaces, math mode is not preserved.
func DefaultLaTeXEscaper() *LaTeXEscaper {
	return &LaTeXEscaper{
		PreserveNewlines: false,
		PreserveMath:     false,
	}
}

// Escape converts a string to be safely included in LaTeX documents.
// Special characters are escaped to prevent LaTeX compilation errors.
func Escape(s string) string {
	return DefaultLaTeXEscaper().Escape(s)
}

// EscapeLaTeX is an alias for Escape for clarity in contexts where
// the LaTeX-specific nature of the escaping should be explicit.
var EscapeLaTeX = Escape

// Escape converts a string to be safely included in LaTeX documents.
// Respects the escaper's configuration for newlines and math mode.
func (e *LaTeXEscaper) Escape(s string) string {
	if s == "" {
		return ""
	}

	if e.PreserveMath {
		return e.escapeWithMath(s)
	}

	return e.escapeAll(s)
}

// escapeAll escapes all special characters without preserving math mode.
func (e *LaTeXEscaper) escapeAll(s string) string {
	var sb strings.Builder
	sb.Grow(len(s) * 2) // Preallocate for potential expansion

	for _, r := range s {
		if r == '\n' {
			if e.PreserveNewlines {
				sb.WriteString(`\\`)
			} else {
				sb.WriteRune(' ')
			}
			continue
		}

		if escaped, ok := latexSpecialChars[r]; ok {
			sb.WriteString(escaped)
		} else {
			sb.WriteRune(r)
		}
	}

	return sb.String()
}

// escapeWithMath escapes text while preserving inline math expressions.
// Text within $...$ delimiters is passed through unescaped.
func (e *LaTeXEscaper) escapeWithMath(s string) string {
	var sb strings.Builder
	sb.Grow(len(s) * 2)

	inMath := false
	for _, r := range s {
		if r == '$' {
			inMath = !inMath
			sb.WriteRune(r)
			continue
		}

		if inMath {
			// Inside math mode, pass through unescaped
			sb.WriteRune(r)
			continue
		}

		// Outside math mode, apply escaping
		if r == '\n' {
			if e.PreserveNewlines {
				sb.WriteString(`\\`)
			} else {
				sb.WriteRune(' ')
			}
			continue
		}

		if escaped, ok := latexSpecialChars[r]; ok {
			sb.WriteString(escaped)
		} else {
			sb.WriteRune(r)
		}
	}

	return sb.String()
}

// EscapeForCell escapes text specifically for use in table cells.
// This handles additional considerations like alignment and spacing.
func EscapeForCell(s string) string {
	// Trim whitespace and escape
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	return Escape(s)
}

// TableConfig specifies options for LaTeX table generation.
type TableConfig struct {
	// Style determines the table formatting style.
	// Supported: "plain" (default), "booktabs"
	Style string

	// Caption is the table caption (optional).
	Caption string

	// Label is the LaTeX label for cross-references (optional).
	// Example: "tab:results"
	Label string

	// ColumnAlignments specifies alignment for each column.
	// Use AlignLeft, AlignCenter, AlignRight constants.
	// If nil, all columns default to left alignment.
	ColumnAlignments []string

	// IncludeRowNumbers adds a row number column.
	IncludeRowNumbers bool
}

// DefaultTableConfig returns a TableConfig with sensible defaults.
// Uses booktabs style with left-aligned columns.
func DefaultTableConfig() *TableConfig {
	return &TableConfig{
		Style:             StyleBooktabs,
		ColumnAlignments:  nil, // Will use AlignLeft for all
		IncludeRowNumbers: false,
	}
}

// TableBuilder constructs LaTeX tables from data.
type TableBuilder struct {
	config  *TableConfig
	headers []string
	rows    [][]string
}

// NewTableBuilder creates a new table builder with the given configuration.
// If config is nil, DefaultTableConfig() is used.
func NewTableBuilder(config *TableConfig) *TableBuilder {
	if config == nil {
		config = DefaultTableConfig()
	}
	return &TableBuilder{
		config:  config,
		headers: nil,
		rows:    nil,
	}
}

// SetHeaders sets the column headers for the table.
// Headers are escaped for safe LaTeX output.
func (tb *TableBuilder) SetHeaders(headers ...string) *TableBuilder {
	tb.headers = make([]string, len(headers))
	for i, h := range headers {
		tb.headers[i] = EscapeForCell(h)
	}
	return tb
}

// SetHeadersRaw sets the column headers without escaping.
// Use this when headers contain pre-formatted LaTeX (e.g., math mode like $D_{eff}$).
func (tb *TableBuilder) SetHeadersRaw(headers ...string) *TableBuilder {
	tb.headers = make([]string, len(headers))
	copy(tb.headers, headers)
	return tb
}

// AddRow adds a data row to the table.
// All values are escaped for safe LaTeX output.
func (tb *TableBuilder) AddRow(values ...string) *TableBuilder {
	row := make([]string, len(values))
	for i, v := range values {
		row[i] = EscapeForCell(v)
	}
	tb.rows = append(tb.rows, row)
	return tb
}

// AddRowRaw adds a data row without escaping.
// Use this when values contain pre-formatted LaTeX (e.g., math mode like $\beta$).
func (tb *TableBuilder) AddRowRaw(values ...string) *TableBuilder {
	row := make([]string, len(values))
	copy(row, values)
	tb.rows = append(tb.rows, row)
	return tb
}

// Build generates the complete LaTeX table code.
// Returns the table as a string ready for inclusion in a LaTeX document.
func (tb *TableBuilder) Build() string {
	var sb strings.Builder

	numCols := tb.columnCount()
	if numCols == 0 {
		return ""
	}

	// Build column specification
	colSpec := tb.buildColumnSpec(numCols)

	// Begin table environment
	if tb.config.Caption != "" || tb.config.Label != "" {
		sb.WriteString("\\begin{table}[htbp]\n")
		sb.WriteString("\\centering\n")
	}

	// Begin tabular
	sb.WriteString("\\begin{tabular}{")
	sb.WriteString(colSpec)
	sb.WriteString("}\n")

	// Top rule
	if tb.config.Style == StyleBooktabs {
		sb.WriteString("\\toprule\n")
	} else {
		sb.WriteString("\\hline\n")
	}

	// Headers
	if len(tb.headers) > 0 {
		tb.writeRow(&sb, tb.headers, numCols)
		sb.WriteString("\n")
		if tb.config.Style == StyleBooktabs {
			sb.WriteString("\\midrule\n")
		} else {
			sb.WriteString("\\hline\n")
		}
	}

	// Data rows
	for i, row := range tb.rows {
		if tb.config.IncludeRowNumbers {
			numberedRow := append([]string{fmt.Sprintf("%d", i+1)}, row...)
			tb.writeRow(&sb, numberedRow, numCols)
		} else {
			tb.writeRow(&sb, row, numCols)
		}
		sb.WriteString("\n")
	}

	// Bottom rule
	if tb.config.Style == StyleBooktabs {
		sb.WriteString("\\bottomrule\n")
	} else {
		sb.WriteString("\\hline\n")
	}

	// End tabular
	sb.WriteString("\\end{tabular}\n")

	// Caption and label
	if tb.config.Caption != "" {
		sb.WriteString("\\caption{")
		sb.WriteString(Escape(tb.config.Caption))
		sb.WriteString("}\n")
	}
	if tb.config.Label != "" {
		sb.WriteString("\\label{")
		sb.WriteString(tb.config.Label)
		sb.WriteString("}\n")
	}

	// End table environment
	if tb.config.Caption != "" || tb.config.Label != "" {
		sb.WriteString("\\end{table}\n")
	}

	return sb.String()
}

// columnCount returns the number of columns based on headers or first row.
func (tb *TableBuilder) columnCount() int {
	if len(tb.headers) > 0 {
		count := len(tb.headers)
		if tb.config.IncludeRowNumbers {
			count++
		}
		return count
	}
	if len(tb.rows) > 0 {
		count := len(tb.rows[0])
		if tb.config.IncludeRowNumbers {
			count++
		}
		return count
	}
	return 0
}

// buildColumnSpec creates the column alignment specification.
func (tb *TableBuilder) buildColumnSpec(numCols int) string {
	var sb strings.Builder

	for i := 0; i < numCols; i++ {
		if i < len(tb.config.ColumnAlignments) {
			sb.WriteString(tb.config.ColumnAlignments[i])
		} else {
			sb.WriteString(AlignLeft) // Default to left alignment
		}
	}

	return sb.String()
}

// writeRow writes a single table row with proper column separators.
func (tb *TableBuilder) writeRow(sb *strings.Builder, row []string, numCols int) {
	for i := 0; i < numCols; i++ {
		if i > 0 {
			sb.WriteString(" & ")
		}
		if i < len(row) {
			sb.WriteString(row[i])
		}
	}
	sb.WriteString(" \\\\")
}

// FormatNumber formats a float64 for LaTeX output with specified precision.
func FormatNumber(value float64, precision int) string {
	format := fmt.Sprintf("%%.%df", precision)
	return fmt.Sprintf(format, value)
}

// FormatPercent formats a value as a percentage for LaTeX output.
// The value should be a decimal (0.0-1.0), which is converted to percentage.
func FormatPercent(value float64, precision int) string {
	pct := value * 100
	format := fmt.Sprintf("%%.%df", precision)
	return fmt.Sprintf(format, pct) + `\%`
}

// -----------------------------------------------------------------------------
// Measurement Summary Tables
// -----------------------------------------------------------------------------

// MeasurementRow represents a single row of measurement data for table export.
// This is a simplified view of measurement data optimized for academic tables.
type MeasurementRow struct {
	Turn       int     // Turn number in conversation
	Sender     string  // Sender name or identifier
	Receiver   string  // Receiver name or identifier
	DEff       int     // Effective dimensionality
	Beta       float64 // Collapse indicator
	Alignment  float64 // Cosine similarity [-1, 1]
	CPair      float64 // Bilateral conveyance
	BetaStatus string  // Quality indicator (optimal, monitor, concerning, critical, unknown)
}

// MeasurementTableConfig specifies options for measurement summary tables.
type MeasurementTableConfig struct {
	// TableConfig provides base table configuration.
	TableConfig

	// IncludeTurn includes the turn number column.
	IncludeTurn bool

	// IncludeParticipants includes sender/receiver columns.
	IncludeParticipants bool

	// IncludeBetaStatus includes the beta status column.
	IncludeBetaStatus bool

	// Precision is the number of decimal places for floating-point values.
	// Default: 3
	Precision int

	// AlignmentAsPercent formats alignment as percentage instead of decimal.
	AlignmentAsPercent bool
}

// DefaultMeasurementTableConfig returns a config with sensible defaults.
// Includes DEff, Beta, Alignment, CPair columns with booktabs style.
func DefaultMeasurementTableConfig() *MeasurementTableConfig {
	return &MeasurementTableConfig{
		TableConfig: TableConfig{
			Style: StyleBooktabs,
		},
		IncludeTurn:         true,
		IncludeParticipants: true,
		IncludeBetaStatus:   false,
		Precision:           3,
		AlignmentAsPercent:  false,
	}
}

// MeasurementTableBuilder builds LaTeX tables from measurement data.
type MeasurementTableBuilder struct {
	config *MeasurementTableConfig
	rows   []MeasurementRow
}

// NewMeasurementTableBuilder creates a new measurement table builder.
// If config is nil, DefaultMeasurementTableConfig() is used.
func NewMeasurementTableBuilder(config *MeasurementTableConfig) *MeasurementTableBuilder {
	if config == nil {
		config = DefaultMeasurementTableConfig()
	}
	return &MeasurementTableBuilder{
		config: config,
		rows:   make([]MeasurementRow, 0),
	}
}

// AddRow adds a measurement row to the table.
func (mtb *MeasurementTableBuilder) AddRow(row MeasurementRow) *MeasurementTableBuilder {
	mtb.rows = append(mtb.rows, row)
	return mtb
}

// AddRows adds multiple measurement rows to the table.
func (mtb *MeasurementTableBuilder) AddRows(rows []MeasurementRow) *MeasurementTableBuilder {
	mtb.rows = append(mtb.rows, rows...)
	return mtb
}

// Build generates the complete LaTeX table code for measurements.
func (mtb *MeasurementTableBuilder) Build() string {
	if len(mtb.rows) == 0 {
		return ""
	}

	// Build headers based on config
	headers := mtb.buildHeaders()

	// Build column alignments
	alignments := mtb.buildAlignments()

	// Create table config
	tableConfig := &TableConfig{
		Style:            mtb.config.Style,
		Caption:          mtb.config.Caption,
		Label:            mtb.config.Label,
		ColumnAlignments: alignments,
	}

	// Create table builder
	tb := NewTableBuilder(tableConfig)
	// Use SetHeadersRaw since headers contain LaTeX math ($D_{eff}$, $\beta$, etc.)
	tb.SetHeadersRaw(headers...)

	// Add data rows
	for _, row := range mtb.rows {
		rowData := mtb.formatRow(row)
		tb.AddRow(rowData...)
	}

	return tb.Build()
}

// buildHeaders generates the column headers based on configuration.
func (mtb *MeasurementTableBuilder) buildHeaders() []string {
	headers := make([]string, 0, 8)

	if mtb.config.IncludeTurn {
		headers = append(headers, "Turn")
	}
	if mtb.config.IncludeParticipants {
		headers = append(headers, "Sender", "Receiver")
	}

	// Core metric columns are always included
	headers = append(headers, "$D_{eff}$", "$\\beta$", "Alignment", "$C_{pair}$")

	if mtb.config.IncludeBetaStatus {
		headers = append(headers, "$\\beta$ Status")
	}

	return headers
}

// buildAlignments generates column alignment specifications.
func (mtb *MeasurementTableBuilder) buildAlignments() []string {
	alignments := make([]string, 0, 8)

	if mtb.config.IncludeTurn {
		alignments = append(alignments, AlignCenter) // Turn number centered
	}
	if mtb.config.IncludeParticipants {
		alignments = append(alignments, AlignLeft, AlignLeft) // Names left-aligned
	}

	// Numeric columns right-aligned
	alignments = append(alignments, AlignRight, AlignRight, AlignRight, AlignRight)

	if mtb.config.IncludeBetaStatus {
		alignments = append(alignments, AlignLeft) // Status text left-aligned
	}

	return alignments
}

// formatRow converts a MeasurementRow to string values for the table.
func (mtb *MeasurementTableBuilder) formatRow(row MeasurementRow) []string {
	values := make([]string, 0, 8)
	precision := mtb.config.Precision

	if mtb.config.IncludeTurn {
		values = append(values, fmt.Sprintf("%d", row.Turn))
	}
	if mtb.config.IncludeParticipants {
		values = append(values, row.Sender, row.Receiver)
	}

	// Core metrics
	values = append(values, fmt.Sprintf("%d", row.DEff))
	values = append(values, FormatNumber(row.Beta, precision))

	if mtb.config.AlignmentAsPercent {
		values = append(values, FormatPercent(row.Alignment, precision-1))
	} else {
		values = append(values, FormatNumber(row.Alignment, precision))
	}

	values = append(values, FormatNumber(row.CPair, precision))

	if mtb.config.IncludeBetaStatus {
		values = append(values, row.BetaStatus)
	}

	return values
}

// GenerateMeasurementTable is a convenience function to generate a measurement
// summary table from a slice of MeasurementRow data.
// If config is nil, DefaultMeasurementTableConfig() is used.
func GenerateMeasurementTable(rows []MeasurementRow, config *MeasurementTableConfig) string {
	mtb := NewMeasurementTableBuilder(config)
	mtb.AddRows(rows)
	return mtb.Build()
}

// SummaryStats holds aggregate statistics for a measurement set.
type SummaryStats struct {
	MeasurementCount int
	AvgDEff          float64
	AvgBeta          float64
	AvgAlignment     float64
	AvgCPair         float64
	MinBeta          float64
	MaxBeta          float64
	BilateralCount   int
}

// ComputeSummaryStats calculates aggregate statistics from measurement rows.
func ComputeSummaryStats(rows []MeasurementRow) SummaryStats {
	if len(rows) == 0 {
		return SummaryStats{}
	}

	stats := SummaryStats{
		MeasurementCount: len(rows),
		MinBeta:          rows[0].Beta,
		MaxBeta:          rows[0].Beta,
	}

	var totalDEff, totalBeta, totalAlignment, totalCPair float64

	for _, row := range rows {
		totalDEff += float64(row.DEff)
		totalBeta += row.Beta
		totalAlignment += row.Alignment
		totalCPair += row.CPair

		if row.Beta < stats.MinBeta {
			stats.MinBeta = row.Beta
		}
		if row.Beta > stats.MaxBeta {
			stats.MaxBeta = row.Beta
		}

		// Count bilateral measurements (non-zero CPair indicates bilateral)
		if row.CPair > 0 {
			stats.BilateralCount++
		}
	}

	n := float64(len(rows))
	stats.AvgDEff = totalDEff / n
	stats.AvgBeta = totalBeta / n
	stats.AvgAlignment = totalAlignment / n
	stats.AvgCPair = totalCPair / n

	return stats
}

// SummaryTableConfig specifies options for summary statistics tables.
type SummaryTableConfig struct {
	TableConfig

	// Precision is the number of decimal places for floating-point values.
	Precision int

	// IncludeMinMax includes min/max values for Beta.
	IncludeMinMax bool

	// IncludeBilateralCount includes count of bilateral measurements.
	IncludeBilateralCount bool
}

// DefaultSummaryTableConfig returns a config with sensible defaults.
func DefaultSummaryTableConfig() *SummaryTableConfig {
	return &SummaryTableConfig{
		TableConfig: TableConfig{
			Style: StyleBooktabs,
		},
		Precision:             3,
		IncludeMinMax:         true,
		IncludeBilateralCount: true,
	}
}

// GenerateSummaryTable creates a LaTeX table showing aggregate statistics.
func GenerateSummaryTable(stats SummaryStats, config *SummaryTableConfig) string {
	if config == nil {
		config = DefaultSummaryTableConfig()
	}

	tableConfig := &TableConfig{
		Style:            config.Style,
		Caption:          config.Caption,
		Label:            config.Label,
		ColumnAlignments: []string{AlignLeft, AlignRight},
	}

	tb := NewTableBuilder(tableConfig)
	tb.SetHeaders("Metric", "Value")

	precision := config.Precision

	tb.AddRow("Measurements", fmt.Sprintf("%d", stats.MeasurementCount))
	// Use AddRowRaw for rows containing LaTeX math in metric labels
	tb.AddRowRaw("Avg. $D_{eff}$", FormatNumber(stats.AvgDEff, precision))
	tb.AddRowRaw("Avg. $\\beta$", FormatNumber(stats.AvgBeta, precision))
	tb.AddRow("Avg. Alignment", FormatNumber(stats.AvgAlignment, precision))
	tb.AddRowRaw("Avg. $C_{pair}$", FormatNumber(stats.AvgCPair, precision))

	if config.IncludeMinMax {
		tb.AddRowRaw("Min $\\beta$", FormatNumber(stats.MinBeta, precision))
		tb.AddRowRaw("Max $\\beta$", FormatNumber(stats.MaxBeta, precision))
	}

	if config.IncludeBilateralCount {
		tb.AddRow("Bilateral Count", fmt.Sprintf("%d", stats.BilateralCount))
	}

	return tb.Build()
}
