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
			numberedRow := append([]string{Escape(string(rune('0' + i + 1)))}, row...)
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
