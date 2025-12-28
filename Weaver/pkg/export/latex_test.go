// Package export tests for LaTeX formatting and escaping.
package export

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// Escape Function Tests
// -----------------------------------------------------------------------------

func TestEscape_EmptyString(t *testing.T) {
	result := Escape("")
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

func TestEscape_NoSpecialChars(t *testing.T) {
	input := "Hello World"
	result := Escape(input)
	if result != input {
		t.Errorf("expected %q, got %q", input, result)
	}
}

func TestEscape_Underscore(t *testing.T) {
	input := "variable_name"
	result := Escape(input)
	expected := `variable\_name`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEscape_Ampersand(t *testing.T) {
	input := "A & B"
	result := Escape(input)
	expected := `A \& B`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEscape_Percent(t *testing.T) {
	input := "100%"
	result := Escape(input)
	expected := `100\%`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEscape_Dollar(t *testing.T) {
	input := "$100"
	result := Escape(input)
	expected := `\$100`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEscape_Hash(t *testing.T) {
	input := "#1"
	result := Escape(input)
	expected := `\#1`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEscape_Braces(t *testing.T) {
	input := "{data}"
	result := Escape(input)
	expected := `\{data\}`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEscape_Tilde(t *testing.T) {
	input := "~approx"
	result := Escape(input)
	expected := `\textasciitilde{}approx`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEscape_Caret(t *testing.T) {
	input := "x^2"
	result := Escape(input)
	expected := `x\textasciicircum{}2`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEscape_Backslash(t *testing.T) {
	input := `path\to\file`
	result := Escape(input)
	expected := `path\textbackslash{}to\textbackslash{}file`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEscape_AllSpecialChars(t *testing.T) {
	input := `\_&%$#{}~^\`
	result := Escape(input)

	// Verify all special chars are escaped
	specialChars := []struct {
		original string
		escaped  string
	}{
		{`\`, `\textbackslash{}`},
		{`_`, `\_`},
		{`&`, `\&`},
		{`%`, `\%`},
		{`$`, `\$`},
		{`#`, `\#`},
		{`{`, `\{`},
		{`}`, `\}`},
		{`~`, `\textasciitilde{}`},
		{`^`, `\textasciicircum{}`},
	}

	for _, sc := range specialChars {
		if !strings.Contains(result, sc.escaped) {
			t.Errorf("expected %q to be escaped to %q in result %q", sc.original, sc.escaped, result)
		}
	}
}

func TestEscape_MixedContent(t *testing.T) {
	input := "The value is $100 & includes 50% tax"
	result := Escape(input)

	if !strings.Contains(result, `\$100`) {
		t.Error("dollar sign not escaped")
	}
	if !strings.Contains(result, `\&`) {
		t.Error("ampersand not escaped")
	}
	if !strings.Contains(result, `\%`) {
		t.Error("percent not escaped")
	}
}

// -----------------------------------------------------------------------------
// LaTeXEscaper Tests
// -----------------------------------------------------------------------------

func TestDefaultLaTeXEscaper(t *testing.T) {
	e := DefaultLaTeXEscaper()

	if e == nil {
		t.Fatal("DefaultLaTeXEscaper() returned nil")
	}
	if e.PreserveNewlines {
		t.Error("expected PreserveNewlines to be false by default")
	}
	if e.PreserveMath {
		t.Error("expected PreserveMath to be false by default")
	}
}

func TestLaTeXEscaper_NewlineToSpace(t *testing.T) {
	e := &LaTeXEscaper{PreserveNewlines: false}
	input := "line1\nline2"
	result := e.Escape(input)
	expected := "line1 line2"

	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestLaTeXEscaper_PreserveNewlines(t *testing.T) {
	e := &LaTeXEscaper{PreserveNewlines: true}
	input := "line1\nline2"
	result := e.Escape(input)
	expected := `line1\\line2`

	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestLaTeXEscaper_PreserveMath(t *testing.T) {
	e := &LaTeXEscaper{PreserveMath: true}
	input := "The equation $x^2 + y^2 = z^2$ is famous"
	result := e.Escape(input)

	// Math content should NOT be escaped
	if !strings.Contains(result, "$x^2 + y^2 = z^2$") {
		t.Errorf("math mode should be preserved, got %q", result)
	}
}

func TestLaTeXEscaper_PreserveMathWithEscapedText(t *testing.T) {
	e := &LaTeXEscaper{PreserveMath: true}
	input := "For 100% confidence, use $\\alpha = 0.05$"
	result := e.Escape(input)

	// Text outside math should be escaped
	if !strings.Contains(result, `\%`) {
		t.Errorf("percent outside math should be escaped, got %q", result)
	}
	// Math content should NOT be escaped
	if !strings.Contains(result, `$\alpha = 0.05$`) {
		t.Errorf("math mode should be preserved, got %q", result)
	}
}

func TestLaTeXEscaper_EmptyString(t *testing.T) {
	e := &LaTeXEscaper{}
	result := e.Escape("")
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

// -----------------------------------------------------------------------------
// EscapeForCell Tests
// -----------------------------------------------------------------------------

func TestEscapeForCell_TrimsWhitespace(t *testing.T) {
	input := "  value  "
	result := EscapeForCell(input)
	if result != "value" {
		t.Errorf("expected trimmed value, got %q", result)
	}
}

func TestEscapeForCell_EmptyAfterTrim(t *testing.T) {
	input := "   "
	result := EscapeForCell(input)
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

func TestEscapeForCell_EscapesSpecialChars(t *testing.T) {
	input := " $100 "
	result := EscapeForCell(input)
	expected := `\$100`
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

// -----------------------------------------------------------------------------
// TableConfig Tests
// -----------------------------------------------------------------------------

func TestDefaultTableConfig(t *testing.T) {
	config := DefaultTableConfig()

	if config == nil {
		t.Fatal("DefaultTableConfig() returned nil")
	}
	if config.Style != StyleBooktabs {
		t.Errorf("expected Style %q, got %q", StyleBooktabs, config.Style)
	}
	if config.IncludeRowNumbers {
		t.Error("expected IncludeRowNumbers to be false by default")
	}
}

// -----------------------------------------------------------------------------
// TableBuilder Tests
// -----------------------------------------------------------------------------

func TestNewTableBuilder_NilConfig(t *testing.T) {
	tb := NewTableBuilder(nil)
	if tb == nil {
		t.Fatal("NewTableBuilder(nil) returned nil")
	}
	if tb.config == nil {
		t.Error("expected default config to be set")
	}
}

func TestTableBuilder_EmptyTable(t *testing.T) {
	tb := NewTableBuilder(nil)
	result := tb.Build()
	if result != "" {
		t.Errorf("expected empty string for empty table, got %q", result)
	}
}

func TestTableBuilder_SimpleTable(t *testing.T) {
	tb := NewTableBuilder(&TableConfig{Style: StylePlain})
	tb.SetHeaders("Name", "Value")
	tb.AddRow("item1", "100")
	tb.AddRow("item2", "200")

	result := tb.Build()

	// Check for table structure
	if !strings.Contains(result, "\\begin{tabular}") {
		t.Error("expected \\begin{tabular}")
	}
	if !strings.Contains(result, "\\end{tabular}") {
		t.Error("expected \\end{tabular}")
	}
	if !strings.Contains(result, "Name") {
		t.Error("expected header 'Name'")
	}
	if !strings.Contains(result, "item1 & 100") {
		t.Error("expected row data with & separator")
	}
	if !strings.Contains(result, "\\hline") {
		t.Error("expected \\hline for plain style")
	}
}

func TestTableBuilder_BooktabsStyle(t *testing.T) {
	tb := NewTableBuilder(&TableConfig{Style: StyleBooktabs})
	tb.SetHeaders("Column")
	tb.AddRow("data")

	result := tb.Build()

	if !strings.Contains(result, "\\toprule") {
		t.Error("expected \\toprule for booktabs style")
	}
	if !strings.Contains(result, "\\midrule") {
		t.Error("expected \\midrule for booktabs style")
	}
	if !strings.Contains(result, "\\bottomrule") {
		t.Error("expected \\bottomrule for booktabs style")
	}
}

func TestTableBuilder_WithCaption(t *testing.T) {
	tb := NewTableBuilder(&TableConfig{
		Style:   StyleBooktabs,
		Caption: "Test Results",
	})
	tb.SetHeaders("Test", "Result")
	tb.AddRow("test1", "pass")

	result := tb.Build()

	if !strings.Contains(result, "\\begin{table}") {
		t.Error("expected table environment with caption")
	}
	if !strings.Contains(result, "\\caption{Test Results}") {
		t.Error("expected caption")
	}
	if !strings.Contains(result, "\\end{table}") {
		t.Error("expected \\end{table}")
	}
}

func TestTableBuilder_WithLabel(t *testing.T) {
	tb := NewTableBuilder(&TableConfig{
		Style: StyleBooktabs,
		Label: "tab:results",
	})
	tb.SetHeaders("A")
	tb.AddRow("1")

	result := tb.Build()

	if !strings.Contains(result, "\\label{tab:results}") {
		t.Error("expected label")
	}
}

func TestTableBuilder_ColumnAlignments(t *testing.T) {
	tb := NewTableBuilder(&TableConfig{
		Style:            StyleBooktabs,
		ColumnAlignments: []string{AlignLeft, AlignCenter, AlignRight},
	})
	tb.SetHeaders("Left", "Center", "Right")
	tb.AddRow("a", "b", "c")

	result := tb.Build()

	if !strings.Contains(result, "{lcr}") {
		t.Errorf("expected column spec {lcr}, got %q", result)
	}
}

func TestTableBuilder_EscapesHeaders(t *testing.T) {
	tb := NewTableBuilder(nil)
	tb.SetHeaders("Price ($)", "Rate (%)")
	tb.AddRow("100", "5")

	result := tb.Build()

	if !strings.Contains(result, `\$`) {
		t.Error("expected dollar sign to be escaped in header")
	}
	if !strings.Contains(result, `\%`) {
		t.Error("expected percent sign to be escaped in header")
	}
}

func TestTableBuilder_EscapesRowData(t *testing.T) {
	tb := NewTableBuilder(nil)
	tb.SetHeaders("Formula")
	tb.AddRow("x^2 & y^2")

	result := tb.Build()

	if !strings.Contains(result, `\textasciicircum{}`) {
		t.Error("expected caret to be escaped in row data")
	}
	if !strings.Contains(result, `\&`) {
		t.Error("expected ampersand to be escaped in row data")
	}
}

func TestTableBuilder_NoHeadersWithRows(t *testing.T) {
	tb := NewTableBuilder(&TableConfig{Style: StylePlain})
	tb.AddRow("a", "b")
	tb.AddRow("c", "d")

	result := tb.Build()

	// Should still produce valid table without headers
	if !strings.Contains(result, "\\begin{tabular}") {
		t.Error("expected table without headers")
	}
	if !strings.Contains(result, "a & b") {
		t.Error("expected row data")
	}
}

func TestTableBuilder_CaptionEscaping(t *testing.T) {
	tb := NewTableBuilder(&TableConfig{
		Style:   StyleBooktabs,
		Caption: "Results for $100 & 50%",
	})
	tb.SetHeaders("Data")
	tb.AddRow("value")

	result := tb.Build()

	// Caption should have special chars escaped
	if !strings.Contains(result, `\$100`) {
		t.Error("expected dollar sign escaped in caption")
	}
	if !strings.Contains(result, `\&`) {
		t.Error("expected ampersand escaped in caption")
	}
}

// -----------------------------------------------------------------------------
// FormatNumber Tests
// -----------------------------------------------------------------------------

func TestFormatNumber_Integer(t *testing.T) {
	result := FormatNumber(42.0, 0)
	if result != "42" {
		t.Errorf("expected '42', got %q", result)
	}
}

func TestFormatNumber_OneDecimal(t *testing.T) {
	result := FormatNumber(3.14159, 1)
	if result != "3.1" {
		t.Errorf("expected '3.1', got %q", result)
	}
}

func TestFormatNumber_TwoDecimals(t *testing.T) {
	result := FormatNumber(3.14159, 2)
	if result != "3.14" {
		t.Errorf("expected '3.14', got %q", result)
	}
}

func TestFormatNumber_ThreeDecimals(t *testing.T) {
	result := FormatNumber(3.14159, 3)
	if result != "3.142" {
		t.Errorf("expected '3.142', got %q", result)
	}
}

func TestFormatNumber_NegativeValue(t *testing.T) {
	result := FormatNumber(-1.5, 1)
	if result != "-1.5" {
		t.Errorf("expected '-1.5', got %q", result)
	}
}

func TestFormatNumber_Zero(t *testing.T) {
	result := FormatNumber(0.0, 2)
	if result != "0.00" {
		t.Errorf("expected '0.00', got %q", result)
	}
}

// -----------------------------------------------------------------------------
// FormatPercent Tests
// -----------------------------------------------------------------------------

func TestFormatPercent_WholePercent(t *testing.T) {
	result := FormatPercent(0.5, 0)
	if result != `50\%` {
		t.Errorf("expected '50\\%%', got %q", result)
	}
}

func TestFormatPercent_DecimalPercent(t *testing.T) {
	result := FormatPercent(0.456, 1)
	if result != `45.6\%` {
		t.Errorf("expected '45.6\\%%', got %q", result)
	}
}

func TestFormatPercent_Zero(t *testing.T) {
	result := FormatPercent(0.0, 0)
	if result != `0\%` {
		t.Errorf("expected '0\\%%', got %q", result)
	}
}

func TestFormatPercent_Hundred(t *testing.T) {
	result := FormatPercent(1.0, 0)
	if result != `100\%` {
		t.Errorf("expected '100\\%%', got %q", result)
	}
}

// -----------------------------------------------------------------------------
// Constants Tests
// -----------------------------------------------------------------------------

func TestAlignmentConstants(t *testing.T) {
	if AlignLeft != "l" {
		t.Errorf("AlignLeft should be 'l', got %q", AlignLeft)
	}
	if AlignCenter != "c" {
		t.Errorf("AlignCenter should be 'c', got %q", AlignCenter)
	}
	if AlignRight != "r" {
		t.Errorf("AlignRight should be 'r', got %q", AlignRight)
	}
}

func TestStyleConstants(t *testing.T) {
	if StylePlain != "plain" {
		t.Errorf("StylePlain should be 'plain', got %q", StylePlain)
	}
	if StyleBooktabs != "booktabs" {
		t.Errorf("StyleBooktabs should be 'booktabs', got %q", StyleBooktabs)
	}
}

// -----------------------------------------------------------------------------
// Integration Tests
// -----------------------------------------------------------------------------

func TestIntegration_FullTable(t *testing.T) {
	config := &TableConfig{
		Style:            StyleBooktabs,
		Caption:          "Measurement Results",
		Label:            "tab:measurements",
		ColumnAlignments: []string{AlignLeft, AlignRight, AlignRight},
	}

	tb := NewTableBuilder(config)
	tb.SetHeaders("Agent", "D_eff", "Beta")
	tb.AddRow("Agent_A", "0.85", "1.23")
	tb.AddRow("Agent_B", "0.92", "1.45")

	result := tb.Build()

	// Verify complete table structure
	mustContain := []string{
		"\\begin{table}[htbp]",
		"\\centering",
		"\\begin{tabular}{lrr}",
		"\\toprule",
		"Agent & D\\_eff & Beta \\\\",
		"\\midrule",
		"Agent\\_A & 0.85 & 1.23 \\\\",
		"Agent\\_B & 0.92 & 1.45 \\\\",
		"\\bottomrule",
		"\\end{tabular}",
		"\\caption{Measurement Results}",
		"\\label{tab:measurements}",
		"\\end{table}",
	}

	for _, s := range mustContain {
		if !strings.Contains(result, s) {
			t.Errorf("expected output to contain %q, got:\n%s", s, result)
		}
	}
}

func TestIntegration_TableWithSpecialCharacters(t *testing.T) {
	tb := NewTableBuilder(&TableConfig{Style: StyleBooktabs})
	tb.SetHeaders("Symbol", "Meaning")
	tb.AddRow("$", "Dollar sign")
	tb.AddRow("&", "Ampersand")
	tb.AddRow("%", "Percent")
	tb.AddRow("#", "Hash")
	tb.AddRow("_", "Underscore")

	result := tb.Build()

	// All special characters should be escaped
	escapedSymbols := []string{`\$`, `\&`, `\%`, `\#`, `\_`}
	for _, es := range escapedSymbols {
		if !strings.Contains(result, es) {
			t.Errorf("expected %q to be in output:\n%s", es, result)
		}
	}
}
