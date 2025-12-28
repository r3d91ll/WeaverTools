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

// -----------------------------------------------------------------------------
// Measurement Table Generation Tests
// -----------------------------------------------------------------------------

func TestLatexTableGeneration_DefaultConfig(t *testing.T) {
	config := DefaultMeasurementTableConfig()

	if config == nil {
		t.Fatal("DefaultMeasurementTableConfig() returned nil")
	}
	if config.Style != StyleBooktabs {
		t.Errorf("expected Style %q, got %q", StyleBooktabs, config.Style)
	}
	if !config.IncludeTurn {
		t.Error("expected IncludeTurn to be true by default")
	}
	if !config.IncludeParticipants {
		t.Error("expected IncludeParticipants to be true by default")
	}
	if config.IncludeBetaStatus {
		t.Error("expected IncludeBetaStatus to be false by default")
	}
	if config.Precision != 3 {
		t.Errorf("expected Precision 3, got %d", config.Precision)
	}
	if config.AlignmentAsPercent {
		t.Error("expected AlignmentAsPercent to be false by default")
	}
}

func TestLatexTableGeneration_EmptyRows(t *testing.T) {
	mtb := NewMeasurementTableBuilder(nil)
	result := mtb.Build()

	if result != "" {
		t.Errorf("expected empty string for empty table, got %q", result)
	}
}

func TestLatexTableGeneration_SingleRow(t *testing.T) {
	mtb := NewMeasurementTableBuilder(nil)
	mtb.AddRow(MeasurementRow{
		Turn:      1,
		Sender:    "Agent_A",
		Receiver:  "Agent_B",
		DEff:      128,
		Beta:      1.75,
		Alignment: 0.85,
		CPair:     0.92,
	})

	result := mtb.Build()

	// Verify table structure
	if !strings.Contains(result, "\\begin{tabular}") {
		t.Error("expected \\begin{tabular}")
	}
	if !strings.Contains(result, "\\toprule") {
		t.Error("expected \\toprule for booktabs style")
	}
	if !strings.Contains(result, "\\midrule") {
		t.Error("expected \\midrule")
	}
	if !strings.Contains(result, "\\bottomrule") {
		t.Error("expected \\bottomrule")
	}

	// Verify headers (with escaping applied)
	if !strings.Contains(result, "Turn") {
		t.Error("expected Turn header")
	}
	if !strings.Contains(result, "Sender") {
		t.Error("expected Sender header")
	}
	if !strings.Contains(result, "Receiver") {
		t.Error("expected Receiver header")
	}

	// Verify data row contains expected values
	if !strings.Contains(result, "Agent\\_A") {
		t.Error("expected Agent_A with escaped underscore")
	}
	if !strings.Contains(result, "Agent\\_B") {
		t.Error("expected Agent_B with escaped underscore")
	}
	if !strings.Contains(result, "128") {
		t.Error("expected DEff value 128")
	}
	if !strings.Contains(result, "1.750") {
		t.Error("expected Beta value 1.750")
	}
	if !strings.Contains(result, "0.850") {
		t.Error("expected Alignment value 0.850")
	}
	if !strings.Contains(result, "0.920") {
		t.Error("expected CPair value 0.920")
	}
}

func TestLatexTableGeneration_MultipleRows(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85},
		{Turn: 2, Sender: "B", Receiver: "A", DEff: 110, Beta: 1.8, Alignment: 0.8, CPair: 0.80},
		{Turn: 3, Sender: "A", Receiver: "B", DEff: 105, Beta: 2.1, Alignment: 0.7, CPair: 0.75},
	}

	result := GenerateMeasurementTable(rows, nil)

	// Verify all three rows are present
	if !strings.Contains(result, "& 1 &") {
		t.Error("expected turn 1")
	}
	if !strings.Contains(result, "& 2 &") {
		t.Error("expected turn 2")
	}
	if !strings.Contains(result, "& 3 &") {
		t.Error("expected turn 3")
	}
}

func TestLatexTableGeneration_WithCaption(t *testing.T) {
	config := DefaultMeasurementTableConfig()
	config.Caption = "Conveyance Measurements"
	config.Label = "tab:measurements"

	mtb := NewMeasurementTableBuilder(config)
	mtb.AddRow(MeasurementRow{
		Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85,
	})

	result := mtb.Build()

	if !strings.Contains(result, "\\begin{table}") {
		t.Error("expected table environment with caption")
	}
	if !strings.Contains(result, "\\caption{Conveyance Measurements}") {
		t.Error("expected caption")
	}
	if !strings.Contains(result, "\\label{tab:measurements}") {
		t.Error("expected label")
	}
	if !strings.Contains(result, "\\end{table}") {
		t.Error("expected \\end{table}")
	}
}

func TestLatexTableGeneration_MinimalColumns(t *testing.T) {
	config := &MeasurementTableConfig{
		TableConfig: TableConfig{
			Style: StyleBooktabs,
		},
		IncludeTurn:         false,
		IncludeParticipants: false,
		IncludeBetaStatus:   false,
		Precision:           2,
	}

	mtb := NewMeasurementTableBuilder(config)
	mtb.AddRow(MeasurementRow{
		Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85,
	})

	result := mtb.Build()

	// Should NOT contain Turn or Participant columns
	if strings.Contains(result, "Turn &") {
		t.Error("Turn column should not be present")
	}
	if strings.Contains(result, "Sender") {
		t.Error("Sender column should not be present")
	}

	// Should contain core metrics
	if !strings.Contains(result, "1.50") {
		t.Error("expected Beta with 2 decimal precision")
	}
	if !strings.Contains(result, "0.90") {
		t.Error("expected Alignment with 2 decimal precision")
	}
}

func TestLatexTableGeneration_WithBetaStatus(t *testing.T) {
	config := DefaultMeasurementTableConfig()
	config.IncludeBetaStatus = true

	mtb := NewMeasurementTableBuilder(config)
	mtb.AddRow(MeasurementRow{
		Turn:       1,
		Sender:     "A",
		Receiver:   "B",
		DEff:       100,
		Beta:       1.75,
		Alignment:  0.9,
		CPair:      0.85,
		BetaStatus: "optimal",
	})

	result := mtb.Build()

	if !strings.Contains(result, "optimal") {
		t.Error("expected beta status 'optimal' in output")
	}
}

func TestLatexTableGeneration_AlignmentAsPercent(t *testing.T) {
	config := DefaultMeasurementTableConfig()
	config.AlignmentAsPercent = true
	config.Precision = 2

	mtb := NewMeasurementTableBuilder(config)
	mtb.AddRow(MeasurementRow{
		Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.856, CPair: 0.85,
	})

	result := mtb.Build()

	// Alignment should be formatted as percentage
	if !strings.Contains(result, `85.6\%`) {
		t.Errorf("expected alignment as percentage '85.6\\%%', got:\n%s", result)
	}
}

func TestLatexTableGeneration_ColumnAlignments(t *testing.T) {
	config := DefaultMeasurementTableConfig()
	config.IncludeTurn = true
	config.IncludeParticipants = true
	config.IncludeBetaStatus = false

	mtb := NewMeasurementTableBuilder(config)
	mtb.AddRow(MeasurementRow{
		Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85,
	})

	result := mtb.Build()

	// Expected alignment: Turn(c), Sender(l), Receiver(l), DEff(r), Beta(r), Alignment(r), CPair(r)
	if !strings.Contains(result, "{cllrrrr}") {
		t.Errorf("expected column spec {cllrrrr}, got:\n%s", result)
	}
}

func TestLatexTableGeneration_SpecialCharactersInNames(t *testing.T) {
	mtb := NewMeasurementTableBuilder(nil)
	mtb.AddRow(MeasurementRow{
		Turn:      1,
		Sender:    "Agent_1",
		Receiver:  "Agent & Co",
		DEff:      100,
		Beta:      1.5,
		Alignment: 0.9,
		CPair:     0.85,
	})

	result := mtb.Build()

	// Special characters should be escaped
	if !strings.Contains(result, "Agent\\_1") {
		t.Error("expected underscore to be escaped")
	}
	if !strings.Contains(result, `Agent \& Co`) {
		t.Error("expected ampersand to be escaped")
	}
}

// -----------------------------------------------------------------------------
// Summary Statistics Tests
// -----------------------------------------------------------------------------

func TestLatexTableGeneration_ComputeSummaryStats_Empty(t *testing.T) {
	rows := []MeasurementRow{}
	stats := ComputeSummaryStats(rows)

	if stats.MeasurementCount != 0 {
		t.Errorf("expected 0 measurements, got %d", stats.MeasurementCount)
	}
	if stats.AvgDEff != 0 {
		t.Errorf("expected 0 AvgDEff, got %f", stats.AvgDEff)
	}
}

func TestLatexTableGeneration_ComputeSummaryStats_SingleRow(t *testing.T) {
	rows := []MeasurementRow{
		{DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85},
	}
	stats := ComputeSummaryStats(rows)

	if stats.MeasurementCount != 1 {
		t.Errorf("expected 1 measurement, got %d", stats.MeasurementCount)
	}
	if stats.AvgDEff != 100 {
		t.Errorf("expected AvgDEff 100, got %f", stats.AvgDEff)
	}
	if stats.AvgBeta != 1.5 {
		t.Errorf("expected AvgBeta 1.5, got %f", stats.AvgBeta)
	}
	if stats.MinBeta != 1.5 {
		t.Errorf("expected MinBeta 1.5, got %f", stats.MinBeta)
	}
	if stats.MaxBeta != 1.5 {
		t.Errorf("expected MaxBeta 1.5, got %f", stats.MaxBeta)
	}
}

func TestLatexTableGeneration_ComputeSummaryStats_MultipleRows(t *testing.T) {
	rows := []MeasurementRow{
		{DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.8},
		{DEff: 120, Beta: 2.0, Alignment: 0.8, CPair: 0.7},
		{DEff: 110, Beta: 1.8, Alignment: 0.85, CPair: 0.0}, // CPair = 0 (unilateral)
	}
	stats := ComputeSummaryStats(rows)

	if stats.MeasurementCount != 3 {
		t.Errorf("expected 3 measurements, got %d", stats.MeasurementCount)
	}

	// Avg DEff = (100 + 120 + 110) / 3 = 110
	if stats.AvgDEff != 110 {
		t.Errorf("expected AvgDEff 110, got %f", stats.AvgDEff)
	}

	// Avg Beta = (1.5 + 2.0 + 1.8) / 3 = 1.766...
	expectedAvgBeta := (1.5 + 2.0 + 1.8) / 3
	if stats.AvgBeta != expectedAvgBeta {
		t.Errorf("expected AvgBeta %f, got %f", expectedAvgBeta, stats.AvgBeta)
	}

	if stats.MinBeta != 1.5 {
		t.Errorf("expected MinBeta 1.5, got %f", stats.MinBeta)
	}
	if stats.MaxBeta != 2.0 {
		t.Errorf("expected MaxBeta 2.0, got %f", stats.MaxBeta)
	}

	// 2 rows have non-zero CPair
	if stats.BilateralCount != 2 {
		t.Errorf("expected BilateralCount 2, got %d", stats.BilateralCount)
	}
}

func TestLatexTableGeneration_GenerateSummaryTable(t *testing.T) {
	stats := SummaryStats{
		MeasurementCount: 10,
		AvgDEff:          105.5,
		AvgBeta:          1.85,
		AvgAlignment:     0.82,
		AvgCPair:         0.75,
		MinBeta:          1.5,
		MaxBeta:          2.3,
		BilateralCount:   8,
	}

	result := GenerateSummaryTable(stats, nil)

	// Verify table structure
	if !strings.Contains(result, "\\begin{tabular}") {
		t.Error("expected \\begin{tabular}")
	}
	if !strings.Contains(result, "Metric") {
		t.Error("expected Metric header")
	}
	if !strings.Contains(result, "Value") {
		t.Error("expected Value header")
	}

	// Verify metrics are present
	if !strings.Contains(result, "Measurements") {
		t.Error("expected Measurements row")
	}
	if !strings.Contains(result, "10") {
		t.Error("expected measurement count 10")
	}

	// Verify avg values
	if !strings.Contains(result, "105.500") {
		t.Error("expected AvgDEff 105.500")
	}
	if !strings.Contains(result, "1.850") {
		t.Error("expected AvgBeta 1.850")
	}

	// Verify min/max are present (default config includes them)
	if !strings.Contains(result, "Min") {
		t.Error("expected Min row")
	}
	if !strings.Contains(result, "Max") {
		t.Error("expected Max row")
	}

	// Verify bilateral count
	if !strings.Contains(result, "Bilateral") {
		t.Error("expected Bilateral Count row")
	}
	if !strings.Contains(result, "8") {
		t.Error("expected bilateral count 8")
	}
}

func TestLatexTableGeneration_SummaryTableWithCaption(t *testing.T) {
	stats := SummaryStats{
		MeasurementCount: 5,
		AvgDEff:          100,
		AvgBeta:          1.7,
		AvgAlignment:     0.85,
		AvgCPair:         0.78,
		MinBeta:          1.5,
		MaxBeta:          2.0,
		BilateralCount:   4,
	}

	config := DefaultSummaryTableConfig()
	config.Caption = "Session Summary Statistics"
	config.Label = "tab:summary"

	result := GenerateSummaryTable(stats, config)

	if !strings.Contains(result, "\\caption{Session Summary Statistics}") {
		t.Error("expected caption")
	}
	if !strings.Contains(result, "\\label{tab:summary}") {
		t.Error("expected label")
	}
}

func TestLatexTableGeneration_SummaryTableWithoutMinMax(t *testing.T) {
	stats := SummaryStats{
		MeasurementCount: 5,
		AvgBeta:          1.7,
		MinBeta:          1.5,
		MaxBeta:          2.0,
	}

	config := DefaultSummaryTableConfig()
	config.IncludeMinMax = false

	result := GenerateSummaryTable(stats, config)

	// Min/Max should NOT be present
	if strings.Contains(result, "Min") {
		t.Error("Min row should not be present when IncludeMinMax is false")
	}
	if strings.Contains(result, "Max") {
		t.Error("Max row should not be present when IncludeMinMax is false")
	}
}

// -----------------------------------------------------------------------------
// Measurement Table Builder API Tests
// -----------------------------------------------------------------------------

func TestLatexTableGeneration_BuilderChaining(t *testing.T) {
	config := DefaultMeasurementTableConfig()

	result := NewMeasurementTableBuilder(config).
		AddRow(MeasurementRow{Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85}).
		AddRow(MeasurementRow{Turn: 2, Sender: "B", Receiver: "A", DEff: 110, Beta: 1.7, Alignment: 0.8, CPair: 0.80}).
		Build()

	if result == "" {
		t.Error("expected non-empty result from chained builder")
	}
	if !strings.Contains(result, "100") {
		t.Error("expected first row DEff value")
	}
	if !strings.Contains(result, "110") {
		t.Error("expected second row DEff value")
	}
}

func TestLatexTableGeneration_AddRowsSlice(t *testing.T) {
	rows := []MeasurementRow{
		{Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85},
		{Turn: 2, Sender: "B", Receiver: "A", DEff: 110, Beta: 1.7, Alignment: 0.8, CPair: 0.80},
	}

	mtb := NewMeasurementTableBuilder(nil)
	mtb.AddRows(rows)
	result := mtb.Build()

	if result == "" {
		t.Error("expected non-empty result from AddRows")
	}
}

func TestLatexTableGeneration_NilConfig(t *testing.T) {
	mtb := NewMeasurementTableBuilder(nil)

	if mtb == nil {
		t.Fatal("NewMeasurementTableBuilder(nil) returned nil")
	}

	// Should use default config
	mtb.AddRow(MeasurementRow{
		Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.9, CPair: 0.85,
	})

	result := mtb.Build()
	if result == "" {
		t.Error("expected non-empty result with nil config")
	}
}

// -----------------------------------------------------------------------------
// Full Integration Test for Measurement Tables
// -----------------------------------------------------------------------------

func TestLatexTableGeneration_FullMeasurementWorkflow(t *testing.T) {
	// Create measurement data as would come from a session
	rows := []MeasurementRow{
		{Turn: 1, Sender: "Claude", Receiver: "User", DEff: 128, Beta: 1.65, Alignment: 0.92, CPair: 0.88, BetaStatus: "optimal"},
		{Turn: 2, Sender: "User", Receiver: "Claude", DEff: 135, Beta: 1.72, Alignment: 0.89, CPair: 0.85, BetaStatus: "optimal"},
		{Turn: 3, Sender: "Claude", Receiver: "User", DEff: 142, Beta: 1.95, Alignment: 0.78, CPair: 0.82, BetaStatus: "monitor"},
		{Turn: 4, Sender: "User", Receiver: "Claude", DEff: 130, Beta: 2.15, Alignment: 0.75, CPair: 0.79, BetaStatus: "monitor"},
		{Turn: 5, Sender: "Claude", Receiver: "User", DEff: 125, Beta: 1.85, Alignment: 0.88, CPair: 0.86, BetaStatus: "optimal"},
	}

	// Generate detailed table with all columns
	config := &MeasurementTableConfig{
		TableConfig: TableConfig{
			Style:   StyleBooktabs,
			Caption: "Conveyance Measurements Over Session",
			Label:   "tab:measurements",
		},
		IncludeTurn:         true,
		IncludeParticipants: true,
		IncludeBetaStatus:   true,
		Precision:           2,
		AlignmentAsPercent:  false,
	}

	result := GenerateMeasurementTable(rows, config)

	// Verify complete table structure
	mustContain := []string{
		"\\begin{table}[htbp]",
		"\\centering",
		"\\toprule",
		"Turn",
		"Sender",
		"Receiver",
		"\\midrule",
		"Claude",
		"User",
		"optimal",
		"monitor",
		"\\bottomrule",
		"\\end{tabular}",
		"\\caption{Conveyance Measurements Over Session}",
		"\\label{tab:measurements}",
		"\\end{table}",
	}

	for _, s := range mustContain {
		if !strings.Contains(result, s) {
			t.Errorf("expected output to contain %q, got:\n%s", s, result)
		}
	}

	// Compute and generate summary table
	stats := ComputeSummaryStats(rows)

	if stats.MeasurementCount != 5 {
		t.Errorf("expected 5 measurements, got %d", stats.MeasurementCount)
	}
	if stats.BilateralCount != 5 {
		t.Errorf("expected 5 bilateral measurements, got %d", stats.BilateralCount)
	}

	summaryConfig := &SummaryTableConfig{
		TableConfig: TableConfig{
			Style:   StyleBooktabs,
			Caption: "Session Statistics",
			Label:   "tab:summary",
		},
		Precision:             2,
		IncludeMinMax:         true,
		IncludeBilateralCount: true,
	}

	summaryResult := GenerateSummaryTable(stats, summaryConfig)

	if summaryResult == "" {
		t.Error("expected non-empty summary table")
	}
	if !strings.Contains(summaryResult, "\\caption{Session Statistics}") {
		t.Error("expected summary table caption")
	}
}
