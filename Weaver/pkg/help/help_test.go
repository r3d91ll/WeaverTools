package help

import (
	"bytes"
	"strings"
	"testing"
)

// =============================================================================
// Box Drawing Tests
// =============================================================================

func TestBoxTop(t *testing.T) {
	box := NewBox(10)
	result := box.Top()

	// Should start with top-left corner and end with top-right corner
	if !strings.HasPrefix(result, BoxTopLeft) {
		t.Errorf("Top() should start with BoxTopLeft, got: %s", result)
	}
	if !strings.HasSuffix(result, BoxTopRight) {
		t.Errorf("Top() should end with BoxTopRight, got: %s", result)
	}

	// Should contain horizontal lines
	if !strings.Contains(result, BoxHorizontal) {
		t.Errorf("Top() should contain horizontal lines, got: %s", result)
	}

	// Visible length should be width + 2 (for corners)
	if visibleLength(result) != 12 {
		t.Errorf("Top() visible length should be 12, got: %d", visibleLength(result))
	}
}

func TestBoxMid(t *testing.T) {
	box := NewBox(10)
	result := box.Mid()

	// Should start with tee-left and end with tee-right
	if !strings.HasPrefix(result, BoxTeeLeft) {
		t.Errorf("Mid() should start with BoxTeeLeft, got: %s", result)
	}
	if !strings.HasSuffix(result, BoxTeeRight) {
		t.Errorf("Mid() should end with BoxTeeRight, got: %s", result)
	}

	// Visible length should be width + 2
	if visibleLength(result) != 12 {
		t.Errorf("Mid() visible length should be 12, got: %d", visibleLength(result))
	}
}

func TestBoxBottom(t *testing.T) {
	box := NewBox(10)
	result := box.Bottom()

	// Should start with bottom-left corner and end with bottom-right corner
	if !strings.HasPrefix(result, BoxBottomLeft) {
		t.Errorf("Bottom() should start with BoxBottomLeft, got: %s", result)
	}
	if !strings.HasSuffix(result, BoxBottomRight) {
		t.Errorf("Bottom() should end with BoxBottomRight, got: %s", result)
	}
}

func TestBoxRow(t *testing.T) {
	box := NewBox(10)

	// Test normal content
	result := box.Row("test")
	if !strings.HasPrefix(result, BoxVertical) {
		t.Errorf("Row() should start with BoxVertical, got: %s", result)
	}
	if !strings.HasSuffix(result, BoxVertical) {
		t.Errorf("Row() should end with BoxVertical, got: %s", result)
	}
	if !strings.Contains(result, "test") {
		t.Errorf("Row() should contain content, got: %s", result)
	}

	// Visible length should be width + 2 (for vertical borders)
	if visibleLength(result) != 12 {
		t.Errorf("Row() visible length should be 12, got: %d", visibleLength(result))
	}
}

func TestBoxRowCenter(t *testing.T) {
	box := NewBox(10)
	result := box.RowCenter("hi")

	// Content should be centered
	if !strings.Contains(result, "hi") {
		t.Errorf("RowCenter() should contain content, got: %s", result)
	}

	// Should have equal (or near-equal) padding on both sides
	if !strings.HasPrefix(result, BoxVertical) {
		t.Errorf("RowCenter() should start with BoxVertical, got: %s", result)
	}
	if !strings.HasSuffix(result, BoxVertical) {
		t.Errorf("RowCenter() should end with BoxVertical, got: %s", result)
	}

	// Visible length should be width + 2
	if visibleLength(result) != 12 {
		t.Errorf("RowCenter() visible length should be 12, got: %d", visibleLength(result))
	}
}

func TestBoxEmptyRow(t *testing.T) {
	box := NewBox(10)
	result := box.EmptyRow()

	// Should be bordered with only spaces inside
	if !strings.HasPrefix(result, BoxVertical) {
		t.Errorf("EmptyRow() should start with BoxVertical, got: %s", result)
	}
	if !strings.HasSuffix(result, BoxVertical) {
		t.Errorf("EmptyRow() should end with BoxVertical, got: %s", result)
	}

	// Should have spaces in the middle
	inner := result[len(BoxVertical) : len(result)-len(BoxVertical)]
	if strings.TrimSpace(inner) != "" {
		t.Errorf("EmptyRow() inner content should be spaces only, got: %q", inner)
	}
}

func TestBoxRowWithANSI(t *testing.T) {
	box := NewBox(20)
	content := ColorCyan + "colored" + ColorReset
	result := box.Row(content)

	// Should contain the ANSI codes
	if !strings.Contains(result, ColorCyan) {
		t.Errorf("Row() should preserve ANSI codes, got: %s", result)
	}

	// Visible length should still be correct (ignoring ANSI codes)
	// Width 20 + 2 borders = 22
	if visibleLength(result) != 22 {
		t.Errorf("Row() visible length with ANSI should be 22, got: %d", visibleLength(result))
	}
}

func TestVisibleLength(t *testing.T) {
	tests := []struct {
		input    string
		expected int
	}{
		{"hello", 5},
		{"", 0},
		{ColorCyan + "hello" + ColorReset, 5},
		{ColorBold + ColorCyan + "test" + ColorReset, 4},
		{"\033[32mgreen\033[0m", 5},
		{"no codes here", 13},
	}

	for _, test := range tests {
		result := visibleLength(test.input)
		if result != test.expected {
			t.Errorf("visibleLength(%q) = %d, want %d", test.input, result, test.expected)
		}
	}
}

func TestTruncateVisible(t *testing.T) {
	// Plain text truncation
	result := truncateVisible("hello world", 5)
	if result != "hello" {
		t.Errorf("truncateVisible should truncate to 5 chars, got: %s", result)
	}

	// ANSI-colored text truncation
	colored := ColorCyan + "hello world" + ColorReset
	result = truncateVisible(colored, 5)
	if visibleLength(result) != 5 {
		t.Errorf("truncateVisible visible length should be 5, got: %d", visibleLength(result))
	}

	// Should still contain ANSI codes (or reset)
	if !strings.Contains(result, "\033[") {
		t.Errorf("truncateVisible should preserve or add ANSI codes, got: %s", result)
	}
}

func TestPadRight(t *testing.T) {
	result := PadRight("hi", 5)
	if result != "hi   " {
		t.Errorf("PadRight('hi', 5) should be 'hi   ', got: %q", result)
	}

	// With ANSI codes
	colored := ColorCyan + "hi" + ColorReset
	result = PadRight(colored, 5)
	if visibleLength(result) != 5 {
		t.Errorf("PadRight visible length should be 5, got: %d", visibleLength(result))
	}
}

func TestPadLeft(t *testing.T) {
	result := PadLeft("hi", 5)
	if result != "   hi" {
		t.Errorf("PadLeft('hi', 5) should be '   hi', got: %q", result)
	}
}

// =============================================================================
// Style Function Tests
// =============================================================================

func TestHeaderStyle(t *testing.T) {
	result := Header("Test Header")

	if !strings.Contains(result, ColorBold) {
		t.Error("Header() should contain ColorBold")
	}
	if !strings.Contains(result, ColorCyan) {
		t.Error("Header() should contain ColorCyan")
	}
	if !strings.Contains(result, ColorReset) {
		t.Error("Header() should contain ColorReset")
	}
	if !strings.Contains(result, "Test Header") {
		t.Error("Header() should contain the text content")
	}
}

func TestCategoryStyle(t *testing.T) {
	result := StyleCategory("Session Management")

	if !strings.Contains(result, ColorBold) {
		t.Error("StyleCategory() should contain ColorBold")
	}
	if !strings.Contains(result, ColorGreen) {
		t.Error("StyleCategory() should contain ColorGreen")
	}
	if !strings.Contains(result, ColorReset) {
		t.Error("StyleCategory() should contain ColorReset")
	}
}

func TestCommandStyle(t *testing.T) {
	result := StyleCommand("/help")

	if !strings.Contains(result, ColorCyan) {
		t.Error("StyleCommand() should contain ColorCyan")
	}
	if !strings.Contains(result, ColorReset) {
		t.Error("StyleCommand() should contain ColorReset")
	}
	if !strings.Contains(result, "/help") {
		t.Error("StyleCommand() should contain the command text")
	}
}

func TestExampleStyle(t *testing.T) {
	result := StyleExample("/extract honor 20")

	if !strings.Contains(result, ColorYellow) {
		t.Error("StyleExample() should contain ColorYellow")
	}
	if !strings.Contains(result, ColorReset) {
		t.Error("StyleExample() should contain ColorReset")
	}
}

func TestShortcutStyle(t *testing.T) {
	result := Shortcut("Ctrl+C")

	if !strings.Contains(result, ColorBold) {
		t.Error("Shortcut() should contain ColorBold")
	}
	if !strings.Contains(result, ColorYellow) {
		t.Error("Shortcut() should contain ColorYellow")
	}
	if !strings.Contains(result, ColorReset) {
		t.Error("Shortcut() should contain ColorReset")
	}
}

func TestDimStyle(t *testing.T) {
	result := Dim("muted text")

	if !strings.Contains(result, ColorGray) {
		t.Error("Dim() should contain ColorGray")
	}
	if !strings.Contains(result, ColorReset) {
		t.Error("Dim() should contain ColorReset")
	}
}

func TestBoldStyle(t *testing.T) {
	result := Bold("important")

	if !strings.Contains(result, ColorBold) {
		t.Error("Bold() should contain ColorBold")
	}
	if !strings.Contains(result, ColorReset) {
		t.Error("Bold() should contain ColorReset")
	}
}

func TestArgumentStyle(t *testing.T) {
	result := Argument("<file>")

	if !strings.Contains(result, ColorYellow) {
		t.Error("Argument() should contain ColorYellow")
	}
	if !strings.Contains(result, ColorReset) {
		t.Error("Argument() should contain ColorReset")
	}
}

func TestHighlightExampleCommand(t *testing.T) {
	result := HighlightExampleCommand("/extract honor 20")

	// Should have command styled in cyan
	if !strings.Contains(result, ColorCyan) {
		t.Error("HighlightExampleCommand() should style command with cyan")
	}

	// Should have arguments styled in yellow
	if !strings.Contains(result, ColorYellow) {
		t.Error("HighlightExampleCommand() should style arguments with yellow")
	}

	// Should contain both parts
	if !strings.Contains(result, "/extract") {
		t.Error("HighlightExampleCommand() should contain the command")
	}
	if !strings.Contains(result, "honor 20") {
		t.Error("HighlightExampleCommand() should contain the arguments")
	}
}

func TestCommandWithShortcut(t *testing.T) {
	// With shortcut
	result := CommandWithShortcut("/help", "/h")
	if !strings.Contains(result, "/help") {
		t.Error("CommandWithShortcut() should contain main command")
	}
	if !strings.Contains(result, "/h") {
		t.Error("CommandWithShortcut() should contain shortcut")
	}
	if !strings.Contains(result, "or") {
		t.Error("CommandWithShortcut() should contain 'or'")
	}

	// Without shortcut
	result = CommandWithShortcut("/agents", "")
	if !strings.Contains(result, "/agents") {
		t.Error("CommandWithShortcut() without shortcut should contain command")
	}
	if strings.Contains(result, "or") {
		t.Error("CommandWithShortcut() without shortcut should not contain 'or'")
	}
}

func TestExampleLine(t *testing.T) {
	result := ExampleLine("/extract main.go", "Extract from main.go")

	// Should contain the command
	if !strings.Contains(result, "/extract") {
		t.Error("ExampleLine() should contain the command")
	}
	// Should contain the description
	if !strings.Contains(result, "Extract from main.go") {
		t.Error("ExampleLine() should contain the description")
	}
	// Should contain arrow separator
	if !strings.Contains(result, "->") {
		t.Error("ExampleLine() should contain arrow separator")
	}
}

// =============================================================================
// Command Registry Completeness Tests
// =============================================================================

// AllShellCommands lists all primary commands in the help registry.
// Note: Aliases like /exit are handled in shell.go but not separately registered
// in help since they just redirect to the primary command.
var AllShellCommands = []string{
	"/quit",   // primary, with /q shortcut
	"/help",   // primary, with /h shortcut
	"/agents",
	"/session",
	"/history",
	"/clear",
	"/default",
	"/extract",
	"/analyze",
	"/compare",
	"/validate",
	"/concepts",
	"/metrics",
	"/clear_concepts",
}

func TestCommandRegistryCompleteness(t *testing.T) {
	// Test that all shell commands have entries in the registry
	for _, cmdName := range AllShellCommands {
		cmd, found := GetCommand(cmdName)
		if !found {
			t.Errorf("Command %s is not in the help registry", cmdName)
			continue
		}

		// Check that each command has required fields
		if cmd.Name == "" {
			t.Errorf("Command %s has empty Name", cmdName)
		}
		if cmd.Description == "" {
			t.Errorf("Command %s has empty Description", cmdName)
		}
		if cmd.Category == "" {
			t.Errorf("Command %s has empty Category", cmdName)
		}
		if cmd.Usage == "" {
			t.Errorf("Command %s has empty Usage", cmdName)
		}
	}
}

func TestCommandRegistryCategories(t *testing.T) {
	// Test that all categories have at least one command
	for _, cat := range CategoryOrder {
		commands := GetCommandsByCategory(cat)
		if len(commands) == 0 {
			t.Errorf("Category %s has no commands", cat)
		}
	}
}

func TestGetCommand(t *testing.T) {
	// Test with leading slash
	cmd, found := GetCommand("/help")
	if !found {
		t.Error("GetCommand('/help') should find the command")
	}
	if cmd.Name != "/help" {
		t.Errorf("GetCommand('/help') Name should be '/help', got %s", cmd.Name)
	}

	// Test without leading slash
	cmd, found = GetCommand("help")
	if !found {
		t.Error("GetCommand('help') should find the command")
	}
	if cmd.Name != "/help" {
		t.Errorf("GetCommand('help') Name should be '/help', got %s", cmd.Name)
	}

	// Test shortcut lookup
	cmd, found = GetCommand("/h")
	if !found {
		t.Error("GetCommand('/h') should find the command via shortcut")
	}
	if cmd.Name != "/help" {
		t.Errorf("GetCommand('/h') should return /help, got %s", cmd.Name)
	}

	// Test non-existent command
	_, found = GetCommand("/nonexistent")
	if found {
		t.Error("GetCommand('/nonexistent') should not find a command")
	}
}

func TestGetCommandsByCategory(t *testing.T) {
	// Session category should have specific commands
	sessionCmds := GetCommandsByCategory(CategorySession)
	sessionNames := make(map[string]bool)
	for _, cmd := range sessionCmds {
		sessionNames[cmd.Name] = true
	}

	expectedSession := []string{"/agents", "/session", "/history", "/clear", "/default"}
	for _, expected := range expectedSession {
		if !sessionNames[expected] {
			t.Errorf("CategorySession should contain %s", expected)
		}
	}

	// Analysis category should have specific commands
	analysisCmds := GetCommandsByCategory(CategoryAnalysis)
	analysisNames := make(map[string]bool)
	for _, cmd := range analysisCmds {
		analysisNames[cmd.Name] = true
	}

	expectedAnalysis := []string{"/extract", "/analyze", "/compare", "/validate", "/metrics", "/concepts", "/clear_concepts"}
	for _, expected := range expectedAnalysis {
		if !analysisNames[expected] {
			t.Errorf("CategoryAnalysis should contain %s", expected)
		}
	}

	// General category should have specific commands
	generalCmds := GetCommandsByCategory(CategoryGeneral)
	generalNames := make(map[string]bool)
	for _, cmd := range generalCmds {
		generalNames[cmd.Name] = true
	}

	expectedGeneral := []string{"/help", "/quit"}
	for _, expected := range expectedGeneral {
		if !generalNames[expected] {
			t.Errorf("CategoryGeneral should contain %s", expected)
		}
	}
}

func TestCategoryMetadata(t *testing.T) {
	// Test that all categories have display metadata
	for _, cat := range CategoryOrder {
		info, ok := Categories[cat]
		if !ok {
			t.Errorf("Category %s has no metadata in Categories map", cat)
			continue
		}

		if info.DisplayName == "" {
			t.Errorf("Category %s has empty DisplayName", cat)
		}
		if info.Icon == "" {
			t.Errorf("Category %s has empty Icon", cat)
		}

		// Test helper methods
		if cat.DisplayName() != info.DisplayName {
			t.Errorf("Category.DisplayName() mismatch for %s", cat)
		}
		if cat.Icon() != info.Icon {
			t.Errorf("Category.Icon() mismatch for %s", cat)
		}
	}
}

// =============================================================================
// Renderer Output Structure Tests
// =============================================================================

func TestRenderFullStructure(t *testing.T) {
	var buf bytes.Buffer
	r := NewRenderer(&buf)
	r.RenderFull()
	output := buf.String()

	// Should contain the header
	if !strings.Contains(output, "Weaver Commands") {
		t.Error("RenderFull() should contain 'Weaver Commands' header")
	}

	// Should contain all category names
	for _, cat := range CategoryOrder {
		displayName := cat.DisplayName()
		if !strings.Contains(output, displayName) {
			t.Errorf("RenderFull() should contain category '%s'", displayName)
		}
	}

	// Should contain Shortcuts section
	if !strings.Contains(output, "Shortcuts") {
		t.Error("RenderFull() should contain 'Shortcuts' section")
	}

	// Should contain box drawing characters
	if !strings.Contains(output, BoxVertical) {
		t.Error("RenderFull() should contain box drawing characters")
	}
	if !strings.Contains(output, BoxTeeLeft) {
		t.Error("RenderFull() should contain box drawing separator")
	}

	// Should contain some commands
	if !strings.Contains(output, "/help") {
		t.Error("RenderFull() should contain /help command")
	}
	if !strings.Contains(output, "/extract") {
		t.Error("RenderFull() should contain /extract command")
	}
}

func TestRenderFullContainsAllCommands(t *testing.T) {
	var buf bytes.Buffer
	r := NewRenderer(&buf)
	r.RenderFull()
	output := buf.String()

	// Check that all primary commands appear in output
	primaryCommands := []string{
		"/agents", "/session", "/history", "/clear", "/default",
		"/extract", "/analyze", "/compare", "/validate",
		"/metrics", "/concepts", "/clear_concepts",
		"/help", "/quit",
	}

	for _, cmd := range primaryCommands {
		if !strings.Contains(output, cmd) {
			t.Errorf("RenderFull() should contain command %s", cmd)
		}
	}
}

func TestRenderFullContainsExamples(t *testing.T) {
	var buf bytes.Buffer
	r := NewRenderer(&buf)
	r.RenderFull()
	output := buf.String()

	// Should contain example markers
	if !strings.Contains(output, "e.g.") {
		t.Error("RenderFull() should contain inline examples with 'e.g.' marker")
	}
}

func TestRenderCommand(t *testing.T) {
	var buf bytes.Buffer
	r := NewRenderer(&buf)
	found := r.RenderCommand("extract")
	output := buf.String()

	if !found {
		t.Error("RenderCommand('extract') should return true")
	}

	// Should contain command name
	if !strings.Contains(output, "/extract") {
		t.Error("RenderCommand should contain command name")
	}

	// Should contain description
	if !strings.Contains(output, "Extract") {
		t.Error("RenderCommand should contain description")
	}

	// Should contain usage
	if !strings.Contains(output, "Usage:") {
		t.Error("RenderCommand should contain 'Usage:' section")
	}

	// Should contain examples section for commands with examples
	if !strings.Contains(output, "Examples:") {
		t.Error("RenderCommand should contain 'Examples:' section for /extract")
	}
}

func TestRenderCommandNotFound(t *testing.T) {
	var buf bytes.Buffer
	r := NewRenderer(&buf)
	found := r.RenderCommand("nonexistent")
	output := buf.String()

	if found {
		t.Error("RenderCommand('nonexistent') should return false")
	}

	// Should contain error message
	if !strings.Contains(output, "not found") {
		t.Error("RenderCommand for unknown command should contain 'not found'")
	}
}

func TestRenderShortcuts(t *testing.T) {
	var buf bytes.Buffer
	r := NewRenderer(&buf)
	r.RenderShortcuts()
	output := buf.String()

	// Should contain shortcuts section title
	if !strings.Contains(output, "Shortcuts") {
		t.Error("RenderShortcuts should contain 'Shortcuts' title")
	}

	// Should contain aliases
	if !strings.Contains(output, "/h") {
		t.Error("RenderShortcuts should contain /h alias")
	}
	if !strings.Contains(output, "/q") {
		t.Error("RenderShortcuts should contain /q alias")
	}

	// Should contain keyboard shortcuts
	if !strings.Contains(output, "Ctrl+C") {
		t.Error("RenderShortcuts should contain Ctrl+C")
	}
	if !strings.Contains(output, "Ctrl+D") {
		t.Error("RenderShortcuts should contain Ctrl+D")
	}

	// Should contain @agent syntax
	if !strings.Contains(output, "@agent") {
		t.Error("RenderShortcuts should contain @agent messaging syntax")
	}
}

func TestNewRenderer(t *testing.T) {
	var buf bytes.Buffer
	r := NewRenderer(&buf)

	if r == nil {
		t.Error("NewRenderer should return non-nil Renderer")
	}

	// Verify it writes to the provided writer
	r.RenderFull()
	if buf.Len() == 0 {
		t.Error("Renderer should write to the provided io.Writer")
	}
}

// =============================================================================
// Helper Function Tests
// =============================================================================

func TestSplitFirstWord(t *testing.T) {
	tests := []struct {
		input string
		first string
		rest  string
	}{
		{"/extract honor 20", "/extract", "honor 20"},
		{"/help", "/help", ""},
		{"", "", ""},
		{"single", "single", ""},
		{"two words", "two", "words"},
		{"  leading", "", "leading"},
		{"trailing  spaces", "trailing", "spaces"},
	}

	for _, test := range tests {
		result := splitFirstWord(test.input)
		if len(result) == 0 {
			if test.first != "" || test.rest != "" {
				t.Errorf("splitFirstWord(%q) returned empty, expected [%q, %q]",
					test.input, test.first, test.rest)
			}
			continue
		}

		if result[0] != test.first {
			t.Errorf("splitFirstWord(%q)[0] = %q, want %q",
				test.input, result[0], test.first)
		}
		if len(result) > 1 && result[1] != test.rest {
			t.Errorf("splitFirstWord(%q)[1] = %q, want %q",
				test.input, result[1], test.rest)
		}
	}
}

func TestTrimLeft(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"  hello", "hello"},
		{"hello", "hello"},
		{"   ", ""},
		{"", ""},
		{" a b c", "a b c"},
	}

	for _, test := range tests {
		result := trimLeft(test.input)
		if result != test.expected {
			t.Errorf("trimLeft(%q) = %q, want %q",
				test.input, result, test.expected)
		}
	}
}

func TestHorizontalLine(t *testing.T) {
	result := HorizontalLine(5)
	expected := strings.Repeat(BoxHorizontal, 5)
	if result != expected {
		t.Errorf("HorizontalLine(5) = %q, want %q", result, expected)
	}
}

func TestPadding(t *testing.T) {
	result := Padding(5)
	if result != "     " {
		t.Errorf("Padding(5) = %q, want '     '", result)
	}
}

func TestDisplayWidth(t *testing.T) {
	// DisplayWidth is an alias for visibleLength
	result := DisplayWidth("hello")
	if result != 5 {
		t.Errorf("DisplayWidth('hello') = %d, want 5", result)
	}

	// With ANSI codes
	result = DisplayWidth(ColorCyan + "hello" + ColorReset)
	if result != 5 {
		t.Errorf("DisplayWidth with ANSI = %d, want 5", result)
	}
}

func TestStringWidth(t *testing.T) {
	// StringWidth is an alias for visibleLength
	result := StringWidth("test")
	if result != 4 {
		t.Errorf("StringWidth('test') = %d, want 4", result)
	}
}
