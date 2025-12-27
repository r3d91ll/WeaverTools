package help

// style.go provides text styling functions using ANSI escape codes.
// These functions wrap text with color/formatting codes for consistent styling
// across all help output. Each function applies a specific semantic style.

// Styled text output helpers for help formatting.
// These functions wrap text with ANSI escape codes for consistent styling.

// Header returns text styled as a header (bold + cyan).
// Used for main section titles like "Commands" or "Shortcuts".
func Header(text string) string {
	return ColorBold + ColorCyan + text + ColorReset
}

// StyleCategory returns text styled as a category label (bold + green).
// Used for command category names like "Session Management" or "Analysis".
func StyleCategory(text string) string {
	return ColorBold + ColorGreen + text + ColorReset
}

// StyleCommand returns text styled as a command name (cyan).
// Used for command names like "/help" or "/extract".
func StyleCommand(text string) string {
	return ColorCyan + text + ColorReset
}

// StyleExample returns text styled as an example command (yellow).
// Used for example command syntax and arguments.
func StyleExample(text string) string {
	return ColorYellow + text + ColorReset
}

// Shortcut returns text styled as a keyboard shortcut (bold + yellow).
// Used for shortcut keys like "Ctrl+C" or aliases like "/q".
func Shortcut(text string) string {
	return ColorBold + ColorYellow + text + ColorReset
}

// Dim returns text in dim/muted style (gray).
// Used for secondary information, descriptions, and separators.
func Dim(text string) string {
	return ColorGray + text + ColorReset
}

// Bold returns text in bold style.
// Used for emphasis within normal text.
func Bold(text string) string {
	return ColorBold + text + ColorReset
}

// Argument returns text styled as a command argument (yellow).
// Used for command parameters and placeholders.
func Argument(text string) string {
	return ColorYellow + text + ColorReset
}

// Description returns text styled as a description (dim).
// Alias for Dim(), used for semantic clarity when styling descriptions.
func Description(text string) string {
	return Dim(text)
}

// Arrow returns a styled arrow separator for examples.
// Returns a dim arrow: " -> "
func Arrow() string {
	return Dim(" -> ")
}

// Bullet returns a styled bullet point.
// Returns a dim bullet: "  - "
func Bullet() string {
	return Dim("  - ")
}

// CommandWithShortcut formats a command with its shortcut alias.
// Example: CommandWithShortcut("/help", "/h") returns "/help (or /h)"
func CommandWithShortcut(cmd, shortcut string) string {
	if shortcut == "" {
		return StyleCommand(cmd)
	}
	return StyleCommand(cmd) + Dim(" (or ") + Shortcut(shortcut) + Dim(")")
}

// HighlightExampleCommand formats an example command with syntax highlighting.
// The command name (first token starting with /) is shown in cyan.
// Arguments (remaining tokens) are shown in yellow.
// Example: "/extract honor 20" -> cyan("/extract") + yellow(" honor 20")
func HighlightExampleCommand(cmd string) string {
	parts := splitFirstWord(cmd)
	if len(parts) == 0 {
		return ""
	}

	// First word is the command (cyan)
	result := StyleCommand(parts[0])

	// Remaining words are arguments (yellow)
	if len(parts) > 1 && parts[1] != "" {
		result += Argument(" " + parts[1])
	}

	return result
}

// splitFirstWord splits a string into [first_word, rest].
// Returns ["word", ""] if there's only one word.
// Returns ["", ""] if the string is empty.
func splitFirstWord(s string) []string {
	// Find first space
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' {
			return []string{s[:i], trimLeft(s[i+1:])}
		}
	}
	// No space found - entire string is the first word
	if s != "" {
		return []string{s, ""}
	}
	return []string{}
}

// trimLeft removes leading spaces from a string.
func trimLeft(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] != ' ' {
			return s[i:]
		}
	}
	return ""
}

// ExampleLine formats a complete example line with command and description.
// Uses syntax highlighting: command in cyan, arguments in yellow, arrow and description in gray.
// Example: ExampleLine("/extract file.go", "Extract concepts from file.go")
// Returns: "  /extract file.go  -> Extract concepts from file.go" (with colors)
func ExampleLine(cmd, desc string) string {
	return "  " + HighlightExampleCommand(cmd) + Arrow() + Dim(desc)
}
