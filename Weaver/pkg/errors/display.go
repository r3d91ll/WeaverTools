// Package errors provides error formatting and display functions.
// Renders WeaverErrors with color coding for TTY output.
package errors

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// ANSI color codes for terminal output.
const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"    // Error type/code
	colorYellow = "\033[33m"    // Context information
	colorCyan   = "\033[36m"    // Suggestions
	colorDim    = "\033[90m"    // Secondary/cause info
	colorBold   = "\033[1m"     // Emphasis
	colorGreen  = "\033[32m"    // Success hints
)

// Formatter handles error display with optional color support.
type Formatter struct {
	// UseColor enables ANSI color codes in output.
	// When false, output is plain text suitable for logs.
	UseColor bool

	// Writer is the output destination. Defaults to os.Stderr.
	Writer io.Writer

	// Indent is the prefix for context and suggestion lines.
	Indent string
}

// DefaultFormatter returns a Formatter configured for standard error output.
// Color is enabled if stderr is a TTY.
func DefaultFormatter() *Formatter {
	return &Formatter{
		UseColor: IsTTY(os.Stderr),
		Writer:   os.Stderr,
		Indent:   "  ",
	}
}

// IsTTY returns true if the given file is a terminal.
func IsTTY(f *os.File) bool {
	if f == nil {
		return false
	}
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

// Format renders a WeaverError with color coding and structured display.
// Returns a formatted string suitable for display to users.
func Format(err error) string {
	return DefaultFormatter().Format(err)
}

// Format renders an error with color coding based on formatter settings.
// For WeaverError, displays code, message, context, cause, and suggestions.
// For standard errors, displays a simple error message.
func (f *Formatter) Format(err error) string {
	if err == nil {
		return ""
	}

	we, ok := AsWeaverError(err)
	if !ok {
		// Standard error: just display with error prefix
		return f.formatStandardError(err)
	}

	return f.formatWeaverError(we)
}

// formatStandardError formats a non-WeaverError error.
func (f *Formatter) formatStandardError(err error) string {
	var sb strings.Builder

	if f.UseColor {
		sb.WriteString(colorRed)
		sb.WriteString("Error: ")
		sb.WriteString(colorReset)
	} else {
		sb.WriteString("Error: ")
	}
	sb.WriteString(err.Error())

	return sb.String()
}

// formatWeaverError formats a WeaverError with full context and suggestions.
func (f *Formatter) formatWeaverError(we *WeaverError) string {
	var sb strings.Builder

	// Error header: ERROR [CODE]: Message
	f.writeErrorHeader(&sb, we)

	// Context (key=value pairs)
	if we.HasContext() {
		f.writeContext(&sb, we)
	}

	// Cause (wrapped error)
	if we.Cause != nil {
		f.writeCause(&sb, we)
	}

	// Suggestions
	if we.HasSuggestions() {
		f.writeSuggestions(&sb, we)
	}

	return sb.String()
}

// writeErrorHeader writes the error type and message.
func (f *Formatter) writeErrorHeader(sb *strings.Builder, we *WeaverError) {
	if f.UseColor {
		sb.WriteString(colorRed)
		sb.WriteString(colorBold)
		sb.WriteString("ERROR")
		sb.WriteString(colorReset)
		sb.WriteString(colorRed)
		sb.WriteString(" [")
		sb.WriteString(we.Code)
		sb.WriteString("]: ")
		sb.WriteString(colorReset)
	} else {
		sb.WriteString("ERROR [")
		sb.WriteString(we.Code)
		sb.WriteString("]: ")
	}
	sb.WriteString(we.Message)
	sb.WriteString("\n")
}

// writeContext writes the context key-value pairs.
func (f *Formatter) writeContext(sb *strings.Builder, we *WeaverError) {
	// Sort keys for consistent output
	keys := make([]string, 0, len(we.Context))
	for k := range we.Context {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		value := we.Context[key]
		sb.WriteString(f.Indent)
		if f.UseColor {
			sb.WriteString(colorYellow)
			sb.WriteString(key)
			sb.WriteString(": ")
			sb.WriteString(colorReset)
		} else {
			sb.WriteString(key)
			sb.WriteString(": ")
		}
		sb.WriteString(value)
		sb.WriteString("\n")
	}
}

// writeCause writes the underlying cause of the error.
func (f *Formatter) writeCause(sb *strings.Builder, we *WeaverError) {
	sb.WriteString(f.Indent)
	if f.UseColor {
		sb.WriteString(colorDim)
		sb.WriteString("cause: ")
		sb.WriteString(we.Cause.Error())
		sb.WriteString(colorReset)
	} else {
		sb.WriteString("cause: ")
		sb.WriteString(we.Cause.Error())
	}
	sb.WriteString("\n")
}

// writeSuggestions writes actionable remediation suggestions.
func (f *Formatter) writeSuggestions(sb *strings.Builder, we *WeaverError) {
	// Add a blank line before suggestions for visual separation
	if we.HasContext() || we.Cause != nil {
		sb.WriteString("\n")
	}

	for i, suggestion := range we.Suggestions {
		sb.WriteString(f.Indent)
		if f.UseColor {
			sb.WriteString(colorCyan)
			sb.WriteString("→ ")
			sb.WriteString(suggestion)
			sb.WriteString(colorReset)
		} else {
			sb.WriteString("→ ")
			sb.WriteString(suggestion)
		}
		if i < len(we.Suggestions)-1 {
			sb.WriteString("\n")
		}
	}
}

// Display writes a formatted error to the formatter's writer.
// This is a convenience method that combines Format and Write.
func (f *Formatter) Display(err error) {
	if err == nil {
		return
	}
	formatted := f.Format(err)
	fmt.Fprintln(f.Writer, formatted)
}

// Display writes a formatted error to stderr with default settings.
// This is the primary function for displaying errors to users.
func Display(err error) {
	DefaultFormatter().Display(err)
}

// Sprint returns a formatted error string without colors.
// Useful for logging or non-TTY environments.
func Sprint(err error) string {
	f := &Formatter{
		UseColor: false,
		Writer:   io.Discard,
		Indent:   "  ",
	}
	return f.Format(err)
}

// Sprintc returns a formatted error string with colors.
// Forces color output regardless of terminal detection.
func Sprintc(err error) string {
	f := &Formatter{
		UseColor: true,
		Writer:   io.Discard,
		Indent:   "  ",
	}
	return f.Format(err)
}

// FormatMultiple formats multiple errors for display.
// Useful when multiple errors need to be shown together.
func FormatMultiple(errs []error) string {
	if len(errs) == 0 {
		return ""
	}

	f := DefaultFormatter()
	var sb strings.Builder

	for i, err := range errs {
		if err == nil {
			continue
		}
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(f.Format(err))
	}

	return sb.String()
}

// CategoryLabel returns a human-readable label for an error category.
func CategoryLabel(cat Category) string {
	switch cat {
	case CategoryConfig:
		return "Configuration Error"
	case CategoryAgent:
		return "Agent Error"
	case CategoryBackend:
		return "Backend Error"
	case CategoryCommand:
		return "Command Error"
	case CategoryValidation:
		return "Validation Error"
	case CategoryNetwork:
		return "Network Error"
	case CategoryIO:
		return "I/O Error"
	case CategoryInternal:
		return "Internal Error"
	default:
		return "Error"
	}
}
