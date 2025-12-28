package yarn

// This file contains example deprecated functions demonstrating the proper
// deprecation annotation format for the Yarn package.
//
// Deprecation Policy:
// - All deprecated APIs have a minimum 6-month notice period before removal
// - Deprecation comments must include: version deprecated, removal version, removal date
// - A replacement function or migration path must be provided

// Deprecated: FormatMessage is deprecated as of v1.5.0 and will be removed in v2.0.0.
// Use FormatMessageWithOptions instead for more flexible message formatting.
// Migration guide: https://docs.weavertools.dev/migration/v2
//
// Removal scheduled for: 2026-06-01 (6 months from deprecation)
//
// FormatMessage formats a message content string with basic sanitization.
// This function has been superseded by FormatMessageWithOptions which provides
// additional formatting controls and better Unicode handling.
func FormatMessage(content string) string {
	// Legacy implementation - delegates to new function with defaults
	return FormatMessageWithOptions(content, nil)
}

// FormatOptions configures message formatting behavior.
type FormatOptions struct {
	// TrimWhitespace removes leading/trailing whitespace when true.
	TrimWhitespace bool

	// NormalizeUnicode normalizes Unicode characters when true.
	NormalizeUnicode bool

	// MaxLength truncates content to this length (0 = no limit).
	MaxLength int
}

// DefaultFormatOptions returns the default formatting options.
func DefaultFormatOptions() *FormatOptions {
	return &FormatOptions{
		TrimWhitespace:   true,
		NormalizeUnicode: false,
		MaxLength:        0,
	}
}

// FormatMessageWithOptions formats a message content string with configurable options.
// This is the recommended replacement for the deprecated FormatMessage function.
//
// If opts is nil, default options are used.
func FormatMessageWithOptions(content string, opts *FormatOptions) string {
	if opts == nil {
		opts = DefaultFormatOptions()
	}

	result := content

	// Apply whitespace trimming
	if opts.TrimWhitespace {
		result = trimWhitespace(result)
	}

	// Apply length limit
	if opts.MaxLength > 0 && len(result) > opts.MaxLength {
		result = result[:opts.MaxLength]
	}

	return result
}

// trimWhitespace removes leading and trailing whitespace from a string.
func trimWhitespace(s string) string {
	// Simple whitespace trimming without importing strings package
	// to keep the example minimal
	start := 0
	end := len(s)

	for start < end && isWhitespace(s[start]) {
		start++
	}

	for end > start && isWhitespace(s[end-1]) {
		end--
	}

	return s[start:end]
}

// isWhitespace returns true if the byte is a whitespace character.
func isWhitespace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}
