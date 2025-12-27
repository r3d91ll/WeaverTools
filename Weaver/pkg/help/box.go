package help

// box.go provides utilities for drawing bordered boxes and ANSI-aware string handling.
// These utilities are used by the Renderer to create visually structured output.

import "strings"

// Box provides methods for rendering box drawing structures with a fixed width.
// Uses rounded corners (╭╮╰╯) for a modern aesthetic.
type Box struct {
	// Width is the inner content width (excluding border characters)
	Width int
}

// NewBox creates a new Box with the specified inner content width.
func NewBox(width int) *Box {
	return &Box{Width: width}
}

// Top returns the top border of the box: ╭───────────╮
func (b *Box) Top() string {
	return BoxTopLeft + strings.Repeat(BoxHorizontal, b.Width) + BoxTopRight
}

// Mid returns a middle separator line: ├───────────┤
func (b *Box) Mid() string {
	return BoxTeeLeft + strings.Repeat(BoxHorizontal, b.Width) + BoxTeeRight
}

// Bottom returns the bottom border of the box: ╰───────────╯
func (b *Box) Bottom() string {
	return BoxBottomLeft + strings.Repeat(BoxHorizontal, b.Width) + BoxBottomRight
}

// Row returns a content row with vertical borders: │ content   │
// Content is left-aligned and padded to fill the box width.
// If content exceeds width, it is truncated.
func (b *Box) Row(content string) string {
	// Calculate visible length (excluding ANSI escape codes)
	visibleLen := visibleLength(content)

	if visibleLen >= b.Width {
		// Truncate content if too long (simple truncation, doesn't handle ANSI mid-sequence)
		return BoxVertical + truncateVisible(content, b.Width) + BoxVertical
	}

	// Pad content to fill width
	padding := strings.Repeat(" ", b.Width-visibleLen)
	return BoxVertical + content + padding + BoxVertical
}

// RowCenter returns a content row with content centered.
func (b *Box) RowCenter(content string) string {
	visibleLen := visibleLength(content)

	if visibleLen >= b.Width {
		return BoxVertical + truncateVisible(content, b.Width) + BoxVertical
	}

	totalPadding := b.Width - visibleLen
	leftPad := totalPadding / 2
	rightPad := totalPadding - leftPad

	return BoxVertical + strings.Repeat(" ", leftPad) + content + strings.Repeat(" ", rightPad) + BoxVertical
}

// EmptyRow returns an empty content row: │           │
func (b *Box) EmptyRow() string {
	return BoxVertical + strings.Repeat(" ", b.Width) + BoxVertical
}

// visibleLength returns the visible length of a string, excluding ANSI escape codes.
func visibleLength(s string) int {
	length := 0
	inEscape := false

	for _, r := range s {
		if r == '\033' {
			inEscape = true
			continue
		}
		if inEscape {
			if r == 'm' {
				inEscape = false
			}
			continue
		}
		length++
	}

	return length
}

// truncateVisible truncates a string to the specified visible width,
// preserving ANSI escape codes and appending a reset code if needed.
func truncateVisible(s string, width int) string {
	var result strings.Builder
	visible := 0
	inEscape := false
	hasOpenEscape := false

	for _, r := range s {
		if r == '\033' {
			inEscape = true
			hasOpenEscape = true
			result.WriteRune(r)
			continue
		}
		if inEscape {
			result.WriteRune(r)
			if r == 'm' {
				inEscape = false
				// Check if this is a reset code
				if strings.HasSuffix(result.String(), ColorReset) {
					hasOpenEscape = false
				}
			}
			continue
		}

		if visible >= width {
			break
		}
		result.WriteRune(r)
		visible++
	}

	// Append reset if we truncated mid-styled text
	if hasOpenEscape {
		result.WriteString(ColorReset)
	}

	return result.String()
}

// HorizontalLine returns a simple horizontal line of the specified width.
func HorizontalLine(width int) string {
	return strings.Repeat(BoxHorizontal, width)
}

// Padding returns spaces of the specified width, useful for alignment.
func Padding(width int) string {
	return strings.Repeat(" ", width)
}

// DisplayWidth returns the display width of a string, accounting for
// wide characters (like CJK) and ANSI escape codes.
// For simplicity, this treats all non-ANSI runes as width 1.
// A more complete implementation would check Unicode width properties.
func DisplayWidth(s string) int {
	return visibleLength(s)
}

// PadRight pads a string to the specified visible width with spaces on the right.
func PadRight(s string, width int) string {
	visLen := visibleLength(s)
	if visLen >= width {
		return s
	}
	return s + strings.Repeat(" ", width-visLen)
}

// PadLeft pads a string to the specified visible width with spaces on the left.
func PadLeft(s string, width int) string {
	visLen := visibleLength(s)
	if visLen >= width {
		return s
	}
	return strings.Repeat(" ", width-visLen) + s
}

// StringWidth is a convenience alias for visibleLength.
// It returns the number of visible characters (excluding ANSI codes).
var StringWidth = visibleLength
