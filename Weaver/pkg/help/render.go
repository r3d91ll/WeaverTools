package help

// render.go implements the Renderer type that composes styled help output.
// It brings together box drawing, styling, and command metadata to produce
// the formatted help displays shown to users.

import (
	"fmt"
	"strings"
)

// Layout constants define the visual spacing for help output.
// These values are tuned for readability in 80-column terminals.
const (
	// commandColumnWidth is the width allocated for command name + shortcut display.
	// Set to 22 to accommodate the longest command with shortcut: "/help (or /h)".
	commandColumnWidth = 22

	// indentCategory is the left margin for category headers (2 spaces).
	indentCategory = "  "

	// indentCommand is the left margin for command lines (4 spaces).
	// Provides visual hierarchy under category headers.
	indentCommand = "    "

	// indentExample is the left margin for inline examples (6 spaces).
	// Provides visual hierarchy under command lines.
	indentExample = "      "
)

// RenderFull renders the complete help output with all categories.
// Commands are grouped by category with visual separators.
func (r *Renderer) RenderFull() {
	r.writeln("")
	r.writeln(Header(indentCategory + "Weaver Commands"))
	r.writeln("")

	// Render each category in order
	for _, cat := range CategoryOrder {
		r.renderCategory(cat)
	}

	// Render shortcuts section
	r.RenderShortcuts()
}

// RenderCommand renders detailed help for a single command.
// Shows full description, usage, and all examples.
// Returns false if the command is not found.
func (r *Renderer) RenderCommand(name string) bool {
	cmd, found := GetCommand(name)
	if !found {
		r.writeln(fmt.Sprintf(indentCategory+"Command '%s' not found. Use /help to see all commands.", name))
		return false
	}

	r.writeln("")

	// Command name with shortcut
	if cmd.Shortcut != "" {
		r.writeln(indentCategory + CommandWithShortcut(cmd.Name, cmd.Shortcut))
	} else {
		r.writeln(indentCategory + StyleCommand(cmd.Name))
	}

	// Description
	r.writeln(indentCategory + Dim(cmd.Description))
	r.writeln("")

	// Usage
	r.writeln(indentCategory + Bold("Usage:") + " " + StyleExample(cmd.Usage))
	r.writeln("")

	// Examples
	if len(cmd.Examples) > 0 {
		r.writeln(indentCategory + Bold("Examples:"))
		for _, ex := range cmd.Examples {
			r.writeln(indentCommand + ExampleLine(ex.Command, ex.Description))
		}
		r.writeln("")
	}

	return true
}

// RenderShortcuts renders the shortcuts and tips reference section.
// Uses consistent indentation and box drawing characters for visual alignment.
// Formatted as a compact reference with aligned columns.
func (r *Renderer) RenderShortcuts() {
	r.writeln("")
	r.writeln(indentCategory + StyleCategory("💡 Shortcuts & Tips"))

	// Box drawing separator matching category style
	separatorWidth := commandColumnWidth + 20
	r.writeln(indentCategory + Dim(BoxTeeLeft+strings.Repeat(BoxHorizontal, separatorWidth)))

	// Command aliases - compact single line
	r.writeln(indentCommand + Dim(BoxVertical+" ") + Dim("Aliases: ") +
		Shortcut("/h") + Dim("→help  ") +
		Shortcut("/q") + Dim("→quit  ") +
		Shortcut("/exit") + Dim("→quit"))

	// Agent messaging - compact format
	r.writeln(indentCommand + Dim(BoxVertical+" ") + Dim("Message: ") +
		StyleExample("@agent text") + Dim(" (to agent)  ") +
		StyleExample("text") + Dim(" (to default)"))

	// Keyboard shortcuts - compact single line
	r.writeln(indentCommand + Dim(BoxVertical+" ") + Dim("Keys:    ") +
		Shortcut("Ctrl+C") + Dim(" cancel  ") +
		Shortcut("Ctrl+D") + Dim(" exit  ") +
		Shortcut("↑↓") + Dim(" history"))

	r.writeln("")
}

// renderCategory renders a single category with its commands.
// Uses box drawing characters for visual hierarchy and aligns command columns.
func (r *Renderer) renderCategory(cat Category) {
	commands := GetCommandsByCategory(cat)
	if len(commands) == 0 {
		return
	}

	// Category header with icon
	icon := cat.Icon()
	displayName := cat.DisplayName()
	r.writeln(indentCategory + StyleCategory(icon+" "+displayName))

	// Box drawing separator: ├───────────
	separatorWidth := commandColumnWidth + 20 // Extend across command + description area
	r.writeln(indentCategory + Dim(BoxTeeLeft+strings.Repeat(BoxHorizontal, separatorWidth)))

	// Render each command with aligned columns
	for _, cmd := range commands {
		r.renderCommandLine(cmd)
	}

	r.writeln("")
}

// renderCommandLine renders a single command line in the category listing.
// Uses column alignment for consistent visual layout.
func (r *Renderer) renderCommandLine(cmd Command) {
	// Build the command name + shortcut part
	var cmdPart string
	if cmd.Shortcut != "" {
		cmdPart = CommandWithShortcut(cmd.Name, cmd.Shortcut)
	} else {
		cmdPart = StyleCommand(cmd.Name)
	}

	// Calculate visible length for padding (excluding ANSI codes)
	visLen := visibleLength(cmdPart)

	// Pad command part to fixed column width for alignment
	padding := ""
	if visLen < commandColumnWidth {
		padding = strings.Repeat(" ", commandColumnWidth-visLen)
	}

	// Format: │ command (shortcut)     description
	line := indentCommand + Dim(BoxVertical+" ") + cmdPart + padding + Dim(cmd.Description)
	r.writeln(line)

	// Show 1-2 inline examples for complex commands
	// Uses syntax highlighting: command in cyan, arguments in yellow
	exampleCount := len(cmd.Examples)
	if exampleCount > 0 {
		// Limit to 2 inline examples in the full help view
		maxInline := 2
		if exampleCount < maxInline {
			maxInline = exampleCount
		}

		for i := 0; i < maxInline; i++ {
			ex := cmd.Examples[i]
			r.writeln(indentExample + Dim(BoxVertical+"   e.g. ") + HighlightExampleCommand(ex.Command))
		}
	}
}

// writeln writes a line to the renderer's output.
func (r *Renderer) writeln(s string) {
	fmt.Fprintln(r.w, s)
}

// write writes to the renderer's output without newline.
func (r *Renderer) write(s string) {
	fmt.Fprint(r.w, s)
}
