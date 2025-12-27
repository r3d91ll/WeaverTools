// Package help provides formatted help output for the Weaver interactive shell.
//
// This package creates visually structured help displays using Unicode box
// drawing characters, ANSI color codes, and category-grouped command listings
// with inline examples. The output is designed to be scannable, informative,
// and aesthetically consistent with modern CLI tools.
//
// # Package Structure
//
// The package is organized into several focused files:
//
//   - help.go: Core types, constants, and package documentation
//   - box.go: Box drawing utilities for bordered sections and ANSI-aware string handling
//   - style.go: Text styling functions using ANSI color codes
//   - commands.go: Command metadata, categories, and the command registry
//   - render.go: The Renderer type that composes styled output
//
// # Visual Design
//
// The help output uses a consistent visual design language:
//
//   - Rounded box corners (╭╮╰╯) for a modern aesthetic
//   - Green bold headers for categories
//   - Cyan command names for quick scanning
//   - Yellow for examples, arguments, and shortcuts
//   - Gray (dim) for descriptions and structural elements
//   - Vertical box lines (│) for visual continuity in listings
//
// # Usage
//
// Basic usage with the Renderer type:
//
//	renderer := help.NewRenderer(os.Stdout)
//	renderer.RenderFull()              // Render complete help with all categories
//	renderer.RenderCommand("extract")  // Render detailed help for a specific command
//	renderer.RenderShortcuts()         // Render shortcuts and tips section only
//
// Style functions can also be used independently:
//
//	fmt.Println(help.Command("/help"))           // Cyan command name
//	fmt.Println(help.Example("/extract honor"))  // Yellow example text
//	fmt.Println(help.Dim("description text"))    // Gray dimmed text
//
// Command metadata is accessible via registry functions:
//
//	cmd, found := help.GetCommand("extract")     // Lookup by name (with or without /)
//	cmds := help.GetCommandsByCategory(help.CategoryAnalysis)
//
// # Terminal Compatibility
//
// The output is designed for 80-column terminals and uses ANSI escape codes
// that are widely supported. In terminals without color support, the text
// remains readable as the styling degrades gracefully to plain text.
package help

import "io"

// Box drawing characters for visual structure.
// Uses rounded corners for a modern aesthetic.
const (
	// Corners (rounded)
	BoxTopLeft     = "╭"
	BoxTopRight    = "╮"
	BoxBottomLeft  = "╰"
	BoxBottomRight = "╯"

	// Lines
	BoxHorizontal = "─"
	BoxVertical   = "│"

	// T-junctions
	BoxTeeLeft  = "├"
	BoxTeeRight = "┤"
	BoxTeeDown  = "┬"
	BoxTeeUp    = "┴"

	// Cross
	BoxCross = "┼"
)

// ANSI color codes for styled output.
const (
	ColorReset  = "\033[0m"
	ColorBold   = "\033[1m"
	ColorDim    = "\033[2m"
	ColorCyan   = "\033[36m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorGray   = "\033[90m"
)

// Renderer formats and writes help output.
type Renderer struct {
	w io.Writer
}

// NewRenderer creates a new help renderer that writes to w.
func NewRenderer(w io.Writer) *Renderer {
	return &Renderer{w: w}
}
