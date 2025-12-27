package help

// commands.go defines command metadata, categories, and the command registry.
// The Commands slice is the source of truth for all shell command documentation.

// Category represents a command category for grouping in help output.
type Category string

// Command categories for organizing help output.
const (
	// CategorySession contains session and agent management commands:
	// /agents, /session, /history, /clear, /default
	CategorySession Category = "session"

	// CategoryAnalysis contains concept extraction and analysis commands:
	// /extract, /analyze, /compare, /validate, /metrics, /concepts, /clear_concepts
	CategoryAnalysis Category = "analysis"

	// CategoryGeneral contains general utility commands:
	// /help, /quit
	CategoryGeneral Category = "general"
)

// CategoryInfo provides display metadata for a command category.
type CategoryInfo struct {
	// DisplayName is the human-readable name shown in help output
	DisplayName string

	// Icon is a Unicode icon/emoji displayed before the category name
	Icon string
}

// CategoryOrder defines the order in which categories appear in help output.
var CategoryOrder = []Category{
	CategorySession,
	CategoryAnalysis,
	CategoryGeneral,
}

// Categories maps each Category to its display information.
var Categories = map[Category]CategoryInfo{
	CategorySession: {
		DisplayName: "Session Management",
		Icon:        "📋",
	},
	CategoryAnalysis: {
		DisplayName: "Concept Extraction & Analysis",
		Icon:        "🔍",
	},
	CategoryGeneral: {
		DisplayName: "General",
		Icon:        "ℹ️",
	},
}

// DisplayName returns the human-readable display name for the category.
func (c Category) DisplayName() string {
	if info, ok := Categories[c]; ok {
		return info.DisplayName
	}
	return string(c)
}

// Icon returns the icon for the category.
func (c Category) Icon() string {
	if info, ok := Categories[c]; ok {
		return info.Icon
	}
	return ""
}

// Command represents a shell command with its metadata for help display.
// Commands are organized by category and can include usage examples.
type Command struct {
	// Name is the command name including the leading slash (e.g., "/help")
	Name string

	// Shortcut is an optional short alias for the command (e.g., "/h")
	Shortcut string

	// Category groups related commands together in help output
	Category Category

	// Description is a brief explanation of what the command does
	Description string

	// Usage shows the command syntax with any required/optional arguments
	// Example: "/extract <file|directory> [options]"
	Usage string

	// Examples provides sample usages with descriptions
	Examples []Example
}

// Example represents a usage example for a command.
// Each example shows a concrete command invocation and explains what it does.
type Example struct {
	// Command is the example command line (e.g., "/extract main.go")
	Command string

	// Description explains what this example does
	Description string
}

// Commands contains metadata for all shell commands.
// This registry is used to generate help output and command documentation.
var Commands = []Command{
	// Session Management Commands
	{
		Name:        "/agents",
		Category:    CategorySession,
		Description: "List available agents with status and capabilities",
		Usage:       "/agents",
		Examples:    nil, // Simple command, no examples needed
	},
	{
		Name:        "/session",
		Category:    CategorySession,
		Description: "Show current session info (ID, conversations, messages)",
		Usage:       "/session",
		Examples:    nil,
	},
	{
		Name:        "/history",
		Category:    CategorySession,
		Description: "Show last 10 messages in conversation",
		Usage:       "/history",
		Examples:    nil,
	},
	{
		Name:        "/clear",
		Category:    CategorySession,
		Description: "Start a new conversation (clears history)",
		Usage:       "/clear",
		Examples:    nil,
	},
	{
		Name:        "/default",
		Category:    CategorySession,
		Description: "Set or show the default agent for messages",
		Usage:       "/default [agent]",
		Examples: []Example{
			{Command: "/default", Description: "Show current default agent"},
			{Command: "/default senior", Description: "Set senior as default"},
		},
	},

	// Concept Extraction & Analysis Commands
	{
		Name:        "/extract",
		Category:    CategoryAnalysis,
		Description: "Extract concept vectors using hidden states",
		Usage:       "/extract <concept> [count]",
		Examples: []Example{
			{Command: "/extract honor 20", Description: "Extract 20 samples for 'honor'"},
			{Command: "/extract love 15", Description: "Extract 15 samples for 'love'"},
			{Command: "/extract random 20", Description: "Extract baseline random vectors"},
		},
	},
	{
		Name:        "/analyze",
		Category:    CategoryAnalysis,
		Description: "Run Kakeya geometry analysis on concept",
		Usage:       "/analyze <concept>",
		Examples: []Example{
			{Command: "/analyze honor", Description: "Analyze geometry of 'honor' vectors"},
		},
	},
	{
		Name:        "/compare",
		Category:    CategoryAnalysis,
		Description: "Compare geometric properties of two concepts",
		Usage:       "/compare <concept1> <concept2>",
		Examples: []Example{
			{Command: "/compare honor duty", Description: "Compare 'honor' vs 'duty'"},
			{Command: "/compare love random", Description: "Compare 'love' to baseline"},
		},
	},
	{
		Name:        "/validate",
		Category:    CategoryAnalysis,
		Description: "Test concept extraction consistency",
		Usage:       "/validate <concept> [iterations]",
		Examples: []Example{
			{Command: "/validate honor 5", Description: "Run 5 extraction iterations"},
		},
	},
	{
		Name:        "/metrics",
		Category:    CategoryAnalysis,
		Description: "Show raw metric values for a concept",
		Usage:       "/metrics <concept>",
		Examples: []Example{
			{Command: "/metrics honor", Description: "Show detailed metrics"},
		},
	},
	{
		Name:        "/concepts",
		Category:    CategoryAnalysis,
		Description: "List all stored concepts with sample counts",
		Usage:       "/concepts",
		Examples:    nil,
	},
	{
		Name:        "/clear_concepts",
		Category:    CategoryAnalysis,
		Description: "Remove all stored concepts",
		Usage:       "/clear_concepts",
		Examples:    nil,
	},

	// General Commands
	{
		Name:        "/help",
		Shortcut:    "/h",
		Category:    CategoryGeneral,
		Description: "Show this help message",
		Usage:       "/help [command]",
		Examples: []Example{
			{Command: "/help", Description: "Show all commands"},
			{Command: "/help extract", Description: "Show detailed /extract help"},
		},
	},
	{
		Name:        "/quit",
		Shortcut:    "/q",
		Category:    CategoryGeneral,
		Description: "Exit Weaver",
		Usage:       "/quit",
		Examples:    nil,
	},
}

// GetCommandsByCategory returns all commands in a given category.
func GetCommandsByCategory(cat Category) []Command {
	var result []Command
	for _, cmd := range Commands {
		if cmd.Category == cat {
			result = append(result, cmd)
		}
	}
	return result
}

// GetCommand returns a command by name (with or without leading slash).
func GetCommand(name string) (Command, bool) {
	// Normalize name to include leading slash
	if len(name) > 0 && name[0] != '/' {
		name = "/" + name
	}

	for _, cmd := range Commands {
		if cmd.Name == name || cmd.Shortcut == name {
			return cmd, true
		}
	}
	return Command{}, false
}
