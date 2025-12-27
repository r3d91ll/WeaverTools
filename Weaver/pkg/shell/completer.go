// Package shell provides the interactive REPL for Weaver.
package shell

import (
	"strings"

	"github.com/chzyer/readline"
	"github.com/r3d91ll/weaver/pkg/runtime"
)

// commands is the static list of available shell commands (without the / prefix).
var commands = []string{
	"quit",
	"exit",
	"q",
	"help",
	"h",
	"agents",
	"session",
	"history",
	"clear",
	"default",
	"extract",
	"analyze",
	"compare",
	"validate",
	"concepts",
	"metrics",
	"clear_concepts",
}

// ShellCompleter provides tab completion for commands and agent names.
// It implements the readline.AutoCompleter interface.
type ShellCompleter struct {
	agents *runtime.Manager
}

// NewShellCompleter creates a new completer with access to the agent manager.
func NewShellCompleter(agents *runtime.Manager) *ShellCompleter {
	return &ShellCompleter{
		agents: agents,
	}
}

// Ensure ShellCompleter implements readline.AutoCompleter at compile time.
var _ readline.AutoCompleter = (*ShellCompleter)(nil)

// Do implements readline.AutoCompleter.
// It provides completions for:
//   - Commands starting with "/" (e.g., /help, /agents)
//   - Agent names starting with "@" (e.g., @senior, @junior)
//
// Parameters:
//   - line: The whole line of input as runes
//   - pos: The current cursor position in the line
//
// Returns:
//   - newLine: All candidate completions (as suffixes after the common prefix)
//   - length: The number of characters in the common prefix
func (c *ShellCompleter) Do(line []rune, pos int) (newLine [][]rune, length int) {
	// Edge case: empty input or cursor at beginning
	if len(line) == 0 || pos <= 0 {
		return nil, 0
	}

	// Clamp pos to valid range (safety check)
	if pos > len(line) {
		pos = len(line)
	}

	// Extract text up to cursor position
	lineStr := string(line[:pos])

	// Find the start of the current word
	// Check for both space and tab as word separators
	wordStart := findWordStart(lineStr)
	currentWord := lineStr[wordStart:]

	// Edge case: empty word (e.g., trailing space)
	if currentWord == "" {
		return nil, 0
	}

	// Handle command completion (starts with /)
	if strings.HasPrefix(currentWord, "/") {
		return c.completeCommand(currentWord)
	}

	// Handle agent completion (starts with @)
	if strings.HasPrefix(currentWord, "@") {
		return c.completeAgent(currentWord)
	}

	return nil, 0
}

// findWordStart returns the index where the current word begins.
// It looks for the last whitespace character (space or tab) and returns
// the position after it. If no whitespace is found, returns 0 (start of line).
func findWordStart(s string) int {
	// Look for the last whitespace character
	lastSpace := strings.LastIndex(s, " ")
	lastTab := strings.LastIndex(s, "\t")

	// Use whichever is later (closer to the end)
	wordStart := lastSpace
	if lastTab > wordStart {
		wordStart = lastTab
	}

	// Return position after the whitespace, or 0 if none found
	return wordStart + 1
}

// completeCommand returns completions for commands starting with the given prefix.
// The prefix includes the leading "/" character.
func (c *ShellCompleter) completeCommand(prefix string) ([][]rune, int) {
	// Remove the leading "/" to match against command names
	cmdPrefix := strings.TrimPrefix(prefix, "/")

	var matches [][]rune
	for _, cmd := range commands {
		if strings.HasPrefix(cmd, cmdPrefix) {
			// Return the suffix that completes the command (add space after)
			suffix := cmd[len(cmdPrefix):] + " "
			matches = append(matches, []rune(suffix))
		}
	}

	// Return the length of the prefix (including "/") that we're completing
	return matches, len(prefix)
}

// completeAgent returns completions for agent names starting with the given prefix.
// The prefix includes the leading "@" character.
// Agent names are fetched dynamically from the runtime manager.
func (c *ShellCompleter) completeAgent(prefix string) ([][]rune, int) {
	// If no agent manager is available, return no completions
	if c.agents == nil {
		return nil, 0
	}

	// Remove the leading "@" to match against agent names
	agentPrefix := strings.TrimPrefix(prefix, "@")

	// Get the current list of agents dynamically
	agentNames := c.agents.List()

	var matches [][]rune
	for _, name := range agentNames {
		if strings.HasPrefix(name, agentPrefix) {
			// Return the suffix that completes the agent name (add space after)
			suffix := name[len(agentPrefix):] + " "
			matches = append(matches, []rune(suffix))
		}
	}

	// Return the length of the prefix (including "@") that we're completing
	return matches, len(prefix)
}
