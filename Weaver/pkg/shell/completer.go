// Package shell provides the interactive REPL for Weaver.
package shell

import (
	"strings"

	"github.com/chzyer/readline"
	"github.com/r3d91ll/weaver/pkg/concepts"
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

// conceptCommands is the list of commands that expect concept names as arguments.
// These commands will trigger concept name tab completion for their arguments.
var conceptCommands = []string{
	"analyze",
	"compare",
	"validate",
	"metrics",
}

// ShellCompleter provides tab completion for commands and agent names.
// It implements the readline.AutoCompleter interface.
type ShellCompleter struct {
	agents   *runtime.Manager
	concepts *concepts.Store
}

// NewShellCompleter creates a new completer with access to the agent manager
// and concept store for dynamic tab completion.
func NewShellCompleter(agents *runtime.Manager, conceptStore *concepts.Store) *ShellCompleter {
	return &ShellCompleter{
		agents:   agents,
		concepts: conceptStore,
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

	// Handle concept completion for arguments to concept-expecting commands
	// Check if the line contains a concept command before the current word
	if c.isConceptCommandContext(lineStr, wordStart) {
		return c.completeConcept(currentWord)
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

// isConceptCommandContext checks if the line contains a concept-expecting command
// before the current word position. This is used to trigger concept name completion
// for arguments to commands like /analyze, /compare, /validate, /metrics.
//
// Parameters:
//   - line: The input line up to the cursor position
//   - wordStart: The starting position of the current word being typed
//
// Returns true if the line starts with a concept command followed by whitespace.
func (c *ShellCompleter) isConceptCommandContext(line string, wordStart int) bool {
	// Get the part of the line before the current word
	beforeWord := line[:wordStart]

	// Trim trailing whitespace to get the command portion
	beforeWord = strings.TrimRight(beforeWord, " \t")

	// Check if the line starts with / (command prefix)
	if !strings.HasPrefix(beforeWord, "/") {
		// No command on this line, might still have concept command earlier
		// Look for the last command in the line
		lastCmdIdx := strings.LastIndex(beforeWord, "/")
		if lastCmdIdx == -1 {
			return false
		}
		beforeWord = beforeWord[lastCmdIdx:]
	}

	// Extract the command name (first word after /)
	cmdPart := strings.TrimPrefix(beforeWord, "/")

	// The command is the first word (up to the first space/tab)
	cmdName := cmdPart
	spaceIdx := strings.IndexAny(cmdPart, " \t")
	if spaceIdx != -1 {
		cmdName = cmdPart[:spaceIdx]
	}

	// Check if this is a concept-expecting command
	for _, conceptCmd := range conceptCommands {
		if cmdName == conceptCmd {
			return true
		}
	}

	return false
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

// completeConcept returns completions for concept names starting with the given prefix.
// Concept names are fetched dynamically from the concept store.
func (c *ShellCompleter) completeConcept(prefix string) ([][]rune, int) {
	// If no concept store is available, return no completions
	if c.concepts == nil {
		return nil, 0
	}

	// Get the current list of concepts dynamically
	// List() returns map[string]int with concept names and sample counts
	conceptMap := c.concepts.List()

	var matches [][]rune
	for name := range conceptMap {
		if strings.HasPrefix(name, prefix) {
			// Return the suffix that completes the concept name (add space after)
			suffix := name[len(prefix):] + " "
			matches = append(matches, []rune(suffix))
		}
	}

	// Return the length of the prefix that we're completing
	return matches, len(prefix)
}
