// Package shell provides the interactive REPL for Weaver.
package shell

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

// Prompter defines the interface for interactive confirmation prompts.
// It allows commands to ask for user confirmation before destructive operations.
// The interface enables easy mocking for testing purposes.
type Prompter interface {
	// Confirm displays a message and waits for the user to confirm.
	// Returns true if the user confirms (enters "yes" or "y"), false otherwise.
	// The message should describe what action requires confirmation.
	Confirm(message string) (bool, error)
}

// InteractivePrompter implements Prompter for real user interaction.
// It reads from stdin and writes prompts to stdout.
type InteractivePrompter struct {
	reader io.Reader
	writer io.Writer
}

// NewInteractivePrompter creates a new InteractivePrompter using stdin/stdout.
func NewInteractivePrompter() *InteractivePrompter {
	return &InteractivePrompter{
		reader: os.Stdin,
		writer: os.Stdout,
	}
}

// NewInteractivePrompterWithIO creates an InteractivePrompter with custom I/O.
// This is useful for testing with simulated input/output streams.
func NewInteractivePrompterWithIO(reader io.Reader, writer io.Writer) *InteractivePrompter {
	return &InteractivePrompter{
		reader: reader,
		writer: writer,
	}
}

// Confirm implements Prompter.Confirm for interactive prompts.
// It displays the message followed by " [y/N]: " and reads user input.
// Returns true only if the user enters "yes" or "y" (case-insensitive).
// Returns false for any other input or empty response (default is No).
func (p *InteractivePrompter) Confirm(message string) (bool, error) {
	// Display the prompt with [y/N] to indicate N is the default
	fmt.Fprintf(p.writer, "%s [y/N]: ", message)

	// Read the user's response
	scanner := bufio.NewScanner(p.reader)
	if !scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return false, fmt.Errorf("failed to read confirmation: %w", err)
		}
		// EOF without input, treat as "no"
		return false, nil
	}

	response := strings.TrimSpace(strings.ToLower(scanner.Text()))

	// Only explicit "yes" or "y" confirms; everything else is "no"
	return response == "yes" || response == "y", nil
}

// Ensure InteractivePrompter implements Prompter at compile time.
var _ Prompter = (*InteractivePrompter)(nil)

// MockPrompter is a test implementation of Prompter that returns predefined responses.
// It records all prompts for verification in tests.
type MockPrompter struct {
	// Response is the predefined response to return from Confirm.
	Response bool
	// Error is an optional error to return from Confirm.
	Error error
	// Prompts records all messages passed to Confirm.
	Prompts []string
	// CallCount tracks how many times Confirm was called.
	CallCount int
}

// NewMockPrompter creates a MockPrompter that will return the given response.
func NewMockPrompter(response bool) *MockPrompter {
	return &MockPrompter{
		Response: response,
		Prompts:  make([]string, 0),
	}
}

// NewMockPrompterWithError creates a MockPrompter that will return an error.
func NewMockPrompterWithError(err error) *MockPrompter {
	return &MockPrompter{
		Error:   err,
		Prompts: make([]string, 0),
	}
}

// Confirm implements Prompter.Confirm for testing.
// It records the message and returns the predefined response.
func (m *MockPrompter) Confirm(message string) (bool, error) {
	m.CallCount++
	m.Prompts = append(m.Prompts, message)

	if m.Error != nil {
		return false, m.Error
	}
	return m.Response, nil
}

// LastPrompt returns the most recent prompt message, or empty string if none.
func (m *MockPrompter) LastPrompt() string {
	if len(m.Prompts) == 0 {
		return ""
	}
	return m.Prompts[len(m.Prompts)-1]
}

// Reset clears the recorded prompts and call count.
func (m *MockPrompter) Reset() {
	m.Prompts = make([]string, 0)
	m.CallCount = 0
}

// Ensure MockPrompter implements Prompter at compile time.
var _ Prompter = (*MockPrompter)(nil)
