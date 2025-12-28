// Package shell provides the interactive REPL for Weaver.
package shell

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/r3d91ll/weaver/pkg/concepts"
)

// =============================================================================
// InteractivePrompter Tests
// =============================================================================

func TestInteractivePrompter_ConfirmYes(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		message string
		want    bool
	}{
		{
			name:    "lowercase yes",
			input:   "yes\n",
			message: "Delete all data?",
			want:    true,
		},
		{
			name:    "uppercase YES",
			input:   "YES\n",
			message: "Delete all data?",
			want:    true,
		},
		{
			name:    "mixed case Yes",
			input:   "Yes\n",
			message: "Delete all data?",
			want:    true,
		},
		{
			name:    "lowercase y",
			input:   "y\n",
			message: "Delete all data?",
			want:    true,
		},
		{
			name:    "uppercase Y",
			input:   "Y\n",
			message: "Delete all data?",
			want:    true,
		},
		{
			name:    "yes with leading whitespace",
			input:   "  yes\n",
			message: "Delete all data?",
			want:    true,
		},
		{
			name:    "yes with trailing whitespace",
			input:   "yes  \n",
			message: "Delete all data?",
			want:    true,
		},
		{
			name:    "y with whitespace",
			input:   "  y  \n",
			message: "Delete all data?",
			want:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			writer := &bytes.Buffer{}
			prompter := NewInteractivePrompterWithIO(reader, writer)

			got, err := prompter.Confirm(tt.message)
			if err != nil {
				t.Errorf("Confirm() unexpected error: %v", err)
				return
			}
			if got != tt.want {
				t.Errorf("Confirm() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestInteractivePrompter_ConfirmNo(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		message string
		want    bool
	}{
		{
			name:    "lowercase no",
			input:   "no\n",
			message: "Delete all data?",
			want:    false,
		},
		{
			name:    "uppercase NO",
			input:   "NO\n",
			message: "Delete all data?",
			want:    false,
		},
		{
			name:    "lowercase n",
			input:   "n\n",
			message: "Delete all data?",
			want:    false,
		},
		{
			name:    "empty input (default is no)",
			input:   "\n",
			message: "Delete all data?",
			want:    false,
		},
		{
			name:    "random text",
			input:   "maybe\n",
			message: "Delete all data?",
			want:    false,
		},
		{
			name:    "numbers",
			input:   "123\n",
			message: "Delete all data?",
			want:    false,
		},
		{
			name:    "yess (not exact match)",
			input:   "yess\n",
			message: "Delete all data?",
			want:    false,
		},
		{
			name:    "yeah",
			input:   "yeah\n",
			message: "Delete all data?",
			want:    false,
		},
		{
			name:    "only whitespace",
			input:   "   \n",
			message: "Delete all data?",
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			writer := &bytes.Buffer{}
			prompter := NewInteractivePrompterWithIO(reader, writer)

			got, err := prompter.Confirm(tt.message)
			if err != nil {
				t.Errorf("Confirm() unexpected error: %v", err)
				return
			}
			if got != tt.want {
				t.Errorf("Confirm() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestInteractivePrompter_PromptFormat(t *testing.T) {
	reader := strings.NewReader("no\n")
	writer := &bytes.Buffer{}
	prompter := NewInteractivePrompterWithIO(reader, writer)

	message := "This will clear 5 concepts. Are you sure?"
	_, err := prompter.Confirm(message)
	if err != nil {
		t.Fatalf("Confirm() unexpected error: %v", err)
	}

	// Verify the prompt format includes [y/N]
	output := writer.String()
	expectedPrompt := message + " [y/N]: "
	if output != expectedPrompt {
		t.Errorf("Prompt output = %q, want %q", output, expectedPrompt)
	}
}

func TestInteractivePrompter_EOF(t *testing.T) {
	// Empty reader simulates EOF
	reader := strings.NewReader("")
	writer := &bytes.Buffer{}
	prompter := NewInteractivePrompterWithIO(reader, writer)

	got, err := prompter.Confirm("Delete all data?")
	if err != nil {
		t.Errorf("Confirm() unexpected error on EOF: %v", err)
	}
	// EOF should return false (default is no)
	if got != false {
		t.Errorf("Confirm() on EOF = %v, want false", got)
	}
}

func TestNewInteractivePrompter(t *testing.T) {
	// Test that the default constructor doesn't panic
	prompter := NewInteractivePrompter()
	if prompter == nil {
		t.Error("NewInteractivePrompter() returned nil")
	}
}

// =============================================================================
// MockPrompter Tests
// =============================================================================

func TestMockPrompter_ReturnsConfiguredResponse(t *testing.T) {
	tests := []struct {
		name     string
		response bool
	}{
		{
			name:     "returns true",
			response: true,
		},
		{
			name:     "returns false",
			response: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := NewMockPrompter(tt.response)

			got, err := mock.Confirm("Any message")
			if err != nil {
				t.Errorf("Confirm() unexpected error: %v", err)
				return
			}
			if got != tt.response {
				t.Errorf("Confirm() = %v, want %v", got, tt.response)
			}
		})
	}
}

func TestMockPrompter_ReturnsError(t *testing.T) {
	expectedErr := errors.New("test error")
	mock := NewMockPrompterWithError(expectedErr)

	got, err := mock.Confirm("Any message")
	if err == nil {
		t.Error("Confirm() expected error, got nil")
		return
	}
	if !errors.Is(err, expectedErr) {
		t.Errorf("Confirm() error = %v, want %v", err, expectedErr)
	}
	if got != false {
		t.Errorf("Confirm() on error = %v, want false", got)
	}
}

func TestMockPrompter_RecordsPrompts(t *testing.T) {
	mock := NewMockPrompter(true)

	messages := []string{
		"Delete 5 concepts?",
		"Clear conversation history?",
		"Are you sure?",
	}

	for _, msg := range messages {
		_, _ = mock.Confirm(msg)
	}

	// Verify all prompts were recorded
	if len(mock.Prompts) != len(messages) {
		t.Errorf("Prompts count = %d, want %d", len(mock.Prompts), len(messages))
	}

	for i, msg := range messages {
		if mock.Prompts[i] != msg {
			t.Errorf("Prompts[%d] = %q, want %q", i, mock.Prompts[i], msg)
		}
	}
}

func TestMockPrompter_CallCount(t *testing.T) {
	mock := NewMockPrompter(false)

	if mock.CallCount != 0 {
		t.Errorf("Initial CallCount = %d, want 0", mock.CallCount)
	}

	// Call Confirm multiple times
	for i := 0; i < 5; i++ {
		_, _ = mock.Confirm("test")
	}

	if mock.CallCount != 5 {
		t.Errorf("CallCount after 5 calls = %d, want 5", mock.CallCount)
	}
}

func TestMockPrompter_LastPrompt(t *testing.T) {
	mock := NewMockPrompter(true)

	// Initially empty
	if last := mock.LastPrompt(); last != "" {
		t.Errorf("LastPrompt() initial = %q, want empty", last)
	}

	// After first call
	_, _ = mock.Confirm("First message")
	if last := mock.LastPrompt(); last != "First message" {
		t.Errorf("LastPrompt() = %q, want 'First message'", last)
	}

	// After second call
	_, _ = mock.Confirm("Second message")
	if last := mock.LastPrompt(); last != "Second message" {
		t.Errorf("LastPrompt() = %q, want 'Second message'", last)
	}
}

func TestMockPrompter_Reset(t *testing.T) {
	mock := NewMockPrompter(true)

	// Make some calls
	_, _ = mock.Confirm("Message 1")
	_, _ = mock.Confirm("Message 2")

	if mock.CallCount != 2 {
		t.Errorf("CallCount before reset = %d, want 2", mock.CallCount)
	}
	if len(mock.Prompts) != 2 {
		t.Errorf("Prompts length before reset = %d, want 2", len(mock.Prompts))
	}

	// Reset
	mock.Reset()

	// Verify reset state
	if mock.CallCount != 0 {
		t.Errorf("CallCount after reset = %d, want 0", mock.CallCount)
	}
	if len(mock.Prompts) != 0 {
		t.Errorf("Prompts length after reset = %d, want 0", len(mock.Prompts))
	}

	// Response should still be configured
	got, _ := mock.Confirm("New message")
	if got != true {
		t.Errorf("Response after reset = %v, want true", got)
	}
}

func TestMockPrompter_ImplementsPrompter(t *testing.T) {
	// Verify MockPrompter satisfies the Prompter interface
	var _ Prompter = (*MockPrompter)(nil)

	// Test with interface type
	var p Prompter = NewMockPrompter(true)
	got, err := p.Confirm("test")
	if err != nil {
		t.Errorf("Prompter.Confirm() error = %v", err)
	}
	if got != true {
		t.Errorf("Prompter.Confirm() = %v, want true", got)
	}
}

func TestInteractivePrompter_ImplementsPrompter(t *testing.T) {
	// Verify InteractivePrompter satisfies the Prompter interface
	var _ Prompter = (*InteractivePrompter)(nil)

	// Test with interface type
	reader := strings.NewReader("y\n")
	writer := &bytes.Buffer{}
	var p Prompter = NewInteractivePrompterWithIO(reader, writer)

	got, err := p.Confirm("test")
	if err != nil {
		t.Errorf("Prompter.Confirm() error = %v", err)
	}
	if got != true {
		t.Errorf("Prompter.Confirm() = %v, want true", got)
	}
}

// =============================================================================
// MockPrompter Usage Pattern Tests
// =============================================================================

func TestMockPrompter_TypicalTestUsage(t *testing.T) {
	// This test demonstrates the typical pattern for using MockPrompter in tests

	// Scenario 1: Test behavior when user confirms
	t.Run("user confirms action", func(t *testing.T) {
		mock := NewMockPrompter(true)

		// Simulate code that uses the prompter
		confirmed, err := mock.Confirm("This will delete 10 items. Continue?")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Verify confirmation was granted
		if !confirmed {
			t.Error("expected confirmation to be granted")
		}

		// Verify the correct message was shown to user
		if mock.LastPrompt() != "This will delete 10 items. Continue?" {
			t.Errorf("unexpected prompt message: %q", mock.LastPrompt())
		}
	})

	// Scenario 2: Test behavior when user rejects
	t.Run("user rejects action", func(t *testing.T) {
		mock := NewMockPrompter(false)

		confirmed, err := mock.Confirm("This will delete 10 items. Continue?")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Verify confirmation was denied
		if confirmed {
			t.Error("expected confirmation to be denied")
		}
	})

	// Scenario 3: Test error handling
	t.Run("confirmation fails with error", func(t *testing.T) {
		mock := NewMockPrompterWithError(errors.New("input stream closed"))

		_, err := mock.Confirm("This will delete 10 items. Continue?")
		if err == nil {
			t.Error("expected error from confirmation")
		}
	})
}

// =============================================================================
// Edge Cases and Error Conditions
// =============================================================================

func TestMockPrompter_MultipleErrorsRecordPrompts(t *testing.T) {
	// Even when returning an error, prompts should be recorded
	mock := NewMockPrompterWithError(errors.New("test error"))

	_, _ = mock.Confirm("First prompt")
	_, _ = mock.Confirm("Second prompt")

	if mock.CallCount != 2 {
		t.Errorf("CallCount = %d, want 2", mock.CallCount)
	}
	if len(mock.Prompts) != 2 {
		t.Errorf("Prompts length = %d, want 2", len(mock.Prompts))
	}
}

func TestMockPrompter_EmptyMessage(t *testing.T) {
	mock := NewMockPrompter(true)

	_, err := mock.Confirm("")
	if err != nil {
		t.Errorf("Confirm with empty message returned error: %v", err)
	}

	if mock.LastPrompt() != "" {
		t.Errorf("LastPrompt() = %q, want empty string", mock.LastPrompt())
	}
}

func TestInteractivePrompter_LongMessage(t *testing.T) {
	longMessage := strings.Repeat("This is a very long message. ", 100)
	reader := strings.NewReader("y\n")
	writer := &bytes.Buffer{}
	prompter := NewInteractivePrompterWithIO(reader, writer)

	got, err := prompter.Confirm(longMessage)
	if err != nil {
		t.Errorf("Confirm() with long message returned error: %v", err)
	}
	if got != true {
		t.Errorf("Confirm() = %v, want true", got)
	}

	// Verify the full message was written
	expectedPrompt := longMessage + " [y/N]: "
	if writer.String() != expectedPrompt {
		t.Error("Long message was not fully written to output")
	}
}

// =============================================================================
// /clear_concepts Command Tests with Confirmation
// =============================================================================

// newTestShellForClearConcepts creates a minimal Shell for testing /clear_concepts.
// It only sets up the fields needed for the clear_concepts command.
func newTestShellForClearConcepts(prompter Prompter) *Shell {
	return &Shell{
		conceptStore: concepts.NewStore(),
		Prompter:     prompter,
	}
}

// addTestConcepts adds test concepts to the shell's concept store.
func addTestConcepts(s *Shell, count int) {
	for i := 0; i < count; i++ {
		conceptName := "concept" + string(rune('A'+i))
		sample := concepts.Sample{
			ID:      "sample-" + conceptName,
			Content: "Test content for " + conceptName,
		}
		s.conceptStore.Add(conceptName, sample)
	}
}

func TestClearConcepts_UserConfirms(t *testing.T) {
	// Setup: Create shell with MockPrompter that returns true (user confirms)
	mock := NewMockPrompter(true)
	shell := newTestShellForClearConcepts(mock)

	// Add some test concepts
	addTestConcepts(shell, 3)

	if shell.conceptStore.Count() != 3 {
		t.Fatalf("Expected 3 concepts before clear, got %d", shell.conceptStore.Count())
	}

	// Execute /clear_concepts command
	ctx := context.Background()
	err := shell.handleCommand(ctx, "/clear_concepts")
	if err != nil {
		t.Fatalf("handleCommand() returned unexpected error: %v", err)
	}

	// Verify: confirmation was requested
	if mock.CallCount != 1 {
		t.Errorf("Expected 1 confirmation call, got %d", mock.CallCount)
	}

	// Verify: prompt message includes concept count
	if !strings.Contains(mock.LastPrompt(), "3 concept(s)") {
		t.Errorf("Expected prompt to mention '3 concept(s)', got: %q", mock.LastPrompt())
	}

	// Verify: concepts were cleared (user confirmed)
	if shell.conceptStore.Count() != 0 {
		t.Errorf("Expected 0 concepts after confirmed clear, got %d", shell.conceptStore.Count())
	}
}

func TestClearConcepts_UserRejects(t *testing.T) {
	// Setup: Create shell with MockPrompter that returns false (user rejects)
	mock := NewMockPrompter(false)
	shell := newTestShellForClearConcepts(mock)

	// Add some test concepts
	addTestConcepts(shell, 3)

	if shell.conceptStore.Count() != 3 {
		t.Fatalf("Expected 3 concepts before clear, got %d", shell.conceptStore.Count())
	}

	// Execute /clear_concepts command
	ctx := context.Background()
	err := shell.handleCommand(ctx, "/clear_concepts")
	if err != nil {
		t.Fatalf("handleCommand() returned unexpected error: %v", err)
	}

	// Verify: confirmation was requested
	if mock.CallCount != 1 {
		t.Errorf("Expected 1 confirmation call, got %d", mock.CallCount)
	}

	// Verify: concepts were NOT cleared (user rejected)
	if shell.conceptStore.Count() != 3 {
		t.Errorf("Expected 3 concepts after rejected clear, got %d", shell.conceptStore.Count())
	}
}

func TestClearConcepts_ForceFlag(t *testing.T) {
	tests := []struct {
		name    string
		command string
	}{
		{
			name:    "--force flag",
			command: "/clear_concepts --force",
		},
		{
			name:    "-f flag",
			command: "/clear_concepts -f",
		},
		{
			name:    "--force with extra space",
			command: "/clear_concepts  --force",
		},
		{
			name:    "-f with extra space",
			command: "/clear_concepts  -f",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup: Create shell with MockPrompter (should NOT be called with --force)
			mock := NewMockPrompter(false) // Would reject if called
			shell := newTestShellForClearConcepts(mock)

			// Add some test concepts
			addTestConcepts(shell, 3)

			if shell.conceptStore.Count() != 3 {
				t.Fatalf("Expected 3 concepts before clear, got %d", shell.conceptStore.Count())
			}

			// Execute /clear_concepts with force flag
			ctx := context.Background()
			err := shell.handleCommand(ctx, tt.command)
			if err != nil {
				t.Fatalf("handleCommand() returned unexpected error: %v", err)
			}

			// Verify: confirmation was NOT requested (force flag skips it)
			if mock.CallCount != 0 {
				t.Errorf("Expected 0 confirmation calls with force flag, got %d", mock.CallCount)
			}

			// Verify: concepts were cleared even though mock would have rejected
			if shell.conceptStore.Count() != 0 {
				t.Errorf("Expected 0 concepts after forced clear, got %d", shell.conceptStore.Count())
			}
		})
	}
}

func TestClearConcepts_EmptyStore(t *testing.T) {
	// Setup: Create shell with empty concept store
	mock := NewMockPrompter(true)
	shell := newTestShellForClearConcepts(mock)

	// Verify store is empty
	if shell.conceptStore.Count() != 0 {
		t.Fatalf("Expected 0 concepts in empty store, got %d", shell.conceptStore.Count())
	}

	// Execute /clear_concepts command
	ctx := context.Background()
	err := shell.handleCommand(ctx, "/clear_concepts")
	if err != nil {
		t.Fatalf("handleCommand() returned unexpected error: %v", err)
	}

	// Verify: confirmation was NOT requested (no concepts to clear)
	if mock.CallCount != 0 {
		t.Errorf("Expected 0 confirmation calls for empty store, got %d", mock.CallCount)
	}
}

func TestClearConcepts_ConfirmationError(t *testing.T) {
	// Setup: Create shell with MockPrompter that returns an error
	expectedErr := errors.New("stdin closed unexpectedly")
	mock := NewMockPrompterWithError(expectedErr)
	shell := newTestShellForClearConcepts(mock)

	// Add some test concepts
	addTestConcepts(shell, 3)

	// Execute /clear_concepts command
	ctx := context.Background()
	err := shell.handleCommand(ctx, "/clear_concepts")

	// Verify: error was returned
	if err == nil {
		t.Fatal("Expected error from handleCommand(), got nil")
	}

	// Verify: error message contains context about confirmation failure
	if !strings.Contains(err.Error(), "confirmation") {
		t.Errorf("Expected error to mention 'confirmation', got: %v", err)
	}

	// Verify: concepts were NOT cleared (error occurred)
	if shell.conceptStore.Count() != 3 {
		t.Errorf("Expected 3 concepts after error, got %d", shell.conceptStore.Count())
	}
}

func TestClearConcepts_SingleConcept(t *testing.T) {
	// Setup: Create shell with MockPrompter that confirms
	mock := NewMockPrompter(true)
	shell := newTestShellForClearConcepts(mock)

	// Add a single concept
	addTestConcepts(shell, 1)

	if shell.conceptStore.Count() != 1 {
		t.Fatalf("Expected 1 concept before clear, got %d", shell.conceptStore.Count())
	}

	// Execute /clear_concepts command
	ctx := context.Background()
	err := shell.handleCommand(ctx, "/clear_concepts")
	if err != nil {
		t.Fatalf("handleCommand() returned unexpected error: %v", err)
	}

	// Verify: prompt message uses correct grammar for single concept
	if !strings.Contains(mock.LastPrompt(), "1 concept(s)") {
		t.Errorf("Expected prompt to mention '1 concept(s)', got: %q", mock.LastPrompt())
	}

	// Verify: concept was cleared
	if shell.conceptStore.Count() != 0 {
		t.Errorf("Expected 0 concepts after clear, got %d", shell.conceptStore.Count())
	}
}

func TestClearConcepts_ForceWithEmptyStore(t *testing.T) {
	// Setup: Create shell with empty concept store
	mock := NewMockPrompter(true)
	shell := newTestShellForClearConcepts(mock)

	// Execute /clear_concepts --force command on empty store
	ctx := context.Background()
	err := shell.handleCommand(ctx, "/clear_concepts --force")
	if err != nil {
		t.Fatalf("handleCommand() returned unexpected error: %v", err)
	}

	// Verify: confirmation was NOT requested
	if mock.CallCount != 0 {
		t.Errorf("Expected 0 confirmation calls for empty store with force, got %d", mock.CallCount)
	}
}
