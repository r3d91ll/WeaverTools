// Package shell provides the interactive REPL for Weaver.
package shell

import (
	"context"
	"reflect"
	"sort"
	"testing"

	"github.com/r3d91ll/weaver/pkg/backend"
	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/wool"
)

// mockBackend implements backend.Backend for testing purposes.
type mockBackend struct {
	name string
}

func (m *mockBackend) Name() string                     { return m.name }
func (m *mockBackend) Type() backend.Type               { return backend.TypeLoom }
func (m *mockBackend) IsAvailable(ctx context.Context) bool { return true }
func (m *mockBackend) Capabilities() backend.Capabilities {
	return backend.Capabilities{}
}
func (m *mockBackend) Chat(ctx context.Context, req backend.ChatRequest) (*backend.ChatResponse, error) {
	return &backend.ChatResponse{Content: "mock response"}, nil
}
func (m *mockBackend) ChatStream(ctx context.Context, req backend.ChatRequest) (<-chan backend.StreamChunk, <-chan error) {
	return nil, nil
}

// newTestManager creates a Manager with test agents.
func newTestManager(agentNames ...string) *runtime.Manager {
	registry := backend.NewRegistry()
	registry.Register("mock", &mockBackend{name: "mock"})

	manager := runtime.NewManager(registry)
	for i, name := range agentNames {
		def := wool.Agent{
			ID:      name,
			Name:    name,
			Backend: "mock",
			Role:    wool.RoleJunior,
		}
		// Use unique ID to avoid collisions
		def.ID = name + "-" + string(rune('0'+i))
		manager.Create(def)
	}
	return manager
}

// TestNewShellCompleter tests completer creation.
func TestNewShellCompleter(t *testing.T) {
	t.Run("with nil manager", func(t *testing.T) {
		c := NewShellCompleter(nil)
		if c == nil {
			t.Fatal("expected non-nil completer")
		}
		if c.agents != nil {
			t.Error("expected nil agents field")
		}
	})

	t.Run("with manager", func(t *testing.T) {
		m := newTestManager()
		c := NewShellCompleter(m)
		if c == nil {
			t.Fatal("expected non-nil completer")
		}
		if c.agents != m {
			t.Error("expected agents field to match provided manager")
		}
	})
}

// TestDoCommandCompletion tests command completion functionality.
func TestDoCommandCompletion(t *testing.T) {
	c := NewShellCompleter(nil)

	tests := []struct {
		name     string
		line     string
		pos      int
		wantLen  int
		contains []string // Expected completions (suffixes)
	}{
		{
			name:     "slash shows all commands",
			line:     "/",
			pos:      1,
			wantLen:  1,
			contains: []string{"quit ", "exit ", "help ", "agents ", "session ", "history ", "clear ", "default ", "extract ", "analyze ", "compare ", "validate ", "concepts ", "metrics ", "clear_concepts ", "q ", "h "},
		},
		{
			name:     "slash a shows agents and analyze",
			line:     "/a",
			pos:      2,
			wantLen:  2,
			contains: []string{"gents ", "nalyze "},
		},
		{
			name:     "slash q shows quit and q",
			line:     "/q",
			pos:      2,
			wantLen:  2,
			contains: []string{"uit ", " "},
		},
		{
			name:     "slash c shows clear compare concepts clear_concepts",
			line:     "/c",
			pos:      2,
			wantLen:  2,
			contains: []string{"lear ", "ompare ", "oncepts ", "lear_concepts "},
		},
		{
			name:     "slash he shows help",
			line:     "/he",
			pos:      3,
			wantLen:  3,
			contains: []string{"lp "},
		},
		{
			name:     "slash help (exact match)",
			line:     "/help",
			pos:      5,
			wantLen:  5,
			contains: []string{" "},
		},
		{
			name:     "slash unknown returns nothing",
			line:     "/xyz",
			pos:      4,
			wantLen:  0,
			contains: nil,
		},
		{
			name:     "slash cl shows clear and clear_concepts",
			line:     "/cl",
			pos:      3,
			wantLen:  3,
			contains: []string{"ear ", "ear_concepts "},
		},
		{
			name:     "slash clear_ shows clear_concepts",
			line:     "/clear_",
			pos:      7,
			wantLen:  7,
			contains: []string{"concepts "},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, length := c.Do([]rune(tt.line), tt.pos)

			if tt.wantLen > 0 && length != tt.wantLen {
				t.Errorf("length = %d, want %d", length, tt.wantLen)
			}

			if tt.contains == nil {
				if len(results) != 0 {
					t.Errorf("expected no results, got %d", len(results))
				}
				return
			}

			// Convert results to strings for comparison
			gotStrings := make([]string, len(results))
			for i, r := range results {
				gotStrings[i] = string(r)
			}

			// Check that all expected completions are present
			for _, want := range tt.contains {
				found := false
				for _, got := range gotStrings {
					if got == want {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("missing expected completion %q in %v", want, gotStrings)
				}
			}
		})
	}
}

// TestDoAgentCompletion tests agent name completion functionality.
func TestDoAgentCompletion(t *testing.T) {
	manager := newTestManager("senior", "junior", "specialist", "supervisor")
	c := NewShellCompleter(manager)

	tests := []struct {
		name     string
		line     string
		pos      int
		wantLen  int
		contains []string
	}{
		{
			name:     "at shows all agents",
			line:     "@",
			pos:      1,
			wantLen:  1,
			contains: []string{"senior ", "junior ", "specialist ", "supervisor "},
		},
		{
			name:     "at s shows senior specialist supervisor",
			line:     "@s",
			pos:      2,
			wantLen:  2,
			contains: []string{"enior ", "pecialist ", "upervisor "},
		},
		{
			name:     "at se shows senior",
			line:     "@se",
			pos:      3,
			wantLen:  3,
			contains: []string{"nior "},
		},
		{
			name:     "at senior (exact match)",
			line:     "@senior",
			pos:      7,
			wantLen:  7,
			contains: []string{" "},
		},
		{
			name:     "at j shows junior",
			line:     "@j",
			pos:      2,
			wantLen:  2,
			contains: []string{"unior "},
		},
		{
			name:     "at unknown returns nothing",
			line:     "@xyz",
			pos:      4,
			wantLen:  0,
			contains: nil,
		},
		{
			name:     "at sp shows specialist",
			line:     "@sp",
			pos:      3,
			wantLen:  3,
			contains: []string{"ecialist "},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, length := c.Do([]rune(tt.line), tt.pos)

			if tt.wantLen > 0 && length != tt.wantLen {
				t.Errorf("length = %d, want %d", length, tt.wantLen)
			}

			if tt.contains == nil {
				if len(results) != 0 {
					t.Errorf("expected no results, got %d", len(results))
				}
				return
			}

			// Convert results to strings for comparison
			gotStrings := make([]string, len(results))
			for i, r := range results {
				gotStrings[i] = string(r)
			}

			// Check that all expected completions are present
			for _, want := range tt.contains {
				found := false
				for _, got := range gotStrings {
					if got == want {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("missing expected completion %q in %v", want, gotStrings)
				}
			}
		})
	}
}

// TestDoNilAgentManager tests agent completion with nil manager.
func TestDoNilAgentManager(t *testing.T) {
	c := NewShellCompleter(nil)

	results, length := c.Do([]rune("@se"), 3)
	if len(results) != 0 {
		t.Errorf("expected no results with nil manager, got %d", len(results))
	}
	if length != 0 {
		t.Errorf("expected length 0 with nil manager, got %d", length)
	}
}

// TestDoEdgeCases tests edge cases and boundary conditions.
func TestDoEdgeCases(t *testing.T) {
	c := NewShellCompleter(newTestManager("agent1", "agent2"))

	tests := []struct {
		name       string
		line       string
		pos        int
		wantLen    int
		wantCount  int
		wantNil    bool
	}{
		{
			name:    "empty line",
			line:    "",
			pos:     0,
			wantNil: true,
		},
		{
			name:    "cursor at position 0",
			line:    "/help",
			pos:     0,
			wantNil: true,
		},
		{
			name:    "cursor at negative position",
			line:    "/help",
			pos:     -1,
			wantNil: true,
		},
		{
			name:    "cursor beyond line length",
			line:    "/h",
			pos:     100,
			wantLen: 2, // Should clamp to line length
		},
		{
			name:    "plain text (no prefix)",
			line:    "hello",
			pos:     5,
			wantNil: true,
		},
		{
			name:    "trailing space",
			line:    "/help ",
			pos:     6,
			wantNil: true,
		},
		{
			name:    "trailing tab",
			line:    "/help\t",
			pos:     6,
			wantNil: true,
		},
		{
			name:    "multiple words - complete second command",
			line:    "@agent1 /he",
			pos:     11,
			wantLen: 3, // Length of "/he"
		},
		{
			name:    "multiple words - complete second agent",
			line:    "/help @ag",
			pos:     9,
			wantLen: 3, // Length of "@ag"
		},
		{
			name:      "mid-line cursor on command",
			line:      "/help @agent",
			pos:       5,
			wantLen:   5,
			wantCount: 1, // Just "/help"
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, length := c.Do([]rune(tt.line), tt.pos)

			if tt.wantNil {
				if results != nil {
					t.Errorf("expected nil results, got %v", results)
				}
				if length != 0 {
					t.Errorf("expected length 0, got %d", length)
				}
				return
			}

			if tt.wantLen > 0 && length != tt.wantLen {
				t.Errorf("length = %d, want %d", length, tt.wantLen)
			}

			if tt.wantCount > 0 && len(results) != tt.wantCount {
				t.Errorf("result count = %d, want %d", len(results), tt.wantCount)
			}
		})
	}
}

// TestDoMixedContent tests completion with mixed command and agent content.
func TestDoMixedContent(t *testing.T) {
	c := NewShellCompleter(newTestManager("senior", "junior"))

	tests := []struct {
		name     string
		line     string
		pos      int
		wantLen  int
		contains []string
	}{
		{
			name:     "agent then command",
			line:     "@senior /he",
			pos:      11,
			wantLen:  3,
			contains: []string{"lp "},
		},
		{
			name:     "command then agent",
			line:     "/help @se",
			pos:      9,
			wantLen:  3,
			contains: []string{"nior "},
		},
		{
			name:     "text then command",
			line:     "some text /ag",
			pos:      13,
			wantLen:  3,
			contains: []string{"ents "},
		},
		{
			name:     "text then agent",
			line:     "some text @ju",
			pos:      13,
			wantLen:  3,
			contains: []string{"nior "},
		},
		{
			name:     "multiple commands complete last",
			line:     "/help /ag",
			pos:      9,
			wantLen:  3,
			contains: []string{"ents "},
		},
		{
			name:     "multiple agents complete last",
			line:     "@senior @ju",
			pos:      11,
			wantLen:  3,
			contains: []string{"nior "},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, length := c.Do([]rune(tt.line), tt.pos)

			if length != tt.wantLen {
				t.Errorf("length = %d, want %d", length, tt.wantLen)
			}

			gotStrings := make([]string, len(results))
			for i, r := range results {
				gotStrings[i] = string(r)
			}

			for _, want := range tt.contains {
				found := false
				for _, got := range gotStrings {
					if got == want {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("missing expected completion %q in %v", want, gotStrings)
				}
			}
		})
	}
}

// TestFindWordStart tests the word boundary detection helper.
func TestFindWordStart(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"", 0},
		{"hello", 0},
		{"hello world", 6},
		{"hello\tworld", 6},
		{"one two three", 8},
		{" leading", 1},
		{"\tleading", 1},
		{"mixed\t and spaces", 9},
		{"no-spaces-here", 0},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := findWordStart(tt.input)
			if got != tt.want {
				t.Errorf("findWordStart(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

// TestCommandsListCompleteness verifies all commands are in the list.
func TestCommandsListCompleteness(t *testing.T) {
	expectedCommands := []string{
		"quit", "exit", "q", "help", "h",
		"agents", "session", "history", "clear", "default",
		"extract", "analyze", "compare", "validate",
		"concepts", "metrics", "clear_concepts",
	}

	// Create a set of actual commands
	actualSet := make(map[string]bool)
	for _, cmd := range commands {
		actualSet[cmd] = true
	}

	// Verify all expected commands are present
	for _, expected := range expectedCommands {
		if !actualSet[expected] {
			t.Errorf("missing command: %s", expected)
		}
	}

	// Verify count matches
	if len(commands) != len(expectedCommands) {
		t.Errorf("command count mismatch: got %d, want %d", len(commands), len(expectedCommands))
	}
}

// TestDynamicAgentList verifies agents are fetched dynamically.
func TestDynamicAgentList(t *testing.T) {
	registry := backend.NewRegistry()
	registry.Register("mock", &mockBackend{name: "mock"})
	manager := runtime.NewManager(registry)

	c := NewShellCompleter(manager)

	// Initially no agents
	results, _ := c.Do([]rune("@"), 1)
	if len(results) != 0 {
		t.Errorf("expected no completions with empty manager, got %d", len(results))
	}

	// Add an agent
	manager.Create(wool.Agent{
		ID:      "agent1",
		Name:    "agent1",
		Backend: "mock",
		Role:    wool.RoleJunior,
	})

	// Should now see the agent
	results, _ = c.Do([]rune("@"), 1)
	if len(results) != 1 {
		t.Errorf("expected 1 completion after adding agent, got %d", len(results))
	}

	// Add another agent
	manager.Create(wool.Agent{
		ID:      "agent2",
		Name:    "agent2",
		Backend: "mock",
		Role:    wool.RoleJunior,
	})

	// Should see both agents
	results, _ = c.Do([]rune("@"), 1)
	if len(results) != 2 {
		t.Errorf("expected 2 completions after adding second agent, got %d", len(results))
	}
}

// TestCompletionSuffix verifies completions include trailing space.
func TestCompletionSuffix(t *testing.T) {
	c := NewShellCompleter(newTestManager("agent"))

	// Test command completion includes space
	results, _ := c.Do([]rune("/help"), 5)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if string(results[0]) != " " {
		t.Errorf("expected completion to be ' ' (just space), got %q", string(results[0]))
	}

	// Test agent completion includes space
	results, _ = c.Do([]rune("@agent"), 6)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if string(results[0]) != " " {
		t.Errorf("expected completion to be ' ' (just space), got %q", string(results[0]))
	}
}

// TestReadlineAutoCompleterInterface verifies ShellCompleter implements the interface.
func TestReadlineAutoCompleterInterface(t *testing.T) {
	// This is a compile-time check that exists in completer.go,
	// but we include a runtime check here for documentation purposes.
	c := NewShellCompleter(nil)

	// Test that Do method exists and has correct signature
	results, length := c.Do([]rune("/h"), 2)
	if results == nil {
		t.Error("Do() should return non-nil slice for valid input")
	}
	if length != 2 {
		t.Errorf("Do() length should be 2, got %d", length)
	}
}

// TestCaseSensitivity verifies completion is case-sensitive.
func TestCaseSensitivity(t *testing.T) {
	c := NewShellCompleter(newTestManager("Senior", "junior"))

	// Lowercase prefix should match lowercase agent
	results, _ := c.Do([]rune("@j"), 2)
	found := false
	for _, r := range results {
		if string(r) == "unior " {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected @j to match 'junior'")
	}

	// Uppercase prefix should match uppercase agent
	results, _ = c.Do([]rune("@S"), 2)
	found = false
	for _, r := range results {
		if string(r) == "enior " {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected @S to match 'Senior'")
	}
}

// TestSortedCompletions tests that results are returned in a consistent order.
func TestSortedCompletions(t *testing.T) {
	// Note: The current implementation doesn't guarantee sorted order.
	// This test documents the behavior and can be updated if sorting is added.
	c := NewShellCompleter(nil)

	results1, _ := c.Do([]rune("/c"), 2)
	results2, _ := c.Do([]rune("/c"), 2)

	// Convert to strings for comparison
	strs1 := make([]string, len(results1))
	strs2 := make([]string, len(results2))
	for i, r := range results1 {
		strs1[i] = string(r)
	}
	for i, r := range results2 {
		strs2[i] = string(r)
	}

	// Sort both for comparison (since order may vary)
	sort.Strings(strs1)
	sort.Strings(strs2)

	if !reflect.DeepEqual(strs1, strs2) {
		t.Errorf("completions should be consistent across calls: %v vs %v", strs1, strs2)
	}
}
