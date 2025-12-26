// Package backend tests for backend registry and structured error handling.
package backend

import (
	"context"
	"strings"
	"testing"
	"time"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
)

// mockBackend is a simple backend implementation for testing.
type mockBackend struct {
	name         string
	available    bool
	backendType  Type
	capabilities Capabilities
}

func (m *mockBackend) Name() string                 { return m.name }
func (m *mockBackend) Type() Type                   { return m.backendType }
func (m *mockBackend) IsAvailable(ctx context.Context) bool { return m.available }
func (m *mockBackend) Capabilities() Capabilities   { return m.capabilities }
func (m *mockBackend) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	return &ChatResponse{Content: "mock response"}, nil
}
func (m *mockBackend) ChatStream(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent)
	go func() {
		ch <- StreamEvent{Content: "mock", Done: true}
		close(ch)
	}()
	return ch, nil
}

// -----------------------------------------------------------------------------
// Registry Basic Tests
// -----------------------------------------------------------------------------

func TestNewRegistry(t *testing.T) {
	r := NewRegistry()
	if r == nil {
		t.Fatal("expected non-nil registry")
	}
	if r.backends == nil {
		t.Error("expected backends map to be initialized")
	}
}

func TestRegistry_Register(t *testing.T) {
	r := NewRegistry()
	mock := &mockBackend{name: "test", available: true}

	if err := r.Register("test", mock); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify backend was registered
	backend, ok := r.Get("test")
	if !ok {
		t.Fatal("expected backend to be registered")
	}
	if backend.Name() != "test" {
		t.Errorf("expected name 'test', got %q", backend.Name())
	}
}

func TestRegistry_Register_Duplicate(t *testing.T) {
	r := NewRegistry()
	mock1 := &mockBackend{name: "test1"}
	mock2 := &mockBackend{name: "test2"}

	if err := r.Register("test", mock1); err != nil {
		t.Fatalf("first registration failed: %v", err)
	}

	// Second registration with same name should fail
	err := r.Register("test", mock2)
	if err == nil {
		t.Fatal("expected error for duplicate registration")
	}

	// Should be a WeaverError
	werr, ok := err.(*werrors.WeaverError)
	if !ok {
		t.Fatalf("expected *werrors.WeaverError, got %T", err)
	}

	if werr.Code != werrors.ErrBackendAlreadyRegistered {
		t.Errorf("expected code %q, got %q", werrors.ErrBackendAlreadyRegistered, werr.Code)
	}

	if werr.Category != werrors.CategoryBackend {
		t.Errorf("expected category %v, got %v", werrors.CategoryBackend, werr.Category)
	}

	// Should have context about the duplicate
	if werr.Context["backend"] != "test" {
		t.Errorf("expected backend context 'test', got %q", werr.Context["backend"])
	}

	// Should have suggestions
	if len(werr.Suggestions) == 0 {
		t.Error("expected suggestions to be attached")
	}

	// Should mention choosing different name
	foundSuggestion := false
	for _, s := range werr.Suggestions {
		if strings.Contains(s, "different name") {
			foundSuggestion = true
			break
		}
	}
	if !foundSuggestion {
		t.Error("expected suggestion about choosing different name")
	}
}

// -----------------------------------------------------------------------------
// Registry Get Tests
// -----------------------------------------------------------------------------

func TestRegistry_Get_Exists(t *testing.T) {
	r := NewRegistry()
	mock := &mockBackend{name: "test", available: true}
	_ = r.Register("test", mock)

	backend, ok := r.Get("test")
	if !ok {
		t.Fatal("expected backend to be found")
	}
	if backend != mock {
		t.Error("expected same backend instance")
	}
}

func TestRegistry_Get_NotExists(t *testing.T) {
	r := NewRegistry()
	mock := &mockBackend{name: "test", available: true}
	_ = r.Register("test", mock)

	_, ok := r.Get("nonexistent")
	if ok {
		t.Error("expected backend to not be found")
	}
}

func TestRegistry_GetWithError_Exists(t *testing.T) {
	r := NewRegistry()
	mock := &mockBackend{name: "test", available: true}
	_ = r.Register("test", mock)

	backend, err := r.GetWithError("test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if backend != mock {
		t.Error("expected same backend instance")
	}
}

func TestRegistry_GetWithError_NotExists(t *testing.T) {
	r := NewRegistry()
	mock := &mockBackend{name: "test", available: true}
	_ = r.Register("test", mock)

	_, err := r.GetWithError("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent backend")
	}

	// Should be a WeaverError
	werr, ok := err.(*werrors.WeaverError)
	if !ok {
		t.Fatalf("expected *werrors.WeaverError, got %T", err)
	}

	if werr.Code != werrors.ErrBackendNotFound {
		t.Errorf("expected code %q, got %q", werrors.ErrBackendNotFound, werr.Code)
	}

	if werr.Category != werrors.CategoryBackend {
		t.Errorf("expected category %v, got %v", werrors.CategoryBackend, werr.Category)
	}

	// Should have context about the request
	if werr.Context["backend"] != "nonexistent" {
		t.Errorf("expected backend context 'nonexistent', got %q", werr.Context["backend"])
	}

	// Should list available backends
	if !strings.Contains(werr.Context["available_backends"], "test") {
		t.Error("expected available_backends to include 'test'")
	}

	// Should have suggestions
	if len(werr.Suggestions) == 0 {
		t.Error("expected suggestions to be attached")
	}
}

func TestRegistry_GetWithError_SuggestsSimilarName(t *testing.T) {
	r := NewRegistry()
	_ = r.Register("claudecode", &mockBackend{name: "claudecode"})
	_ = r.Register("loom", &mockBackend{name: "loom"})

	// Try with wrong case
	_, err := r.GetWithError("ClaudeCode")
	if err == nil {
		t.Fatal("expected error")
	}

	werr := err.(*werrors.WeaverError)

	// Should suggest correct name
	foundSuggestion := false
	for _, s := range werr.Suggestions {
		if strings.Contains(s, "claudecode") && strings.Contains(strings.ToLower(s), "case") {
			foundSuggestion = true
			break
		}
	}
	if !foundSuggestion {
		t.Error("expected suggestion about case mismatch")
	}
}

func TestRegistry_GetWithError_SuggestsPartialMatch(t *testing.T) {
	r := NewRegistry()
	_ = r.Register("claudecode", &mockBackend{name: "claudecode"})
	_ = r.Register("loom", &mockBackend{name: "loom"})

	// Try with partial name
	_, err := r.GetWithError("claude")
	if err == nil {
		t.Fatal("expected error")
	}

	werr := err.(*werrors.WeaverError)

	// Should suggest similar backend
	foundSuggestion := false
	for _, s := range werr.Suggestions {
		if strings.Contains(s, "claudecode") && strings.Contains(strings.ToLower(s), "mean") {
			foundSuggestion = true
			break
		}
	}
	if !foundSuggestion {
		t.Error("expected 'Did you mean' suggestion for partial match")
	}
}

func TestRegistry_GetWithError_EmptyRegistry(t *testing.T) {
	r := NewRegistry()

	_, err := r.GetWithError("anybackend")
	if err == nil {
		t.Fatal("expected error for empty registry")
	}

	werr := err.(*werrors.WeaverError)

	// Should indicate no backends
	if werr.Context["available_backends"] != "(none)" {
		t.Errorf("expected '(none)' for empty registry, got %q", werr.Context["available_backends"])
	}

	// Should suggest registering backend
	foundSuggestion := false
	for _, s := range werr.Suggestions {
		if strings.Contains(s, "Register") {
			foundSuggestion = true
			break
		}
	}
	if !foundSuggestion {
		t.Error("expected suggestion about registering backend")
	}
}

// -----------------------------------------------------------------------------
// Registry List Tests
// -----------------------------------------------------------------------------

func TestRegistry_List(t *testing.T) {
	r := NewRegistry()
	_ = r.Register("backend1", &mockBackend{name: "backend1"})
	_ = r.Register("backend2", &mockBackend{name: "backend2"})
	_ = r.Register("backend3", &mockBackend{name: "backend3"})

	list := r.List()
	if len(list) != 3 {
		t.Errorf("expected 3 backends, got %d", len(list))
	}

	// Verify all names are present
	found := make(map[string]bool)
	for _, name := range list {
		found[name] = true
	}

	if !found["backend1"] || !found["backend2"] || !found["backend3"] {
		t.Error("expected all backends to be in list")
	}
}

func TestRegistry_List_Empty(t *testing.T) {
	r := NewRegistry()
	list := r.List()
	if len(list) != 0 {
		t.Errorf("expected empty list, got %d items", len(list))
	}
}

// -----------------------------------------------------------------------------
// Registry Available Tests
// -----------------------------------------------------------------------------

func TestRegistry_Available(t *testing.T) {
	r := NewRegistry()
	_ = r.Register("available1", &mockBackend{name: "available1", available: true})
	_ = r.Register("unavailable", &mockBackend{name: "unavailable", available: false})
	_ = r.Register("available2", &mockBackend{name: "available2", available: true})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	available := r.Available(ctx)
	if len(available) != 2 {
		t.Errorf("expected 2 available backends, got %d", len(available))
	}

	// Verify the right backends are returned
	for _, b := range available {
		if !b.IsAvailable(ctx) {
			t.Error("returned backend should be available")
		}
	}
}

func TestRegistry_Available_NoneAvailable(t *testing.T) {
	r := NewRegistry()
	_ = r.Register("unavailable1", &mockBackend{name: "unavailable1", available: false})
	_ = r.Register("unavailable2", &mockBackend{name: "unavailable2", available: false})

	ctx := context.Background()
	available := r.Available(ctx)
	if len(available) != 0 {
		t.Errorf("expected 0 available backends, got %d", len(available))
	}
}

// -----------------------------------------------------------------------------
// Registry Status Tests
// -----------------------------------------------------------------------------

func TestRegistry_Status(t *testing.T) {
	r := NewRegistry()
	_ = r.Register("backend1", &mockBackend{
		name:         "backend1",
		available:    true,
		backendType:  TypeLocal,
		capabilities: Capabilities{Streaming: true, Tools: true},
	})
	_ = r.Register("backend2", &mockBackend{
		name:         "backend2",
		available:    false,
		backendType:  TypeRemote,
		capabilities: Capabilities{Streaming: false, Tools: false},
	})

	ctx := context.Background()
	status := r.Status(ctx)

	if len(status) != 2 {
		t.Errorf("expected 2 status entries, got %d", len(status))
	}

	// Check backend1 status
	if s, ok := status["backend1"]; ok {
		if s.Name != "backend1" {
			t.Errorf("expected name 'backend1', got %q", s.Name)
		}
		if !s.Available {
			t.Error("expected backend1 to be available")
		}
		if s.Type != TypeLocal {
			t.Errorf("expected TypeLocal, got %v", s.Type)
		}
		if !s.Capabilities.Streaming {
			t.Error("expected streaming capability")
		}
	} else {
		t.Error("backend1 not in status")
	}

	// Check backend2 status
	if s, ok := status["backend2"]; ok {
		if s.Available {
			t.Error("expected backend2 to be unavailable")
		}
	} else {
		t.Error("backend2 not in status")
	}
}

// -----------------------------------------------------------------------------
// Default Registry Tests
// -----------------------------------------------------------------------------

func TestDefault(t *testing.T) {
	r := Default("http://localhost:8080")

	if r == nil {
		t.Fatal("expected non-nil registry")
	}

	// Should have claudecode
	if _, ok := r.Get("claudecode"); !ok {
		t.Error("expected claudecode backend to be registered")
	}

	// Should have loom
	if _, ok := r.Get("loom"); !ok {
		t.Error("expected loom backend to be registered")
	}

	// Should not have others
	if len(r.List()) != 2 {
		t.Errorf("expected exactly 2 backends, got %d", len(r.List()))
	}
}

// -----------------------------------------------------------------------------
// Concurrency Tests
// -----------------------------------------------------------------------------

func TestRegistry_ConcurrentAccess(t *testing.T) {
	r := NewRegistry()

	// Pre-register some backends
	for i := 0; i < 10; i++ {
		name := strings.Repeat("a", i+1) // "a", "aa", "aaa", etc.
		_ = r.Register(name, &mockBackend{name: name, available: true})
	}

	done := make(chan bool)
	ctx := context.Background()

	// Concurrent readers
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				_ = r.List()
				_, _ = r.Get("a")
				_, _ = r.GetWithError("aa")
				_ = r.Available(ctx)
				_ = r.Status(ctx)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
}

// -----------------------------------------------------------------------------
// SuggestSimilarBackends Tests
// -----------------------------------------------------------------------------

func TestSuggestSimilarBackends_CaseMismatch(t *testing.T) {
	available := []string{"claudecode", "loom"}

	suggestions := suggestSimilarBackends("ClaudeCode", available)
	if len(suggestions) == 0 {
		t.Error("expected case mismatch suggestion")
	}

	found := false
	for _, s := range suggestions {
		if strings.Contains(s, "claudecode") && strings.Contains(s, "case") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected suggestion about case mismatch")
	}
}

func TestSuggestSimilarBackends_PartialMatch(t *testing.T) {
	available := []string{"claudecode", "loom"}

	suggestions := suggestSimilarBackends("claude", available)
	if len(suggestions) == 0 {
		t.Error("expected partial match suggestion")
	}

	found := false
	for _, s := range suggestions {
		if strings.Contains(s, "claudecode") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected suggestion for claudecode")
	}
}

func TestSuggestSimilarBackends_NoMatch(t *testing.T) {
	available := []string{"claudecode", "loom"}

	suggestions := suggestSimilarBackends("xyz", available)
	if len(suggestions) != 0 {
		t.Errorf("expected no suggestions for completely different name, got %v", suggestions)
	}
}

func TestSuggestSimilarBackends_ExactMatch(t *testing.T) {
	available := []string{"claudecode", "loom"}

	// Exact match shouldn't produce suggestions (no correction needed)
	suggestions := suggestSimilarBackends("claudecode", available)
	if len(suggestions) != 0 {
		t.Errorf("expected no suggestions for exact match, got %v", suggestions)
	}
}

// -----------------------------------------------------------------------------
// Helper Function Tests
// -----------------------------------------------------------------------------

func TestListBackendNamesLocked_Sorted(t *testing.T) {
	r := NewRegistry()
	_ = r.Register("zebra", &mockBackend{name: "zebra"})
	_ = r.Register("alpha", &mockBackend{name: "alpha"})
	_ = r.Register("middle", &mockBackend{name: "middle"})

	// Access via GetWithError to trigger listBackendNamesLocked
	_, err := r.GetWithError("nonexistent")
	werr := err.(*werrors.WeaverError)

	// The list should be sorted
	expected := "alpha, middle, zebra"
	if werr.Context["available_backends"] != expected {
		t.Errorf("expected sorted list %q, got %q", expected, werr.Context["available_backends"])
	}
}
