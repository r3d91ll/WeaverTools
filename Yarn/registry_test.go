package yarn

import (
	"sort"
	"testing"
)

// TestNewSessionRegistry verifies the constructor creates a properly initialized registry.
func TestNewSessionRegistry(t *testing.T) {
	registry := NewSessionRegistry()

	if registry == nil {
		t.Fatal("NewSessionRegistry returned nil")
	}

	if registry.sessions == nil {
		t.Error("sessions map is nil, expected initialized map")
	}

	// Should start empty
	if len(registry.sessions) != 0 {
		t.Errorf("expected 0 sessions, got %d", len(registry.sessions))
	}
}

// TestRegister verifies session registration functionality.
func TestRegister(t *testing.T) {
	t.Run("successful registration", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "test description")

		err := registry.Register("test", session)
		if err != nil {
			t.Errorf("expected nil error, got %v", err)
		}

		// Verify session was added
		if got, ok := registry.sessions["test"]; !ok {
			t.Error("session not found in registry after registration")
		} else if got != session {
			t.Error("registered session does not match original")
		}
	})

	t.Run("duplicate name returns error", func(t *testing.T) {
		registry := NewSessionRegistry()
		session1 := NewSession("session-1", "first session")
		session2 := NewSession("session-2", "second session")

		// First registration should succeed
		if err := registry.Register("dup", session1); err != nil {
			t.Fatalf("first registration failed: %v", err)
		}

		// Second registration with same name should fail
		err := registry.Register("dup", session2)
		if err == nil {
			t.Error("expected error for duplicate name, got nil")
		}
	})

	t.Run("different names allowed", func(t *testing.T) {
		registry := NewSessionRegistry()
		session1 := NewSession("session-1", "first session")
		session2 := NewSession("session-2", "second session")

		if err := registry.Register("name1", session1); err != nil {
			t.Errorf("first registration failed: %v", err)
		}
		if err := registry.Register("name2", session2); err != nil {
			t.Errorf("second registration failed: %v", err)
		}

		if len(registry.sessions) != 2 {
			t.Errorf("expected 2 sessions, got %d", len(registry.sessions))
		}
	})

	t.Run("empty name allowed", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("empty-name-session", "description")

		err := registry.Register("", session)
		if err != nil {
			t.Errorf("expected empty name to be allowed, got error: %v", err)
		}
	})

	t.Run("nil session allowed", func(t *testing.T) {
		registry := NewSessionRegistry()

		err := registry.Register("nil-session", nil)
		if err != nil {
			t.Errorf("expected nil session to be allowed, got error: %v", err)
		}
	})
}

// TestGet verifies session retrieval functionality.
func TestGet(t *testing.T) {
	t.Run("existing session returns session and true", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "test description")
		_ = registry.Register("test", session)

		got, ok := registry.Get("test")
		if !ok {
			t.Error("expected ok to be true for existing session")
		}
		if got != session {
			t.Error("returned session does not match registered session")
		}
	})

	t.Run("non-existent session returns nil and false", func(t *testing.T) {
		registry := NewSessionRegistry()

		got, ok := registry.Get("nonexistent")
		if ok {
			t.Error("expected ok to be false for non-existent session")
		}
		if got != nil {
			t.Error("expected nil session for non-existent name")
		}
	})

	t.Run("empty name lookup", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("empty-name-session", "description")
		_ = registry.Register("", session)

		got, ok := registry.Get("")
		if !ok {
			t.Error("expected ok to be true for empty name session")
		}
		if got != session {
			t.Error("returned session does not match registered session")
		}
	})

	t.Run("multiple sessions retrieval", func(t *testing.T) {
		registry := NewSessionRegistry()
		session1 := NewSession("session-1", "first")
		session2 := NewSession("session-2", "second")
		session3 := NewSession("session-3", "third")

		_ = registry.Register("s1", session1)
		_ = registry.Register("s2", session2)
		_ = registry.Register("s3", session3)

		if got, ok := registry.Get("s1"); !ok || got != session1 {
			t.Error("failed to retrieve session s1")
		}
		if got, ok := registry.Get("s2"); !ok || got != session2 {
			t.Error("failed to retrieve session s2")
		}
		if got, ok := registry.Get("s3"); !ok || got != session3 {
			t.Error("failed to retrieve session s3")
		}
	})

	t.Run("nil session retrieval", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("nil-session", nil)

		got, ok := registry.Get("nil-session")
		if !ok {
			t.Error("expected ok to be true for nil session")
		}
		if got != nil {
			t.Error("expected nil session to be returned")
		}
	})
}

// TestList verifies listing of all session names.
func TestList(t *testing.T) {
	t.Run("empty registry returns empty slice", func(t *testing.T) {
		registry := NewSessionRegistry()

		names := registry.List()
		if names == nil {
			t.Error("expected non-nil slice for empty registry")
		}
		if len(names) != 0 {
			t.Errorf("expected 0 names, got %d", len(names))
		}
	})

	t.Run("single session", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "description")
		_ = registry.Register("only", session)

		names := registry.List()
		if len(names) != 1 {
			t.Fatalf("expected 1 name, got %d", len(names))
		}
		if names[0] != "only" {
			t.Errorf("expected 'only', got %q", names[0])
		}
	})

	t.Run("multiple sessions", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("alpha", NewSession("s1", "d1"))
		_ = registry.Register("beta", NewSession("s2", "d2"))
		_ = registry.Register("gamma", NewSession("s3", "d3"))

		names := registry.List()
		if len(names) != 3 {
			t.Fatalf("expected 3 names, got %d", len(names))
		}

		// Sort to ensure consistent comparison (map iteration order is random)
		sort.Strings(names)
		expected := []string{"alpha", "beta", "gamma"}
		for i, name := range names {
			if name != expected[i] {
				t.Errorf("expected %q at index %d, got %q", expected[i], i, name)
			}
		}
	})

	t.Run("includes empty name", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("", NewSession("empty-name", "d"))
		_ = registry.Register("named", NewSession("named", "d"))

		names := registry.List()
		if len(names) != 2 {
			t.Fatalf("expected 2 names, got %d", len(names))
		}

		hasEmpty := false
		hasNamed := false
		for _, name := range names {
			if name == "" {
				hasEmpty = true
			}
			if name == "named" {
				hasNamed = true
			}
		}
		if !hasEmpty {
			t.Error("empty name not found in list")
		}
		if !hasNamed {
			t.Error("'named' not found in list")
		}
	})

	t.Run("list returns new slice each time", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("test", NewSession("test", "d"))

		list1 := registry.List()
		list2 := registry.List()

		// Modify first list
		if len(list1) > 0 {
			list1[0] = "modified"
		}

		// Second list should be unaffected
		if len(list2) > 0 && list2[0] == "modified" {
			t.Error("List did not return independent slices")
		}
	})
}
