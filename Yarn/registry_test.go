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

// TestStatus verifies the Status method functionality.
func TestStatus(t *testing.T) {
	t.Run("empty registry returns empty map", func(t *testing.T) {
		registry := NewSessionRegistry()

		status := registry.Status()
		if status == nil {
			t.Error("expected non-nil map for empty registry")
		}
		if len(status) != 0 {
			t.Errorf("expected 0 statuses, got %d", len(status))
		}
	})

	t.Run("single active session", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "description")
		_ = registry.Register("test", session)

		status := registry.Status()
		if len(status) != 1 {
			t.Fatalf("expected 1 status, got %d", len(status))
		}

		s, ok := status["test"]
		if !ok {
			t.Fatal("expected 'test' key in status map")
		}
		if s.Name != "test-session" {
			t.Errorf("expected Name 'test-session', got %q", s.Name)
		}
		if s.ID != session.ID {
			t.Errorf("expected ID %q, got %q", session.ID, s.ID)
		}
		if !s.IsActive {
			t.Error("expected IsActive to be true for non-ended session")
		}
		if s.EndedAt != nil {
			t.Error("expected EndedAt to be nil for active session")
		}
	})

	t.Run("ended session shows inactive", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("ended-session", "description")
		session.End() // Mark as ended
		_ = registry.Register("ended", session)

		status := registry.Status()
		s, ok := status["ended"]
		if !ok {
			t.Fatal("expected 'ended' key in status map")
		}
		if s.IsActive {
			t.Error("expected IsActive to be false for ended session")
		}
		if s.EndedAt == nil {
			t.Error("expected EndedAt to be set for ended session")
		}
	})

	t.Run("multiple sessions", func(t *testing.T) {
		registry := NewSessionRegistry()
		session1 := NewSession("session-1", "first")
		session2 := NewSession("session-2", "second")
		session3 := NewSession("session-3", "third")
		session2.End() // Mark second session as ended

		_ = registry.Register("s1", session1)
		_ = registry.Register("s2", session2)
		_ = registry.Register("s3", session3)

		status := registry.Status()
		if len(status) != 3 {
			t.Fatalf("expected 3 statuses, got %d", len(status))
		}

		// Verify each session status
		if !status["s1"].IsActive {
			t.Error("expected s1 to be active")
		}
		if status["s2"].IsActive {
			t.Error("expected s2 to be inactive (ended)")
		}
		if !status["s3"].IsActive {
			t.Error("expected s3 to be active")
		}
	})

	t.Run("status includes correct stats", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "description")
		_ = registry.Register("test", session)

		status := registry.Status()
		s := status["test"]

		// Initial session should have zero stats
		if s.Stats.ConversationCount != 0 {
			t.Errorf("expected 0 conversations, got %d", s.Stats.ConversationCount)
		}
		if s.Stats.MeasurementCount != 0 {
			t.Errorf("expected 0 measurements, got %d", s.Stats.MeasurementCount)
		}
	})
}

// TestActive verifies the Active method functionality.
func TestActive(t *testing.T) {
	t.Run("empty registry returns empty slice", func(t *testing.T) {
		registry := NewSessionRegistry()

		active := registry.Active()
		if active == nil {
			t.Error("expected non-nil slice for empty registry")
		}
		if len(active) != 0 {
			t.Errorf("expected 0 active sessions, got %d", len(active))
		}
	})

	t.Run("single active session", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("active-session", "description")
		_ = registry.Register("test", session)

		active := registry.Active()
		if len(active) != 1 {
			t.Fatalf("expected 1 active session, got %d", len(active))
		}
		if active[0] != session {
			t.Error("returned session does not match registered session")
		}
	})

	t.Run("ended session not included", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("ended-session", "description")
		session.End() // Mark as ended
		_ = registry.Register("test", session)

		active := registry.Active()
		if len(active) != 0 {
			t.Errorf("expected 0 active sessions (session ended), got %d", len(active))
		}
	})

	t.Run("all sessions ended returns empty", func(t *testing.T) {
		registry := NewSessionRegistry()
		session1 := NewSession("session-1", "first")
		session2 := NewSession("session-2", "second")
		session1.End()
		session2.End()

		_ = registry.Register("s1", session1)
		_ = registry.Register("s2", session2)

		active := registry.Active()
		if len(active) != 0 {
			t.Errorf("expected 0 active sessions (all ended), got %d", len(active))
		}
	})

	t.Run("mix of active and ended sessions", func(t *testing.T) {
		registry := NewSessionRegistry()
		active1 := NewSession("active-1", "active session 1")
		ended1 := NewSession("ended-1", "ended session 1")
		active2 := NewSession("active-2", "active session 2")
		ended2 := NewSession("ended-2", "ended session 2")
		ended1.End()
		ended2.End()

		_ = registry.Register("a1", active1)
		_ = registry.Register("e1", ended1)
		_ = registry.Register("a2", active2)
		_ = registry.Register("e2", ended2)

		active := registry.Active()
		if len(active) != 2 {
			t.Fatalf("expected 2 active sessions, got %d", len(active))
		}

		// Verify only active sessions are returned
		foundActive1 := false
		foundActive2 := false
		for _, s := range active {
			if s == active1 {
				foundActive1 = true
			}
			if s == active2 {
				foundActive2 = true
			}
			if s == ended1 || s == ended2 {
				t.Error("ended session found in active list")
			}
		}
		if !foundActive1 {
			t.Error("active1 not found in active list")
		}
		if !foundActive2 {
			t.Error("active2 not found in active list")
		}
	})

	t.Run("active returns new slice each time", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("test", NewSession("test", "d"))

		active1 := registry.Active()
		active2 := registry.Active()

		// Modify first slice
		if len(active1) > 0 {
			active1[0] = nil
		}

		// Second slice should be unaffected
		if len(active2) > 0 && active2[0] == nil {
			t.Error("Active did not return independent slices")
		}
	})
}

// TestUnregister verifies the Unregister method functionality.
func TestUnregister(t *testing.T) {
	t.Run("successful unregistration", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "description")
		_ = registry.Register("test", session)

		err := registry.Unregister("test")
		if err != nil {
			t.Errorf("expected nil error, got %v", err)
		}

		// Verify session was removed
		if _, ok := registry.Get("test"); ok {
			t.Error("session still found after unregistration")
		}
	})

	t.Run("non-existent session returns error", func(t *testing.T) {
		registry := NewSessionRegistry()

		err := registry.Unregister("nonexistent")
		if err == nil {
			t.Error("expected error for non-existent session, got nil")
		}
	})

	t.Run("unregister reduces count", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("s1", NewSession("s1", "d1"))
		_ = registry.Register("s2", NewSession("s2", "d2"))
		_ = registry.Register("s3", NewSession("s3", "d3"))

		if registry.Count() != 3 {
			t.Fatalf("expected 3 sessions before unregister, got %d", registry.Count())
		}

		_ = registry.Unregister("s2")

		if registry.Count() != 2 {
			t.Errorf("expected 2 sessions after unregister, got %d", registry.Count())
		}

		// Verify remaining sessions
		if _, ok := registry.Get("s1"); !ok {
			t.Error("s1 should still exist")
		}
		if _, ok := registry.Get("s2"); ok {
			t.Error("s2 should not exist")
		}
		if _, ok := registry.Get("s3"); !ok {
			t.Error("s3 should still exist")
		}
	})

	t.Run("unregister empty name", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("empty-name-session", "description")
		_ = registry.Register("", session)

		err := registry.Unregister("")
		if err != nil {
			t.Errorf("expected nil error for empty name unregister, got %v", err)
		}

		if _, ok := registry.Get(""); ok {
			t.Error("empty name session still found after unregistration")
		}
	})

	t.Run("can re-register after unregister", func(t *testing.T) {
		registry := NewSessionRegistry()
		session1 := NewSession("session-1", "first")
		_ = registry.Register("reuse", session1)

		_ = registry.Unregister("reuse")

		session2 := NewSession("session-2", "second")
		err := registry.Register("reuse", session2)
		if err != nil {
			t.Errorf("expected nil error for re-registration, got %v", err)
		}

		got, ok := registry.Get("reuse")
		if !ok {
			t.Error("re-registered session not found")
		}
		if got != session2 {
			t.Error("wrong session returned after re-registration")
		}
	})

	t.Run("unregister nil session", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("nil-session", nil)

		err := registry.Unregister("nil-session")
		if err != nil {
			t.Errorf("expected nil error for nil session unregister, got %v", err)
		}
	})
}
