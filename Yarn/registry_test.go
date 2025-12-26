package yarn

import (
	"fmt"
	"sort"
	"sync"
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

// TestCreate verifies the Create method functionality.
func TestCreate(t *testing.T) {
	t.Run("successful creation", func(t *testing.T) {
		registry := NewSessionRegistry()

		session, err := registry.Create("test", "test description")
		if err != nil {
			t.Errorf("expected nil error, got %v", err)
		}
		if session == nil {
			t.Fatal("expected non-nil session")
		}

		// Verify session properties
		if session.Name != "test" {
			t.Errorf("expected Name 'test', got %q", session.Name)
		}
		if session.Description != "test description" {
			t.Errorf("expected Description 'test description', got %q", session.Description)
		}

		// Verify session was registered
		got, ok := registry.Get("test")
		if !ok {
			t.Error("session not found in registry after creation")
		}
		if got != session {
			t.Error("registered session does not match created session")
		}
	})

	t.Run("duplicate name returns error", func(t *testing.T) {
		registry := NewSessionRegistry()

		// First creation should succeed
		_, err := registry.Create("dup", "first session")
		if err != nil {
			t.Fatalf("first creation failed: %v", err)
		}

		// Second creation with same name should fail
		session, err := registry.Create("dup", "second session")
		if err == nil {
			t.Error("expected error for duplicate name, got nil")
		}
		if session != nil {
			t.Error("expected nil session on error")
		}
	})

	t.Run("different names allowed", func(t *testing.T) {
		registry := NewSessionRegistry()

		s1, err := registry.Create("name1", "first session")
		if err != nil {
			t.Errorf("first creation failed: %v", err)
		}
		s2, err := registry.Create("name2", "second session")
		if err != nil {
			t.Errorf("second creation failed: %v", err)
		}

		if registry.Count() != 2 {
			t.Errorf("expected 2 sessions, got %d", registry.Count())
		}
		if s1 == s2 {
			t.Error("expected different session instances")
		}
	})

	t.Run("empty name returns error", func(t *testing.T) {
		registry := NewSessionRegistry()

		session, err := registry.Create("", "empty name session")
		if err != ErrEmptySessionName {
			t.Errorf("expected ErrEmptySessionName, got %v", err)
		}
		if session != nil {
			t.Error("expected nil session for empty name")
		}

		// Verify nothing was registered
		if registry.Count() != 0 {
			t.Errorf("expected 0 sessions, got %d", registry.Count())
		}
	})

	t.Run("whitespace-only name returns error", func(t *testing.T) {
		registry := NewSessionRegistry()

		session, err := registry.Create("   ", "whitespace name session")
		if err != ErrEmptySessionName {
			t.Errorf("expected ErrEmptySessionName for whitespace name, got %v", err)
		}
		if session != nil {
			t.Error("expected nil session for whitespace name")
		}

		// Verify nothing was registered
		if registry.Count() != 0 {
			t.Errorf("expected 0 sessions, got %d", registry.Count())
		}
	})

	t.Run("created session is active", func(t *testing.T) {
		registry := NewSessionRegistry()

		session, _ := registry.Create("test", "description")

		if session.EndedAt != nil {
			t.Error("expected new session to have nil EndedAt")
		}

		// Verify shows up in Active list
		active := registry.Active()
		if len(active) != 1 {
			t.Fatalf("expected 1 active session, got %d", len(active))
		}
		if active[0] != session {
			t.Error("created session not found in active list")
		}
	})

	t.Run("create increments count", func(t *testing.T) {
		registry := NewSessionRegistry()

		if registry.Count() != 0 {
			t.Fatalf("expected 0 count initially, got %d", registry.Count())
		}

		_, _ = registry.Create("s1", "d1")
		if registry.Count() != 1 {
			t.Errorf("expected 1 after first create, got %d", registry.Count())
		}

		_, _ = registry.Create("s2", "d2")
		if registry.Count() != 2 {
			t.Errorf("expected 2 after second create, got %d", registry.Count())
		}
	})
}

// TestGetOrCreate verifies the GetOrCreate method functionality.
func TestGetOrCreate(t *testing.T) {
	t.Run("creates new session when not found", func(t *testing.T) {
		registry := NewSessionRegistry()

		session, created := registry.GetOrCreate("new", "new session")
		if !created {
			t.Error("expected created to be true for new session")
		}
		if session == nil {
			t.Fatal("expected non-nil session")
		}

		// Verify session properties
		if session.Name != "new" {
			t.Errorf("expected Name 'new', got %q", session.Name)
		}
		if session.Description != "new session" {
			t.Errorf("expected Description 'new session', got %q", session.Description)
		}
	})

	t.Run("returns existing session when found", func(t *testing.T) {
		registry := NewSessionRegistry()

		// Create first
		original, _ := registry.Create("existing", "original description")

		// GetOrCreate should return existing
		session, created := registry.GetOrCreate("existing", "different description")
		if created {
			t.Error("expected created to be false for existing session")
		}
		if session != original {
			t.Error("expected to return the same session instance")
		}
		// Original description should be preserved
		if session.Description != "original description" {
			t.Errorf("expected original description, got %q", session.Description)
		}
	})

	t.Run("registers created session", func(t *testing.T) {
		registry := NewSessionRegistry()

		session, _ := registry.GetOrCreate("test", "description")

		// Should be retrievable via Get
		got, ok := registry.Get("test")
		if !ok {
			t.Error("GetOrCreate did not register the session")
		}
		if got != session {
			t.Error("retrieved session does not match created session")
		}
	})

	t.Run("empty name returns nil", func(t *testing.T) {
		registry := NewSessionRegistry()

		s1, created1 := registry.GetOrCreate("", "first empty")
		if s1 != nil {
			t.Error("expected nil session for empty name")
		}
		if created1 {
			t.Error("expected created=false for empty name")
		}

		// Verify nothing was registered
		if registry.Count() != 0 {
			t.Errorf("expected 0 sessions, got %d", registry.Count())
		}
	})

	t.Run("whitespace-only name returns nil", func(t *testing.T) {
		registry := NewSessionRegistry()

		session, created := registry.GetOrCreate("   ", "whitespace name")
		if session != nil {
			t.Error("expected nil session for whitespace-only name")
		}
		if created {
			t.Error("expected created=false for whitespace-only name")
		}

		// Verify nothing was registered
		if registry.Count() != 0 {
			t.Errorf("expected 0 sessions, got %d", registry.Count())
		}
	})

	t.Run("multiple GetOrCreate calls", func(t *testing.T) {
		registry := NewSessionRegistry()

		// First call creates
		s1, created1 := registry.GetOrCreate("test", "description")
		if !created1 {
			t.Error("expected first call to create")
		}

		// Second call returns existing
		s2, created2 := registry.GetOrCreate("test", "description")
		if created2 {
			t.Error("expected second call to return existing")
		}

		// Third call also returns existing
		s3, created3 := registry.GetOrCreate("test", "description")
		if created3 {
			t.Error("expected third call to return existing")
		}

		// All should be the same instance
		if s1 != s2 || s2 != s3 {
			t.Error("expected all calls to return same session instance")
		}

		// Count should be 1
		if registry.Count() != 1 {
			t.Errorf("expected count of 1, got %d", registry.Count())
		}
	})

	t.Run("different names create different sessions", func(t *testing.T) {
		registry := NewSessionRegistry()

		s1, created1 := registry.GetOrCreate("name1", "first")
		s2, created2 := registry.GetOrCreate("name2", "second")

		if !created1 || !created2 {
			t.Error("expected both calls to create new sessions")
		}
		if s1 == s2 {
			t.Error("expected different session instances for different names")
		}
		if registry.Count() != 2 {
			t.Errorf("expected count of 2, got %d", registry.Count())
		}
	})

	t.Run("created session is active", func(t *testing.T) {
		registry := NewSessionRegistry()

		session, _ := registry.GetOrCreate("test", "description")

		if session.EndedAt != nil {
			t.Error("expected new session to have nil EndedAt")
		}

		active := registry.Active()
		if len(active) != 1 {
			t.Fatalf("expected 1 active session, got %d", len(active))
		}
		if active[0] != session {
			t.Error("created session not found in active list")
		}
	})
}

// ============================================================================
// Concurrency Tests
// ============================================================================

// TestConcurrentRegister verifies thread-safety of Register operations.
func TestConcurrentRegister(t *testing.T) {
	t.Run("concurrent register different names", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numGoroutines = 100

		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("session-%d", idx)
				session := NewSession(name, "description")
				if err := registry.Register(name, session); err != nil {
					errors <- err
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		// Check for errors
		for err := range errors {
			t.Errorf("unexpected error during concurrent register: %v", err)
		}

		// Verify all sessions were registered
		if registry.Count() != numGoroutines {
			t.Errorf("expected %d sessions, got %d", numGoroutines, registry.Count())
		}
	})

	t.Run("concurrent register same name", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numGoroutines = 50

		var wg sync.WaitGroup
		successCount := make(chan struct{}, numGoroutines)
		errorCount := make(chan struct{}, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				session := NewSession(fmt.Sprintf("session-%d", idx), "description")
				if err := registry.Register("same-name", session); err != nil {
					errorCount <- struct{}{}
				} else {
					successCount <- struct{}{}
				}
			}(i)
		}

		wg.Wait()
		close(successCount)
		close(errorCount)

		// Count successes and errors
		successes := 0
		for range successCount {
			successes++
		}
		errors := 0
		for range errorCount {
			errors++
		}

		// Exactly one should succeed
		if successes != 1 {
			t.Errorf("expected exactly 1 success, got %d", successes)
		}
		if errors != numGoroutines-1 {
			t.Errorf("expected %d errors, got %d", numGoroutines-1, errors)
		}
		if registry.Count() != 1 {
			t.Errorf("expected 1 session, got %d", registry.Count())
		}
	})
}

// TestConcurrentGet verifies thread-safety of Get operations.
func TestConcurrentGet(t *testing.T) {
	t.Run("concurrent get same session", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "description")
		_ = registry.Register("test", session)

		const numGoroutines = 100
		var wg sync.WaitGroup
		results := make(chan *Session, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				got, ok := registry.Get("test")
				if !ok {
					t.Error("expected to find session")
				}
				results <- got
			}()
		}

		wg.Wait()
		close(results)

		// All should return the same session
		for got := range results {
			if got != session {
				t.Error("concurrent Get returned different session")
			}
		}
	})

	t.Run("concurrent get different sessions", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numSessions = 10

		// Register sessions
		sessions := make(map[string]*Session)
		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			s := NewSession(name, "description")
			sessions[name] = s
			_ = registry.Register(name, s)
		}

		const numGoroutines = 100
		var wg sync.WaitGroup

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("session-%d", idx%numSessions)
				got, ok := registry.Get(name)
				if !ok {
					t.Errorf("expected to find session %s", name)
				}
				if got != sessions[name] {
					t.Errorf("wrong session returned for %s", name)
				}
			}(i)
		}

		wg.Wait()
	})
}

// TestConcurrentList verifies thread-safety of List operations.
func TestConcurrentList(t *testing.T) {
	t.Run("concurrent list operations", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numSessions = 10

		// Register sessions
		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			_ = registry.Register(name, NewSession(name, "description"))
		}

		const numGoroutines = 50
		var wg sync.WaitGroup

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				names := registry.List()
				if len(names) != numSessions {
					t.Errorf("expected %d names, got %d", numSessions, len(names))
				}
			}()
		}

		wg.Wait()
	})
}

// TestConcurrentMixedOperations verifies thread-safety with mixed read/write operations.
func TestConcurrentMixedOperations(t *testing.T) {
	t.Run("concurrent register and get", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numGoroutines = 50

		var wg sync.WaitGroup

		// Half register, half get
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			if i%2 == 0 {
				// Register
				go func(idx int) {
					defer wg.Done()
					name := fmt.Sprintf("session-%d", idx)
					session := NewSession(name, "description")
					_ = registry.Register(name, session)
				}(i)
			} else {
				// Get (may or may not exist yet)
				go func(idx int) {
					defer wg.Done()
					name := fmt.Sprintf("session-%d", idx-1)
					_, _ = registry.Get(name) // Result doesn't matter, testing for race
				}(i)
			}
		}

		wg.Wait()
	})

	t.Run("concurrent register and list", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numRegisters = 50
		const numLists = 50

		var wg sync.WaitGroup

		// Register goroutines
		for i := 0; i < numRegisters; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("session-%d", idx)
				session := NewSession(name, "description")
				_ = registry.Register(name, session)
			}(i)
		}

		// List goroutines
		for i := 0; i < numLists; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = registry.List()
			}()
		}

		wg.Wait()

		// All registers should have completed
		if registry.Count() != numRegisters {
			t.Errorf("expected %d sessions, got %d", numRegisters, registry.Count())
		}
	})

	t.Run("concurrent register and unregister", func(t *testing.T) {
		registry := NewSessionRegistry()

		var wg sync.WaitGroup

		// Pre-register some sessions
		for i := 0; i < 25; i++ {
			name := fmt.Sprintf("pre-session-%d", i)
			_ = registry.Register(name, NewSession(name, "description"))
		}

		// Concurrently register new and unregister existing
		for i := 0; i < 25; i++ {
			wg.Add(2)

			// Register new
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("new-session-%d", idx)
				session := NewSession(name, "description")
				_ = registry.Register(name, session)
			}(i)

			// Unregister existing
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("pre-session-%d", idx)
				_ = registry.Unregister(name)
			}(i)
		}

		wg.Wait()

		// Should have 25 new sessions (pre-sessions removed, new ones added)
		if registry.Count() != 25 {
			t.Errorf("expected 25 sessions, got %d", registry.Count())
		}
	})

	t.Run("concurrent create and get", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numGoroutines = 50

		var wg sync.WaitGroup

		// Create goroutines
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("session-%d", idx)
				_, _ = registry.Create(name, "description")
			}(i)
		}

		// Get goroutines (running concurrently with creates)
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("session-%d", idx)
				_, _ = registry.Get(name)
			}(i)
		}

		wg.Wait()

		// All creates should have completed
		if registry.Count() != numGoroutines {
			t.Errorf("expected %d sessions, got %d", numGoroutines, registry.Count())
		}
	})

	t.Run("concurrent GetOrCreate same name", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numGoroutines = 100

		var wg sync.WaitGroup
		sessions := make(chan *Session, numGoroutines)
		createdCount := make(chan bool, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				session, created := registry.GetOrCreate("same-name", "description")
				sessions <- session
				createdCount <- created
			}()
		}

		wg.Wait()
		close(sessions)
		close(createdCount)

		// Count creations (should be exactly 1)
		creates := 0
		for created := range createdCount {
			if created {
				creates++
			}
		}
		if creates != 1 {
			t.Errorf("expected exactly 1 creation, got %d", creates)
		}

		// All should return the same session
		var firstSession *Session
		for s := range sessions {
			if firstSession == nil {
				firstSession = s
			} else if s != firstSession {
				t.Error("GetOrCreate returned different sessions for same name")
			}
		}

		if registry.Count() != 1 {
			t.Errorf("expected 1 session, got %d", registry.Count())
		}
	})

	t.Run("concurrent status and active", func(t *testing.T) {
		registry := NewSessionRegistry()

		// Register mix of active and ended sessions
		for i := 0; i < 10; i++ {
			session := NewSession(fmt.Sprintf("session-%d", i), "description")
			if i%2 == 0 {
				session.End()
			}
			_ = registry.Register(fmt.Sprintf("session-%d", i), session)
		}

		const numGoroutines = 50
		var wg sync.WaitGroup

		// Status goroutines
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				status := registry.Status()
				if len(status) != 10 {
					t.Errorf("expected 10 statuses, got %d", len(status))
				}
			}()
		}

		// Active goroutines
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				active := registry.Active()
				if len(active) != 5 {
					t.Errorf("expected 5 active sessions, got %d", len(active))
				}
			}()
		}

		wg.Wait()
	})

	t.Run("concurrent count operations", func(t *testing.T) {
		registry := NewSessionRegistry()

		// Register some sessions
		for i := 0; i < 10; i++ {
			_ = registry.Register(fmt.Sprintf("session-%d", i), NewSession(fmt.Sprintf("session-%d", i), "d"))
		}

		const numGoroutines = 100
		var wg sync.WaitGroup

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				count := registry.Count()
				if count != 10 {
					t.Errorf("expected count of 10, got %d", count)
				}
			}()
		}

		wg.Wait()
	})
}

// TestConcurrentStress performs stress testing with high concurrency.
func TestConcurrentStress(t *testing.T) {
	t.Run("stress test all operations", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numGoroutines = 200

		var wg sync.WaitGroup

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()

				// Perform various operations
				name := fmt.Sprintf("session-%d", idx%50) // Some name collisions

				switch idx % 7 {
				case 0:
					_, _ = registry.Create(name, "description")
				case 1:
					_, _ = registry.Get(name)
				case 2:
					_ = registry.List()
				case 3:
					_ = registry.Register(name, NewSession(name, "d"))
				case 4:
					_ = registry.Status()
				case 5:
					_ = registry.Active()
				case 6:
					_ = registry.Count()
				}
			}(i)
		}

		wg.Wait()

		// Just verify registry is in consistent state
		count := registry.Count()
		list := registry.List()
		if len(list) != count {
			t.Errorf("count (%d) and list length (%d) mismatch", count, len(list))
		}
	})
}

// TestCount verifies the Count method functionality.
func TestCount(t *testing.T) {
	t.Run("empty registry returns zero", func(t *testing.T) {
		registry := NewSessionRegistry()

		count := registry.Count()
		if count != 0 {
			t.Errorf("expected 0 for empty registry, got %d", count)
		}
	})

	t.Run("single session", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("test", NewSession("test", "d"))

		count := registry.Count()
		if count != 1 {
			t.Errorf("expected 1, got %d", count)
		}
	})

	t.Run("multiple sessions", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("s1", NewSession("s1", "d1"))
		_ = registry.Register("s2", NewSession("s2", "d2"))
		_ = registry.Register("s3", NewSession("s3", "d3"))

		count := registry.Count()
		if count != 3 {
			t.Errorf("expected 3, got %d", count)
		}
	})

	t.Run("count after unregister", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("s1", NewSession("s1", "d1"))
		_ = registry.Register("s2", NewSession("s2", "d2"))

		if registry.Count() != 2 {
			t.Fatalf("expected 2 before unregister, got %d", registry.Count())
		}

		_ = registry.Unregister("s1")

		if registry.Count() != 1 {
			t.Errorf("expected 1 after unregister, got %d", registry.Count())
		}
	})

	t.Run("count with Create method", func(t *testing.T) {
		registry := NewSessionRegistry()

		if registry.Count() != 0 {
			t.Fatalf("expected 0 initially, got %d", registry.Count())
		}

		_, _ = registry.Create("test1", "d1")
		if registry.Count() != 1 {
			t.Errorf("expected 1 after first Create, got %d", registry.Count())
		}

		_, _ = registry.Create("test2", "d2")
		if registry.Count() != 2 {
			t.Errorf("expected 2 after second Create, got %d", registry.Count())
		}
	})

	t.Run("count with GetOrCreate method", func(t *testing.T) {
		registry := NewSessionRegistry()

		_, _ = registry.GetOrCreate("test", "d")
		if registry.Count() != 1 {
			t.Errorf("expected 1 after first GetOrCreate, got %d", registry.Count())
		}

		// Same name should not increase count
		_, _ = registry.GetOrCreate("test", "d")
		if registry.Count() != 1 {
			t.Errorf("expected 1 after duplicate GetOrCreate, got %d", registry.Count())
		}

		// Different name should increase count
		_, _ = registry.GetOrCreate("test2", "d")
		if registry.Count() != 2 {
			t.Errorf("expected 2 after new name GetOrCreate, got %d", registry.Count())
		}
	})

	t.Run("count includes nil sessions", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("nil-session", nil)
		_ = registry.Register("real-session", NewSession("real", "d"))

		if registry.Count() != 2 {
			t.Errorf("expected 2 (including nil session), got %d", registry.Count())
		}
	})

	t.Run("count includes ended sessions", func(t *testing.T) {
		registry := NewSessionRegistry()
		active := NewSession("active", "d")
		ended := NewSession("ended", "d")
		ended.End()

		_ = registry.Register("active", active)
		_ = registry.Register("ended", ended)

		if registry.Count() != 2 {
			t.Errorf("expected 2 (including ended session), got %d", registry.Count())
		}
	})
}
