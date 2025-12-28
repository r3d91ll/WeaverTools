package yarn

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
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

		// Verify error type is SessionAlreadyRegisteredError
		sarErr, ok := err.(*SessionAlreadyRegisteredError)
		if !ok {
			t.Fatalf("expected *SessionAlreadyRegisteredError, got %T", err)
		}
		if sarErr.Name != "dup" {
			t.Errorf("expected Name 'dup', got %q", sarErr.Name)
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

		// Verify error type is SessionNotFoundError
		snfErr, ok := err.(*SessionNotFoundError)
		if !ok {
			t.Fatalf("expected *SessionNotFoundError, got %T", err)
		}
		if snfErr.Name != "nonexistent" {
			t.Errorf("expected Name 'nonexistent', got %q", snfErr.Name)
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

		// Verify error type is SessionAlreadyRegisteredError
		sarErr, ok := err.(*SessionAlreadyRegisteredError)
		if !ok {
			t.Fatalf("expected *SessionAlreadyRegisteredError, got %T", err)
		}
		if sarErr.Name != "dup" {
			t.Errorf("expected Name 'dup', got %q", sarErr.Name)
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
		errorCount := make(chan error, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				session := NewSession(fmt.Sprintf("session-%d", idx), "description")
				if err := registry.Register("same-name", session); err != nil {
					errorCount <- err
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
		for err := range errorCount {
			errors++
			// Verify error type is SessionAlreadyRegisteredError
			if _, ok := err.(*SessionAlreadyRegisteredError); !ok {
				t.Errorf("expected *SessionAlreadyRegisteredError, got %T", err)
			}
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

// ============================================================================
// suggestSimilarSessions Tests
// ============================================================================

// TestSuggestSimilarSessions_CaseMismatch verifies detection of case-insensitive matches.
func TestSuggestSimilarSessions_CaseMismatch(t *testing.T) {
	available := []string{"my-experiment", "loom-test"}

	suggestions := suggestSimilarSessions("My-Experiment", available)
	if len(suggestions) == 0 {
		t.Error("expected case mismatch suggestion")
	}

	found := false
	for _, s := range suggestions {
		if strings.Contains(s, "my-experiment") && strings.Contains(s, "case") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected suggestion about case mismatch")
	}
}

// TestSuggestSimilarSessions_PartialMatch verifies detection of substring matches.
func TestSuggestSimilarSessions_PartialMatch(t *testing.T) {
	t.Run("input is substring of available", func(t *testing.T) {
		available := []string{"experiment-2024", "loom-test"}

		suggestions := suggestSimilarSessions("experiment", available)
		if len(suggestions) == 0 {
			t.Error("expected partial match suggestion")
		}

		found := false
		for _, s := range suggestions {
			if strings.Contains(s, "experiment-2024") {
				found = true
				break
			}
		}
		if !found {
			t.Error("expected suggestion for experiment-2024")
		}
	})

	t.Run("available is substring of input", func(t *testing.T) {
		available := []string{"my", "test"}

		suggestions := suggestSimilarSessions("my-session", available)
		if len(suggestions) == 0 {
			t.Error("expected partial match suggestion when available is substring of input")
		}

		found := false
		for _, s := range suggestions {
			if strings.Contains(s, "my") {
				found = true
				break
			}
		}
		if !found {
			t.Error("expected suggestion for 'my'")
		}
	})
}

// TestSuggestSimilarSessions_NoMatch verifies no suggestions for unrelated names.
func TestSuggestSimilarSessions_NoMatch(t *testing.T) {
	available := []string{"experiment-2024", "loom-test"}

	suggestions := suggestSimilarSessions("xyz-unrelated", available)
	if len(suggestions) != 0 {
		t.Errorf("expected no suggestions for completely different name, got %v", suggestions)
	}
}

// TestSuggestSimilarSessions_ExactMatch verifies no suggestions for exact matches.
func TestSuggestSimilarSessions_ExactMatch(t *testing.T) {
	available := []string{"experiment-2024", "loom-test"}

	// Exact match shouldn't produce suggestions (no correction needed)
	suggestions := suggestSimilarSessions("experiment-2024", available)
	if len(suggestions) != 0 {
		t.Errorf("expected no suggestions for exact match, got %v", suggestions)
	}
}

// TestSuggestSimilarSessions_EmptyAvailable verifies behavior with no available sessions.
func TestSuggestSimilarSessions_EmptyAvailable(t *testing.T) {
	available := []string{}

	suggestions := suggestSimilarSessions("my-session", available)
	if len(suggestions) != 0 {
		t.Errorf("expected no suggestions when no sessions available, got %v", suggestions)
	}
}

// TestSuggestSimilarSessions_EmptyInput verifies behavior with empty input name.
func TestSuggestSimilarSessions_EmptyInput(t *testing.T) {
	available := []string{"experiment-2024", "loom-test"}

	suggestions := suggestSimilarSessions("", available)
	// Empty string should match as substring of everything, but we're looking for typo help
	// The function may or may not suggest - just ensure no panic
	_ = suggestions
}

// TestSuggestSimilarSessions_MultipleSuggestions verifies multiple matches are returned.
func TestSuggestSimilarSessions_MultipleSuggestions(t *testing.T) {
	available := []string{"test-alpha", "test-beta", "test-gamma"}

	suggestions := suggestSimilarSessions("test", available)
	if len(suggestions) < 3 {
		t.Errorf("expected 3 suggestions for 'test' matching test-*, got %d: %v", len(suggestions), suggestions)
	}
}

// TestSuggestSimilarSessions_CaseMismatchFormat verifies the format of case mismatch suggestions.
func TestSuggestSimilarSessions_CaseMismatchFormat(t *testing.T) {
	available := []string{"mysession"}

	suggestions := suggestSimilarSessions("MySession", available)
	if len(suggestions) != 1 {
		t.Fatalf("expected 1 suggestion, got %d", len(suggestions))
	}

	expected := `Did you mean "mysession"? (case mismatch)`
	if suggestions[0] != expected {
		t.Errorf("expected %q, got %q", expected, suggestions[0])
	}
}

// TestSuggestSimilarSessions_PartialMatchFormat verifies the format of partial match suggestions.
func TestSuggestSimilarSessions_PartialMatchFormat(t *testing.T) {
	available := []string{"experiment-2024"}

	suggestions := suggestSimilarSessions("experiment", available)
	if len(suggestions) != 1 {
		t.Fatalf("expected 1 suggestion, got %d", len(suggestions))
	}

	expected := `Did you mean "experiment-2024"?`
	if suggestions[0] != expected {
		t.Errorf("expected %q, got %q", expected, suggestions[0])
	}
}

// TestSuggestSimilarSessions_CaseMismatchPrioritized verifies case mismatch is distinct from partial match.
func TestSuggestSimilarSessions_CaseMismatchPrioritized(t *testing.T) {
	// When there's both a case mismatch and partial match, case mismatch should be noted
	available := []string{"MySession"}

	suggestions := suggestSimilarSessions("mysession", available)
	if len(suggestions) != 1 {
		t.Fatalf("expected 1 suggestion, got %d: %v", len(suggestions), suggestions)
	}

	// Should indicate case mismatch, not just partial match
	if !strings.Contains(suggestions[0], "case") {
		t.Errorf("expected case mismatch indication, got %q", suggestions[0])
	}
}

// TestSuggestSimilarSessions_NilAvailable handles nil slice (defensive).
func TestSuggestSimilarSessions_NilAvailable(t *testing.T) {
	// The function signature takes []string which can be nil
	suggestions := suggestSimilarSessions("test", nil)
	if len(suggestions) != 0 {
		t.Errorf("expected no suggestions for nil available, got %v", suggestions)
	}
}

// ============================================================================
// GetWithError Tests
// ============================================================================

// TestGetWithError_Exists verifies successful retrieval returns session without error.
func TestGetWithError_Exists(t *testing.T) {
	registry := NewSessionRegistry()
	session := NewSession("test-session", "test description")
	_ = registry.Register("test", session)

	got, err := registry.GetWithError("test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != session {
		t.Error("expected same session instance")
	}
}

// TestGetWithError_NotExists verifies non-existent session returns error.
func TestGetWithError_NotExists(t *testing.T) {
	registry := NewSessionRegistry()
	session := NewSession("test-session", "test description")
	_ = registry.Register("test", session)

	_, err := registry.GetWithError("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent session")
	}

	// Should be a SessionNotFoundError
	snfErr, ok := err.(*SessionNotFoundError)
	if !ok {
		t.Fatalf("expected *SessionNotFoundError, got %T", err)
	}

	// Should have correct name
	if snfErr.Name != "nonexistent" {
		t.Errorf("expected Name 'nonexistent', got %q", snfErr.Name)
	}
}

// TestGetWithError_ReturnsAvailableSessions verifies error includes available sessions.
func TestGetWithError_ReturnsAvailableSessions(t *testing.T) {
	registry := NewSessionRegistry()
	_ = registry.Register("alpha", NewSession("alpha", "d"))
	_ = registry.Register("beta", NewSession("beta", "d"))
	_ = registry.Register("gamma", NewSession("gamma", "d"))

	_, err := registry.GetWithError("nonexistent")
	if err == nil {
		t.Fatal("expected error")
	}

	snfErr := err.(*SessionNotFoundError)

	// Should list all available sessions (sorted)
	if len(snfErr.AvailableSessions) != 3 {
		t.Errorf("expected 3 available sessions, got %d", len(snfErr.AvailableSessions))
	}

	// Verify sorted order
	expected := []string{"alpha", "beta", "gamma"}
	for i, name := range expected {
		if i >= len(snfErr.AvailableSessions) {
			break
		}
		if snfErr.AvailableSessions[i] != name {
			t.Errorf("expected %q at index %d, got %q", name, i, snfErr.AvailableSessions[i])
		}
	}
}

// TestGetWithError_IncludesSuggestions verifies error includes similar name suggestions.
func TestGetWithError_IncludesSuggestions(t *testing.T) {
	registry := NewSessionRegistry()
	_ = registry.Register("my-experiment", NewSession("my-experiment", "d"))
	_ = registry.Register("loom-test", NewSession("loom-test", "d"))

	// Try with case mismatch
	_, err := registry.GetWithError("My-Experiment")
	if err == nil {
		t.Fatal("expected error")
	}

	snfErr := err.(*SessionNotFoundError)

	// Should have suggestions
	if len(snfErr.Suggestions) == 0 {
		t.Error("expected suggestions to be attached")
	}

	// Should suggest correct name with case mismatch note
	foundSuggestion := false
	for _, s := range snfErr.Suggestions {
		if strings.Contains(s, "my-experiment") && strings.Contains(s, "case") {
			foundSuggestion = true
			break
		}
	}
	if !foundSuggestion {
		t.Errorf("expected suggestion about case mismatch, got %v", snfErr.Suggestions)
	}
}

// TestGetWithError_SuggestsPartialMatch verifies partial name match suggestions.
func TestGetWithError_SuggestsPartialMatch(t *testing.T) {
	registry := NewSessionRegistry()
	_ = registry.Register("experiment-2024", NewSession("experiment-2024", "d"))
	_ = registry.Register("loom-test", NewSession("loom-test", "d"))

	// Try with partial name
	_, err := registry.GetWithError("experiment")
	if err == nil {
		t.Fatal("expected error")
	}

	snfErr := err.(*SessionNotFoundError)

	// Should suggest similar session
	foundSuggestion := false
	for _, s := range snfErr.Suggestions {
		if strings.Contains(s, "experiment-2024") && strings.Contains(strings.ToLower(s), "mean") {
			foundSuggestion = true
			break
		}
	}
	if !foundSuggestion {
		t.Errorf("expected 'Did you mean' suggestion for partial match, got %v", snfErr.Suggestions)
	}
}

// TestGetWithError_EmptyRegistry verifies behavior with no registered sessions.
func TestGetWithError_EmptyRegistry(t *testing.T) {
	registry := NewSessionRegistry()

	_, err := registry.GetWithError("anysession")
	if err == nil {
		t.Fatal("expected error for empty registry")
	}

	snfErr := err.(*SessionNotFoundError)

	// Should indicate no sessions
	if len(snfErr.AvailableSessions) != 0 {
		t.Errorf("expected empty available sessions, got %v", snfErr.AvailableSessions)
	}

	// Error message should indicate no sessions
	errMsg := snfErr.Error()
	if !strings.Contains(errMsg, "no sessions registered") {
		t.Errorf("expected 'no sessions registered' in error message, got %q", errMsg)
	}
}

// TestGetWithError_ErrorMessageFormat verifies error message format.
func TestGetWithError_ErrorMessageFormat(t *testing.T) {
	registry := NewSessionRegistry()
	_ = registry.Register("session-1", NewSession("session-1", "d"))
	_ = registry.Register("session-2", NewSession("session-2", "d"))

	_, err := registry.GetWithError("nonexistent")
	if err == nil {
		t.Fatal("expected error")
	}

	errMsg := err.Error()

	// Should contain session name
	if !strings.Contains(errMsg, `"nonexistent"`) {
		t.Errorf("expected session name in error message, got %q", errMsg)
	}

	// Should contain available sessions
	if !strings.Contains(errMsg, "available:") {
		t.Errorf("expected 'available:' in error message, got %q", errMsg)
	}

	// Should list sessions
	if !strings.Contains(errMsg, "session-1") || !strings.Contains(errMsg, "session-2") {
		t.Errorf("expected available sessions listed, got %q", errMsg)
	}
}

// TestGetWithError_ErrorMessageWithSuggestions verifies error message includes suggestions.
func TestGetWithError_ErrorMessageWithSuggestions(t *testing.T) {
	registry := NewSessionRegistry()
	_ = registry.Register("mySession", NewSession("mySession", "d"))

	_, err := registry.GetWithError("mysession") // case mismatch
	if err == nil {
		t.Fatal("expected error")
	}

	errMsg := err.Error()

	// Should contain suggestion
	if !strings.Contains(errMsg, "Did you mean") {
		t.Errorf("expected 'Did you mean' in error message, got %q", errMsg)
	}
}

// TestGetWithError_MultipleRetrievals verifies multiple sessions can be retrieved.
func TestGetWithError_MultipleRetrievals(t *testing.T) {
	registry := NewSessionRegistry()
	session1 := NewSession("session-1", "first")
	session2 := NewSession("session-2", "second")
	session3 := NewSession("session-3", "third")

	_ = registry.Register("s1", session1)
	_ = registry.Register("s2", session2)
	_ = registry.Register("s3", session3)

	got1, err1 := registry.GetWithError("s1")
	if err1 != nil || got1 != session1 {
		t.Error("failed to retrieve session s1")
	}

	got2, err2 := registry.GetWithError("s2")
	if err2 != nil || got2 != session2 {
		t.Error("failed to retrieve session s2")
	}

	got3, err3 := registry.GetWithError("s3")
	if err3 != nil || got3 != session3 {
		t.Error("failed to retrieve session s3")
	}
}

// TestGetWithError_NilSession verifies retrieval of nil session.
func TestGetWithError_NilSession(t *testing.T) {
	registry := NewSessionRegistry()
	_ = registry.Register("nil-session", nil)

	got, err := registry.GetWithError("nil-session")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Error("expected nil session to be returned")
	}
}

// TestGetWithError_EmptyNameSession verifies retrieval with empty name key.
func TestGetWithError_EmptyNameSession(t *testing.T) {
	registry := NewSessionRegistry()
	session := NewSession("empty-name-session", "description")
	_ = registry.Register("", session)

	got, err := registry.GetWithError("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != session {
		t.Error("returned session does not match registered session")
	}
}

// TestGetWithError_ErrorImplementsErrorInterface verifies error type implements error.
func TestGetWithError_ErrorImplementsErrorInterface(t *testing.T) {
	registry := NewSessionRegistry()

	_, err := registry.GetWithError("nonexistent")
	if err == nil {
		t.Fatal("expected error")
	}

	// Should implement error interface
	var _ error = err

	// Error() should return non-empty string
	errMsg := err.Error()
	if errMsg == "" {
		t.Error("expected non-empty error message")
	}
}

// TestConcurrentGetWithError verifies thread-safety of GetWithError operations.
func TestConcurrentGetWithError(t *testing.T) {
	t.Run("concurrent GetWithError same session", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "description")
		_ = registry.Register("test", session)

		const numGoroutines = 100
		var wg sync.WaitGroup
		results := make(chan *Session, numGoroutines)
		errs := make(chan error, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				got, err := registry.GetWithError("test")
				if err != nil {
					errs <- err
				} else {
					results <- got
				}
			}()
		}

		wg.Wait()
		close(results)
		close(errs)

		// Check for errors
		for err := range errs {
			t.Errorf("unexpected error: %v", err)
		}

		// All should return the same session
		for got := range results {
			if got != session {
				t.Error("concurrent GetWithError returned different session")
			}
		}
	})

	t.Run("concurrent GetWithError different sessions", func(t *testing.T) {
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
				got, err := registry.GetWithError(name)
				if err != nil {
					t.Errorf("unexpected error for %s: %v", name, err)
				}
				if got != sessions[name] {
					t.Errorf("wrong session returned for %s", name)
				}
			}(i)
		}

		wg.Wait()
	})

	t.Run("concurrent GetWithError with nonexistent", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("exists", NewSession("exists", "d"))

		const numGoroutines = 50
		var wg sync.WaitGroup

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, err := registry.GetWithError("nonexistent")
				if err == nil {
					t.Error("expected error for nonexistent session")
				}
				// Verify error type
				if _, ok := err.(*SessionNotFoundError); !ok {
					t.Errorf("expected *SessionNotFoundError, got %T", err)
				}
			}()
		}

		wg.Wait()
	})

	t.Run("concurrent GetWithError while modifying registry", func(t *testing.T) {
		registry := NewSessionRegistry()

		// Pre-register some sessions
		for i := 0; i < 10; i++ {
			_ = registry.Register(fmt.Sprintf("session-%d", i), NewSession(fmt.Sprintf("session-%d", i), "d"))
		}

		var wg sync.WaitGroup
		const numGoroutines = 50

		// Half doing GetWithError, half doing Register/Unregister
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			if i%2 == 0 {
				go func(idx int) {
					defer wg.Done()
					name := fmt.Sprintf("session-%d", idx%10)
					_, _ = registry.GetWithError(name) // May succeed or fail depending on timing
				}(i)
			} else {
				go func(idx int) {
					defer wg.Done()
					name := fmt.Sprintf("new-session-%d", idx)
					_, _ = registry.Create(name, "d")
				}(i)
			}
		}

		wg.Wait()
	})
}

// ============================================================================
// Error Type Tests
// ============================================================================

// TestSessionNotFoundError_Error verifies the Error() method output.
func TestSessionNotFoundError_Error(t *testing.T) {
	t.Run("basic error message format", func(t *testing.T) {
		err := &SessionNotFoundError{
			Name:              "my-session",
			AvailableSessions: []string{"session-1", "session-2"},
			Suggestions:       nil,
		}

		msg := err.Error()

		// Should contain session name
		if !strings.Contains(msg, `"my-session"`) {
			t.Errorf("expected session name in error, got %q", msg)
		}

		// Should contain 'not found'
		if !strings.Contains(msg, "not found") {
			t.Errorf("expected 'not found' in error, got %q", msg)
		}

		// Should list available sessions
		if !strings.Contains(msg, "available:") {
			t.Errorf("expected 'available:' in error, got %q", msg)
		}
		if !strings.Contains(msg, "session-1") || !strings.Contains(msg, "session-2") {
			t.Errorf("expected available sessions listed, got %q", msg)
		}
	})

	t.Run("empty available sessions", func(t *testing.T) {
		err := &SessionNotFoundError{
			Name:              "nonexistent",
			AvailableSessions: []string{},
			Suggestions:       nil,
		}

		msg := err.Error()

		// Should indicate no sessions
		if !strings.Contains(msg, "no sessions registered") {
			t.Errorf("expected 'no sessions registered' in error, got %q", msg)
		}
	})

	t.Run("with suggestions", func(t *testing.T) {
		err := &SessionNotFoundError{
			Name:              "mysession",
			AvailableSessions: []string{"MySession"},
			Suggestions:       []string{`Did you mean "MySession"? (case mismatch)`},
		}

		msg := err.Error()

		// Should include suggestion
		if !strings.Contains(msg, "Did you mean") {
			t.Errorf("expected suggestion in error, got %q", msg)
		}
		if !strings.Contains(msg, "case mismatch") {
			t.Errorf("expected 'case mismatch' in suggestion, got %q", msg)
		}
	})

	t.Run("with multiple suggestions", func(t *testing.T) {
		err := &SessionNotFoundError{
			Name:              "test",
			AvailableSessions: []string{"test-alpha", "test-beta"},
			Suggestions: []string{
				`Did you mean "test-alpha"?`,
				`Did you mean "test-beta"?`,
			},
		}

		msg := err.Error()

		// Should include both suggestions
		if !strings.Contains(msg, "test-alpha") {
			t.Errorf("expected first suggestion in error, got %q", msg)
		}
		if !strings.Contains(msg, "test-beta") {
			t.Errorf("expected second suggestion in error, got %q", msg)
		}
	})

	t.Run("nil slices handled gracefully", func(t *testing.T) {
		err := &SessionNotFoundError{
			Name:              "test",
			AvailableSessions: nil,
			Suggestions:       nil,
		}

		// Should not panic
		msg := err.Error()
		if msg == "" {
			t.Error("expected non-empty error message")
		}
		// Should indicate no sessions
		if !strings.Contains(msg, "no sessions registered") {
			t.Errorf("expected 'no sessions registered' for nil available, got %q", msg)
		}
	})
}

// TestSessionNotFoundError_Fields verifies the error contains expected fields.
func TestSessionNotFoundError_Fields(t *testing.T) {
	err := &SessionNotFoundError{
		Name:              "requested-session",
		AvailableSessions: []string{"a", "b", "c"},
		Suggestions:       []string{"suggestion-1", "suggestion-2"},
	}

	if err.Name != "requested-session" {
		t.Errorf("expected Name 'requested-session', got %q", err.Name)
	}

	if len(err.AvailableSessions) != 3 {
		t.Errorf("expected 3 available sessions, got %d", len(err.AvailableSessions))
	}

	if len(err.Suggestions) != 2 {
		t.Errorf("expected 2 suggestions, got %d", len(err.Suggestions))
	}
}

// TestSessionNotFoundError_ImplementsError verifies the error implements the error interface.
func TestSessionNotFoundError_ImplementsError(t *testing.T) {
	err := &SessionNotFoundError{
		Name: "test",
	}

	// Compile-time check: error interface implementation
	var _ error = err

	// Runtime check: Error() returns non-empty string
	if err.Error() == "" {
		t.Error("expected non-empty error message")
	}
}

// TestSessionAlreadyRegisteredError_Error verifies the Error() method output.
func TestSessionAlreadyRegisteredError_Error(t *testing.T) {
	t.Run("basic error message format", func(t *testing.T) {
		err := &SessionAlreadyRegisteredError{
			Name:               "my-session",
			RegisteredSessions: []string{"my-session", "other-session"},
		}

		msg := err.Error()

		// Should contain session name
		if !strings.Contains(msg, `"my-session"`) {
			t.Errorf("expected session name in error, got %q", msg)
		}

		// Should contain 'already registered'
		if !strings.Contains(msg, "already registered") {
			t.Errorf("expected 'already registered' in error, got %q", msg)
		}

		// Should list registered sessions
		if !strings.Contains(msg, "registered:") {
			t.Errorf("expected 'registered:' in error, got %q", msg)
		}
		if !strings.Contains(msg, "my-session") || !strings.Contains(msg, "other-session") {
			t.Errorf("expected registered sessions listed, got %q", msg)
		}
	})

	t.Run("contains actionable suggestions", func(t *testing.T) {
		err := &SessionAlreadyRegisteredError{
			Name:               "duplicate",
			RegisteredSessions: []string{"duplicate"},
		}

		msg := err.Error()

		// Should suggest choosing different name
		if !strings.Contains(msg, "choose a different name") {
			t.Errorf("expected 'choose a different name' suggestion, got %q", msg)
		}

		// Should suggest using Get()
		if !strings.Contains(msg, "Get()") {
			t.Errorf("expected 'Get()' suggestion, got %q", msg)
		}

		// Should suggest using GetOrCreate()
		if !strings.Contains(msg, "GetOrCreate()") {
			t.Errorf("expected 'GetOrCreate()' suggestion, got %q", msg)
		}
	})

	t.Run("empty registered sessions", func(t *testing.T) {
		err := &SessionAlreadyRegisteredError{
			Name:               "session",
			RegisteredSessions: []string{},
		}

		// Should not panic
		msg := err.Error()
		if msg == "" {
			t.Error("expected non-empty error message")
		}

		// Should still contain the duplicate name
		if !strings.Contains(msg, `"session"`) {
			t.Errorf("expected session name in error, got %q", msg)
		}
	})

	t.Run("nil registered sessions handled gracefully", func(t *testing.T) {
		err := &SessionAlreadyRegisteredError{
			Name:               "test",
			RegisteredSessions: nil,
		}

		// Should not panic
		msg := err.Error()
		if msg == "" {
			t.Error("expected non-empty error message")
		}
	})

	t.Run("multiple registered sessions listed", func(t *testing.T) {
		err := &SessionAlreadyRegisteredError{
			Name:               "conflict",
			RegisteredSessions: []string{"session-a", "session-b", "session-c"},
		}

		msg := err.Error()

		// All sessions should be listed
		for _, name := range []string{"session-a", "session-b", "session-c"} {
			if !strings.Contains(msg, name) {
				t.Errorf("expected %q in registered sessions, got %q", name, msg)
			}
		}
	})
}

// TestSessionAlreadyRegisteredError_Fields verifies the error contains expected fields.
func TestSessionAlreadyRegisteredError_Fields(t *testing.T) {
	err := &SessionAlreadyRegisteredError{
		Name:               "duplicate-name",
		RegisteredSessions: []string{"a", "b", "duplicate-name"},
	}

	if err.Name != "duplicate-name" {
		t.Errorf("expected Name 'duplicate-name', got %q", err.Name)
	}

	if len(err.RegisteredSessions) != 3 {
		t.Errorf("expected 3 registered sessions, got %d", len(err.RegisteredSessions))
	}
}

// TestSessionAlreadyRegisteredError_ImplementsError verifies the error implements the error interface.
func TestSessionAlreadyRegisteredError_ImplementsError(t *testing.T) {
	err := &SessionAlreadyRegisteredError{
		Name: "test",
	}

	// Compile-time check: error interface implementation
	var _ error = err

	// Runtime check: Error() returns non-empty string
	if err.Error() == "" {
		t.Error("expected non-empty error message")
	}
}

// TestErrorTypeAssertions verifies error types can be type-asserted correctly.
func TestErrorTypeAssertions(t *testing.T) {
	t.Run("SessionNotFoundError from GetWithError", func(t *testing.T) {
		registry := NewSessionRegistry()

		_, err := registry.GetWithError("nonexistent")
		if err == nil {
			t.Fatal("expected error")
		}

		// Type assertion should succeed
		snfErr, ok := err.(*SessionNotFoundError)
		if !ok {
			t.Fatalf("expected *SessionNotFoundError, got %T", err)
		}

		// Fields should be populated
		if snfErr.Name != "nonexistent" {
			t.Errorf("expected Name 'nonexistent', got %q", snfErr.Name)
		}
	})

	t.Run("SessionNotFoundError from Unregister", func(t *testing.T) {
		registry := NewSessionRegistry()

		err := registry.Unregister("nonexistent")
		if err == nil {
			t.Fatal("expected error")
		}

		// Type assertion should succeed
		snfErr, ok := err.(*SessionNotFoundError)
		if !ok {
			t.Fatalf("expected *SessionNotFoundError, got %T", err)
		}

		if snfErr.Name != "nonexistent" {
			t.Errorf("expected Name 'nonexistent', got %q", snfErr.Name)
		}
	})

	t.Run("SessionAlreadyRegisteredError from Register", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test", "d")
		_ = registry.Register("duplicate", session)

		err := registry.Register("duplicate", session)
		if err == nil {
			t.Fatal("expected error")
		}

		// Type assertion should succeed
		sarErr, ok := err.(*SessionAlreadyRegisteredError)
		if !ok {
			t.Fatalf("expected *SessionAlreadyRegisteredError, got %T", err)
		}

		// Fields should be populated
		if sarErr.Name != "duplicate" {
			t.Errorf("expected Name 'duplicate', got %q", sarErr.Name)
		}
		if len(sarErr.RegisteredSessions) == 0 {
			t.Error("expected non-empty RegisteredSessions")
		}
	})

	t.Run("SessionAlreadyRegisteredError from Create", func(t *testing.T) {
		registry := NewSessionRegistry()
		_, _ = registry.Create("duplicate", "first")

		_, err := registry.Create("duplicate", "second")
		if err == nil {
			t.Fatal("expected error")
		}

		// Type assertion should succeed
		sarErr, ok := err.(*SessionAlreadyRegisteredError)
		if !ok {
			t.Fatalf("expected *SessionAlreadyRegisteredError, got %T", err)
		}

		if sarErr.Name != "duplicate" {
			t.Errorf("expected Name 'duplicate', got %q", sarErr.Name)
		}
	})
}

// TestErrorMessageHumanReadability verifies error messages are helpful for users.
func TestErrorMessageHumanReadability(t *testing.T) {
	t.Run("SessionNotFoundError is helpful", func(t *testing.T) {
		err := &SessionNotFoundError{
			Name:              "experment", // typo
			AvailableSessions: []string{"experiment", "test-session"},
			Suggestions:       []string{`Did you mean "experiment"?`},
		}

		msg := err.Error()

		// Should clearly indicate the problem
		if !strings.Contains(msg, "not found") {
			t.Error("error should indicate session not found")
		}

		// Should show what was requested
		if !strings.Contains(msg, "experment") {
			t.Error("error should show requested session name")
		}

		// Should show what's available
		if !strings.Contains(msg, "experiment") {
			t.Error("error should show available sessions")
		}

		// Should provide suggestion
		if !strings.Contains(msg, "Did you mean") {
			t.Error("error should include suggestion")
		}
	})

	t.Run("SessionAlreadyRegisteredError is actionable", func(t *testing.T) {
		err := &SessionAlreadyRegisteredError{
			Name:               "my-session",
			RegisteredSessions: []string{"my-session", "other"},
		}

		msg := err.Error()

		// Should clearly indicate the problem
		if !strings.Contains(msg, "already registered") {
			t.Error("error should indicate already registered")
		}

		// Should provide actionable suggestions
		suggestions := []string{
			"choose a different name",
			"Get()",
			"GetOrCreate()",
		}
		for _, suggestion := range suggestions {
			if !strings.Contains(msg, suggestion) {
				t.Errorf("error should include suggestion %q", suggestion)
			}
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

// ============================================================================
// Save/Load Tests
// ============================================================================

// TestSave verifies the Save method creates correct directory and files.
func TestSave(t *testing.T) {
	t.Run("creates directory if it doesn't exist", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "description")
		_ = registry.Register("test", session)

		dir := t.TempDir()
		savePath := filepath.Join(dir, "new-subdir")

		err := registry.Save(savePath)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Verify directory was created
		info, err := os.Stat(savePath)
		if err != nil {
			t.Fatalf("directory not created: %v", err)
		}
		if !info.IsDir() {
			t.Error("expected directory, got file")
		}
	})

	t.Run("creates manifest.json", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "description")
		_ = registry.Register("test", session)

		dir := t.TempDir()

		err := registry.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Verify manifest exists
		manifestPath := filepath.Join(dir, "manifest.json")
		data, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Fatalf("manifest.json not created: %v", err)
		}

		// Verify manifest contents
		var manifest struct {
			Version  int               `json:"version"`
			Sessions map[string]string `json:"sessions"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
			t.Fatalf("failed to parse manifest: %v", err)
		}
		if manifest.Version != 1 {
			t.Errorf("expected version 1, got %d", manifest.Version)
		}
		if len(manifest.Sessions) != 1 {
			t.Errorf("expected 1 session in manifest, got %d", len(manifest.Sessions))
		}
	})

	t.Run("creates session file", func(t *testing.T) {
		registry := NewSessionRegistry()
		session := NewSession("test-session", "test description")
		_ = registry.Register("test", session)

		dir := t.TempDir()

		err := registry.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Find session file (should be test.json based on sanitized name)
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("failed to read directory: %v", err)
		}

		sessionFileFound := false
		for _, entry := range entries {
			if entry.Name() != "manifest.json" && filepath.Ext(entry.Name()) == ".json" {
				sessionFileFound = true

				// Verify session file contents
				data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
				if err != nil {
					t.Fatalf("failed to read session file: %v", err)
				}

				var savedSession Session
				if err := json.Unmarshal(data, &savedSession); err != nil {
					t.Fatalf("failed to parse session file: %v", err)
				}

				if savedSession.Name != "test-session" {
					t.Errorf("expected Name 'test-session', got %q", savedSession.Name)
				}
				if savedSession.Description != "test description" {
					t.Errorf("expected Description 'test description', got %q", savedSession.Description)
				}
			}
		}

		if !sessionFileFound {
			t.Error("no session file was created")
		}
	})

	t.Run("empty registry saves empty manifest", func(t *testing.T) {
		registry := NewSessionRegistry()
		dir := t.TempDir()

		err := registry.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Verify manifest exists with empty sessions
		manifestPath := filepath.Join(dir, "manifest.json")
		data, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Fatalf("manifest.json not created: %v", err)
		}

		var manifest struct {
			Sessions map[string]string `json:"sessions"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
			t.Fatalf("failed to parse manifest: %v", err)
		}
		if len(manifest.Sessions) != 0 {
			t.Errorf("expected 0 sessions in manifest, got %d", len(manifest.Sessions))
		}
	})

	t.Run("multiple sessions saved", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("s1", NewSession("session-1", "desc 1"))
		_ = registry.Register("s2", NewSession("session-2", "desc 2"))
		_ = registry.Register("s3", NewSession("session-3", "desc 3"))

		dir := t.TempDir()

		err := registry.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Verify manifest contains all sessions
		manifestPath := filepath.Join(dir, "manifest.json")
		data, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Fatalf("failed to read manifest: %v", err)
		}

		var manifest struct {
			Sessions map[string]string `json:"sessions"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
			t.Fatalf("failed to parse manifest: %v", err)
		}
		if len(manifest.Sessions) != 3 {
			t.Errorf("expected 3 sessions in manifest, got %d", len(manifest.Sessions))
		}

		// Verify each registry name is in manifest
		for _, name := range []string{"s1", "s2", "s3"} {
			if _, ok := manifest.Sessions[name]; !ok {
				t.Errorf("session %q not found in manifest", name)
			}
		}
	})

	t.Run("nil session skipped", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("nil-session", nil)
		_ = registry.Register("real-session", NewSession("real", "d"))

		dir := t.TempDir()

		err := registry.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Verify only real session in manifest
		manifestPath := filepath.Join(dir, "manifest.json")
		data, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Fatalf("failed to read manifest: %v", err)
		}

		var manifest struct {
			Sessions map[string]string `json:"sessions"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
			t.Fatalf("failed to parse manifest: %v", err)
		}
		if len(manifest.Sessions) != 1 {
			t.Errorf("expected 1 session in manifest (nil skipped), got %d", len(manifest.Sessions))
		}
		if _, ok := manifest.Sessions["real-session"]; !ok {
			t.Error("real-session not found in manifest")
		}
	})
}

// TestLoad verifies the Load method restores registry state correctly.
func TestLoad(t *testing.T) {
	t.Run("loads single session", func(t *testing.T) {
		// First save a session
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")
		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Load into new registry
		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Verify session was loaded
		got, ok := loaded.Get("test")
		if !ok {
			t.Fatal("session 'test' not found after load")
		}
		if got.Name != "test-session" {
			t.Errorf("expected Name 'test-session', got %q", got.Name)
		}
		if got.Description != "test description" {
			t.Errorf("expected Description 'test description', got %q", got.Description)
		}
	})

	t.Run("loads multiple sessions", func(t *testing.T) {
		// Save multiple sessions
		original := NewSessionRegistry()
		_ = original.Register("s1", NewSession("session-1", "desc 1"))
		_ = original.Register("s2", NewSession("session-2", "desc 2"))
		_ = original.Register("s3", NewSession("session-3", "desc 3"))

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Load into new registry
		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Verify all sessions were loaded
		if loaded.Count() != 3 {
			t.Errorf("expected 3 sessions, got %d", loaded.Count())
		}

		for _, name := range []string{"s1", "s2", "s3"} {
			if _, ok := loaded.Get(name); !ok {
				t.Errorf("session %q not found after load", name)
			}
		}
	})

	t.Run("merges into existing registry", func(t *testing.T) {
		// Save a session
		original := NewSessionRegistry()
		_ = original.Register("saved", NewSession("saved-session", "saved desc"))

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Load into registry with existing session
		existing := NewSessionRegistry()
		_ = existing.Register("existing", NewSession("existing-session", "existing desc"))

		if err := existing.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Verify both sessions exist
		if existing.Count() != 2 {
			t.Errorf("expected 2 sessions after merge, got %d", existing.Count())
		}
		if _, ok := existing.Get("existing"); !ok {
			t.Error("existing session not found after load")
		}
		if _, ok := existing.Get("saved"); !ok {
			t.Error("saved session not found after load")
		}
	})

	t.Run("overwrites existing session with same name", func(t *testing.T) {
		// Save a session
		original := NewSessionRegistry()
		originalSession := NewSession("original", "original desc")
		_ = original.Register("test", originalSession)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Load into registry with session of same name
		existing := NewSessionRegistry()
		existingSession := NewSession("existing", "existing desc")
		_ = existing.Register("test", existingSession)

		if err := existing.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Verify loaded session overwrote existing
		got, ok := existing.Get("test")
		if !ok {
			t.Fatal("session 'test' not found after load")
		}
		if got.Name != "original" {
			t.Errorf("expected loaded session name 'original', got %q", got.Name)
		}
	})

	t.Run("empty directory loads nothing", func(t *testing.T) {
		dir := t.TempDir()

		registry := NewSessionRegistry()
		if err := registry.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		if registry.Count() != 0 {
			t.Errorf("expected 0 sessions from empty dir, got %d", registry.Count())
		}
	})

	t.Run("preserves session ID", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "description")
		originalID := session.ID
		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		if got.ID != originalID {
			t.Errorf("expected ID %q, got %q", originalID, got.ID)
		}
	})

	t.Run("preserves timestamps", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "description")
		originalStartedAt := session.StartedAt
		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		if !got.StartedAt.Equal(originalStartedAt) {
			t.Errorf("expected StartedAt %v, got %v", originalStartedAt, got.StartedAt)
		}
	})
}

// TestLoadNonExistentDir verifies Load returns nil for non-existent directories.
func TestLoadNonExistentDir(t *testing.T) {
	registry := NewSessionRegistry()

	err := registry.Load("/nonexistent/path/that/does/not/exist")
	if err != nil {
		t.Errorf("expected nil error for non-existent dir, got %v", err)
	}

	if registry.Count() != 0 {
		t.Errorf("expected 0 sessions, got %d", registry.Count())
	}
}

// TestSaveLoadRoundtrip verifies Save then Load preserves all data.
func TestSaveLoadRoundtrip(t *testing.T) {
	t.Run("empty registry roundtrip", func(t *testing.T) {
		original := NewSessionRegistry()

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		if loaded.Count() != 0 {
			t.Errorf("expected 0 sessions after roundtrip, got %d", loaded.Count())
		}
	})

	t.Run("single session roundtrip", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")
		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		if loaded.Count() != 1 {
			t.Fatalf("expected 1 session, got %d", loaded.Count())
		}

		got, ok := loaded.Get("test")
		if !ok {
			t.Fatal("session 'test' not found")
		}
		if got.Name != session.Name {
			t.Errorf("Name mismatch: %q != %q", got.Name, session.Name)
		}
		if got.Description != session.Description {
			t.Errorf("Description mismatch: %q != %q", got.Description, session.Description)
		}
		if got.ID != session.ID {
			t.Errorf("ID mismatch: %q != %q", got.ID, session.ID)
		}
	})

	t.Run("multiple sessions roundtrip", func(t *testing.T) {
		original := NewSessionRegistry()
		sessions := map[string]*Session{
			"alpha": NewSession("alpha-session", "first"),
			"beta":  NewSession("beta-session", "second"),
			"gamma": NewSession("gamma-session", "third"),
		}
		for name, s := range sessions {
			_ = original.Register(name, s)
		}

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		if loaded.Count() != 3 {
			t.Errorf("expected 3 sessions, got %d", loaded.Count())
		}

		for name, origSession := range sessions {
			got, ok := loaded.Get(name)
			if !ok {
				t.Errorf("session %q not found after roundtrip", name)
				continue
			}
			if got.Name != origSession.Name {
				t.Errorf("session %q Name mismatch: %q != %q", name, got.Name, origSession.Name)
			}
			if got.ID != origSession.ID {
				t.Errorf("session %q ID mismatch: %q != %q", name, got.ID, origSession.ID)
			}
		}
	})

	t.Run("active session status preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("active-session", "description")
		_ = original.Register("active", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("active")
		if got.EndedAt != nil {
			t.Error("expected active session to have nil EndedAt")
		}

		// Verify appears in Active list
		active := loaded.Active()
		if len(active) != 1 {
			t.Errorf("expected 1 active session, got %d", len(active))
		}
	})

	t.Run("ended session status preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("ended-session", "description")
		session.End() // Mark as ended
		_ = original.Register("ended", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("ended")
		if got.EndedAt == nil {
			t.Error("expected ended session to have non-nil EndedAt")
		}

		// Verify does NOT appear in Active list
		active := loaded.Active()
		if len(active) != 0 {
			t.Errorf("expected 0 active sessions, got %d", len(active))
		}
	})

	t.Run("mix of active and ended sessions", func(t *testing.T) {
		original := NewSessionRegistry()

		activeSession := NewSession("active-session", "active desc")
		endedSession := NewSession("ended-session", "ended desc")
		endedSession.End()

		_ = original.Register("active", activeSession)
		_ = original.Register("ended", endedSession)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Verify counts
		if loaded.Count() != 2 {
			t.Errorf("expected 2 total sessions, got %d", loaded.Count())
		}

		activeList := loaded.Active()
		if len(activeList) != 1 {
			t.Errorf("expected 1 active session, got %d", len(activeList))
		}

		// Verify active session
		gotActive, _ := loaded.Get("active")
		if gotActive.EndedAt != nil {
			t.Error("active session should have nil EndedAt")
		}

		// Verify ended session
		gotEnded, _ := loaded.Get("ended")
		if gotEnded.EndedAt == nil {
			t.Error("ended session should have non-nil EndedAt")
		}
	})

	t.Run("registry name differs from session name", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("session-internal-name", "description")
		_ = original.Register("registry-key", session) // Different from session.Name

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Should be retrievable by registry name, not session name
		got, ok := loaded.Get("registry-key")
		if !ok {
			t.Fatal("session not found by registry key")
		}
		if got.Name != "session-internal-name" {
			t.Errorf("expected session Name 'session-internal-name', got %q", got.Name)
		}

		// Session name should NOT be a registry key
		_, ok = loaded.Get("session-internal-name")
		if ok {
			t.Error("session should not be retrievable by session.Name when registry key differs")
		}
	})
}

// ============================================================================
// Edge Case and Error Tests
// ============================================================================

// TestLoadCorruptedJSON verifies Load returns an error with context for corrupted files.
func TestLoadCorruptedJSON(t *testing.T) {
	t.Run("corrupted session file returns error", func(t *testing.T) {
		dir := t.TempDir()

		// Create a valid manifest
		manifest := `{"version": 1, "sessions": {"test": "test"}}`
		if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(manifest), 0644); err != nil {
			t.Fatalf("failed to write manifest: %v", err)
		}

		// Create a corrupted session file
		corrupted := `{"id": "abc", "name": "test", invalid json here`
		if err := os.WriteFile(filepath.Join(dir, "test.json"), []byte(corrupted), 0644); err != nil {
			t.Fatalf("failed to write corrupted file: %v", err)
		}

		registry := NewSessionRegistry()
		err := registry.Load(dir)
		if err == nil {
			t.Fatal("expected error for corrupted JSON, got nil")
		}

		// Verify error wraps ErrRegistryLoadFailed
		if !errors.Is(err, ErrRegistryLoadFailed) {
			t.Errorf("expected error to wrap ErrRegistryLoadFailed, got %v", err)
		}

		// Verify error message contains file context
		errStr := err.Error()
		if !strings.Contains(errStr, "test.json") {
			t.Errorf("expected error to mention filename, got %q", errStr)
		}
	})

	t.Run("corrupted manifest returns error", func(t *testing.T) {
		dir := t.TempDir()

		// Create a corrupted manifest
		corrupted := `{"version": 1, "sessions": { invalid }`
		if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(corrupted), 0644); err != nil {
			t.Fatalf("failed to write corrupted manifest: %v", err)
		}

		registry := NewSessionRegistry()
		err := registry.Load(dir)
		if err == nil {
			t.Fatal("expected error for corrupted manifest, got nil")
		}

		// Verify error wraps ErrRegistryLoadFailed
		if !errors.Is(err, ErrRegistryLoadFailed) {
			t.Errorf("expected error to wrap ErrRegistryLoadFailed, got %v", err)
		}
	})

	t.Run("empty session file returns error", func(t *testing.T) {
		dir := t.TempDir()

		// Create a valid manifest
		manifest := `{"version": 1, "sessions": {"test": "test"}}`
		if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(manifest), 0644); err != nil {
			t.Fatalf("failed to write manifest: %v", err)
		}

		// Create an empty session file
		if err := os.WriteFile(filepath.Join(dir, "test.json"), []byte(""), 0644); err != nil {
			t.Fatalf("failed to write empty file: %v", err)
		}

		registry := NewSessionRegistry()
		err := registry.Load(dir)
		if err == nil {
			t.Fatal("expected error for empty JSON file, got nil")
		}
	})

	t.Run("valid JSON but wrong structure returns no error but empty fields", func(t *testing.T) {
		dir := t.TempDir()

		// Create a valid manifest
		manifest := `{"version": 1, "sessions": {"test": "test"}}`
		if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(manifest), 0644); err != nil {
			t.Fatalf("failed to write manifest: %v", err)
		}

		// Create a session file with valid JSON but different structure
		// (e.g., array instead of object)
		wrongStructure := `{"foo": "bar", "baz": 123}`
		if err := os.WriteFile(filepath.Join(dir, "test.json"), []byte(wrongStructure), 0644); err != nil {
			t.Fatalf("failed to write file: %v", err)
		}

		registry := NewSessionRegistry()
		err := registry.Load(dir)
		// JSON unmarshaling into struct with missing fields is valid in Go
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Session should exist but with empty/zero fields
		got, ok := registry.Get("test")
		if !ok {
			t.Fatal("session not found after load")
		}
		// Session fields should be empty/zero since JSON didn't match
		if got.ID != "" {
			t.Errorf("expected empty ID for mismatched JSON, got %q", got.ID)
		}
	})
}

// TestSaveSessionWithSpecialChars verifies Save handles special characters in names safely.
func TestSaveSessionWithSpecialChars(t *testing.T) {
	testCases := []struct {
		name        string
		registryKey string
		sessionName string
	}{
		{"slash in name", "test/session", "session/with/slashes"},
		{"backslash in name", "test\\session", "session\\with\\backslashes"},
		{"colon in name", "test:session", "session:with:colons"},
		{"asterisk in name", "test*session", "session*with*asterisks"},
		{"question mark in name", "test?session", "session?with?questions"},
		{"quote in name", `test"session`, `session"with"quotes`},
		{"less than in name", "test<session", "session<with<lessthan"},
		{"greater than in name", "test>session", "session>with>greaterthan"},
		{"pipe in name", "test|session", "session|with|pipes"},
		{"percent in name", "test%session", "session%with%percent"},
		{"mixed special chars", "a/b\\c:d*e?f", "session<a>b|c%d"},
		{"unicode characters", "test-日本語", "セッション-unicode"},
		{"emoji in name", "test-🎉-session", "session-🚀-emoji"},
		{"spaces and dots", "test . session", "  session...dots  "},
		{"empty registry key", "", "empty-key-session"},
		{"very long name", string(make([]byte, 300)), "very-long-session"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			original := NewSessionRegistry()
			session := NewSession(tc.sessionName, "test description")
			_ = original.Register(tc.registryKey, session)

			dir := t.TempDir()

			// Save should not fail
			err := original.Save(dir)
			if err != nil {
				t.Fatalf("Save failed for %q: %v", tc.name, err)
			}

			// Verify manifest exists and is valid JSON
			manifestPath := filepath.Join(dir, "manifest.json")
			data, err := os.ReadFile(manifestPath)
			if err != nil {
				t.Fatalf("failed to read manifest: %v", err)
			}

			var manifest struct {
				Sessions map[string]string `json:"sessions"`
			}
			if err := json.Unmarshal(data, &manifest); err != nil {
				t.Fatalf("manifest is not valid JSON: %v", err)
			}

			// Verify session can be loaded back
			loaded := NewSessionRegistry()
			if err := loaded.Load(dir); err != nil {
				t.Fatalf("Load failed: %v", err)
			}

			got, ok := loaded.Get(tc.registryKey)
			if !ok {
				t.Fatalf("session not found after roundtrip with key %q", tc.registryKey)
			}
			if got.Name != tc.sessionName {
				t.Errorf("session Name mismatch: got %q, want %q", got.Name, tc.sessionName)
			}
			if got.ID != session.ID {
				t.Errorf("session ID mismatch: got %q, want %q", got.ID, session.ID)
			}
		})
	}

	t.Run("multiple sessions with similar names after sanitization", func(t *testing.T) {
		// Test that sessions with names that sanitize to similar values don't collide
		original := NewSessionRegistry()
		session1 := NewSession("session-1", "desc 1")
		session2 := NewSession("session-2", "desc 2")
		session3 := NewSession("session-3", "desc 3")

		// These names might sanitize to similar filenames
		_ = original.Register("test/a", session1)
		_ = original.Register("test\\a", session2)
		_ = original.Register("test:a", session3)

		dir := t.TempDir()

		err := original.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Load and verify all three sessions are distinct
		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		if loaded.Count() != 3 {
			t.Fatalf("expected 3 sessions, got %d", loaded.Count())
		}

		for _, key := range []string{"test/a", "test\\a", "test:a"} {
			if _, ok := loaded.Get(key); !ok {
				t.Errorf("session %q not found after roundtrip", key)
			}
		}
	})
}

// TestLoadSessionWithConversations verifies Load preserves session conversations.
func TestLoadSessionWithConversations(t *testing.T) {
	t.Run("single conversation preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")

		// Add a conversation with messages
		conv := NewConversation("test-conversation")
		conv.Add(&Message{
			ID:        "msg-1",
			Role:      RoleUser,
			Content:   "Hello",
			Timestamp: time.Now(),
		})
		conv.Add(&Message{
			ID:        "msg-2",
			Role:      RoleAssistant,
			Content:   "Hi there!",
			AgentID:   "agent-1",
			AgentName: "TestAgent",
			Timestamp: time.Now(),
		})
		session.AddConversation(conv)

		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, ok := loaded.Get("test")
		if !ok {
			t.Fatal("session not found after load")
		}

		if len(got.Conversations) != 1 {
			t.Fatalf("expected 1 conversation, got %d", len(got.Conversations))
		}

		loadedConv := got.Conversations[0]
		if loadedConv.ID != conv.ID {
			t.Errorf("conversation ID mismatch: got %q, want %q", loadedConv.ID, conv.ID)
		}
		if loadedConv.Name != conv.Name {
			t.Errorf("conversation Name mismatch: got %q, want %q", loadedConv.Name, conv.Name)
		}
		if len(loadedConv.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(loadedConv.Messages))
		}
		if loadedConv.Messages[0].Content != "Hello" {
			t.Errorf("message content mismatch: got %q, want %q", loadedConv.Messages[0].Content, "Hello")
		}
		if loadedConv.Messages[1].AgentID != "agent-1" {
			t.Errorf("message AgentID mismatch: got %q, want %q", loadedConv.Messages[1].AgentID, "agent-1")
		}
	})

	t.Run("multiple conversations preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")

		// Add multiple conversations
		for i := 1; i <= 3; i++ {
			conv := NewConversation(fmt.Sprintf("conversation-%d", i))
			conv.Add(&Message{
				ID:        fmt.Sprintf("msg-%d", i),
				Role:      RoleUser,
				Content:   fmt.Sprintf("Message %d", i),
				Timestamp: time.Now(),
			})
			session.AddConversation(conv)
		}

		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		if len(got.Conversations) != 3 {
			t.Errorf("expected 3 conversations, got %d", len(got.Conversations))
		}
	})

	t.Run("empty conversations slice preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")
		// No conversations added - should preserve empty slice
		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		if got.Conversations == nil {
			t.Error("expected non-nil Conversations slice after load")
		}
		if len(got.Conversations) != 0 {
			t.Errorf("expected 0 conversations, got %d", len(got.Conversations))
		}
	})

	t.Run("conversation participants preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")

		conv := NewConversation("test-conversation")
		conv.Add(&Message{
			ID:        "msg-1",
			Role:      RoleAssistant,
			Content:   "Hello",
			AgentID:   "agent-1",
			AgentName: "Agent One",
			Timestamp: time.Now(),
		})
		conv.Add(&Message{
			ID:        "msg-2",
			Role:      RoleAssistant,
			Content:   "World",
			AgentID:   "agent-2",
			AgentName: "Agent Two",
			Timestamp: time.Now(),
		})
		session.AddConversation(conv)

		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		loadedConv := got.Conversations[0]

		if len(loadedConv.Participants) != 2 {
			t.Errorf("expected 2 participants, got %d", len(loadedConv.Participants))
		}

		if p, ok := loadedConv.Participants["agent-1"]; !ok {
			t.Error("participant agent-1 not found")
		} else if p.AgentName != "Agent One" {
			t.Errorf("participant name mismatch: got %q, want %q", p.AgentName, "Agent One")
		}
	})
}

// TestLoadSessionWithMeasurements verifies Load preserves session measurements.
func TestLoadSessionWithMeasurements(t *testing.T) {
	t.Run("single measurement preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")

		// Add a measurement
		m := NewMeasurement()
		m.DEff = 256
		m.Beta = 1.8
		m.Alignment = 0.85
		m.CPair = 0.72
		m.BetaStatus = BetaOptimal
		m.SenderID = "sender-1"
		m.SenderName = "Sender"
		m.ReceiverID = "receiver-1"
		m.ReceiverName = "Receiver"
		m.TurnNumber = 5
		m.MessageContent = "Test message"
		m.TokenCount = 42

		session.AddMeasurement(m)
		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, ok := loaded.Get("test")
		if !ok {
			t.Fatal("session not found after load")
		}

		if len(got.Measurements) != 1 {
			t.Fatalf("expected 1 measurement, got %d", len(got.Measurements))
		}

		loadedM := got.Measurements[0]
		if loadedM.ID != m.ID {
			t.Errorf("measurement ID mismatch: got %q, want %q", loadedM.ID, m.ID)
		}
		if loadedM.DEff != 256 {
			t.Errorf("DEff mismatch: got %d, want %d", loadedM.DEff, 256)
		}
		if loadedM.Beta != 1.8 {
			t.Errorf("Beta mismatch: got %f, want %f", loadedM.Beta, 1.8)
		}
		if loadedM.Alignment != 0.85 {
			t.Errorf("Alignment mismatch: got %f, want %f", loadedM.Alignment, 0.85)
		}
		if loadedM.CPair != 0.72 {
			t.Errorf("CPair mismatch: got %f, want %f", loadedM.CPair, 0.72)
		}
		if loadedM.BetaStatus != BetaOptimal {
			t.Errorf("BetaStatus mismatch: got %q, want %q", loadedM.BetaStatus, BetaOptimal)
		}
		if loadedM.SenderID != "sender-1" {
			t.Errorf("SenderID mismatch: got %q, want %q", loadedM.SenderID, "sender-1")
		}
		if loadedM.TurnNumber != 5 {
			t.Errorf("TurnNumber mismatch: got %d, want %d", loadedM.TurnNumber, 5)
		}
		if loadedM.MessageContent != "Test message" {
			t.Errorf("MessageContent mismatch: got %q, want %q", loadedM.MessageContent, "Test message")
		}
		if loadedM.TokenCount != 42 {
			t.Errorf("TokenCount mismatch: got %d, want %d", loadedM.TokenCount, 42)
		}
	})

	t.Run("multiple measurements preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")

		// Add multiple measurements with different statuses
		for i := 1; i <= 5; i++ {
			m := NewMeasurement()
			m.DEff = i * 100
			m.Beta = float64(i) * 0.5
			m.SenderID = fmt.Sprintf("sender-%d", i)
			session.AddMeasurement(m)
		}

		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		if len(got.Measurements) != 5 {
			t.Errorf("expected 5 measurements, got %d", len(got.Measurements))
		}
	})

	t.Run("measurement with hidden states preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")

		m := NewMeasurement()
		m.SenderID = "sender-1"
		m.SenderHidden = &HiddenState{
			Vector: []float32{0.1, 0.2, 0.3, 0.4, 0.5},
			Shape:  []int{5},
			Layer:  12,
			DType:  "float32",
		}
		m.ReceiverID = "receiver-1"
		m.ReceiverHidden = &HiddenState{
			Vector: []float32{0.5, 0.4, 0.3, 0.2, 0.1},
			Shape:  []int{5},
			Layer:  24,
			DType:  "float32",
		}

		session.AddMeasurement(m)
		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		loadedM := got.Measurements[0]

		if loadedM.SenderHidden == nil {
			t.Fatal("SenderHidden is nil after load")
		}
		if len(loadedM.SenderHidden.Vector) != 5 {
			t.Errorf("SenderHidden vector length mismatch: got %d, want 5", len(loadedM.SenderHidden.Vector))
		}
		if loadedM.SenderHidden.Layer != 12 {
			t.Errorf("SenderHidden Layer mismatch: got %d, want 12", loadedM.SenderHidden.Layer)
		}

		if loadedM.ReceiverHidden == nil {
			t.Fatal("ReceiverHidden is nil after load")
		}
		if loadedM.ReceiverHidden.DType != "float32" {
			t.Errorf("ReceiverHidden DType mismatch: got %q, want %q", loadedM.ReceiverHidden.DType, "float32")
		}
	})

	t.Run("empty measurements slice preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")
		// No measurements added
		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		if got.Measurements == nil {
			t.Error("expected non-nil Measurements slice after load")
		}
		if len(got.Measurements) != 0 {
			t.Errorf("expected 0 measurements, got %d", len(got.Measurements))
		}
	})

	t.Run("session stats reflect loaded measurements", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")

		// Add measurements with known values
		for i := 0; i < 4; i++ {
			m := NewMeasurement()
			m.DEff = 100
			m.Beta = 2.0
			m.Alignment = 0.5
			m.SenderID = "sender-1"
			if i%2 == 0 {
				m.SenderHidden = &HiddenState{Vector: []float32{0.1}, Shape: []int{1}}
				m.ReceiverHidden = &HiddenState{Vector: []float32{0.2}, Shape: []int{1}}
			}
			session.AddMeasurement(m)
		}

		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		stats := got.Stats()

		if stats.MeasurementCount != 4 {
			t.Errorf("MeasurementCount mismatch: got %d, want 4", stats.MeasurementCount)
		}
		if stats.BilateralCount != 2 {
			t.Errorf("BilateralCount mismatch: got %d, want 2", stats.BilateralCount)
		}
	})
}

// TestSaveNilSession verifies Save handles nil sessions gracefully.
func TestSaveNilSession(t *testing.T) {
	t.Run("nil session is skipped", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("nil-session", nil)
		_ = registry.Register("real-session", NewSession("real", "d"))

		dir := t.TempDir()

		// Save should succeed
		err := registry.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Verify only real session in manifest
		manifestPath := filepath.Join(dir, "manifest.json")
		data, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Fatalf("failed to read manifest: %v", err)
		}

		var manifest struct {
			Sessions map[string]string `json:"sessions"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
			t.Fatalf("failed to parse manifest: %v", err)
		}

		if len(manifest.Sessions) != 1 {
			t.Errorf("expected 1 session in manifest (nil skipped), got %d", len(manifest.Sessions))
		}
		if _, ok := manifest.Sessions["real-session"]; !ok {
			t.Error("real-session not found in manifest")
		}
		if _, ok := manifest.Sessions["nil-session"]; ok {
			t.Error("nil-session should not be in manifest")
		}

		// Verify only one session file exists (plus manifest)
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("failed to read directory: %v", err)
		}

		jsonCount := 0
		for _, entry := range entries {
			if filepath.Ext(entry.Name()) == ".json" {
				jsonCount++
			}
		}
		// Should have 2: manifest.json and real-session.json
		if jsonCount != 2 {
			t.Errorf("expected 2 JSON files, got %d", jsonCount)
		}
	})

	t.Run("all nil sessions results in empty manifest", func(t *testing.T) {
		registry := NewSessionRegistry()
		_ = registry.Register("nil-1", nil)
		_ = registry.Register("nil-2", nil)
		_ = registry.Register("nil-3", nil)

		dir := t.TempDir()

		err := registry.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Verify empty manifest
		manifestPath := filepath.Join(dir, "manifest.json")
		data, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Fatalf("failed to read manifest: %v", err)
		}

		var manifest struct {
			Sessions map[string]string `json:"sessions"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
			t.Fatalf("failed to parse manifest: %v", err)
		}

		if len(manifest.Sessions) != 0 {
			t.Errorf("expected 0 sessions in manifest, got %d", len(manifest.Sessions))
		}
	})
}

// TestLoadFallbackWithoutManifest verifies Load works without manifest.json.
func TestLoadFallbackWithoutManifest(t *testing.T) {
	t.Run("loads sessions by filename when no manifest", func(t *testing.T) {
		dir := t.TempDir()

		// Create session files directly without manifest
		session1 := NewSession("session-1", "description 1")
		data1, _ := json.MarshalIndent(session1, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, "my-session.json"), data1, 0644); err != nil {
			t.Fatalf("failed to write session file: %v", err)
		}

		registry := NewSessionRegistry()
		if err := registry.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Session should be loaded with filename as registry name
		got, ok := registry.Get("my-session")
		if !ok {
			t.Fatal("session not found with filename as key")
		}
		if got.Name != "session-1" {
			t.Errorf("session Name mismatch: got %q, want %q", got.Name, "session-1")
		}
	})

	t.Run("ignores non-json files when no manifest", func(t *testing.T) {
		dir := t.TempDir()

		// Create a JSON session file
		session := NewSession("test-session", "description")
		data, _ := json.MarshalIndent(session, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, "valid.json"), data, 0644); err != nil {
			t.Fatalf("failed to write session file: %v", err)
		}

		// Create some non-JSON files
		if err := os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("readme"), 0644); err != nil {
			t.Fatalf("failed to write text file: %v", err)
		}
		if err := os.WriteFile(filepath.Join(dir, "config.yaml"), []byte("config: value"), 0644); err != nil {
			t.Fatalf("failed to write yaml file: %v", err)
		}

		registry := NewSessionRegistry()
		if err := registry.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Only the JSON session should be loaded
		if registry.Count() != 1 {
			t.Errorf("expected 1 session, got %d", registry.Count())
		}
		if _, ok := registry.Get("valid"); !ok {
			t.Error("valid session not found")
		}
	})

	t.Run("ignores subdirectories", func(t *testing.T) {
		dir := t.TempDir()

		// Create a JSON session file
		session := NewSession("test-session", "description")
		data, _ := json.MarshalIndent(session, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, "valid.json"), data, 0644); err != nil {
			t.Fatalf("failed to write session file: %v", err)
		}

		// Create a subdirectory with JSON files (should be ignored)
		subdir := filepath.Join(dir, "subdir")
		if err := os.MkdirAll(subdir, 0755); err != nil {
			t.Fatalf("failed to create subdir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(subdir, "nested.json"), data, 0644); err != nil {
			t.Fatalf("failed to write nested file: %v", err)
		}

		registry := NewSessionRegistry()
		if err := registry.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Only the top-level JSON session should be loaded
		if registry.Count() != 1 {
			t.Errorf("expected 1 session, got %d", registry.Count())
		}
	})
}

// TestLoadWithSessionMetadata verifies session metadata is preserved.
func TestLoadWithSessionMetadata(t *testing.T) {
	t.Run("session metadata preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")
		session.Metadata["key1"] = "value1"
		session.Metadata["key2"] = 42.5
		session.Metadata["key3"] = true
		session.Metadata["key4"] = []any{"a", "b", "c"}

		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")
		if got.Metadata == nil {
			t.Fatal("Metadata is nil after load")
		}

		if got.Metadata["key1"] != "value1" {
			t.Errorf("Metadata key1 mismatch: got %v, want %v", got.Metadata["key1"], "value1")
		}
		if got.Metadata["key2"] != 42.5 {
			t.Errorf("Metadata key2 mismatch: got %v, want %v", got.Metadata["key2"], 42.5)
		}
		if got.Metadata["key3"] != true {
			t.Errorf("Metadata key3 mismatch: got %v, want %v", got.Metadata["key3"], true)
		}
	})

	t.Run("session config preserved", func(t *testing.T) {
		original := NewSessionRegistry()
		session := NewSession("test-session", "test description")
		session.Config.MeasurementMode = MeasureTriggered
		session.Config.AutoExport = false
		session.Config.ExportPath = "/custom/path"

		_ = original.Register("test", session)

		dir := t.TempDir()
		if err := original.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loaded := NewSessionRegistry()
		if err := loaded.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		got, _ := loaded.Get("test")

		if got.Config.MeasurementMode != MeasureTriggered {
			t.Errorf("MeasurementMode mismatch: got %q, want %q", got.Config.MeasurementMode, MeasureTriggered)
		}
		if got.Config.AutoExport != false {
			t.Errorf("AutoExport mismatch: got %v, want %v", got.Config.AutoExport, false)
		}
		if got.Config.ExportPath != "/custom/path" {
			t.Errorf("ExportPath mismatch: got %q, want %q", got.Config.ExportPath, "/custom/path")
		}
	})
}

// ============================================================================
// Save/Load Concurrency Tests
// ============================================================================

// TestConcurrentSave verifies thread-safety of Save operations.
func TestConcurrentSave(t *testing.T) {
	t.Run("multiple concurrent Save calls", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numSessions = 10

		// Pre-populate registry with sessions
		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = registry.Register(name, session)
		}

		const numGoroutines = 20
		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines)

		// Multiple concurrent saves to different directories
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				dir := filepath.Join(t.TempDir(), fmt.Sprintf("save-%d", idx))
				if err := registry.Save(dir); err != nil {
					errors <- err
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			t.Errorf("unexpected error during concurrent save: %v", err)
		}
	})

	t.Run("concurrent Save to same directory", func(t *testing.T) {
		registry := NewSessionRegistry()
		const numSessions = 5

		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = registry.Register(name, session)
		}

		dir := t.TempDir()
		const numGoroutines = 10
		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines)

		// Multiple concurrent saves to the same directory
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := registry.Save(dir); err != nil {
					errors <- err
				}
			}()
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			t.Errorf("unexpected error during concurrent save: %v", err)
		}

		// Verify directory contains expected files
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("failed to read directory: %v", err)
		}
		// Should have manifest.json + numSessions session files
		if len(entries) != numSessions+1 {
			t.Errorf("expected %d files, got %d", numSessions+1, len(entries))
		}
	})
}

// TestConcurrentLoad verifies thread-safety of Load operations.
func TestConcurrentLoad(t *testing.T) {
	t.Run("multiple concurrent Load calls", func(t *testing.T) {
		// Create and save a registry to disk
		source := NewSessionRegistry()
		const numSessions = 10

		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = source.Register(name, session)
		}

		dir := t.TempDir()
		if err := source.Save(dir); err != nil {
			t.Fatalf("failed to save source registry: %v", err)
		}

		const numGoroutines = 20
		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines)
		results := make(chan int, numGoroutines)

		// Multiple concurrent loads from the same directory into different registries
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				registry := NewSessionRegistry()
				if err := registry.Load(dir); err != nil {
					errors <- err
					return
				}
				results <- registry.Count()
			}()
		}

		wg.Wait()
		close(errors)
		close(results)

		for err := range errors {
			t.Errorf("unexpected error during concurrent load: %v", err)
		}

		for count := range results {
			if count != numSessions {
				t.Errorf("expected %d sessions after load, got %d", numSessions, count)
			}
		}
	})

	t.Run("concurrent Load into same registry", func(t *testing.T) {
		// Create and save a registry to disk
		source := NewSessionRegistry()
		const numSessions = 5

		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = source.Register(name, session)
		}

		dir := t.TempDir()
		if err := source.Save(dir); err != nil {
			t.Fatalf("failed to save source registry: %v", err)
		}

		registry := NewSessionRegistry()
		const numGoroutines = 20
		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines)

		// Multiple concurrent loads into the same registry
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := registry.Load(dir); err != nil {
					errors <- err
				}
			}()
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			t.Errorf("unexpected error during concurrent load: %v", err)
		}

		// All sessions should be present after concurrent loads
		if registry.Count() != numSessions {
			t.Errorf("expected %d sessions, got %d", numSessions, registry.Count())
		}
	})
}

// TestSaveDuringModifications verifies Save works correctly when registry is being modified.
func TestSaveDuringModifications(t *testing.T) {
	t.Run("Save while registering sessions", func(t *testing.T) {
		registry := NewSessionRegistry()
		dir := t.TempDir()

		const numRegisters = 50
		const numSaves = 10
		var wg sync.WaitGroup

		// Register goroutines
		for i := 0; i < numRegisters; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("session-%d", idx)
				session := NewSession(name, fmt.Sprintf("description %d", idx))
				_ = registry.Register(name, session) // Ignore duplicate errors
			}(i)
		}

		// Save goroutines
		saveErrors := make(chan error, numSaves)
		for i := 0; i < numSaves; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				saveDir := filepath.Join(dir, fmt.Sprintf("save-%d", idx))
				if err := registry.Save(saveDir); err != nil {
					saveErrors <- err
				}
			}(i)
		}

		wg.Wait()
		close(saveErrors)

		for err := range saveErrors {
			t.Errorf("unexpected error during concurrent save: %v", err)
		}
	})

	t.Run("Save while unregistering sessions", func(t *testing.T) {
		registry := NewSessionRegistry()
		dir := t.TempDir()

		// Pre-populate registry
		const numSessions = 50
		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = registry.Register(name, session)
		}

		const numUnregisters = 25
		const numSaves = 10
		var wg sync.WaitGroup

		// Unregister goroutines
		for i := 0; i < numUnregisters; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("session-%d", idx)
				_ = registry.Unregister(name) // Ignore not-found errors
			}(i)
		}

		// Save goroutines
		saveErrors := make(chan error, numSaves)
		for i := 0; i < numSaves; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				saveDir := filepath.Join(dir, fmt.Sprintf("save-%d", idx))
				if err := registry.Save(saveDir); err != nil {
					saveErrors <- err
				}
			}(i)
		}

		wg.Wait()
		close(saveErrors)

		for err := range saveErrors {
			t.Errorf("unexpected error during concurrent save/unregister: %v", err)
		}
	})

	t.Run("Save while using GetOrCreate", func(t *testing.T) {
		registry := NewSessionRegistry()
		dir := t.TempDir()

		const numCreates = 50
		const numSaves = 10
		var wg sync.WaitGroup

		// GetOrCreate goroutines
		for i := 0; i < numCreates; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("session-%d", idx%10) // Some will collide
				_, _ = registry.GetOrCreate(name, fmt.Sprintf("description %d", idx))
			}(i)
		}

		// Save goroutines
		saveErrors := make(chan error, numSaves)
		for i := 0; i < numSaves; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				saveDir := filepath.Join(dir, fmt.Sprintf("save-%d", idx))
				if err := registry.Save(saveDir); err != nil {
					saveErrors <- err
				}
			}(i)
		}

		wg.Wait()
		close(saveErrors)

		for err := range saveErrors {
			t.Errorf("unexpected error during concurrent save/create: %v", err)
		}
	})
}

// TestLoadWithConcurrentReads verifies Load works correctly with concurrent reads.
func TestLoadWithConcurrentReads(t *testing.T) {
	t.Run("Load with concurrent Get calls", func(t *testing.T) {
		// Create and save source registry
		source := NewSessionRegistry()
		const numSessions = 10

		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = source.Register(name, session)
		}

		dir := t.TempDir()
		if err := source.Save(dir); err != nil {
			t.Fatalf("failed to save source registry: %v", err)
		}

		registry := NewSessionRegistry()
		const numLoads = 5
		const numGets = 50
		var wg sync.WaitGroup

		// Load goroutines
		loadErrors := make(chan error, numLoads)
		for i := 0; i < numLoads; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := registry.Load(dir); err != nil {
					loadErrors <- err
				}
			}()
		}

		// Get goroutines - accessing potentially loaded sessions
		for i := 0; i < numGets; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				name := fmt.Sprintf("session-%d", idx%numSessions)
				_, _ = registry.Get(name) // Result doesn't matter, testing for race
			}(i)
		}

		wg.Wait()
		close(loadErrors)

		for err := range loadErrors {
			t.Errorf("unexpected error during concurrent load/get: %v", err)
		}
	})

	t.Run("Load with concurrent List calls", func(t *testing.T) {
		// Create and save source registry
		source := NewSessionRegistry()
		const numSessions = 10

		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = source.Register(name, session)
		}

		dir := t.TempDir()
		if err := source.Save(dir); err != nil {
			t.Fatalf("failed to save source registry: %v", err)
		}

		registry := NewSessionRegistry()
		const numLoads = 5
		const numLists = 50
		var wg sync.WaitGroup

		// Load goroutines
		loadErrors := make(chan error, numLoads)
		for i := 0; i < numLoads; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := registry.Load(dir); err != nil {
					loadErrors <- err
				}
			}()
		}

		// List goroutines
		for i := 0; i < numLists; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = registry.List() // Result doesn't matter, testing for race
			}()
		}

		wg.Wait()
		close(loadErrors)

		for err := range loadErrors {
			t.Errorf("unexpected error during concurrent load/list: %v", err)
		}
	})

	t.Run("Load with concurrent Status calls", func(t *testing.T) {
		// Create and save source registry
		source := NewSessionRegistry()
		const numSessions = 10

		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = source.Register(name, session)
		}

		dir := t.TempDir()
		if err := source.Save(dir); err != nil {
			t.Fatalf("failed to save source registry: %v", err)
		}

		registry := NewSessionRegistry()
		const numLoads = 5
		const numStatus = 50
		var wg sync.WaitGroup

		// Load goroutines
		loadErrors := make(chan error, numLoads)
		for i := 0; i < numLoads; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := registry.Load(dir); err != nil {
					loadErrors <- err
				}
			}()
		}

		// Status goroutines
		for i := 0; i < numStatus; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = registry.Status() // Result doesn't matter, testing for race
			}()
		}

		wg.Wait()
		close(loadErrors)

		for err := range loadErrors {
			t.Errorf("unexpected error during concurrent load/status: %v", err)
		}
	})
}

// TestConcurrentSaveLoad verifies combined Save and Load operations are thread-safe.
func TestConcurrentSaveLoad(t *testing.T) {
	t.Run("concurrent Save and Load on different registries", func(t *testing.T) {
		dir := t.TempDir()
		const numSessions = 5

		// Create source registry and save it
		source := NewSessionRegistry()
		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = source.Register(name, session)
		}
		if err := source.Save(dir); err != nil {
			t.Fatalf("failed to save source: %v", err)
		}

		const numSaves = 10
		const numLoads = 10
		var wg sync.WaitGroup

		// Save goroutines - saving source registry
		for i := 0; i < numSaves; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				// Ignore errors - concurrent save/load to same directory may produce
				// partial reads, which is expected filesystem behavior
				_ = source.Save(dir)
			}()
		}

		// Load goroutines - loading into new registries
		for i := 0; i < numLoads; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				registry := NewSessionRegistry()
				// Ignore errors - concurrent save/load to same directory may produce
				// partial reads, which is expected filesystem behavior
				_ = registry.Load(dir)
			}()
		}

		wg.Wait()
		// No assertions - this test verifies no data races occur (run with -race)
	})

	t.Run("stress test Save and Load", func(t *testing.T) {
		const numSessions = 10
		const numGoroutines = 50
		var wg sync.WaitGroup

		// Create shared registry with sessions
		registry := NewSessionRegistry()
		for i := 0; i < numSessions; i++ {
			name := fmt.Sprintf("session-%d", i)
			session := NewSession(name, fmt.Sprintf("description %d", i))
			_ = registry.Register(name, session)
		}

		baseDir := t.TempDir()

		// Mix of saves and loads
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				dir := filepath.Join(baseDir, fmt.Sprintf("dir-%d", idx%5))
				if idx%2 == 0 {
					_ = registry.Save(dir) // Some will succeed, some may fail
				} else {
					_ = registry.Load(dir) // Some will succeed, some may fail (dir may not exist)
				}
			}(i)
		}

		wg.Wait()
		// No assertions - just checking for race conditions
	})
}
