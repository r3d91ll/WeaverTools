package yarn

import (
	"testing"
	"time"
)

// TestNewConversationStore verifies the constructor initializes correctly.
func TestNewConversationStore(t *testing.T) {
	store := NewConversationStore()
	if store == nil {
		t.Fatal("NewConversationStore returned nil")
	}
	if store.conversations == nil {
		t.Fatal("conversations map is nil")
	}
	if store.Count() != 0 {
		t.Errorf("new store should have count 0, got %d", store.Count())
	}
}

// TestAdd tests adding conversations to the store.
func TestAdd(t *testing.T) {
	t.Run("add single conversation", func(t *testing.T) {
		store := NewConversationStore()
		conv := NewConversation("test-conversation")

		beforeAdd := time.Now()
		store.Add(conv)
		afterAdd := time.Now()

		if store.Count() != 1 {
			t.Errorf("expected count 1, got %d", store.Count())
		}

		// Verify UpdatedAt was set
		retrieved, ok := store.Get(conv.ID)
		if !ok {
			t.Fatal("failed to retrieve added conversation")
		}
		if retrieved.UpdatedAt.Before(beforeAdd) || retrieved.UpdatedAt.After(afterAdd) {
			t.Error("UpdatedAt timestamp not set correctly")
		}
	})

	t.Run("add multiple conversations", func(t *testing.T) {
		store := NewConversationStore()

		conv1 := NewConversation("conversation-1")
		conv2 := NewConversation("conversation-2")
		conv3 := NewConversation("conversation-3")

		store.Add(conv1)
		store.Add(conv2)
		store.Add(conv3)

		if store.Count() != 3 {
			t.Errorf("expected count 3, got %d", store.Count())
		}
	})

	t.Run("update existing conversation", func(t *testing.T) {
		store := NewConversationStore()
		conv := NewConversation("test-conversation")
		store.Add(conv)

		// Modify and re-add
		conv.Name = "updated-name"
		store.Add(conv)

		if store.Count() != 1 {
			t.Errorf("expected count 1 after update, got %d", store.Count())
		}

		retrieved, ok := store.Get(conv.ID)
		if !ok {
			t.Fatal("failed to retrieve updated conversation")
		}
		if retrieved.Name != "updated-name" {
			t.Errorf("expected name 'updated-name', got '%s'", retrieved.Name)
		}
	})
}

// TestGet tests retrieving conversations from the store.
func TestGet(t *testing.T) {
	t.Run("get existing conversation", func(t *testing.T) {
		store := NewConversationStore()
		conv := NewConversation("test-conversation")
		conv.Metadata["key"] = "value"
		msg := NewMessage(RoleUser, "Hello")
		conv.Add(msg)
		store.Add(conv)

		retrieved, ok := store.Get(conv.ID)
		if !ok {
			t.Fatal("expected ok=true for existing conversation")
		}
		if retrieved.ID != conv.ID {
			t.Errorf("expected ID %s, got %s", conv.ID, retrieved.ID)
		}
		if retrieved.Name != conv.Name {
			t.Errorf("expected Name %s, got %s", conv.Name, retrieved.Name)
		}
		if len(retrieved.Messages) != 1 {
			t.Errorf("expected 1 message, got %d", len(retrieved.Messages))
		}
		if retrieved.Metadata["key"] != "value" {
			t.Errorf("expected metadata key=value, got %v", retrieved.Metadata["key"])
		}
	})

	t.Run("get non-existent conversation", func(t *testing.T) {
		store := NewConversationStore()
		retrieved, ok := store.Get("non-existent")
		if ok {
			t.Error("expected ok=false for non-existent conversation")
		}
		if retrieved != nil {
			t.Error("expected nil for non-existent conversation")
		}
	})

	t.Run("get returns copy", func(t *testing.T) {
		store := NewConversationStore()
		conv := NewConversation("test-conversation")
		msg := NewMessage(RoleUser, "Hello")
		conv.Add(msg)
		conv.Metadata["key"] = "original"
		store.Add(conv)

		// Get and modify the copy
		retrieved, _ := store.Get(conv.ID)
		retrieved.Name = "modified-name"
		retrieved.Messages = append(retrieved.Messages, NewMessage(RoleAssistant, "World"))
		retrieved.Metadata["key"] = "modified"
		retrieved.Participants["new-participant"] = Participant{AgentID: "agent-1"}

		// Verify original is unchanged
		original, _ := store.Get(conv.ID)
		if original.Name == "modified-name" {
			t.Error("modifying copy affected original Name")
		}
		if len(original.Messages) != 1 {
			t.Errorf("modifying copy affected original Messages: expected 1, got %d", len(original.Messages))
		}
		if original.Metadata["key"] == "modified" {
			t.Error("modifying copy affected original Metadata")
		}
		if _, exists := original.Participants["new-participant"]; exists {
			t.Error("modifying copy affected original Participants")
		}
	})
}

// TestList tests listing all conversations.
func TestList(t *testing.T) {
	t.Run("list empty store", func(t *testing.T) {
		store := NewConversationStore()
		list := store.List()
		if len(list) != 0 {
			t.Errorf("expected empty list, got %d items", len(list))
		}
	})

	t.Run("list with conversations", func(t *testing.T) {
		store := NewConversationStore()

		conv1 := NewConversation("conversation-1")
		conv1.Add(NewMessage(RoleUser, "msg1"))
		conv1.Add(NewMessage(RoleAssistant, "msg2"))

		conv2 := NewConversation("conversation-2")
		conv2.Add(NewMessage(RoleUser, "msg1"))

		store.Add(conv1)
		store.Add(conv2)

		list := store.List()
		if len(list) != 2 {
			t.Errorf("expected 2 items, got %d", len(list))
		}

		// Verify message counts
		if list[conv1.ID] != 2 {
			t.Errorf("expected conv1 message count 2, got %d", list[conv1.ID])
		}
		if list[conv2.ID] != 1 {
			t.Errorf("expected conv2 message count 1, got %d", list[conv2.ID])
		}
	})
}

// TestCount tests counting conversations.
func TestCount(t *testing.T) {
	store := NewConversationStore()

	if store.Count() != 0 {
		t.Errorf("expected count 0, got %d", store.Count())
	}

	store.Add(NewConversation("conv-1"))
	if store.Count() != 1 {
		t.Errorf("expected count 1, got %d", store.Count())
	}

	store.Add(NewConversation("conv-2"))
	if store.Count() != 2 {
		t.Errorf("expected count 2, got %d", store.Count())
	}
}

// TestClear tests removing a specific conversation.
func TestClear(t *testing.T) {
	t.Run("clear existing conversation", func(t *testing.T) {
		store := NewConversationStore()
		conv := NewConversation("test-conversation")
		store.Add(conv)

		removed := store.Clear(conv.ID)
		if !removed {
			t.Error("expected true for removing existing conversation")
		}
		if store.Count() != 0 {
			t.Errorf("expected count 0 after clear, got %d", store.Count())
		}
	})

	t.Run("clear non-existent conversation", func(t *testing.T) {
		store := NewConversationStore()
		removed := store.Clear("non-existent")
		if removed {
			t.Error("expected false for removing non-existent conversation")
		}
	})

	t.Run("clear one of many", func(t *testing.T) {
		store := NewConversationStore()
		conv1 := NewConversation("conv-1")
		conv2 := NewConversation("conv-2")
		conv3 := NewConversation("conv-3")

		store.Add(conv1)
		store.Add(conv2)
		store.Add(conv3)

		store.Clear(conv2.ID)

		if store.Count() != 2 {
			t.Errorf("expected count 2, got %d", store.Count())
		}

		_, exists := store.Get(conv2.ID)
		if exists {
			t.Error("conv2 should have been removed")
		}

		// Verify others still exist
		if _, exists := store.Get(conv1.ID); !exists {
			t.Error("conv1 should still exist")
		}
		if _, exists := store.Get(conv3.ID); !exists {
			t.Error("conv3 should still exist")
		}
	})
}

// TestClearAll tests removing all conversations.
func TestClearAll(t *testing.T) {
	t.Run("clear all from empty store", func(t *testing.T) {
		store := NewConversationStore()
		count := store.ClearAll()
		if count != 0 {
			t.Errorf("expected 0 removed from empty store, got %d", count)
		}
	})

	t.Run("clear all from populated store", func(t *testing.T) {
		store := NewConversationStore()
		store.Add(NewConversation("conv-1"))
		store.Add(NewConversation("conv-2"))
		store.Add(NewConversation("conv-3"))

		count := store.ClearAll()
		if count != 3 {
			t.Errorf("expected 3 removed, got %d", count)
		}
		if store.Count() != 0 {
			t.Errorf("expected count 0 after ClearAll, got %d", store.Count())
		}
	})

	t.Run("store is usable after ClearAll", func(t *testing.T) {
		store := NewConversationStore()
		store.Add(NewConversation("conv-1"))
		store.ClearAll()

		// Should be able to add new conversations
		conv := NewConversation("new-conv")
		store.Add(conv)

		if store.Count() != 1 {
			t.Errorf("expected count 1, got %d", store.Count())
		}

		retrieved, ok := store.Get(conv.ID)
		if !ok || retrieved.Name != "new-conv" {
			t.Error("failed to add conversation after ClearAll")
		}
	})
}

// TestEdgeCases tests various edge cases.
func TestEdgeCases(t *testing.T) {
	t.Run("empty conversation name", func(t *testing.T) {
		store := NewConversationStore()
		conv := NewConversation("")
		store.Add(conv)

		retrieved, ok := store.Get(conv.ID)
		if !ok {
			t.Fatal("failed to retrieve conversation with empty name")
		}
		if retrieved.Name != "" {
			t.Errorf("expected empty name, got '%s'", retrieved.Name)
		}
	})

	t.Run("conversation with nil slices/maps after creation", func(t *testing.T) {
		store := NewConversationStore()
		// Create a conversation manually without using NewConversation
		conv := &Conversation{
			ID:   "manual-id",
			Name: "manual-conversation",
			// Messages, Participants, Metadata are nil
		}
		store.Add(conv)

		retrieved, ok := store.Get(conv.ID)
		if !ok {
			t.Fatal("failed to retrieve manually created conversation")
		}
		if retrieved.ID != "manual-id" {
			t.Errorf("expected ID 'manual-id', got '%s'", retrieved.ID)
		}
	})

	t.Run("conversation with participants and metadata", func(t *testing.T) {
		store := NewConversationStore()
		conv := NewConversation("full-conversation")
		conv.Participants["agent-1"] = Participant{
			AgentID:      "agent-1",
			AgentName:    "Test Agent",
			Role:         "assistant",
			JoinedAt:     time.Now(),
			MessageCount: 5,
		}
		conv.Metadata["session_id"] = "session-123"
		conv.Metadata["nested"] = map[string]any{"a": 1, "b": 2}
		store.Add(conv)

		retrieved, ok := store.Get(conv.ID)
		if !ok {
			t.Fatal("failed to retrieve conversation")
		}

		// Verify participant
		if p, exists := retrieved.Participants["agent-1"]; !exists {
			t.Error("expected participant agent-1 to exist")
		} else if p.AgentName != "Test Agent" {
			t.Errorf("expected AgentName 'Test Agent', got '%s'", p.AgentName)
		}

		// Verify metadata
		if retrieved.Metadata["session_id"] != "session-123" {
			t.Errorf("expected session_id 'session-123', got '%v'", retrieved.Metadata["session_id"])
		}
	})
}
