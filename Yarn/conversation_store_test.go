package yarn

import (
	"encoding/json"
	"os"
	"path/filepath"
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

// TestSave tests that conversations are persisted to JSON files.
func TestSave(t *testing.T) {
	t.Run("save creates JSON files", func(t *testing.T) {
		store := NewConversationStore()
		conv1 := NewConversation("conversation-1")
		conv1.Add(NewMessage(RoleUser, "Hello"))
		conv2 := NewConversation("conversation-2")
		conv2.Add(NewMessage(RoleAssistant, "Hi there"))
		store.Add(conv1)
		store.Add(conv2)

		dir := t.TempDir()
		err := store.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Verify files were created
		file1 := filepath.Join(dir, conv1.ID+".json")
		file2 := filepath.Join(dir, conv2.ID+".json")

		if _, err := os.Stat(file1); os.IsNotExist(err) {
			t.Errorf("expected file %s to exist", file1)
		}
		if _, err := os.Stat(file2); os.IsNotExist(err) {
			t.Errorf("expected file %s to exist", file2)
		}
	})

	t.Run("save creates valid JSON", func(t *testing.T) {
		store := NewConversationStore()
		conv := NewConversation("test-conversation")
		conv.Add(NewMessage(RoleUser, "Hello"))
		conv.Metadata["key"] = "value"
		store.Add(conv)

		dir := t.TempDir()
		if err := store.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Read and parse the JSON file
		filePath := filepath.Join(dir, conv.ID+".json")
		data, err := os.ReadFile(filePath)
		if err != nil {
			t.Fatalf("failed to read saved file: %v", err)
		}

		var loaded Conversation
		if err := json.Unmarshal(data, &loaded); err != nil {
			t.Fatalf("saved file is not valid JSON: %v", err)
		}

		if loaded.ID != conv.ID {
			t.Errorf("expected ID %s, got %s", conv.ID, loaded.ID)
		}
		if loaded.Name != conv.Name {
			t.Errorf("expected Name %s, got %s", conv.Name, loaded.Name)
		}
		if len(loaded.Messages) != 1 {
			t.Errorf("expected 1 message, got %d", len(loaded.Messages))
		}
	})

	t.Run("save empty store", func(t *testing.T) {
		store := NewConversationStore()
		dir := t.TempDir()

		err := store.Save(dir)
		if err != nil {
			t.Fatalf("Save failed for empty store: %v", err)
		}

		// Should have no JSON files
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("failed to read directory: %v", err)
		}
		if len(entries) != 0 {
			t.Errorf("expected empty directory, got %d entries", len(entries))
		}
	})

	t.Run("save creates directory if needed", func(t *testing.T) {
		store := NewConversationStore()
		conv := NewConversation("test")
		store.Add(conv)

		dir := filepath.Join(t.TempDir(), "nested", "path")
		err := store.Save(dir)
		if err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Verify directory and file exist
		filePath := filepath.Join(dir, conv.ID+".json")
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			t.Errorf("expected file %s to exist", filePath)
		}
	})
}

// TestLoad tests loading conversations from JSON files.
func TestLoad(t *testing.T) {
	t.Run("load from saved conversations", func(t *testing.T) {
		// First save some conversations
		originalStore := NewConversationStore()
		conv := NewConversation("test-conversation")
		conv.Add(NewMessage(RoleUser, "Hello"))
		conv.Add(NewMessage(RoleAssistant, "Hi there"))
		originalStore.Add(conv)

		dir := t.TempDir()
		if err := originalStore.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Now load into a new store
		newStore := NewConversationStore()
		if err := newStore.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		if newStore.Count() != 1 {
			t.Errorf("expected 1 conversation, got %d", newStore.Count())
		}

		loaded, ok := newStore.Get(conv.ID)
		if !ok {
			t.Fatal("failed to get loaded conversation")
		}
		if loaded.Name != "test-conversation" {
			t.Errorf("expected name 'test-conversation', got '%s'", loaded.Name)
		}
		if len(loaded.Messages) != 2 {
			t.Errorf("expected 2 messages, got %d", len(loaded.Messages))
		}
	})

	t.Run("load skips non-JSON files", func(t *testing.T) {
		dir := t.TempDir()

		// Create a conversation JSON file
		conv := NewConversation("test")
		data, _ := json.Marshal(conv)
		if err := os.WriteFile(filepath.Join(dir, conv.ID+".json"), data, 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		// Create a non-JSON file
		if err := os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("not json"), 0644); err != nil {
			t.Fatalf("failed to write txt file: %v", err)
		}

		store := NewConversationStore()
		if err := store.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		if store.Count() != 1 {
			t.Errorf("expected 1 conversation (skipping txt), got %d", store.Count())
		}
	})

	t.Run("load skips directories", func(t *testing.T) {
		dir := t.TempDir()

		// Create a subdirectory with .json extension (unusual but possible)
		subdir := filepath.Join(dir, "subdir.json")
		if err := os.Mkdir(subdir, 0755); err != nil {
			t.Fatalf("failed to create subdirectory: %v", err)
		}

		// Create a valid conversation file
		conv := NewConversation("test")
		data, _ := json.Marshal(conv)
		if err := os.WriteFile(filepath.Join(dir, conv.ID+".json"), data, 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		store := NewConversationStore()
		if err := store.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		if store.Count() != 1 {
			t.Errorf("expected 1 conversation (skipping directory), got %d", store.Count())
		}
	})

	t.Run("load merges with existing conversations", func(t *testing.T) {
		dir := t.TempDir()

		// Create a conversation file
		conv1 := NewConversation("conv-from-file")
		data, _ := json.Marshal(conv1)
		if err := os.WriteFile(filepath.Join(dir, conv1.ID+".json"), data, 0644); err != nil {
			t.Fatalf("failed to write test file: %v", err)
		}

		// Create a store with an existing conversation
		store := NewConversationStore()
		conv2 := NewConversation("existing-conv")
		store.Add(conv2)

		// Load should merge
		if err := store.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		if store.Count() != 2 {
			t.Errorf("expected 2 conversations after merge, got %d", store.Count())
		}

		if _, ok := store.Get(conv1.ID); !ok {
			t.Error("loaded conversation not found")
		}
		if _, ok := store.Get(conv2.ID); !ok {
			t.Error("existing conversation lost after load")
		}
	})
}

// TestLoadNonExistent tests loading from a non-existent directory.
func TestLoadNonExistent(t *testing.T) {
	store := NewConversationStore()
	err := store.Load("/non/existent/path/that/does/not/exist")
	if err != nil {
		t.Errorf("Load should return nil for non-existent directory, got: %v", err)
	}
	if store.Count() != 0 {
		t.Errorf("expected count 0, got %d", store.Count())
	}
}

// TestSaveLoad tests round-trip persistence.
func TestSaveLoad(t *testing.T) {
	t.Run("round-trip preserves conversation data", func(t *testing.T) {
		// Create a store with rich conversation data
		originalStore := NewConversationStore()
		conv := NewConversation("rich-conversation")

		// Add messages
		msg1 := NewAgentMessage(RoleUser, "Hello, how are you?", "user-1", "Alice")
		msg2 := NewAgentMessage(RoleAssistant, "I'm doing well, thanks!", "agent-1", "Bot")
		conv.Add(msg1)
		conv.Add(msg2)

		// Add participant data
		conv.Participants["agent-1"] = Participant{
			AgentID:      "agent-1",
			AgentName:    "Bot",
			Role:         "assistant",
			JoinedAt:     time.Now().Truncate(time.Second), // Truncate for JSON precision
			MessageCount: 1,
		}

		// Add metadata
		conv.Metadata["session_id"] = "session-123"
		conv.Metadata["tags"] = []string{"test", "round-trip"}

		originalStore.Add(conv)

		// Save to temp directory
		dir := t.TempDir()
		if err := originalStore.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Load into new store
		loadedStore := NewConversationStore()
		if err := loadedStore.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Verify loaded data
		loaded, ok := loadedStore.Get(conv.ID)
		if !ok {
			t.Fatal("failed to get loaded conversation")
		}

		// Verify basic fields
		if loaded.ID != conv.ID {
			t.Errorf("ID mismatch: expected %s, got %s", conv.ID, loaded.ID)
		}
		if loaded.Name != conv.Name {
			t.Errorf("Name mismatch: expected %s, got %s", conv.Name, loaded.Name)
		}

		// Verify messages
		if len(loaded.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(loaded.Messages))
		}
		if loaded.Messages[0].Content != "Hello, how are you?" {
			t.Errorf("first message content mismatch: got %s", loaded.Messages[0].Content)
		}
		if loaded.Messages[1].AgentName != "Bot" {
			t.Errorf("second message agent name mismatch: got %s", loaded.Messages[1].AgentName)
		}

		// Verify participants
		if p, exists := loaded.Participants["agent-1"]; !exists {
			t.Error("participant agent-1 not found after load")
		} else {
			if p.AgentName != "Bot" {
				t.Errorf("participant AgentName mismatch: expected 'Bot', got '%s'", p.AgentName)
			}
			if p.MessageCount != 1 {
				t.Errorf("participant MessageCount mismatch: expected 1, got %d", p.MessageCount)
			}
		}

		// Verify metadata
		if loaded.Metadata["session_id"] != "session-123" {
			t.Errorf("metadata session_id mismatch: got %v", loaded.Metadata["session_id"])
		}
	})

	t.Run("round-trip with multiple conversations", func(t *testing.T) {
		originalStore := NewConversationStore()

		// Create multiple conversations
		for i := 0; i < 5; i++ {
			conv := NewConversation("conversation-" + string(rune('A'+i)))
			conv.Add(NewMessage(RoleUser, "Message in conv "+string(rune('A'+i))))
			originalStore.Add(conv)
		}

		// Save and load
		dir := t.TempDir()
		if err := originalStore.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loadedStore := NewConversationStore()
		if err := loadedStore.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Verify count
		if loadedStore.Count() != 5 {
			t.Errorf("expected 5 conversations, got %d", loadedStore.Count())
		}

		// Verify each conversation exists with correct data
		originalList := originalStore.List()
		for id, msgCount := range originalList {
			loaded, ok := loadedStore.Get(id)
			if !ok {
				t.Errorf("conversation %s not found after load", id)
				continue
			}
			if len(loaded.Messages) != msgCount {
				t.Errorf("conversation %s message count mismatch: expected %d, got %d",
					id, msgCount, len(loaded.Messages))
			}
		}
	})

	t.Run("round-trip with hidden states", func(t *testing.T) {
		originalStore := NewConversationStore()
		conv := NewConversation("hidden-state-conversation")

		// Create message with hidden state
		msg := NewMessage(RoleAssistant, "Response with hidden state")
		msg.WithHiddenState(&HiddenState{
			Vector: []float32{0.1, 0.2, 0.3, 0.4, 0.5},
			Shape:  []int{1, 1, 5},
			Layer:  12,
			DType:  "float32",
		})
		conv.Add(msg)
		originalStore.Add(conv)

		// Save and load
		dir := t.TempDir()
		if err := originalStore.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loadedStore := NewConversationStore()
		if err := loadedStore.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		// Verify hidden state was preserved
		loaded, _ := loadedStore.Get(conv.ID)
		if len(loaded.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(loaded.Messages))
		}

		hs := loaded.Messages[0].HiddenState
		if hs == nil {
			t.Fatal("hidden state is nil after load")
		}
		if len(hs.Vector) != 5 {
			t.Errorf("hidden state vector length mismatch: expected 5, got %d", len(hs.Vector))
		}
		if hs.Layer != 12 {
			t.Errorf("hidden state layer mismatch: expected 12, got %d", hs.Layer)
		}
		if hs.DType != "float32" {
			t.Errorf("hidden state dtype mismatch: expected 'float32', got '%s'", hs.DType)
		}
	})

	t.Run("round-trip empty conversation", func(t *testing.T) {
		originalStore := NewConversationStore()
		conv := NewConversation("empty-conversation")
		// No messages added
		originalStore.Add(conv)

		dir := t.TempDir()
		if err := originalStore.Save(dir); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		loadedStore := NewConversationStore()
		if err := loadedStore.Load(dir); err != nil {
			t.Fatalf("Load failed: %v", err)
		}

		loaded, ok := loadedStore.Get(conv.ID)
		if !ok {
			t.Fatal("empty conversation not found after load")
		}
		if len(loaded.Messages) != 0 {
			t.Errorf("expected 0 messages, got %d", len(loaded.Messages))
		}
	})
}
