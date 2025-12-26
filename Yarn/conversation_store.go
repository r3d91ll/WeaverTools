package yarn

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ConversationStore manages conversations in memory with optional persistence.
// It provides thread-safe storage for conversations with JSON file persistence.
type ConversationStore struct {
	mu            sync.RWMutex
	conversations map[string]*Conversation
}

// NewConversationStore creates a new conversation store.
func NewConversationStore() *ConversationStore {
	return &ConversationStore{
		conversations: make(map[string]*Conversation),
	}
}

// Add stores a conversation by ID, creating or updating as needed.
// Updates the UpdatedAt timestamp on the conversation.
func (s *ConversationStore) Add(conversation *Conversation) {
	s.mu.Lock()
	defer s.mu.Unlock()

	conversation.UpdatedAt = time.Now()
	s.conversations[conversation.ID] = conversation
}

// Get retrieves a conversation by ID.
// The returned Conversation is a copy to prevent external mutation of internal state.
func (s *ConversationStore) Get(id string) (*Conversation, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	conversation, ok := s.conversations[id]
	if !ok {
		return nil, false
	}

	// Return a shallow copy to protect internal state
	cpy := &Conversation{
		ID:           conversation.ID,
		Name:         conversation.Name,
		Messages:     make([]*Message, len(conversation.Messages)),
		Participants: make(map[string]Participant, len(conversation.Participants)),
		CreatedAt:    conversation.CreatedAt,
		UpdatedAt:    conversation.UpdatedAt,
		Metadata:     make(map[string]any, len(conversation.Metadata)),
	}
	copy(cpy.Messages, conversation.Messages)
	for k, v := range conversation.Participants {
		cpy.Participants[k] = v
	}
	for k, v := range conversation.Metadata {
		cpy.Metadata[k] = v
	}
	return cpy, true
}

// List returns all conversation IDs with message counts.
func (s *ConversationStore) List() map[string]int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]int)
	for id, conversation := range s.conversations {
		result[id] = len(conversation.Messages)
	}
	return result
}

// Count returns the number of conversations.
func (s *ConversationStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.conversations)
}

// Clear removes a conversation by ID.
func (s *ConversationStore) Clear(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.conversations[id]; ok {
		delete(s.conversations, id)
		return true
	}
	return false
}

// ClearAll removes all conversations.
func (s *ConversationStore) ClearAll() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	count := len(s.conversations)
	s.conversations = make(map[string]*Conversation)
	return count
}

// Save persists all conversations to a directory.
// Each conversation is saved as a separate JSON file named by ID.
// Data is marshaled under the lock, but I/O is performed outside the lock.
func (s *ConversationStore) Save(dir string) error {
	// Copy data under lock, then release before I/O
	s.mu.RLock()
	toSave := make(map[string][]byte)
	for id, conversation := range s.conversations {
		data, err := json.MarshalIndent(conversation, "", "  ")
		if err != nil {
			s.mu.RUnlock()
			return fmt.Errorf("marshal %s: %w", id, err)
		}
		toSave[id] = data
	}
	s.mu.RUnlock()

	// Perform I/O without holding the lock
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	for id, data := range toSave {
		path := filepath.Join(dir, id+".json")
		if err := os.WriteFile(path, data, 0644); err != nil {
			return fmt.Errorf("write %s: %w", id, err)
		}
	}

	return nil
}

// Load loads conversations from a directory.
// Each JSON file in the directory is read and unmarshaled into a conversation.
// I/O is performed outside the lock, with only the map update under lock.
func (s *ConversationStore) Load(dir string) error {
	// Perform I/O outside the lock
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No saved conversations
		}
		return err
	}

	// Read and unmarshal files outside the lock
	loaded := make(map[string]*Conversation)
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read %s: %w", entry.Name(), err)
		}

		var conversation Conversation
		if err := json.Unmarshal(data, &conversation); err != nil {
			return fmt.Errorf("unmarshal %s: %w", entry.Name(), err)
		}

		loaded[conversation.ID] = &conversation
	}

	// Acquire lock only for map update
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, conversation := range loaded {
		s.conversations[id] = conversation
	}

	return nil
}
