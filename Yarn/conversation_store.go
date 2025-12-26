package yarn

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ConversationStore provides thread-safe storage for conversations with JSON file persistence.
// It mirrors the Concept Store pattern from Weaver/pkg/concepts/store.go.
//
// All methods are safe for concurrent use. Read operations use a read lock (RLock),
// while write operations use an exclusive lock (Lock).
//
// The Save and Load methods implement a lock-minimizing pattern: data is copied or prepared
// outside the lock, and I/O operations are performed without holding the lock.
type ConversationStore struct {
	mu            sync.RWMutex
	conversations map[string]*Conversation
}

// NewConversationStore creates and returns a new, empty conversation store.
func NewConversationStore() *ConversationStore {
	return &ConversationStore{
		conversations: make(map[string]*Conversation),
	}
}

// Add stores a conversation by ID, creating or updating as needed.
// It updates the conversation's UpdatedAt timestamp to the current time.
// This method is safe for concurrent use.
func (s *ConversationStore) Add(conversation *Conversation) {
	s.mu.Lock()
	defer s.mu.Unlock()

	conversation.UpdatedAt = time.Now()
	s.conversations[conversation.ID] = conversation
}

// Get retrieves a conversation by ID.
// Returns the conversation and true if found, or nil and false if not found.
// The returned Conversation is a shallow copy to prevent external mutation of internal state.
// This method is safe for concurrent use.
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

// List returns a map of all conversation IDs to their message counts.
// This method is safe for concurrent use.
func (s *ConversationStore) List() map[string]int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]int)
	for id, conversation := range s.conversations {
		result[id] = len(conversation.Messages)
	}
	return result
}

// Count returns the number of conversations in the store.
// This method is safe for concurrent use.
func (s *ConversationStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.conversations)
}

// Clear removes a conversation by ID.
// Returns true if the conversation was found and removed, false otherwise.
// This method is safe for concurrent use.
func (s *ConversationStore) Clear(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.conversations[id]; ok {
		delete(s.conversations, id)
		return true
	}
	return false
}

// ClearAll removes all conversations from the store.
// Returns the number of conversations that were removed.
// This method is safe for concurrent use.
func (s *ConversationStore) ClearAll() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	count := len(s.conversations)
	s.conversations = make(map[string]*Conversation)
	return count
}

// Save persists all conversations to a directory as JSON files.
// Each conversation is saved as a separate file named {id}.json.
// Creates the directory if it doesn't exist.
//
// This method is safe for concurrent use. Data is marshaled under the read lock,
// but file I/O is performed outside the lock to minimize lock contention.
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

// Load reads conversations from a directory of JSON files.
// Each .json file in the directory is read and unmarshaled into a conversation.
// Non-JSON files and subdirectories are skipped.
// Returns nil if the directory doesn't exist (graceful handling of no saved data).
//
// This method is safe for concurrent use. File I/O is performed outside the lock,
// with only the final map update acquiring the write lock.
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
