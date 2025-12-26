package yarn

import (
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
