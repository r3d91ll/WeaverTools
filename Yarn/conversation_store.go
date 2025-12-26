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
