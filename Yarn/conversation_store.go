package yarn

import (
	"sync"
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
