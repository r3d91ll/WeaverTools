package yarn

import (
	"sync"
)

// SessionRegistry manages multiple research sessions with thread-safe access.
type SessionRegistry struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

// NewSessionRegistry creates a new session registry.
func NewSessionRegistry() *SessionRegistry {
	return &SessionRegistry{
		sessions: make(map[string]*Session),
	}
}
