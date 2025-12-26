package yarn

import (
	"fmt"
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

// Register adds a session to the registry.
func (r *SessionRegistry) Register(name string, session *Session) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.sessions[name]; exists {
		return fmt.Errorf("session %q already registered", name)
	}
	r.sessions[name] = session
	return nil
}

// Get retrieves a session by name.
func (r *SessionRegistry) Get(name string) (*Session, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	session, ok := r.sessions[name]
	return session, ok
}
