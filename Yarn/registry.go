// Package yarn provides session management for research workflows.
package yarn

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// ErrEmptySessionName is returned when a session name is empty or whitespace-only.
var ErrEmptySessionName = errors.New("session name cannot be empty")

// SessionRegistry manages multiple research sessions with thread-safe access.
// It provides a registry pattern for storing, retrieving, and managing sessions
// by name. All methods are safe for concurrent use by multiple goroutines.
//
// SessionRegistry mirrors the Backend Registry pattern from Weaver, enabling
// multiple concurrent research experiments to be tracked and managed.
type SessionRegistry struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

// NewSessionRegistry creates and returns a new, empty SessionRegistry.
// The returned registry is ready for use and safe for concurrent access.
func NewSessionRegistry() *SessionRegistry {
	return &SessionRegistry{
		sessions: make(map[string]*Session),
	}
}

// Register adds a session to the registry with the given name.
// The session can later be retrieved using Get with the same name.
//
// Register returns an error if a session with the given name is already
// registered. Use GetOrCreate if you want to reuse existing sessions.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Register(name string, session *Session) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.sessions[name]; exists {
		return fmt.Errorf("session %q already registered", name)
	}
	r.sessions[name] = session
	return nil
}

// Get retrieves a session by name from the registry.
// It returns the session and true if found, or nil and false if no session
// with the given name is registered.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Get(name string) (*Session, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	session, ok := r.sessions[name]
	return session, ok
}

// List returns the names of all registered sessions.
// The order of names in the returned slice is not guaranteed.
// Returns an empty slice if no sessions are registered.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]string, 0, len(r.sessions))
	for name := range r.sessions {
		result = append(result, name)
	}
	return result
}

// SessionStatus represents the current status of a session in the registry.
// It provides a snapshot of session metadata and statistics that is safe
// to use after the registry lock has been released.
type SessionStatus struct {
	// Name is the registered name of the session.
	Name string `json:"name"`
	// ID is the unique identifier of the session.
	ID string `json:"id"`
	// IsActive is true if the session has not been ended.
	IsActive bool `json:"is_active"`
	// StartedAt is when the session was created.
	StartedAt time.Time `json:"started_at"`
	// EndedAt is when the session was ended, or nil if still active.
	EndedAt *time.Time `json:"ended_at,omitempty"`
	// Stats contains session statistics like conversation and measurement counts.
	Stats SessionStats `json:"stats"`
}

// Status returns a map of session names to their current status for all
// registered sessions. The returned map is independent of the registry
// and can be safely used without holding any locks.
//
// This method copies sessions before computing stats to minimize lock
// contention, following the Backend Registry pattern.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Status() map[string]SessionStatus {
	// Copy sessions to avoid holding lock during stats computation
	r.mu.RLock()
	sessions := make(map[string]*Session, len(r.sessions))
	for name, s := range r.sessions {
		sessions[name] = s
	}
	r.mu.RUnlock()

	result := make(map[string]SessionStatus)
	for name, session := range sessions {
		if session == nil {
			continue
		}
		result[name] = SessionStatus{
			Name:      session.Name,
			ID:        session.ID,
			IsActive:  session.EndedAt == nil,
			StartedAt: session.StartedAt,
			EndedAt:   session.EndedAt,
			Stats:     session.Stats(),
		}
	}
	return result
}

// Active returns all sessions that have not been ended (EndedAt is nil).
// This is analogous to the Backend Registry's Available method.
// Returns an empty slice if no active sessions exist.
//
// This method copies sessions before filtering to minimize lock contention.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Active() []*Session {
	// Copy sessions to avoid holding lock during filtering
	r.mu.RLock()
	sessions := make([]*Session, 0, len(r.sessions))
	for _, s := range r.sessions {
		sessions = append(sessions, s)
	}
	r.mu.RUnlock()

	result := make([]*Session, 0)
	for _, session := range sessions {
		if session != nil && session.EndedAt == nil {
			result = append(result, session)
		}
	}
	return result
}

// Unregister removes a session from the registry by name.
// The session itself is not modified or ended; it is simply removed from
// the registry. After unregistration, the same name can be used to register
// a new session.
//
// Unregister returns an error if no session with the given name is registered.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Unregister(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.sessions[name]; !exists {
		return fmt.Errorf("session %q not registered", name)
	}
	delete(r.sessions, name)
	return nil
}

// Create creates a new session with the given name and description, and
// registers it in the registry atomically. This is a convenience method
// that combines NewSession and Register into a single operation.
//
// Create returns the new session and nil on success. It returns nil and
// an error if a session with the given name already exists.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Create(name, description string) (*Session, error) {
	// Validate input: name cannot be empty or whitespace-only
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, ErrEmptySessionName
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.sessions[name]; exists {
		return nil, fmt.Errorf("session %q already registered", name)
	}

	session := NewSession(name, description)
	r.sessions[name] = session
	return session, nil
}

// GetOrCreate returns an existing session with the given name, or creates
// and registers a new one if it doesn't exist. This is useful when you want
// to ensure a session exists without checking first.
//
// The description parameter is only used if a new session is created. If
// a session with the given name already exists, the description is ignored
// and the existing session's description is preserved.
//
// Returns the session and a bool indicating whether a new session was created
// (true) or an existing session was returned (false). Returns nil and false
// if the name is empty or whitespace-only.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) GetOrCreate(name, description string) (*Session, bool) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, false
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if session, exists := r.sessions[name]; exists {
		return session, false
	}

	session := NewSession(name, description)
	r.sessions[name] = session
	return session, true
}

// Count returns the total number of sessions currently registered.
// This includes both active and ended sessions.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.sessions)
}

// registryManifest stores the mapping between registry names and filenames.
// This is saved as manifest.json to preserve the original names when loading.
type registryManifest struct {
	// Version is the manifest format version for future compatibility.
	Version int `json:"version"`
	// Sessions maps registry names to their sanitized filenames (without .json extension).
	Sessions map[string]string `json:"sessions"`
}

// manifestFilename is the name of the manifest file in the save directory.
const manifestFilename = "manifest.json"

// Save persists all sessions to a directory as JSON files.
// Each session is saved as {sanitized-name}.json, with a manifest.json
// that maps registry names to filenames.
//
// Save follows the copy-under-lock pattern: session data is copied while
// holding the read lock, then the lock is released before performing I/O.
// This minimizes lock contention during potentially slow file operations.
//
// The directory is created if it doesn't exist (using os.MkdirAll).
// Returns an error if the directory cannot be created or if any session
// fails to save.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Save(dir string) error {
	// Copy data under lock, then release before I/O
	r.mu.RLock()
	toSave := make(map[string][]byte)
	manifest := registryManifest{
		Version:  1,
		Sessions: make(map[string]string),
	}

	for name, session := range r.sessions {
		if session == nil {
			continue
		}

		// Serialize the session while holding the lock
		data, err := json.MarshalIndent(session, "", "  ")
		if err != nil {
			r.mu.RUnlock()
			return fmt.Errorf("failed to marshal session %q: %w", name, err)
		}

		// Create sanitized filename and store mapping
		filename := sanitizeFilename(name)
		toSave[filename] = data
		manifest.Sessions[name] = filename
	}
	r.mu.RUnlock()

	// Perform I/O without holding the lock
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %q: %w", dir, err)
	}

	// Save the manifest first
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}
	manifestPath := filepath.Join(dir, manifestFilename)
	if err := os.WriteFile(manifestPath, manifestData, 0644); err != nil {
		return fmt.Errorf("failed to write manifest to %q: %w", manifestPath, err)
	}

	// Save each session
	for filename, data := range toSave {
		path := filepath.Join(dir, filename+".json")
		if err := os.WriteFile(path, data, 0644); err != nil {
			return fmt.Errorf("failed to write session to %q: %w", path, err)
		}
	}

	return nil
}

// unsafeFilenameChars matches characters that are unsafe for filenames.
// This includes: / \ : * ? " < > | and control characters.
var unsafeFilenameChars = regexp.MustCompile(`[/\\:*?"<>|\x00-\x1f]`)

// sanitizeFilename converts a session name to a safe filename.
// Unsafe characters are replaced with underscores, and the result is
// truncated to a reasonable length. The original name is preserved in
// the manifest for accurate restoration during Load.
func sanitizeFilename(name string) string {
	// Replace unsafe characters with underscores
	safe := unsafeFilenameChars.ReplaceAllString(name, "_")

	// Trim leading/trailing whitespace and dots (problematic on some filesystems)
	safe = strings.Trim(safe, " .")

	// Handle empty result
	if safe == "" {
		safe = "_unnamed_"
	}

	// Truncate to reasonable length (200 chars, leaving room for .json extension)
	if len(safe) > 200 {
		safe = safe[:200]
	}

	return safe
}
