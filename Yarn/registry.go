// Package yarn provides session management for research workflows.
package yarn

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// ErrEmptySessionName is returned when a session name is empty or whitespace-only.
var ErrEmptySessionName = errors.New("session name cannot be empty")

// ErrRegistrySaveFailed is returned when the registry fails to save to disk.
// The underlying error provides additional context about the specific failure.
var ErrRegistrySaveFailed = errors.New("registry save failed")

// ErrRegistryLoadFailed is returned when the registry fails to load from disk.
// The underlying error provides additional context about the specific failure.
var ErrRegistryLoadFailed = errors.New("registry load failed")

// -----------------------------------------------------------------------------
// Error Types
// -----------------------------------------------------------------------------

// SessionNotFoundError is returned when a session lookup fails.
// It provides helpful context including the requested session name,
// available sessions, and suggestions for similar names.
// This follows the structured error pattern used by Yarn's ValidationError.
type SessionNotFoundError struct {
	// Name is the session name that was requested but not found.
	Name string
	// AvailableSessions lists all currently registered session names.
	AvailableSessions []string
	// Suggestions contains helpful hints like "Did you mean X?".
	Suggestions []string
}

// Error implements the error interface.
// It formats a helpful message that includes the session name, available sessions,
// and suggestions for similar names when applicable.
func (e *SessionNotFoundError) Error() string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("session %q not found", e.Name))

	// Add available sessions context
	if len(e.AvailableSessions) > 0 {
		sb.WriteString(fmt.Sprintf("; available: [%s]", strings.Join(e.AvailableSessions, ", ")))
	} else {
		sb.WriteString("; no sessions registered")
	}

	// Add suggestions if any
	for _, suggestion := range e.Suggestions {
		sb.WriteString("; ")
		sb.WriteString(suggestion)
	}

	return sb.String()
}

// SessionAlreadyRegisteredError is returned when attempting to register a session
// with a name that is already in use.
// It provides helpful context including the conflicting name and registered sessions
// to help users resolve the conflict.
type SessionAlreadyRegisteredError struct {
	// Name is the session name that was already registered.
	Name string
	// RegisteredSessions lists all currently registered session names.
	RegisteredSessions []string
}

// Error implements the error interface.
// It formats a helpful message that includes the conflicting session name,
// currently registered sessions, and actionable suggestions for resolution.
func (e *SessionAlreadyRegisteredError) Error() string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("session %q is already registered", e.Name))

	// Add registered sessions context
	if len(e.RegisteredSessions) > 0 {
		sb.WriteString(fmt.Sprintf("; registered: [%s]", strings.Join(e.RegisteredSessions, ", ")))
	}

	// Add actionable suggestions
	sb.WriteString(fmt.Sprintf("; choose a different name (not %q)", e.Name))
	sb.WriteString("; or use Get() to retrieve the existing session")
	sb.WriteString("; or use GetOrCreate() to reuse existing sessions")

	return sb.String()
}

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
// Register returns a SessionAlreadyRegisteredError if a session with the given
// name is already registered. The error includes the list of registered sessions
// and actionable suggestions for resolving the conflict.
// Use GetOrCreate if you want to reuse existing sessions.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Register(name string, session *Session) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.sessions[name]; exists {
		return &SessionAlreadyRegisteredError{
			Name:               name,
			RegisteredSessions: r.listSessionNamesLocked(),
		}
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

// GetWithError retrieves a session by name, returning a structured error if not found.
// Use this when you want detailed error information for user-facing error messages.
// The returned error includes available session names and suggestions for similar names
// (case mismatches, substring matches) to help users identify typos.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) GetWithError(name string) (*Session, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	session, ok := r.sessions[name]
	if !ok {
		available := r.listSessionNamesLocked()
		return nil, &SessionNotFoundError{
			Name:              name,
			AvailableSessions: available,
			Suggestions:       suggestSimilarSessions(name, available),
		}
	}
	return session, nil
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
// Unregister returns a SessionNotFoundError if no session with the given name
// is registered. The error includes available session names and suggestions
// for similar names to help identify typos.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Unregister(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.sessions[name]; !exists {
		available := r.listSessionNamesLocked()
		return &SessionNotFoundError{
			Name:              name,
			AvailableSessions: available,
			Suggestions:       suggestSimilarSessions(name, available),
		}
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
		return nil, &SessionAlreadyRegisteredError{
			Name:               name,
			RegisteredSessions: r.listSessionNamesLocked(),
		}
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

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// listSessionNamesLocked returns a sorted list of registered session names.
// Must be called while holding at least a read lock on r.mu.
func (r *SessionRegistry) listSessionNamesLocked() []string {
	names := make([]string, 0, len(r.sessions))
	for name := range r.sessions {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// suggestSimilarSessions returns suggestions for similar session names.
// It checks for case-insensitive matches and substring/partial matches.
// This helps users identify typos or find sessions with similar names.
func suggestSimilarSessions(name string, available []string) []string {
	var suggestions []string
	nameLower := strings.ToLower(name)

	// Check for common variations
	for _, session := range available {
		sessionLower := strings.ToLower(session)

		// Check for case-insensitive match
		if nameLower == sessionLower && name != session {
			suggestions = append(suggestions, fmt.Sprintf("Did you mean %q? (case mismatch)", session))
			continue
		}

		// Check for common typos or variations
		// e.g., "experiment" -> "experiment-2024", "my-session" -> "my"
		// Skip exact matches (no suggestion needed)
		if nameLower == sessionLower {
			continue
		}
		if strings.Contains(sessionLower, nameLower) || strings.Contains(nameLower, sessionLower) {
			suggestions = append(suggestions, fmt.Sprintf("Did you mean %q?", session))
		}
	}

	return suggestions
}

// -----------------------------------------------------------------------------
// Persistence
// -----------------------------------------------------------------------------

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

	// Track used filenames to handle collisions
	usedFilenames := make(map[string]struct{})

	for name, session := range r.sessions {
		if session == nil {
			continue
		}

		// Serialize the session while holding the lock
		data, err := json.MarshalIndent(session, "", "  ")
		if err != nil {
			r.mu.RUnlock()
			return fmt.Errorf("%w: failed to marshal session %q: %v", ErrRegistrySaveFailed, name, err)
		}

		// Create sanitized filename with collision handling
		baseFilename := sanitizeFilename(name)
		filename := makeUniqueFilename(baseFilename, usedFilenames)
		toSave[filename] = data
		manifest.Sessions[name] = filename
	}
	r.mu.RUnlock()

	// Perform I/O without holding the lock
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("%w: failed to create directory %q: %v", ErrRegistrySaveFailed, dir, err)
	}

	// Save each session first
	for filename, data := range toSave {
		path := filepath.Join(dir, filename+".json")
		if err := os.WriteFile(path, data, 0644); err != nil {
			return fmt.Errorf("%w: failed to write session to %q: %v", ErrRegistrySaveFailed, path, err)
		}
	}

	// Save the manifest last for atomic behavior
	// If we crash after writing sessions but before manifest, Load will still work
	// using filename-based fallback. Writing manifest last ensures consistency.
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("%w: failed to marshal manifest: %v", ErrRegistrySaveFailed, err)
	}
	manifestPath := filepath.Join(dir, manifestFilename)
	if err := os.WriteFile(manifestPath, manifestData, 0644); err != nil {
		return fmt.Errorf("%w: failed to write manifest to %q: %v", ErrRegistrySaveFailed, manifestPath, err)
	}

	return nil
}

// Load restores sessions from a directory of JSON files.
// It reads the manifest.json for name mappings, or falls back to using
// filenames (without .json extension) as registry names.
//
// Load follows the copy-under-lock pattern: files are read and unmarshaled
// outside the lock, then the lock is acquired only for updating the map.
// This minimizes lock contention during potentially slow file operations.
//
// If the directory doesn't exist, Load returns nil (no error) to handle
// the case where no sessions have been saved yet.
//
// Loaded sessions are merged into the registry; existing sessions with
// the same names are overwritten.
//
// This method is safe for concurrent use.
func (r *SessionRegistry) Load(dir string) error {
	// Perform I/O outside the lock
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No saved sessions - this is not an error
		}
		return fmt.Errorf("%w: failed to read directory %q: %v", ErrRegistryLoadFailed, dir, err)
	}

	// Try to load the manifest for name mappings
	var manifest registryManifest
	manifestPath := filepath.Join(dir, manifestFilename)
	manifestData, err := os.ReadFile(manifestPath)
	if err == nil {
		// Manifest exists - parse it
		if err := json.Unmarshal(manifestData, &manifest); err != nil {
			return fmt.Errorf("%w: failed to parse manifest %q: %v", ErrRegistryLoadFailed, manifestPath, err)
		}
	} else if !os.IsNotExist(err) {
		// Manifest exists but couldn't be read - that's an error
		return fmt.Errorf("%w: failed to read manifest %q: %v", ErrRegistryLoadFailed, manifestPath, err)
	}
	// If manifest doesn't exist, manifest.Sessions will be nil (fallback to filenames)

	// Build reverse lookup: filename -> registry name
	filenameToName := make(map[string]string)
	for name, filename := range manifest.Sessions {
		filenameToName[filename] = name
	}

	// Read and unmarshal session files outside the lock
	loaded := make(map[string]*Session)
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if filepath.Ext(name) != ".json" || name == manifestFilename {
			continue
		}

		// Read the session file
		path := filepath.Join(dir, name)
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("%w: failed to read session file %q: %v", ErrRegistryLoadFailed, path, err)
		}

		// Unmarshal the session
		var session Session
		if err := json.Unmarshal(data, &session); err != nil {
			return fmt.Errorf("%w: failed to parse session file %q: %v", ErrRegistryLoadFailed, path, err)
		}

		// Determine the registry name
		baseFilename := strings.TrimSuffix(name, ".json")
		registryName := baseFilename
		if mappedName, ok := filenameToName[baseFilename]; ok {
			registryName = mappedName
		}

		loaded[registryName] = &session
	}

	// Acquire lock only for map update
	r.mu.Lock()
	defer r.mu.Unlock()
	for name, session := range loaded {
		r.sessions[name] = session
	}

	return nil
}

// unsafeFilenameChars matches characters that are unsafe for filenames.
// This includes: / \ : * ? " < > | % and control characters.
// The percent sign is included since we use percent-encoding.
var unsafeFilenameChars = regexp.MustCompile(`[/\\:*?"<>|%\x00-\x1f]`)

// sanitizeFilename converts a session name to a safe filename.
// Unsafe characters are percent-encoded (e.g., "/" becomes "%2F") to preserve
// uniqueness and avoid collisions between different names. The original name
// is also preserved in the manifest for accurate restoration during Load.
//
// This function handles:
//   - Percent-encoding of unsafe characters (/, \, :, *, ?, ", <, >, |, %)
//   - Removing leading/trailing whitespace and dots
//   - Empty names (returns "_unnamed_")
//   - Long names (truncated to 190 runes for UTF-8 safety)
func sanitizeFilename(name string) string {
	// Percent-encode unsafe characters to avoid collisions
	// e.g., "a/b" -> "a%2Fb", "a:b" -> "a%3Ab"
	safe := unsafeFilenameChars.ReplaceAllStringFunc(name, func(s string) string {
		return fmt.Sprintf("%%%02X", s[0])
	})

	// Trim leading/trailing whitespace and dots (problematic on some filesystems)
	safe = strings.Trim(safe, " .")

	// Handle empty result
	if safe == "" {
		safe = "_unnamed_"
	}

	// Truncate to reasonable length using runes for UTF-8 safety
	// (190 runes, leaving room for .json extension and suffix)
	runes := []rune(safe)
	if len(runes) > 190 {
		safe = string(runes[:190])
	}

	return safe
}

// makeUniqueFilename ensures the filename is unique within the given set.
// If the base filename already exists, it appends a numeric suffix (e.g., "_1", "_2").
// Returns the unique filename (without extension) and adds it to the used set.
func makeUniqueFilename(base string, used map[string]struct{}) string {
	filename := base
	counter := 1
	for {
		if _, exists := used[filename]; !exists {
			used[filename] = struct{}{}
			return filename
		}
		filename = fmt.Sprintf("%s_%d", base, counter)
		counter++
	}
}
