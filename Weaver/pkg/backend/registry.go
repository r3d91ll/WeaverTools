package backend

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
)

// Registry manages available backends.
type Registry struct {
	backends map[string]Backend
	mu       sync.RWMutex
}

// NewRegistry creates a new backend registry.
func NewRegistry() *Registry {
	return &Registry{
		backends: make(map[string]Backend),
	}
}

// Register adds a backend to the registry.
func (r *Registry) Register(name string, backend Backend) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.backends[name]; exists {
		return createBackendAlreadyRegisteredError(name, r.listBackendNamesLocked())
	}
	r.backends[name] = backend
	return nil
}

// Get retrieves a backend by name.
func (r *Registry) Get(name string) (Backend, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	backend, ok := r.backends[name]
	return backend, ok
}

// GetWithError retrieves a backend by name, returning a structured error if not found.
// Use this when you want detailed error information for user-facing error messages.
func (r *Registry) GetWithError(name string) (Backend, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	backend, ok := r.backends[name]
	if !ok {
		return nil, createBackendNotFoundError(name, r.listBackendNamesLocked())
	}
	return backend, nil
}

// List returns all registered backend names.
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]string, 0, len(r.backends))
	for name := range r.backends {
		result = append(result, name)
	}
	return result
}

// Available returns all backends that are currently available.
func (r *Registry) Available(ctx context.Context) []Backend {
	// Copy backends slice to avoid holding lock during I/O
	r.mu.RLock()
	backends := make([]Backend, 0, len(r.backends))
	for _, b := range r.backends {
		backends = append(backends, b)
	}
	r.mu.RUnlock()

	var result []Backend
	for _, backend := range backends {
		if backend.IsAvailable(ctx) {
			result = append(result, backend)
		}
	}
	return result
}

// Status returns availability status for all backends.
func (r *Registry) Status(ctx context.Context) map[string]Status {
	// Copy backends to avoid holding lock during I/O (IsAvailable may do network calls)
	r.mu.RLock()
	backends := make(map[string]Backend, len(r.backends))
	for name, b := range r.backends {
		backends[name] = b
	}
	r.mu.RUnlock()

	result := make(map[string]Status)
	for name, backend := range backends {
		result[name] = Status{
			Name:         name,
			Type:         backend.Type(),
			Available:    backend.IsAvailable(ctx),
			Capabilities: backend.Capabilities(),
		}
	}
	return result
}

// Status represents backend status.
type Status struct {
	Name         string       `json:"name"`
	Type         Type         `json:"type"`
	Available    bool         `json:"available"`
	Capabilities Capabilities `json:"capabilities"`
}

// Default creates a registry with default backends.
// Panics if registration fails (indicates a programming error).
func Default(loomURL string) *Registry {
	registry := NewRegistry()

	if err := registry.Register("claudecode", NewClaudeCode(ClaudeCodeConfig{})); err != nil {
		panic(fmt.Sprintf("failed to register backend 'claudecode': %v", err))
	}
	if err := registry.Register("loom", NewLoom(LoomConfig{URL: loomURL})); err != nil {
		panic(fmt.Sprintf("failed to register backend 'loom': %v", err))
	}

	return registry
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// listBackendNamesLocked returns a sorted list of registered backend names.
// Must be called while holding at least a read lock on r.mu.
func (r *Registry) listBackendNamesLocked() []string {
	names := make([]string, 0, len(r.backends))
	for name := range r.backends {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// -----------------------------------------------------------------------------
// Error Creation Helpers
// -----------------------------------------------------------------------------

// createBackendAlreadyRegisteredError creates a structured error for duplicate backend registration.
func createBackendAlreadyRegisteredError(name string, existingBackends []string) *werrors.WeaverError {
	err := werrors.Backend(werrors.ErrBackendAlreadyRegistered,
		fmt.Sprintf("backend %q is already registered", name))

	err.WithContext("backend", name)
	err.WithContext("registered_backends", strings.Join(existingBackends, ", "))

	// Add suggestions for resolving the conflict
	err.WithSuggestion(fmt.Sprintf("Choose a different name for the new backend (not %q)", name))
	err.WithSuggestion("Use Registry.Get() to retrieve the existing backend instead of registering a new one")
	err.WithSuggestion("If replacing the backend, unregister the existing one first")

	return err
}

// createBackendNotFoundError creates a structured error for backend lookup failures.
func createBackendNotFoundError(name string, availableBackends []string) *werrors.WeaverError {
	err := werrors.Backend(werrors.ErrBackendNotFound,
		fmt.Sprintf("backend %q is not registered", name))

	err.WithContext("backend", name)

	if len(availableBackends) > 0 {
		err.WithContext("available_backends", strings.Join(availableBackends, ", "))
		err.WithSuggestion(fmt.Sprintf("Available backends: %s", strings.Join(availableBackends, ", ")))
	} else {
		err.WithContext("available_backends", "(none)")
		err.WithSuggestion("No backends are currently registered")
	}

	// Check for common typos/alternatives
	suggestions := suggestSimilarBackends(name, availableBackends)
	for _, s := range suggestions {
		err.WithSuggestion(s)
	}

	err.WithSuggestion("Use Registry.Register() to add a backend before using it")

	return err
}

// suggestSimilarBackends returns suggestions for similar backend names.
func suggestSimilarBackends(name string, available []string) []string {
	var suggestions []string
	nameLower := strings.ToLower(name)

	// Check for common variations
	for _, backend := range available {
		backendLower := strings.ToLower(backend)

		// Check for case-insensitive match
		if nameLower == backendLower && name != backend {
			suggestions = append(suggestions, fmt.Sprintf("Did you mean %q? (case mismatch)", backend))
			continue
		}

		// Check for common typos or variations
		// e.g., "claude" -> "claudecode", "loom-server" -> "loom"
		if strings.Contains(backendLower, nameLower) || strings.Contains(nameLower, backendLower) {
			suggestions = append(suggestions, fmt.Sprintf("Did you mean %q?", backend))
		}
	}

	return suggestions
}
