package backend

import (
	"context"
	"fmt"
	"sync"
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
		return fmt.Errorf("backend %q already registered", name)
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
func Default(loomURL string) *Registry {
	registry := NewRegistry()
	// Errors impossible here since registry is freshly created (no duplicates)
	_ = registry.Register("claudecode", NewClaudeCode(ClaudeCodeConfig{}))
	_ = registry.Register("loom", NewLoom(LoomConfig{URL: loomURL}))
	return registry
}
