// Package concepts provides storage and management for concept hidden states.
// Used for Kakeya geometry analysis to validate geometric signatures.
package concepts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/r3d91ll/yarn"
)

// Sample represents a single extracted sample for a concept.
type Sample struct {
	ID          string           `json:"id"`
	Content     string           `json:"content"`      // The generated example text
	HiddenState *yarn.HiddenState `json:"hidden_state"` // The hidden state vector
	ExtractedAt time.Time        `json:"extracted_at"`
	Model       string           `json:"model,omitempty"`
}

// Concept holds all samples for a named concept.
type Concept struct {
	Name      string    `json:"name"`
	Samples   []Sample  `json:"samples"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Dimension returns the hidden state dimension for this concept.
// Returns 0 if no samples or no hidden states.
func (c *Concept) Dimension() int {
	if len(c.Samples) == 0 {
		return 0
	}
	for _, s := range c.Samples {
		if s.HiddenState != nil {
			return s.HiddenState.Dimension()
		}
	}
	return 0
}

// ValidateDimensions checks that all samples have consistent dimensions.
// Returns the dimension and any mismatched sample IDs.
func (c *Concept) ValidateDimensions() (dim int, mismatched []string) {
	if len(c.Samples) == 0 {
		return 0, nil
	}

	// Find the first valid dimension
	for _, s := range c.Samples {
		if s.HiddenState != nil && s.HiddenState.Dimension() > 0 {
			dim = s.HiddenState.Dimension()
			break
		}
	}

	if dim == 0 {
		return 0, nil
	}

	// Check all samples against the expected dimension
	for _, s := range c.Samples {
		if s.HiddenState == nil {
			continue
		}
		if s.HiddenState.Dimension() != dim {
			mismatched = append(mismatched, s.ID)
		}
	}

	return dim, mismatched
}

// Vectors returns all hidden state vectors as [][]float32.
// Skips samples without hidden states.
func (c *Concept) Vectors() [][]float32 {
	var vectors [][]float32
	for _, s := range c.Samples {
		if s.HiddenState != nil && len(s.HiddenState.Vector) > 0 {
			vectors = append(vectors, s.HiddenState.Vector)
		}
	}
	return vectors
}

// VectorsAsFloat64 returns vectors as [][]float64 for analysis APIs.
func (c *Concept) VectorsAsFloat64() [][]float64 {
	vectors := c.Vectors()
	result := make([][]float64, len(vectors))
	for i, v := range vectors {
		result[i] = make([]float64, len(v))
		for j, f := range v {
			result[i][j] = float64(f)
		}
	}
	return result
}

// Store manages concepts in memory with optional persistence.
type Store struct {
	mu       sync.RWMutex
	concepts map[string]*Concept
}

// NewStore creates a new concept store.
func NewStore() *Store {
	return &Store{
		concepts: make(map[string]*Concept),
	}
}

// Add adds a sample to a concept, creating the concept if it doesn't exist.
func (s *Store) Add(conceptName string, sample Sample) {
	s.mu.Lock()
	defer s.mu.Unlock()

	concept, ok := s.concepts[conceptName]
	if !ok {
		concept = &Concept{
			Name:      conceptName,
			Samples:   []Sample{},
			CreatedAt: time.Now(),
		}
		s.concepts[conceptName] = concept
	}

	concept.Samples = append(concept.Samples, sample)
	concept.UpdatedAt = time.Now()
}

// Get retrieves a concept by name.
// The returned Concept is a copy to prevent external mutation of internal state.
func (s *Store) Get(name string) (*Concept, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	concept, ok := s.concepts[name]
	if !ok {
		return nil, false
	}

	// Return a shallow copy to protect internal state
	cpy := &Concept{
		Name:      concept.Name,
		Samples:   make([]Sample, len(concept.Samples)),
		CreatedAt: concept.CreatedAt,
		UpdatedAt: concept.UpdatedAt,
	}
	copy(cpy.Samples, concept.Samples)
	return cpy, true
}

// List returns all concept names with sample counts.
func (s *Store) List() map[string]int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]int)
	for name, concept := range s.concepts {
		result[name] = len(concept.Samples)
	}
	return result
}

// Clear removes a concept by name.
func (s *Store) Clear(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.concepts[name]; ok {
		delete(s.concepts, name)
		return true
	}
	return false
}

// ClearAll removes all concepts.
func (s *Store) ClearAll() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	count := len(s.concepts)
	s.concepts = make(map[string]*Concept)
	return count
}

// Count returns the number of concepts.
func (s *Store) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.concepts)
}

// Save persists all concepts to a directory.
func (s *Store) Save(dir string) error {
	// Copy data under lock, then release before I/O
	s.mu.RLock()
	toSave := make(map[string][]byte)
	for name, concept := range s.concepts {
		data, err := json.MarshalIndent(concept, "", "  ")
		if err != nil {
			s.mu.RUnlock()
			return fmt.Errorf("marshal %s: %w", name, err)
		}
		toSave[name] = data
	}
	s.mu.RUnlock()

	// Perform I/O without holding the lock
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	for name, data := range toSave {
		path := filepath.Join(dir, name+".json")
		if err := os.WriteFile(path, data, 0644); err != nil {
			return fmt.Errorf("write %s: %w", name, err)
		}
	}

	return nil
}

// Load loads concepts from a directory.
func (s *Store) Load(dir string) error {
	// Perform I/O outside the lock
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No saved concepts
		}
		return err
	}

	// Read and unmarshal files outside the lock
	loaded := make(map[string]*Concept)
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read %s: %w", entry.Name(), err)
		}

		var concept Concept
		if err := json.Unmarshal(data, &concept); err != nil {
			return fmt.Errorf("unmarshal %s: %w", entry.Name(), err)
		}

		loaded[concept.Name] = &concept
	}

	// Acquire lock only for map update
	s.mu.Lock()
	defer s.mu.Unlock()
	for name, concept := range loaded {
		s.concepts[name] = concept
	}

	return nil
}
