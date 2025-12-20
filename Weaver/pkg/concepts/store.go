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
func (s *Store) Get(name string) (*Concept, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	concept, ok := s.concepts[name]
	return concept, ok
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
	s.mu.RLock()
	defer s.mu.RUnlock()

	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	for name, concept := range s.concepts {
		path := filepath.Join(dir, name+".json")
		data, err := json.MarshalIndent(concept, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal %s: %w", name, err)
		}
		if err := os.WriteFile(path, data, 0644); err != nil {
			return fmt.Errorf("write %s: %w", name, err)
		}
	}

	return nil
}

// Load loads concepts from a directory.
func (s *Store) Load(dir string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No saved concepts
		}
		return err
	}

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

		s.concepts[concept.Name] = &concept
	}

	return nil
}
