// Package concepts provides storage and management for concept hidden states.
// Used for Kakeya geometry analysis to validate geometric signatures.
package concepts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
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

// ConceptStats holds detailed statistics for a single concept.
type ConceptStats struct {
	// Name is the concept name.
	Name string `json:"name"`

	// SampleCount is the number of samples for this concept.
	SampleCount int `json:"sample_count"`

	// Dimension is the hidden state dimension (0 if no samples with hidden states).
	Dimension int `json:"dimension"`

	// MismatchedIDs contains IDs of samples with dimensions different from the expected.
	// Empty if all dimensions are consistent.
	MismatchedIDs []string `json:"mismatched_ids,omitempty"`

	// CreatedAt is when the concept was first created.
	CreatedAt time.Time `json:"created_at"`

	// UpdatedAt is when the concept was last modified.
	UpdatedAt time.Time `json:"updated_at"`

	// Models lists unique model identifiers used to extract samples.
	Models []string `json:"models,omitempty"`

	// OldestSampleAt is the timestamp of the oldest sample extraction.
	// Zero time if no samples.
	OldestSampleAt time.Time `json:"oldest_sample_at,omitempty"`

	// NewestSampleAt is the timestamp of the newest sample extraction.
	// Zero time if no samples.
	NewestSampleAt time.Time `json:"newest_sample_at,omitempty"`
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
			return createMarshalError(name, len(concept.Samples), err)
		}
		toSave[name] = data
	}
	s.mu.RUnlock()

	// Perform I/O without holding the lock
	if err := os.MkdirAll(dir, 0755); err != nil {
		return createDirectoryCreateError(dir, err)
	}

	for name, data := range toSave {
		path := filepath.Join(dir, name+".json")
		if err := os.WriteFile(path, data, 0644); err != nil {
			return createWriteError(name, path, dir, err)
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
		return createDirectoryReadError(dir, err)
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
			return createReadError(entry.Name(), path, dir, err)
		}

		var concept Concept
		if err := json.Unmarshal(data, &concept); err != nil {
			return createUnmarshalError(entry.Name(), path, err)
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

// -----------------------------------------------------------------------------
// Error Helper Functions
// -----------------------------------------------------------------------------
// These functions create structured WeaverErrors for various store
// failure scenarios with appropriate context and suggestions.

// createMarshalError creates a structured error when JSON marshaling fails.
func createMarshalError(conceptName string, sampleCount int, cause error) *werrors.WeaverError {
	return werrors.IOWrap(cause, werrors.ErrIOMarshalFailed,
		"failed to serialize concept data to JSON").
		WithContext("concept", conceptName).
		WithContext("sample_count", formatSampleCount(sampleCount)).
		WithContext("operation", "save").
		WithSuggestion("This may indicate corrupted hidden state data").
		WithSuggestion("Try re-extracting samples for this concept with '/extract'").
		WithSuggestion("Check if the concept contains unusual or very large hidden states")
}

// createDirectoryCreateError creates a structured error when directory creation fails.
func createDirectoryCreateError(dir string, cause error) *werrors.WeaverError {
	errStr := strings.ToLower(cause.Error())

	// Detect specific error types
	if isStorePermissionError(errStr) {
		return werrors.IOWrap(cause, werrors.ErrIOPermissionDenied,
			"permission denied when creating concepts directory").
			WithContext("directory", dir).
			WithContext("operation", "mkdir").
			WithSuggestion("Check that you have write permissions for the parent directory").
			WithSuggestion("Try running: chmod 755 " + filepath.Dir(dir)).
			WithSuggestion("Verify the path is accessible by the current user")
	}

	if isStoreDiskFullError(errStr) {
		return werrors.IOWrap(cause, werrors.ErrIODiskFull,
			"disk full when creating concepts directory").
			WithContext("directory", dir).
			WithContext("operation", "mkdir").
			WithSuggestion("Free up disk space and try again").
			WithSuggestion("Check available space with 'df -h'").
			WithSuggestion("Consider clearing old concept data with '/concepts clear'")
	}

	if isStoreReadOnlyFilesystemError(errStr) {
		return werrors.IOWrap(cause, werrors.ErrIOWriteFailed,
			"cannot create directory on read-only filesystem").
			WithContext("directory", dir).
			WithContext("operation", "mkdir").
			WithSuggestion("The filesystem may be mounted read-only").
			WithSuggestion("Check if the disk is mounted correctly with 'mount'").
			WithSuggestion("Try a different storage location for concept data")
	}

	// Generic directory creation failure
	return werrors.IOWrap(cause, werrors.ErrIOWriteFailed,
		"failed to create concepts storage directory").
		WithContext("directory", dir).
		WithContext("operation", "mkdir").
		WithSuggestion("Check that the parent directory exists").
		WithSuggestion("Verify write permissions for the storage path").
		WithSuggestion("Check if the path is valid and accessible")
}

// createWriteError creates a structured error when file write fails.
func createWriteError(conceptName, path, dir string, cause error) *werrors.WeaverError {
	errStr := strings.ToLower(cause.Error())

	// Detect specific error types
	if isStorePermissionError(errStr) {
		return werrors.IOWrap(cause, werrors.ErrIOPermissionDenied,
			"permission denied when saving concept file").
			WithContext("concept", conceptName).
			WithContext("file", path).
			WithContext("directory", dir).
			WithContext("operation", "write").
			WithSuggestion("Check file permissions: ls -la " + path).
			WithSuggestion("Try: chmod 644 " + path).
			WithSuggestion("Verify directory permissions: ls -la " + dir)
	}

	if isStoreDiskFullError(errStr) {
		return werrors.IOWrap(cause, werrors.ErrIODiskFull,
			"disk full when saving concept file").
			WithContext("concept", conceptName).
			WithContext("file", path).
			WithContext("operation", "write").
			WithSuggestion("Free up disk space and try again").
			WithSuggestion("Check available space with 'df -h'").
			WithSuggestion("Consider clearing old concept data with '/concepts clear <name>'")
	}

	if isStoreReadOnlyFilesystemError(errStr) {
		return werrors.IOWrap(cause, werrors.ErrIOWriteFailed,
			"cannot write concept file on read-only filesystem").
			WithContext("concept", conceptName).
			WithContext("file", path).
			WithContext("operation", "write").
			WithSuggestion("The filesystem may be mounted read-only").
			WithSuggestion("Check if the disk is mounted correctly").
			WithSuggestion("Remount with write permissions if possible")
	}

	// Generic write failure
	return werrors.IOWrap(cause, werrors.ErrIOWriteFailed,
		"failed to save concept file").
		WithContext("concept", conceptName).
		WithContext("file", path).
		WithContext("directory", dir).
		WithContext("operation", "write").
		WithSuggestion("Check that the directory exists and is writable").
		WithSuggestion("Verify disk space is available").
		WithSuggestion("Check if another process has the file locked")
}

// createDirectoryReadError creates a structured error when directory read fails.
func createDirectoryReadError(dir string, cause error) *werrors.WeaverError {
	errStr := strings.ToLower(cause.Error())

	if isStorePermissionError(errStr) {
		return werrors.IOWrap(cause, werrors.ErrIOPermissionDenied,
			"permission denied when reading concepts directory").
			WithContext("directory", dir).
			WithContext("operation", "readdir").
			WithSuggestion("Check directory permissions: ls -la " + filepath.Dir(dir)).
			WithSuggestion("Try: chmod 755 " + dir).
			WithSuggestion("Verify the directory is accessible by the current user")
	}

	// Generic directory read failure
	return werrors.IOWrap(cause, werrors.ErrIOReadFailed,
		"failed to read concepts directory").
		WithContext("directory", dir).
		WithContext("operation", "readdir").
		WithSuggestion("Check that the directory exists and is readable").
		WithSuggestion("Verify the path is correct").
		WithSuggestion("Check if the filesystem is mounted correctly")
}

// createReadError creates a structured error when file read fails.
func createReadError(fileName, path, dir string, cause error) *werrors.WeaverError {
	errStr := strings.ToLower(cause.Error())

	if isStorePermissionError(errStr) {
		return werrors.IOWrap(cause, werrors.ErrIOPermissionDenied,
			"permission denied when reading concept file").
			WithContext("file", fileName).
			WithContext("path", path).
			WithContext("directory", dir).
			WithContext("operation", "read").
			WithSuggestion("Check file permissions: ls -la " + path).
			WithSuggestion("Try: chmod 644 " + path).
			WithSuggestion("Verify the file owner and permissions")
	}

	if os.IsNotExist(cause) {
		return werrors.IOWrap(cause, werrors.ErrIOFileNotFound,
			"concept file not found").
			WithContext("file", fileName).
			WithContext("path", path).
			WithContext("directory", dir).
			WithContext("operation", "read").
			WithSuggestion("The concept file may have been deleted").
			WithSuggestion("Re-extract samples with '/extract <concept> <count>'").
			WithSuggestion("Check if the file exists: ls -la " + path)
	}

	// Generic read failure
	return werrors.IOWrap(cause, werrors.ErrIOReadFailed,
		"failed to read concept file").
		WithContext("file", fileName).
		WithContext("path", path).
		WithContext("directory", dir).
		WithContext("operation", "read").
		WithSuggestion("Check that the file exists and is readable").
		WithSuggestion("Verify the file is not corrupted").
		WithSuggestion("Try re-extracting the concept if the file is damaged")
}

// createUnmarshalError creates a structured error when JSON unmarshaling fails.
func createUnmarshalError(fileName, path string, cause error) *werrors.WeaverError {
	errStr := strings.ToLower(cause.Error())

	// Try to extract line/offset info from JSON error
	var context string
	if strings.Contains(errStr, "offset") {
		context = extractStoreJSONErrorContext(errStr)
	}

	err := werrors.IOWrap(cause, werrors.ErrIOUnmarshalFailed,
		"failed to parse concept file as JSON").
		WithContext("file", fileName).
		WithContext("path", path).
		WithContext("operation", "unmarshal")

	if context != "" {
		err.WithContext("error_location", context)
	}

	return err.
		WithSuggestion("The concept file may be corrupted or in an invalid format").
		WithSuggestion("Try deleting and re-extracting: /concepts clear <name> && /extract <name> <count>").
		WithSuggestion("Check if the file was modified manually").
		WithSuggestion("Inspect the file for syntax errors: cat " + path)
}

// -----------------------------------------------------------------------------
// Error Detection Helpers
// -----------------------------------------------------------------------------

// isStorePermissionError checks if the error indicates a permission problem.
func isStorePermissionError(errStr string) bool {
	patterns := []string{
		"permission denied",
		"access denied",
		"operation not permitted",
		"eacces",
		"eperm",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isStoreDiskFullError checks if the error indicates disk is full.
func isStoreDiskFullError(errStr string) bool {
	patterns := []string{
		"no space left",
		"disk full",
		"quota exceeded",
		"enospc",
		"edquot",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isStoreReadOnlyFilesystemError checks if the error indicates read-only filesystem.
func isStoreReadOnlyFilesystemError(errStr string) bool {
	patterns := []string{
		"read-only file system",
		"read only file system",
		"erofs",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// extractStoreJSONErrorContext attempts to extract useful context from a JSON error.
func extractStoreJSONErrorContext(errStr string) string {
	// Look for patterns like "offset 123" or "at offset 123"
	if idx := strings.Index(errStr, "offset"); idx >= 0 {
		// Extract a substring around "offset"
		start := idx
		end := idx + 20
		if end > len(errStr) {
			end = len(errStr)
		}
		return strings.TrimSpace(errStr[start:end])
	}
	return ""
}

// formatSampleCount formats a sample count as a string.
func formatSampleCount(n int) string {
	// Use a simple approach for small numbers
	if n < 0 {
		return "invalid"
	}
	if n == 0 {
		return "0"
	}

	// Convert to string using fmt.Sprintf
	result := ""
	for n > 0 {
		digit := n % 10
		result = string(rune('0'+digit)) + result
		n /= 10
	}
	return result
}
