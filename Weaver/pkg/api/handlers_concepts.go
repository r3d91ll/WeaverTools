// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/r3d91ll/weaver/pkg/concepts"
	"github.com/r3d91ll/yarn"
)

// ConceptStore is the interface for managing concepts.
// This interface allows for dependency injection and testing.
type ConceptStore interface {
	// List returns all concept names with sample counts.
	List() map[string]int
	// Get retrieves a concept by name.
	Get(name string) (*concepts.Concept, bool)
	// Add adds a sample to a concept, creating the concept if needed.
	Add(conceptName string, sample concepts.Sample) error
	// Clear removes a concept by name.
	Clear(name string) bool
	// Stats returns detailed statistics for all concepts.
	Stats() concepts.StoreStats
}

// ConceptsHandler handles concept-related API requests.
type ConceptsHandler struct {
	// store is the concept store (real store or mock)
	store ConceptStore

	// mu protects concurrent access
	mu sync.RWMutex
}

// NewConceptsHandler creates a new ConceptsHandler with the given concept store.
func NewConceptsHandler(store ConceptStore) *ConceptsHandler {
	return &ConceptsHandler{
		store: store,
	}
}

// RegisterRoutes registers the concept API routes on the router.
func (h *ConceptsHandler) RegisterRoutes(router *Router) {
	router.GET("/api/concepts", h.ListConcepts)
	router.GET("/api/concepts/stats", h.GetStats)
	router.GET("/api/concepts/:name", h.GetConcept)
	router.POST("/api/concepts/:name/samples", h.AddSample)
	router.DELETE("/api/concepts/:name", h.DeleteConcept)
}

// -----------------------------------------------------------------------------
// API Request Types
// -----------------------------------------------------------------------------

// AddSampleRequest is the expected JSON body for POST /api/concepts/:name/samples.
type AddSampleRequest struct {
	Content     string           `json:"content"`
	HiddenState *HiddenStateAPI  `json:"hiddenState,omitempty"`
	Model       string           `json:"model,omitempty"`
}

// HiddenStateAPI represents a hidden state for the API.
type HiddenStateAPI struct {
	Vector    []float32 `json:"vector"`
	Layer     int       `json:"layer"`
	TokenIdx  int       `json:"tokenIdx"`
	Dtype     string    `json:"dtype"`
}

// -----------------------------------------------------------------------------
// API Response Types
// -----------------------------------------------------------------------------

// ConceptListResponse is the JSON response for GET /api/concepts.
type ConceptListResponse struct {
	Concepts []ConceptSummary `json:"concepts"`
}

// ConceptSummary provides a summary of a concept for listing.
type ConceptSummary struct {
	Name        string    `json:"name"`
	SampleCount int       `json:"sampleCount"`
	CreatedAt   time.Time `json:"createdAt,omitempty"`
	UpdatedAt   time.Time `json:"updatedAt,omitempty"`
}

// ConceptDetailResponse is the JSON response for GET /api/concepts/:name.
type ConceptDetailResponse struct {
	Name        string      `json:"name"`
	SampleCount int         `json:"sampleCount"`
	Dimension   int         `json:"dimension"`
	Samples     []SampleAPI `json:"samples"`
	Models      []string    `json:"models,omitempty"`
	CreatedAt   time.Time   `json:"createdAt"`
	UpdatedAt   time.Time   `json:"updatedAt"`
}

// SampleAPI is the API representation of a concept sample.
type SampleAPI struct {
	ID          string    `json:"id"`
	Content     string    `json:"content"`
	Model       string    `json:"model,omitempty"`
	HasVector   bool      `json:"hasVector"`
	Dimension   int       `json:"dimension,omitempty"`
	ExtractedAt time.Time `json:"extractedAt"`
}

// StatsResponse is the JSON response for GET /api/concepts/stats.
type StatsResponse struct {
	ConceptCount       int                       `json:"conceptCount"`
	TotalSamples       int                       `json:"totalSamples"`
	HealthyConcepts    int                       `json:"healthyConcepts"`
	ConceptsWithIssues int                       `json:"conceptsWithIssues"`
	Dimensions         map[int]int               `json:"dimensions"`
	Models             map[string]int            `json:"models"`
	Concepts           map[string]ConceptStatAPI `json:"concepts,omitempty"`
	OldestExtraction   *time.Time                `json:"oldestExtraction,omitempty"`
	NewestExtraction   *time.Time                `json:"newestExtraction,omitempty"`
}

// ConceptStatAPI is the API representation of concept statistics.
type ConceptStatAPI struct {
	Name           string     `json:"name"`
	SampleCount    int        `json:"sampleCount"`
	Dimension      int        `json:"dimension"`
	MismatchedIDs  []string   `json:"mismatchedIds,omitempty"`
	Models         []string   `json:"models,omitempty"`
	CreatedAt      time.Time  `json:"createdAt"`
	UpdatedAt      time.Time  `json:"updatedAt"`
	OldestSampleAt *time.Time `json:"oldestSampleAt,omitempty"`
	NewestSampleAt *time.Time `json:"newestSampleAt,omitempty"`
}

// AddSampleResponse is the JSON response for POST /api/concepts/:name/samples.
type AddSampleResponse struct {
	ID          string    `json:"id"`
	ConceptName string    `json:"conceptName"`
	Content     string    `json:"content"`
	Model       string    `json:"model,omitempty"`
	ExtractedAt time.Time `json:"extractedAt"`
}

// DeleteConceptResponse is the JSON response for DELETE /api/concepts/:name.
type DeleteConceptResponse struct {
	Message string `json:"message"`
	Name    string `json:"name"`
}

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------

// ListConcepts handles GET /api/concepts.
// It returns a list of all concepts with their sample counts.
func (h *ConceptsHandler) ListConcepts(w http.ResponseWriter, r *http.Request) {
	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Concept store is not available")
		return
	}

	conceptsMap := store.List()

	// Build response with sorted concept names for consistent ordering
	conceptList := make([]ConceptSummary, 0, len(conceptsMap))
	for name, count := range conceptsMap {
		summary := ConceptSummary{
			Name:        name,
			SampleCount: count,
		}
		// Try to get additional details if available
		if concept, ok := store.Get(name); ok {
			summary.CreatedAt = concept.CreatedAt
			summary.UpdatedAt = concept.UpdatedAt
		}
		conceptList = append(conceptList, summary)
	}

	// Sort by name for consistent ordering
	sort.Slice(conceptList, func(i, j int) bool {
		return conceptList[i].Name < conceptList[j].Name
	})

	response := ConceptListResponse{
		Concepts: conceptList,
	}

	WriteJSON(w, http.StatusOK, response)
}

// GetConcept handles GET /api/concepts/:name.
// It returns detailed information about a specific concept.
func (h *ConceptsHandler) GetConcept(w http.ResponseWriter, r *http.Request) {
	conceptName := PathParam(r, "name")
	if conceptName == "" {
		WriteError(w, http.StatusBadRequest, "missing_concept_name",
			"Concept name is required in the URL path")
		return
	}

	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Concept store is not available")
		return
	}

	concept, ok := store.Get(conceptName)
	if !ok {
		WriteError(w, http.StatusNotFound, "concept_not_found",
			"Concept '"+conceptName+"' not found")
		return
	}

	// Convert samples to API format
	samples := make([]SampleAPI, 0, len(concept.Samples))
	for _, s := range concept.Samples {
		sample := SampleAPI{
			ID:          s.ID,
			Content:     s.Content,
			Model:       s.Model,
			ExtractedAt: s.ExtractedAt,
			HasVector:   s.HiddenState != nil && len(s.HiddenState.Vector) > 0,
		}
		if s.HiddenState != nil {
			sample.Dimension = s.HiddenState.Dimension()
		}
		samples = append(samples, sample)
	}

	response := ConceptDetailResponse{
		Name:        concept.Name,
		SampleCount: len(concept.Samples),
		Dimension:   concept.Dimension(),
		Samples:     samples,
		Models:      concept.Models(),
		CreatedAt:   concept.CreatedAt,
		UpdatedAt:   concept.UpdatedAt,
	}

	WriteJSON(w, http.StatusOK, response)
}

// GetStats handles GET /api/concepts/stats.
// It returns aggregate statistics for all concepts.
func (h *ConceptsHandler) GetStats(w http.ResponseWriter, r *http.Request) {
	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Concept store is not available")
		return
	}

	stats := store.Stats()

	// Convert to API response format with camelCase
	response := StatsResponse{
		ConceptCount:       stats.ConceptCount,
		TotalSamples:       stats.TotalSamples,
		HealthyConcepts:    stats.HealthyConcepts,
		ConceptsWithIssues: stats.ConceptsWithIssues,
		Dimensions:         stats.Dimensions,
		Models:             stats.Models,
	}

	// Set time pointers only if not zero
	if !stats.OldestExtraction.IsZero() {
		response.OldestExtraction = &stats.OldestExtraction
	}
	if !stats.NewestExtraction.IsZero() {
		response.NewestExtraction = &stats.NewestExtraction
	}

	// Convert concept stats to API format
	if len(stats.Concepts) > 0 {
		response.Concepts = make(map[string]ConceptStatAPI, len(stats.Concepts))
		for name, cs := range stats.Concepts {
			stat := ConceptStatAPI{
				Name:          cs.Name,
				SampleCount:   cs.SampleCount,
				Dimension:     cs.Dimension,
				MismatchedIDs: cs.MismatchedIDs,
				Models:        cs.Models,
				CreatedAt:     cs.CreatedAt,
				UpdatedAt:     cs.UpdatedAt,
			}
			// Set time pointers only if not zero
			if !cs.OldestSampleAt.IsZero() {
				stat.OldestSampleAt = &cs.OldestSampleAt
			}
			if !cs.NewestSampleAt.IsZero() {
				stat.NewestSampleAt = &cs.NewestSampleAt
			}
			response.Concepts[name] = stat
		}
	}

	WriteJSON(w, http.StatusOK, response)
}

// AddSample handles POST /api/concepts/:name/samples.
// It adds a new sample to a concept, creating the concept if needed.
func (h *ConceptsHandler) AddSample(w http.ResponseWriter, r *http.Request) {
	conceptName := PathParam(r, "name")
	if conceptName == "" {
		WriteError(w, http.StatusBadRequest, "missing_concept_name",
			"Concept name is required in the URL path")
		return
	}

	// Parse request body
	var req AddSampleRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	// Validate content
	if req.Content == "" {
		WriteError(w, http.StatusBadRequest, "missing_content",
			"Sample content is required")
		return
	}

	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Concept store is not available")
		return
	}

	// Create sample
	now := time.Now().UTC()
	sample := concepts.Sample{
		ID:          uuid.New().String(),
		Content:     req.Content,
		ExtractedAt: now,
		Model:       req.Model,
	}

	// Convert hidden state if provided
	if req.HiddenState != nil && len(req.HiddenState.Vector) > 0 {
		sample.HiddenState = &yarn.HiddenState{
			Vector: req.HiddenState.Vector,
			Layer:  req.HiddenState.Layer,
			Dtype:  req.HiddenState.Dtype,
		}
	}

	// Add sample to store
	if err := store.Add(conceptName, sample); err != nil {
		WriteError(w, http.StatusBadRequest, "add_sample_failed",
			"Failed to add sample: "+err.Error())
		return
	}

	response := AddSampleResponse{
		ID:          sample.ID,
		ConceptName: conceptName,
		Content:     sample.Content,
		Model:       sample.Model,
		ExtractedAt: sample.ExtractedAt,
	}

	WriteJSON(w, http.StatusCreated, response)
}

// DeleteConcept handles DELETE /api/concepts/:name.
// It removes a concept and all its samples.
func (h *ConceptsHandler) DeleteConcept(w http.ResponseWriter, r *http.Request) {
	conceptName := PathParam(r, "name")
	if conceptName == "" {
		WriteError(w, http.StatusBadRequest, "missing_concept_name",
			"Concept name is required in the URL path")
		return
	}

	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Concept store is not available")
		return
	}

	// Check if concept exists before deleting
	if _, ok := store.Get(conceptName); !ok {
		WriteError(w, http.StatusNotFound, "concept_not_found",
			"Concept '"+conceptName+"' not found")
		return
	}

	if !store.Clear(conceptName) {
		WriteError(w, http.StatusInternalServerError, "delete_failed",
			"Failed to delete concept")
		return
	}

	response := DeleteConceptResponse{
		Message: "Concept deleted successfully",
		Name:    conceptName,
	}

	WriteJSON(w, http.StatusOK, response)
}

// -----------------------------------------------------------------------------
// Mock Store for Testing
// -----------------------------------------------------------------------------

// MockConceptStore is a mock implementation of ConceptStore for testing.
type MockConceptStore struct {
	concepts map[string]*concepts.Concept
	mu       sync.RWMutex
}

// Ensure MockConceptStore implements ConceptStore.
var _ ConceptStore = (*MockConceptStore)(nil)

// NewMockConceptStore creates a new mock concept store.
func NewMockConceptStore() *MockConceptStore {
	return &MockConceptStore{
		concepts: make(map[string]*concepts.Concept),
	}
}

// List returns all concept names with sample counts.
func (m *MockConceptStore) List() map[string]int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]int)
	for name, concept := range m.concepts {
		result[name] = len(concept.Samples)
	}
	return result
}

// Get retrieves a concept by name.
func (m *MockConceptStore) Get(name string) (*concepts.Concept, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	concept, ok := m.concepts[name]
	if !ok {
		return nil, false
	}

	// Return a copy to protect internal state
	cpy := &concepts.Concept{
		Name:      concept.Name,
		Samples:   make([]concepts.Sample, len(concept.Samples)),
		CreatedAt: concept.CreatedAt,
		UpdatedAt: concept.UpdatedAt,
	}
	copy(cpy.Samples, concept.Samples)
	return cpy, true
}

// Add adds a sample to a concept, creating the concept if needed.
func (m *MockConceptStore) Add(conceptName string, sample concepts.Sample) error {
	if conceptName == "" {
		return &ConceptError{Code: "empty_name", Message: "concept name cannot be empty"}
	}
	if sample.ID == "" {
		return &ConceptError{Code: "empty_sample_id", Message: "sample ID cannot be empty"}
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	concept, ok := m.concepts[conceptName]
	if !ok {
		concept = &concepts.Concept{
			Name:      conceptName,
			Samples:   []concepts.Sample{},
			CreatedAt: time.Now().UTC(),
		}
		m.concepts[conceptName] = concept
	}

	concept.Samples = append(concept.Samples, sample)
	concept.UpdatedAt = time.Now().UTC()

	return nil
}

// Clear removes a concept by name.
func (m *MockConceptStore) Clear(name string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.concepts[name]; ok {
		delete(m.concepts, name)
		return true
	}
	return false
}

// Stats returns detailed statistics for all concepts.
func (m *MockConceptStore) Stats() concepts.StoreStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := concepts.StoreStats{
		ConceptCount: len(m.concepts),
		Dimensions:   make(map[int]int),
		Models:       make(map[string]int),
		Concepts:     make(map[string]concepts.ConceptStats),
	}

	for name, concept := range m.concepts {
		stats.TotalSamples += len(concept.Samples)
		cs := concept.Stats()
		stats.Concepts[name] = cs

		if cs.Dimension > 0 {
			stats.Dimensions[cs.Dimension]++
		}

		if len(cs.MismatchedIDs) > 0 {
			stats.ConceptsWithIssues++
		} else {
			stats.HealthyConcepts++
		}

		for _, sample := range concept.Samples {
			if sample.Model != "" {
				stats.Models[sample.Model]++
			}
		}
	}

	return stats
}

// AddMockConcept adds a mock concept for testing.
func (m *MockConceptStore) AddMockConcept(concept *concepts.Concept) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.concepts[concept.Name] = concept
}

// ConceptError represents an error from concept operations.
type ConceptError struct {
	Code    string
	Message string
}

func (e *ConceptError) Error() string {
	return e.Message
}
