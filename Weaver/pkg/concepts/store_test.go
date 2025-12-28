// Package concepts tests for concept store operations.
package concepts

import (
	"testing"
	"time"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
	"github.com/r3d91ll/yarn"
)

// -----------------------------------------------------------------------------
// Test Helpers
// -----------------------------------------------------------------------------

// makeValidSample creates a valid sample with the given ID and hidden state dimension.
func makeValidSample(id string, dim int) Sample {
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = float32(i) * 0.1
	}
	return Sample{
		ID:      id,
		Content: "test content for " + id,
		HiddenState: &yarn.HiddenState{
			Vector: vector,
			Layer:  0,
			DType:  "float32",
		},
		ExtractedAt: time.Now(),
		Model:       "test-model",
	}
}

// makeInvalidHiddenStateSample creates a sample with an invalid hidden state.
func makeInvalidHiddenStateSample(id string) Sample {
	return Sample{
		ID:      id,
		Content: "test content",
		HiddenState: &yarn.HiddenState{
			Vector: []float32{}, // Empty vector is invalid
			Layer:  0,
			DType:  "float32",
		},
		ExtractedAt: time.Now(),
	}
}

// -----------------------------------------------------------------------------
// Add Method Validation Tests
// -----------------------------------------------------------------------------

func TestAdd_EmptyConceptName(t *testing.T) {
	store := NewStore()
	sample := makeValidSample("sample-1", 128)

	err := store.Add("", sample)

	if err == nil {
		t.Fatal("expected error for empty concept name, got nil")
	}

	we, ok := werrors.AsWeaverError(err)
	if !ok {
		t.Fatalf("expected WeaverError, got %T", err)
	}

	if we.Code != werrors.ErrConceptsEmptyName {
		t.Errorf("expected code %q, got %q", werrors.ErrConceptsEmptyName, we.Code)
	}

	if we.Category != werrors.CategoryValidation {
		t.Errorf("expected category %v, got %v", werrors.CategoryValidation, we.Category)
	}

	// Verify context
	if we.Context["operation"] != "add" {
		t.Errorf("expected context operation 'add', got %q", we.Context["operation"])
	}

	// Verify suggestions are present
	if len(we.Suggestions) == 0 {
		t.Error("expected suggestions, got none")
	}
}

func TestAdd_EmptySampleID(t *testing.T) {
	store := NewStore()
	sample := Sample{
		ID:      "", // Empty ID
		Content: "test content",
		HiddenState: &yarn.HiddenState{
			Vector: []float32{1.0, 2.0, 3.0, 4.0},
			Layer:  0,
			DType:  "float32",
		},
		ExtractedAt: time.Now(),
	}

	err := store.Add("test-concept", sample)

	if err == nil {
		t.Fatal("expected error for empty sample ID, got nil")
	}

	we, ok := werrors.AsWeaverError(err)
	if !ok {
		t.Fatalf("expected WeaverError, got %T", err)
	}

	if we.Code != werrors.ErrConceptsEmptySampleID {
		t.Errorf("expected code %q, got %q", werrors.ErrConceptsEmptySampleID, we.Code)
	}

	if we.Category != werrors.CategoryValidation {
		t.Errorf("expected category %v, got %v", werrors.CategoryValidation, we.Category)
	}

	// Verify context includes concept name
	if we.Context["concept"] != "test-concept" {
		t.Errorf("expected context concept 'test-concept', got %q", we.Context["concept"])
	}

	if we.Context["operation"] != "add" {
		t.Errorf("expected context operation 'add', got %q", we.Context["operation"])
	}

	// Verify suggestions are present
	if len(we.Suggestions) == 0 {
		t.Error("expected suggestions, got none")
	}
}

func TestAdd_InvalidHiddenState(t *testing.T) {
	tests := []struct {
		name        string
		hiddenState *yarn.HiddenState
		wantField   string
		wantMessage string
	}{
		{
			name: "empty vector",
			hiddenState: &yarn.HiddenState{
				Vector: []float32{},
				Layer:  0,
				DType:  "float32",
			},
			wantField:   "vector",
			wantMessage: "vector is required",
		},
		{
			name: "nil vector",
			hiddenState: &yarn.HiddenState{
				Vector: nil,
				Layer:  0,
				DType:  "float32",
			},
			wantField:   "vector",
			wantMessage: "vector is required",
		},
		{
			name: "negative layer",
			hiddenState: &yarn.HiddenState{
				Vector: []float32{1.0, 2.0, 3.0},
				Layer:  -1,
				DType:  "float32",
			},
			wantField:   "layer",
			wantMessage: "layer must be non-negative",
		},
		{
			name: "invalid dtype",
			hiddenState: &yarn.HiddenState{
				Vector: []float32{1.0, 2.0, 3.0},
				Layer:  0,
				DType:  "float64", // Invalid - must be float32 or float16
			},
			wantField:   "dtype",
			wantMessage: "dtype must be 'float32' or 'float16'",
		},
		{
			name: "shape inconsistent with vector length",
			hiddenState: &yarn.HiddenState{
				Vector: []float32{1.0, 2.0, 3.0, 4.0}, // 4 elements
				Shape:  []int{2, 3},                   // 6 != 4
				Layer:  0,
				DType:  "float32",
			},
			wantField:   "shape",
			wantMessage: "shape is inconsistent with vector length",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := NewStore()
			sample := Sample{
				ID:          "sample-1",
				Content:     "test content",
				HiddenState: tt.hiddenState,
				ExtractedAt: time.Now(),
			}

			err := store.Add("test-concept", sample)

			if err == nil {
				t.Fatal("expected error for invalid hidden state, got nil")
			}

			we, ok := werrors.AsWeaverError(err)
			if !ok {
				t.Fatalf("expected WeaverError, got %T", err)
			}

			if we.Code != werrors.ErrConceptsSampleInvalid {
				t.Errorf("expected code %q, got %q", werrors.ErrConceptsSampleInvalid, we.Code)
			}

			if we.Category != werrors.CategoryValidation {
				t.Errorf("expected category %v, got %v", werrors.CategoryValidation, we.Category)
			}

			// Verify context includes validation details
			if we.Context["concept"] != "test-concept" {
				t.Errorf("expected context concept 'test-concept', got %q", we.Context["concept"])
			}
			if we.Context["sample_id"] != "sample-1" {
				t.Errorf("expected context sample_id 'sample-1', got %q", we.Context["sample_id"])
			}
			if we.Context["validation_field"] != tt.wantField {
				t.Errorf("expected context validation_field %q, got %q", tt.wantField, we.Context["validation_field"])
			}
			if we.Context["validation_error"] != tt.wantMessage {
				t.Errorf("expected context validation_error %q, got %q", tt.wantMessage, we.Context["validation_error"])
			}
			if we.Context["operation"] != "add" {
				t.Errorf("expected context operation 'add', got %q", we.Context["operation"])
			}

			// Verify suggestions are present
			if len(we.Suggestions) == 0 {
				t.Error("expected suggestions, got none")
			}
		})
	}
}

func TestAdd_DimensionMismatch(t *testing.T) {
	store := NewStore()

	// Add first sample with dimension 128
	sample1 := makeValidSample("sample-1", 128)
	if err := store.Add("test-concept", sample1); err != nil {
		t.Fatalf("failed to add first sample: %v", err)
	}

	// Try to add second sample with different dimension (256)
	sample2 := makeValidSample("sample-2", 256)
	err := store.Add("test-concept", sample2)

	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}

	we, ok := werrors.AsWeaverError(err)
	if !ok {
		t.Fatalf("expected WeaverError, got %T", err)
	}

	if we.Code != werrors.ErrConceptsDimensionMismatch {
		t.Errorf("expected code %q, got %q", werrors.ErrConceptsDimensionMismatch, we.Code)
	}

	if we.Category != werrors.CategoryValidation {
		t.Errorf("expected category %v, got %v", werrors.CategoryValidation, we.Category)
	}

	// Verify context includes dimension details
	if we.Context["concept"] != "test-concept" {
		t.Errorf("expected context concept 'test-concept', got %q", we.Context["concept"])
	}
	if we.Context["sample_id"] != "sample-2" {
		t.Errorf("expected context sample_id 'sample-2', got %q", we.Context["sample_id"])
	}
	if we.Context["expected_dimension"] != "128" {
		t.Errorf("expected context expected_dimension '128', got %q", we.Context["expected_dimension"])
	}
	if we.Context["actual_dimension"] != "256" {
		t.Errorf("expected context actual_dimension '256', got %q", we.Context["actual_dimension"])
	}
	if we.Context["operation"] != "add" {
		t.Errorf("expected context operation 'add', got %q", we.Context["operation"])
	}

	// Verify suggestions are present
	if len(we.Suggestions) == 0 {
		t.Error("expected suggestions, got none")
	}

	// Verify the failed sample was not added
	concept, ok := store.Get("test-concept")
	if !ok {
		t.Fatal("expected concept to exist")
	}
	if len(concept.Samples) != 1 {
		t.Errorf("expected 1 sample, got %d", len(concept.Samples))
	}
}

func TestAdd_Success(t *testing.T) {
	store := NewStore()
	sample := makeValidSample("sample-1", 128)

	err := store.Add("test-concept", sample)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the sample was added
	concept, ok := store.Get("test-concept")
	if !ok {
		t.Fatal("expected concept to exist")
	}

	if len(concept.Samples) != 1 {
		t.Errorf("expected 1 sample, got %d", len(concept.Samples))
	}

	if concept.Samples[0].ID != "sample-1" {
		t.Errorf("expected sample ID 'sample-1', got %q", concept.Samples[0].ID)
	}

	if concept.Name != "test-concept" {
		t.Errorf("expected concept name 'test-concept', got %q", concept.Name)
	}
}

func TestAdd_SuccessWithNilHiddenState(t *testing.T) {
	store := NewStore()
	sample := Sample{
		ID:          "sample-1",
		Content:     "test content",
		HiddenState: nil, // Nil hidden state is valid (skips validation)
		ExtractedAt: time.Now(),
	}

	err := store.Add("test-concept", sample)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the sample was added
	concept, ok := store.Get("test-concept")
	if !ok {
		t.Fatal("expected concept to exist")
	}

	if len(concept.Samples) != 1 {
		t.Errorf("expected 1 sample, got %d", len(concept.Samples))
	}
}

func TestAdd_MultipleSuccessfulSamples(t *testing.T) {
	store := NewStore()

	// Add multiple samples with same dimension
	sampleIDs := []string{"sample-1", "sample-2", "sample-3", "sample-4", "sample-5"}
	for _, id := range sampleIDs {
		sample := makeValidSample(id, 128)
		if err := store.Add("test-concept", sample); err != nil {
			t.Fatalf("failed to add sample %s: %v", id, err)
		}
	}

	// Verify all samples were added
	concept, ok := store.Get("test-concept")
	if !ok {
		t.Fatal("expected concept to exist")
	}

	if len(concept.Samples) != len(sampleIDs) {
		t.Errorf("expected %d samples, got %d", len(sampleIDs), len(concept.Samples))
	}

	// Verify sample IDs
	for i, expectedID := range sampleIDs {
		if concept.Samples[i].ID != expectedID {
			t.Errorf("sample %d: expected ID %q, got %q", i, expectedID, concept.Samples[i].ID)
		}
	}

	// Verify dimension is consistent
	dim, mismatched := concept.ValidateDimensions()
	if dim != 128 {
		t.Errorf("expected dimension 128, got %d", dim)
	}
	if len(mismatched) != 0 {
		t.Errorf("expected no mismatched samples, got %v", mismatched)
	}
}

func TestAdd_MultipleConcepts(t *testing.T) {
	store := NewStore()

	// Add samples to different concepts
	concepts := []struct {
		name      string
		dimension int
	}{
		{"concept-a", 128},
		{"concept-b", 256},
		{"concept-c", 512},
	}

	for _, c := range concepts {
		for i := 1; i <= 3; i++ {
			sample := makeValidSample("sample-"+string(rune('0'+i)), c.dimension)
			if err := store.Add(c.name, sample); err != nil {
				t.Fatalf("failed to add sample to %s: %v", c.name, err)
			}
		}
	}

	// Verify all concepts exist with correct sample counts
	list := store.List()
	if len(list) != 3 {
		t.Errorf("expected 3 concepts, got %d", len(list))
	}

	for _, c := range concepts {
		count, ok := list[c.name]
		if !ok {
			t.Errorf("concept %s not found", c.name)
			continue
		}
		if count != 3 {
			t.Errorf("concept %s: expected 3 samples, got %d", c.name, count)
		}
	}
}

func TestAdd_DimensionMismatchDoesNotAffectOtherConcepts(t *testing.T) {
	store := NewStore()

	// Add sample to concept-a with dim 128
	sample1 := makeValidSample("sample-1", 128)
	if err := store.Add("concept-a", sample1); err != nil {
		t.Fatalf("failed to add sample to concept-a: %v", err)
	}

	// Add sample to concept-b with dim 256 (should succeed - different concept)
	sample2 := makeValidSample("sample-2", 256)
	if err := store.Add("concept-b", sample2); err != nil {
		t.Fatalf("failed to add sample to concept-b: %v", err)
	}

	// Try to add sample to concept-a with dim 256 (should fail - dimension mismatch)
	sample3 := makeValidSample("sample-3", 256)
	err := store.Add("concept-a", sample3)
	if err == nil {
		t.Fatal("expected dimension mismatch error")
	}

	// Verify concept-b still has its sample
	conceptB, ok := store.Get("concept-b")
	if !ok {
		t.Fatal("concept-b should exist")
	}
	if len(conceptB.Samples) != 1 {
		t.Errorf("concept-b: expected 1 sample, got %d", len(conceptB.Samples))
	}
}

// -----------------------------------------------------------------------------
// Validation Order Tests
// -----------------------------------------------------------------------------

func TestAdd_ValidationOrder(t *testing.T) {
	// Test that validations happen in the correct order:
	// 1. Empty concept name
	// 2. Empty sample ID
	// 3. Hidden state validation
	// 4. Dimension mismatch

	store := NewStore()

	// First add a valid sample so we can test dimension mismatch
	sample := makeValidSample("sample-0", 128)
	if err := store.Add("test-concept", sample); err != nil {
		t.Fatalf("setup failed: %v", err)
	}

	t.Run("empty concept name checked first", func(t *testing.T) {
		// All fields are invalid, but empty concept name should be caught first
		badSample := Sample{
			ID:          "",
			HiddenState: &yarn.HiddenState{Vector: nil},
		}
		err := store.Add("", badSample)
		we, ok := werrors.AsWeaverError(err)
		if !ok {
			t.Fatalf("expected WeaverError, got %T", err)
		}
		if we.Code != werrors.ErrConceptsEmptyName {
			t.Errorf("expected %q to be caught first, got %q", werrors.ErrConceptsEmptyName, we.Code)
		}
	})

	t.Run("empty sample ID checked second", func(t *testing.T) {
		// Sample ID is empty, hidden state is invalid
		badSample := Sample{
			ID:          "",
			HiddenState: &yarn.HiddenState{Vector: nil},
		}
		err := store.Add("test-concept", badSample)
		we, ok := werrors.AsWeaverError(err)
		if !ok {
			t.Fatalf("expected WeaverError, got %T", err)
		}
		if we.Code != werrors.ErrConceptsEmptySampleID {
			t.Errorf("expected %q to be caught second, got %q", werrors.ErrConceptsEmptySampleID, we.Code)
		}
	})

	t.Run("hidden state validation checked third", func(t *testing.T) {
		// Hidden state is invalid, dimension would mismatch
		badSample := Sample{
			ID:          "sample-1",
			HiddenState: &yarn.HiddenState{Vector: nil}, // Invalid
		}
		err := store.Add("test-concept", badSample)
		we, ok := werrors.AsWeaverError(err)
		if !ok {
			t.Fatalf("expected WeaverError, got %T", err)
		}
		if we.Code != werrors.ErrConceptsSampleInvalid {
			t.Errorf("expected %q to be caught third, got %q", werrors.ErrConceptsSampleInvalid, we.Code)
		}
	})

	t.Run("dimension mismatch checked last", func(t *testing.T) {
		// All other validations pass, but dimension mismatches
		badSample := makeValidSample("sample-1", 256) // 256 != 128
		err := store.Add("test-concept", badSample)
		we, ok := werrors.AsWeaverError(err)
		if !ok {
			t.Fatalf("expected WeaverError, got %T", err)
		}
		if we.Code != werrors.ErrConceptsDimensionMismatch {
			t.Errorf("expected %q to be caught last, got %q", werrors.ErrConceptsDimensionMismatch, we.Code)
		}
	})
}
