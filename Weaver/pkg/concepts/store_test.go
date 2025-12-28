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
// Error Helper Function Tests
// -----------------------------------------------------------------------------
// These tests verify that the error helper functions create WeaverErrors with
// the correct error codes, context fields, and suggestions.

func TestCreateEmptyConceptNameError(t *testing.T) {
	err := createEmptyConceptNameError()

	if err == nil {
		t.Fatal("expected error, got nil")
	}

	// Verify error code
	if err.Code != werrors.ErrConceptsEmptyName {
		t.Errorf("expected code %q, got %q", werrors.ErrConceptsEmptyName, err.Code)
	}

	// Verify category
	if err.Category != werrors.CategoryValidation {
		t.Errorf("expected category %v, got %v", werrors.CategoryValidation, err.Category)
	}

	// Verify message
	if err.Message != "concept name cannot be empty" {
		t.Errorf("expected message 'concept name cannot be empty', got %q", err.Message)
	}

	// Verify context
	if err.Context == nil {
		t.Fatal("expected context, got nil")
	}
	if err.Context["operation"] != "add" {
		t.Errorf("expected context operation 'add', got %q", err.Context["operation"])
	}

	// Verify suggestions
	if len(err.Suggestions) != 3 {
		t.Errorf("expected 3 suggestions, got %d", len(err.Suggestions))
	}

	// Verify suggestion content
	foundDescriptiveName := false
	foundNamingConvention := false
	foundExtractCommand := false
	for _, s := range err.Suggestions {
		if s == "Provide a descriptive name for the concept (e.g., 'recursion', 'sorting', 'authentication')" {
			foundDescriptiveName = true
		}
		if s == "Concept names should be lowercase with optional hyphens or underscores" {
			foundNamingConvention = true
		}
		if s == "Use '/extract <concept-name> <count>' to specify a valid concept name" {
			foundExtractCommand = true
		}
	}
	if !foundDescriptiveName {
		t.Error("missing suggestion about descriptive concept names")
	}
	if !foundNamingConvention {
		t.Error("missing suggestion about naming conventions")
	}
	if !foundExtractCommand {
		t.Error("missing suggestion about /extract command")
	}
}

func TestCreateEmptySampleIDError(t *testing.T) {
	err := createEmptySampleIDError("test-concept")

	if err == nil {
		t.Fatal("expected error, got nil")
	}

	// Verify error code
	if err.Code != werrors.ErrConceptsEmptySampleID {
		t.Errorf("expected code %q, got %q", werrors.ErrConceptsEmptySampleID, err.Code)
	}

	// Verify category
	if err.Category != werrors.CategoryValidation {
		t.Errorf("expected category %v, got %v", werrors.CategoryValidation, err.Category)
	}

	// Verify message
	if err.Message != "sample ID cannot be empty" {
		t.Errorf("expected message 'sample ID cannot be empty', got %q", err.Message)
	}

	// Verify context
	if err.Context == nil {
		t.Fatal("expected context, got nil")
	}
	if err.Context["concept"] != "test-concept" {
		t.Errorf("expected context concept 'test-concept', got %q", err.Context["concept"])
	}
	if err.Context["operation"] != "add" {
		t.Errorf("expected context operation 'add', got %q", err.Context["operation"])
	}

	// Verify suggestions
	if len(err.Suggestions) != 3 {
		t.Errorf("expected 3 suggestions, got %d", len(err.Suggestions))
	}

	// Verify suggestion content
	foundUUID := false
	foundUniqueness := false
	foundTimestamp := false
	for _, s := range err.Suggestions {
		if s == "Generate a unique sample ID using UUID (e.g., uuid.New().String())" {
			foundUUID = true
		}
		if s == "Sample IDs must be unique within a concept to allow proper tracking" {
			foundUniqueness = true
		}
		if s == "Consider using a combination of timestamp and random suffix if UUID is unavailable" {
			foundTimestamp = true
		}
	}
	if !foundUUID {
		t.Error("missing suggestion about UUID generation")
	}
	if !foundUniqueness {
		t.Error("missing suggestion about uniqueness requirement")
	}
	if !foundTimestamp {
		t.Error("missing suggestion about timestamp alternative")
	}
}

func TestCreateInvalidHiddenStateError(t *testing.T) {
	validationErr := &yarn.ValidationError{
		Field:   "vector",
		Message: "vector is required",
	}

	err := createInvalidHiddenStateError("test-concept", "sample-123", validationErr)

	if err == nil {
		t.Fatal("expected error, got nil")
	}

	// Verify error code
	if err.Code != werrors.ErrConceptsSampleInvalid {
		t.Errorf("expected code %q, got %q", werrors.ErrConceptsSampleInvalid, err.Code)
	}

	// Verify category
	if err.Category != werrors.CategoryValidation {
		t.Errorf("expected category %v, got %v", werrors.CategoryValidation, err.Category)
	}

	// Verify message
	if err.Message != "hidden state validation failed" {
		t.Errorf("expected message 'hidden state validation failed', got %q", err.Message)
	}

	// Verify context
	if err.Context == nil {
		t.Fatal("expected context, got nil")
	}
	if err.Context["concept"] != "test-concept" {
		t.Errorf("expected context concept 'test-concept', got %q", err.Context["concept"])
	}
	if err.Context["sample_id"] != "sample-123" {
		t.Errorf("expected context sample_id 'sample-123', got %q", err.Context["sample_id"])
	}
	if err.Context["validation_field"] != "vector" {
		t.Errorf("expected context validation_field 'vector', got %q", err.Context["validation_field"])
	}
	if err.Context["validation_error"] != "vector is required" {
		t.Errorf("expected context validation_error 'vector is required', got %q", err.Context["validation_error"])
	}
	if err.Context["operation"] != "add" {
		t.Errorf("expected context operation 'add', got %q", err.Context["operation"])
	}

	// Verify cause is wrapped
	if err.Cause != validationErr {
		t.Errorf("expected cause to be the validation error, got %v", err.Cause)
	}

	// Verify suggestions
	if len(err.Suggestions) != 4 {
		t.Errorf("expected 4 suggestions, got %d", len(err.Suggestions))
	}

	// Verify suggestion content
	foundReExtract := false
	foundModelSupport := false
	foundVectorCheck := false
	foundDtypeCheck := false
	for _, s := range err.Suggestions {
		if s == "Re-extract the sample with valid hidden state data using '/extract'" {
			foundReExtract = true
		}
		if s == "Ensure the model supports hidden state extraction" {
			foundModelSupport = true
		}
		if s == "Check that the hidden state vector is not empty and has valid dimensions" {
			foundVectorCheck = true
		}
		if s == "Verify the dtype is 'float32' or 'float16'" {
			foundDtypeCheck = true
		}
	}
	if !foundReExtract {
		t.Error("missing suggestion about re-extracting sample")
	}
	if !foundModelSupport {
		t.Error("missing suggestion about model support")
	}
	if !foundVectorCheck {
		t.Error("missing suggestion about vector check")
	}
	if !foundDtypeCheck {
		t.Error("missing suggestion about dtype check")
	}
}

func TestCreateInvalidHiddenStateError_DifferentValidationErrors(t *testing.T) {
	tests := []struct {
		name        string
		field       string
		message     string
		wantField   string
		wantMessage string
	}{
		{
			name:        "layer validation error",
			field:       "layer",
			message:     "layer must be non-negative",
			wantField:   "layer",
			wantMessage: "layer must be non-negative",
		},
		{
			name:        "dtype validation error",
			field:       "dtype",
			message:     "dtype must be 'float32' or 'float16'",
			wantField:   "dtype",
			wantMessage: "dtype must be 'float32' or 'float16'",
		},
		{
			name:        "shape validation error",
			field:       "shape",
			message:     "shape is inconsistent with vector length",
			wantField:   "shape",
			wantMessage: "shape is inconsistent with vector length",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validationErr := &yarn.ValidationError{
				Field:   tt.field,
				Message: tt.message,
			}

			err := createInvalidHiddenStateError("my-concept", "sample-abc", validationErr)

			if err.Context["validation_field"] != tt.wantField {
				t.Errorf("expected validation_field %q, got %q", tt.wantField, err.Context["validation_field"])
			}
			if err.Context["validation_error"] != tt.wantMessage {
				t.Errorf("expected validation_error %q, got %q", tt.wantMessage, err.Context["validation_error"])
			}
		})
	}
}

func TestCreateDimensionMismatchError(t *testing.T) {
	err := createDimensionMismatchError("test-concept", "sample-456", 128, 256)

	if err == nil {
		t.Fatal("expected error, got nil")
	}

	// Verify error code
	if err.Code != werrors.ErrConceptsDimensionMismatch {
		t.Errorf("expected code %q, got %q", werrors.ErrConceptsDimensionMismatch, err.Code)
	}

	// Verify category
	if err.Category != werrors.CategoryValidation {
		t.Errorf("expected category %v, got %v", werrors.CategoryValidation, err.Category)
	}

	// Verify message
	if err.Message != "sample dimension does not match existing concept dimension" {
		t.Errorf("expected message 'sample dimension does not match existing concept dimension', got %q", err.Message)
	}

	// Verify context
	if err.Context == nil {
		t.Fatal("expected context, got nil")
	}
	if err.Context["concept"] != "test-concept" {
		t.Errorf("expected context concept 'test-concept', got %q", err.Context["concept"])
	}
	if err.Context["sample_id"] != "sample-456" {
		t.Errorf("expected context sample_id 'sample-456', got %q", err.Context["sample_id"])
	}
	if err.Context["expected_dimension"] != "128" {
		t.Errorf("expected context expected_dimension '128', got %q", err.Context["expected_dimension"])
	}
	if err.Context["actual_dimension"] != "256" {
		t.Errorf("expected context actual_dimension '256', got %q", err.Context["actual_dimension"])
	}
	if err.Context["operation"] != "add" {
		t.Errorf("expected context operation 'add', got %q", err.Context["operation"])
	}

	// Verify suggestions
	if len(err.Suggestions) != 4 {
		t.Errorf("expected 4 suggestions, got %d", len(err.Suggestions))
	}

	// Verify suggestion content
	foundConsistency := false
	foundReExtract := false
	foundListCommand := false
	foundNewConcept := false
	for _, s := range err.Suggestions {
		if s == "All samples in a concept must have the same hidden state dimension" {
			foundConsistency = true
		}
		if s == "Re-extract this sample using the same model as existing samples" {
			foundReExtract = true
		}
		if s == "Use '/concepts list' to see existing concepts and their sample counts" {
			foundListCommand = true
		}
		if s == "Consider creating a new concept if using a different model" {
			foundNewConcept = true
		}
	}
	if !foundConsistency {
		t.Error("missing suggestion about dimension consistency")
	}
	if !foundReExtract {
		t.Error("missing suggestion about re-extracting with same model")
	}
	if !foundListCommand {
		t.Error("missing suggestion about /concepts list command")
	}
	if !foundNewConcept {
		t.Error("missing suggestion about creating new concept")
	}
}

func TestCreateDimensionMismatchError_DifferentDimensions(t *testing.T) {
	tests := []struct {
		name        string
		expected    int
		actual      int
		wantExpStr  string
		wantActStr  string
	}{
		{
			name:       "small dimensions",
			expected:   64,
			actual:     128,
			wantExpStr: "64",
			wantActStr: "128",
		},
		{
			name:       "large dimensions",
			expected:   4096,
			actual:     8192,
			wantExpStr: "4096",
			wantActStr: "8192",
		},
		{
			name:       "actual smaller than expected",
			expected:   512,
			actual:     256,
			wantExpStr: "512",
			wantActStr: "256",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := createDimensionMismatchError("concept-x", "sample-y", tt.expected, tt.actual)

			if err.Context["expected_dimension"] != tt.wantExpStr {
				t.Errorf("expected expected_dimension %q, got %q", tt.wantExpStr, err.Context["expected_dimension"])
			}
			if err.Context["actual_dimension"] != tt.wantActStr {
				t.Errorf("expected actual_dimension %q, got %q", tt.wantActStr, err.Context["actual_dimension"])
			}
		})
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
