package concepts

import (
	"testing"
	"time"

	"github.com/r3d91ll/yarn"
)

// Helper function to create a HiddenState with a specific dimension.
func makeHiddenState(dim int) *yarn.HiddenState {
	if dim == 0 {
		return nil
	}
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = float32(i)
	}
	return &yarn.HiddenState{
		Vector: vector,
		Shape:  []int{1, 1, dim},
		Layer:  12,
		DType:  "float32",
	}
}

// Helper function to create a sample with the given parameters.
func makeSample(id, content, model string, dim int, extractedAt time.Time) Sample {
	return Sample{
		ID:          id,
		Content:     content,
		HiddenState: makeHiddenState(dim),
		ExtractedAt: extractedAt,
		Model:       model,
	}
}

// =============================================================================
// Concept.Stats() Tests
// =============================================================================

// TestConceptStats_Empty tests Stats() on a concept with no samples.
func TestConceptStats_Empty(t *testing.T) {
	now := time.Now()
	concept := &Concept{
		Name:      "empty-concept",
		Samples:   []Sample{},
		CreatedAt: now,
		UpdatedAt: now,
	}

	stats := concept.Stats()

	if stats.Name != "empty-concept" {
		t.Errorf("expected Name 'empty-concept', got %q", stats.Name)
	}
	if stats.SampleCount != 0 {
		t.Errorf("expected SampleCount 0, got %d", stats.SampleCount)
	}
	if stats.Dimension != 0 {
		t.Errorf("expected Dimension 0, got %d", stats.Dimension)
	}
	if len(stats.MismatchedIDs) != 0 {
		t.Errorf("expected no MismatchedIDs, got %v", stats.MismatchedIDs)
	}
	if len(stats.Models) != 0 {
		t.Errorf("expected no Models, got %v", stats.Models)
	}
	if !stats.OldestSampleAt.IsZero() {
		t.Errorf("expected zero OldestSampleAt, got %v", stats.OldestSampleAt)
	}
	if !stats.NewestSampleAt.IsZero() {
		t.Errorf("expected zero NewestSampleAt, got %v", stats.NewestSampleAt)
	}
	if !stats.CreatedAt.Equal(now) {
		t.Errorf("expected CreatedAt %v, got %v", now, stats.CreatedAt)
	}
	if !stats.UpdatedAt.Equal(now) {
		t.Errorf("expected UpdatedAt %v, got %v", now, stats.UpdatedAt)
	}
}

// TestConceptStats_SingleSample tests Stats() on a concept with one sample.
func TestConceptStats_SingleSample(t *testing.T) {
	now := time.Now()
	extractedAt := now.Add(-1 * time.Hour)

	concept := &Concept{
		Name: "single-sample-concept",
		Samples: []Sample{
			makeSample("sample-1", "test content", "gpt-4", 768, extractedAt),
		},
		CreatedAt: now.Add(-2 * time.Hour),
		UpdatedAt: now,
	}

	stats := concept.Stats()

	if stats.Name != "single-sample-concept" {
		t.Errorf("expected Name 'single-sample-concept', got %q", stats.Name)
	}
	if stats.SampleCount != 1 {
		t.Errorf("expected SampleCount 1, got %d", stats.SampleCount)
	}
	if stats.Dimension != 768 {
		t.Errorf("expected Dimension 768, got %d", stats.Dimension)
	}
	if len(stats.MismatchedIDs) != 0 {
		t.Errorf("expected no MismatchedIDs, got %v", stats.MismatchedIDs)
	}
	if len(stats.Models) != 1 || stats.Models[0] != "gpt-4" {
		t.Errorf("expected Models ['gpt-4'], got %v", stats.Models)
	}
	if !stats.OldestSampleAt.Equal(extractedAt) {
		t.Errorf("expected OldestSampleAt %v, got %v", extractedAt, stats.OldestSampleAt)
	}
	if !stats.NewestSampleAt.Equal(extractedAt) {
		t.Errorf("expected NewestSampleAt %v, got %v", extractedAt, stats.NewestSampleAt)
	}
}

// TestConceptStats_MultipleSamples tests Stats() on a concept with multiple samples.
func TestConceptStats_MultipleSamples(t *testing.T) {
	now := time.Now()
	oldTime := now.Add(-24 * time.Hour)
	midTime := now.Add(-12 * time.Hour)
	newTime := now.Add(-1 * time.Hour)

	concept := &Concept{
		Name: "multi-sample-concept",
		Samples: []Sample{
			makeSample("sample-1", "content 1", "gpt-4", 1024, midTime),
			makeSample("sample-2", "content 2", "gpt-4", 1024, oldTime),
			makeSample("sample-3", "content 3", "gpt-4", 1024, newTime),
		},
		CreatedAt: now.Add(-48 * time.Hour),
		UpdatedAt: now,
	}

	stats := concept.Stats()

	if stats.SampleCount != 3 {
		t.Errorf("expected SampleCount 3, got %d", stats.SampleCount)
	}
	if stats.Dimension != 1024 {
		t.Errorf("expected Dimension 1024, got %d", stats.Dimension)
	}
	if len(stats.MismatchedIDs) != 0 {
		t.Errorf("expected no MismatchedIDs, got %v", stats.MismatchedIDs)
	}
	if !stats.OldestSampleAt.Equal(oldTime) {
		t.Errorf("expected OldestSampleAt %v, got %v", oldTime, stats.OldestSampleAt)
	}
	if !stats.NewestSampleAt.Equal(newTime) {
		t.Errorf("expected NewestSampleAt %v, got %v", newTime, stats.NewestSampleAt)
	}
}

// TestConceptStats_MismatchedDimensions tests Stats() with samples of different dimensions.
func TestConceptStats_MismatchedDimensions(t *testing.T) {
	t.Run("single mismatch", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "mismatched-concept",
			Samples: []Sample{
				makeSample("sample-1", "content 1", "gpt-4", 1024, now),
				makeSample("sample-2", "content 2", "gpt-4", 1024, now),
				makeSample("sample-3", "content 3", "gpt-4", 768, now), // Different dimension
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		if stats.Dimension != 1024 {
			t.Errorf("expected Dimension 1024 (first valid), got %d", stats.Dimension)
		}
		if len(stats.MismatchedIDs) != 1 {
			t.Errorf("expected 1 MismatchedID, got %d: %v", len(stats.MismatchedIDs), stats.MismatchedIDs)
		}
		if len(stats.MismatchedIDs) > 0 && stats.MismatchedIDs[0] != "sample-3" {
			t.Errorf("expected MismatchedID 'sample-3', got %v", stats.MismatchedIDs)
		}
	})

	t.Run("multiple mismatches", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "multiple-mismatches",
			Samples: []Sample{
				makeSample("sample-1", "content 1", "gpt-4", 1024, now),
				makeSample("sample-2", "content 2", "gpt-4", 512, now),  // Mismatch
				makeSample("sample-3", "content 3", "gpt-4", 768, now),  // Mismatch
				makeSample("sample-4", "content 4", "gpt-4", 1024, now), // Matches
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		if stats.Dimension != 1024 {
			t.Errorf("expected Dimension 1024, got %d", stats.Dimension)
		}
		if len(stats.MismatchedIDs) != 2 {
			t.Errorf("expected 2 MismatchedIDs, got %d: %v", len(stats.MismatchedIDs), stats.MismatchedIDs)
		}
	})

	t.Run("nil hidden states ignored", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "nil-hidden-states",
			Samples: []Sample{
				makeSample("sample-1", "content 1", "gpt-4", 1024, now),
				{ID: "sample-2", Content: "content 2", Model: "gpt-4", ExtractedAt: now, HiddenState: nil},
				makeSample("sample-3", "content 3", "gpt-4", 1024, now),
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		if stats.Dimension != 1024 {
			t.Errorf("expected Dimension 1024, got %d", stats.Dimension)
		}
		// nil HiddenState should be skipped in mismatch checking
		if len(stats.MismatchedIDs) != 0 {
			t.Errorf("expected 0 MismatchedIDs (nil should be ignored), got %d: %v", len(stats.MismatchedIDs), stats.MismatchedIDs)
		}
	})
}

// TestConceptStats_Models tests Stats() model collection.
func TestConceptStats_Models(t *testing.T) {
	t.Run("single model", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "single-model",
			Samples: []Sample{
				makeSample("sample-1", "content 1", "gpt-4", 1024, now),
				makeSample("sample-2", "content 2", "gpt-4", 1024, now),
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		if len(stats.Models) != 1 {
			t.Errorf("expected 1 model, got %d: %v", len(stats.Models), stats.Models)
		}
		if len(stats.Models) > 0 && stats.Models[0] != "gpt-4" {
			t.Errorf("expected model 'gpt-4', got %v", stats.Models)
		}
	})

	t.Run("multiple models sorted", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "multiple-models",
			Samples: []Sample{
				makeSample("sample-1", "content 1", "gpt-4", 1024, now),
				makeSample("sample-2", "content 2", "claude-3", 1024, now),
				makeSample("sample-3", "content 3", "gpt-3.5", 1024, now),
				makeSample("sample-4", "content 4", "claude-3", 1024, now), // Duplicate
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		// Should have 3 unique models, sorted
		if len(stats.Models) != 3 {
			t.Errorf("expected 3 unique models, got %d: %v", len(stats.Models), stats.Models)
		}
		expectedModels := []string{"claude-3", "gpt-3.5", "gpt-4"}
		for i, expected := range expectedModels {
			if i < len(stats.Models) && stats.Models[i] != expected {
				t.Errorf("expected Models[%d] = %q, got %q", i, expected, stats.Models[i])
			}
		}
	})

	t.Run("empty model field", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "empty-model",
			Samples: []Sample{
				makeSample("sample-1", "content 1", "", 1024, now),       // Empty model
				makeSample("sample-2", "content 2", "gpt-4", 1024, now),  // Has model
				makeSample("sample-3", "content 3", "", 1024, now),       // Empty model
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		// Should only include non-empty model
		if len(stats.Models) != 1 {
			t.Errorf("expected 1 model (empty should be excluded), got %d: %v", len(stats.Models), stats.Models)
		}
		if len(stats.Models) > 0 && stats.Models[0] != "gpt-4" {
			t.Errorf("expected model 'gpt-4', got %v", stats.Models)
		}
	})

	t.Run("all empty models", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "all-empty-models",
			Samples: []Sample{
				makeSample("sample-1", "content 1", "", 1024, now),
				makeSample("sample-2", "content 2", "", 1024, now),
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		if len(stats.Models) != 0 {
			t.Errorf("expected 0 models, got %d: %v", len(stats.Models), stats.Models)
		}
	})
}

// TestConceptStats_NilHiddenState tests Stats() with nil hidden states.
func TestConceptStats_NilHiddenState(t *testing.T) {
	t.Run("all nil hidden states", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "all-nil-hidden-states",
			Samples: []Sample{
				{ID: "sample-1", Content: "content 1", Model: "gpt-4", ExtractedAt: now, HiddenState: nil},
				{ID: "sample-2", Content: "content 2", Model: "gpt-4", ExtractedAt: now, HiddenState: nil},
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		if stats.SampleCount != 2 {
			t.Errorf("expected SampleCount 2, got %d", stats.SampleCount)
		}
		if stats.Dimension != 0 {
			t.Errorf("expected Dimension 0 (no hidden states), got %d", stats.Dimension)
		}
		if len(stats.MismatchedIDs) != 0 {
			t.Errorf("expected no MismatchedIDs, got %v", stats.MismatchedIDs)
		}
	})

	t.Run("mixed nil and valid hidden states", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "mixed-hidden-states",
			Samples: []Sample{
				{ID: "sample-1", Content: "content 1", Model: "gpt-4", ExtractedAt: now, HiddenState: nil},
				makeSample("sample-2", "content 2", "gpt-4", 1024, now),
				{ID: "sample-3", Content: "content 3", Model: "gpt-4", ExtractedAt: now, HiddenState: nil},
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		if stats.SampleCount != 3 {
			t.Errorf("expected SampleCount 3, got %d", stats.SampleCount)
		}
		if stats.Dimension != 1024 {
			t.Errorf("expected Dimension 1024, got %d", stats.Dimension)
		}
	})
}

// TestConceptStats_ZeroExtractedAt tests Stats() with zero extraction times.
func TestConceptStats_ZeroExtractedAt(t *testing.T) {
	t.Run("all zero times", func(t *testing.T) {
		now := time.Now()
		zeroTime := time.Time{}
		concept := &Concept{
			Name: "zero-times",
			Samples: []Sample{
				{ID: "sample-1", Content: "content 1", Model: "gpt-4", ExtractedAt: zeroTime, HiddenState: makeHiddenState(1024)},
				{ID: "sample-2", Content: "content 2", Model: "gpt-4", ExtractedAt: zeroTime, HiddenState: makeHiddenState(1024)},
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		if !stats.OldestSampleAt.IsZero() {
			t.Errorf("expected zero OldestSampleAt with all zero times, got %v", stats.OldestSampleAt)
		}
		if !stats.NewestSampleAt.IsZero() {
			t.Errorf("expected zero NewestSampleAt with all zero times, got %v", stats.NewestSampleAt)
		}
	})

	t.Run("mixed zero and valid times", func(t *testing.T) {
		now := time.Now()
		zeroTime := time.Time{}
		validTime := now.Add(-1 * time.Hour)
		concept := &Concept{
			Name: "mixed-times",
			Samples: []Sample{
				{ID: "sample-1", Content: "content 1", Model: "gpt-4", ExtractedAt: zeroTime, HiddenState: makeHiddenState(1024)},
				{ID: "sample-2", Content: "content 2", Model: "gpt-4", ExtractedAt: validTime, HiddenState: makeHiddenState(1024)},
				{ID: "sample-3", Content: "content 3", Model: "gpt-4", ExtractedAt: zeroTime, HiddenState: makeHiddenState(1024)},
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		stats := concept.Stats()

		if !stats.OldestSampleAt.Equal(validTime) {
			t.Errorf("expected OldestSampleAt %v, got %v", validTime, stats.OldestSampleAt)
		}
		if !stats.NewestSampleAt.Equal(validTime) {
			t.Errorf("expected NewestSampleAt %v, got %v", validTime, stats.NewestSampleAt)
		}
	})
}

// =============================================================================
// Concept.Models() Tests
// =============================================================================

// TestConceptModels tests the Models() method.
func TestConceptModels(t *testing.T) {
	t.Run("empty samples", func(t *testing.T) {
		concept := &Concept{
			Name:    "empty",
			Samples: []Sample{},
		}

		models := concept.Models()
		if models != nil {
			t.Errorf("expected nil for empty samples, got %v", models)
		}
	})

	t.Run("unique models sorted", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "multi-model",
			Samples: []Sample{
				makeSample("s1", "c1", "gpt-4", 1024, now),
				makeSample("s2", "c2", "claude-3", 1024, now),
				makeSample("s3", "c3", "gpt-3.5", 1024, now),
				makeSample("s4", "c4", "gpt-4", 1024, now), // Duplicate
			},
		}

		models := concept.Models()
		expected := []string{"claude-3", "gpt-3.5", "gpt-4"}

		if len(models) != len(expected) {
			t.Fatalf("expected %d models, got %d: %v", len(expected), len(models), models)
		}
		for i, exp := range expected {
			if models[i] != exp {
				t.Errorf("expected models[%d] = %q, got %q", i, exp, models[i])
			}
		}
	})

	t.Run("all empty models returns nil", func(t *testing.T) {
		now := time.Now()
		concept := &Concept{
			Name: "no-models",
			Samples: []Sample{
				makeSample("s1", "c1", "", 1024, now),
				makeSample("s2", "c2", "", 1024, now),
			},
		}

		models := concept.Models()
		if models != nil {
			t.Errorf("expected nil for all empty models, got %v", models)
		}
	})
}

// =============================================================================
// Store.Stats() Tests
// =============================================================================

// TestStoreStats_Empty tests Stats() on an empty store.
func TestStoreStats_Empty(t *testing.T) {
	store := NewStore()

	stats := store.Stats()

	if stats.ConceptCount != 0 {
		t.Errorf("expected ConceptCount 0, got %d", stats.ConceptCount)
	}
	if stats.TotalSamples != 0 {
		t.Errorf("expected TotalSamples 0, got %d", stats.TotalSamples)
	}
	if stats.HealthyConcepts != 0 {
		t.Errorf("expected HealthyConcepts 0, got %d", stats.HealthyConcepts)
	}
	if stats.ConceptsWithIssues != 0 {
		t.Errorf("expected ConceptsWithIssues 0, got %d", stats.ConceptsWithIssues)
	}
	if len(stats.Dimensions) != 0 {
		t.Errorf("expected empty Dimensions map, got %v", stats.Dimensions)
	}
	if len(stats.Models) != 0 {
		t.Errorf("expected empty Models map, got %v", stats.Models)
	}
	if len(stats.Concepts) != 0 {
		t.Errorf("expected empty Concepts map, got %v", stats.Concepts)
	}
	if !stats.OldestExtraction.IsZero() {
		t.Errorf("expected zero OldestExtraction, got %v", stats.OldestExtraction)
	}
	if !stats.NewestExtraction.IsZero() {
		t.Errorf("expected zero NewestExtraction, got %v", stats.NewestExtraction)
	}
}

// TestStoreStats_SingleConcept tests Stats() with a single concept.
func TestStoreStats_SingleConcept(t *testing.T) {
	store := NewStore()
	now := time.Now()

	store.Add("concept-1", makeSample("s1", "c1", "gpt-4", 1024, now))
	store.Add("concept-1", makeSample("s2", "c2", "gpt-4", 1024, now.Add(-1*time.Hour)))

	stats := store.Stats()

	if stats.ConceptCount != 1 {
		t.Errorf("expected ConceptCount 1, got %d", stats.ConceptCount)
	}
	if stats.TotalSamples != 2 {
		t.Errorf("expected TotalSamples 2, got %d", stats.TotalSamples)
	}
	if stats.HealthyConcepts != 1 {
		t.Errorf("expected HealthyConcepts 1, got %d", stats.HealthyConcepts)
	}
	if stats.ConceptsWithIssues != 0 {
		t.Errorf("expected ConceptsWithIssues 0, got %d", stats.ConceptsWithIssues)
	}
	if stats.Dimensions[1024] != 1 {
		t.Errorf("expected Dimensions[1024] = 1, got %v", stats.Dimensions)
	}
	if stats.Models["gpt-4"] != 2 {
		t.Errorf("expected Models['gpt-4'] = 2, got %v", stats.Models)
	}

	// Verify concept stats are included
	cs, ok := stats.Concepts["concept-1"]
	if !ok {
		t.Fatal("expected concept-1 in Concepts map")
	}
	if cs.SampleCount != 2 {
		t.Errorf("expected concept SampleCount 2, got %d", cs.SampleCount)
	}
}

// TestStoreStats_MultipleConcepts tests Stats() with multiple concepts.
func TestStoreStats_MultipleConcepts(t *testing.T) {
	store := NewStore()
	now := time.Now()

	// Add concept 1 with 2 samples
	store.Add("concept-1", makeSample("s1", "c1", "gpt-4", 1024, now))
	store.Add("concept-1", makeSample("s2", "c2", "gpt-4", 1024, now))

	// Add concept 2 with 3 samples
	store.Add("concept-2", makeSample("s3", "c3", "claude-3", 768, now))
	store.Add("concept-2", makeSample("s4", "c4", "claude-3", 768, now))
	store.Add("concept-2", makeSample("s5", "c5", "claude-3", 768, now))

	// Add concept 3 with 1 sample
	store.Add("concept-3", makeSample("s6", "c6", "gpt-3.5", 512, now))

	stats := store.Stats()

	if stats.ConceptCount != 3 {
		t.Errorf("expected ConceptCount 3, got %d", stats.ConceptCount)
	}
	if stats.TotalSamples != 6 {
		t.Errorf("expected TotalSamples 6, got %d", stats.TotalSamples)
	}
	if stats.HealthyConcepts != 3 {
		t.Errorf("expected HealthyConcepts 3, got %d", stats.HealthyConcepts)
	}
	if stats.ConceptsWithIssues != 0 {
		t.Errorf("expected ConceptsWithIssues 0, got %d", stats.ConceptsWithIssues)
	}
	if len(stats.Concepts) != 3 {
		t.Errorf("expected 3 concepts in map, got %d", len(stats.Concepts))
	}
}

// TestStoreStats_DimensionDistribution tests dimension distribution counting.
func TestStoreStats_DimensionDistribution(t *testing.T) {
	store := NewStore()
	now := time.Now()

	// Create concepts with different dimensions
	store.Add("dim-1024-a", makeSample("s1", "c1", "gpt-4", 1024, now))
	store.Add("dim-1024-b", makeSample("s2", "c2", "gpt-4", 1024, now))
	store.Add("dim-768", makeSample("s3", "c3", "claude-3", 768, now))
	store.Add("dim-512", makeSample("s4", "c4", "gpt-3.5", 512, now))

	stats := store.Stats()

	if stats.Dimensions[1024] != 2 {
		t.Errorf("expected Dimensions[1024] = 2, got %d", stats.Dimensions[1024])
	}
	if stats.Dimensions[768] != 1 {
		t.Errorf("expected Dimensions[768] = 1, got %d", stats.Dimensions[768])
	}
	if stats.Dimensions[512] != 1 {
		t.Errorf("expected Dimensions[512] = 1, got %d", stats.Dimensions[512])
	}
	if len(stats.Dimensions) != 3 {
		t.Errorf("expected 3 dimensions, got %d: %v", len(stats.Dimensions), stats.Dimensions)
	}
}

// TestStoreStats_ModelDistribution tests model usage counting.
func TestStoreStats_ModelDistribution(t *testing.T) {
	store := NewStore()
	now := time.Now()

	// Add samples with different models
	store.Add("concept-1", makeSample("s1", "c1", "gpt-4", 1024, now))
	store.Add("concept-1", makeSample("s2", "c2", "gpt-4", 1024, now))
	store.Add("concept-1", makeSample("s3", "c3", "claude-3", 1024, now))

	store.Add("concept-2", makeSample("s4", "c4", "gpt-4", 768, now))
	store.Add("concept-2", makeSample("s5", "c5", "gpt-3.5", 768, now))

	stats := store.Stats()

	if stats.Models["gpt-4"] != 3 {
		t.Errorf("expected Models['gpt-4'] = 3, got %d", stats.Models["gpt-4"])
	}
	if stats.Models["claude-3"] != 1 {
		t.Errorf("expected Models['claude-3'] = 1, got %d", stats.Models["claude-3"])
	}
	if stats.Models["gpt-3.5"] != 1 {
		t.Errorf("expected Models['gpt-3.5'] = 1, got %d", stats.Models["gpt-3.5"])
	}
	if len(stats.Models) != 3 {
		t.Errorf("expected 3 models, got %d: %v", len(stats.Models), stats.Models)
	}
}

// TestStoreStats_TimeRanges tests time range tracking.
func TestStoreStats_TimeRanges(t *testing.T) {
	store := NewStore()

	oldestTime := time.Now().Add(-72 * time.Hour)
	middleTime := time.Now().Add(-24 * time.Hour)
	newestTime := time.Now().Add(-1 * time.Hour)

	// Add samples with different times
	store.Add("concept-1", makeSample("s1", "c1", "gpt-4", 1024, middleTime))
	store.Add("concept-2", makeSample("s2", "c2", "gpt-4", 1024, oldestTime))
	store.Add("concept-3", makeSample("s3", "c3", "gpt-4", 1024, newestTime))

	stats := store.Stats()

	if !stats.OldestExtraction.Equal(oldestTime) {
		t.Errorf("expected OldestExtraction %v, got %v", oldestTime, stats.OldestExtraction)
	}
	if !stats.NewestExtraction.Equal(newestTime) {
		t.Errorf("expected NewestExtraction %v, got %v", newestTime, stats.NewestExtraction)
	}
}

// TestStoreStats_HealthyVsIssues tests healthy vs concepts with issues counting.
func TestStoreStats_HealthyVsIssues(t *testing.T) {
	store := NewStore()
	now := time.Now()

	// Add healthy concept (consistent dimensions)
	store.Add("healthy-1", makeSample("s1", "c1", "gpt-4", 1024, now))
	store.Add("healthy-1", makeSample("s2", "c2", "gpt-4", 1024, now))

	// Add healthy concept (consistent dimensions)
	store.Add("healthy-2", makeSample("s3", "c3", "gpt-4", 768, now))
	store.Add("healthy-2", makeSample("s4", "c4", "gpt-4", 768, now))

	// Add concept with issues (mismatched dimensions)
	store.Add("issues-1", makeSample("s5", "c5", "gpt-4", 1024, now))
	store.Add("issues-1", makeSample("s6", "c6", "gpt-4", 512, now)) // Mismatch!

	stats := store.Stats()

	if stats.HealthyConcepts != 2 {
		t.Errorf("expected HealthyConcepts 2, got %d", stats.HealthyConcepts)
	}
	if stats.ConceptsWithIssues != 1 {
		t.Errorf("expected ConceptsWithIssues 1, got %d", stats.ConceptsWithIssues)
	}
}

// TestStoreStats_EmptyModelField tests that empty model fields are handled correctly.
func TestStoreStats_EmptyModelField(t *testing.T) {
	store := NewStore()
	now := time.Now()

	store.Add("concept-1", makeSample("s1", "c1", "", 1024, now))      // Empty model
	store.Add("concept-1", makeSample("s2", "c2", "gpt-4", 1024, now)) // Has model
	store.Add("concept-1", makeSample("s3", "c3", "", 1024, now))      // Empty model

	stats := store.Stats()

	// Only the non-empty model should be counted
	if stats.Models["gpt-4"] != 1 {
		t.Errorf("expected Models['gpt-4'] = 1, got %d", stats.Models["gpt-4"])
	}
	if len(stats.Models) != 1 {
		t.Errorf("expected 1 model (empty excluded), got %d: %v", len(stats.Models), stats.Models)
	}
}

// TestStoreStats_NilHiddenState tests that nil hidden states are handled correctly.
func TestStoreStats_NilHiddenState(t *testing.T) {
	store := NewStore()
	now := time.Now()

	store.Add("concept-1", Sample{
		ID:          "s1",
		Content:     "c1",
		Model:       "gpt-4",
		ExtractedAt: now,
		HiddenState: nil,
	})
	store.Add("concept-1", makeSample("s2", "c2", "gpt-4", 1024, now))

	stats := store.Stats()

	if stats.TotalSamples != 2 {
		t.Errorf("expected TotalSamples 2, got %d", stats.TotalSamples)
	}
	// Should still count the dimension from the valid sample
	if stats.Dimensions[1024] != 1 {
		t.Errorf("expected Dimensions[1024] = 1, got %v", stats.Dimensions)
	}
}

// TestStoreStats_ZeroSamples tests concepts with zero samples (edge case).
func TestStoreStats_ZeroSamples(t *testing.T) {
	// This is a bit artificial since Add always adds a sample,
	// but we can test via Get after clearing
	store := NewStore()
	now := time.Now()

	// Add and then clear
	store.Add("concept-1", makeSample("s1", "c1", "gpt-4", 1024, now))
	store.Clear("concept-1")

	stats := store.Stats()

	if stats.ConceptCount != 0 {
		t.Errorf("expected ConceptCount 0, got %d", stats.ConceptCount)
	}
	if stats.TotalSamples != 0 {
		t.Errorf("expected TotalSamples 0, got %d", stats.TotalSamples)
	}
}

// TestStoreStats_Concurrency tests that Stats() is thread-safe.
func TestStoreStats_Concurrency(t *testing.T) {
	store := NewStore()
	now := time.Now()

	// Pre-populate with some data
	for i := 0; i < 10; i++ {
		name := "concept-" + string(rune('a'+i))
		for j := 0; j < 5; j++ {
			id := name + "-sample-" + string(rune('0'+j))
			store.Add(name, makeSample(id, "content", "gpt-4", 1024, now))
		}
	}

	// Run concurrent Stats calls
	const numGoroutines = 50
	done := make(chan struct{}, numGoroutines)
	errors := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer func() { done <- struct{}{} }()

			stats := store.Stats()

			// Verify consistency
			if stats.ConceptCount != 10 {
				errors <- nil // Can't return error from goroutine, just signal
			}
			if stats.TotalSamples != 50 {
				errors <- nil
			}
		}()
	}

	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// Check for any errors
	select {
	case <-errors:
		t.Error("concurrent Stats() returned inconsistent results")
	default:
		// All good
	}
}

// TestStoreStats_ConcurrentAddAndStats tests concurrent Add and Stats operations.
func TestStoreStats_ConcurrentAddAndStats(t *testing.T) {
	store := NewStore()
	now := time.Now()

	const numAdders = 20
	const numStatsCalls = 20
	done := make(chan struct{}, numAdders+numStatsCalls)

	// Add goroutines
	for i := 0; i < numAdders; i++ {
		go func(idx int) {
			defer func() { done <- struct{}{} }()

			name := "concept-" + string(rune('a'+idx%10))
			id := name + "-sample-" + string(rune('0'+idx%10))
			store.Add(name, makeSample(id, "content", "gpt-4", 1024, now))
		}(i)
	}

	// Stats goroutines
	for i := 0; i < numStatsCalls; i++ {
		go func() {
			defer func() { done <- struct{}{} }()

			// Just verify Stats doesn't panic
			_ = store.Stats()
		}()
	}

	// Wait for all goroutines
	for i := 0; i < numAdders+numStatsCalls; i++ {
		<-done
	}
}
