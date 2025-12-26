package concepts

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/r3d91ll/weaver/pkg/backend"
	"github.com/r3d91ll/yarn"
)

// Extractor generates concept examples and extracts hidden states.
type Extractor struct {
	backend backend.Backend
	store   *Store
	model   string // Model to use for extraction
}

// NewExtractor creates a new concept extractor.
// The model parameter specifies which model to use; if empty, uses the backend's default.
func NewExtractor(b backend.Backend, store *Store) *Extractor {
	return &Extractor{
		backend: b,
		store:   store,
		model:   "", // Use backend default
	}
}

// WithModel sets the model to use for extraction.
func (e *Extractor) WithModel(model string) *Extractor {
	e.model = model
	return e
}

// ExtractionConfig controls the extraction process.
type ExtractionConfig struct {
	Concept     string  // The concept to extract examples for
	NumSamples  int     // Number of samples to extract
	Temperature float64 // Temperature for generation (higher = more variety)
	MaxTokens   int     // Max tokens per response
}

// DefaultExtractionConfig returns sensible defaults.
func DefaultExtractionConfig(concept string, n int) ExtractionConfig {
	return ExtractionConfig{
		Concept:     concept,
		NumSamples:  n,
		Temperature: 0.8, // Higher temp for variety
		MaxTokens:   150, // Short examples
	}
}

// ExtractionResult summarizes an extraction run.
type ExtractionResult struct {
	Concept       string
	SamplesAdded  int
	TotalSamples  int
	Dimension     int
	Errors        []string
	DurationMs    float64
}

// Extract generates concept examples and stores their hidden states.
func (e *Extractor) Extract(ctx context.Context, cfg ExtractionConfig) (*ExtractionResult, error) {
	if !e.backend.Capabilities().SupportsHidden {
		return nil, fmt.Errorf("backend %q does not support hidden state extraction", e.backend.Name())
	}

	start := time.Now()
	result := &ExtractionResult{
		Concept: cfg.Concept,
		Errors:  []string{},
	}

	prompt := e.buildPrompt(cfg.Concept)

	for i := 0; i < cfg.NumSamples; i++ {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}

		sample, err := e.extractSingle(ctx, prompt, cfg)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("sample %d: %v", i+1, err))
			continue
		}

		e.store.Add(cfg.Concept, sample)
		result.SamplesAdded++
	}

	// Get final stats
	if concept, ok := e.store.Get(cfg.Concept); ok {
		result.TotalSamples = len(concept.Samples)
		result.Dimension = concept.Dimension()
	}

	result.DurationMs = float64(time.Since(start).Milliseconds())
	return result, nil
}

// extractSingle extracts a single concept example.
func (e *Extractor) extractSingle(ctx context.Context, prompt string, cfg ExtractionConfig) (Sample, error) {
	req := backend.ChatRequest{
		Model: e.model, // Use configured model (empty uses backend default)
		Messages: []backend.ChatMessage{
			{Role: "user", Content: prompt},
		},
		Temperature:        cfg.Temperature,
		MaxTokens:          cfg.MaxTokens,
		ReturnHiddenStates: true,
	}

	resp, err := e.backend.Chat(ctx, req)
	if err != nil {
		return Sample{}, err
	}

	if resp.HiddenState == nil {
		return Sample{}, fmt.Errorf("no hidden state returned")
	}

	return Sample{
		ID:          uuid.New().String(),
		Content:     resp.Content,
		HiddenState: resp.HiddenState,
		ExtractedAt: time.Now(),
		Model:       resp.Model,
	}, nil
}

// buildPrompt creates the extraction prompt for a concept.
func (e *Extractor) buildPrompt(concept string) string {
	if concept == "random" {
		return `Generate a random sentence on any topic. Be varied and creative. Just output the sentence, no explanation.`
	}

	return fmt.Sprintf(
		`Provide a brief example (1-2 sentences) illustrating the concept of "%s" as it might appear in classical literature. Be creative and varied. Output only the example, no explanation or preamble.`,
		concept,
	)
}

// ExtractRandom generates random samples for baseline comparison.
func (e *Extractor) ExtractRandom(ctx context.Context, n int) (*ExtractionResult, error) {
	cfg := DefaultExtractionConfig("random", n)
	cfg.Temperature = 1.0 // Maximum variety for random
	return e.Extract(ctx, cfg)
}
