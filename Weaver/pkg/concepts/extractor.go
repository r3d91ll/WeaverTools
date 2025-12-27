package concepts

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	werrors "github.com/r3d91ll/weaver/pkg/errors"
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
		return nil, createBackendNoHiddenStateError(e.backend.Name(), string(e.backend.Type()))
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
			return result, createExtractionCancelledError(ctx, cfg.Concept, i+1, cfg.NumSamples, e.backend.Name())
		default:
		}

		sample, err := e.extractSingle(ctx, prompt, cfg)
		if err != nil {
			// Store a user-friendly error message for display
			result.Errors = append(result.Errors, formatSampleError(i+1, err))
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
		return Sample{}, createChatExtractionError(err, cfg.Concept, e.backend.Name(), e.model)
	}

	if resp.HiddenState == nil {
		return Sample{}, createNoHiddenStateReturnedError(cfg.Concept, e.backend.Name(), resp.Model)
	}

	return Sample{
		ID:          uuid.New().String(),
		Content:     resp.Content,
		HiddenState: convertHiddenState(resp.HiddenState),
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

// convertHiddenState converts backend.HiddenState to yarn.HiddenState.
func convertHiddenState(hs *backend.HiddenState) *yarn.HiddenState {
	if hs == nil {
		return nil
	}
	return &yarn.HiddenState{
		Vector: hs.Vector,
		Shape:  hs.Shape,
		Layer:  hs.Layer,
		DType:  hs.DType,
	}
}

// ExtractRandom generates random samples for baseline comparison.
func (e *Extractor) ExtractRandom(ctx context.Context, n int) (*ExtractionResult, error) {
	cfg := DefaultExtractionConfig("random", n)
	cfg.Temperature = 1.0 // Maximum variety for random
	return e.Extract(ctx, cfg)
}

// -----------------------------------------------------------------------------
// Error Helper Functions
// -----------------------------------------------------------------------------
// These functions create structured WeaverErrors for various extraction
// failure scenarios with appropriate context and suggestions.

// createBackendNoHiddenStateError creates a structured error when the backend
// doesn't support hidden state extraction.
func createBackendNoHiddenStateError(backendName, backendType string) *werrors.WeaverError {
	err := werrors.Backend(werrors.ErrConceptsNoHiddenState,
		fmt.Sprintf("backend %q does not support hidden state extraction", backendName))

	err.WithContext("backend", backendName).
		WithContext("backend_type", backendType).
		WithContext("required_capability", "hidden state extraction")

	// Add suggestions based on backend type
	if backendType == "claudecode" {
		err.WithSuggestion("Claude Code does not support hidden state extraction").
			WithSuggestion("Switch to a Loom backend agent for hidden state support").
			WithSuggestion("Use '/agents' to see available agents and their capabilities").
			WithSuggestion("Create a Loom agent with: weaver agent add --name <name> --backend loom")
	} else {
		err.WithSuggestion("Use a Loom backend for hidden state extraction").
			WithSuggestion("Loom (TheLoom) provides access to model internals").
			WithSuggestion("Configure Loom in your weaver.yaml under backends.loom").
			WithSuggestion("See Loom documentation: https://github.com/r3d91ll/TheLoom")
	}

	return err
}

// createExtractionCancelledError creates a structured error when extraction
// is cancelled due to context cancellation or timeout.
func createExtractionCancelledError(ctx context.Context, concept string, currentSample, totalSamples int, backendName string) *werrors.WeaverError {
	var code, message string

	if ctx.Err() == context.DeadlineExceeded {
		code = werrors.ErrBackendTimeout
		message = fmt.Sprintf("concept extraction timed out after %d/%d samples", currentSample-1, totalSamples)
	} else {
		code = werrors.ErrConceptsExtractionFailed
		message = fmt.Sprintf("concept extraction cancelled after %d/%d samples", currentSample-1, totalSamples)
	}

	err := werrors.Backend(code, message).
		WithContext("concept", concept).
		WithContext("completed_samples", fmt.Sprintf("%d", currentSample-1)).
		WithContext("requested_samples", fmt.Sprintf("%d", totalSamples)).
		WithContext("backend", backendName)

	if ctx.Err() == context.DeadlineExceeded {
		err.WithSuggestion("Try extracting fewer samples at a time").
			WithSuggestion("Check backend server responsiveness").
			WithSuggestion("Already extracted samples are saved - resume with '/extract' to add more")
	} else {
		err.WithSuggestion("Extraction was interrupted by user or system").
			WithSuggestion("Already extracted samples are saved in the concept store").
			WithSuggestion("Use '/extract' again to resume adding samples")
	}

	return err.WithCause(ctx.Err())
}

// createChatExtractionError creates a structured error when the chat request
// for extraction fails. It wraps and categorizes the underlying error.
func createChatExtractionError(cause error, concept, backendName, model string) *werrors.WeaverError {
	errStr := strings.ToLower(cause.Error())

	// Preserve WeaverErrors from the backend layer
	if we, ok := werrors.AsWeaverError(cause); ok {
		// Add extraction context to the existing error
		we.WithContext("concept", concept)
		if model != "" {
			we.WithContext("model", model)
		}
		return we
	}

	// Detect specific error types
	if isConnectionError(errStr) {
		return werrors.BackendWrap(cause, werrors.ErrBackendConnectionFailed,
			"cannot connect to backend for hidden state extraction").
			WithContext("concept", concept).
			WithContext("backend", backendName).
			WithContext("model", model).
			WithSuggestion("Check that the backend server is running").
			WithSuggestion("Verify the server URL in your configuration").
			WithSuggestion("For Loom: Check if TheLoom server is accessible")
	}

	if isTimeoutError(errStr) {
		return werrors.BackendWrap(cause, werrors.ErrBackendTimeout,
			"backend request timed out during extraction").
			WithContext("concept", concept).
			WithContext("backend", backendName).
			WithContext("model", model).
			WithSuggestion("The extraction request took too long").
			WithSuggestion("Try with simpler concepts or shorter prompts").
			WithSuggestion("Check backend server health and load")
	}

	if isAuthError(errStr) {
		return werrors.BackendWrap(cause, werrors.ErrBackendAuthFailed,
			"authentication failed during extraction").
			WithContext("concept", concept).
			WithContext("backend", backendName).
			WithSuggestion("Verify your backend credentials").
			WithSuggestion("For Claude Code: Run 'claude auth login'").
			WithSuggestion("For Loom: Check if auth is enabled on the server")
	}

	if isRateLimitError(errStr) {
		return werrors.BackendWrap(cause, werrors.ErrBackendAPIError,
			"rate limited during extraction").
			WithContext("concept", concept).
			WithContext("backend", backendName).
			WithSuggestion("Wait a few minutes and try again").
			WithSuggestion("Consider extracting fewer samples per session")
	}

	if isModelError(errStr) {
		return werrors.BackendWrap(cause, werrors.ErrBackendAPIError,
			"model error during extraction").
			WithContext("concept", concept).
			WithContext("backend", backendName).
			WithContext("model", model).
			WithSuggestion("Check that the model is available on the backend").
			WithSuggestion("Verify model name in your agent configuration").
			WithSuggestion("For Loom: Check the model is loaded with 'curl <url>/v1/models'")
	}

	// Generic extraction failure
	return werrors.BackendWrap(cause, werrors.ErrConceptsExtractionFailed,
		"failed to extract concept sample").
		WithContext("concept", concept).
		WithContext("backend", backendName).
		WithContext("model", model).
		WithSuggestion("Check backend logs for more details").
		WithSuggestion("Try the extraction again").
		WithSuggestion("Use '/agents' to verify agent configuration")
}

// createNoHiddenStateReturnedError creates a structured error when the backend
// doesn't return hidden state despite being requested.
func createNoHiddenStateReturnedError(concept, backendName, model string) *werrors.WeaverError {
	return werrors.Backend(werrors.ErrConceptsNoHiddenState,
		"backend did not return hidden state in response").
		WithContext("concept", concept).
		WithContext("backend", backendName).
		WithContext("model", model).
		WithContext("issue", "hidden_state field was null in response").
		WithSuggestion("Ensure the model supports hidden state extraction").
		WithSuggestion("For Loom: Check 'return_hidden_states' is enabled in the request").
		WithSuggestion("Some model configurations may disable hidden state output").
		WithSuggestion("Check backend server logs for errors during extraction").
		WithSuggestion("Verify the Loom server version supports hidden state extraction")
}

// formatSampleError formats an extraction error for display in the result.
// This provides a user-friendly string representation of the error.
func formatSampleError(sampleNum int, err error) string {
	if we, ok := werrors.AsWeaverError(err); ok {
		// Use the WeaverError message which is more descriptive
		return fmt.Sprintf("sample %d: %s", sampleNum, we.Message)
	}
	return fmt.Sprintf("sample %d: %v", sampleNum, err)
}

// -----------------------------------------------------------------------------
// Error Detection Helpers
// -----------------------------------------------------------------------------

// isConnectionError checks if the error indicates a connection problem.
func isConnectionError(errStr string) bool {
	patterns := []string{
		"connection refused",
		"no such host",
		"connection reset",
		"network unreachable",
		"no route to host",
		"dial tcp",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isTimeoutError checks if the error indicates a timeout.
func isTimeoutError(errStr string) bool {
	patterns := []string{
		"timeout",
		"deadline exceeded",
		"timed out",
		"context deadline",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isAuthError checks if the error indicates an authentication failure.
func isAuthError(errStr string) bool {
	patterns := []string{
		"unauthorized",
		"401",
		"authentication",
		"auth failed",
		"invalid api key",
		"credentials",
		"forbidden",
		"403",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isRateLimitError checks if the error indicates rate limiting.
func isRateLimitError(errStr string) bool {
	patterns := []string{
		"rate limit",
		"429",
		"too many requests",
		"quota exceeded",
		"throttled",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isModelError checks if the error is related to model availability.
func isModelError(errStr string) bool {
	patterns := []string{
		"model not found",
		"unknown model",
		"model does not exist",
		"model unavailable",
		"model error",
		"no model loaded",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}
