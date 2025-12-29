package concepts

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/r3d91ll/weaver/pkg/backend"
	yarn "github.com/r3d91ll/yarn"
)

// -----------------------------------------------------------------------------
// Mock Backend for Testing
// -----------------------------------------------------------------------------

// mockExtractorBackend is a mock backend implementation for testing the extractor.
type mockExtractorBackend struct {
	name            string
	supportsHidden  bool
	chatFunc        func(ctx context.Context, req backend.ChatRequest) (*backend.ChatResponse, error)
	chatDelay       time.Duration
	failOnSample    int  // Sample index (0-based) to fail on; -1 means never fail
	failedSamples   int  // Counter for tracking failed samples
	mu              sync.Mutex
}

func (m *mockExtractorBackend) Name() string { return m.name }
func (m *mockExtractorBackend) Type() backend.Type { return backend.TypeLoom }
func (m *mockExtractorBackend) IsAvailable(ctx context.Context) bool { return true }
func (m *mockExtractorBackend) Capabilities() backend.Capabilities {
	return backend.Capabilities{
		ContextLimit:   4096,
		SupportsHidden: m.supportsHidden,
		MaxTokens:      2048,
	}
}

func (m *mockExtractorBackend) Chat(ctx context.Context, req backend.ChatRequest) (*backend.ChatResponse, error) {
	// Apply delay if configured
	if m.chatDelay > 0 {
		time.Sleep(m.chatDelay)
	}

	// Track sample number for failure simulation
	m.mu.Lock()
	currentSample := m.failedSamples
	m.failedSamples++
	m.mu.Unlock()

	// Check if this sample should fail
	if m.failOnSample >= 0 && currentSample == m.failOnSample {
		return nil, errors.New("simulated extraction error")
	}

	// Use custom chat function if provided
	if m.chatFunc != nil {
		return m.chatFunc(ctx, req)
	}

	// Default: return a successful response with hidden state
	return &backend.ChatResponse{
		Content: "Sample content for testing",
		Model:   "test-model",
		HiddenState: &yarn.HiddenState{
			Vector: []float32{0.1, 0.2, 0.3, 0.4, 0.5},
			Shape:  []int{1, 1, 5},
			Layer:  0,
			DType:  "float32",
		},
	}, nil
}

func (m *mockExtractorBackend) ChatStream(ctx context.Context, req backend.ChatRequest) (<-chan backend.StreamChunk, <-chan error) {
	ch := make(chan backend.StreamChunk)
	errCh := make(chan error)
	go func() {
		ch <- backend.StreamChunk{Content: "mock", Done: true}
		close(ch)
		close(errCh)
	}()
	return ch, errCh
}

// reset resets the failed samples counter for reuse between tests
func (m *mockExtractorBackend) reset() {
	m.mu.Lock()
	m.failedSamples = 0
	m.mu.Unlock()
}

// -----------------------------------------------------------------------------
// Progress Callback Tests
// -----------------------------------------------------------------------------

// TestProgressCallbackCalledForEachSample verifies callback is called for each sample.
func TestProgressCallbackCalledForEachSample(t *testing.T) {
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
	}
	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	numSamples := 5
	callCount := 0
	var mu sync.Mutex

	cfg := ExtractionConfig{
		Concept:    "test-concept",
		NumSamples: numSamples,
		OnProgress: func(current, total int, elapsed time.Duration) {
			mu.Lock()
			callCount++
			mu.Unlock()
		},
	}

	ctx := context.Background()
	_, err := extractor.Extract(ctx, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	if callCount != numSamples {
		t.Errorf("expected callback to be called %d times, got %d", numSamples, callCount)
	}
}

// TestProgressCallbackReceivesCorrectValues verifies callback receives correct current/total values.
func TestProgressCallbackReceivesCorrectValues(t *testing.T) {
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
	}
	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	numSamples := 5

	// Track all callback invocations
	type progressCall struct {
		current int
		total   int
		elapsed time.Duration
	}
	var calls []progressCall
	var mu sync.Mutex

	cfg := ExtractionConfig{
		Concept:    "test-concept",
		NumSamples: numSamples,
		OnProgress: func(current, total int, elapsed time.Duration) {
			mu.Lock()
			calls = append(calls, progressCall{current: current, total: total, elapsed: elapsed})
			mu.Unlock()
		},
	}

	ctx := context.Background()
	_, err := extractor.Extract(ctx, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	if len(calls) != numSamples {
		t.Fatalf("expected %d calls, got %d", numSamples, len(calls))
	}

	// Verify each call has correct values
	for i, call := range calls {
		expectedCurrent := i // 0-indexed: 0, 1, 2, 3, 4
		if call.current != expectedCurrent {
			t.Errorf("call %d: expected current=%d, got %d", i, expectedCurrent, call.current)
		}

		if call.total != numSamples {
			t.Errorf("call %d: expected total=%d, got %d", i, numSamples, call.total)
		}

		// Elapsed time should be non-negative
		if call.elapsed < 0 {
			t.Errorf("call %d: elapsed time should be non-negative, got %v", i, call.elapsed)
		}
	}

	// Verify elapsed times are increasing (or at least not decreasing)
	for i := 1; i < len(calls); i++ {
		if calls[i].elapsed < calls[i-1].elapsed {
			t.Errorf("elapsed time should not decrease: call %d (%v) < call %d (%v)",
				i, calls[i].elapsed, i-1, calls[i-1].elapsed)
		}
	}
}

// TestProgressCallbackNotCalledIfNil verifies callback is not called if nil.
func TestProgressCallbackNotCalledIfNil(t *testing.T) {
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
	}
	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	// Create config without OnProgress callback (should be nil by default)
	cfg := DefaultExtractionConfig("test-concept", 5)

	// Verify OnProgress is nil
	if cfg.OnProgress != nil {
		t.Error("DefaultExtractionConfig should leave OnProgress nil")
	}

	ctx := context.Background()
	result, err := extractor.Extract(ctx, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Extraction should complete successfully without panic
	if result.SamplesAdded != 5 {
		t.Errorf("expected 5 samples added, got %d", result.SamplesAdded)
	}
}

// TestProgressCallbackCalledOnExtractionErrors verifies callback is called even when extraction errors occur.
func TestProgressCallbackCalledOnExtractionErrors(t *testing.T) {
	// Create mock backend that fails on sample index 2
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
		failOnSample:   2, // Fail on the 3rd sample (0-indexed)
	}
	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	numSamples := 5
	var callCurrentValues []int
	var mu sync.Mutex

	cfg := ExtractionConfig{
		Concept:    "test-concept",
		NumSamples: numSamples,
		OnProgress: func(current, total int, elapsed time.Duration) {
			mu.Lock()
			callCurrentValues = append(callCurrentValues, current)
			mu.Unlock()
		},
	}

	ctx := context.Background()
	result, err := extractor.Extract(ctx, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	// Callback should still be called for all samples (including the one that errors)
	if len(callCurrentValues) != numSamples {
		t.Errorf("expected callback to be called %d times (including errored sample), got %d",
			numSamples, len(callCurrentValues))
	}

	// Verify we got calls for 0, 1, 2, 3, 4
	expectedCurrent := []int{0, 1, 2, 3, 4}
	for i, expected := range expectedCurrent {
		if i >= len(callCurrentValues) {
			t.Errorf("missing call for sample %d", expected)
			continue
		}
		if callCurrentValues[i] != expected {
			t.Errorf("call %d: expected current=%d, got %d", i, expected, callCurrentValues[i])
		}
	}

	// Verify that we have one error in the result
	if len(result.Errors) != 1 {
		t.Errorf("expected 1 error in result, got %d", len(result.Errors))
	}

	// Verify 4 samples were successfully added (all except the failed one)
	if result.SamplesAdded != 4 {
		t.Errorf("expected 4 samples added (one failed), got %d", result.SamplesAdded)
	}
}

// TestProgressCallbackWithMultipleErrors verifies callback behavior with multiple errors.
func TestProgressCallbackWithMultipleErrors(t *testing.T) {
	// Create mock backend that fails on every other sample
	failingSamples := map[int]bool{0: true, 2: true, 4: true}
	callCount := 0
	var callMu sync.Mutex

	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
		chatFunc: func(ctx context.Context, req backend.ChatRequest) (*backend.ChatResponse, error) {
			callMu.Lock()
			currentCall := callCount
			callCount++
			callMu.Unlock()

			if failingSamples[currentCall] {
				return nil, errors.New("simulated extraction error")
			}
			return &backend.ChatResponse{
				Content: "Sample content",
				Model:   "test-model",
				HiddenState: &yarn.HiddenState{
					Vector: []float32{0.1, 0.2, 0.3, 0.4, 0.5},
					Shape:  []int{1, 1, 5},
					Layer:  0,
					DType:  "float32",
				},
			}, nil
		},
	}

	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	numSamples := 5
	progressCallCount := 0
	var mu sync.Mutex

	cfg := ExtractionConfig{
		Concept:    "test-concept",
		NumSamples: numSamples,
		OnProgress: func(current, total int, elapsed time.Duration) {
			mu.Lock()
			progressCallCount++
			mu.Unlock()
		},
	}

	ctx := context.Background()
	result, err := extractor.Extract(ctx, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	// Callback should be called for all samples
	if progressCallCount != numSamples {
		t.Errorf("expected callback to be called %d times, got %d", numSamples, progressCallCount)
	}

	// Should have 3 errors (samples 0, 2, 4)
	if len(result.Errors) != 3 {
		t.Errorf("expected 3 errors, got %d", len(result.Errors))
	}

	// Should have 2 successful samples (samples 1, 3)
	if result.SamplesAdded != 2 {
		t.Errorf("expected 2 samples added, got %d", result.SamplesAdded)
	}
}

// TestProgressCallbackWithContextCancellation verifies callback behavior when context is cancelled.
func TestProgressCallbackWithContextCancellation(t *testing.T) {
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
		chatDelay:      50 * time.Millisecond, // Add delay to allow cancellation
	}
	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	numSamples := 10
	var callCurrentValues []int
	var mu sync.Mutex

	cfg := ExtractionConfig{
		Concept:    "test-concept",
		NumSamples: numSamples,
		OnProgress: func(current, total int, elapsed time.Duration) {
			mu.Lock()
			callCurrentValues = append(callCurrentValues, current)
			mu.Unlock()
		},
	}

	// Create a context that will be cancelled after a short time
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err := extractor.Extract(ctx, cfg)

	// Should return an error due to cancellation
	if err == nil {
		t.Log("Note: Extraction completed before cancellation (timing-dependent)")
	}

	mu.Lock()
	defer mu.Unlock()

	// At least one callback should have been called before cancellation
	if len(callCurrentValues) == 0 {
		t.Error("expected at least one callback before cancellation")
	}

	// Callback should have been called for samples before cancellation
	t.Logf("Progress callback was called %d times before cancellation", len(callCurrentValues))
}

// TestProgressCallbackConcurrencySafety verifies callback is called safely in the extraction loop.
func TestProgressCallbackConcurrencySafety(t *testing.T) {
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
	}
	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	numSamples := 50
	var callCount int64
	var mu sync.Mutex

	// Use a callback that does some work to simulate real-world usage
	cfg := ExtractionConfig{
		Concept:    "test-concept",
		NumSamples: numSamples,
		OnProgress: func(current, total int, elapsed time.Duration) {
			mu.Lock()
			callCount++
			// Simulate some work in the callback
			_ = current * total
			mu.Unlock()
		},
	}

	ctx := context.Background()
	result, err := extractor.Extract(ctx, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	if callCount != int64(numSamples) {
		t.Errorf("expected %d callbacks, got %d", numSamples, callCount)
	}

	if result.SamplesAdded != numSamples {
		t.Errorf("expected %d samples added, got %d", numSamples, result.SamplesAdded)
	}
}

// TestExtractorWithoutProgressCallback verifies extraction works without callback.
func TestExtractorWithoutProgressCallback(t *testing.T) {
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
	}
	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	cfg := ExtractionConfig{
		Concept:    "test-concept",
		NumSamples: 5,
		// OnProgress intentionally left nil
	}

	ctx := context.Background()
	result, err := extractor.Extract(ctx, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.SamplesAdded != 5 {
		t.Errorf("expected 5 samples added, got %d", result.SamplesAdded)
	}
}

// TestDefaultExtractionConfigLeavesCallbackNil verifies default config has nil callback.
func TestDefaultExtractionConfigLeavesCallbackNil(t *testing.T) {
	cfg := DefaultExtractionConfig("test-concept", 10)

	if cfg.OnProgress != nil {
		t.Error("DefaultExtractionConfig should leave OnProgress nil")
	}

	if cfg.Concept != "test-concept" {
		t.Errorf("expected concept 'test-concept', got %q", cfg.Concept)
	}

	if cfg.NumSamples != 10 {
		t.Errorf("expected NumSamples 10, got %d", cfg.NumSamples)
	}
}

// -----------------------------------------------------------------------------
// Extractor Basic Tests
// -----------------------------------------------------------------------------

// TestExtractorReturnsErrorForNonHiddenBackend verifies error when backend doesn't support hidden states.
func TestExtractorReturnsErrorForNonHiddenBackend(t *testing.T) {
	mockBackend := &mockExtractorBackend{
		name:           "no-hidden-backend",
		supportsHidden: false, // Does not support hidden states
	}
	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	cfg := DefaultExtractionConfig("test-concept", 5)
	ctx := context.Background()

	_, err := extractor.Extract(ctx, cfg)
	if err == nil {
		t.Error("expected error for backend that doesn't support hidden states")
	}
}

// TestExtractorWithModel verifies WithModel sets the model correctly.
func TestExtractorWithModel(t *testing.T) {
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
		chatFunc: func(ctx context.Context, req backend.ChatRequest) (*backend.ChatResponse, error) {
			// Verify the model is set correctly
			if req.Model != "custom-model" {
				t.Errorf("expected model 'custom-model', got %q", req.Model)
			}
			return &backend.ChatResponse{
				Content: "Sample content",
				Model:   req.Model,
				HiddenState: &yarn.HiddenState{
					Vector: []float32{0.1, 0.2, 0.3},
					Shape:  []int{1, 1, 3},
					Layer:  0,
					DType:  "float32",
				},
			}, nil
		},
	}

	store := NewStore()
	extractor := NewExtractor(mockBackend, store).WithModel("custom-model")

	cfg := DefaultExtractionConfig("test-concept", 1)
	ctx := context.Background()

	_, err := extractor.Extract(ctx, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestExtractionResultContainsCorrectConcept verifies result has correct concept name.
func TestExtractionResultContainsCorrectConcept(t *testing.T) {
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
	}
	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	cfg := DefaultExtractionConfig("my-test-concept", 3)
	ctx := context.Background()

	result, err := extractor.Extract(ctx, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Concept != "my-test-concept" {
		t.Errorf("expected concept 'my-test-concept', got %q", result.Concept)
	}

	if result.SamplesAdded != 3 {
		t.Errorf("expected 3 samples added, got %d", result.SamplesAdded)
	}
}

// TestExtractRandomUsesMaxTemperature verifies ExtractRandom sets temperature to 1.0.
func TestExtractRandomUsesMaxTemperature(t *testing.T) {
	var capturedTemp float64
	mockBackend := &mockExtractorBackend{
		name:           "test-backend",
		supportsHidden: true,
		chatFunc: func(ctx context.Context, req backend.ChatRequest) (*backend.ChatResponse, error) {
			capturedTemp = req.Temperature
			return &backend.ChatResponse{
				Content: "Random content",
				Model:   "test-model",
				HiddenState: &yarn.HiddenState{
					Vector: []float32{0.1, 0.2, 0.3},
					Shape:  []int{1, 1, 3},
					Layer:  0,
					DType:  "float32",
				},
			}, nil
		},
	}

	store := NewStore()
	extractor := NewExtractor(mockBackend, store)

	ctx := context.Background()
	_, err := extractor.ExtractRandom(ctx, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedTemp != 1.0 {
		t.Errorf("expected temperature 1.0 for random extraction, got %f", capturedTemp)
	}
}
