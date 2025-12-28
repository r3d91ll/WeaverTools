package backend

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
	yarn "github.com/r3d91ll/yarn"
)

// Loom connects to The Loom server for inference with hidden state extraction.
type Loom struct {
	name       string
	baseURL    string
	model      string
	httpClient *http.Client
	mu         sync.RWMutex
}

// LoomConfig holds configuration for The Loom backend.
type LoomConfig struct {
	Name    string        `yaml:"name"`
	URL     string        `yaml:"url"`
	Model   string        `yaml:"model"`
	Timeout time.Duration `yaml:"timeout"`
}

// NewLoom creates a new Loom backend.
func NewLoom(cfg LoomConfig) *Loom {
	name := cfg.Name
	if name == "" {
		name = "loom"
	}
	url := cfg.URL
	if url == "" {
		url = "http://localhost:8080"
	}
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	return &Loom{
		name:    name,
		baseURL: url,
		model:   cfg.Model,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (l *Loom) Name() string { return l.name }
func (l *Loom) Type() Type   { return TypeLoom }

func (l *Loom) IsAvailable(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, "GET", l.baseURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := l.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (l *Loom) Capabilities() Capabilities {
	return Capabilities{
		ContextLimit:      32768,
		SupportsTools:     false,
		SupportsStreaming: true,
		SupportsHidden:    true,
		MaxTokens:         2048,
	}
}

type loomRequest struct {
	Model              string        `json:"model"`
	Messages           []ChatMessage `json:"messages"`
	MaxTokens          int           `json:"max_tokens,omitempty"`
	Temperature        float64       `json:"temperature,omitempty"`
	ReturnHiddenStates bool          `json:"return_hidden_states,omitempty"`
	HiddenStateLayers  interface{}   `json:"hidden_state_layers,omitempty"` // Layer selection: "all", int, or []int
	Device             string        `json:"device,omitempty"`              // GPU: "auto", "cuda:0", "cuda:1"
}

// loomStreamingRequest extends loomRequest with streaming flag.
type loomStreamingRequest struct {
	Model              string        `json:"model"`
	Messages           []ChatMessage `json:"messages"`
	MaxTokens          int           `json:"max_tokens,omitempty"`
	Temperature        float64       `json:"temperature,omitempty"`
	ReturnHiddenStates bool          `json:"return_hidden_states,omitempty"`
	HiddenStateLayers  interface{}   `json:"hidden_state_layers,omitempty"` // Layer selection: "all", int, or []int
	Device             string        `json:"device,omitempty"`
	Stream             bool          `json:"stream"`
}

type loomResponse struct {
	Text        string `json:"text"`
	Usage       struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	HiddenState *struct {
		Final []float32 `json:"final"`
		Shape []int     `json:"shape"`
		Layer int       `json:"layer"`
		DType string    `json:"dtype"`
	} `json:"hidden_state,omitempty"`
	// HiddenStates contains per-layer hidden states when multiple layers are requested.
	// Key is layer index (as string due to JSON), value contains the hidden state for that layer.
	HiddenStates map[string]*loomLayerHiddenState `json:"hidden_states,omitempty"`
	Metadata     map[string]any                   `json:"metadata,omitempty"`
}

// loomLayerHiddenState represents a hidden state from a specific layer.
type loomLayerHiddenState struct {
	Vector []float32 `json:"vector"` // Hidden state vector for this layer
	Shape  []int     `json:"shape"`  // Tensor shape
	Layer  int       `json:"layer"`  // Layer index
	DType  string    `json:"dtype"`  // Data type
	DEff   float64   `json:"d_eff"`  // Effective dimensionality (if computed)
}

// sseEvent represents a parsed Server-Sent Event from The Loom server.
type sseEvent struct {
	Event string // Event type (e.g., "content_block_delta", "message_delta", "error")
	Data  string // JSON data payload
}

// loomSSEEvent represents the common structure of SSE event data from The Loom.
// This is used to parse the "type" field before further parsing.
type loomSSEEvent struct {
	Type string `json:"type"`
}

// loomContentDelta represents a content_block_delta event payload.
// These events contain individual tokens during streaming.
type loomContentDelta struct {
	Type  string `json:"type"`
	Delta struct {
		Type string `json:"type"` // "text_delta"
		Text string `json:"text"`
	} `json:"delta"`
}

// loomMessageDelta represents a message_delta event payload.
// This is the final event in a stream, containing completion info.
type loomMessageDelta struct {
	Type  string `json:"type"`
	Delta struct {
		StopReason string `json:"stop_reason"` // "end_turn", "max_tokens", etc.
	} `json:"delta"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	// Metadata contains generation statistics (optional).
	Metadata *loomMessageMetadata `json:"metadata,omitempty"`
	// HiddenState contains the final hidden state vector if requested (optional).
	HiddenState *loomHiddenState `json:"hidden_state,omitempty"`
}

// loomMessageMetadata contains generation statistics from The Loom server.
type loomMessageMetadata struct {
	Model           string  `json:"model"`
	LatencyMS       float64 `json:"latency_ms"`
	TokensPerSecond float64 `json:"tokens_per_second"`
}

// loomHiddenState represents the hidden state returned in streaming responses.
// This is the "boundary object" - the geometric representation of meaning.
type loomHiddenState struct {
	Final []float32 `json:"final"` // Final layer hidden state vector
	Shape []int     `json:"shape"` // Tensor shape [batch, hidden_dim]
	Layer int       `json:"layer"` // Layer index (-1 = last)
	DType string    `json:"dtype"` // Data type (e.g., "float32")
}

// loomErrorEvent represents an error event payload.
type loomErrorEvent struct {
	Type  string `json:"type"`
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}

// parseSSE parses Server-Sent Events from a response body using bufio.Scanner.
// It follows the SSE spec: event: and data: lines, empty lines separate events,
// lines starting with : are comments.
//
// The returned channel emits parsed SSE events as they arrive. The channel is
// closed when the stream ends or when the context is canceled. Any parse errors
// are logged but do not stop parsing.
func parseSSE(ctx context.Context, body io.Reader) <-chan sseEvent {
	events := make(chan sseEvent, 100)

	go func() {
		defer close(events)

		scanner := bufio.NewScanner(body)
		var currentEvent string
		var dataLines []string

		for scanner.Scan() {
			// Check for context cancellation
			select {
			case <-ctx.Done():
				return
			default:
			}

			line := scanner.Text()

			// Empty line signals end of event
			if line == "" {
				if len(dataLines) > 0 {
					// Combine multi-line data fields with newlines
					data := strings.Join(dataLines, "\n")
					events <- sseEvent{
						Event: currentEvent,
						Data:  data,
					}
				}
				// Reset for next event
				currentEvent = ""
				dataLines = nil
				continue
			}

			// Skip comment lines (start with :)
			if strings.HasPrefix(line, ":") {
				continue
			}

			// Parse field: value format
			if strings.HasPrefix(line, "event:") {
				currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			} else if strings.HasPrefix(line, "data:") {
				data := strings.TrimPrefix(line, "data:")
				// Only trim the single leading space after "data:" per SSE spec
				if len(data) > 0 && data[0] == ' ' {
					data = data[1:]
				}
				dataLines = append(dataLines, data)
			}
			// Other fields like id: and retry: are ignored
		}

		// Handle any remaining event at end of stream
		if len(dataLines) > 0 {
			data := strings.Join(dataLines, "\n")
			events <- sseEvent{
				Event: currentEvent,
				Data:  data,
			}
		}
	}()

	return events
}

// buildStreamingRequest creates an HTTP request for streaming chat completions.
// It sets stream=true in the JSON body and Accept: text/event-stream header
// to enable Server-Sent Events streaming from The Loom server.
func (l *Loom) buildStreamingRequest(ctx context.Context, req ChatRequest) (*http.Request, error) {
	model := req.Model
	if model == "" {
		l.mu.RLock()
		model = l.model
		l.mu.RUnlock()
	}

	streamReq := loomStreamingRequest{
		Model:              model,
		Messages:           req.Messages,
		MaxTokens:          req.MaxTokens,
		Temperature:        req.Temperature,
		ReturnHiddenStates: req.ReturnHiddenStates,
		HiddenStateLayers:  req.HiddenStateLayers,
		Device:             req.Device,
		Stream:             true,
	}
	if streamReq.MaxTokens == 0 {
		streamReq.MaxTokens = 1024
	}
	if streamReq.Temperature == 0 {
		streamReq.Temperature = 0.7
	}

	body, err := json.Marshal(streamReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal streaming request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", l.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create streaming request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	return httpReq, nil
}

func (l *Loom) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	start := time.Now()

	model := req.Model
	if model == "" {
		l.mu.RLock()
		model = l.model
		l.mu.RUnlock()
	}

	loomReq := loomRequest{
		Model:              model,
		Messages:           req.Messages,
		MaxTokens:          req.MaxTokens,
		Temperature:        req.Temperature,
		ReturnHiddenStates: req.ReturnHiddenStates,
		HiddenStateLayers:  req.HiddenStateLayers,
		Device:             req.Device,
	}
	if loomReq.MaxTokens == 0 {
		loomReq.MaxTokens = 1024 // Increased from 256 to avoid truncated responses
	}
	if loomReq.Temperature == 0 {
		loomReq.Temperature = 0.7
	}

	body, err := json.Marshal(loomReq)
	if err != nil {
		return nil, createLoomMarshalError(err, l.baseURL)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", l.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, createLoomRequestError(err, l.baseURL)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := l.httpClient.Do(httpReq)
	if err != nil {
		return nil, createLoomConnectionError(ctx, err, l.baseURL)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, createLoomReadError(err, l.baseURL)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, createLoomHTTPError(resp.StatusCode, string(respBody), model, l.baseURL)
	}

	var loomResp loomResponse
	if err := json.Unmarshal(respBody, &loomResp); err != nil {
		return nil, createLoomUnmarshalError(err, l.baseURL, string(respBody))
	}

	result := &ChatResponse{
		Content:      loomResp.Text,
		Model:        model,
		FinishReason: "stop",
		LatencyMS:    float64(time.Since(start).Milliseconds()),
		Usage: TokenUsage{
			PromptTokens:     loomResp.Usage.PromptTokens,
			CompletionTokens: loomResp.Usage.CompletionTokens,
			TotalTokens:      loomResp.Usage.TotalTokens,
		},
		Metadata: loomResp.Metadata,
	}

	// Handle single-layer hidden state (legacy)
	if loomResp.HiddenState != nil {
		result.HiddenState = &yarn.HiddenState{
			Vector: loomResp.HiddenState.Final,
			Shape:  loomResp.HiddenState.Shape,
			Layer:  loomResp.HiddenState.Layer,
			DType:  loomResp.HiddenState.DType,
		}
	}

	// Handle multi-layer hidden states
	if len(loomResp.HiddenStates) > 0 {
		result.HiddenStates = make(map[int]*yarn.HiddenState, len(loomResp.HiddenStates))
		for layerKey, layerState := range loomResp.HiddenStates {
			if layerState != nil {
				layerIdx := layerState.Layer // Use the layer index from the state itself
				result.HiddenStates[layerIdx] = &yarn.HiddenState{
					Vector: layerState.Vector,
					Shape:  layerState.Shape,
					Layer:  layerState.Layer,
					DType:  layerState.DType,
				}
				// Store D_eff in metadata if computed
				if layerState.DEff > 0 {
					if result.Metadata == nil {
						result.Metadata = make(map[string]any)
					}
					if _, ok := result.Metadata["layer_d_eff"]; !ok {
						result.Metadata["layer_d_eff"] = make(map[string]float64)
					}
					if deffMap, ok := result.Metadata["layer_d_eff"].(map[string]float64); ok {
						deffMap[layerKey] = layerState.DEff
					}
				}
			}
		}
	}

	return result, nil
}

func (l *Loom) ChatStream(ctx context.Context, req ChatRequest) (<-chan StreamChunk, <-chan error) {
	chunks := make(chan StreamChunk, 100)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		// Build streaming request with stream=true and Accept: text/event-stream
		httpReq, err := l.buildStreamingRequest(ctx, req)
		if err != nil {
			errs <- fmt.Errorf("failed to build streaming request: %w", err)
			return
		}

		// Make the HTTP request
		resp, err := l.httpClient.Do(httpReq)
		if err != nil {
			errs <- fmt.Errorf("streaming request failed: %w", err)
			return
		}
		defer resp.Body.Close()

		// Check for HTTP errors
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			errs <- fmt.Errorf("loom streaming returned status %d: %s", resp.StatusCode, string(body))
			return
		}

		// Parse SSE events from response body
		events := parseSSE(ctx, resp.Body)

		// Process events and convert to StreamChunks
		for event := range events {
			// Check for context cancellation
			select {
			case <-ctx.Done():
				return
			default:
			}

			// Skip events without data
			if event.Data == "" {
				continue
			}

			// First, determine event type from the event name or parse JSON
			eventType := event.Event
			if eventType == "" {
				// Try to get type from JSON data
				var baseEvent loomSSEEvent
				if err := json.Unmarshal([]byte(event.Data), &baseEvent); err == nil {
					eventType = baseEvent.Type
				}
			}

			switch eventType {
			case "content_block_delta":
				// Parse content delta event
				var delta loomContentDelta
				if err := json.Unmarshal([]byte(event.Data), &delta); err != nil {
					// Skip malformed events rather than failing
					continue
				}
				if delta.Delta.Text != "" {
					chunks <- StreamChunk{Content: delta.Delta.Text}
				}

			case "message_delta":
				// Parse message delta (completion) event
				var msgDelta loomMessageDelta
				if err := json.Unmarshal([]byte(event.Data), &msgDelta); err != nil {
					// Still mark as done even if we can't parse details
					chunks <- StreamChunk{Done: true, FinishReason: "stop"}
					continue
				}
				// Map stop_reason to finish_reason
				finishReason := msgDelta.Delta.StopReason
				if finishReason == "end_turn" {
					finishReason = "stop"
				}
				chunks <- StreamChunk{Done: true, FinishReason: finishReason}

			case "error":
				// Parse error event
				var errEvent loomErrorEvent
				if err := json.Unmarshal([]byte(event.Data), &errEvent); err != nil {
					errs <- fmt.Errorf("loom streaming error: %s", event.Data)
				} else {
					errs <- fmt.Errorf("loom streaming error: %s", errEvent.Error.Message)
				}
				return
			}
		}
	}()

	return chunks, errs
}

// SetModel updates the default model.
func (l *Loom) SetModel(model string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.model = model
}

// Model returns the current model.
func (l *Loom) Model() string {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.model
}

// -----------------------------------------------------------------------------
// Layer Analysis Helper Functions
// -----------------------------------------------------------------------------
// These functions provide utilities for working with layer-by-layer hidden states
// for interpretability research and multi-agent analysis.

// LayerAnalysisResult contains the result of analyzing hidden states across layers.
type LayerAnalysisResult struct {
	Layers       []int               `json:"layers"`        // Layer indices analyzed
	DEff         map[int]float64     `json:"d_eff"`         // D_eff per layer
	HiddenStates map[int][]float32   `json:"hidden_states"` // Hidden state vectors per layer
	Shape        []int               `json:"shape"`         // Common shape (if uniform)
	Model        string              `json:"model"`         // Model used
}

// ExtractLayerStates is a convenience method to request hidden states from specific layers.
// layers can be: "all" for all layers, a single int, or a slice of ints.
// Returns the response with HiddenStates populated for the requested layers.
func (l *Loom) ExtractLayerStates(ctx context.Context, text string, layers interface{}) (*ChatResponse, error) {
	req := ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: text},
		},
		ReturnHiddenStates: true,
		HiddenStateLayers:  layers,
	}
	return l.Chat(ctx, req)
}

// GetLayerDEff extracts D_eff values from a response's HiddenStates metadata.
// Returns a map of layer index to D_eff value.
func GetLayerDEff(resp *ChatResponse) map[int]float64 {
	result := make(map[int]float64)

	if resp == nil || resp.Metadata == nil {
		return result
	}

	if deffData, ok := resp.Metadata["layer_d_eff"]; ok {
		if deffMap, ok := deffData.(map[string]float64); ok {
			for layerStr, deff := range deffMap {
				var layerIdx int
				if _, err := fmt.Sscanf(layerStr, "%d", &layerIdx); err == nil {
					result[layerIdx] = deff
				}
			}
		}
	}

	return result
}

// CompareLayerStates compares hidden states between two responses at specified layers.
// Returns a comparison result with per-layer similarity metrics.
type LayerComparisonResult struct {
	Layers           []int              `json:"layers"`             // Layers compared
	CosineSimilarity map[int]float64    `json:"cosine_similarity"`  // Per-layer cosine similarity
	DEffDiff         map[int]float64    `json:"d_eff_diff"`         // Per-layer D_eff difference
	MeanSimilarity   float64            `json:"mean_similarity"`    // Average similarity across layers
}

// CompareLayerStates computes similarity metrics between two responses' hidden states.
func CompareLayerStates(resp1, resp2 *ChatResponse) *LayerComparisonResult {
	result := &LayerComparisonResult{
		Layers:           make([]int, 0),
		CosineSimilarity: make(map[int]float64),
		DEffDiff:         make(map[int]float64),
	}

	if resp1 == nil || resp2 == nil {
		return result
	}

	// Find common layers
	if resp1.HiddenStates == nil || resp2.HiddenStates == nil {
		return result
	}

	for layer := range resp1.HiddenStates {
		if _, ok := resp2.HiddenStates[layer]; ok {
			result.Layers = append(result.Layers, layer)
		}
	}

	// Compute similarity for each common layer
	deff1 := GetLayerDEff(resp1)
	deff2 := GetLayerDEff(resp2)

	var totalSim float64
	for _, layer := range result.Layers {
		hs1 := resp1.HiddenStates[layer]
		hs2 := resp2.HiddenStates[layer]

		if hs1 != nil && hs2 != nil && len(hs1.Vector) > 0 && len(hs2.Vector) > 0 {
			sim := cosineSimilarity(hs1.Vector, hs2.Vector)
			result.CosineSimilarity[layer] = sim
			totalSim += sim
		}

		// D_eff difference
		if d1, ok := deff1[layer]; ok {
			if d2, ok := deff2[layer]; ok {
				result.DEffDiff[layer] = d1 - d2
			}
		}
	}

	if len(result.Layers) > 0 {
		result.MeanSimilarity = totalSim / float64(len(result.Layers))
	}

	return result
}

// cosineSimilarity computes the cosine similarity between two float32 vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	// Import math for Sqrt is already available
	normA = sqrt(normA)
	normB = sqrt(normB)

	return dotProduct / (normA * normB)
}

// sqrt computes the square root using Newton's method to avoid math import dependency.
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 20; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}

// ParseLayerSelection parses a layer selection value and returns a normalized slice of layer indices.
// Supports: "all", single int, []int, or []interface{} (from JSON unmarshal).
func ParseLayerSelection(layers interface{}) ([]int, bool, error) {
	if layers == nil {
		return nil, false, nil
	}

	switch v := layers.(type) {
	case string:
		if v == "all" {
			return nil, true, nil // true indicates "all" was specified
		}
		return nil, false, fmt.Errorf("invalid layer selection string: %s", v)
	case int:
		return []int{v}, false, nil
	case []int:
		return v, false, nil
	case []interface{}:
		result := make([]int, 0, len(v))
		for _, item := range v {
			if num, ok := item.(float64); ok {
				result = append(result, int(num))
			} else if num, ok := item.(int); ok {
				result = append(result, num)
			}
		}
		return result, false, nil
	case float64:
		return []int{int(v)}, false, nil
	default:
		return nil, false, fmt.Errorf("unsupported layer selection type: %T", layers)
	}
}

// -----------------------------------------------------------------------------
// Error Helper Functions
// -----------------------------------------------------------------------------
// These functions create structured WeaverErrors for different Loom server
// failure scenarios with appropriate context and suggestions.

// createLoomMarshalError creates a structured error for JSON marshaling failures.
func createLoomMarshalError(err error, serverURL string) *werrors.WeaverError {
	return werrors.BackendWrap(err, werrors.ErrIOMarshalFailed,
		"failed to marshal request for Loom server").
		WithContext("backend", "loom").
		WithContext("server_url", serverURL).
		WithSuggestion("This is likely an internal error - please report it").
		WithSuggestion("Check that your message content is valid UTF-8")
}

// createLoomRequestError creates a structured error for HTTP request creation failures.
func createLoomRequestError(err error, serverURL string) *werrors.WeaverError {
	return werrors.BackendWrap(err, werrors.ErrBackendConnectionFailed,
		"failed to create request for Loom server").
		WithContext("backend", "loom").
		WithContext("server_url", serverURL).
		WithSuggestion("Check that the Loom server URL is valid").
		WithSuggestion("Verify the URL format: http://host:port or https://host:port")
}

// createLoomConnectionError creates a structured error for HTTP connection failures.
// It distinguishes between: timeout, connection refused, DNS failures, and other network issues.
func createLoomConnectionError(ctx context.Context, err error, serverURL string) *werrors.WeaverError {
	errStr := strings.ToLower(err.Error())

	// Check for context timeout/cancellation first
	if ctx.Err() != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return werrors.BackendWithContext(
				werrors.ErrBackendTimeout,
				"Loom server request timed out",
				map[string]string{
					werrors.ContextBackend: werrors.BackendLoom,
					"server_url":           serverURL,
				},
			).WithCause(err).
				WithSuggestion("The request took too long to complete").
				WithSuggestion("Check if the Loom server is responsive: curl " + serverURL + "/health").
				WithSuggestion("For large prompts, try reducing the input size").
				WithSuggestion("Check server logs for processing bottlenecks")
		}
		return werrors.BackendWithContext(
			werrors.ErrBackendConnectionFailed,
			"Loom server request was cancelled",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
			},
		).WithCause(err).
			WithSuggestion("The request was interrupted before completion")
	}

	// Connection refused - server not running
	if isLoomConnectionRefused(errStr) {
		return werrors.BackendWithContext(
			werrors.ErrBackendConnectionFailed,
			"cannot connect to Loom server - connection refused",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
			},
		).WithCause(err).
			WithSuggestion("Start the Loom server: cd TheLoom && python -m loom.server").
			WithSuggestion("Verify the server URL in your config (default: http://localhost:8080)").
			WithSuggestion("Check if the correct port is open and not blocked by firewall").
			WithSuggestion("See TheLoom documentation for server setup instructions")
	}

	// Timeout during connection
	if isLoomTimeout(errStr) {
		return werrors.BackendWithContext(
			werrors.ErrBackendTimeout,
			"Loom server connection timed out",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
			},
		).WithCause(err).
			WithSuggestion("The server may be overloaded or unresponsive").
			WithSuggestion("Check server health: curl " + serverURL + "/health").
			WithSuggestion("If running locally, ensure the server process is running").
			WithSuggestion("Check server logs for errors or GPU issues")
	}

	// DNS resolution failure
	if isLoomDNSError(errStr) {
		return werrors.BackendWithContext(
			werrors.ErrNetworkDNSFailed,
			"cannot resolve Loom server hostname",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
			},
		).WithCause(err).
			WithSuggestion("Check that the server hostname is correct").
			WithSuggestion("Verify your DNS settings and network connectivity").
			WithSuggestion("Try using an IP address instead of hostname")
	}

	// Generic network error
	return werrors.BackendWithContext(
		werrors.ErrBackendConnectionFailed,
		"cannot connect to Loom server",
		map[string]string{
			werrors.ContextBackend: werrors.BackendLoom,
			"server_url":           serverURL,
		},
	).WithCause(err).
		WithSuggestion("Ensure the Loom server is running at " + serverURL).
		WithSuggestion("Check your network connection").
		WithSuggestion("Verify firewall settings allow the connection")
}

// createLoomReadError creates a structured error for response reading failures.
func createLoomReadError(err error, serverURL string) *werrors.WeaverError {
	return werrors.BackendWrap(err, werrors.ErrIOReadFailed,
		"failed to read response from Loom server").
		WithContext("backend", "loom").
		WithContext("server_url", serverURL).
		WithSuggestion("The connection may have been interrupted").
		WithSuggestion("Check server logs for errors").
		WithSuggestion("Try the request again")
}

// createLoomHTTPError creates a structured error from HTTP error status codes.
// It parses the response body to distinguish between: model not found, GPU OOM,
// authentication errors, rate limiting, and other server errors.
func createLoomHTTPError(statusCode int, responseBody, model, serverURL string) *werrors.WeaverError {
	respLower := strings.ToLower(responseBody)

	// GPU Out of Memory
	if isLoomGPUOOM(respLower) {
		return werrors.BackendWithContext(
			werrors.ErrBackendAPIError,
			"GPU out of memory on Loom server",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
				"model":                model,
				"status_code":          fmt.Sprintf("%d", statusCode),
			},
		).
			WithContext("details", truncateLoomResponse(responseBody)).
			WithSuggestion("Reduce the prompt size or max_tokens setting").
			WithSuggestion("Try a smaller model if available").
			WithSuggestion("Close other GPU-intensive applications").
			WithSuggestion("Restart the Loom server to clear GPU memory").
			WithSuggestion("Consider adding 'device: cpu' to use CPU inference")
	}

	// Model not found
	if isLoomModelNotFound(respLower) {
		return werrors.BackendWithContext(
			werrors.ErrBackendAPIError,
			fmt.Sprintf("model '%s' not found on Loom server", model),
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
				"model":                model,
				"status_code":          fmt.Sprintf("%d", statusCode),
			},
		).
			WithContext("details", truncateLoomResponse(responseBody)).
			WithSuggestion("Check the model name in your config matches the server's loaded model").
			WithSuggestion("List available models: curl " + serverURL + "/v1/models").
			WithSuggestion("Ensure the model is downloaded and loaded on the server").
			WithSuggestion("Check server logs for model loading errors")
	}

	// CUDA/GPU errors
	if isLoomCUDAError(respLower) {
		return werrors.BackendWithContext(
			werrors.ErrBackendAPIError,
			"GPU/CUDA error on Loom server",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
				"model":                model,
				"status_code":          fmt.Sprintf("%d", statusCode),
			},
		).
			WithContext("details", truncateLoomResponse(responseBody)).
			WithSuggestion("Check that CUDA drivers are properly installed on the server").
			WithSuggestion("Verify GPU is available and not in use by other processes").
			WithSuggestion("Restart the Loom server to reinitialize GPU").
			WithSuggestion("Try 'device: cpu' in config for CPU-only inference")
	}

	// Server internal error (5xx)
	if statusCode >= 500 {
		return werrors.BackendWithContext(
			werrors.ErrBackendAPIError,
			"Loom server internal error",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
				"model":                model,
				"status_code":          fmt.Sprintf("%d", statusCode),
			},
		).
			WithContext("details", truncateLoomResponse(responseBody)).
			WithSuggestion("Check the Loom server logs for detailed error information").
			WithSuggestion("The server may need to be restarted").
			WithSuggestion("Verify the model is properly loaded: curl " + serverURL + "/health")
	}

	// Client errors (4xx)
	if statusCode == 400 {
		return werrors.BackendWithContext(
			werrors.ErrBackendAPIError,
			"invalid request to Loom server",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
				"model":                model,
				"status_code":          fmt.Sprintf("%d", statusCode),
			},
		).
			WithContext("details", truncateLoomResponse(responseBody)).
			WithSuggestion("Check the request format matches Loom API expectations").
			WithSuggestion("Verify message roles are valid (user/assistant/system)")
	}

	if statusCode == 401 || statusCode == 403 {
		return werrors.BackendWithContext(
			werrors.ErrBackendAuthFailed,
			"authentication failed for Loom server",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
				"status_code":          fmt.Sprintf("%d", statusCode),
			},
		).
			WithContext("details", truncateLoomResponse(responseBody)).
			WithSuggestion("Check if the Loom server requires authentication").
			WithSuggestion("Verify your API key or credentials are correct")
	}

	if statusCode == 429 {
		return werrors.BackendWithContext(
			werrors.ErrBackendAPIError,
			"Loom server rate limit exceeded",
			map[string]string{
				werrors.ContextBackend: werrors.BackendLoom,
				"server_url":           serverURL,
				"status_code":          fmt.Sprintf("%d", statusCode),
			},
		).
			WithContext("details", truncateLoomResponse(responseBody)).
			WithSuggestion("Wait a moment before sending more requests").
			WithSuggestion("Reduce request frequency")
	}

	// Generic error with status code
	return werrors.BackendWithContext(
		werrors.ErrBackendAPIError,
		fmt.Sprintf("Loom server returned error status %d", statusCode),
		map[string]string{
			werrors.ContextBackend: werrors.BackendLoom,
			"server_url":           serverURL,
			"model":                model,
			"status_code":          fmt.Sprintf("%d", statusCode),
		},
	).
		WithContext("details", truncateLoomResponse(responseBody)).
		WithSuggestion("Check the Loom server logs for more details").
		WithSuggestion("Verify server health: curl " + serverURL + "/health")
}

// createLoomUnmarshalError creates a structured error for JSON unmarshaling failures.
func createLoomUnmarshalError(err error, serverURL, responseBody string) *werrors.WeaverError {
	return werrors.BackendWrap(err, werrors.ErrIOUnmarshalFailed,
		"failed to parse response from Loom server").
		WithContext("backend", "loom").
		WithContext("server_url", serverURL).
		WithContext("response_preview", truncateLoomResponse(responseBody)).
		WithSuggestion("The server response format may be unexpected").
		WithSuggestion("Check that you're connecting to a compatible Loom server version").
		WithSuggestion("Verify server logs for response generation errors")
}

// -----------------------------------------------------------------------------
// Error Detection Helpers
// -----------------------------------------------------------------------------

// isLoomConnectionRefused checks if the error indicates connection refused.
func isLoomConnectionRefused(errStr string) bool {
	return strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "connect: connection refused") ||
		strings.Contains(errStr, "no connection could be made")
}

// isLoomTimeout checks if the error indicates a timeout.
func isLoomTimeout(errStr string) bool {
	return strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "timed out") ||
		strings.Contains(errStr, "deadline exceeded") ||
		strings.Contains(errStr, "i/o timeout")
}

// isLoomDNSError checks if the error indicates a DNS resolution failure.
func isLoomDNSError(errStr string) bool {
	return strings.Contains(errStr, "no such host") ||
		strings.Contains(errStr, "dns") ||
		strings.Contains(errStr, "lookup") && strings.Contains(errStr, "no such")
}

// isLoomGPUOOM checks if the response indicates GPU out of memory.
func isLoomGPUOOM(respLower string) bool {
	return strings.Contains(respLower, "out of memory") ||
		strings.Contains(respLower, "cuda out of memory") ||
		strings.Contains(respLower, "oom") && strings.Contains(respLower, "gpu") ||
		strings.Contains(respLower, "memory allocation failed") ||
		strings.Contains(respLower, "cudaerroroutofmemory") ||
		strings.Contains(respLower, "cuda error: out of memory")
}

// isLoomModelNotFound checks if the response indicates model not found.
func isLoomModelNotFound(respLower string) bool {
	return strings.Contains(respLower, "model not found") ||
		strings.Contains(respLower, "model does not exist") ||
		strings.Contains(respLower, "unknown model") ||
		strings.Contains(respLower, "invalid model") ||
		strings.Contains(respLower, "model") && strings.Contains(respLower, "not available") ||
		strings.Contains(respLower, "no such model")
}

// isLoomCUDAError checks if the response indicates a CUDA/GPU error.
func isLoomCUDAError(respLower string) bool {
	return strings.Contains(respLower, "cuda error") ||
		strings.Contains(respLower, "cuda runtime error") ||
		strings.Contains(respLower, "cudnn error") ||
		strings.Contains(respLower, "gpu error") ||
		strings.Contains(respLower, "cuda device") && strings.Contains(respLower, "error") ||
		strings.Contains(respLower, "no cuda gpus") ||
		strings.Contains(respLower, "cuda unavailable")
}

// truncateLoomResponse truncates a response body to a reasonable length for error context.
func truncateLoomResponse(body string) string {
	body = strings.TrimSpace(body)
	if len(body) > 300 {
		return body[:300] + "..."
	}
	return body
}
