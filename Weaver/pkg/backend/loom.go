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
	Device             string        `json:"device,omitempty"` // GPU: "auto", "cuda:0", "cuda:1"
}

// loomStreamingRequest extends loomRequest with streaming flag.
type loomStreamingRequest struct {
	Model              string        `json:"model"`
	Messages           []ChatMessage `json:"messages"`
	MaxTokens          int           `json:"max_tokens,omitempty"`
	Temperature        float64       `json:"temperature,omitempty"`
	ReturnHiddenStates bool          `json:"return_hidden_states,omitempty"`
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
	Metadata map[string]any `json:"metadata,omitempty"`
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
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", l.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := l.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("loom returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var loomResp loomResponse
	if err := json.Unmarshal(respBody, &loomResp); err != nil {
		return nil, err
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

	if loomResp.HiddenState != nil {
		result.HiddenState = &HiddenState{
			Vector: loomResp.HiddenState.Final,
			Shape:  loomResp.HiddenState.Shape,
			Layer:  loomResp.HiddenState.Layer,
			DType:  loomResp.HiddenState.DType,
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
