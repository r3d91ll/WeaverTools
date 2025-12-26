package backend

import (
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

		// For now, use non-streaming and send as single chunk
		resp, err := l.Chat(ctx, req)
		if err != nil {
			errs <- err
			return
		}

		chunks <- StreamChunk{Content: resp.Content, Done: true, FinishReason: "stop"}
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
