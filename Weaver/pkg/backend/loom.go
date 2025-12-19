package backend

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Loom connects to The Loom server for inference with hidden state extraction.
type Loom struct {
	name       string
	baseURL    string
	model      string
	httpClient *http.Client
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

func (l *Loom) SupportsHiddenStates() bool { return true }

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
		model = l.model
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
		loomReq.MaxTokens = 256
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
func (l *Loom) SetModel(model string) { l.model = model }

// Model returns the current model.
func (l *Loom) Model() string { return l.model }
