// Package backend provides the unified interface for model communication.
// Weaver orchestrates agents through these backends.
package backend

import (
	"context"
	"time"

	yarn "github.com/r3d91ll/yarn"
)

// Type identifies the backend type.
type Type string

const (
	TypeClaudeCode Type = "claudecode" // Claude CLI subprocess
	TypeLoom       Type = "loom"       // The Loom server
)

// Capabilities describes what a backend can do.
type Capabilities struct {
	ContextLimit      int  `json:"contextLimit"`
	SupportsTools     bool `json:"supportsTools"`
	SupportsStreaming bool `json:"supportsStreaming"`
	SupportsHidden    bool `json:"supportsHidden"` // Hidden state extraction
	MaxTokens         int  `json:"maxTokens"`
}

// ChatMessage represents a single message.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// TokenUsage tracks token consumption.
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatRequest contains parameters for a chat request.
type ChatRequest struct {
	Model              string        `json:"model"`
	Messages           []ChatMessage `json:"messages"`
	MaxTokens          int           `json:"max_tokens,omitempty"`
	Temperature        float64       `json:"temperature,omitempty"`
	Stream             bool          `json:"stream,omitempty"`
	ReturnHiddenStates bool          `json:"return_hidden_states,omitempty"`
	Device             string        `json:"device,omitempty"` // GPU device: "auto", "cuda:0", "cuda:1", etc.
}

// ChatResponse contains the model's response.
type ChatResponse struct {
	Content      string            `json:"content"`
	Usage        TokenUsage        `json:"usage"`
	HiddenState  *yarn.HiddenState `json:"hidden_state,omitempty"`
	Metadata     map[string]any    `json:"metadata,omitempty"`
	LatencyMS    float64           `json:"latency_ms"`
	Model        string            `json:"model"`
	FinishReason string            `json:"finish_reason"`
}

// StreamChunk represents a chunk of streamed response.
type StreamChunk struct {
	Content      string `json:"content"`
	Done         bool   `json:"done"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// Backend is the unified interface for model communication.
type Backend interface {
	Name() string
	Type() Type
	IsAvailable(ctx context.Context) bool
	Capabilities() Capabilities
	Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error)
	ChatStream(ctx context.Context, req ChatRequest) (<-chan StreamChunk, <-chan error)
}

// Config holds common backend configuration.
type Config struct {
	Name    string        `yaml:"name"`
	Type    Type          `yaml:"type"`
	URL     string        `yaml:"url,omitempty"`
	Model   string        `yaml:"model,omitempty"`
	Timeout time.Duration `yaml:"timeout,omitempty"`
}
