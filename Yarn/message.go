// Package yarn manages conversations, measurements, and data storage.
// Yarn is the thread that connects everything - tracking WHAT HAPPENED.
package yarn

import (
	"time"

	"github.com/google/uuid"
)

// MessageRole represents the sender type.
type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleTool      MessageRole = "tool"
)

// IsValid returns true if this is a valid message role.
func (r MessageRole) IsValid() bool {
	switch r {
	case RoleSystem, RoleUser, RoleAssistant, RoleTool:
		return true
	default:
		return false
	}
}

// Message is the atomic unit of communication between agents.
type Message struct {
	ID          string         `json:"id"`
	Role        MessageRole    `json:"role"`
	Content     string         `json:"content"`
	Timestamp   time.Time      `json:"timestamp"`
	AgentID     string         `json:"agent_id,omitempty"`
	AgentName   string         `json:"agent_name,omitempty"`
	HiddenState *HiddenState   `json:"hidden_state,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`

	// Tool-related fields
	ToolCallID string `json:"tool_call_id,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
}

// HiddenState represents the boundary object - semantic state before text projection.
// Memory note: Vector can be large (e.g., 4096 floats = 16KB for typical LLMs).
// For models with larger hidden dimensions (e.g., 8192), expect ~32KB per state.
// Consider streaming or lazy loading for batch processing of many messages.
type HiddenState struct {
	Vector []float32 `json:"vector"` // Hidden state vector, typically 2048-8192 float32 values
	Shape  []int     `json:"shape"`  // Original tensor shape, e.g., [1, seq_len, hidden_dim]
	Layer  int       `json:"layer"`  // Layer index this state was extracted from
	DType  string    `json:"dtype"`  // Data type, typically "float32"
}

// NewMessage creates a new Message with a generated UUID.
func NewMessage(role MessageRole, content string) *Message {
	return &Message{
		ID:        uuid.New().String(),
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
		Metadata:  make(map[string]any),
	}
}

// NewAgentMessage creates a Message attributed to a specific agent.
func NewAgentMessage(role MessageRole, content, agentID, agentName string) *Message {
	msg := NewMessage(role, content)
	msg.AgentID = agentID
	msg.AgentName = agentName
	return msg
}

// WithHiddenState attaches a hidden state to the message.
func (m *Message) WithHiddenState(hs *HiddenState) *Message {
	m.HiddenState = hs
	return m
}

// WithMetadata adds a key-value pair to the message metadata.
func (m *Message) WithMetadata(key string, value any) *Message {
	if m.Metadata == nil {
		m.Metadata = make(map[string]any)
	}
	m.Metadata[key] = value
	return m
}

// HasHiddenState returns true if this message has hidden state data.
func (m *Message) HasHiddenState() bool {
	return m.HiddenState != nil && len(m.HiddenState.Vector) > 0
}

// Validate checks if the message is valid.
// Returns a ValidationError if invalid, nil if valid.
func (m *Message) Validate() *ValidationError {
	if m.ID == "" {
		return &ValidationError{Field: "id", Message: "id is required"}
	}
	if !m.Role.IsValid() {
		return &ValidationError{Field: "role", Message: "invalid role"}
	}
	if m.Timestamp.IsZero() {
		return &ValidationError{Field: "timestamp", Message: "timestamp is required"}
	}

	// Tool messages require ToolCallID
	if m.Role == RoleTool {
		if m.ToolCallID == "" {
			return &ValidationError{Field: "tool_call_id", Message: "tool_call_id is required for tool messages"}
		}
	} else {
		// Non-tool messages require content
		if m.Content == "" {
			return &ValidationError{Field: "content", Message: "content is required for non-tool messages"}
		}
	}

	return nil
}

// Dimension returns the hidden dimension size.
// Returns 0 if the HiddenState is nil.
func (h *HiddenState) Dimension() int {
	if h == nil {
		return 0
	}
	if len(h.Shape) < 2 {
		return len(h.Vector)
	}
	return h.Shape[len(h.Shape)-1]
}
