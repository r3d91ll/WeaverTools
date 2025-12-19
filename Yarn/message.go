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
type HiddenState struct {
	Vector []float32 `json:"vector"`
	Shape  []int     `json:"shape"`
	Layer  int       `json:"layer"`
	DType  string    `json:"dtype"`
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

// Dimension returns the hidden dimension size.
func (h *HiddenState) Dimension() int {
	if h == nil || len(h.Shape) < 2 {
		return len(h.Vector)
	}
	return h.Shape[len(h.Shape)-1]
}
