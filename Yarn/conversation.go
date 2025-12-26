package yarn

import (
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Conversation is an ordered sequence of messages with participant tracking.
type Conversation struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Messages     []*Message             `json:"messages"`
	Participants map[string]Participant `json:"participants"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	Metadata     map[string]any         `json:"metadata,omitempty"`

	mu sync.RWMutex
}

// Participant tracks an agent's involvement in the conversation.
type Participant struct {
	AgentID      string    `json:"agent_id"`
	AgentName    string    `json:"agent_name"`
	Role         string    `json:"role"`
	JoinedAt     time.Time `json:"joined_at"`
	MessageCount int       `json:"message_count"`
}

// NewConversation creates a new conversation.
func NewConversation(name string) *Conversation {
	now := time.Now()
	return &Conversation{
		ID:           uuid.New().String(),
		Name:         name,
		Messages:     make([]*Message, 0),
		Participants: make(map[string]Participant),
		CreatedAt:    now,
		UpdatedAt:    now,
		Metadata:     make(map[string]any),
	}
}

// Add appends a message to the conversation.
func (c *Conversation) Add(msg *Message) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.Messages = append(c.Messages, msg)
	c.UpdatedAt = time.Now()

	// Track participant
	if msg.AgentID != "" {
		if p, exists := c.Participants[msg.AgentID]; exists {
			p.MessageCount++
			c.Participants[msg.AgentID] = p
		} else {
			c.Participants[msg.AgentID] = Participant{
				AgentID:      msg.AgentID,
				AgentName:    msg.AgentName,
				Role:         string(msg.Role),
				JoinedAt:     msg.Timestamp,
				MessageCount: 1,
			}
		}
	}
}

// History returns the last n messages (or all if n <= 0).
func (c *Conversation) History(limit int) []*Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if limit <= 0 || limit >= len(c.Messages) {
		result := make([]*Message, len(c.Messages))
		copy(result, c.Messages)
		return result
	}

	start := len(c.Messages) - limit
	result := make([]*Message, limit)
	copy(result, c.Messages[start:])
	return result
}

// LastMessage returns the most recent message, or nil if empty.
func (c *Conversation) LastMessage() *Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.Messages) == 0 {
		return nil
	}
	return c.Messages[len(c.Messages)-1]
}

// Length returns the number of messages.
func (c *Conversation) Length() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.Messages)
}

// MessagesWithHiddenStates returns only messages that have hidden state data.
func (c *Conversation) MessagesWithHiddenStates() []*Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var result []*Message
	for _, msg := range c.Messages {
		if msg.HasHiddenState() {
			result = append(result, msg)
		}
	}
	return result
}

// MessagesByRole returns only messages that match the specified role.
func (c *Conversation) MessagesByRole(role MessageRole) []*Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var result []*Message
	for _, msg := range c.Messages {
		if msg.Role == role {
			result = append(result, msg)
		}
	}
	return result
}

// MessagesByAgent returns only messages that match the specified agent ID.
// If agentID is empty, returns messages with empty AgentID (literal match).
func (c *Conversation) MessagesByAgent(agentID string) []*Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var result []*Message
	for _, msg := range c.Messages {
		if msg.AgentID == agentID {
			result = append(result, msg)
		}
	}
	return result
}

// MessagesSince returns only messages with Timestamp strictly after the given time.
// Messages are returned in chronological order (as stored).
func (c *Conversation) MessagesSince(since time.Time) []*Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var result []*Message
	for _, msg := range c.Messages {
		if msg.Timestamp.After(since) {
			result = append(result, msg)
		}
	}
	return result
}

// MessagesWithMetadata returns only messages that have the specified key present
// in their Metadata map (regardless of value). Messages with nil Metadata are skipped.
func (c *Conversation) MessagesWithMetadata(key string) []*Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var result []*Message
	for _, msg := range c.Messages {
		if msg.Metadata != nil {
			if _, exists := msg.Metadata[key]; exists {
				result = append(result, msg)
			}
		}
	}
	return result
}

// Validate checks if the conversation is valid.
// Returns a ValidationError if invalid, nil if valid.
func (c *Conversation) Validate() *ValidationError {
	if c.ID == "" {
		return &ValidationError{Field: "id", Message: "id is required"}
	}
	if c.Name == "" {
		return &ValidationError{Field: "name", Message: "name is required"}
	}
	if c.CreatedAt.IsZero() {
		return &ValidationError{Field: "created_at", Message: "created_at is required"}
	}
	if c.UpdatedAt.Before(c.CreatedAt) {
		return &ValidationError{Field: "updated_at", Message: "updated_at must not be before created_at"}
	}

	// Cascade validation: validate all messages
	for i, msg := range c.Messages {
		if msg == nil {
			return &ValidationError{
				Field:   "messages",
				Message: "message at index " + strconv.Itoa(i) + " is nil",
			}
		}
		if err := msg.Validate(); err != nil {
			return &ValidationError{
				Field:   "messages[" + strconv.Itoa(i) + "]." + err.Field,
				Message: err.Message,
			}
		}
	}

	return nil
}

// Validate checks if the participant is valid.
// Returns a ValidationError if invalid, nil if valid.
func (p *Participant) Validate() *ValidationError {
	if p.AgentID == "" {
		return &ValidationError{Field: "agent_id", Message: "agent_id is required"}
	}
	if p.Role == "" {
		return &ValidationError{Field: "role", Message: "role is required"}
	}
	if p.JoinedAt.IsZero() {
		return &ValidationError{Field: "joined_at", Message: "joined_at is required"}
	}
	return nil
}
