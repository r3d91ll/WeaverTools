package yarn

import (
	"sync"
	"testing"
	"time"
)

// TestConversationValidate tests the Validate method for Conversation.
func TestConversationValidate(t *testing.T) {
	now := time.Now()
	validMessage := &Message{
		ID:        "msg-1",
		Role:      RoleUser,
		Content:   "Hello",
		Timestamp: now,
	}

	tests := []struct {
		name        string
		conv        Conversation
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name: "valid conversation with no messages",
			conv: Conversation{
				ID:        "conv-1",
				Name:      "Test Conversation",
				Messages:  []*Message{},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr: false,
		},
		{
			name: "valid conversation with messages",
			conv: Conversation{
				ID:        "conv-2",
				Name:      "Test Conversation",
				Messages:  []*Message{validMessage},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr: false,
		},
		{
			name: "valid conversation with updated_at after created_at",
			conv: Conversation{
				ID:        "conv-3",
				Name:      "Test Conversation",
				Messages:  []*Message{},
				CreatedAt: now,
				UpdatedAt: now.Add(time.Hour),
			},
			wantErr: false,
		},
		{
			name: "valid conversation with participants",
			conv: Conversation{
				ID:       "conv-4",
				Name:     "Test Conversation",
				Messages: []*Message{},
				Participants: map[string]Participant{
					"agent-1": {
						AgentID:      "agent-1",
						AgentName:    "Test Agent",
						Role:         "assistant",
						JoinedAt:     now,
						MessageCount: 5,
					},
				},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr: false,
		},
		{
			name: "valid conversation with metadata",
			conv: Conversation{
				ID:        "conv-5",
				Name:      "Test Conversation",
				Messages:  []*Message{},
				CreatedAt: now,
				UpdatedAt: now,
				Metadata:  map[string]any{"key": "value"},
			},
			wantErr: false,
		},
		{
			name: "missing id",
			conv: Conversation{
				ID:        "",
				Name:      "Test Conversation",
				Messages:  []*Message{},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "id",
			wantMessage: "id is required",
		},
		{
			name: "missing name",
			conv: Conversation{
				ID:        "conv-6",
				Name:      "",
				Messages:  []*Message{},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "name",
			wantMessage: "name is required",
		},
		{
			name: "zero created_at",
			conv: Conversation{
				ID:        "conv-7",
				Name:      "Test Conversation",
				Messages:  []*Message{},
				CreatedAt: time.Time{},
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "created_at",
			wantMessage: "created_at is required",
		},
		{
			name: "updated_at before created_at",
			conv: Conversation{
				ID:        "conv-8",
				Name:      "Test Conversation",
				Messages:  []*Message{},
				CreatedAt: now,
				UpdatedAt: now.Add(-time.Hour),
			},
			wantErr:     true,
			wantField:   "updated_at",
			wantMessage: "updated_at must not be before created_at",
		},
		{
			name: "nil message in messages slice",
			conv: Conversation{
				ID:        "conv-9",
				Name:      "Test Conversation",
				Messages:  []*Message{nil},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "messages",
			wantMessage: "message at index 0 is nil",
		},
		{
			name: "nil message at index 1",
			conv: Conversation{
				ID:        "conv-10",
				Name:      "Test Conversation",
				Messages:  []*Message{validMessage, nil},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "messages",
			wantMessage: "message at index 1 is nil",
		},
		{
			name: "invalid message - missing id",
			conv: Conversation{
				ID:   "conv-11",
				Name: "Test Conversation",
				Messages: []*Message{
					{
						ID:        "",
						Role:      RoleUser,
						Content:   "Hello",
						Timestamp: now,
					},
				},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "messages[0].id",
			wantMessage: "id is required",
		},
		{
			name: "invalid message - invalid role",
			conv: Conversation{
				ID:   "conv-12",
				Name: "Test Conversation",
				Messages: []*Message{
					{
						ID:        "msg-invalid-role",
						Role:      MessageRole("invalid"),
						Content:   "Hello",
						Timestamp: now,
					},
				},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "messages[0].role",
			wantMessage: "invalid role",
		},
		{
			name: "invalid message at index 2",
			conv: Conversation{
				ID:   "conv-13",
				Name: "Test Conversation",
				Messages: []*Message{
					validMessage,
					{
						ID:        "msg-2",
						Role:      RoleAssistant,
						Content:   "Hi there!",
						Timestamp: now,
					},
					{
						ID:        "",
						Role:      RoleUser,
						Content:   "Invalid",
						Timestamp: now,
					},
				},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "messages[2].id",
			wantMessage: "id is required",
		},
		{
			name: "invalid message - zero timestamp",
			conv: Conversation{
				ID:   "conv-14",
				Name: "Test Conversation",
				Messages: []*Message{
					{
						ID:        "msg-zero-ts",
						Role:      RoleUser,
						Content:   "Hello",
						Timestamp: time.Time{},
					},
				},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "messages[0].timestamp",
			wantMessage: "timestamp is required",
		},
		{
			name: "invalid message - empty content for non-tool message",
			conv: Conversation{
				ID:   "conv-15",
				Name: "Test Conversation",
				Messages: []*Message{
					{
						ID:        "msg-empty-content",
						Role:      RoleUser,
						Content:   "",
						Timestamp: now,
					},
				},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "messages[0].content",
			wantMessage: "content is required for non-tool messages",
		},
		{
			name: "invalid message - tool without tool_call_id",
			conv: Conversation{
				ID:   "conv-16",
				Name: "Test Conversation",
				Messages: []*Message{
					{
						ID:        "msg-tool-no-id",
						Role:      RoleTool,
						Content:   "Tool response",
						Timestamp: now,
					},
				},
				CreatedAt: now,
				UpdatedAt: now,
			},
			wantErr:     true,
			wantField:   "messages[0].tool_call_id",
			wantMessage: "tool_call_id is required for tool messages",
		},
	}

	for i := range tests {
		tt := &tests[i]
		t.Run(tt.name, func(t *testing.T) {
			err := tt.conv.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Conversation.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Conversation.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("Conversation.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("Conversation.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestParticipantValidate tests the Validate method for Participant.
func TestParticipantValidate(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name        string
		participant Participant
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name: "valid participant with all fields",
			participant: Participant{
				AgentID:      "agent-1",
				AgentName:    "Test Agent",
				Role:         "assistant",
				JoinedAt:     now,
				MessageCount: 10,
			},
			wantErr: false,
		},
		{
			name: "valid participant with minimal fields",
			participant: Participant{
				AgentID:  "agent-2",
				Role:     "user",
				JoinedAt: now,
			},
			wantErr: false,
		},
		{
			name: "valid participant with zero message count",
			participant: Participant{
				AgentID:      "agent-3",
				AgentName:    "New Agent",
				Role:         "system",
				JoinedAt:     now,
				MessageCount: 0,
			},
			wantErr: false,
		},
		{
			name: "missing agent_id",
			participant: Participant{
				AgentID:  "",
				Role:     "assistant",
				JoinedAt: now,
			},
			wantErr:     true,
			wantField:   "agent_id",
			wantMessage: "agent_id is required",
		},
		{
			name: "missing role",
			participant: Participant{
				AgentID:  "agent-4",
				Role:     "",
				JoinedAt: now,
			},
			wantErr:     true,
			wantField:   "role",
			wantMessage: "role is required",
		},
		{
			name: "zero joined_at",
			participant: Participant{
				AgentID:  "agent-5",
				Role:     "assistant",
				JoinedAt: time.Time{},
			},
			wantErr:     true,
			wantField:   "joined_at",
			wantMessage: "joined_at is required",
		},
		{
			name: "all required fields missing",
			participant: Participant{
				AgentID:  "",
				Role:     "",
				JoinedAt: time.Time{},
			},
			wantErr:     true,
			wantField:   "agent_id",
			wantMessage: "agent_id is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.participant.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Participant.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Participant.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("Participant.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("Participant.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestNewConversationValidation tests that NewConversation creates valid conversations.
func TestNewConversationValidation(t *testing.T) {
	tests := []struct {
		name     string
		convName string
		wantErr  bool
	}{
		{
			name:     "NewConversation creates valid conversation",
			convName: "Test Conversation",
			wantErr:  false,
		},
		{
			name:     "NewConversation with simple name",
			convName: "Chat",
			wantErr:  false,
		},
		{
			name:     "NewConversation with special characters in name",
			convName: "Conversation #1 - Test!",
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv := NewConversation(tt.convName)
			err := conv.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("NewConversation() created conversation with Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestConversationValidateWithMultipleMessages tests cascade validation with multiple messages.
func TestConversationValidateWithMultipleMessages(t *testing.T) {
	now := time.Now()

	t.Run("valid conversation with multiple messages", func(t *testing.T) {
		conv := Conversation{
			ID:   "conv-multi",
			Name: "Multi-Message Conversation",
			Messages: []*Message{
				{
					ID:        "msg-1",
					Role:      RoleSystem,
					Content:   "You are a helpful assistant.",
					Timestamp: now,
				},
				{
					ID:        "msg-2",
					Role:      RoleUser,
					Content:   "Hello!",
					Timestamp: now.Add(time.Second),
				},
				{
					ID:        "msg-3",
					Role:      RoleAssistant,
					Content:   "Hi there! How can I help you today?",
					Timestamp: now.Add(2 * time.Second),
				},
				{
					ID:         "msg-4",
					Role:       RoleTool,
					Content:    "Tool result",
					Timestamp:  now.Add(3 * time.Second),
					ToolCallID: "call-123",
					ToolName:   "get_data",
				},
			},
			CreatedAt: now,
			UpdatedAt: now.Add(3 * time.Second),
		}

		if err := conv.Validate(); err != nil {
			t.Errorf("Conversation.Validate() unexpected error: %v", err)
		}
	})

	t.Run("invalid message in the middle of conversation", func(t *testing.T) {
		conv := Conversation{
			ID:   "conv-middle-invalid",
			Name: "Conversation with Invalid Middle Message",
			Messages: []*Message{
				{
					ID:        "msg-1",
					Role:      RoleUser,
					Content:   "Hello!",
					Timestamp: now,
				},
				{
					ID:        "msg-2",
					Role:      RoleAssistant,
					Content:   "", // Invalid: empty content for assistant
					Timestamp: now.Add(time.Second),
				},
				{
					ID:        "msg-3",
					Role:      RoleUser,
					Content:   "Thanks!",
					Timestamp: now.Add(2 * time.Second),
				},
			},
			CreatedAt: now,
			UpdatedAt: now.Add(2 * time.Second),
		}

		err := conv.Validate()
		if err == nil {
			t.Errorf("Conversation.Validate() expected error, got nil")
			return
		}
		if err.Field != "messages[1].content" {
			t.Errorf("Conversation.Validate() error field = %q, want %q", err.Field, "messages[1].content")
		}
	})
}

// TestConversationValidateTimeConsistency tests time-related validation rules.
func TestConversationValidateTimeConsistency(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name        string
		createdAt   time.Time
		updatedAt   time.Time
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name:      "same created_at and updated_at is valid",
			createdAt: now,
			updatedAt: now,
			wantErr:   false,
		},
		{
			name:      "updated_at 1 nanosecond after created_at is valid",
			createdAt: now,
			updatedAt: now.Add(time.Nanosecond),
			wantErr:   false,
		},
		{
			name:      "updated_at 1 day after created_at is valid",
			createdAt: now,
			updatedAt: now.Add(24 * time.Hour),
			wantErr:   false,
		},
		{
			name:        "updated_at 1 nanosecond before created_at is invalid",
			createdAt:   now,
			updatedAt:   now.Add(-time.Nanosecond),
			wantErr:     true,
			wantField:   "updated_at",
			wantMessage: "updated_at must not be before created_at",
		},
		{
			name:        "updated_at 1 hour before created_at is invalid",
			createdAt:   now,
			updatedAt:   now.Add(-time.Hour),
			wantErr:     true,
			wantField:   "updated_at",
			wantMessage: "updated_at must not be before created_at",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv := Conversation{
				ID:        "conv-time-test",
				Name:      "Time Test Conversation",
				Messages:  []*Message{},
				CreatedAt: tt.createdAt,
				UpdatedAt: tt.updatedAt,
			}

			err := conv.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Conversation.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Conversation.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("Conversation.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("Conversation.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestConversationAddThenValidate tests that Add method maintains valid state.
func TestConversationAddThenValidate(t *testing.T) {
	conv := NewConversation("Test Conversation")

	// Validate initial state
	if err := conv.Validate(); err != nil {
		t.Fatalf("NewConversation() created invalid conversation: %v", err)
	}

	// Add a valid message
	msg := NewMessage(RoleUser, "Hello!")
	conv.Add(msg)

	// Should still be valid
	if err := conv.Validate(); err != nil {
		t.Errorf("Conversation.Validate() after Add() unexpected error: %v", err)
	}

	// Add another message
	resp := NewAgentMessage(RoleAssistant, "Hi there!", "agent-1", "Assistant")
	conv.Add(resp)

	// Should still be valid
	if err := conv.Validate(); err != nil {
		t.Errorf("Conversation.Validate() after second Add() unexpected error: %v", err)
	}

	// Verify message count
	if conv.Length() != 2 {
		t.Errorf("Conversation.Length() = %d, want %d", conv.Length(), 2)
	}
}

// TestParticipantValidatePointerReceiver tests Participant.Validate with pointer receiver.
func TestParticipantValidatePointerReceiver(t *testing.T) {
	now := time.Now()

	t.Run("pointer to valid participant", func(t *testing.T) {
		p := &Participant{
			AgentID:  "agent-1",
			Role:     "assistant",
			JoinedAt: now,
		}
		if err := p.Validate(); err != nil {
			t.Errorf("Participant.Validate() unexpected error: %v", err)
		}
	})

	t.Run("pointer to invalid participant", func(t *testing.T) {
		p := &Participant{
			AgentID:  "",
			Role:     "assistant",
			JoinedAt: now,
		}
		err := p.Validate()
		if err == nil {
			t.Errorf("Participant.Validate() expected error, got nil")
			return
		}
		if err.Field != "agent_id" {
			t.Errorf("Participant.Validate() error field = %q, want %q", err.Field, "agent_id")
		}
	})
}

// TestMessagesByRole_Basic tests basic filtering by role.
func TestMessagesByRole_Basic(t *testing.T) {
	conv := NewConversation("test")

	// Add messages with different roles
	conv.Add(NewMessage(RoleSystem, "system message"))
	conv.Add(NewMessage(RoleUser, "user message 1"))
	conv.Add(NewMessage(RoleAssistant, "assistant message"))
	conv.Add(NewMessage(RoleUser, "user message 2"))
	conv.Add(NewMessage(RoleTool, "tool message"))

	// Test filtering by each role
	tests := []struct {
		role     MessageRole
		expected int
	}{
		{RoleSystem, 1},
		{RoleUser, 2},
		{RoleAssistant, 1},
		{RoleTool, 1},
	}

	for _, tc := range tests {
		result := conv.MessagesByRole(tc.role)
		if len(result) != tc.expected {
			t.Errorf("MessagesByRole(%s): expected %d, got %d", tc.role, tc.expected, len(result))
		}

		// Verify all returned messages have the correct role
		for _, msg := range result {
			if msg.Role != tc.role {
				t.Errorf("MessagesByRole(%s): returned message with role %s", tc.role, msg.Role)
			}
		}
	}
}

// TestMessagesByRole_EmptyConversation tests filtering on empty conversation.
func TestMessagesByRole_EmptyConversation(t *testing.T) {
	conv := NewConversation("empty")

	result := conv.MessagesByRole(RoleUser)
	if result == nil {
		t.Error("MessagesByRole on empty conversation should return empty slice, not nil")
	}
	if len(result) != 0 {
		t.Errorf("MessagesByRole on empty conversation: expected 0, got %d", len(result))
	}
}

// TestMessagesByRole_NoMatches tests when no messages match the role.
func TestMessagesByRole_NoMatches(t *testing.T) {
	conv := NewConversation("test")
	conv.Add(NewMessage(RoleUser, "user message"))
	conv.Add(NewMessage(RoleAssistant, "assistant message"))

	result := conv.MessagesByRole(RoleTool)
	if len(result) != 0 {
		t.Errorf("MessagesByRole with no matches: expected 0, got %d", len(result))
	}
}

// TestMessagesByRole_AllMatch tests when all messages match.
func TestMessagesByRole_AllMatch(t *testing.T) {
	conv := NewConversation("test")
	conv.Add(NewMessage(RoleUser, "message 1"))
	conv.Add(NewMessage(RoleUser, "message 2"))
	conv.Add(NewMessage(RoleUser, "message 3"))

	result := conv.MessagesByRole(RoleUser)
	if len(result) != 3 {
		t.Errorf("MessagesByRole all match: expected 3, got %d", len(result))
	}
}

// TestMessagesByAgent_Basic tests basic filtering by agent ID.
func TestMessagesByAgent_Basic(t *testing.T) {
	conv := NewConversation("test")

	conv.Add(NewAgentMessage(RoleAssistant, "message 1", "agent-1", "Agent One"))
	conv.Add(NewAgentMessage(RoleAssistant, "message 2", "agent-2", "Agent Two"))
	conv.Add(NewAgentMessage(RoleAssistant, "message 3", "agent-1", "Agent One"))
	conv.Add(NewAgentMessage(RoleAssistant, "message 4", "agent-3", "Agent Three"))

	result := conv.MessagesByAgent("agent-1")
	if len(result) != 2 {
		t.Errorf("MessagesByAgent(agent-1): expected 2, got %d", len(result))
	}

	// Verify all returned messages have the correct agent ID
	for _, msg := range result {
		if msg.AgentID != "agent-1" {
			t.Errorf("MessagesByAgent(agent-1): returned message with agent ID %s", msg.AgentID)
		}
	}
}

// TestMessagesByAgent_EmptyConversation tests filtering on empty conversation.
func TestMessagesByAgent_EmptyConversation(t *testing.T) {
	conv := NewConversation("empty")

	result := conv.MessagesByAgent("any-agent")
	if result == nil {
		t.Error("MessagesByAgent on empty conversation should return empty slice, not nil")
	}
	if len(result) != 0 {
		t.Errorf("MessagesByAgent on empty conversation: expected 0, got %d", len(result))
	}
}

// TestMessagesByAgent_NoMatches tests when no messages match the agent ID.
func TestMessagesByAgent_NoMatches(t *testing.T) {
	conv := NewConversation("test")
	conv.Add(NewAgentMessage(RoleAssistant, "message", "agent-1", "Agent One"))
	conv.Add(NewAgentMessage(RoleAssistant, "message", "agent-2", "Agent Two"))

	result := conv.MessagesByAgent("agent-99")
	if len(result) != 0 {
		t.Errorf("MessagesByAgent with no matches: expected 0, got %d", len(result))
	}
}

// TestMessagesByAgent_EmptyAgentID tests filtering for empty agent ID.
func TestMessagesByAgent_EmptyAgentID(t *testing.T) {
	conv := NewConversation("test")

	// Add messages with and without agent IDs
	conv.Add(NewMessage(RoleUser, "user message")) // No agent ID
	conv.Add(NewAgentMessage(RoleAssistant, "agent message", "agent-1", "Agent One"))
	conv.Add(NewMessage(RoleUser, "another user message")) // No agent ID

	result := conv.MessagesByAgent("")
	if len(result) != 2 {
		t.Errorf("MessagesByAgent(''): expected 2 messages with empty AgentID, got %d", len(result))
	}

	// Verify all returned messages have empty agent ID
	for _, msg := range result {
		if msg.AgentID != "" {
			t.Errorf("MessagesByAgent(''): returned message with non-empty agent ID %s", msg.AgentID)
		}
	}
}

// TestMessagesSince_Basic tests basic filtering by time.
func TestMessagesSince_Basic(t *testing.T) {
	conv := NewConversation("test")

	// Create messages with controlled timestamps
	baseTime := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)

	msg1 := NewMessage(RoleUser, "old message 1")
	msg1.Timestamp = baseTime.Add(-2 * time.Hour)
	conv.Add(msg1)

	msg2 := NewMessage(RoleUser, "old message 2")
	msg2.Timestamp = baseTime.Add(-1 * time.Hour)
	conv.Add(msg2)

	msg3 := NewMessage(RoleUser, "new message 1")
	msg3.Timestamp = baseTime.Add(1 * time.Hour)
	conv.Add(msg3)

	msg4 := NewMessage(RoleUser, "new message 2")
	msg4.Timestamp = baseTime.Add(2 * time.Hour)
	conv.Add(msg4)

	result := conv.MessagesSince(baseTime)
	if len(result) != 2 {
		t.Errorf("MessagesSince: expected 2 messages after baseTime, got %d", len(result))
	}

	// Verify messages are in chronological order
	for i := 1; i < len(result); i++ {
		if result[i].Timestamp.Before(result[i-1].Timestamp) {
			t.Error("MessagesSince: messages not in chronological order")
		}
	}
}

// TestMessagesSince_EmptyConversation tests filtering on empty conversation.
func TestMessagesSince_EmptyConversation(t *testing.T) {
	conv := NewConversation("empty")

	result := conv.MessagesSince(time.Now())
	if result == nil {
		t.Error("MessagesSince on empty conversation should return empty slice, not nil")
	}
	if len(result) != 0 {
		t.Errorf("MessagesSince on empty conversation: expected 0, got %d", len(result))
	}
}

// TestMessagesSince_NoMatches tests when no messages are after the given time.
func TestMessagesSince_NoMatches(t *testing.T) {
	conv := NewConversation("test")

	// Add old messages
	oldTime := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	msg := NewMessage(RoleUser, "old message")
	msg.Timestamp = oldTime
	conv.Add(msg)

	// Filter for messages after current time
	result := conv.MessagesSince(time.Now())
	if len(result) != 0 {
		t.Errorf("MessagesSince with no matches: expected 0, got %d", len(result))
	}
}

// TestMessagesSince_AllMatch tests when all messages are after the given time.
func TestMessagesSince_AllMatch(t *testing.T) {
	conv := NewConversation("test")

	// Add messages with recent timestamps
	conv.Add(NewMessage(RoleUser, "message 1"))
	conv.Add(NewMessage(RoleUser, "message 2"))
	conv.Add(NewMessage(RoleUser, "message 3"))

	// Filter for messages after a very old time
	oldTime := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
	result := conv.MessagesSince(oldTime)
	if len(result) != 3 {
		t.Errorf("MessagesSince all match: expected 3, got %d", len(result))
	}
}

// TestMessagesSince_ExactTimestamp tests that messages at exact timestamp are excluded.
func TestMessagesSince_ExactTimestamp(t *testing.T) {
	conv := NewConversation("test")

	exactTime := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)

	msg := NewMessage(RoleUser, "exact time message")
	msg.Timestamp = exactTime
	conv.Add(msg)

	// MessagesSince should be strictly after, so exact match should NOT be included
	result := conv.MessagesSince(exactTime)
	if len(result) != 0 {
		t.Errorf("MessagesSince at exact timestamp: expected 0 (strictly after), got %d", len(result))
	}
}

// TestMessagesWithMetadata_Basic tests basic filtering by metadata key.
func TestMessagesWithMetadata_Basic(t *testing.T) {
	conv := NewConversation("test")

	msg1 := NewMessage(RoleUser, "message 1")
	msg1.WithMetadata("important", true)
	conv.Add(msg1)

	msg2 := NewMessage(RoleUser, "message 2")
	msg2.WithMetadata("other_key", "value")
	conv.Add(msg2)

	msg3 := NewMessage(RoleUser, "message 3")
	msg3.WithMetadata("important", false)
	conv.Add(msg3)

	result := conv.MessagesWithMetadata("important")
	if len(result) != 2 {
		t.Errorf("MessagesWithMetadata('important'): expected 2, got %d", len(result))
	}

	// Verify all returned messages have the key
	for _, msg := range result {
		if _, exists := msg.Metadata["important"]; !exists {
			t.Error("MessagesWithMetadata: returned message without the specified key")
		}
	}
}

// TestMessagesWithMetadata_EmptyConversation tests filtering on empty conversation.
func TestMessagesWithMetadata_EmptyConversation(t *testing.T) {
	conv := NewConversation("empty")

	result := conv.MessagesWithMetadata("any_key")
	if result == nil {
		t.Error("MessagesWithMetadata on empty conversation should return empty slice, not nil")
	}
	if len(result) != 0 {
		t.Errorf("MessagesWithMetadata on empty conversation: expected 0, got %d", len(result))
	}
}

// TestMessagesWithMetadata_NoMatches tests when no messages have the key.
func TestMessagesWithMetadata_NoMatches(t *testing.T) {
	conv := NewConversation("test")

	msg := NewMessage(RoleUser, "message")
	msg.WithMetadata("other_key", "value")
	conv.Add(msg)

	result := conv.MessagesWithMetadata("nonexistent_key")
	if len(result) != 0 {
		t.Errorf("MessagesWithMetadata with no matches: expected 0, got %d", len(result))
	}
}

// TestMessagesWithMetadata_NilMetadata tests handling of nil Metadata maps.
func TestMessagesWithMetadata_NilMetadata(t *testing.T) {
	conv := NewConversation("test")

	// Create a message and explicitly set Metadata to nil
	msg1 := &Message{
		ID:        "msg-1",
		Role:      RoleUser,
		Content:   "message with nil metadata",
		Timestamp: time.Now(),
		Metadata:  nil, // Explicitly nil
	}
	conv.Add(msg1)

	msg2 := NewMessage(RoleUser, "message with metadata")
	msg2.WithMetadata("key", "value")
	conv.Add(msg2)

	// Should not panic and should only return the message with metadata
	result := conv.MessagesWithMetadata("key")
	if len(result) != 1 {
		t.Errorf("MessagesWithMetadata with nil Metadata: expected 1, got %d", len(result))
	}
}

// TestMessagesWithMetadata_EmptyKey tests filtering with empty key string.
func TestMessagesWithMetadata_EmptyKey(t *testing.T) {
	conv := NewConversation("test")

	msg1 := NewMessage(RoleUser, "message 1")
	msg1.WithMetadata("", "empty key value")
	conv.Add(msg1)

	msg2 := NewMessage(RoleUser, "message 2")
	msg2.WithMetadata("normal_key", "value")
	conv.Add(msg2)

	result := conv.MessagesWithMetadata("")
	if len(result) != 1 {
		t.Errorf("MessagesWithMetadata(''): expected 1, got %d", len(result))
	}
}

// TestMessagesWithMetadata_NilValue tests that nil values in metadata are matched.
func TestMessagesWithMetadata_NilValue(t *testing.T) {
	conv := NewConversation("test")

	msg := NewMessage(RoleUser, "message")
	msg.WithMetadata("key_with_nil", nil)
	conv.Add(msg)

	result := conv.MessagesWithMetadata("key_with_nil")
	if len(result) != 1 {
		t.Errorf("MessagesWithMetadata with nil value: expected 1, got %d", len(result))
	}
}

// TestConcurrentAccess tests thread safety of filter methods.
func TestConcurrentAccess(t *testing.T) {
	conv := NewConversation("concurrent-test")

	// Add some initial messages
	for i := 0; i < 10; i++ {
		msg := NewAgentMessage(RoleAssistant, "message", "agent-1", "Agent")
		msg.WithMetadata("key", i)
		conv.Add(msg)
	}

	var wg sync.WaitGroup
	errChan := make(chan error, 100)

	// Concurrent readers
	for i := 0; i < 10; i++ {
		wg.Add(4)

		go func() {
			defer wg.Done()
			_ = conv.MessagesByRole(RoleAssistant)
		}()

		go func() {
			defer wg.Done()
			_ = conv.MessagesByAgent("agent-1")
		}()

		go func() {
			defer wg.Done()
			_ = conv.MessagesSince(time.Now().Add(-1 * time.Hour))
		}()

		go func() {
			defer wg.Done()
			_ = conv.MessagesWithMetadata("key")
		}()
	}

	// Concurrent writer
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			msg := NewMessage(RoleUser, "new message")
			msg.WithMetadata("new_key", i)
			conv.Add(msg)
		}
	}()

	wg.Wait()
	close(errChan)

	// Check for any errors
	for err := range errChan {
		t.Error(err)
	}
}

// TestFilterMethodsReturnCopy verifies that filter methods return copies, not references.
func TestFilterMethodsReturnCopy(t *testing.T) {
	conv := NewConversation("test")
	conv.Add(NewMessage(RoleUser, "message 1"))
	conv.Add(NewMessage(RoleUser, "message 2"))

	result1 := conv.MessagesByRole(RoleUser)
	result2 := conv.MessagesByRole(RoleUser)

	// Modifying one result shouldn't affect the other
	if len(result1) < 2 {
		t.Fatal("Expected at least 2 messages")
	}

	// Change the first element of result1
	result1[0] = NewMessage(RoleAssistant, "modified")

	// result2 should still have the original message
	if result2[0].Role != RoleUser {
		t.Error("Filter methods should return slice copies, not shared references")
	}
}

// TestMessagesByRole_PreservesOrder tests that messages are returned in order.
func TestMessagesByRole_PreservesOrder(t *testing.T) {
	conv := NewConversation("test")

	msg1 := NewMessage(RoleUser, "first")
	msg2 := NewMessage(RoleAssistant, "second")
	msg3 := NewMessage(RoleUser, "third")
	msg4 := NewMessage(RoleUser, "fourth")

	conv.Add(msg1)
	conv.Add(msg2)
	conv.Add(msg3)
	conv.Add(msg4)

	result := conv.MessagesByRole(RoleUser)
	if len(result) != 3 {
		t.Fatalf("Expected 3 messages, got %d", len(result))
	}

	if result[0].Content != "first" || result[1].Content != "third" || result[2].Content != "fourth" {
		t.Error("MessagesByRole should preserve message order")
	}
}
