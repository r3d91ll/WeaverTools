package yarn

import (
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

	for _, tt := range tests {
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
