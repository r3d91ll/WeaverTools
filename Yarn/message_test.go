package yarn

import (
	"testing"
	"time"
)

// TestMessageRoleIsValid tests the IsValid method for MessageRole.
func TestMessageRoleIsValid(t *testing.T) {
	tests := []struct {
		name  string
		role  MessageRole
		valid bool
	}{
		{"system role is valid", RoleSystem, true},
		{"user role is valid", RoleUser, true},
		{"assistant role is valid", RoleAssistant, true},
		{"tool role is valid", RoleTool, true},
		{"empty role is invalid", MessageRole(""), false},
		{"unknown role is invalid", MessageRole("unknown"), false},
		{"typo role is invalid", MessageRole("asistant"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.role.IsValid(); got != tt.valid {
				t.Errorf("MessageRole(%q).IsValid() = %v, want %v", tt.role, got, tt.valid)
			}
		})
	}
}

// TestMessageValidate tests the Validate method for Message.
func TestMessageValidate(t *testing.T) {
	validTimestamp := time.Now()

	tests := []struct {
		name        string
		message     Message
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name: "valid user message",
			message: Message{
				ID:        "msg-1",
				Role:      RoleUser,
				Content:   "Hello, world!",
				Timestamp: validTimestamp,
			},
			wantErr: false,
		},
		{
			name: "valid system message",
			message: Message{
				ID:        "msg-2",
				Role:      RoleSystem,
				Content:   "You are a helpful assistant.",
				Timestamp: validTimestamp,
			},
			wantErr: false,
		},
		{
			name: "valid assistant message",
			message: Message{
				ID:        "msg-3",
				Role:      RoleAssistant,
				Content:   "Hello! How can I help you?",
				Timestamp: validTimestamp,
			},
			wantErr: false,
		},
		{
			name: "valid tool message with tool_call_id",
			message: Message{
				ID:         "msg-4",
				Role:       RoleTool,
				Content:    "Tool response",
				Timestamp:  validTimestamp,
				ToolCallID: "call-123",
				ToolName:   "get_weather",
			},
			wantErr: false,
		},
		{
			name: "valid tool message with empty content but has tool_call_id",
			message: Message{
				ID:         "msg-5",
				Role:       RoleTool,
				Content:    "",
				Timestamp:  validTimestamp,
				ToolCallID: "call-456",
			},
			wantErr: false,
		},
		{
			name: "missing id",
			message: Message{
				ID:        "",
				Role:      RoleUser,
				Content:   "Hello",
				Timestamp: validTimestamp,
			},
			wantErr:     true,
			wantField:   "id",
			wantMessage: "id is required",
		},
		{
			name: "invalid role",
			message: Message{
				ID:        "msg-6",
				Role:      MessageRole("invalid"),
				Content:   "Hello",
				Timestamp: validTimestamp,
			},
			wantErr:     true,
			wantField:   "role",
			wantMessage: "invalid role",
		},
		{
			name: "empty role",
			message: Message{
				ID:        "msg-7",
				Role:      MessageRole(""),
				Content:   "Hello",
				Timestamp: validTimestamp,
			},
			wantErr:     true,
			wantField:   "role",
			wantMessage: "invalid role",
		},
		{
			name: "zero timestamp",
			message: Message{
				ID:        "msg-8",
				Role:      RoleUser,
				Content:   "Hello",
				Timestamp: time.Time{},
			},
			wantErr:     true,
			wantField:   "timestamp",
			wantMessage: "timestamp is required",
		},
		{
			name: "empty content for non-tool message",
			message: Message{
				ID:        "msg-9",
				Role:      RoleUser,
				Content:   "",
				Timestamp: validTimestamp,
			},
			wantErr:     true,
			wantField:   "content",
			wantMessage: "content is required for non-tool messages",
		},
		{
			name: "empty content for assistant message",
			message: Message{
				ID:        "msg-10",
				Role:      RoleAssistant,
				Content:   "",
				Timestamp: validTimestamp,
			},
			wantErr:     true,
			wantField:   "content",
			wantMessage: "content is required for non-tool messages",
		},
		{
			name: "empty content for system message",
			message: Message{
				ID:        "msg-11",
				Role:      RoleSystem,
				Content:   "",
				Timestamp: validTimestamp,
			},
			wantErr:     true,
			wantField:   "content",
			wantMessage: "content is required for non-tool messages",
		},
		{
			name: "tool message without tool_call_id",
			message: Message{
				ID:         "msg-12",
				Role:       RoleTool,
				Content:    "Tool response",
				Timestamp:  validTimestamp,
				ToolCallID: "",
			},
			wantErr:     true,
			wantField:   "tool_call_id",
			wantMessage: "tool_call_id is required for tool messages",
		},
		{
			name: "message with optional fields",
			message: Message{
				ID:        "msg-13",
				Role:      RoleAssistant,
				Content:   "Response with metadata",
				Timestamp: validTimestamp,
				AgentID:   "agent-1",
				AgentName: "Assistant Agent",
				Metadata:  map[string]any{"key": "value"},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.message.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Message.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Message.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("Message.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("Message.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestHiddenStateValidate tests the Validate method for HiddenState.
func TestHiddenStateValidate(t *testing.T) {
	tests := []struct {
		name        string
		state       *HiddenState
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name:    "nil hidden state is valid",
			state:   nil,
			wantErr: false,
		},
		{
			name: "valid hidden state with minimal fields",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0, 4.0},
				Layer:  0,
			},
			wantErr: false,
		},
		{
			name: "valid hidden state with all fields",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
				Shape:  []int{2, 3},
				Layer:  5,
				DType:  "float32",
			},
			wantErr: false,
		},
		{
			name: "valid hidden state with float16 dtype",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0},
				Layer:  0,
				DType:  "float16",
			},
			wantErr: false,
		},
		{
			name: "valid hidden state with empty dtype",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0},
				Layer:  1,
				DType:  "",
			},
			wantErr: false,
		},
		{
			name: "empty vector",
			state: &HiddenState{
				Vector: []float32{},
				Layer:  0,
			},
			wantErr:     true,
			wantField:   "vector",
			wantMessage: "vector is required",
		},
		{
			name: "nil vector",
			state: &HiddenState{
				Vector: nil,
				Layer:  0,
			},
			wantErr:     true,
			wantField:   "vector",
			wantMessage: "vector is required",
		},
		{
			name: "negative layer",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0},
				Layer:  -1,
			},
			wantErr:     true,
			wantField:   "layer",
			wantMessage: "layer must be non-negative",
		},
		{
			name: "zero shape dimension",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0},
				Shape:  []int{0, 3},
				Layer:  0,
			},
			wantErr:     true,
			wantField:   "shape",
			wantMessage: "shape dimensions must be positive",
		},
		{
			name: "negative shape dimension",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0},
				Shape:  []int{-1, 3},
				Layer:  0,
			},
			wantErr:     true,
			wantField:   "shape",
			wantMessage: "shape dimensions must be positive",
		},
		{
			name: "shape inconsistent with vector length",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0, 4.0},
				Shape:  []int{2, 3}, // 6 != 4
				Layer:  0,
			},
			wantErr:     true,
			wantField:   "shape",
			wantMessage: "shape is inconsistent with vector length",
		},
		{
			name: "invalid dtype",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0},
				Layer:  0,
				DType:  "float64",
			},
			wantErr:     true,
			wantField:   "dtype",
			wantMessage: "dtype must be 'float32' or 'float16'",
		},
		{
			name: "invalid dtype - typo",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0},
				Layer:  0,
				DType:  "flot32",
			},
			wantErr:     true,
			wantField:   "dtype",
			wantMessage: "dtype must be 'float32' or 'float16'",
		},
		{
			name: "valid shape with 3 dimensions",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
				Shape:  []int{2, 2, 3},
				Layer:  2,
				DType:  "float32",
			},
			wantErr: false,
		},
		{
			name: "valid shape with single dimension",
			state: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0, 4.0},
				Shape:  []int{4},
				Layer:  0,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.state.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("HiddenState.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("HiddenState.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("HiddenState.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("HiddenState.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestMessageValidateWithHiddenState tests that Message.Validate does not
// currently cascade to HiddenState validation.
// Note: This test documents current behavior. If cascade validation is added,
// update these tests accordingly.
func TestMessageValidateWithHiddenState(t *testing.T) {
	validTimestamp := time.Now()

	t.Run("message with valid hidden state", func(t *testing.T) {
		msg := Message{
			ID:        "msg-hs-1",
			Role:      RoleAssistant,
			Content:   "Response with hidden state",
			Timestamp: validTimestamp,
			HiddenState: &HiddenState{
				Vector: []float32{1.0, 2.0, 3.0, 4.0},
				Layer:  0,
				DType:  "float32",
			},
		}
		if err := msg.Validate(); err != nil {
			t.Errorf("Message.Validate() unexpected error: %v", err)
		}
	})

	t.Run("message with nil hidden state", func(t *testing.T) {
		msg := Message{
			ID:          "msg-hs-2",
			Role:        RoleUser,
			Content:     "Message without hidden state",
			Timestamp:   validTimestamp,
			HiddenState: nil,
		}
		if err := msg.Validate(); err != nil {
			t.Errorf("Message.Validate() unexpected error: %v", err)
		}
	})
}

// TestValidationErrorString tests the Error() method of ValidationError.
func TestValidationErrorString(t *testing.T) {
	tests := []struct {
		name    string
		err     ValidationError
		wantStr string
	}{
		{
			name:    "standard error format",
			err:     ValidationError{Field: "id", Message: "id is required"},
			wantStr: "id: id is required",
		},
		{
			name:    "complex message",
			err:     ValidationError{Field: "tool_call_id", Message: "tool_call_id is required for tool messages"},
			wantStr: "tool_call_id: tool_call_id is required for tool messages",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.wantStr {
				t.Errorf("ValidationError.Error() = %q, want %q", got, tt.wantStr)
			}
		})
	}
}

// TestNewMessageValidation tests that NewMessage creates valid messages.
func TestNewMessageValidation(t *testing.T) {
	tests := []struct {
		name    string
		role    MessageRole
		content string
		wantErr bool
	}{
		{
			name:    "NewMessage creates valid user message",
			role:    RoleUser,
			content: "Hello",
			wantErr: false,
		},
		{
			name:    "NewMessage creates valid assistant message",
			role:    RoleAssistant,
			content: "Hi there!",
			wantErr: false,
		},
		{
			name:    "NewMessage creates valid system message",
			role:    RoleSystem,
			content: "You are helpful.",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := NewMessage(tt.role, tt.content)
			err := msg.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("NewMessage() created message with Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestNewAgentMessageValidation tests that NewAgentMessage creates valid messages.
func TestNewAgentMessageValidation(t *testing.T) {
	msg := NewAgentMessage(RoleAssistant, "Hello from agent", "agent-123", "Test Agent")
	if err := msg.Validate(); err != nil {
		t.Errorf("NewAgentMessage() created invalid message: %v", err)
	}
	if msg.AgentID != "agent-123" {
		t.Errorf("NewAgentMessage() AgentID = %q, want %q", msg.AgentID, "agent-123")
	}
	if msg.AgentName != "Test Agent" {
		t.Errorf("NewAgentMessage() AgentName = %q, want %q", msg.AgentName, "Test Agent")
	}
}
