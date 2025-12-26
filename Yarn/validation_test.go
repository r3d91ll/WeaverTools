package yarn

import (
	"testing"
	"time"
)

// ============================================================================
// Integration Tests for Cascade Validation
// These tests verify that validation properly cascades through nested types:
// - Session -> Conversations -> Messages -> HiddenState
// - Session -> Measurements -> HiddenState
// - Conversation -> Messages -> HiddenState
// ============================================================================

// TestCascadeSessionToConversationToMessage tests the full cascade from Session to Message.
func TestCascadeSessionToConversationToMessage(t *testing.T) {
	now := time.Now()

	t.Run("valid session with valid nested structure", func(t *testing.T) {
		session := &Session{
			ID:        "session-cascade-1",
			Name:      "Cascade Test Session",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:   "conv-1",
					Name: "Valid Conversation",
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
							Content:   "Hi there! How can I help?",
							Timestamp: now.Add(2 * time.Second),
						},
					},
					CreatedAt: now,
					UpdatedAt: now.Add(2 * time.Second),
				},
			},
			Measurements: []*Measurement{},
		}

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})

	t.Run("invalid message ID deep in session causes session validation failure", func(t *testing.T) {
		session := &Session{
			ID:        "session-cascade-2",
			Name:      "Cascade Test Session",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:   "conv-1",
					Name: "First Conversation",
					Messages: []*Message{
						{
							ID:        "msg-1",
							Role:      RoleUser,
							Content:   "Valid message",
							Timestamp: now,
						},
					},
					CreatedAt: now,
					UpdatedAt: now,
				},
				{
					ID:   "conv-2",
					Name: "Second Conversation",
					Messages: []*Message{
						{
							ID:        "msg-2",
							Role:      RoleAssistant,
							Content:   "Valid message",
							Timestamp: now,
						},
						{
							ID:        "", // Invalid: empty ID
							Role:      RoleUser,
							Content:   "Invalid message",
							Timestamp: now.Add(time.Second),
						},
					},
					CreatedAt: now,
					UpdatedAt: now,
				},
			},
			Measurements: []*Measurement{},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for invalid nested message, got nil")
		}
		if err.Field != "conversations[1].messages[1].id" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "conversations[1].messages[1].id")
		}
		if err.Message != "id is required" {
			t.Errorf("Session.Validate() error message = %q, want %q", err.Message, "id is required")
		}
	})

	t.Run("invalid message role deep in session causes session validation failure", func(t *testing.T) {
		session := &Session{
			ID:        "session-cascade-3",
			Name:      "Cascade Test Session",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:   "conv-1",
					Name: "Conversation with Invalid Role",
					Messages: []*Message{
						{
							ID:        "msg-1",
							Role:      MessageRole("invalid_role"),
							Content:   "Message with bad role",
							Timestamp: now,
						},
					},
					CreatedAt: now,
					UpdatedAt: now,
				},
			},
			Measurements: []*Measurement{},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for invalid role, got nil")
		}
		if err.Field != "conversations[0].messages[0].role" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "conversations[0].messages[0].role")
		}
	})

	t.Run("empty content for non-tool message deep in session", func(t *testing.T) {
		session := &Session{
			ID:        "session-cascade-4",
			Name:      "Cascade Test Session",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:   "conv-1",
					Name: "Conversation with Empty Content",
					Messages: []*Message{
						{
							ID:        "msg-1",
							Role:      RoleUser,
							Content:   "Hello",
							Timestamp: now,
						},
						{
							ID:        "msg-2",
							Role:      RoleAssistant,
							Content:   "", // Invalid: empty content for assistant
							Timestamp: now.Add(time.Second),
						},
					},
					CreatedAt: now,
					UpdatedAt: now,
				},
			},
			Measurements: []*Measurement{},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for empty content, got nil")
		}
		if err.Field != "conversations[0].messages[1].content" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "conversations[0].messages[1].content")
		}
		if err.Message != "content is required for non-tool messages" {
			t.Errorf("Session.Validate() error message = %q, want %q", err.Message, "content is required for non-tool messages")
		}
	})

	t.Run("tool message without tool_call_id deep in session", func(t *testing.T) {
		session := &Session{
			ID:        "session-cascade-5",
			Name:      "Cascade Test Session",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:   "conv-1",
					Name: "Conversation with Invalid Tool Message",
					Messages: []*Message{
						{
							ID:         "msg-1",
							Role:       RoleTool,
							Content:    "Tool response",
							Timestamp:  now,
							ToolCallID: "", // Invalid: tool message without tool_call_id
						},
					},
					CreatedAt: now,
					UpdatedAt: now,
				},
			},
			Measurements: []*Measurement{},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for missing tool_call_id, got nil")
		}
		if err.Field != "conversations[0].messages[0].tool_call_id" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "conversations[0].messages[0].tool_call_id")
		}
	})

	t.Run("zero message timestamp deep in session", func(t *testing.T) {
		session := &Session{
			ID:        "session-cascade-6",
			Name:      "Cascade Test Session",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:   "conv-1",
					Name: "Conversation with Zero Timestamp",
					Messages: []*Message{
						{
							ID:        "msg-1",
							Role:      RoleUser,
							Content:   "Message with zero timestamp",
							Timestamp: time.Time{}, // Invalid: zero timestamp
						},
					},
					CreatedAt: now,
					UpdatedAt: now,
				},
			},
			Measurements: []*Measurement{},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for zero timestamp, got nil")
		}
		if err.Field != "conversations[0].messages[0].timestamp" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "conversations[0].messages[0].timestamp")
		}
	})
}

// TestCascadeSessionToMeasurement tests the full cascade from Session to Measurement.
func TestCascadeSessionToMeasurement(t *testing.T) {
	now := time.Now()

	t.Run("valid session with valid measurements", func(t *testing.T) {
		session := &Session{
			ID:            "session-meas-1",
			Name:          "Measurement Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					DEff:       100,
					Beta:       1.8,
					Alignment:  0.75,
					BetaStatus: BetaOptimal,
				},
				{
					ID:         "meas-2",
					Timestamp:  now.Add(time.Second),
					SenderID:   "agent-1",
					ReceiverID: "agent-2",
					TurnNumber: 1,
					DEff:       150,
					Beta:       2.0,
					Alignment:  0.85,
					BetaStatus: BetaMonitor,
				},
			},
		}

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})

	t.Run("invalid measurement ID in session causes session validation failure", func(t *testing.T) {
		session := &Session{
			ID:            "session-meas-2",
			Name:          "Measurement Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
				},
				{
					ID:         "", // Invalid: empty ID
					Timestamp:  now.Add(time.Second),
					SenderID:   "agent-1",
					TurnNumber: 1,
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for invalid measurement ID, got nil")
		}
		if err.Field != "measurements[1].id" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[1].id")
		}
		if err.Message != "id is required" {
			t.Errorf("Session.Validate() error message = %q, want %q", err.Message, "id is required")
		}
	})

	t.Run("measurement with zero timestamp in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-meas-3",
			Name:          "Measurement Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  time.Time{}, // Invalid: zero timestamp
					SenderID:   "agent-1",
					TurnNumber: 0,
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for zero timestamp, got nil")
		}
		if err.Field != "measurements[0].timestamp" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].timestamp")
		}
	})

	t.Run("measurement with negative turn number in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-meas-4",
			Name:          "Measurement Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: -1, // Invalid: negative turn number
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for negative turn number, got nil")
		}
		if err.Field != "measurements[0].turn_number" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].turn_number")
		}
	})

	t.Run("measurement with no participants in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-meas-5",
			Name:          "Measurement Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "", // No sender
					ReceiverID: "", // No receiver
					TurnNumber: 0,
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for no participants, got nil")
		}
		if err.Field != "measurements[0].sender_id" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].sender_id")
		}
	})

	t.Run("measurement with invalid metric values in session", func(t *testing.T) {
		tests := []struct {
			name      string
			meas      *Measurement
			wantField string
		}{
			{
				name: "negative d_eff",
				meas: &Measurement{
					ID:         "meas-neg-deff",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					DEff:       -1,
				},
				wantField: "measurements[0].d_eff",
			},
			{
				name: "negative beta",
				meas: &Measurement{
					ID:         "meas-neg-beta",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					Beta:       -0.5,
				},
				wantField: "measurements[0].beta",
			},
			{
				name: "alignment below -1",
				meas: &Measurement{
					ID:         "meas-low-align",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					Alignment:  -1.5,
				},
				wantField: "measurements[0].alignment",
			},
			{
				name: "alignment above 1",
				meas: &Measurement{
					ID:         "meas-high-align",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					Alignment:  1.5,
				},
				wantField: "measurements[0].alignment",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				session := &Session{
					ID:            "session-meas-invalid",
					Name:          "Measurement Test Session",
					StartedAt:     now,
					Conversations: []*Conversation{},
					Measurements:  []*Measurement{tt.meas},
				}

				err := session.Validate()
				if err == nil {
					t.Fatalf("Session.Validate() expected error for %s, got nil", tt.name)
				}
				if err.Field != tt.wantField {
					t.Errorf("Session.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
			})
		}
	})

	t.Run("measurement with invalid beta status in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-meas-6",
			Name:          "Measurement Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					BetaStatus: BetaStatus("invalid_status"),
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for invalid beta status, got nil")
		}
		if err.Field != "measurements[0].beta_status" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].beta_status")
		}
	})
}

// TestCascadeSessionToMeasurementToHiddenState tests the full cascade to HiddenState.
func TestCascadeSessionToMeasurementToHiddenState(t *testing.T) {
	now := time.Now()

	t.Run("valid session with measurement containing valid hidden states", func(t *testing.T) {
		session := &Session{
			ID:            "session-hs-1",
			Name:          "Hidden State Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					ReceiverID: "agent-2",
					TurnNumber: 0,
					SenderHidden: &HiddenState{
						Vector: []float32{1.0, 2.0, 3.0, 4.0},
						Layer:  0,
						DType:  "float32",
					},
					ReceiverHidden: &HiddenState{
						Vector: []float32{5.0, 6.0, 7.0, 8.0},
						Layer:  1,
						DType:  "float16",
					},
				},
			},
		}

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})

	t.Run("invalid sender hidden state empty vector deep in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-hs-2",
			Name:          "Hidden State Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					SenderHidden: &HiddenState{
						Vector: []float32{}, // Invalid: empty vector
						Layer:  0,
					},
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for empty vector, got nil")
		}
		if err.Field != "measurements[0].sender_hidden.vector" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].sender_hidden.vector")
		}
		if err.Message != "vector is required" {
			t.Errorf("Session.Validate() error message = %q, want %q", err.Message, "vector is required")
		}
	})

	t.Run("invalid sender hidden state negative layer deep in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-hs-3",
			Name:          "Hidden State Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					SenderHidden: &HiddenState{
						Vector: []float32{1.0, 2.0, 3.0},
						Layer:  -5, // Invalid: negative layer
					},
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for negative layer, got nil")
		}
		if err.Field != "measurements[0].sender_hidden.layer" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].sender_hidden.layer")
		}
	})

	t.Run("invalid sender hidden state invalid dtype deep in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-hs-4",
			Name:          "Hidden State Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					SenderHidden: &HiddenState{
						Vector: []float32{1.0, 2.0, 3.0},
						Layer:  0,
						DType:  "float64", // Invalid: unsupported dtype
					},
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for invalid dtype, got nil")
		}
		if err.Field != "measurements[0].sender_hidden.dtype" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].sender_hidden.dtype")
		}
	})

	t.Run("invalid sender hidden state inconsistent shape deep in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-hs-5",
			Name:          "Hidden State Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					SenderHidden: &HiddenState{
						Vector: []float32{1.0, 2.0, 3.0, 4.0},
						Shape:  []int{2, 3}, // 6 != 4, inconsistent with vector
						Layer:  0,
					},
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for inconsistent shape, got nil")
		}
		if err.Field != "measurements[0].sender_hidden.shape" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].sender_hidden.shape")
		}
	})

	t.Run("invalid receiver hidden state deep in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-hs-6",
			Name:          "Hidden State Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					ReceiverID: "agent-2",
					TurnNumber: 0,
					ReceiverHidden: &HiddenState{
						Vector: []float32{}, // Invalid: empty vector
						Layer:  0,
					},
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for invalid receiver hidden state, got nil")
		}
		if err.Field != "measurements[0].receiver_hidden.vector" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].receiver_hidden.vector")
		}
	})

	t.Run("sender hidden state error takes precedence over receiver", func(t *testing.T) {
		session := &Session{
			ID:            "session-hs-7",
			Name:          "Hidden State Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					ReceiverID: "agent-2",
					TurnNumber: 0,
					SenderHidden: &HiddenState{
						Vector: []float32{}, // Invalid: empty vector (checked first)
						Layer:  0,
					},
					ReceiverHidden: &HiddenState{
						Vector: []float32{}, // Also invalid
						Layer:  0,
					},
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error, got nil")
		}
		// Sender is validated before receiver, so sender error should be first
		if err.Field != "measurements[0].sender_hidden.vector" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[0].sender_hidden.vector")
		}
	})
}

// TestCascadeConversationToMessage tests cascade from Conversation to Message.
func TestCascadeConversationToMessage(t *testing.T) {
	now := time.Now()

	t.Run("valid conversation with multiple message types", func(t *testing.T) {
		conv := &Conversation{
			ID:   "conv-cascade-1",
			Name: "Multi-Type Message Conversation",
			Messages: []*Message{
				{
					ID:        "msg-1",
					Role:      RoleSystem,
					Content:   "System prompt",
					Timestamp: now,
				},
				{
					ID:        "msg-2",
					Role:      RoleUser,
					Content:   "User input",
					Timestamp: now.Add(time.Second),
				},
				{
					ID:        "msg-3",
					Role:      RoleAssistant,
					Content:   "Assistant response",
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

	t.Run("invalid message in conversation at various indices", func(t *testing.T) {
		tests := []struct {
			name       string
			index      int
			invalidMsg *Message
			wantField  string
		}{
			{
				name:  "first message invalid",
				index: 0,
				invalidMsg: &Message{
					ID:        "",
					Role:      RoleUser,
					Content:   "Hello",
					Timestamp: now,
				},
				wantField: "messages[0].id",
			},
			{
				name:  "middle message invalid",
				index: 1,
				invalidMsg: &Message{
					ID:        "msg-middle",
					Role:      MessageRole("bad"),
					Content:   "Hello",
					Timestamp: now,
				},
				wantField: "messages[1].role",
			},
			{
				name:  "last message invalid",
				index: 2,
				invalidMsg: &Message{
					ID:        "msg-last",
					Role:      RoleUser,
					Content:   "Hello",
					Timestamp: time.Time{},
				},
				wantField: "messages[2].timestamp",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				messages := make([]*Message, 3)
				for i := range messages {
					if i == tt.index {
						messages[i] = tt.invalidMsg
					} else {
						messages[i] = &Message{
							ID:        "msg-valid-" + string(rune('0'+i)),
							Role:      RoleUser,
							Content:   "Valid content",
							Timestamp: now,
						}
					}
				}

				conv := &Conversation{
					ID:        "conv-test",
					Name:      "Test Conversation",
					Messages:  messages,
					CreatedAt: now,
					UpdatedAt: now,
				}

				err := conv.Validate()
				if err == nil {
					t.Fatalf("Conversation.Validate() expected error for %s, got nil", tt.name)
				}
				if err.Field != tt.wantField {
					t.Errorf("Conversation.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
			})
		}
	})

	t.Run("nil message in conversation", func(t *testing.T) {
		conv := &Conversation{
			ID:   "conv-nil-msg",
			Name: "Conversation with Nil Message",
			Messages: []*Message{
				{
					ID:        "msg-1",
					Role:      RoleUser,
					Content:   "Valid",
					Timestamp: now,
				},
				nil, // Invalid: nil message
				{
					ID:        "msg-3",
					Role:      RoleUser,
					Content:   "Valid",
					Timestamp: now,
				},
			},
			CreatedAt: now,
			UpdatedAt: now,
		}

		err := conv.Validate()
		if err == nil {
			t.Fatal("Conversation.Validate() expected error for nil message, got nil")
		}
		if err.Field != "messages" {
			t.Errorf("Conversation.Validate() error field = %q, want %q", err.Field, "messages")
		}
		if err.Message != "message at index 1 is nil" {
			t.Errorf("Conversation.Validate() error message = %q, want %q", err.Message, "message at index 1 is nil")
		}
	})
}

// TestCascadeComplexSession tests a complex session with multiple conversations and measurements.
func TestCascadeComplexSession(t *testing.T) {
	now := time.Now()

	t.Run("valid complex session with all types of nested data", func(t *testing.T) {
		session := &Session{
			ID:        "session-complex-1",
			Name:      "Complex Integration Test Session",
			StartedAt: now,
			Config: SessionConfig{
				MeasurementMode: MeasureActive,
				AutoExport:      true,
				ExportPath:      "./experiments",
			},
			Conversations: []*Conversation{
				{
					ID:   "conv-1",
					Name: "First Conversation",
					Messages: []*Message{
						{
							ID:        "msg-1-1",
							Role:      RoleSystem,
							Content:   "You are helpful.",
							Timestamp: now,
						},
						{
							ID:        "msg-1-2",
							Role:      RoleUser,
							Content:   "Hello!",
							Timestamp: now.Add(time.Second),
						},
					},
					CreatedAt: now,
					UpdatedAt: now.Add(time.Second),
				},
				{
					ID:   "conv-2",
					Name: "Second Conversation",
					Messages: []*Message{
						{
							ID:         "msg-2-1",
							Role:       RoleTool,
							Content:    "Tool output",
							Timestamp:  now.Add(2 * time.Second),
							ToolCallID: "call-1",
							ToolName:   "search",
						},
					},
					CreatedAt: now,
					UpdatedAt: now.Add(2 * time.Second),
				},
			},
			Measurements: []*Measurement{
				{
					ID:         "meas-1",
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
					DEff:       100,
					Beta:       1.8,
					Alignment:  0.75,
					BetaStatus: BetaOptimal,
					SenderHidden: &HiddenState{
						Vector: []float32{1.0, 2.0, 3.0},
						Layer:  0,
						DType:  "float32",
					},
				},
				{
					ID:         "meas-2",
					Timestamp:  now.Add(time.Second),
					SenderID:   "agent-1",
					ReceiverID: "agent-2",
					TurnNumber: 1,
					DEff:       150,
					Beta:       2.1,
					Alignment:  0.80,
					BetaStatus: BetaMonitor,
					SenderHidden: &HiddenState{
						Vector: []float32{4.0, 5.0, 6.0},
						Shape:  []int{3},
						Layer:  1,
						DType:  "float16",
					},
					ReceiverHidden: &HiddenState{
						Vector: []float32{7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
						Shape:  []int{2, 3},
						Layer:  2,
						DType:  "float32",
					},
				},
			},
		}

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})

	t.Run("first error in session takes precedence", func(t *testing.T) {
		// Session with multiple validation errors - should return the first one
		session := &Session{
			ID:        "", // First error - empty ID
			Name:      "", // Second error - empty name
			StartedAt: now,
			Config: SessionConfig{
				MeasurementMode: MeasurementMode("invalid"), // Third error
			},
			Conversations: []*Conversation{},
			Measurements:  []*Measurement{},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error, got nil")
		}
		// ID is validated first
		if err.Field != "id" {
			t.Errorf("Session.Validate() should fail on id first, got field = %q", err.Field)
		}
	})

	t.Run("conversation error takes precedence over measurement error", func(t *testing.T) {
		// With both invalid conversation and measurement, conversation is checked first
		session := &Session{
			ID:        "session-precedence",
			Name:      "Precedence Test",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:        "", // Invalid conversation
					Name:      "Invalid",
					Messages:  []*Message{},
					CreatedAt: now,
					UpdatedAt: now,
				},
			},
			Measurements: []*Measurement{
				{
					ID:         "", // Invalid measurement
					Timestamp:  now,
					SenderID:   "agent-1",
					TurnNumber: 0,
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error, got nil")
		}
		// Conversations are validated before measurements
		if err.Field != "conversations[0].id" {
			t.Errorf("Session.Validate() should fail on conversation first, got field = %q", err.Field)
		}
	})
}

// TestCascadeErrorContextPropagation tests that error context is properly propagated.
func TestCascadeErrorContextPropagation(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name         string
		session      *Session
		wantField    string
		wantMessage  string
		description  string
	}{
		{
			name: "deeply nested message error",
			session: &Session{
				ID:        "session-deep",
				Name:      "Deep Nesting Test",
				StartedAt: now,
				Conversations: []*Conversation{
					{
						ID:   "conv-1",
						Name: "Outer Conversation",
						Messages: []*Message{
							{
								ID:        "msg-1",
								Role:      RoleUser,
								Content:   "Valid",
								Timestamp: now,
							},
							{
								ID:        "msg-2",
								Role:      RoleTool,
								Content:   "Tool output",
								Timestamp: now,
								// Missing ToolCallID
							},
						},
						CreatedAt: now,
						UpdatedAt: now,
					},
				},
				Measurements: []*Measurement{},
			},
			wantField:   "conversations[0].messages[1].tool_call_id",
			wantMessage: "tool_call_id is required for tool messages",
			description: "error path should include full nesting context",
		},
		{
			name: "measurement hidden state error",
			session: &Session{
				ID:            "session-hs-error",
				Name:          "Hidden State Error Test",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements: []*Measurement{
					{
						ID:         "meas-1",
						Timestamp:  now,
						SenderID:   "agent-1",
						TurnNumber: 0,
						SenderHidden: &HiddenState{
							Vector: nil, // Invalid
							Layer:  0,
						},
					},
				},
			},
			wantField:   "measurements[0].sender_hidden.vector",
			wantMessage: "vector is required",
			description: "hidden state error should include parent context",
		},
		{
			name: "second measurement error at index 3",
			session: &Session{
				ID:            "session-index",
				Name:          "Index Error Test",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements: []*Measurement{
					{ID: "m-1", Timestamp: now, SenderID: "a", TurnNumber: 0},
					{ID: "m-2", Timestamp: now, SenderID: "a", TurnNumber: 1},
					{ID: "m-3", Timestamp: now, SenderID: "a", TurnNumber: 2},
					{ID: "", Timestamp: now, SenderID: "a", TurnNumber: 3}, // Invalid at index 3
				},
			},
			wantField:   "measurements[3].id",
			wantMessage: "id is required",
			description: "correct index should be reported",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.session.Validate()
			if err == nil {
				t.Fatalf("Session.Validate() expected error (%s), got nil", tt.description)
			}
			if err.Field != tt.wantField {
				t.Errorf("Session.Validate() error field = %q, want %q (%s)", err.Field, tt.wantField, tt.description)
			}
			if err.Message != tt.wantMessage {
				t.Errorf("Session.Validate() error message = %q, want %q (%s)", err.Message, tt.wantMessage, tt.description)
			}
		})
	}
}

// TestCascadeWithFactoryFunctions tests that factory-created objects validate correctly.
func TestCascadeWithFactoryFunctions(t *testing.T) {
	t.Run("factory created session with added conversation validates", func(t *testing.T) {
		session := NewSession("Test Session", "A test session")
		conv := NewConversation("Test Conversation")
		msg := NewMessage(RoleUser, "Hello!")

		conv.Add(msg)
		session.AddConversation(conv)

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})

	t.Run("factory created session with added measurement validates", func(t *testing.T) {
		session := NewSession("Test Session", "A test session")
		meas := NewMeasurement()
		meas.SenderID = "agent-1" // Required field

		session.AddMeasurement(meas)

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})

	t.Run("factory created measurement for turn validates after setting participant", func(t *testing.T) {
		session := NewSession("Test Session", "A test session")
		conv := NewConversation("Test Conversation")
		session.AddConversation(conv)

		meas := NewMeasurementForTurn(session.ID, conv.ID, 0)
		meas.SetSender("agent-1", "Agent One", "assistant", &HiddenState{
			Vector: []float32{1.0, 2.0, 3.0, 4.0},
			Layer:  0,
		})

		session.AddMeasurement(meas)

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})
}

// TestNilHandlingInCascade tests how nil values are handled in cascade validation.
func TestNilHandlingInCascade(t *testing.T) {
	now := time.Now()

	t.Run("nil conversation in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-nil-conv",
			Name:          "Session with Nil Conversation",
			StartedAt:     now,
			Conversations: []*Conversation{nil},
			Measurements:  []*Measurement{},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for nil conversation, got nil")
		}
		if err.Field != "conversations" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "conversations")
		}
		if err.Message != "conversation at index 0 is nil" {
			t.Errorf("Session.Validate() error message = %q, want %q", err.Message, "conversation at index 0 is nil")
		}
	})

	t.Run("nil measurement in session", func(t *testing.T) {
		session := &Session{
			ID:            "session-nil-meas",
			Name:          "Session with Nil Measurement",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements:  []*Measurement{nil},
		}

		err := session.Validate()
		if err == nil {
			t.Fatal("Session.Validate() expected error for nil measurement, got nil")
		}
		if err.Field != "measurements" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements")
		}
		if err.Message != "measurement at index 0 is nil" {
			t.Errorf("Session.Validate() error message = %q, want %q", err.Message, "measurement at index 0 is nil")
		}
	})

	t.Run("nil hidden state in measurement is valid", func(t *testing.T) {
		session := &Session{
			ID:            "session-nil-hs",
			Name:          "Session with Nil Hidden States",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:             "meas-1",
					Timestamp:      now,
					SenderID:       "agent-1",
					TurnNumber:     0,
					SenderHidden:   nil, // Valid: nil is allowed
					ReceiverHidden: nil, // Valid: nil is allowed
				},
			},
		}

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})
}
