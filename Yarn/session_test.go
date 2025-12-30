package yarn

import (
	"testing"
	"time"
)

// TestMeasurementModeIsValid tests the IsValid method for MeasurementMode.
func TestMeasurementModeIsValid(t *testing.T) {
	tests := []struct {
		name  string
		mode  MeasurementMode
		valid bool
	}{
		{"passive mode is valid", MeasurePassive, true},
		{"active mode is valid", MeasureActive, true},
		{"triggered mode is valid", MeasureTriggered, true},
		{"empty mode is invalid", MeasurementMode(""), false},
		{"unknown mode is invalid", MeasurementMode("unknown"), false},
		{"typo mode is invalid", MeasurementMode("pasive"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.mode.IsValid(); got != tt.valid {
				t.Errorf("MeasurementMode(%q).IsValid() = %v, want %v", tt.mode, got, tt.valid)
			}
		})
	}
}

// TestSessionConfigValidate tests the Validate method for SessionConfig.
func TestSessionConfigValidate(t *testing.T) {
	tests := []struct {
		name        string
		config      SessionConfig
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name: "valid config with passive mode",
			config: SessionConfig{
				MeasurementMode: MeasurePassive,
				AutoExport:      true,
				ExportPath:      "./output",
			},
			wantErr: false,
		},
		{
			name: "valid config with active mode",
			config: SessionConfig{
				MeasurementMode: MeasureActive,
			},
			wantErr: false,
		},
		{
			name: "valid config with triggered mode",
			config: SessionConfig{
				MeasurementMode: MeasureTriggered,
			},
			wantErr: false,
		},
		{
			name: "valid config with empty mode (default allowed)",
			config: SessionConfig{
				MeasurementMode: "",
			},
			wantErr: false,
		},
		{
			name: "invalid measurement mode",
			config: SessionConfig{
				MeasurementMode: MeasurementMode("invalid"),
			},
			wantErr:     true,
			wantField:   "measurement_mode",
			wantMessage: "invalid measurement mode",
		},
		{
			name: "invalid measurement mode typo",
			config: SessionConfig{
				MeasurementMode: MeasurementMode("actve"),
			},
			wantErr:     true,
			wantField:   "measurement_mode",
			wantMessage: "invalid measurement mode",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("SessionConfig.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("SessionConfig.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("SessionConfig.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("SessionConfig.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestSessionValidate tests the Validate method for Session.
func TestSessionValidate(t *testing.T) {
	now := time.Now()
	later := now.Add(time.Hour)
	validConversation := &Conversation{
		ID:        "conv-1",
		Name:      "Test Conversation",
		Messages:  []*Message{},
		CreatedAt: now,
		UpdatedAt: now,
	}
	validMeasurement := &Measurement{
		ID:        "meas-1",
		Timestamp: now,
		SenderID:  "sender-1",
	}

	tests := []struct {
		name        string
		session     Session
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name: "valid session with no conversations or measurements",
			session: Session{
				ID:            "session-1",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr: false,
		},
		{
			name: "valid session with conversations",
			session: Session{
				ID:            "session-2",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{validConversation},
				Measurements:  []*Measurement{},
			},
			wantErr: false,
		},
		{
			name: "valid session with measurements",
			session: Session{
				ID:            "session-3",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{validMeasurement},
			},
			wantErr: false,
		},
		{
			name: "valid session with ended_at after started_at",
			session: Session{
				ID:            "session-4",
				Name:          "Test Session",
				StartedAt:     now,
				EndedAt:       &later,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr: false,
		},
		{
			name: "valid session with valid config",
			session: Session{
				ID:        "session-5",
				Name:      "Test Session",
				StartedAt: now,
				Config: SessionConfig{
					MeasurementMode: MeasureActive,
					AutoExport:      true,
					ExportPath:      "./output",
				},
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr: false,
		},
		{
			name: "valid session with description and metadata",
			session: Session{
				ID:            "session-6",
				Name:          "Test Session",
				Description:   "A session for testing",
				StartedAt:     now,
				Metadata:      map[string]any{"key": "value"},
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr: false,
		},
		{
			name: "valid session with empty measurement mode (default)",
			session: Session{
				ID:        "session-7",
				Name:      "Test Session",
				StartedAt: now,
				Config: SessionConfig{
					MeasurementMode: "",
				},
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr: false,
		},
		{
			name: "missing id",
			session: Session{
				ID:            "",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr:     true,
			wantField:   "id",
			wantMessage: "id is required",
		},
		{
			name: "missing name",
			session: Session{
				ID:            "session-8",
				Name:          "",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr:     true,
			wantField:   "name",
			wantMessage: "name is required",
		},
		{
			name: "zero started_at",
			session: Session{
				ID:            "session-9",
				Name:          "Test Session",
				StartedAt:     time.Time{},
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr:     true,
			wantField:   "started_at",
			wantMessage: "started_at is required",
		},
		{
			name: "ended_at before started_at",
			session: Session{
				ID:            "session-10",
				Name:          "Test Session",
				StartedAt:     later,
				EndedAt:       &now,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr:     true,
			wantField:   "ended_at",
			wantMessage: "ended_at must be after started_at",
		},
		{
			name: "ended_at equal to started_at",
			session: Session{
				ID:            "session-11",
				Name:          "Test Session",
				StartedAt:     now,
				EndedAt:       &now,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr:     true,
			wantField:   "ended_at",
			wantMessage: "ended_at must be after started_at",
		},
		{
			name: "invalid measurement mode",
			session: Session{
				ID:        "session-12",
				Name:      "Test Session",
				StartedAt: now,
				Config: SessionConfig{
					MeasurementMode: MeasurementMode("invalid"),
				},
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			},
			wantErr:     true,
			wantField:   "config.measurement_mode",
			wantMessage: "invalid measurement mode",
		},
		{
			name: "nil conversation in conversations slice",
			session: Session{
				ID:            "session-13",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{nil},
				Measurements:  []*Measurement{},
			},
			wantErr:     true,
			wantField:   "conversations",
			wantMessage: "conversation at index 0 is nil",
		},
		{
			name: "nil conversation at index 1",
			session: Session{
				ID:            "session-14",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{validConversation, nil},
				Measurements:  []*Measurement{},
			},
			wantErr:     true,
			wantField:   "conversations",
			wantMessage: "conversation at index 1 is nil",
		},
		{
			name: "invalid conversation - missing id",
			session: Session{
				ID:        "session-15",
				Name:      "Test Session",
				StartedAt: now,
				Conversations: []*Conversation{
					{
						ID:        "",
						Name:      "Invalid Conversation",
						Messages:  []*Message{},
						CreatedAt: now,
						UpdatedAt: now,
					},
				},
				Measurements: []*Measurement{},
			},
			wantErr:     true,
			wantField:   "conversations[0].id",
			wantMessage: "id is required",
		},
		{
			name: "invalid conversation - missing name",
			session: Session{
				ID:        "session-16",
				Name:      "Test Session",
				StartedAt: now,
				Conversations: []*Conversation{
					{
						ID:        "conv-invalid",
						Name:      "",
						Messages:  []*Message{},
						CreatedAt: now,
						UpdatedAt: now,
					},
				},
				Measurements: []*Measurement{},
			},
			wantErr:     true,
			wantField:   "conversations[0].name",
			wantMessage: "name is required",
		},
		{
			name: "invalid conversation at index 2",
			session: Session{
				ID:        "session-17",
				Name:      "Test Session",
				StartedAt: now,
				Conversations: []*Conversation{
					validConversation,
					{
						ID:        "conv-2",
						Name:      "Valid Conv 2",
						Messages:  []*Message{},
						CreatedAt: now,
						UpdatedAt: now,
					},
					{
						ID:        "",
						Name:      "Invalid Conv",
						Messages:  []*Message{},
						CreatedAt: now,
						UpdatedAt: now,
					},
				},
				Measurements: []*Measurement{},
			},
			wantErr:     true,
			wantField:   "conversations[2].id",
			wantMessage: "id is required",
		},
		{
			name: "nil measurement in measurements slice",
			session: Session{
				ID:            "session-18",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{nil},
			},
			wantErr:     true,
			wantField:   "measurements",
			wantMessage: "measurement at index 0 is nil",
		},
		{
			name: "nil measurement at index 1",
			session: Session{
				ID:            "session-19",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{validMeasurement, nil},
			},
			wantErr:     true,
			wantField:   "measurements",
			wantMessage: "measurement at index 1 is nil",
		},
		{
			name: "invalid measurement - missing id",
			session: Session{
				ID:            "session-20",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements: []*Measurement{
					{
						ID:        "",
						Timestamp: now,
					},
				},
			},
			wantErr:     true,
			wantField:   "measurements[0].id",
			wantMessage: "id is required",
		},
		{
			name: "invalid measurement - zero timestamp",
			session: Session{
				ID:            "session-21",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements: []*Measurement{
					{
						ID:        "meas-invalid",
						Timestamp: time.Time{},
						SenderID:  "sender-1",
					},
				},
			},
			wantErr:     true,
			wantField:   "measurements[0].timestamp",
			wantMessage: "timestamp is required",
		},
		{
			name: "invalid measurement at index 2",
			session: Session{
				ID:            "session-22",
				Name:          "Test Session",
				StartedAt:     now,
				Conversations: []*Conversation{},
				Measurements: []*Measurement{
					validMeasurement,
					{
						ID:        "meas-2",
						Timestamp: now,
						SenderID:  "sender-2",
					},
					{
						ID:        "",
						Timestamp: now,
						SenderID:  "sender-3",
					},
				},
			},
			wantErr:     true,
			wantField:   "measurements[2].id",
			wantMessage: "id is required",
		},
	}

	for i := range tests {
		tt := &tests[i]
		t.Run(tt.name, func(t *testing.T) {
			err := tt.session.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Session.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Session.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("Session.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("Session.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestSessionValidateTimeConsistency tests time-related validation rules for Session.
func TestSessionValidateTimeConsistency(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name        string
		startedAt   time.Time
		endedAt     *time.Time
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name:      "nil ended_at is valid",
			startedAt: now,
			endedAt:   nil,
			wantErr:   false,
		},
		{
			name:      "ended_at 1 nanosecond after started_at is valid",
			startedAt: now,
			endedAt:   timePtr(now.Add(time.Nanosecond)),
			wantErr:   false,
		},
		{
			name:      "ended_at 1 hour after started_at is valid",
			startedAt: now,
			endedAt:   timePtr(now.Add(time.Hour)),
			wantErr:   false,
		},
		{
			name:      "ended_at 1 day after started_at is valid",
			startedAt: now,
			endedAt:   timePtr(now.Add(24 * time.Hour)),
			wantErr:   false,
		},
		{
			name:        "ended_at equal to started_at is invalid",
			startedAt:   now,
			endedAt:     timePtr(now),
			wantErr:     true,
			wantField:   "ended_at",
			wantMessage: "ended_at must be after started_at",
		},
		{
			name:        "ended_at 1 nanosecond before started_at is invalid",
			startedAt:   now,
			endedAt:     timePtr(now.Add(-time.Nanosecond)),
			wantErr:     true,
			wantField:   "ended_at",
			wantMessage: "ended_at must be after started_at",
		},
		{
			name:        "ended_at 1 hour before started_at is invalid",
			startedAt:   now,
			endedAt:     timePtr(now.Add(-time.Hour)),
			wantErr:     true,
			wantField:   "ended_at",
			wantMessage: "ended_at must be after started_at",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			session := Session{
				ID:            "session-time-test",
				Name:          "Time Test Session",
				StartedAt:     tt.startedAt,
				EndedAt:       tt.endedAt,
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			}

			err := session.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Session.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Session.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("Session.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("Session.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// timePtr is a helper function that returns a pointer to a time.Time.
func timePtr(t time.Time) *time.Time {
	return &t
}

// TestNewSessionValidation tests that NewSession creates valid sessions.
func TestNewSessionValidation(t *testing.T) {
	tests := []struct {
		name        string
		sessionName string
		description string
		wantErr     bool
	}{
		{
			name:        "NewSession creates valid session",
			sessionName: "Test Session",
			description: "A test session",
			wantErr:     false,
		},
		{
			name:        "NewSession with simple name",
			sessionName: "Session",
			description: "",
			wantErr:     false,
		},
		{
			name:        "NewSession with special characters in name",
			sessionName: "Session #1 - Test!",
			description: "Test description",
			wantErr:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			session := NewSession(tt.sessionName, tt.description)
			err := session.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("NewSession() created session with Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestSessionValidateWithMultipleConversations tests cascade validation with multiple conversations.
func TestSessionValidateWithMultipleConversations(t *testing.T) {
	now := time.Now()

	t.Run("valid session with multiple conversations", func(t *testing.T) {
		session := Session{
			ID:        "session-multi-conv",
			Name:      "Multi-Conversation Session",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:        "conv-1",
					Name:      "Conversation 1",
					Messages:  []*Message{},
					CreatedAt: now,
					UpdatedAt: now,
				},
				{
					ID:        "conv-2",
					Name:      "Conversation 2",
					Messages:  []*Message{},
					CreatedAt: now,
					UpdatedAt: now,
				},
				{
					ID:        "conv-3",
					Name:      "Conversation 3",
					Messages:  []*Message{},
					CreatedAt: now,
					UpdatedAt: now,
				},
			},
			Measurements: []*Measurement{},
		}

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})

	t.Run("invalid conversation in the middle of session", func(t *testing.T) {
		session := Session{
			ID:        "session-middle-invalid",
			Name:      "Session with Invalid Middle Conversation",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:        "conv-1",
					Name:      "Valid Conversation 1",
					Messages:  []*Message{},
					CreatedAt: now,
					UpdatedAt: now,
				},
				{
					ID:        "conv-2",
					Name:      "", // Invalid: empty name
					Messages:  []*Message{},
					CreatedAt: now,
					UpdatedAt: now,
				},
				{
					ID:        "conv-3",
					Name:      "Valid Conversation 3",
					Messages:  []*Message{},
					CreatedAt: now,
					UpdatedAt: now,
				},
			},
			Measurements: []*Measurement{},
		}

		err := session.Validate()
		if err == nil {
			t.Errorf("Session.Validate() expected error, got nil")
			return
		}
		if err.Field != "conversations[1].name" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "conversations[1].name")
		}
	})
}

// TestSessionValidateWithMultipleMeasurements tests cascade validation with multiple measurements.
func TestSessionValidateWithMultipleMeasurements(t *testing.T) {
	now := time.Now()

	t.Run("valid session with multiple measurements", func(t *testing.T) {
		session := Session{
			ID:            "session-multi-meas",
			Name:          "Multi-Measurement Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:        "meas-1",
					Timestamp: now,
					SenderID:  "sender-1",
				},
				{
					ID:        "meas-2",
					Timestamp: now.Add(time.Second),
					SenderID:  "sender-2",
				},
				{
					ID:        "meas-3",
					Timestamp: now.Add(2 * time.Second),
					SenderID:  "sender-3",
				},
			},
		}

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})

	t.Run("invalid measurement in the middle of session", func(t *testing.T) {
		session := Session{
			ID:            "session-middle-invalid-meas",
			Name:          "Session with Invalid Middle Measurement",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements: []*Measurement{
				{
					ID:        "meas-1",
					Timestamp: now,
					SenderID:  "sender-1",
				},
				{
					ID:        "meas-2",
					Timestamp: time.Time{}, // Invalid: zero timestamp
					SenderID:  "sender-2",
				},
				{
					ID:        "meas-3",
					Timestamp: now,
					SenderID:  "sender-3",
				},
			},
		}

		err := session.Validate()
		if err == nil {
			t.Errorf("Session.Validate() expected error, got nil")
			return
		}
		if err.Field != "measurements[1].timestamp" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "measurements[1].timestamp")
		}
	})
}

// TestSessionValidateCascadeToMessageLevel tests that validation cascades from session to conversation to message level.
func TestSessionValidateCascadeToMessageLevel(t *testing.T) {
	now := time.Now()

	t.Run("invalid message in conversation in session", func(t *testing.T) {
		session := Session{
			ID:        "session-cascade",
			Name:      "Cascade Test Session",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:   "conv-1",
					Name: "Conversation with Invalid Message",
					Messages: []*Message{
						{
							ID:        "msg-1",
							Role:      RoleUser,
							Content:   "Valid message",
							Timestamp: now,
						},
						{
							ID:        "", // Invalid: empty ID
							Role:      RoleAssistant,
							Content:   "Invalid message",
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
			t.Errorf("Session.Validate() expected error, got nil")
			return
		}
		if err.Field != "conversations[0].messages[1].id" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "conversations[0].messages[1].id")
		}
		if err.Message != "id is required" {
			t.Errorf("Session.Validate() error message = %q, want %q", err.Message, "id is required")
		}
	})

	t.Run("invalid tool message in conversation", func(t *testing.T) {
		session := Session{
			ID:        "session-cascade-tool",
			Name:      "Cascade Test Session with Tool Message",
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
			t.Errorf("Session.Validate() expected error, got nil")
			return
		}
		if err.Field != "conversations[0].messages[0].tool_call_id" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "conversations[0].messages[0].tool_call_id")
		}
	})
}

// TestSessionAddConversationThenValidate tests that AddConversation maintains valid state.
func TestSessionAddConversationThenValidate(t *testing.T) {
	session := NewSession("Test Session", "A test session")

	// Validate initial state
	if err := session.Validate(); err != nil {
		t.Fatalf("NewSession() created invalid session: %v", err)
	}

	// Add a valid conversation
	conv := NewConversation("Conversation 1")
	session.AddConversation(conv)

	// Should still be valid
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() after AddConversation() unexpected error: %v", err)
	}

	// Add another conversation
	conv2 := NewConversation("Conversation 2")
	session.AddConversation(conv2)

	// Should still be valid
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() after second AddConversation() unexpected error: %v", err)
	}

	// Verify conversation count
	if len(session.Conversations) != 2 {
		t.Errorf("Session.Conversations length = %d, want %d", len(session.Conversations), 2)
	}
}

// TestSessionAddMeasurementThenValidate tests that AddMeasurement maintains valid state.
func TestSessionAddMeasurementThenValidate(t *testing.T) {
	session := NewSession("Test Session", "A test session")

	// Validate initial state
	if err := session.Validate(); err != nil {
		t.Fatalf("NewSession() created invalid session: %v", err)
	}

	// Add a valid measurement (must have sender or receiver)
	meas := NewMeasurement()
	meas.SenderID = "agent-1"
	session.AddMeasurement(meas)

	// Should still be valid
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() after AddMeasurement() unexpected error: %v", err)
	}

	// Add another measurement
	meas2 := NewMeasurement()
	meas2.SenderID = "agent-2"
	session.AddMeasurement(meas2)

	// Should still be valid
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() after second AddMeasurement() unexpected error: %v", err)
	}

	// Verify measurement count
	if len(session.Measurements) != 2 {
		t.Errorf("Session.Measurements length = %d, want %d", len(session.Measurements), 2)
	}

	// Verify session ID is set on measurements
	for i, m := range session.Measurements {
		if m.SessionID != session.ID {
			t.Errorf("Session.Measurements[%d].SessionID = %q, want %q", i, m.SessionID, session.ID)
		}
	}
}

// TestSessionValidatePointerReceiver tests Session.Validate with pointer receiver.
func TestSessionValidatePointerReceiver(t *testing.T) {
	now := time.Now()

	t.Run("pointer to valid session", func(t *testing.T) {
		s := &Session{
			ID:            "session-1",
			Name:          "Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements:  []*Measurement{},
		}
		if err := s.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})

	t.Run("pointer to invalid session", func(t *testing.T) {
		s := &Session{
			ID:            "",
			Name:          "Test Session",
			StartedAt:     now,
			Conversations: []*Conversation{},
			Measurements:  []*Measurement{},
		}
		err := s.Validate()
		if err == nil {
			t.Errorf("Session.Validate() expected error, got nil")
			return
		}
		if err.Field != "id" {
			t.Errorf("Session.Validate() error field = %q, want %q", err.Field, "id")
		}
	})
}

// TestSessionEndThenValidate tests that End() maintains valid state.
func TestSessionEndThenValidate(t *testing.T) {
	session := NewSession("Test Session", "A test session")

	// Validate initial state
	if err := session.Validate(); err != nil {
		t.Fatalf("NewSession() created invalid session: %v", err)
	}

	// End the session
	session.End()

	// Should still be valid (ended_at should be after started_at)
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() after End() unexpected error: %v", err)
	}

	// Verify ended_at is set and after started_at
	if session.EndedAt == nil {
		t.Error("Session.EndedAt should not be nil after End()")
	}
	if session.EndedAt != nil && !session.EndedAt.After(session.StartedAt) {
		t.Error("Session.EndedAt should be after StartedAt")
	}
}

// TestSessionValidateZeroEndedAt tests that zero EndedAt (via pointer to zero time) is handled correctly.
func TestSessionValidateZeroEndedAt(t *testing.T) {
	now := time.Now()
	zeroTime := time.Time{}

	t.Run("zero ended_at via pointer is handled", func(t *testing.T) {
		session := Session{
			ID:            "session-zero-end",
			Name:          "Test Session",
			StartedAt:     now,
			EndedAt:       &zeroTime,
			Conversations: []*Conversation{},
			Measurements:  []*Measurement{},
		}

		// The implementation checks IsZero(), so a zero time should skip the after check
		// Looking at the implementation: if s.EndedAt != nil && !s.EndedAt.IsZero()
		// So a zero time pointer should be treated as "not set" and pass validation
		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() with zero EndedAt pointer unexpected error: %v", err)
		}
	})
}

// TestSessionValidateWithConversationMessages tests full cascade validation with messages.
func TestSessionValidateWithConversationMessages(t *testing.T) {
	now := time.Now()

	t.Run("valid session with conversation containing valid messages", func(t *testing.T) {
		session := Session{
			ID:        "session-full",
			Name:      "Full Test Session",
			StartedAt: now,
			Conversations: []*Conversation{
				{
					ID:   "conv-1",
					Name: "Conversation with Messages",
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
							Content:   "Hi there! How can I help you?",
							Timestamp: now.Add(2 * time.Second),
						},
						{
							ID:         "msg-4",
							Role:       RoleTool,
							Content:    "Tool response",
							Timestamp:  now.Add(3 * time.Second),
							ToolCallID: "call-123",
							ToolName:   "get_data",
						},
					},
					CreatedAt: now,
					UpdatedAt: now.Add(3 * time.Second),
				},
			},
			Measurements: []*Measurement{
				{
					ID:        "meas-1",
					Timestamp: now,
					SenderID:  "sender-1",
				},
			},
		}

		if err := session.Validate(); err != nil {
			t.Errorf("Session.Validate() unexpected error: %v", err)
		}
	})
}

// TestSessionValidateMeasurementModeAllVariants tests all valid measurement modes.
func TestSessionValidateMeasurementModeAllVariants(t *testing.T) {
	now := time.Now()

	modes := []MeasurementMode{MeasurePassive, MeasureActive, MeasureTriggered, ""}

	for _, mode := range modes {
		t.Run("mode="+string(mode), func(t *testing.T) {
			session := Session{
				ID:        "session-mode-test",
				Name:      "Mode Test Session",
				StartedAt: now,
				Config: SessionConfig{
					MeasurementMode: mode,
				},
				Conversations: []*Conversation{},
				Measurements:  []*Measurement{},
			}

			if err := session.Validate(); err != nil {
				t.Errorf("Session.Validate() with MeasurementMode %q unexpected error: %v", mode, err)
			}
		})
	}
}
