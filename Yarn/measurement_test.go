package yarn

import (
	"testing"
	"time"
)

// TestBetaStatusIsValid tests the IsValid method for BetaStatus.
func TestBetaStatusIsValid(t *testing.T) {
	tests := []struct {
		name   string
		status BetaStatus
		valid  bool
	}{
		{"optimal status is valid", BetaOptimal, true},
		{"monitor status is valid", BetaMonitor, true},
		{"concerning status is valid", BetaConcerning, true},
		{"critical status is valid", BetaCritical, true},
		{"unknown status is valid", BetaUnknown, true},
		{"empty status is invalid", BetaStatus(""), false},
		{"invalid status string", BetaStatus("invalid"), false},
		{"typo status is invalid", BetaStatus("optmal"), false},
		{"uppercase status is invalid", BetaStatus("OPTIMAL"), false},
		{"mixed case status is invalid", BetaStatus("Optimal"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.status.IsValid(); got != tt.valid {
				t.Errorf("BetaStatus(%q).IsValid() = %v, want %v", tt.status, got, tt.valid)
			}
		})
	}
}

// TestMeasurementValidate tests the Validate method for Measurement.
func TestMeasurementValidate(t *testing.T) {
	validTimestamp := time.Now()

	tests := []struct {
		name        string
		measurement Measurement
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name: "valid measurement with sender only",
			measurement: Measurement{
				ID:         "m-1",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				DEff:       100,
				Beta:       1.8,
				Alignment:  0.75,
				BetaStatus: BetaOptimal,
			},
			wantErr: false,
		},
		{
			name: "valid measurement with receiver only",
			measurement: Measurement{
				ID:         "m-2",
				Timestamp:  validTimestamp,
				ReceiverID: "agent-2",
				TurnNumber: 1,
				DEff:       50,
				Beta:       2.0,
				Alignment:  -0.5,
				BetaStatus: BetaMonitor,
			},
			wantErr: false,
		},
		{
			name: "valid measurement with both participants",
			measurement: Measurement{
				ID:         "m-3",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				ReceiverID: "agent-2",
				TurnNumber: 5,
				DEff:       200,
				Beta:       1.5,
				Alignment:  1.0,
				BetaStatus: BetaOptimal,
			},
			wantErr: false,
		},
		{
			name: "valid measurement with all optional fields",
			measurement: Measurement{
				ID:             "m-4",
				Timestamp:      validTimestamp,
				SessionID:      "session-1",
				ConversationID: "conv-1",
				TurnNumber:     10,
				SenderID:       "agent-1",
				SenderName:     "Agent One",
				SenderRole:     "assistant",
				ReceiverID:     "agent-2",
				ReceiverName:   "Agent Two",
				ReceiverRole:   "user",
				DEff:           150,
				Beta:           2.3,
				Alignment:      0.0,
				CPair:          0.85,
				BetaStatus:     BetaMonitor,
				IsUnilateral:   false,
				MessageContent: "Hello world",
				TokenCount:     5,
			},
			wantErr: false,
		},
		{
			name: "valid measurement with zero metrics",
			measurement: Measurement{
				ID:         "m-5",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				DEff:       0,
				Beta:       0,
				Alignment:  0,
			},
			wantErr: false,
		},
		{
			name: "valid measurement with boundary alignment values",
			measurement: Measurement{
				ID:         "m-6",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				Alignment:  -1.0,
			},
			wantErr: false,
		},
		{
			name: "valid measurement with alignment at +1",
			measurement: Measurement{
				ID:         "m-7",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				Alignment:  1.0,
			},
			wantErr: false,
		},
		{
			name: "valid measurement with empty beta status",
			measurement: Measurement{
				ID:         "m-8",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				BetaStatus: "", // empty is allowed as default
			},
			wantErr: false,
		},
		{
			name: "missing id",
			measurement: Measurement{
				ID:         "",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
			},
			wantErr:     true,
			wantField:   "id",
			wantMessage: "id is required",
		},
		{
			name: "zero timestamp",
			measurement: Measurement{
				ID:         "m-9",
				Timestamp:  time.Time{},
				SenderID:   "agent-1",
				TurnNumber: 0,
			},
			wantErr:     true,
			wantField:   "timestamp",
			wantMessage: "timestamp is required",
		},
		{
			name: "negative turn number",
			measurement: Measurement{
				ID:         "m-10",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: -1,
			},
			wantErr:     true,
			wantField:   "turn_number",
			wantMessage: "turn_number must be non-negative",
		},
		{
			name: "no participants",
			measurement: Measurement{
				ID:         "m-11",
				Timestamp:  validTimestamp,
				SenderID:   "",
				ReceiverID: "",
				TurnNumber: 0,
			},
			wantErr:     true,
			wantField:   "sender_id",
			wantMessage: "at least one of sender_id or receiver_id is required",
		},
		{
			name: "negative d_eff",
			measurement: Measurement{
				ID:         "m-12",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				DEff:       -1,
			},
			wantErr:     true,
			wantField:   "d_eff",
			wantMessage: "d_eff must be non-negative",
		},
		{
			name: "negative beta",
			measurement: Measurement{
				ID:         "m-13",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				Beta:       -0.5,
			},
			wantErr:     true,
			wantField:   "beta",
			wantMessage: "beta must be non-negative",
		},
		{
			name: "alignment below -1",
			measurement: Measurement{
				ID:         "m-14",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				Alignment:  -1.01,
			},
			wantErr:     true,
			wantField:   "alignment",
			wantMessage: "alignment must be in range [-1, 1]",
		},
		{
			name: "alignment above 1",
			measurement: Measurement{
				ID:         "m-15",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				Alignment:  1.01,
			},
			wantErr:     true,
			wantField:   "alignment",
			wantMessage: "alignment must be in range [-1, 1]",
		},
		{
			name: "invalid beta status",
			measurement: Measurement{
				ID:         "m-16",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				BetaStatus: BetaStatus("invalid"),
			},
			wantErr:     true,
			wantField:   "beta_status",
			wantMessage: "invalid beta_status",
		},
		{
			name: "beta status typo",
			measurement: Measurement{
				ID:         "m-17",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				BetaStatus: BetaStatus("optmal"),
			},
			wantErr:     true,
			wantField:   "beta_status",
			wantMessage: "invalid beta_status",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.measurement.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Measurement.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Measurement.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("Measurement.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("Measurement.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestMeasurementValidateHiddenStateCascade tests cascade validation of hidden states.
func TestMeasurementValidateHiddenStateCascade(t *testing.T) {
	validTimestamp := time.Now()

	tests := []struct {
		name        string
		measurement Measurement
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name: "valid measurement with valid sender hidden state",
			measurement: Measurement{
				ID:         "m-hs-1",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				SenderHidden: &HiddenState{
					Vector: []float32{1.0, 2.0, 3.0, 4.0},
					Layer:  0,
					DType:  "float32",
				},
			},
			wantErr: false,
		},
		{
			name: "valid measurement with valid receiver hidden state",
			measurement: Measurement{
				ID:         "m-hs-2",
				Timestamp:  validTimestamp,
				ReceiverID: "agent-2",
				TurnNumber: 0,
				ReceiverHidden: &HiddenState{
					Vector: []float32{1.0, 2.0, 3.0, 4.0},
					Layer:  1,
					DType:  "float16",
				},
			},
			wantErr: false,
		},
		{
			name: "valid measurement with both hidden states",
			measurement: Measurement{
				ID:         "m-hs-3",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				ReceiverID: "agent-2",
				TurnNumber: 0,
				SenderHidden: &HiddenState{
					Vector: []float32{1.0, 2.0, 3.0, 4.0},
					Layer:  0,
				},
				ReceiverHidden: &HiddenState{
					Vector: []float32{5.0, 6.0, 7.0, 8.0},
					Layer:  0,
				},
			},
			wantErr: false,
		},
		{
			name: "valid measurement with nil hidden states",
			measurement: Measurement{
				ID:             "m-hs-4",
				Timestamp:      validTimestamp,
				SenderID:       "agent-1",
				TurnNumber:     0,
				SenderHidden:   nil,
				ReceiverHidden: nil,
			},
			wantErr: false,
		},
		{
			name: "invalid sender hidden state - empty vector",
			measurement: Measurement{
				ID:         "m-hs-5",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				SenderHidden: &HiddenState{
					Vector: []float32{},
					Layer:  0,
				},
			},
			wantErr:     true,
			wantField:   "sender_hidden.vector",
			wantMessage: "vector is required",
		},
		{
			name: "invalid sender hidden state - negative layer",
			measurement: Measurement{
				ID:         "m-hs-6",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				SenderHidden: &HiddenState{
					Vector: []float32{1.0, 2.0, 3.0},
					Layer:  -1,
				},
			},
			wantErr:     true,
			wantField:   "sender_hidden.layer",
			wantMessage: "layer must be non-negative",
		},
		{
			name: "invalid sender hidden state - invalid dtype",
			measurement: Measurement{
				ID:         "m-hs-7",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				SenderHidden: &HiddenState{
					Vector: []float32{1.0, 2.0, 3.0},
					Layer:  0,
					DType:  "float64",
				},
			},
			wantErr:     true,
			wantField:   "sender_hidden.dtype",
			wantMessage: "dtype must be 'float32' or 'float16'",
		},
		{
			name: "invalid sender hidden state - inconsistent shape",
			measurement: Measurement{
				ID:         "m-hs-8",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				TurnNumber: 0,
				SenderHidden: &HiddenState{
					Vector: []float32{1.0, 2.0, 3.0, 4.0},
					Shape:  []int{2, 3}, // 6 != 4
					Layer:  0,
				},
			},
			wantErr:     true,
			wantField:   "sender_hidden.shape",
			wantMessage: "shape is inconsistent with vector length",
		},
		{
			name: "invalid receiver hidden state - empty vector",
			measurement: Measurement{
				ID:         "m-hs-9",
				Timestamp:  validTimestamp,
				ReceiverID: "agent-2",
				TurnNumber: 0,
				ReceiverHidden: &HiddenState{
					Vector: []float32{},
					Layer:  0,
				},
			},
			wantErr:     true,
			wantField:   "receiver_hidden.vector",
			wantMessage: "vector is required",
		},
		{
			name: "invalid receiver hidden state - negative layer",
			measurement: Measurement{
				ID:         "m-hs-10",
				Timestamp:  validTimestamp,
				ReceiverID: "agent-2",
				TurnNumber: 0,
				ReceiverHidden: &HiddenState{
					Vector: []float32{1.0, 2.0, 3.0},
					Layer:  -5,
				},
			},
			wantErr:     true,
			wantField:   "receiver_hidden.layer",
			wantMessage: "layer must be non-negative",
		},
		{
			name: "invalid receiver hidden state - invalid dtype",
			measurement: Measurement{
				ID:         "m-hs-11",
				Timestamp:  validTimestamp,
				ReceiverID: "agent-2",
				TurnNumber: 0,
				ReceiverHidden: &HiddenState{
					Vector: []float32{1.0, 2.0, 3.0},
					Layer:  0,
					DType:  "bfloat16",
				},
			},
			wantErr:     true,
			wantField:   "receiver_hidden.dtype",
			wantMessage: "dtype must be 'float32' or 'float16'",
		},
		{
			name: "invalid receiver hidden state - zero shape dimension",
			measurement: Measurement{
				ID:         "m-hs-12",
				Timestamp:  validTimestamp,
				ReceiverID: "agent-2",
				TurnNumber: 0,
				ReceiverHidden: &HiddenState{
					Vector: []float32{1.0, 2.0, 3.0},
					Shape:  []int{0, 3},
					Layer:  0,
				},
			},
			wantErr:     true,
			wantField:   "receiver_hidden.shape",
			wantMessage: "shape dimensions must be positive",
		},
		{
			name: "sender hidden state error takes precedence over receiver",
			measurement: Measurement{
				ID:         "m-hs-13",
				Timestamp:  validTimestamp,
				SenderID:   "agent-1",
				ReceiverID: "agent-2",
				TurnNumber: 0,
				SenderHidden: &HiddenState{
					Vector: []float32{},
					Layer:  0,
				},
				ReceiverHidden: &HiddenState{
					Vector: []float32{},
					Layer:  0,
				},
			},
			wantErr:     true,
			wantField:   "sender_hidden.vector",
			wantMessage: "vector is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.measurement.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Measurement.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Measurement.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
				if err.Message != tt.wantMessage {
					t.Errorf("Measurement.Validate() error message = %q, want %q", err.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("Measurement.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestNewMeasurementValidation tests that NewMeasurement creates valid measurements.
func TestNewMeasurementValidation(t *testing.T) {
	t.Run("NewMeasurement creates measurement needing participant", func(t *testing.T) {
		m := NewMeasurement()

		// Should fail without a participant
		err := m.Validate()
		if err == nil {
			t.Error("NewMeasurement() created measurement that validates without participant")
			return
		}
		if err.Field != "sender_id" {
			t.Errorf("NewMeasurement() validation error field = %q, want %q", err.Field, "sender_id")
		}

		// Should pass after adding a participant
		m.SenderID = "agent-1"
		if err := m.Validate(); err != nil {
			t.Errorf("NewMeasurement() with SenderID validation error: %v", err)
		}
	})

	t.Run("NewMeasurement sets ID", func(t *testing.T) {
		m := NewMeasurement()
		if m.ID == "" {
			t.Error("NewMeasurement() did not set ID")
		}
	})

	t.Run("NewMeasurement sets Timestamp", func(t *testing.T) {
		m := NewMeasurement()
		if m.Timestamp.IsZero() {
			t.Error("NewMeasurement() did not set Timestamp")
		}
	})

	t.Run("NewMeasurement generates unique IDs", func(t *testing.T) {
		m1 := NewMeasurement()
		m2 := NewMeasurement()
		if m1.ID == m2.ID {
			t.Error("NewMeasurement() generated duplicate IDs")
		}
	})
}

// TestNewMeasurementForTurnValidation tests that NewMeasurementForTurn creates valid measurements.
func TestNewMeasurementForTurnValidation(t *testing.T) {
	t.Run("NewMeasurementForTurn creates measurement needing participant", func(t *testing.T) {
		m := NewMeasurementForTurn("session-1", "conv-1", 5)

		// Should fail without a participant
		err := m.Validate()
		if err == nil {
			t.Error("NewMeasurementForTurn() created measurement that validates without participant")
			return
		}

		// Should pass after adding a participant
		m.SenderID = "agent-1"
		if err := m.Validate(); err != nil {
			t.Errorf("NewMeasurementForTurn() with SenderID validation error: %v", err)
		}
	})

	t.Run("NewMeasurementForTurn sets session context", func(t *testing.T) {
		m := NewMeasurementForTurn("session-1", "conv-1", 5)
		if m.SessionID != "session-1" {
			t.Errorf("NewMeasurementForTurn() SessionID = %q, want %q", m.SessionID, "session-1")
		}
		if m.ConversationID != "conv-1" {
			t.Errorf("NewMeasurementForTurn() ConversationID = %q, want %q", m.ConversationID, "conv-1")
		}
		if m.TurnNumber != 5 {
			t.Errorf("NewMeasurementForTurn() TurnNumber = %d, want %d", m.TurnNumber, 5)
		}
	})

	t.Run("NewMeasurementForTurn sets ID and Timestamp", func(t *testing.T) {
		m := NewMeasurementForTurn("session-1", "conv-1", 0)
		if m.ID == "" {
			t.Error("NewMeasurementForTurn() did not set ID")
		}
		if m.Timestamp.IsZero() {
			t.Error("NewMeasurementForTurn() did not set Timestamp")
		}
	})
}

// TestComputeBetaStatus tests the ComputeBetaStatus function.
func TestComputeBetaStatus(t *testing.T) {
	tests := []struct {
		name   string
		beta   float64
		want   BetaStatus
	}{
		// Unknown range: beta <= 0
		{"negative beta", -1.0, BetaUnknown},
		{"zero beta", 0.0, BetaUnknown},

		// Unknown range: (0, 1.5)
		{"very low beta", 0.5, BetaUnknown},
		{"low beta", 1.0, BetaUnknown},
		{"just under optimal threshold", 1.49, BetaUnknown},

		// Optimal range: [1.5, 2.0)
		{"optimal lower bound", 1.5, BetaOptimal},
		{"optimal mid", 1.75, BetaOptimal},
		{"just under monitor threshold", 1.99, BetaOptimal},

		// Monitor range: [2.0, 2.5)
		{"monitor lower bound", 2.0, BetaMonitor},
		{"monitor mid", 2.25, BetaMonitor},
		{"just under concerning threshold", 2.49, BetaMonitor},

		// Concerning range: [2.5, 3.0)
		{"concerning lower bound", 2.5, BetaConcerning},
		{"concerning mid", 2.75, BetaConcerning},
		{"just under critical threshold", 2.99, BetaConcerning},

		// Critical range: >= 3.0
		{"critical lower bound", 3.0, BetaCritical},
		{"critical high", 5.0, BetaCritical},
		{"critical very high", 10.0, BetaCritical},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ComputeBetaStatus(tt.beta); got != tt.want {
				t.Errorf("ComputeBetaStatus(%v) = %q, want %q", tt.beta, got, tt.want)
			}
		})
	}
}

// TestMeasurementIsBilateral tests the IsBilateral method.
func TestMeasurementIsBilateral(t *testing.T) {
	tests := []struct {
		name        string
		measurement Measurement
		want        bool
	}{
		{
			name:        "no hidden states",
			measurement: Measurement{},
			want:        false,
		},
		{
			name: "sender hidden state only",
			measurement: Measurement{
				SenderHidden: &HiddenState{Vector: []float32{1.0, 2.0, 3.0}},
			},
			want: false,
		},
		{
			name: "receiver hidden state only",
			measurement: Measurement{
				ReceiverHidden: &HiddenState{Vector: []float32{1.0, 2.0, 3.0}},
			},
			want: false,
		},
		{
			name: "both hidden states with vectors",
			measurement: Measurement{
				SenderHidden:   &HiddenState{Vector: []float32{1.0, 2.0, 3.0}},
				ReceiverHidden: &HiddenState{Vector: []float32{4.0, 5.0, 6.0}},
			},
			want: true,
		},
		{
			name: "sender hidden state with empty vector",
			measurement: Measurement{
				SenderHidden:   &HiddenState{Vector: []float32{}},
				ReceiverHidden: &HiddenState{Vector: []float32{1.0, 2.0, 3.0}},
			},
			want: false,
		},
		{
			name: "receiver hidden state with empty vector",
			measurement: Measurement{
				SenderHidden:   &HiddenState{Vector: []float32{1.0, 2.0, 3.0}},
				ReceiverHidden: &HiddenState{Vector: []float32{}},
			},
			want: false,
		},
		{
			name: "both hidden states with nil vectors",
			measurement: Measurement{
				SenderHidden:   &HiddenState{Vector: nil},
				ReceiverHidden: &HiddenState{Vector: nil},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.measurement.IsBilateral(); got != tt.want {
				t.Errorf("Measurement.IsBilateral() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestMeasurementSetSender tests the SetSender method.
func TestMeasurementSetSender(t *testing.T) {
	m := NewMeasurement()
	hidden := &HiddenState{
		Vector: []float32{1.0, 2.0, 3.0},
		Layer:  0,
	}

	m.SetSender("agent-1", "Agent One", "assistant", hidden)

	if m.SenderID != "agent-1" {
		t.Errorf("SetSender() SenderID = %q, want %q", m.SenderID, "agent-1")
	}
	if m.SenderName != "Agent One" {
		t.Errorf("SetSender() SenderName = %q, want %q", m.SenderName, "Agent One")
	}
	if m.SenderRole != "assistant" {
		t.Errorf("SetSender() SenderRole = %q, want %q", m.SenderRole, "assistant")
	}
	if m.SenderHidden != hidden {
		t.Error("SetSender() SenderHidden not set correctly")
	}
}

// TestMeasurementSetReceiver tests the SetReceiver method.
func TestMeasurementSetReceiver(t *testing.T) {
	m := NewMeasurement()
	hidden := &HiddenState{
		Vector: []float32{4.0, 5.0, 6.0},
		Layer:  1,
	}

	m.SetReceiver("agent-2", "Agent Two", "user", hidden)

	if m.ReceiverID != "agent-2" {
		t.Errorf("SetReceiver() ReceiverID = %q, want %q", m.ReceiverID, "agent-2")
	}
	if m.ReceiverName != "Agent Two" {
		t.Errorf("SetReceiver() ReceiverName = %q, want %q", m.ReceiverName, "Agent Two")
	}
	if m.ReceiverRole != "user" {
		t.Errorf("SetReceiver() ReceiverRole = %q, want %q", m.ReceiverRole, "user")
	}
	if m.ReceiverHidden != hidden {
		t.Error("SetReceiver() ReceiverHidden not set correctly")
	}
}

// TestMeasurementValidationOrder tests that validation fails at the first error.
func TestMeasurementValidationOrder(t *testing.T) {
	// A measurement with multiple problems should fail on the first check
	m := Measurement{
		ID:         "", // First error - id required
		Timestamp:  time.Time{},
		TurnNumber: -1,
		SenderID:   "",
		ReceiverID: "",
		DEff:       -1,
	}

	err := m.Validate()
	if err == nil {
		t.Fatal("Measurement.Validate() expected error, got nil")
	}
	if err.Field != "id" {
		t.Errorf("Measurement.Validate() should fail on id first, got field = %q", err.Field)
	}
}
