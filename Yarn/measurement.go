package yarn

import (
	"time"

	"github.com/google/uuid"
)

// Measurement contains conveyance metrics from a single agent interaction.
type Measurement struct {
	ID        string    `json:"id"`
	Timestamp time.Time `json:"timestamp"`

	// Session context
	SessionID      string `json:"session_id"`
	ConversationID string `json:"conversation_id"`
	TurnNumber     int    `json:"turn_number"`

	// Participants
	SenderID   string `json:"sender_id"`
	SenderName string `json:"sender_name"`
	SenderRole string `json:"sender_role"`

	ReceiverID   string `json:"receiver_id"`
	ReceiverName string `json:"receiver_name"`
	ReceiverRole string `json:"receiver_role"`

	// Boundary objects (hidden states)
	SenderHidden   *HiddenState `json:"sender_hidden,omitempty"`
	ReceiverHidden *HiddenState `json:"receiver_hidden,omitempty"`

	// Core conveyance metrics
	DEff      int     `json:"d_eff"`      // Effective dimensionality
	Beta      float64 `json:"beta"`       // Collapse indicator
	Alignment float64 `json:"alignment"`  // Cosine similarity
	CPair     float64 `json:"c_pair"`     // Bilateral conveyance

	// Quality indicators
	BetaStatus   BetaStatus `json:"beta_status"`
	IsUnilateral bool       `json:"is_unilateral"`

	// Message context
	MessageContent string `json:"message_content,omitempty"`
	TokenCount     int    `json:"token_count,omitempty"`
}

// BetaStatus indicates the quality of the β value.
// Beta (β) is the collapse indicator from the Conveyance Framework.
// Lower values indicate better dimensional preservation.
type BetaStatus string

const (
	BetaOptimal    BetaStatus = "optimal"    // β ∈ [1.5, 2.0) - ideal range
	BetaMonitor    BetaStatus = "monitor"    // β ∈ [2.0, 2.5) - acceptable, watch for drift
	BetaConcerning BetaStatus = "concerning" // β ∈ [2.5, 3.0) - dimensional compression detected
	BetaCritical   BetaStatus = "critical"   // β ≥ 3.0 - severe collapse, intervention needed
	BetaUnknown    BetaStatus = "unknown"    // β ≤ 0 or β ∈ (0, 1.5) - invalid or uncategorized
)

// IsValid returns true if this is a valid beta status.
func (s BetaStatus) IsValid() bool {
	switch s {
	case BetaOptimal, BetaMonitor, BetaConcerning, BetaCritical, BetaUnknown:
		return true
	default:
		return false
	}
}

// NewMeasurement creates a new Measurement with a generated ID.
// Note: A SenderID or ReceiverID must be set before the measurement is valid.
// Use SetSender() or SetReceiver() to configure participants after creation.
func NewMeasurement() *Measurement {
	return &Measurement{
		ID:        uuid.New().String(),
		Timestamp: time.Now(),
	}
}

// NewMeasurementForTurn creates a measurement for a specific conversation turn.
func NewMeasurementForTurn(sessionID, convID string, turn int) *Measurement {
	m := NewMeasurement()
	m.SessionID = sessionID
	m.ConversationID = convID
	m.TurnNumber = turn
	return m
}

// SetSender sets the sender information.
func (m *Measurement) SetSender(id, name, role string, hidden *HiddenState) {
	m.SenderID = id
	m.SenderName = name
	m.SenderRole = role
	m.SenderHidden = hidden
}

// SetReceiver sets the receiver information.
func (m *Measurement) SetReceiver(id, name, role string, hidden *HiddenState) {
	m.ReceiverID = id
	m.ReceiverName = name
	m.ReceiverRole = role
	m.ReceiverHidden = hidden
}

// IsBilateral returns true if both sender and receiver have hidden states.
func (m *Measurement) IsBilateral() bool {
	hasSender := m.SenderHidden != nil && len(m.SenderHidden.Vector) > 0
	hasReceiver := m.ReceiverHidden != nil && len(m.ReceiverHidden.Vector) > 0
	return hasSender && hasReceiver
}

// Validate checks if the measurement is valid.
// Returns a ValidationError if invalid, nil if valid.
func (m *Measurement) Validate() *ValidationError {
	// Required fields
	if m.ID == "" {
		return &ValidationError{Field: "id", Message: "id is required"}
	}
	if m.Timestamp.IsZero() {
		return &ValidationError{Field: "timestamp", Message: "timestamp is required"}
	}

	// TurnNumber must be non-negative
	if m.TurnNumber < 0 {
		return &ValidationError{Field: "turn_number", Message: "turn_number must be non-negative"}
	}

	// At least one participant is required
	if m.SenderID == "" && m.ReceiverID == "" {
		return &ValidationError{Field: "sender_id", Message: "at least one of sender_id or receiver_id is required"}
	}

	// Metric validations
	if m.DEff < 0 {
		return &ValidationError{Field: "d_eff", Message: "d_eff must be non-negative"}
	}
	if m.Beta < 0 {
		return &ValidationError{Field: "beta", Message: "beta must be non-negative"}
	}
	if m.Alignment < -1 || m.Alignment > 1 {
		return &ValidationError{Field: "alignment", Message: "alignment must be in range [-1, 1]"}
	}

	// BetaStatus validation (empty is allowed as default)
	if m.BetaStatus != "" && !m.BetaStatus.IsValid() {
		return &ValidationError{Field: "beta_status", Message: "invalid beta_status"}
	}

	// Validate hidden states if present
	if err := m.SenderHidden.Validate(); err != nil {
		return &ValidationError{Field: "sender_hidden." + err.Field, Message: err.Message}
	}
	if err := m.ReceiverHidden.Validate(); err != nil {
		return &ValidationError{Field: "receiver_hidden." + err.Field, Message: err.Message}
	}

	return nil
}

// ComputeBetaStatus determines the status based on β value.
func ComputeBetaStatus(beta float64) BetaStatus {
	switch {
	case beta <= 0:
		return BetaUnknown
	case beta < 1.5:
		return BetaUnknown // Explicitly handle (0, 1.5) range
	case beta < 2.0:
		return BetaOptimal // β ∈ [1.5, 2.0)
	case beta < 2.5:
		return BetaMonitor // β ∈ [2.0, 2.5)
	case beta < 3.0:
		return BetaConcerning // β ∈ [2.5, 3.0)
	default:
		return BetaCritical // β ≥ 3.0
	}
}
