package yarn

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Session is a named research session grouping conversations and measurements.
type Session struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	StartedAt   time.Time      `json:"started_at"`
	EndedAt     *time.Time     `json:"ended_at,omitempty"`
	Config      SessionConfig  `json:"config"`
	Metadata    map[string]any `json:"metadata,omitempty"`

	Conversations []*Conversation `json:"conversations"`
	Measurements  []*Measurement  `json:"measurements"`

	mu sync.RWMutex
}

// SessionConfig holds session configuration.
type SessionConfig struct {
	MeasurementMode MeasurementMode `json:"measurement_mode"`
	AutoExport      bool            `json:"auto_export"`
	ExportPath      string          `json:"export_path"`
}

// MeasurementMode determines when measurements are captured.
type MeasurementMode string

const (
	MeasurePassive   MeasurementMode = "passive"   // Observe only
	MeasureActive    MeasurementMode = "active"    // Every exchange
	MeasureTriggered MeasurementMode = "triggered" // On request
)

// IsValid returns true if this is a valid measurement mode.
func (m MeasurementMode) IsValid() bool {
	switch m {
	case MeasurePassive, MeasureActive, MeasureTriggered:
		return true
	default:
		return false
	}
}

// Validate checks if the session config is valid.
// Returns a ValidationError if invalid, nil if valid.
// Empty MeasurementMode is allowed (treated as default).
func (c *SessionConfig) Validate() *ValidationError {
	// MeasurementMode must be valid if set
	if c.MeasurementMode != "" && !c.MeasurementMode.IsValid() {
		return &ValidationError{Field: "measurement_mode", Message: "invalid measurement mode"}
	}
	return nil
}

// Validate checks if the session is valid.
// Returns a ValidationError if invalid, nil if valid.
func (s *Session) Validate() *ValidationError {
	if s.ID == "" {
		return &ValidationError{Field: "id", Message: "id is required"}
	}
	if s.Name == "" {
		return &ValidationError{Field: "name", Message: "name is required"}
	}
	if s.StartedAt.IsZero() {
		return &ValidationError{Field: "started_at", Message: "started_at is required"}
	}

	// EndedAt must be after StartedAt if set
	if s.EndedAt != nil && !s.EndedAt.IsZero() {
		if s.EndedAt.Before(s.StartedAt) || s.EndedAt.Equal(s.StartedAt) {
			return &ValidationError{Field: "ended_at", Message: "ended_at must be after started_at"}
		}
	}

	// Validate MeasurementMode if set
	if s.Config.MeasurementMode != "" && !s.Config.MeasurementMode.IsValid() {
		return &ValidationError{Field: "config.measurement_mode", Message: "invalid measurement mode"}
	}

	// Cascade validation: validate all conversations
	for i, conv := range s.Conversations {
		if conv == nil {
			return &ValidationError{
				Field:   "conversations",
				Message: "conversation at index " + strconv.Itoa(i) + " is nil",
			}
		}
		if err := conv.Validate(); err != nil {
			return &ValidationError{
				Field:   "conversations[" + strconv.Itoa(i) + "]." + err.Field,
				Message: err.Message,
			}
		}
	}

	// Cascade validation: validate all measurements
	for i, m := range s.Measurements {
		if m == nil {
			return &ValidationError{
				Field:   "measurements",
				Message: "measurement at index " + strconv.Itoa(i) + " is nil",
			}
		}
		if err := m.Validate(); err != nil {
			return &ValidationError{
				Field:   "measurements[" + strconv.Itoa(i) + "]." + err.Field,
				Message: err.Message,
			}
		}
	}

	return nil
}

// NewSession creates a new session.
func NewSession(name, description string) *Session {
	return &Session{
		ID:            uuid.New().String(),
		Name:          name,
		Description:   description,
		StartedAt:     time.Now(),
		Conversations: make([]*Conversation, 0),
		Measurements:  make([]*Measurement, 0),
		Metadata:      make(map[string]any),
		Config: SessionConfig{
			MeasurementMode: MeasureActive,
			AutoExport:      true,
			ExportPath:      "./experiments",
		},
	}
}

// AddConversation adds a conversation to the session.
func (s *Session) AddConversation(conv *Conversation) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Conversations = append(s.Conversations, conv)
}

// AddMeasurement adds a measurement to the session.
func (s *Session) AddMeasurement(m *Measurement) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m.SessionID = s.ID
	s.Measurements = append(s.Measurements, m)
}

// ActiveConversation returns the most recent conversation, or creates one.
func (s *Session) ActiveConversation() *Conversation {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.Conversations) == 0 {
		conv := NewConversation(s.Name + "-conv-1")
		s.Conversations = append(s.Conversations, conv)
	}
	return s.Conversations[len(s.Conversations)-1]
}

// End marks the session as ended.
func (s *Session) End() {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	s.EndedAt = &now
}

// Stats returns session statistics.
func (s *Session) Stats() SessionStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := SessionStats{
		ConversationCount: len(s.Conversations),
		MeasurementCount:  len(s.Measurements),
	}

	for _, conv := range s.Conversations {
		stats.MessageCount += conv.Length()
	}

	if len(s.Measurements) > 0 {
		var totalDEff, totalBeta, totalAlignment float64
		var bilateralCount int

		for _, m := range s.Measurements {
			totalDEff += float64(m.DEff)
			totalBeta += m.Beta
			totalAlignment += m.Alignment
			if m.IsBilateral() {
				bilateralCount++
			}
		}

		n := float64(len(s.Measurements))
		stats.AvgDEff = totalDEff / n
		stats.AvgBeta = totalBeta / n
		stats.AvgAlignment = totalAlignment / n
		stats.BilateralCount = bilateralCount
	}

	return stats
}

// SessionStats holds session statistics.
type SessionStats struct {
	ConversationCount int     `json:"conversation_count"`
	MessageCount      int     `json:"message_count"`
	MeasurementCount  int     `json:"measurement_count"`
	BilateralCount    int     `json:"bilateral_count"`
	AvgDEff           float64 `json:"avg_d_eff"`
	AvgBeta           float64 `json:"avg_beta"`
	AvgAlignment      float64 `json:"avg_alignment"`
}

// Export writes the session to files (JSON + JSONL for measurements).
func (s *Session) Export() (err error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	exportDir := filepath.Join(s.Config.ExportPath, s.ID)
	if err := os.MkdirAll(exportDir, 0755); err != nil {
		return err
	}

	// Export session metadata
	sessionFile := filepath.Join(exportDir, "session.json")
	sessionData, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(sessionFile, sessionData, 0644); err != nil {
		return err
	}

	// Export measurements as JSONL
	measurementsFile := filepath.Join(exportDir, "measurements.jsonl")
	f, err := os.Create(measurementsFile)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()

	for _, m := range s.Measurements {
		data, err := json.Marshal(m)
		if err != nil {
			return err
		}
		if _, err := f.Write(data); err != nil {
			return err
		}
		if _, err := f.WriteString("\n"); err != nil {
			return err
		}
	}

	return nil
}
