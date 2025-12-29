// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"sync"
	"time"
)

// -----------------------------------------------------------------------------
// Event Types and Constants
// -----------------------------------------------------------------------------

// Additional event types for extended measurement data
const (
	EventTypeMeasurementBatch = "measurement_batch"
	EventTypeSessionStart     = "session_start"
	EventTypeSessionEnd       = "session_end"
	EventTypeConversationTurn = "conversation_turn"
	EventTypeBetaAlert        = "beta_alert"
)

// BetaStatus indicates the quality of the beta value.
// Mirrors the Yarn BetaStatus type for API consistency.
type BetaStatus string

const (
	BetaOptimal    BetaStatus = "optimal"    // beta in [1.5, 2.0)
	BetaMonitor    BetaStatus = "monitor"    // beta in [2.0, 2.5)
	BetaConcerning BetaStatus = "concerning" // beta in [2.5, 3.0)
	BetaCritical   BetaStatus = "critical"   // beta >= 3.0
	BetaUnknown    BetaStatus = "unknown"    // beta <= 0 or in (0, 1.5)
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

// ComputeBetaStatus determines the status based on beta value.
// This mirrors the Yarn ComputeBetaStatus function.
func ComputeBetaStatus(beta float64) BetaStatus {
	switch {
	case beta <= 0:
		return BetaUnknown
	case beta < 1.5:
		return BetaUnknown
	case beta < 2.0:
		return BetaOptimal
	case beta < 2.5:
		return BetaMonitor
	case beta < 3.0:
		return BetaConcerning
	default:
		return BetaCritical
	}
}

// -----------------------------------------------------------------------------
// Extended Event Data Types
// -----------------------------------------------------------------------------

// MeasurementEvent provides a richer measurement event structure that aligns
// with the Yarn Measurement type for comprehensive conveyance metrics.
type MeasurementEvent struct {
	// Core identification
	ID        string `json:"id"`
	Timestamp string `json:"timestamp"`

	// Session context
	SessionID      string `json:"sessionId,omitempty"`
	ConversationID string `json:"conversationId,omitempty"`
	Turn           int    `json:"turn"`

	// Participants
	SenderID   string `json:"senderId,omitempty"`
	SenderName string `json:"senderName,omitempty"`
	SenderRole string `json:"senderRole,omitempty"`

	ReceiverID   string `json:"receiverId,omitempty"`
	ReceiverName string `json:"receiverName,omitempty"`
	ReceiverRole string `json:"receiverRole,omitempty"`

	// Core conveyance metrics
	Deff      float64 `json:"deff"`      // Effective dimensionality
	Beta      float64 `json:"beta"`      // Collapse indicator
	Alignment float64 `json:"alignment"` // Cosine similarity
	Cpair     float64 `json:"cpair"`     // Bilateral conveyance

	// Quality indicators
	BetaStatus   BetaStatus `json:"betaStatus,omitempty"`
	IsUnilateral bool       `json:"isUnilateral,omitempty"`

	// Message context
	TokenCount int `json:"tokenCount,omitempty"`
}

// NewMeasurementEvent creates a MeasurementEvent from basic data.
func NewMeasurementEvent(id string, turn int, deff, beta, alignment, cpair float64) *MeasurementEvent {
	return &MeasurementEvent{
		ID:         id,
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Turn:       turn,
		Deff:       deff,
		Beta:       beta,
		Alignment:  alignment,
		Cpair:      cpair,
		BetaStatus: ComputeBetaStatus(beta),
	}
}

// SetParticipants sets the sender and receiver information.
func (e *MeasurementEvent) SetParticipants(senderID, senderName, senderRole, receiverID, receiverName, receiverRole string) {
	e.SenderID = senderID
	e.SenderName = senderName
	e.SenderRole = senderRole
	e.ReceiverID = receiverID
	e.ReceiverName = receiverName
	e.ReceiverRole = receiverRole
}

// SetSession sets the session context.
func (e *MeasurementEvent) SetSession(sessionID, conversationID string) {
	e.SessionID = sessionID
	e.ConversationID = conversationID
}

// ToMeasurementData converts to the simpler MeasurementData type for backward compatibility.
func (e *MeasurementEvent) ToMeasurementData() *MeasurementData {
	return &MeasurementData{
		Turn:      e.Turn,
		Deff:      e.Deff,
		Beta:      e.Beta,
		Alignment: e.Alignment,
		Cpair:     e.Cpair,
		Sender:    e.SenderName,
		Receiver:  e.ReceiverName,
	}
}

// MeasurementBatchEvent contains multiple measurements for batch updates.
type MeasurementBatchEvent struct {
	Measurements []MeasurementEvent `json:"measurements"`
	SessionID    string             `json:"sessionId,omitempty"`
	TurnRange    [2]int             `json:"turnRange"` // [start, end]
}

// SessionStartEvent is sent when a new measurement session begins.
type SessionStartEvent struct {
	SessionID   string   `json:"sessionId"`
	Name        string   `json:"name,omitempty"`
	AgentIDs    []string `json:"agentIds,omitempty"`
	StartedAt   string   `json:"startedAt"`
	Description string   `json:"description,omitempty"`
}

// SessionEndEvent is sent when a measurement session ends.
type SessionEndEvent struct {
	SessionID    string                 `json:"sessionId"`
	EndedAt      string                 `json:"endedAt"`
	TotalTurns   int                    `json:"totalTurns"`
	FinalMetrics *SessionMetricsSummary `json:"finalMetrics,omitempty"`
}

// SessionMetricsSummary provides aggregate statistics for a session.
type SessionMetricsSummary struct {
	TotalMeasurements int     `json:"totalMeasurements"`
	AvgDeff           float64 `json:"avgDeff"`
	AvgBeta           float64 `json:"avgBeta"`
	AvgAlignment      float64 `json:"avgAlignment"`
	AvgCpair          float64 `json:"avgCpair"`
	MinBeta           float64 `json:"minBeta"`
	MaxBeta           float64 `json:"maxBeta"`
	BetaAlertCount    int     `json:"betaAlertCount"`
}

// ConversationTurnEvent is sent for each conversation turn with context.
type ConversationTurnEvent struct {
	SessionID      string `json:"sessionId"`
	ConversationID string `json:"conversationId"`
	Turn           int    `json:"turn"`
	SenderName     string `json:"senderName"`
	ReceiverName   string `json:"receiverName"`
	Timestamp      string `json:"timestamp"`
}

// BetaAlertEvent is sent when beta reaches concerning or critical levels.
type BetaAlertEvent struct {
	MeasurementID  string     `json:"measurementId"`
	SessionID      string     `json:"sessionId,omitempty"`
	Turn           int        `json:"turn"`
	Beta           float64    `json:"beta"`
	Status         BetaStatus `json:"status"`
	PreviousBeta   float64    `json:"previousBeta,omitempty"`
	PreviousStatus BetaStatus `json:"previousStatus,omitempty"`
	AlertMessage   string     `json:"alertMessage"`
	Timestamp      string     `json:"timestamp"`
}

// -----------------------------------------------------------------------------
// EventBroadcaster Interface
// -----------------------------------------------------------------------------

// EventBroadcaster defines the interface for broadcasting events to clients.
// This allows for dependency injection and testing.
type EventBroadcaster interface {
	// BroadcastMeasurementEvent sends a measurement event to subscribed clients.
	BroadcastMeasurementEvent(event *MeasurementEvent) error

	// BroadcastMeasurementBatch sends a batch of measurements.
	BroadcastMeasurementBatch(batch *MeasurementBatchEvent) error

	// BroadcastSessionStart notifies clients of a new session.
	BroadcastSessionStart(event *SessionStartEvent) error

	// BroadcastSessionEnd notifies clients of a session ending.
	BroadcastSessionEnd(event *SessionEndEvent) error

	// BroadcastConversationTurn notifies clients of a conversation turn.
	BroadcastConversationTurn(event *ConversationTurnEvent) error

	// BroadcastBetaAlert sends a beta alert when thresholds are exceeded.
	BroadcastBetaAlert(alert *BetaAlertEvent) error
}

// -----------------------------------------------------------------------------
// HubEventBroadcaster Implementation
// -----------------------------------------------------------------------------

// HubEventBroadcaster wraps the WebSocket Hub to implement EventBroadcaster.
type HubEventBroadcaster struct {
	hub *Hub
}

// NewHubEventBroadcaster creates a new HubEventBroadcaster.
func NewHubEventBroadcaster(hub *Hub) *HubEventBroadcaster {
	return &HubEventBroadcaster{hub: hub}
}

// BroadcastMeasurementEvent sends a measurement event to subscribed clients.
func (b *HubEventBroadcaster) BroadcastMeasurementEvent(event *MeasurementEvent) error {
	msg := &WSMessage{
		Type:      EventTypeMeasurement,
		Data:      event,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return b.hub.BroadcastToChannel(ChannelMeasurements, msg)
}

// BroadcastMeasurementBatch sends a batch of measurements.
func (b *HubEventBroadcaster) BroadcastMeasurementBatch(batch *MeasurementBatchEvent) error {
	msg := &WSMessage{
		Type:      EventTypeMeasurementBatch,
		Data:      batch,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return b.hub.BroadcastToChannel(ChannelMeasurements, msg)
}

// BroadcastSessionStart notifies clients of a new session.
func (b *HubEventBroadcaster) BroadcastSessionStart(event *SessionStartEvent) error {
	msg := &WSMessage{
		Type:      EventTypeSessionStart,
		Data:      event,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return b.hub.BroadcastToChannel(ChannelMeasurements, msg)
}

// BroadcastSessionEnd notifies clients of a session ending.
func (b *HubEventBroadcaster) BroadcastSessionEnd(event *SessionEndEvent) error {
	msg := &WSMessage{
		Type:      EventTypeSessionEnd,
		Data:      event,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return b.hub.BroadcastToChannel(ChannelMeasurements, msg)
}

// BroadcastConversationTurn notifies clients of a conversation turn.
func (b *HubEventBroadcaster) BroadcastConversationTurn(event *ConversationTurnEvent) error {
	msg := &WSMessage{
		Type:      EventTypeConversationTurn,
		Data:      event,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return b.hub.BroadcastToChannel(ChannelMessages, msg)
}

// BroadcastBetaAlert sends a beta alert when thresholds are exceeded.
func (b *HubEventBroadcaster) BroadcastBetaAlert(alert *BetaAlertEvent) error {
	msg := &WSMessage{
		Type:      EventTypeBetaAlert,
		Data:      alert,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	return b.hub.BroadcastToChannel(ChannelMeasurements, msg)
}

// -----------------------------------------------------------------------------
// Event Aggregator
// -----------------------------------------------------------------------------

// EventAggregator collects measurements and computes aggregate statistics.
type EventAggregator struct {
	mu           sync.RWMutex
	measurements []MeasurementEvent
	sessionID    string
	maxSize      int
}

// NewEventAggregator creates a new EventAggregator with the given max size.
func NewEventAggregator(sessionID string, maxSize int) *EventAggregator {
	if maxSize <= 0 {
		maxSize = 1000 // Default max size
	}
	return &EventAggregator{
		measurements: make([]MeasurementEvent, 0, maxSize),
		sessionID:    sessionID,
		maxSize:      maxSize,
	}
}

// Add adds a measurement to the aggregator.
// If maxSize is reached, the oldest measurement is removed.
func (a *EventAggregator) Add(event MeasurementEvent) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.measurements) >= a.maxSize {
		// Remove oldest (first) element
		a.measurements = a.measurements[1:]
	}
	a.measurements = append(a.measurements, event)
}

// Count returns the number of measurements in the aggregator.
func (a *EventAggregator) Count() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return len(a.measurements)
}

// Clear removes all measurements from the aggregator.
func (a *EventAggregator) Clear() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.measurements = make([]MeasurementEvent, 0, a.maxSize)
}

// GetAll returns a copy of all measurements.
func (a *EventAggregator) GetAll() []MeasurementEvent {
	a.mu.RLock()
	defer a.mu.RUnlock()

	result := make([]MeasurementEvent, len(a.measurements))
	copy(result, a.measurements)
	return result
}

// GetSummary computes aggregate statistics for the collected measurements.
func (a *EventAggregator) GetSummary() *SessionMetricsSummary {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if len(a.measurements) == 0 {
		return &SessionMetricsSummary{}
	}

	summary := &SessionMetricsSummary{
		TotalMeasurements: len(a.measurements),
		MinBeta:           a.measurements[0].Beta,
		MaxBeta:           a.measurements[0].Beta,
	}

	var sumDeff, sumBeta, sumAlignment, sumCpair float64

	for _, m := range a.measurements {
		sumDeff += m.Deff
		sumBeta += m.Beta
		sumAlignment += m.Alignment
		sumCpair += m.Cpair

		if m.Beta < summary.MinBeta {
			summary.MinBeta = m.Beta
		}
		if m.Beta > summary.MaxBeta {
			summary.MaxBeta = m.Beta
		}

		status := ComputeBetaStatus(m.Beta)
		if status == BetaConcerning || status == BetaCritical {
			summary.BetaAlertCount++
		}
	}

	n := float64(len(a.measurements))
	summary.AvgDeff = sumDeff / n
	summary.AvgBeta = sumBeta / n
	summary.AvgAlignment = sumAlignment / n
	summary.AvgCpair = sumCpair / n

	return summary
}

// GetBatch returns measurements as a batch event.
func (a *EventAggregator) GetBatch() *MeasurementBatchEvent {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if len(a.measurements) == 0 {
		return &MeasurementBatchEvent{
			Measurements: []MeasurementEvent{},
			SessionID:    a.sessionID,
			TurnRange:    [2]int{0, 0},
		}
	}

	measurements := make([]MeasurementEvent, len(a.measurements))
	copy(measurements, a.measurements)

	minTurn := measurements[0].Turn
	maxTurn := measurements[0].Turn
	for _, m := range measurements {
		if m.Turn < minTurn {
			minTurn = m.Turn
		}
		if m.Turn > maxTurn {
			maxTurn = m.Turn
		}
	}

	return &MeasurementBatchEvent{
		Measurements: measurements,
		SessionID:    a.sessionID,
		TurnRange:    [2]int{minTurn, maxTurn},
	}
}

// -----------------------------------------------------------------------------
// Beta Alert Monitor
// -----------------------------------------------------------------------------

// BetaAlertMonitor tracks beta values and generates alerts when thresholds are exceeded.
type BetaAlertMonitor struct {
	mu            sync.RWMutex
	broadcaster   EventBroadcaster
	lastBeta      map[string]float64 // sessionID -> last beta
	lastStatus    map[string]BetaStatus
	alertEnabled  bool
	alertCallback func(*BetaAlertEvent)
}

// NewBetaAlertMonitor creates a new BetaAlertMonitor.
func NewBetaAlertMonitor(broadcaster EventBroadcaster) *BetaAlertMonitor {
	return &BetaAlertMonitor{
		broadcaster:  broadcaster,
		lastBeta:     make(map[string]float64),
		lastStatus:   make(map[string]BetaStatus),
		alertEnabled: true,
	}
}

// SetAlertCallback sets a callback function to be called when alerts are generated.
func (m *BetaAlertMonitor) SetAlertCallback(callback func(*BetaAlertEvent)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.alertCallback = callback
}

// EnableAlerts enables or disables alert broadcasting.
func (m *BetaAlertMonitor) EnableAlerts(enabled bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.alertEnabled = enabled
}

// CheckMeasurement evaluates a measurement and generates an alert if needed.
func (m *BetaAlertMonitor) CheckMeasurement(event *MeasurementEvent) *BetaAlertEvent {
	m.mu.Lock()
	defer m.mu.Unlock()

	currentStatus := ComputeBetaStatus(event.Beta)
	sessionKey := event.SessionID
	if sessionKey == "" {
		sessionKey = "default"
	}

	previousBeta := m.lastBeta[sessionKey]
	previousStatus := m.lastStatus[sessionKey]

	// Update tracking
	m.lastBeta[sessionKey] = event.Beta
	m.lastStatus[sessionKey] = currentStatus

	// Only alert on concerning or critical status
	if currentStatus != BetaConcerning && currentStatus != BetaCritical {
		return nil
	}

	// Generate alert message
	var alertMessage string
	switch currentStatus {
	case BetaConcerning:
		alertMessage = "Beta value has reached concerning levels. Dimensional compression detected."
	case BetaCritical:
		alertMessage = "CRITICAL: Beta value indicates severe dimensional collapse. Intervention recommended."
	}

	alert := &BetaAlertEvent{
		MeasurementID:  event.ID,
		SessionID:      event.SessionID,
		Turn:           event.Turn,
		Beta:           event.Beta,
		Status:         currentStatus,
		PreviousBeta:   previousBeta,
		PreviousStatus: previousStatus,
		AlertMessage:   alertMessage,
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
	}

	// Call callback if set
	if m.alertCallback != nil {
		m.alertCallback(alert)
	}

	// Broadcast alert if enabled
	if m.alertEnabled && m.broadcaster != nil {
		m.broadcaster.BroadcastBetaAlert(alert)
	}

	return alert
}

// ClearSession removes tracking data for a session.
func (m *BetaAlertMonitor) ClearSession(sessionID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if sessionID == "" {
		sessionID = "default"
	}
	delete(m.lastBeta, sessionID)
	delete(m.lastStatus, sessionID)
}

// -----------------------------------------------------------------------------
// Mock Event Broadcaster for Testing
// -----------------------------------------------------------------------------

// MockEventBroadcaster is a mock implementation of EventBroadcaster for testing.
type MockEventBroadcaster struct {
	mu                      sync.Mutex
	MeasurementEvents       []*MeasurementEvent
	MeasurementBatches      []*MeasurementBatchEvent
	SessionStartEvents      []*SessionStartEvent
	SessionEndEvents        []*SessionEndEvent
	ConversationTurnEvents  []*ConversationTurnEvent
	BetaAlerts              []*BetaAlertEvent
	BroadcastError          error
}

// NewMockEventBroadcaster creates a new MockEventBroadcaster.
func NewMockEventBroadcaster() *MockEventBroadcaster {
	return &MockEventBroadcaster{}
}

// BroadcastMeasurementEvent records the measurement event.
func (m *MockEventBroadcaster) BroadcastMeasurementEvent(event *MeasurementEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.BroadcastError != nil {
		return m.BroadcastError
	}
	m.MeasurementEvents = append(m.MeasurementEvents, event)
	return nil
}

// BroadcastMeasurementBatch records the batch event.
func (m *MockEventBroadcaster) BroadcastMeasurementBatch(batch *MeasurementBatchEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.BroadcastError != nil {
		return m.BroadcastError
	}
	m.MeasurementBatches = append(m.MeasurementBatches, batch)
	return nil
}

// BroadcastSessionStart records the session start event.
func (m *MockEventBroadcaster) BroadcastSessionStart(event *SessionStartEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.BroadcastError != nil {
		return m.BroadcastError
	}
	m.SessionStartEvents = append(m.SessionStartEvents, event)
	return nil
}

// BroadcastSessionEnd records the session end event.
func (m *MockEventBroadcaster) BroadcastSessionEnd(event *SessionEndEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.BroadcastError != nil {
		return m.BroadcastError
	}
	m.SessionEndEvents = append(m.SessionEndEvents, event)
	return nil
}

// BroadcastConversationTurn records the conversation turn event.
func (m *MockEventBroadcaster) BroadcastConversationTurn(event *ConversationTurnEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.BroadcastError != nil {
		return m.BroadcastError
	}
	m.ConversationTurnEvents = append(m.ConversationTurnEvents, event)
	return nil
}

// BroadcastBetaAlert records the beta alert.
func (m *MockEventBroadcaster) BroadcastBetaAlert(alert *BetaAlertEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.BroadcastError != nil {
		return m.BroadcastError
	}
	m.BetaAlerts = append(m.BetaAlerts, alert)
	return nil
}

// Reset clears all recorded events.
func (m *MockEventBroadcaster) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.MeasurementEvents = nil
	m.MeasurementBatches = nil
	m.SessionStartEvents = nil
	m.SessionEndEvents = nil
	m.ConversationTurnEvents = nil
	m.BetaAlerts = nil
	m.BroadcastError = nil
}
