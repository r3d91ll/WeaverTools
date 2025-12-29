// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"testing"
	"time"
)

// -----------------------------------------------------------------------------
// BetaStatus Tests
// -----------------------------------------------------------------------------

func TestBetaStatus_IsValid(t *testing.T) {
	tests := []struct {
		name   string
		status BetaStatus
		want   bool
	}{
		{"optimal is valid", BetaOptimal, true},
		{"monitor is valid", BetaMonitor, true},
		{"concerning is valid", BetaConcerning, true},
		{"critical is valid", BetaCritical, true},
		{"unknown is valid", BetaUnknown, true},
		{"empty is invalid", BetaStatus(""), false},
		{"arbitrary string is invalid", BetaStatus("foo"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.status.IsValid(); got != tt.want {
				t.Errorf("BetaStatus.IsValid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestComputeBetaStatus(t *testing.T) {
	tests := []struct {
		name string
		beta float64
		want BetaStatus
	}{
		{"negative beta", -1.0, BetaUnknown},
		{"zero beta", 0.0, BetaUnknown},
		{"very low beta", 0.5, BetaUnknown},
		{"below optimal", 1.4, BetaUnknown},
		{"optimal lower bound", 1.5, BetaOptimal},
		{"optimal mid", 1.7, BetaOptimal},
		{"optimal upper bound", 1.99, BetaOptimal},
		{"monitor lower bound", 2.0, BetaMonitor},
		{"monitor mid", 2.2, BetaMonitor},
		{"monitor upper bound", 2.49, BetaMonitor},
		{"concerning lower bound", 2.5, BetaConcerning},
		{"concerning mid", 2.7, BetaConcerning},
		{"concerning upper bound", 2.99, BetaConcerning},
		{"critical lower bound", 3.0, BetaCritical},
		{"critical high", 5.0, BetaCritical},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ComputeBetaStatus(tt.beta); got != tt.want {
				t.Errorf("ComputeBetaStatus(%v) = %v, want %v", tt.beta, got, tt.want)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// MeasurementEvent Tests
// -----------------------------------------------------------------------------

func TestNewMeasurementEvent(t *testing.T) {
	event := NewMeasurementEvent("test-id", 5, 64.0, 1.8, 0.95, 0.88)

	if event.ID != "test-id" {
		t.Errorf("expected ID test-id, got %s", event.ID)
	}
	if event.Turn != 5 {
		t.Errorf("expected Turn 5, got %d", event.Turn)
	}
	if event.Deff != 64.0 {
		t.Errorf("expected Deff 64.0, got %f", event.Deff)
	}
	if event.Beta != 1.8 {
		t.Errorf("expected Beta 1.8, got %f", event.Beta)
	}
	if event.Alignment != 0.95 {
		t.Errorf("expected Alignment 0.95, got %f", event.Alignment)
	}
	if event.Cpair != 0.88 {
		t.Errorf("expected Cpair 0.88, got %f", event.Cpair)
	}
	if event.BetaStatus != BetaOptimal {
		t.Errorf("expected BetaStatus optimal, got %s", event.BetaStatus)
	}
	if event.Timestamp == "" {
		t.Error("expected Timestamp to be set")
	}
}

func TestMeasurementEvent_SetParticipants(t *testing.T) {
	event := NewMeasurementEvent("test-id", 1, 64.0, 1.8, 0.95, 0.88)
	event.SetParticipants("sender-1", "Alice", "human", "receiver-1", "Bob", "agent")

	if event.SenderID != "sender-1" {
		t.Errorf("expected SenderID sender-1, got %s", event.SenderID)
	}
	if event.SenderName != "Alice" {
		t.Errorf("expected SenderName Alice, got %s", event.SenderName)
	}
	if event.SenderRole != "human" {
		t.Errorf("expected SenderRole human, got %s", event.SenderRole)
	}
	if event.ReceiverID != "receiver-1" {
		t.Errorf("expected ReceiverID receiver-1, got %s", event.ReceiverID)
	}
	if event.ReceiverName != "Bob" {
		t.Errorf("expected ReceiverName Bob, got %s", event.ReceiverName)
	}
	if event.ReceiverRole != "agent" {
		t.Errorf("expected ReceiverRole agent, got %s", event.ReceiverRole)
	}
}

func TestMeasurementEvent_SetSession(t *testing.T) {
	event := NewMeasurementEvent("test-id", 1, 64.0, 1.8, 0.95, 0.88)
	event.SetSession("session-123", "conv-456")

	if event.SessionID != "session-123" {
		t.Errorf("expected SessionID session-123, got %s", event.SessionID)
	}
	if event.ConversationID != "conv-456" {
		t.Errorf("expected ConversationID conv-456, got %s", event.ConversationID)
	}
}

func TestMeasurementEvent_ToMeasurementData(t *testing.T) {
	event := NewMeasurementEvent("test-id", 5, 64.0, 1.8, 0.95, 0.88)
	event.SetParticipants("s1", "Alice", "human", "r1", "Bob", "agent")

	data := event.ToMeasurementData()

	if data.Turn != 5 {
		t.Errorf("expected Turn 5, got %d", data.Turn)
	}
	if data.Deff != 64.0 {
		t.Errorf("expected Deff 64.0, got %f", data.Deff)
	}
	if data.Beta != 1.8 {
		t.Errorf("expected Beta 1.8, got %f", data.Beta)
	}
	if data.Alignment != 0.95 {
		t.Errorf("expected Alignment 0.95, got %f", data.Alignment)
	}
	if data.Cpair != 0.88 {
		t.Errorf("expected Cpair 0.88, got %f", data.Cpair)
	}
	if data.Sender != "Alice" {
		t.Errorf("expected Sender Alice, got %s", data.Sender)
	}
	if data.Receiver != "Bob" {
		t.Errorf("expected Receiver Bob, got %s", data.Receiver)
	}
}

// -----------------------------------------------------------------------------
// MockEventBroadcaster Tests
// -----------------------------------------------------------------------------

func TestMockEventBroadcaster_BroadcastMeasurementEvent(t *testing.T) {
	mock := NewMockEventBroadcaster()
	event := NewMeasurementEvent("test-id", 1, 64.0, 1.8, 0.95, 0.88)

	err := mock.BroadcastMeasurementEvent(event)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(mock.MeasurementEvents) != 1 {
		t.Errorf("expected 1 event, got %d", len(mock.MeasurementEvents))
	}
	if mock.MeasurementEvents[0].ID != "test-id" {
		t.Errorf("expected event ID test-id, got %s", mock.MeasurementEvents[0].ID)
	}
}

func TestMockEventBroadcaster_BroadcastMeasurementBatch(t *testing.T) {
	mock := NewMockEventBroadcaster()
	batch := &MeasurementBatchEvent{
		Measurements: []MeasurementEvent{
			*NewMeasurementEvent("id-1", 1, 64.0, 1.8, 0.95, 0.88),
			*NewMeasurementEvent("id-2", 2, 65.0, 1.9, 0.96, 0.89),
		},
		SessionID: "session-1",
		TurnRange: [2]int{1, 2},
	}

	err := mock.BroadcastMeasurementBatch(batch)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(mock.MeasurementBatches) != 1 {
		t.Errorf("expected 1 batch, got %d", len(mock.MeasurementBatches))
	}
	if len(mock.MeasurementBatches[0].Measurements) != 2 {
		t.Errorf("expected 2 measurements in batch, got %d", len(mock.MeasurementBatches[0].Measurements))
	}
}

func TestMockEventBroadcaster_BroadcastSessionStart(t *testing.T) {
	mock := NewMockEventBroadcaster()
	event := &SessionStartEvent{
		SessionID: "session-1",
		Name:      "Test Session",
		StartedAt: time.Now().UTC().Format(time.RFC3339),
	}

	err := mock.BroadcastSessionStart(event)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(mock.SessionStartEvents) != 1 {
		t.Errorf("expected 1 event, got %d", len(mock.SessionStartEvents))
	}
	if mock.SessionStartEvents[0].SessionID != "session-1" {
		t.Errorf("expected session ID session-1, got %s", mock.SessionStartEvents[0].SessionID)
	}
}

func TestMockEventBroadcaster_BroadcastSessionEnd(t *testing.T) {
	mock := NewMockEventBroadcaster()
	event := &SessionEndEvent{
		SessionID:  "session-1",
		EndedAt:    time.Now().UTC().Format(time.RFC3339),
		TotalTurns: 10,
	}

	err := mock.BroadcastSessionEnd(event)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(mock.SessionEndEvents) != 1 {
		t.Errorf("expected 1 event, got %d", len(mock.SessionEndEvents))
	}
	if mock.SessionEndEvents[0].TotalTurns != 10 {
		t.Errorf("expected 10 turns, got %d", mock.SessionEndEvents[0].TotalTurns)
	}
}

func TestMockEventBroadcaster_BroadcastConversationTurn(t *testing.T) {
	mock := NewMockEventBroadcaster()
	event := &ConversationTurnEvent{
		SessionID:      "session-1",
		ConversationID: "conv-1",
		Turn:           5,
		SenderName:     "Alice",
		ReceiverName:   "Bob",
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
	}

	err := mock.BroadcastConversationTurn(event)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(mock.ConversationTurnEvents) != 1 {
		t.Errorf("expected 1 event, got %d", len(mock.ConversationTurnEvents))
	}
	if mock.ConversationTurnEvents[0].Turn != 5 {
		t.Errorf("expected turn 5, got %d", mock.ConversationTurnEvents[0].Turn)
	}
}

func TestMockEventBroadcaster_BroadcastBetaAlert(t *testing.T) {
	mock := NewMockEventBroadcaster()
	alert := &BetaAlertEvent{
		MeasurementID: "m-1",
		SessionID:     "session-1",
		Turn:          5,
		Beta:          3.5,
		Status:        BetaCritical,
		AlertMessage:  "Critical beta level",
		Timestamp:     time.Now().UTC().Format(time.RFC3339),
	}

	err := mock.BroadcastBetaAlert(alert)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(mock.BetaAlerts) != 1 {
		t.Errorf("expected 1 alert, got %d", len(mock.BetaAlerts))
	}
	if mock.BetaAlerts[0].Status != BetaCritical {
		t.Errorf("expected critical status, got %s", mock.BetaAlerts[0].Status)
	}
}

func TestMockEventBroadcaster_BroadcastError(t *testing.T) {
	mock := NewMockEventBroadcaster()
	mock.BroadcastError = errTest

	err := mock.BroadcastMeasurementEvent(NewMeasurementEvent("id", 1, 1, 1, 1, 1))
	if err != errTest {
		t.Errorf("expected errTest, got %v", err)
	}

	if len(mock.MeasurementEvents) != 0 {
		t.Errorf("expected 0 events, got %d", len(mock.MeasurementEvents))
	}
}

func TestMockEventBroadcaster_Reset(t *testing.T) {
	mock := NewMockEventBroadcaster()
	mock.BroadcastMeasurementEvent(NewMeasurementEvent("id", 1, 1, 1, 1, 1))
	mock.BroadcastBetaAlert(&BetaAlertEvent{})
	mock.BroadcastError = errTest

	mock.Reset()

	if len(mock.MeasurementEvents) != 0 {
		t.Errorf("expected 0 measurement events after reset")
	}
	if len(mock.BetaAlerts) != 0 {
		t.Errorf("expected 0 beta alerts after reset")
	}
	if mock.BroadcastError != nil {
		t.Errorf("expected nil BroadcastError after reset")
	}
}

// errTest is a test error for error handling tests
var errTest = &testError{}

type testError struct{}

func (e *testError) Error() string { return "test error" }

// -----------------------------------------------------------------------------
// EventAggregator Tests
// -----------------------------------------------------------------------------

func TestEventAggregator_Add(t *testing.T) {
	agg := NewEventAggregator("session-1", 10)

	for i := 0; i < 5; i++ {
		agg.Add(*NewMeasurementEvent("id", i, 64.0, 1.8, 0.95, 0.88))
	}

	if agg.Count() != 5 {
		t.Errorf("expected 5 measurements, got %d", agg.Count())
	}
}

func TestEventAggregator_Add_MaxSize(t *testing.T) {
	agg := NewEventAggregator("session-1", 3)

	for i := 0; i < 5; i++ {
		agg.Add(*NewMeasurementEvent("id", i, 64.0, float64(i)+1.5, 0.95, 0.88))
	}

	if agg.Count() != 3 {
		t.Errorf("expected 3 measurements (max size), got %d", agg.Count())
	}

	// Verify oldest were removed
	all := agg.GetAll()
	if all[0].Turn != 2 {
		t.Errorf("expected first Turn to be 2 after overflow, got %d", all[0].Turn)
	}
}

func TestEventAggregator_DefaultMaxSize(t *testing.T) {
	agg := NewEventAggregator("session-1", 0)

	// Should use default max size of 1000
	if agg.maxSize != 1000 {
		t.Errorf("expected default maxSize of 1000, got %d", agg.maxSize)
	}
}

func TestEventAggregator_Clear(t *testing.T) {
	agg := NewEventAggregator("session-1", 10)

	for i := 0; i < 5; i++ {
		agg.Add(*NewMeasurementEvent("id", i, 64.0, 1.8, 0.95, 0.88))
	}

	agg.Clear()

	if agg.Count() != 0 {
		t.Errorf("expected 0 measurements after clear, got %d", agg.Count())
	}
}

func TestEventAggregator_GetAll(t *testing.T) {
	agg := NewEventAggregator("session-1", 10)

	for i := 0; i < 3; i++ {
		agg.Add(*NewMeasurementEvent("id", i, 64.0, 1.8, 0.95, 0.88))
	}

	all := agg.GetAll()

	if len(all) != 3 {
		t.Errorf("expected 3 measurements, got %d", len(all))
	}

	// Verify it's a copy (modify and check original)
	all[0].Turn = 999
	original := agg.GetAll()
	if original[0].Turn == 999 {
		t.Error("GetAll should return a copy, not the original slice")
	}
}

func TestEventAggregator_GetSummary(t *testing.T) {
	agg := NewEventAggregator("session-1", 10)

	// Add measurements with varied beta values
	agg.Add(*NewMeasurementEvent("id-1", 1, 60.0, 1.8, 0.90, 0.85))
	agg.Add(*NewMeasurementEvent("id-2", 2, 65.0, 2.0, 0.92, 0.87))
	agg.Add(*NewMeasurementEvent("id-3", 3, 70.0, 2.7, 0.94, 0.89)) // concerning
	agg.Add(*NewMeasurementEvent("id-4", 4, 75.0, 3.5, 0.96, 0.91)) // critical

	summary := agg.GetSummary()

	if summary.TotalMeasurements != 4 {
		t.Errorf("expected 4 total measurements, got %d", summary.TotalMeasurements)
	}
	if summary.MinBeta != 1.8 {
		t.Errorf("expected MinBeta 1.8, got %f", summary.MinBeta)
	}
	if summary.MaxBeta != 3.5 {
		t.Errorf("expected MaxBeta 3.5, got %f", summary.MaxBeta)
	}
	if summary.BetaAlertCount != 2 {
		t.Errorf("expected 2 beta alerts (concerning + critical), got %d", summary.BetaAlertCount)
	}

	expectedAvgBeta := (1.8 + 2.0 + 2.7 + 3.5) / 4.0
	if summary.AvgBeta != expectedAvgBeta {
		t.Errorf("expected AvgBeta %f, got %f", expectedAvgBeta, summary.AvgBeta)
	}
}

func TestEventAggregator_GetSummary_Empty(t *testing.T) {
	agg := NewEventAggregator("session-1", 10)

	summary := agg.GetSummary()

	if summary.TotalMeasurements != 0 {
		t.Errorf("expected 0 total measurements for empty aggregator, got %d", summary.TotalMeasurements)
	}
}

func TestEventAggregator_GetBatch(t *testing.T) {
	agg := NewEventAggregator("session-1", 10)

	agg.Add(*NewMeasurementEvent("id-1", 3, 60.0, 1.8, 0.90, 0.85))
	agg.Add(*NewMeasurementEvent("id-2", 1, 65.0, 2.0, 0.92, 0.87))
	agg.Add(*NewMeasurementEvent("id-3", 5, 70.0, 2.2, 0.94, 0.89))

	batch := agg.GetBatch()

	if batch.SessionID != "session-1" {
		t.Errorf("expected SessionID session-1, got %s", batch.SessionID)
	}
	if len(batch.Measurements) != 3 {
		t.Errorf("expected 3 measurements in batch, got %d", len(batch.Measurements))
	}
	if batch.TurnRange[0] != 1 {
		t.Errorf("expected TurnRange[0] to be 1, got %d", batch.TurnRange[0])
	}
	if batch.TurnRange[1] != 5 {
		t.Errorf("expected TurnRange[1] to be 5, got %d", batch.TurnRange[1])
	}
}

func TestEventAggregator_GetBatch_Empty(t *testing.T) {
	agg := NewEventAggregator("session-1", 10)

	batch := agg.GetBatch()

	if batch.SessionID != "session-1" {
		t.Errorf("expected SessionID session-1, got %s", batch.SessionID)
	}
	if len(batch.Measurements) != 0 {
		t.Errorf("expected 0 measurements in empty batch, got %d", len(batch.Measurements))
	}
	if batch.TurnRange[0] != 0 || batch.TurnRange[1] != 0 {
		t.Errorf("expected TurnRange [0, 0] for empty batch, got %v", batch.TurnRange)
	}
}

// -----------------------------------------------------------------------------
// BetaAlertMonitor Tests
// -----------------------------------------------------------------------------

func TestBetaAlertMonitor_CheckMeasurement_NoAlert(t *testing.T) {
	mock := NewMockEventBroadcaster()
	monitor := NewBetaAlertMonitor(mock)

	// Optimal beta should not trigger alert
	event := NewMeasurementEvent("id-1", 1, 64.0, 1.8, 0.95, 0.88)
	alert := monitor.CheckMeasurement(event)

	if alert != nil {
		t.Errorf("expected no alert for optimal beta, got %+v", alert)
	}
	if len(mock.BetaAlerts) != 0 {
		t.Errorf("expected 0 broadcasts, got %d", len(mock.BetaAlerts))
	}
}

func TestBetaAlertMonitor_CheckMeasurement_ConcerningAlert(t *testing.T) {
	mock := NewMockEventBroadcaster()
	monitor := NewBetaAlertMonitor(mock)

	event := NewMeasurementEvent("id-1", 1, 64.0, 2.7, 0.95, 0.88)
	event.SessionID = "session-1"
	alert := monitor.CheckMeasurement(event)

	if alert == nil {
		t.Fatal("expected alert for concerning beta")
	}
	if alert.Status != BetaConcerning {
		t.Errorf("expected concerning status, got %s", alert.Status)
	}
	if len(mock.BetaAlerts) != 1 {
		t.Errorf("expected 1 broadcast, got %d", len(mock.BetaAlerts))
	}
}

func TestBetaAlertMonitor_CheckMeasurement_CriticalAlert(t *testing.T) {
	mock := NewMockEventBroadcaster()
	monitor := NewBetaAlertMonitor(mock)

	event := NewMeasurementEvent("id-1", 1, 64.0, 3.5, 0.95, 0.88)
	alert := monitor.CheckMeasurement(event)

	if alert == nil {
		t.Fatal("expected alert for critical beta")
	}
	if alert.Status != BetaCritical {
		t.Errorf("expected critical status, got %s", alert.Status)
	}
	if alert.AlertMessage == "" {
		t.Error("expected alert message to be set")
	}
}

func TestBetaAlertMonitor_CheckMeasurement_TracksPrevious(t *testing.T) {
	mock := NewMockEventBroadcaster()
	monitor := NewBetaAlertMonitor(mock)

	// First measurement (optimal)
	event1 := NewMeasurementEvent("id-1", 1, 64.0, 1.8, 0.95, 0.88)
	event1.SessionID = "session-1"
	monitor.CheckMeasurement(event1)

	// Second measurement (critical)
	event2 := NewMeasurementEvent("id-2", 2, 64.0, 3.5, 0.95, 0.88)
	event2.SessionID = "session-1"
	alert := monitor.CheckMeasurement(event2)

	if alert == nil {
		t.Fatal("expected alert")
	}
	if alert.PreviousBeta != 1.8 {
		t.Errorf("expected PreviousBeta 1.8, got %f", alert.PreviousBeta)
	}
	if alert.PreviousStatus != BetaOptimal {
		t.Errorf("expected PreviousStatus optimal, got %s", alert.PreviousStatus)
	}
}

func TestBetaAlertMonitor_CheckMeasurement_AlertsDisabled(t *testing.T) {
	mock := NewMockEventBroadcaster()
	monitor := NewBetaAlertMonitor(mock)
	monitor.EnableAlerts(false)

	event := NewMeasurementEvent("id-1", 1, 64.0, 3.5, 0.95, 0.88)
	alert := monitor.CheckMeasurement(event)

	// Alert is still returned for local use
	if alert == nil {
		t.Fatal("expected alert to be returned even when broadcasts disabled")
	}

	// But no broadcast should occur
	if len(mock.BetaAlerts) != 0 {
		t.Errorf("expected 0 broadcasts when alerts disabled, got %d", len(mock.BetaAlerts))
	}
}

func TestBetaAlertMonitor_SetAlertCallback(t *testing.T) {
	mock := NewMockEventBroadcaster()
	monitor := NewBetaAlertMonitor(mock)

	var callbackAlert *BetaAlertEvent
	monitor.SetAlertCallback(func(alert *BetaAlertEvent) {
		callbackAlert = alert
	})

	event := NewMeasurementEvent("id-1", 1, 64.0, 3.5, 0.95, 0.88)
	monitor.CheckMeasurement(event)

	if callbackAlert == nil {
		t.Fatal("expected callback to be called")
	}
	if callbackAlert.Beta != 3.5 {
		t.Errorf("expected Beta 3.5 in callback, got %f", callbackAlert.Beta)
	}
}

func TestBetaAlertMonitor_ClearSession(t *testing.T) {
	mock := NewMockEventBroadcaster()
	monitor := NewBetaAlertMonitor(mock)

	// Add some tracking data
	event := NewMeasurementEvent("id-1", 1, 64.0, 1.8, 0.95, 0.88)
	event.SessionID = "session-1"
	monitor.CheckMeasurement(event)

	// Clear the session
	monitor.ClearSession("session-1")

	// Next measurement should have no previous data
	event2 := NewMeasurementEvent("id-2", 2, 64.0, 3.5, 0.95, 0.88)
	event2.SessionID = "session-1"
	alert := monitor.CheckMeasurement(event2)

	if alert.PreviousBeta != 0 {
		t.Errorf("expected PreviousBeta 0 after clear, got %f", alert.PreviousBeta)
	}
}

func TestBetaAlertMonitor_ClearSession_Default(t *testing.T) {
	mock := NewMockEventBroadcaster()
	monitor := NewBetaAlertMonitor(mock)

	// Add measurement with no session (uses "default")
	event := NewMeasurementEvent("id-1", 1, 64.0, 1.8, 0.95, 0.88)
	monitor.CheckMeasurement(event)

	// Clear with empty string (should clear "default")
	monitor.ClearSession("")

	// Next measurement should have no previous data
	event2 := NewMeasurementEvent("id-2", 2, 64.0, 3.5, 0.95, 0.88)
	alert := monitor.CheckMeasurement(event2)

	if alert.PreviousBeta != 0 {
		t.Errorf("expected PreviousBeta 0 after clear, got %f", alert.PreviousBeta)
	}
}

// -----------------------------------------------------------------------------
// HubEventBroadcaster Tests
// -----------------------------------------------------------------------------

func TestHubEventBroadcaster_BroadcastMeasurementEvent(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	broadcaster := NewHubEventBroadcaster(hub)
	event := NewMeasurementEvent("test-id", 1, 64.0, 1.8, 0.95, 0.88)

	// Should not error even with no connected clients
	err := broadcaster.BroadcastMeasurementEvent(event)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestHubEventBroadcaster_BroadcastMeasurementBatch(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	broadcaster := NewHubEventBroadcaster(hub)
	batch := &MeasurementBatchEvent{
		Measurements: []MeasurementEvent{
			*NewMeasurementEvent("id-1", 1, 64.0, 1.8, 0.95, 0.88),
		},
		SessionID: "session-1",
		TurnRange: [2]int{1, 1},
	}

	err := broadcaster.BroadcastMeasurementBatch(batch)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestHubEventBroadcaster_BroadcastSessionStart(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	broadcaster := NewHubEventBroadcaster(hub)
	event := &SessionStartEvent{
		SessionID: "session-1",
		Name:      "Test Session",
		StartedAt: time.Now().UTC().Format(time.RFC3339),
	}

	err := broadcaster.BroadcastSessionStart(event)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestHubEventBroadcaster_BroadcastSessionEnd(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	broadcaster := NewHubEventBroadcaster(hub)
	event := &SessionEndEvent{
		SessionID:  "session-1",
		EndedAt:    time.Now().UTC().Format(time.RFC3339),
		TotalTurns: 10,
	}

	err := broadcaster.BroadcastSessionEnd(event)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestHubEventBroadcaster_BroadcastConversationTurn(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	broadcaster := NewHubEventBroadcaster(hub)
	event := &ConversationTurnEvent{
		SessionID:      "session-1",
		ConversationID: "conv-1",
		Turn:           5,
		SenderName:     "Alice",
		ReceiverName:   "Bob",
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
	}

	err := broadcaster.BroadcastConversationTurn(event)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestHubEventBroadcaster_BroadcastBetaAlert(t *testing.T) {
	hub := NewHub()
	go hub.Run()
	defer hub.Stop()

	broadcaster := NewHubEventBroadcaster(hub)
	alert := &BetaAlertEvent{
		MeasurementID: "m-1",
		SessionID:     "session-1",
		Turn:          5,
		Beta:          3.5,
		Status:        BetaCritical,
		AlertMessage:  "Critical beta level",
		Timestamp:     time.Now().UTC().Format(time.RFC3339),
	}

	err := broadcaster.BroadcastBetaAlert(alert)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
