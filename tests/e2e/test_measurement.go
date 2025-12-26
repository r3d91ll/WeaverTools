package e2e

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/r3d91ll/weaver/pkg/backend"
	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/wool"
	"github.com/r3d91ll/yarn"
)

// TestMeasurementCreation tests creating and validating measurements.
func TestMeasurementCreation(t *testing.T) {
	tests := []struct {
		name          string
		setupFunc     func() *yarn.Measurement
		wantValid     bool
		wantField     string
		wantMessage   string
	}{
		{
			name: "valid measurement with sender",
			setupFunc: func() *yarn.Measurement {
				m := yarn.NewMeasurement()
				m.SenderID = "agent-1"
				m.SenderName = "Senior"
				m.SenderRole = "assistant"
				return m
			},
			wantValid: true,
		},
		{
			name: "valid measurement with receiver",
			setupFunc: func() *yarn.Measurement {
				m := yarn.NewMeasurement()
				m.ReceiverID = "agent-2"
				m.ReceiverName = "Junior"
				m.ReceiverRole = "assistant"
				return m
			},
			wantValid: true,
		},
		{
			name: "valid measurement with both participants",
			setupFunc: func() *yarn.Measurement {
				m := yarn.NewMeasurement()
				m.SetSender("agent-1", "Senior", "assistant", nil)
				m.SetReceiver("agent-2", "Junior", "assistant", nil)
				return m
			},
			wantValid: true,
		},
		{
			name: "valid measurement with metrics",
			setupFunc: func() *yarn.Measurement {
				m := yarn.NewMeasurement()
				m.SenderID = "agent-1"
				m.DEff = 100
				m.Beta = 1.8
				m.Alignment = 0.75
				m.BetaStatus = yarn.BetaOptimal
				return m
			},
			wantValid: true,
		},
		{
			name: "invalid measurement without participant",
			setupFunc: func() *yarn.Measurement {
				return yarn.NewMeasurement()
			},
			wantValid:   false,
			wantField:   "sender_id",
			wantMessage: "at least one of sender_id or receiver_id is required",
		},
		{
			name: "invalid measurement with negative DEff",
			setupFunc: func() *yarn.Measurement {
				m := yarn.NewMeasurement()
				m.SenderID = "agent-1"
				m.DEff = -1
				return m
			},
			wantValid:   false,
			wantField:   "d_eff",
			wantMessage: "d_eff must be non-negative",
		},
		{
			name: "invalid measurement with alignment out of range",
			setupFunc: func() *yarn.Measurement {
				m := yarn.NewMeasurement()
				m.SenderID = "agent-1"
				m.Alignment = 1.5
				return m
			},
			wantValid:   false,
			wantField:   "alignment",
			wantMessage: "alignment must be in range [-1, 1]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.setupFunc()
			err := m.Validate()

			if tt.wantValid {
				if err != nil {
					t.Errorf("Measurement.Validate() unexpected error: %v", err)
				}
			} else {
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
			}
		})
	}
}

// TestMeasurementBetaStatusComputation tests the ComputeBetaStatus function.
func TestMeasurementBetaStatusComputation(t *testing.T) {
	tests := []struct {
		name   string
		beta   float64
		want   yarn.BetaStatus
	}{
		// Unknown range (invalid or out of optimal range)
		{"zero beta", 0.0, yarn.BetaUnknown},
		{"negative beta", -1.0, yarn.BetaUnknown},
		{"very low beta", 0.5, yarn.BetaUnknown},
		{"low beta", 1.0, yarn.BetaUnknown},
		{"just below optimal", 1.49, yarn.BetaUnknown},

		// Optimal range [1.5, 2.0)
		{"optimal lower bound", 1.5, yarn.BetaOptimal},
		{"optimal mid-range", 1.75, yarn.BetaOptimal},
		{"optimal high", 1.99, yarn.BetaOptimal},

		// Monitor range [2.0, 2.5)
		{"monitor lower bound", 2.0, yarn.BetaMonitor},
		{"monitor mid-range", 2.25, yarn.BetaMonitor},
		{"monitor high", 2.49, yarn.BetaMonitor},

		// Concerning range [2.5, 3.0)
		{"concerning lower bound", 2.5, yarn.BetaConcerning},
		{"concerning mid-range", 2.75, yarn.BetaConcerning},
		{"concerning high", 2.99, yarn.BetaConcerning},

		// Critical range [3.0, ∞)
		{"critical lower bound", 3.0, yarn.BetaCritical},
		{"critical high", 5.0, yarn.BetaCritical},
		{"critical very high", 10.0, yarn.BetaCritical},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := yarn.ComputeBetaStatus(tt.beta); got != tt.want {
				t.Errorf("ComputeBetaStatus(%v) = %q, want %q", tt.beta, got, tt.want)
			}
		})
	}
}

// TestMeasurementBilateralDetection tests the IsBilateral method.
func TestMeasurementBilateralDetection(t *testing.T) {
	tests := []struct {
		name         string
		senderHidden *yarn.HiddenState
		recvHidden   *yarn.HiddenState
		wantBilateral bool
	}{
		{
			name:          "no hidden states",
			senderHidden:  nil,
			recvHidden:    nil,
			wantBilateral: false,
		},
		{
			name:          "sender only",
			senderHidden:  &yarn.HiddenState{Vector: make([]float32, 4096)},
			recvHidden:    nil,
			wantBilateral: false,
		},
		{
			name:          "receiver only",
			senderHidden:  nil,
			recvHidden:    &yarn.HiddenState{Vector: make([]float32, 4096)},
			wantBilateral: false,
		},
		{
			name:          "both hidden states",
			senderHidden:  &yarn.HiddenState{Vector: make([]float32, 4096)},
			recvHidden:    &yarn.HiddenState{Vector: make([]float32, 4096)},
			wantBilateral: true,
		},
		{
			name:          "sender empty vector",
			senderHidden:  &yarn.HiddenState{Vector: []float32{}},
			recvHidden:    &yarn.HiddenState{Vector: make([]float32, 4096)},
			wantBilateral: false,
		},
		{
			name:          "receiver empty vector",
			senderHidden:  &yarn.HiddenState{Vector: make([]float32, 4096)},
			recvHidden:    &yarn.HiddenState{Vector: []float32{}},
			wantBilateral: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := yarn.NewMeasurement()
			m.SenderID = "agent-1"
			m.SenderHidden = tt.senderHidden
			m.ReceiverHidden = tt.recvHidden

			if got := m.IsBilateral(); got != tt.wantBilateral {
				t.Errorf("Measurement.IsBilateral() = %v, want %v", got, tt.wantBilateral)
			}
		})
	}
}

// TestMeasurementForTurnCreation tests NewMeasurementForTurn.
func TestMeasurementForTurnCreation(t *testing.T) {
	sessionID := "session-123"
	convID := "conv-456"
	turnNumber := 5

	m := yarn.NewMeasurementForTurn(sessionID, convID, turnNumber)

	if m.ID == "" {
		t.Error("NewMeasurementForTurn() ID should not be empty")
	}
	if m.Timestamp.IsZero() {
		t.Error("NewMeasurementForTurn() Timestamp should not be zero")
	}
	if m.SessionID != sessionID {
		t.Errorf("NewMeasurementForTurn() SessionID = %q, want %q", m.SessionID, sessionID)
	}
	if m.ConversationID != convID {
		t.Errorf("NewMeasurementForTurn() ConversationID = %q, want %q", m.ConversationID, convID)
	}
	if m.TurnNumber != turnNumber {
		t.Errorf("NewMeasurementForTurn() TurnNumber = %d, want %d", m.TurnNumber, turnNumber)
	}

	// Should require a participant to be valid
	err := m.Validate()
	if err == nil {
		t.Error("NewMeasurementForTurn() should require a participant")
	}

	m.SenderID = "agent-1"
	if err := m.Validate(); err != nil {
		t.Errorf("NewMeasurementForTurn() with participant should be valid: %v", err)
	}
}

// mockLoomServerWithMeasurement creates a mock Loom server for measurement testing.
func mockLoomServerWithMeasurement(t *testing.T, hiddenDim int) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"status": "ok"}`))

		case "/v1/chat/completions":
			body, err := io.ReadAll(r.Body)
			if err != nil {
				http.Error(w, "Bad request", http.StatusBadRequest)
				return
			}
			defer r.Body.Close()

			var req struct {
				Model              string `json:"model"`
				ReturnHiddenStates bool   `json:"return_hidden_states"`
			}
			if err := json.Unmarshal(body, &req); err != nil {
				http.Error(w, "Bad request", http.StatusBadRequest)
				return
			}

			resp := map[string]any{
				"text": "Mock response for measurement testing",
				"usage": map[string]int{
					"prompt_tokens":     15,
					"completion_tokens": 10,
					"total_tokens":      25,
				},
			}

			if req.ReturnHiddenStates {
				mockVector := make([]float32, hiddenDim)
				for i := range mockVector {
					mockVector[i] = float32(i%100) * 0.01
				}

				resp["hidden_state"] = map[string]any{
					"final": mockVector,
					"shape": []int{1, hiddenDim},
					"layer": -1,
					"dtype": "float32",
				}
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)

		default:
			http.NotFound(w, r)
		}
	}))
}

// TestMeasurementWorkflowWithSession tests the full measurement workflow in a session.
func TestMeasurementWorkflowWithSession(t *testing.T) {
	// Create a session
	session := yarn.NewSession("measurement-test", "E2E measurement workflow test")

	// Verify session is valid initially
	if err := session.Validate(); err != nil {
		t.Fatalf("NewSession() created invalid session: %v", err)
	}

	// Create conversation
	conv := yarn.NewConversation("measurement-conv")
	session.AddConversation(conv)

	// Add user message
	userMsg := yarn.NewAgentMessage(yarn.RoleUser, "Explain conveyance theory", "user", "user")
	conv.Add(userMsg)

	// Create measurement for the turn
	m := yarn.NewMeasurementForTurn(session.ID, conv.ID, 1)
	m.SetSender("user", "User", "user", nil)

	// Simulate assistant response with hidden state
	assistantHidden := &yarn.HiddenState{
		Vector: make([]float32, 4096),
		Shape:  []int{1, 4096},
		Layer:  -1,
		DType:  "float32",
	}
	// Populate with mock values
	for i := range assistantHidden.Vector {
		assistantHidden.Vector[i] = float32(i%100) * 0.01
	}

	assistantMsg := yarn.NewAgentMessage(yarn.RoleAssistant, "Conveyance theory explains...", "junior-001", "junior")
	assistantMsg.HiddenState = assistantHidden
	conv.Add(assistantMsg)

	// Update measurement with receiver info
	m.SetReceiver("junior-001", "junior", "assistant", assistantHidden)
	m.DEff = 150
	m.Beta = 1.75
	m.Alignment = 0.85
	m.BetaStatus = yarn.ComputeBetaStatus(m.Beta)
	m.MessageContent = assistantMsg.Content
	m.TokenCount = 25

	// Add measurement to session
	session.AddMeasurement(m)

	// Verify session is still valid
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() after adding measurement: %v", err)
	}

	// Verify session stats
	stats := session.Stats()
	if stats.ConversationCount != 1 {
		t.Errorf("Stats.ConversationCount = %d, want 1", stats.ConversationCount)
	}
	if stats.MessageCount != 2 {
		t.Errorf("Stats.MessageCount = %d, want 2", stats.MessageCount)
	}
	if stats.MeasurementCount != 1 {
		t.Errorf("Stats.MeasurementCount = %d, want 1", stats.MeasurementCount)
	}
	if stats.AvgBeta != m.Beta {
		t.Errorf("Stats.AvgBeta = %v, want %v", stats.AvgBeta, m.Beta)
	}
	if stats.AvgDEff != float64(m.DEff) {
		t.Errorf("Stats.AvgDEff = %v, want %v", stats.AvgDEff, float64(m.DEff))
	}
}

// TestMeasurementWithMockBackend tests measurement capture with mock backend.
func TestMeasurementWithMockBackend(t *testing.T) {
	server := mockLoomServerWithMeasurement(t, 4096)
	defer server.Close()

	// Setup registry and backend
	registry := backend.NewRegistry()
	loomBackend := backend.NewLoom(backend.LoomConfig{
		Name:    "loom",
		URL:     server.URL,
		Model:   "test-model",
		Timeout: 10 * time.Second,
	})
	registry.Register(loomBackend)

	// Create manager and agent
	manager := runtime.NewManager(registry)
	juniorDef := wool.DefaultJunior()
	juniorDef.ID = "junior-measure"
	juniorDef.Model = "test-model"

	agent, err := manager.Create(juniorDef)
	if err != nil {
		t.Fatalf("Failed to create agent: %v", err)
	}

	// Create session and conversation
	session := yarn.NewSession("backend-measurement-test", "Test measurements with backend")
	conv := yarn.NewConversation("backend-measurement-conv")
	session.AddConversation(conv)

	// Send message
	ctx := context.Background()
	userMsg := yarn.NewAgentMessage(yarn.RoleUser, "Test message", "user", "user")
	conv.Add(userMsg)

	resp, err := agent.Chat(ctx, conv.History(-1))
	if err != nil {
		t.Fatalf("Agent.Chat() failed: %v", err)
	}
	conv.Add(resp)

	// Create measurement from response
	m := yarn.NewMeasurementForTurn(session.ID, conv.ID, 1)
	m.SetSender("user", "User", "user", nil)

	if resp.HiddenState != nil {
		m.SetReceiver(resp.AgentID, resp.AgentName, string(resp.Role), resp.HiddenState)
	} else {
		m.SetReceiver(resp.AgentID, resp.AgentName, string(resp.Role), nil)
	}

	// Verify measurement has receiver hidden state
	if resp.HasHiddenState() {
		if m.ReceiverHidden == nil {
			t.Error("Measurement should have receiver hidden state when response has it")
		}
		if m.ReceiverHidden != nil && m.ReceiverHidden.Dimension() != 4096 {
			t.Errorf("ReceiverHidden.Dimension() = %d, want 4096", m.ReceiverHidden.Dimension())
		}
	}

	session.AddMeasurement(m)

	// Validate full session
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() failed: %v", err)
	}
}

// TestMeasurementMultipleTurns tests measurements across multiple conversation turns.
func TestMeasurementMultipleTurns(t *testing.T) {
	session := yarn.NewSession("multi-turn-test", "Multiple turn measurement test")
	conv := yarn.NewConversation("multi-turn-conv")
	session.AddConversation(conv)

	// Simulate multiple turns
	numTurns := 5
	for i := 0; i < numTurns; i++ {
		// User message
		userMsg := yarn.NewAgentMessage(yarn.RoleUser, "Turn "+string(rune('0'+i)), "user", "user")
		conv.Add(userMsg)

		// Assistant response with hidden state
		assistantHidden := &yarn.HiddenState{
			Vector: make([]float32, 4096),
			Shape:  []int{1, 4096},
			Layer:  -1,
			DType:  "float32",
		}
		for j := range assistantHidden.Vector {
			assistantHidden.Vector[j] = float32(i*100+j) * 0.001
		}

		assistantMsg := yarn.NewAgentMessage(yarn.RoleAssistant, "Response "+string(rune('0'+i)), "junior", "junior")
		assistantMsg.HiddenState = assistantHidden
		conv.Add(assistantMsg)

		// Create measurement for this turn
		m := yarn.NewMeasurementForTurn(session.ID, conv.ID, i+1)
		m.SetSender("user", "User", "user", nil)
		m.SetReceiver("junior", "junior", "assistant", assistantHidden)
		m.DEff = 100 + i*10
		m.Beta = 1.5 + float64(i)*0.1
		m.Alignment = 0.8 - float64(i)*0.05
		m.BetaStatus = yarn.ComputeBetaStatus(m.Beta)
		session.AddMeasurement(m)
	}

	// Verify measurements
	if len(session.Measurements) != numTurns {
		t.Errorf("Session has %d measurements, want %d", len(session.Measurements), numTurns)
	}

	// Verify conversation messages
	if conv.Length() != numTurns*2 {
		t.Errorf("Conversation has %d messages, want %d", conv.Length(), numTurns*2)
	}

	// Validate session
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() failed: %v", err)
	}

	// Check session stats
	stats := session.Stats()
	if stats.MeasurementCount != numTurns {
		t.Errorf("Stats.MeasurementCount = %d, want %d", stats.MeasurementCount, numTurns)
	}

	// Verify average beta is computed correctly
	expectedAvgBeta := 0.0
	for i := 0; i < numTurns; i++ {
		expectedAvgBeta += 1.5 + float64(i)*0.1
	}
	expectedAvgBeta /= float64(numTurns)

	tolerance := 0.0001
	if math.Abs(stats.AvgBeta-expectedAvgBeta) > tolerance {
		t.Errorf("Stats.AvgBeta = %v, want %v (tolerance %v)", stats.AvgBeta, expectedAvgBeta, tolerance)
	}
}

// TestMeasurementBilateralExchange tests bilateral measurement with both hidden states.
func TestMeasurementBilateralExchange(t *testing.T) {
	session := yarn.NewSession("bilateral-test", "Bilateral exchange test")
	conv := yarn.NewConversation("bilateral-conv")
	session.AddConversation(conv)

	// Create hidden states for both participants
	senderHidden := &yarn.HiddenState{
		Vector: make([]float32, 4096),
		Shape:  []int{1, 4096},
		Layer:  -1,
		DType:  "float32",
	}
	receiverHidden := &yarn.HiddenState{
		Vector: make([]float32, 4096),
		Shape:  []int{1, 4096},
		Layer:  -1,
		DType:  "float32",
	}

	// Populate with different patterns
	for i := range senderHidden.Vector {
		senderHidden.Vector[i] = float32(i) * 0.001
		receiverHidden.Vector[i] = float32(4095-i) * 0.001
	}

	// Create bilateral measurement
	m := yarn.NewMeasurementForTurn(session.ID, conv.ID, 1)
	m.SetSender("agent-1", "Agent One", "assistant", senderHidden)
	m.SetReceiver("agent-2", "Agent Two", "assistant", receiverHidden)
	m.DEff = 200
	m.Beta = 1.65
	m.Alignment = 0.92
	m.CPair = 0.88
	m.BetaStatus = yarn.BetaOptimal
	m.IsUnilateral = false

	// Verify bilateral detection
	if !m.IsBilateral() {
		t.Error("Measurement with both hidden states should be bilateral")
	}
	if m.IsUnilateral {
		t.Error("Measurement should not be marked as unilateral")
	}

	session.AddMeasurement(m)

	// Validate session
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() failed: %v", err)
	}

	// Verify stats
	stats := session.Stats()
	if stats.BilateralCount != 1 {
		t.Errorf("Stats.BilateralCount = %d, want 1", stats.BilateralCount)
	}
}

// TestMeasurementSessionExport tests exporting session with measurements.
func TestMeasurementSessionExport(t *testing.T) {
	// Create temp directory for export
	tmpDir, err := os.MkdirTemp("", "measurement-export-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create session with export config
	session := yarn.NewSession("export-test", "Export test session")
	session.Config.ExportPath = tmpDir
	session.Config.AutoExport = true

	// Add conversation with messages
	conv := yarn.NewConversation("export-conv")
	session.AddConversation(conv)

	userMsg := yarn.NewAgentMessage(yarn.RoleUser, "Export test message", "user", "user")
	conv.Add(userMsg)

	assistantMsg := yarn.NewAgentMessage(yarn.RoleAssistant, "Export test response", "junior", "junior")
	conv.Add(assistantMsg)

	// Add measurements
	for i := 0; i < 3; i++ {
		m := yarn.NewMeasurementForTurn(session.ID, conv.ID, i+1)
		m.SenderID = "user"
		m.ReceiverID = "junior"
		m.DEff = 100 + i*10
		m.Beta = 1.5 + float64(i)*0.2
		m.Alignment = 0.8 - float64(i)*0.1
		m.BetaStatus = yarn.ComputeBetaStatus(m.Beta)
		session.AddMeasurement(m)
	}

	// End session
	session.End()

	// Export session
	if err := session.Export(); err != nil {
		t.Fatalf("Session.Export() failed: %v", err)
	}

	// Verify export files exist
	exportDir := filepath.Join(tmpDir, session.ID)
	sessionFile := filepath.Join(exportDir, "session.json")
	measurementsFile := filepath.Join(exportDir, "measurements.jsonl")

	if _, err := os.Stat(sessionFile); os.IsNotExist(err) {
		t.Error("session.json file not created")
	}
	if _, err := os.Stat(measurementsFile); os.IsNotExist(err) {
		t.Error("measurements.jsonl file not created")
	}

	// Verify session.json content
	sessionData, err := os.ReadFile(sessionFile)
	if err != nil {
		t.Fatalf("Failed to read session.json: %v", err)
	}

	var exportedSession yarn.Session
	if err := json.Unmarshal(sessionData, &exportedSession); err != nil {
		t.Fatalf("Failed to parse session.json: %v", err)
	}

	if exportedSession.ID != session.ID {
		t.Errorf("Exported session ID = %q, want %q", exportedSession.ID, session.ID)
	}
	if len(exportedSession.Measurements) != 3 {
		t.Errorf("Exported session has %d measurements, want 3", len(exportedSession.Measurements))
	}
}

// TestMeasurementHiddenStateValidation tests hidden state validation in measurements.
func TestMeasurementHiddenStateValidation(t *testing.T) {
	tests := []struct {
		name        string
		hiddenState *yarn.HiddenState
		wantErr     bool
		wantField   string
	}{
		{
			name:        "nil hidden state is valid",
			hiddenState: nil,
			wantErr:     false,
		},
		{
			name: "valid hidden state",
			hiddenState: &yarn.HiddenState{
				Vector: make([]float32, 4096),
				Shape:  []int{1, 4096},
				Layer:  0,
				DType:  "float32",
			},
			wantErr: false,
		},
		{
			name: "valid hidden state with float16",
			hiddenState: &yarn.HiddenState{
				Vector: make([]float32, 768),
				Shape:  []int{1, 768},
				Layer:  12,
				DType:  "float16",
			},
			wantErr: false,
		},
		{
			name: "empty vector is invalid",
			hiddenState: &yarn.HiddenState{
				Vector: []float32{},
				Layer:  0,
			},
			wantErr:   true,
			wantField: "sender_hidden.vector",
		},
		{
			name: "negative layer is invalid",
			hiddenState: &yarn.HiddenState{
				Vector: make([]float32, 100),
				Layer:  -2, // -1 is common for last layer, but implementation might reject
			},
			wantErr:   true,
			wantField: "sender_hidden.layer",
		},
		{
			name: "invalid dtype",
			hiddenState: &yarn.HiddenState{
				Vector: make([]float32, 100),
				Layer:  0,
				DType:  "float64",
			},
			wantErr:   true,
			wantField: "sender_hidden.dtype",
		},
		{
			name: "inconsistent shape",
			hiddenState: &yarn.HiddenState{
				Vector: make([]float32, 100),
				Shape:  []int{2, 100}, // 200 != 100
				Layer:  0,
			},
			wantErr:   true,
			wantField: "sender_hidden.shape",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := yarn.NewMeasurement()
			m.SenderID = "agent-1"
			m.SenderHidden = tt.hiddenState

			err := m.Validate()

			if tt.wantErr {
				if err == nil {
					t.Errorf("Measurement.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Measurement.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
			} else {
				if err != nil {
					t.Errorf("Measurement.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestMeasurementMetricsRanges tests boundary conditions for measurement metrics.
func TestMeasurementMetricsRanges(t *testing.T) {
	tests := []struct {
		name      string
		deff      int
		beta      float64
		alignment float64
		wantValid bool
		wantField string
	}{
		{"zero metrics", 0, 0, 0, true, ""},
		{"positive DEff", 100, 0, 0, true, ""},
		{"large DEff", 10000, 0, 0, true, ""},
		{"optimal beta", 0, 1.75, 0, true, ""},
		{"high beta", 0, 5.0, 0, true, ""},
		{"negative alignment", 0, 0, -1.0, true, ""},
		{"positive alignment", 0, 0, 1.0, true, ""},
		{"zero alignment", 0, 0, 0.0, true, ""},
		{"negative DEff", -1, 0, 0, false, "d_eff"},
		{"negative beta", 0, -0.1, 0, false, "beta"},
		{"alignment below -1", 0, 0, -1.1, false, "alignment"},
		{"alignment above 1", 0, 0, 1.1, false, "alignment"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := yarn.NewMeasurement()
			m.SenderID = "agent-1"
			m.DEff = tt.deff
			m.Beta = tt.beta
			m.Alignment = tt.alignment

			err := m.Validate()

			if tt.wantValid {
				if err != nil {
					t.Errorf("Measurement.Validate() unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("Measurement.Validate() expected error, got nil")
					return
				}
				if err.Field != tt.wantField {
					t.Errorf("Measurement.Validate() error field = %q, want %q", err.Field, tt.wantField)
				}
			}
		})
	}
}

// TestMeasurementSetterMethods tests SetSender and SetReceiver methods.
func TestMeasurementSetterMethods(t *testing.T) {
	t.Run("SetSender sets all fields", func(t *testing.T) {
		m := yarn.NewMeasurement()
		hidden := &yarn.HiddenState{
			Vector: make([]float32, 100),
			Layer:  0,
		}

		m.SetSender("sender-id", "Sender Name", "assistant", hidden)

		if m.SenderID != "sender-id" {
			t.Errorf("SenderID = %q, want %q", m.SenderID, "sender-id")
		}
		if m.SenderName != "Sender Name" {
			t.Errorf("SenderName = %q, want %q", m.SenderName, "Sender Name")
		}
		if m.SenderRole != "assistant" {
			t.Errorf("SenderRole = %q, want %q", m.SenderRole, "assistant")
		}
		if m.SenderHidden != hidden {
			t.Error("SenderHidden not set correctly")
		}
	})

	t.Run("SetReceiver sets all fields", func(t *testing.T) {
		m := yarn.NewMeasurement()
		hidden := &yarn.HiddenState{
			Vector: make([]float32, 100),
			Layer:  0,
		}

		m.SetReceiver("receiver-id", "Receiver Name", "user", hidden)

		if m.ReceiverID != "receiver-id" {
			t.Errorf("ReceiverID = %q, want %q", m.ReceiverID, "receiver-id")
		}
		if m.ReceiverName != "Receiver Name" {
			t.Errorf("ReceiverName = %q, want %q", m.ReceiverName, "Receiver Name")
		}
		if m.ReceiverRole != "user" {
			t.Errorf("ReceiverRole = %q, want %q", m.ReceiverRole, "user")
		}
		if m.ReceiverHidden != hidden {
			t.Error("ReceiverHidden not set correctly")
		}
	})

	t.Run("SetSender with nil hidden", func(t *testing.T) {
		m := yarn.NewMeasurement()
		m.SetSender("sender-id", "Sender", "assistant", nil)

		if m.SenderID != "sender-id" {
			t.Errorf("SenderID = %q, want %q", m.SenderID, "sender-id")
		}
		if m.SenderHidden != nil {
			t.Error("SenderHidden should be nil")
		}
	})
}

// TestMeasurementUniqueness tests that measurements have unique IDs.
func TestMeasurementUniqueness(t *testing.T) {
	const numMeasurements = 100
	ids := make(map[string]bool)

	for i := 0; i < numMeasurements; i++ {
		m := yarn.NewMeasurement()
		if ids[m.ID] {
			t.Errorf("Duplicate measurement ID: %s", m.ID)
		}
		ids[m.ID] = true
	}

	if len(ids) != numMeasurements {
		t.Errorf("Expected %d unique IDs, got %d", numMeasurements, len(ids))
	}
}

// TestMeasurementTimestamp tests that measurements have valid timestamps.
func TestMeasurementTimestamp(t *testing.T) {
	before := time.Now()
	m := yarn.NewMeasurement()
	after := time.Now()

	if m.Timestamp.Before(before) || m.Timestamp.After(after) {
		t.Errorf("Measurement timestamp %v not in expected range [%v, %v]", m.Timestamp, before, after)
	}

	if m.Timestamp.IsZero() {
		t.Error("Measurement timestamp should not be zero")
	}
}
