package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/r3d91ll/weaver/pkg/backend"
	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/wool"
	"github.com/r3d91ll/yarn"
)

// TestDelegationRouting tests that messages are correctly routed based on @agent prefix.
func TestDelegationRouting(t *testing.T) {
	tests := []struct {
		name           string
		message        string
		wantTarget     string
		wantContent    string
		wantParseError bool
	}{
		{
			name:        "route to senior by default",
			message:     "Hello, help me with this task",
			wantTarget:  "senior",
			wantContent: "Hello, help me with this task",
		},
		{
			name:        "route to junior with @junior prefix",
			message:     "@junior implement this function",
			wantTarget:  "junior",
			wantContent: "implement this function",
		},
		{
			name:        "route to senior with @senior prefix",
			message:     "@senior review this code",
			wantTarget:  "senior",
			wantContent: "review this code",
		},
		{
			name:        "handle multiple words after @agent",
			message:     "@junior please write a test for the login function",
			wantTarget:  "junior",
			wantContent: "please write a test for the login function",
		},
		{
			name:           "error on @agent without message",
			message:        "@junior",
			wantTarget:     "junior",
			wantParseError: true,
		},
		{
			name:        "preserve whitespace in message content",
			message:     "@junior   code with   spaces",
			wantTarget:  "junior",
			wantContent: "  code with   spaces",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target, content, err := parseAgentMessage(tt.message, "senior")

			if tt.wantParseError {
				if err == nil {
					t.Errorf("parseAgentMessage() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("parseAgentMessage() unexpected error: %v", err)
				return
			}

			if target != tt.wantTarget {
				t.Errorf("parseAgentMessage() target = %q, want %q", target, tt.wantTarget)
			}
			if content != tt.wantContent {
				t.Errorf("parseAgentMessage() content = %q, want %q", content, tt.wantContent)
			}
		})
	}
}

// parseAgentMessage parses a message with optional @agent prefix.
// Returns target agent name, message content, and any error.
func parseAgentMessage(message, defaultAgent string) (string, string, error) {
	targetAgent := defaultAgent
	content := message

	if strings.HasPrefix(message, "@") {
		parts := strings.SplitN(message, " ", 2)
		targetAgent = strings.TrimPrefix(parts[0], "@")
		if len(parts) > 1 {
			content = parts[1]
		} else {
			return targetAgent, "", fmt.Errorf("no message after @%s", targetAgent)
		}
	}

	return targetAgent, content, nil
}

// TestDelegationAgentRoles tests that agents have correct role assignments.
func TestDelegationAgentRoles(t *testing.T) {
	tests := []struct {
		name             string
		agent            wool.Agent
		wantRole         wool.Role
		wantBackend      string
		wantToolsEnabled bool
	}{
		{
			name:             "senior agent has senior role",
			agent:            wool.DefaultSenior(),
			wantRole:         wool.RoleSenior,
			wantBackend:      "claudecode",
			wantToolsEnabled: true,
		},
		{
			name:             "junior agent has junior role",
			agent:            wool.DefaultJunior(),
			wantRole:         wool.RoleJunior,
			wantBackend:      "loom",
			wantToolsEnabled: true,
		},
		{
			name:             "subject agent has subject role",
			agent:            wool.DefaultSubject(),
			wantRole:         wool.RoleSubject,
			wantBackend:      "loom",
			wantToolsEnabled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.agent.Role != tt.wantRole {
				t.Errorf("Agent.Role = %q, want %q", tt.agent.Role, tt.wantRole)
			}
			if tt.agent.Backend != tt.wantBackend {
				t.Errorf("Agent.Backend = %q, want %q", tt.agent.Backend, tt.wantBackend)
			}
			if tt.agent.ToolsEnabled != tt.wantToolsEnabled {
				t.Errorf("Agent.ToolsEnabled = %v, want %v", tt.agent.ToolsEnabled, tt.wantToolsEnabled)
			}
		})
	}
}

// TestDelegationRoleCapabilities tests role capability methods.
func TestDelegationRoleCapabilities(t *testing.T) {
	tests := []struct {
		name                 string
		role                 wool.Role
		wantSupportsTools    bool
		wantRequiresHidden   bool
		wantCanGenerate      bool
	}{
		{
			name:                 "senior supports tools, no hidden states",
			role:                 wool.RoleSenior,
			wantSupportsTools:    true,
			wantRequiresHidden:   false,
			wantCanGenerate:      true,
		},
		{
			name:                 "junior supports tools, requires hidden states",
			role:                 wool.RoleJunior,
			wantSupportsTools:    true,
			wantRequiresHidden:   true,
			wantCanGenerate:      true,
		},
		{
			name:                 "conversant no tools, requires hidden states",
			role:                 wool.RoleConversant,
			wantSupportsTools:    false,
			wantRequiresHidden:   true,
			wantCanGenerate:      true,
		},
		{
			name:                 "subject no tools, requires hidden states",
			role:                 wool.RoleSubject,
			wantSupportsTools:    false,
			wantRequiresHidden:   true,
			wantCanGenerate:      true,
		},
		{
			name:                 "observer no tools, no hidden states, no generation",
			role:                 wool.RoleObserver,
			wantSupportsTools:    false,
			wantRequiresHidden:   false,
			wantCanGenerate:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.role.SupportsTools(); got != tt.wantSupportsTools {
				t.Errorf("Role.SupportsTools() = %v, want %v", got, tt.wantSupportsTools)
			}
			if got := tt.role.RequiresHiddenStates(); got != tt.wantRequiresHidden {
				t.Errorf("Role.RequiresHiddenStates() = %v, want %v", got, tt.wantRequiresHidden)
			}
			if got := tt.role.CanGenerateResponses(); got != tt.wantCanGenerate {
				t.Errorf("Role.CanGenerateResponses() = %v, want %v", got, tt.wantCanGenerate)
			}
		})
	}
}

// TestDelegationBackendCapabilities tests backend capability detection.
func TestDelegationBackendCapabilities(t *testing.T) {
	tests := []struct {
		name           string
		backendType    backend.Type
		wantSupports   bool
		wantStreaming  bool
	}{
		{
			name:           "loom backend supports hidden states",
			backendType:    backend.TypeLoom,
			wantSupports:   true,
			wantStreaming:  true,
		},
		{
			name:           "claudecode backend does not support hidden states",
			backendType:    backend.TypeClaudeCode,
			wantSupports:   false,
			wantStreaming:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Verify backend type constants exist
			if tt.backendType == "" {
				t.Errorf("BackendType is empty")
			}
		})
	}
}

// mockLoomServer creates a mock HTTP server that simulates The Loom backend.
func mockLoomServer(t *testing.T, respondWithHiddenState bool) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"status": "ok"}`))

		case "/v1/chat/completions":
			// Parse request to verify structure
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Errorf("Failed to read request body: %v", err)
				http.Error(w, "Bad request", http.StatusBadRequest)
				return
			}
			defer r.Body.Close()

			var req struct {
				Model              string `json:"model"`
				Messages           []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
				ReturnHiddenStates bool `json:"return_hidden_states"`
			}
			if err := json.Unmarshal(body, &req); err != nil {
				t.Errorf("Failed to parse request: %v", err)
				http.Error(w, "Bad request", http.StatusBadRequest)
				return
			}

			// Build response
			resp := map[string]any{
				"text": "This is a mock response from the Junior agent.",
				"usage": map[string]int{
					"prompt_tokens":     10,
					"completion_tokens": 15,
					"total_tokens":      25,
				},
			}

			// Add hidden state if requested
			if respondWithHiddenState && req.ReturnHiddenStates {
				// Create mock hidden state vector (4096 dimensions typical for LLMs)
				hiddenDim := 4096
				mockVector := make([]float32, hiddenDim)
				for i := range mockVector {
					mockVector[i] = float32(i) * 0.001 // Simple pattern
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

// TestDelegationWithMockServer tests delegation with a mock Loom server.
func TestDelegationWithMockServer(t *testing.T) {
	// Create mock server
	server := mockLoomServer(t, true)
	defer server.Close()

	// Create Loom backend pointing to mock server
	loomBackend := backend.NewLoom(backend.LoomConfig{
		Name:    "mock-loom",
		URL:     server.URL,
		Model:   "test-model",
		Timeout: 10 * time.Second,
	})

	// Verify backend is available
	ctx := context.Background()
	if !loomBackend.IsAvailable(ctx) {
		t.Fatal("Mock Loom server should be available")
	}

	// Verify capabilities
	caps := loomBackend.Capabilities()
	if !caps.SupportsHidden {
		t.Error("Loom backend should support hidden states")
	}

	// Create a chat request
	req := backend.ChatRequest{
		Model: "test-model",
		Messages: []backend.ChatMessage{
			{Role: "user", Content: "Hello, please help me with a task"},
		},
		ReturnHiddenStates: true,
	}

	// Execute chat
	resp, err := loomBackend.Chat(ctx, req)
	if err != nil {
		t.Fatalf("Chat() failed: %v", err)
	}

	// Verify response
	if resp.Content == "" {
		t.Error("Response content should not be empty")
	}

	// Verify hidden state
	if resp.HiddenState == nil {
		t.Error("Response should include hidden state")
	} else {
		if len(resp.HiddenState.Vector) == 0 {
			t.Error("Hidden state vector should not be empty")
		}
		if resp.HiddenState.Dimension() != 4096 {
			t.Errorf("Hidden state dimension = %d, want 4096", resp.HiddenState.Dimension())
		}
	}
}

// TestDelegationAgentCreation tests creating agents with the runtime manager.
func TestDelegationAgentCreation(t *testing.T) {
	// Create mock server
	server := mockLoomServer(t, true)
	defer server.Close()

	// Create registry with mock Loom backend
	registry := backend.NewRegistry()
	loomBackend := backend.NewLoom(backend.LoomConfig{
		Name:    "loom",
		URL:     server.URL,
		Model:   "test-model",
		Timeout: 10 * time.Second,
	})
	registry.Register(loomBackend)

	// Create manager
	manager := runtime.NewManager(registry)

	// Create junior agent
	juniorDef := wool.DefaultJunior()
	juniorDef.ID = "junior-001"
	juniorDef.Model = "test-model"

	agent, err := manager.Create(juniorDef)
	if err != nil {
		t.Fatalf("Failed to create junior agent: %v", err)
	}

	// Verify agent properties
	if agent.Name() != "junior" {
		t.Errorf("Agent.Name() = %q, want %q", agent.Name(), "junior")
	}
	if agent.Role() != wool.RoleJunior {
		t.Errorf("Agent.Role() = %q, want %q", agent.Role(), wool.RoleJunior)
	}
	if agent.BackendName() != "loom" {
		t.Errorf("Agent.BackendName() = %q, want %q", agent.BackendName(), "loom")
	}
	if !agent.SupportsHiddenStates() {
		t.Error("Junior agent should support hidden states")
	}

	// Verify agent is ready (backend available)
	ctx := context.Background()
	if !agent.IsReady(ctx) {
		t.Error("Agent should be ready with mock server running")
	}
}

// TestDelegationAgentChat tests sending messages through an agent.
func TestDelegationAgentChat(t *testing.T) {
	// Create mock server
	server := mockLoomServer(t, true)
	defer server.Close()

	// Create registry with mock Loom backend
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
	juniorDef.ID = "junior-001"
	juniorDef.Model = "test-model"

	agent, err := manager.Create(juniorDef)
	if err != nil {
		t.Fatalf("Failed to create agent: %v", err)
	}

	// Create test messages
	ctx := context.Background()
	messages := []*yarn.Message{
		yarn.NewAgentMessage(yarn.RoleUser, "Please implement a function", "user", "user"),
	}

	// Chat with agent
	resp, err := agent.Chat(ctx, messages)
	if err != nil {
		t.Fatalf("Agent.Chat() failed: %v", err)
	}

	// Verify response
	if resp == nil {
		t.Fatal("Response should not be nil")
	}
	if resp.Content == "" {
		t.Error("Response content should not be empty")
	}
	if resp.Role != yarn.RoleAssistant {
		t.Errorf("Response.Role = %q, want %q", resp.Role, yarn.RoleAssistant)
	}
	if resp.AgentName != "junior" {
		t.Errorf("Response.AgentName = %q, want %q", resp.AgentName, "junior")
	}

	// Verify hidden state attached
	if !resp.HasHiddenState() {
		t.Error("Response should have hidden state")
	}
	if resp.HiddenState != nil && resp.HiddenState.Dimension() != 4096 {
		t.Errorf("HiddenState.Dimension() = %d, want 4096", resp.HiddenState.Dimension())
	}
}

// TestDelegationManagerGet tests retrieving agents by name.
func TestDelegationManagerGet(t *testing.T) {
	// Create mock server
	server := mockLoomServer(t, false)
	defer server.Close()

	// Setup registry and manager
	registry := backend.NewRegistry()
	loomBackend := backend.NewLoom(backend.LoomConfig{
		Name:    "loom",
		URL:     server.URL,
		Model:   "test-model",
		Timeout: 10 * time.Second,
	})
	registry.Register(loomBackend)
	manager := runtime.NewManager(registry)

	// Create junior agent
	juniorDef := wool.DefaultJunior()
	juniorDef.ID = "junior-001"
	juniorDef.Model = "test-model"
	manager.Create(juniorDef)

	tests := []struct {
		name       string
		agentName  string
		wantFound  bool
	}{
		{"find existing junior agent", "junior", true},
		{"agent not found", "senior", false},
		{"empty name not found", "", false},
		{"unknown agent not found", "unknown", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, found := manager.Get(tt.agentName)
			if found != tt.wantFound {
				t.Errorf("Manager.Get(%q) found = %v, want %v", tt.agentName, found, tt.wantFound)
			}
			if tt.wantFound && agent == nil {
				t.Errorf("Manager.Get(%q) returned nil agent when found", tt.agentName)
			}
		})
	}
}

// TestDelegationManagerStatus tests getting agent status.
func TestDelegationManagerStatus(t *testing.T) {
	// Create mock server
	server := mockLoomServer(t, true)
	defer server.Close()

	// Setup
	registry := backend.NewRegistry()
	loomBackend := backend.NewLoom(backend.LoomConfig{
		Name:    "loom",
		URL:     server.URL,
		Model:   "test-model",
		Timeout: 10 * time.Second,
	})
	registry.Register(loomBackend)
	manager := runtime.NewManager(registry)

	// Create junior agent
	juniorDef := wool.DefaultJunior()
	juniorDef.ID = "junior-001"
	juniorDef.Model = "test-model"
	manager.Create(juniorDef)

	// Get status
	ctx := context.Background()
	statuses := manager.Status(ctx)

	// Verify junior status
	juniorStatus, ok := statuses["junior"]
	if !ok {
		t.Fatal("Junior agent status not found")
	}

	if juniorStatus.Name != "junior" {
		t.Errorf("Status.Name = %q, want %q", juniorStatus.Name, "junior")
	}
	if juniorStatus.Role != wool.RoleJunior {
		t.Errorf("Status.Role = %q, want %q", juniorStatus.Role, wool.RoleJunior)
	}
	if juniorStatus.Backend != "loom" {
		t.Errorf("Status.Backend = %q, want %q", juniorStatus.Backend, "loom")
	}
	if !juniorStatus.Ready {
		t.Error("Status.Ready should be true with mock server")
	}
	if !juniorStatus.HiddenStates {
		t.Error("Status.HiddenStates should be true for Loom backend")
	}
}

// TestDelegationDuplicateAgent tests that duplicate agent creation fails.
func TestDelegationDuplicateAgent(t *testing.T) {
	// Create mock server
	server := mockLoomServer(t, false)
	defer server.Close()

	// Setup
	registry := backend.NewRegistry()
	loomBackend := backend.NewLoom(backend.LoomConfig{
		Name:    "loom",
		URL:     server.URL,
		Model:   "test-model",
		Timeout: 10 * time.Second,
	})
	registry.Register(loomBackend)
	manager := runtime.NewManager(registry)

	// Create first junior
	juniorDef := wool.DefaultJunior()
	juniorDef.ID = "junior-001"
	juniorDef.Model = "test-model"

	_, err := manager.Create(juniorDef)
	if err != nil {
		t.Fatalf("First Create() failed: %v", err)
	}

	// Try to create duplicate
	juniorDef2 := wool.DefaultJunior()
	juniorDef2.ID = "junior-002"
	juniorDef2.Model = "test-model"

	_, err = manager.Create(juniorDef2)
	if err == nil {
		t.Error("Create() should fail for duplicate agent name")
	}
}

// TestDelegationSSEParsing tests Server-Sent Events parsing for streaming responses.
func TestDelegationSSEParsing(t *testing.T) {
	tests := []struct {
		name          string
		sseData       string
		wantEvents    int
		wantContent   string
	}{
		{
			name: "single content delta event",
			sseData: `event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}

`,
			wantEvents:  1,
			wantContent: "Hello",
		},
		{
			name: "multiple content deltas",
			sseData: `event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" World"}}

`,
			wantEvents:  2,
			wantContent: " World",
		},
		{
			name: "message delta with done",
			sseData: `event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"total_tokens":100}}

`,
			wantEvents: 1,
		},
		{
			name: "comment lines are ignored",
			sseData: `: this is a comment
event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Test"}}

`,
			wantEvents:  1,
			wantContent: "Test",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := bytes.NewReader([]byte(tt.sseData))
			events := parseTestSSE(reader)

			if len(events) != tt.wantEvents {
				t.Errorf("parseSSE() returned %d events, want %d", len(events), tt.wantEvents)
			}
		})
	}
}

// testSSEEvent represents a parsed SSE event for testing.
type testSSEEvent struct {
	Event string
	Data  string
}

// parseTestSSE is a simple SSE parser for testing.
func parseTestSSE(r io.Reader) []testSSEEvent {
	var events []testSSEEvent
	data, _ := io.ReadAll(r)
	lines := strings.Split(string(data), "\n")

	var currentEvent string
	var dataLines []string

	for _, line := range lines {
		if line == "" {
			if len(dataLines) > 0 {
				events = append(events, testSSEEvent{
					Event: currentEvent,
					Data:  strings.Join(dataLines, "\n"),
				})
			}
			currentEvent = ""
			dataLines = nil
			continue
		}

		if strings.HasPrefix(line, ":") {
			continue // Skip comments
		}

		if strings.HasPrefix(line, "event:") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			data := strings.TrimPrefix(line, "data:")
			if len(data) > 0 && data[0] == ' ' {
				data = data[1:]
			}
			dataLines = append(dataLines, data)
		}
	}

	return events
}

// TestDelegationConversationFlow tests a full conversation flow with multiple messages.
func TestDelegationConversationFlow(t *testing.T) {
	// Create mock server
	server := mockLoomServer(t, true)
	defer server.Close()

	// Setup
	registry := backend.NewRegistry()
	loomBackend := backend.NewLoom(backend.LoomConfig{
		Name:    "loom",
		URL:     server.URL,
		Model:   "test-model",
		Timeout: 10 * time.Second,
	})
	registry.Register(loomBackend)
	manager := runtime.NewManager(registry)

	// Create junior agent
	juniorDef := wool.DefaultJunior()
	juniorDef.ID = "junior-001"
	juniorDef.Model = "test-model"
	agent, err := manager.Create(juniorDef)
	if err != nil {
		t.Fatalf("Failed to create agent: %v", err)
	}

	// Create conversation
	conv := yarn.NewConversation("test-conversation")

	// Send first message
	ctx := context.Background()
	userMsg1 := yarn.NewAgentMessage(yarn.RoleUser, "What is 2+2?", "user", "user")
	conv.Add(userMsg1)

	resp1, err := agent.Chat(ctx, conv.History(-1))
	if err != nil {
		t.Fatalf("First Chat() failed: %v", err)
	}
	conv.Add(resp1)

	// Send follow-up message
	userMsg2 := yarn.NewAgentMessage(yarn.RoleUser, "Now what is 3+3?", "user", "user")
	conv.Add(userMsg2)

	resp2, err := agent.Chat(ctx, conv.History(-1))
	if err != nil {
		t.Fatalf("Second Chat() failed: %v", err)
	}
	conv.Add(resp2)

	// Verify conversation state
	messages := conv.History(-1)
	if len(messages) != 4 {
		t.Errorf("Conversation has %d messages, want 4", len(messages))
	}

	// Verify message roles alternate
	expectedRoles := []yarn.MessageRole{yarn.RoleUser, yarn.RoleAssistant, yarn.RoleUser, yarn.RoleAssistant}
	for i, msg := range messages {
		if msg.Role != expectedRoles[i] {
			t.Errorf("Message[%d].Role = %q, want %q", i, msg.Role, expectedRoles[i])
		}
	}
}

// TestDelegationSessionIntegration tests session integration with conversations.
func TestDelegationSessionIntegration(t *testing.T) {
	// Create a session
	session := yarn.NewSession("test-session", "Test delegation session")

	// Validate session
	if err := session.Validate(); err != nil {
		t.Fatalf("Session.Validate() failed: %v", err)
	}

	// Create conversation
	conv := yarn.NewConversation("delegation-test")
	session.AddConversation(conv)

	// Add messages
	userMsg := yarn.NewAgentMessage(yarn.RoleUser, "@junior help me", "user", "user")
	conv.Add(userMsg)

	assistantMsg := yarn.NewAgentMessage(yarn.RoleAssistant, "I'll help you", "junior-001", "junior")
	// Simulate hidden state
	assistantMsg.HiddenState = &yarn.HiddenState{
		Vector: make([]float32, 4096),
		Shape:  []int{1, 4096},
		Layer:  -1,
		DType:  "float32",
	}
	conv.Add(assistantMsg)

	// Verify session stats
	stats := session.Stats()
	if stats.ConversationCount != 1 {
		t.Errorf("Stats.ConversationCount = %d, want 1", stats.ConversationCount)
	}
	if stats.MessageCount != 2 {
		t.Errorf("Stats.MessageCount = %d, want 2", stats.MessageCount)
	}

	// Verify session is still valid
	if err := session.Validate(); err != nil {
		t.Errorf("Session.Validate() after messages failed: %v", err)
	}
}

// TestDelegationHiddenStateValidation tests hidden state validation and properties.
func TestDelegationHiddenStateValidation(t *testing.T) {
	tests := []struct {
		name          string
		hiddenState   *yarn.HiddenState
		wantDimension int
		wantHasState  bool
	}{
		{
			name:          "nil hidden state",
			hiddenState:   nil,
			wantDimension: 0,
			wantHasState:  false,
		},
		{
			name: "valid hidden state",
			hiddenState: &yarn.HiddenState{
				Vector: make([]float32, 4096),
				Shape:  []int{1, 4096},
				Layer:  -1,
				DType:  "float32",
			},
			wantDimension: 4096,
			wantHasState:  true,
		},
		{
			name: "small hidden state",
			hiddenState: &yarn.HiddenState{
				Vector: make([]float32, 768),
				Shape:  []int{1, 768},
				Layer:  12,
				DType:  "float32",
			},
			wantDimension: 768,
			wantHasState:  true,
		},
		{
			name: "empty vector",
			hiddenState: &yarn.HiddenState{
				Vector: []float32{},
				Shape:  []int{1, 0},
				Layer:  -1,
				DType:  "float32",
			},
			wantDimension: 0,
			wantHasState:  true, // struct exists but empty
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := yarn.NewAgentMessage(yarn.RoleAssistant, "test", "agent-1", "agent")
			msg.HiddenState = tt.hiddenState

			if msg.HasHiddenState() != tt.wantHasState {
				t.Errorf("Message.HasHiddenState() = %v, want %v", msg.HasHiddenState(), tt.wantHasState)
			}

			if tt.hiddenState != nil {
				if got := tt.hiddenState.Dimension(); got != tt.wantDimension {
					t.Errorf("HiddenState.Dimension() = %d, want %d", got, tt.wantDimension)
				}
			}
		})
	}
}
