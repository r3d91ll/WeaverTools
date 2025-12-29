package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/r3d91ll/weaver/pkg/backend"
	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/wool"
	"github.com/r3d91ll/yarn"
)

// -----------------------------------------------------------------------------
// Test Backend for Agent Tests
// -----------------------------------------------------------------------------

// testBackend implements backend.Backend for testing.
type testBackend struct {
	name         string
	available    bool
	response     string
	err          error
	hiddenStates bool
}

func (b *testBackend) Name() string                         { return b.name }
func (b *testBackend) Type() backend.Type                   { return backend.TypeLoom }
func (b *testBackend) IsAvailable(ctx context.Context) bool { return b.available }
func (b *testBackend) Capabilities() backend.Capabilities {
	return backend.Capabilities{
		ContextLimit:      32768,
		SupportsTools:     true,
		SupportsStreaming: false,
		SupportsHidden:    b.hiddenStates,
		MaxTokens:         4096,
	}
}
func (b *testBackend) Chat(ctx context.Context, req backend.ChatRequest) (*backend.ChatResponse, error) {
	if b.err != nil {
		return nil, b.err
	}
	return &backend.ChatResponse{
		Content:   b.response,
		Model:     "test-model",
		LatencyMS: 42.5,
		Usage: backend.TokenUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
		FinishReason: "stop",
	}, nil
}
func (b *testBackend) ChatStream(ctx context.Context, req backend.ChatRequest) (<-chan backend.StreamChunk, <-chan error) {
	return nil, nil
}

// -----------------------------------------------------------------------------
// NewAgentsHandler Tests
// -----------------------------------------------------------------------------

func TestNewAgentsHandler(t *testing.T) {
	t.Run("with nil manager", func(t *testing.T) {
		h := NewAgentsHandler(nil)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})

	t.Run("with mock manager", func(t *testing.T) {
		mockMgr := NewMockAgentManager()
		h := NewAgentsHandler(mockMgr)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})

	t.Run("with runtime manager via adapter", func(t *testing.T) {
		registry := backend.NewRegistry()
		manager := runtime.NewManager(registry)
		h := NewAgentsHandlerWithRuntime(manager)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})
}

// -----------------------------------------------------------------------------
// ListAgents Tests
// -----------------------------------------------------------------------------

func TestAgentsHandler_ListAgents(t *testing.T) {
	t.Run("returns empty list when no agents", func(t *testing.T) {
		mockMgr := NewMockAgentManager()
		h := NewAgentsHandler(mockMgr)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/agents", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		agents := data["agents"].([]interface{})
		if len(agents) != 0 {
			t.Errorf("Expected empty agents list, got %d agents", len(agents))
		}
	})

	t.Run("returns agents when registered", func(t *testing.T) {
		mockMgr := NewMockAgentManager()
		mockMgr.AddMockAgent(&MockAgent{
			name:         "test-agent",
			role:         "junior",
			backend:      "test-backend",
			model:        "test-model",
			ready:        true,
			hiddenStates: false,
		})

		h := NewAgentsHandler(mockMgr)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/agents", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		agents := data["agents"].([]interface{})
		if len(agents) != 1 {
			t.Errorf("Expected 1 agent, got %d", len(agents))
		}

		agent := agents[0].(map[string]interface{})
		if agent["name"] != "test-agent" {
			t.Errorf("Expected agent name 'test-agent', got %v", agent["name"])
		}
		if agent["backend"] != "test-backend" {
			t.Errorf("Expected backend 'test-backend', got %v", agent["backend"])
		}
		if agent["ready"] != true {
			t.Errorf("Expected agent to be ready")
		}
	})

	t.Run("returns error when manager is nil", func(t *testing.T) {
		h := NewAgentsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/agents", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_manager" {
			t.Error("Expected error code 'no_manager'")
		}
	})

	t.Run("returns agents with runtime manager", func(t *testing.T) {
		// Create backend and register it
		tb := &testBackend{
			name:      "test-backend",
			available: true,
			response:  "Hello from test",
		}
		registry := backend.NewRegistry()
		if err := registry.Register("test-backend", tb); err != nil {
			t.Fatalf("Failed to register backend: %v", err)
		}

		// Create manager and agent
		manager := runtime.NewManager(registry)
		agentDef := wool.Agent{
			ID:           "test-agent-001",
			Name:         "test-agent",
			Role:         wool.RoleJunior,
			Backend:      "test-backend",
			Model:        "test-model",
			SystemPrompt: "You are a test agent",
		}
		if _, err := manager.Create(agentDef); err != nil {
			t.Fatalf("Failed to create agent: %v", err)
		}

		h := NewAgentsHandlerWithRuntime(manager)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/agents", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		agents := data["agents"].([]interface{})
		if len(agents) != 1 {
			t.Errorf("Expected 1 agent, got %d", len(agents))
		}
	})
}

// -----------------------------------------------------------------------------
// ChatWithAgent Tests
// -----------------------------------------------------------------------------

func TestAgentsHandler_ChatWithAgent(t *testing.T) {
	t.Run("successful chat with mock", func(t *testing.T) {
		mockMgr := NewMockAgentManager()
		mockMgr.AddMockAgent(&MockAgent{
			name:         "test-agent",
			role:         "junior",
			backend:      "test-backend",
			model:        "test-model",
			ready:        true,
			chatResponse: "Hello! I am the test agent response.",
		})

		h := NewAgentsHandler(mockMgr)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ChatAgentRequest{
			Message: "Hello, agent!",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/agents/test-agent/chat", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
			t.Logf("Response: %s", rec.Body.String())
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		if data["content"] != "Hello! I am the test agent response." {
			t.Errorf("Expected response content, got %v", data["content"])
		}
		if data["agent"] != "test-agent" {
			t.Errorf("Expected agent 'test-agent', got %v", data["agent"])
		}
	})

	t.Run("successful chat with runtime manager", func(t *testing.T) {
		// Create backend and register it
		tb := &testBackend{
			name:      "test-backend",
			available: true,
			response:  "Hello! I am the test agent response.",
		}
		registry := backend.NewRegistry()
		if err := registry.Register("test-backend", tb); err != nil {
			t.Fatalf("Failed to register backend: %v", err)
		}

		// Create manager and agent
		manager := runtime.NewManager(registry)
		agentDef := wool.Agent{
			ID:           "test-agent-001",
			Name:         "test-agent",
			Role:         wool.RoleJunior,
			Backend:      "test-backend",
			Model:        "test-model",
			SystemPrompt: "You are a test agent",
		}
		if _, err := manager.Create(agentDef); err != nil {
			t.Fatalf("Failed to create agent: %v", err)
		}

		h := NewAgentsHandlerWithRuntime(manager)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ChatAgentRequest{
			Message: "Hello, agent!",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/agents/test-agent/chat", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
			t.Logf("Response: %s", rec.Body.String())
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		if data["content"] != "Hello! I am the test agent response." {
			t.Errorf("Expected response content, got %v", data["content"])
		}
		if data["agent"] != "test-agent" {
			t.Errorf("Expected agent 'test-agent', got %v", data["agent"])
		}
	})

	t.Run("chat with history", func(t *testing.T) {
		mockMgr := NewMockAgentManager()
		mockMgr.AddMockAgent(&MockAgent{
			name:         "test-agent",
			role:         "junior",
			backend:      "test-backend",
			model:        "test-model",
			ready:        true,
			chatResponse: "I remember our previous conversation!",
		})

		h := NewAgentsHandler(mockMgr)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ChatAgentRequest{
			Message: "Do you remember?",
			History: []ChatHistoryMessage{
				{Role: "user", Content: "Hello!"},
				{Role: "assistant", Content: "Hi there!"},
			},
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/agents/test-agent/chat", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}
	})

	t.Run("agent not found", func(t *testing.T) {
		mockMgr := NewMockAgentManager()
		h := NewAgentsHandler(mockMgr)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ChatAgentRequest{
			Message: "Hello!",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/agents/nonexistent/chat", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "agent_not_found" {
			t.Error("Expected error code 'agent_not_found'")
		}
	})

	t.Run("empty message", func(t *testing.T) {
		mockMgr := NewMockAgentManager()
		h := NewAgentsHandler(mockMgr)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ChatAgentRequest{
			Message: "",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/agents/test-agent/chat", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "empty_message" {
			t.Error("Expected error code 'empty_message'")
		}
	})

	t.Run("invalid JSON", func(t *testing.T) {
		mockMgr := NewMockAgentManager()
		h := NewAgentsHandler(mockMgr)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/agents/test-agent/chat", bytes.NewReader([]byte("not valid json")))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "invalid_json" {
			t.Error("Expected error code 'invalid_json'")
		}
	})

	t.Run("chat error from mock agent", func(t *testing.T) {
		mockMgr := NewMockAgentManager()
		mockMgr.AddMockAgent(&MockAgent{
			name:      "test-agent",
			role:      "junior",
			backend:   "test-backend",
			model:     "test-model",
			ready:     true,
			chatError: errors.New("backend connection failed"),
		})

		h := NewAgentsHandler(mockMgr)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ChatAgentRequest{
			Message: "Hello!",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/agents/test-agent/chat", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusInternalServerError {
			t.Errorf("Expected status 500, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "chat_error" {
			t.Error("Expected error code 'chat_error'")
		}
	})

	t.Run("chat error from backend", func(t *testing.T) {
		// Create backend that returns error
		tb := &testBackend{
			name:      "test-backend",
			available: true,
			err:       errors.New("backend connection failed"),
		}
		registry := backend.NewRegistry()
		if err := registry.Register("test-backend", tb); err != nil {
			t.Fatalf("Failed to register backend: %v", err)
		}

		// Create manager and agent
		manager := runtime.NewManager(registry)
		agentDef := wool.Agent{
			ID:           "test-agent-001",
			Name:         "test-agent",
			Role:         wool.RoleJunior,
			Backend:      "test-backend",
			Model:        "test-model",
			SystemPrompt: "You are a test agent",
		}
		if _, err := manager.Create(agentDef); err != nil {
			t.Fatalf("Failed to create agent: %v", err)
		}

		h := NewAgentsHandlerWithRuntime(manager)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ChatAgentRequest{
			Message: "Hello!",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/agents/test-agent/chat", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusInternalServerError {
			t.Errorf("Expected status 500, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "chat_error" {
			t.Error("Expected error code 'chat_error'")
		}
	})

	t.Run("manager is nil", func(t *testing.T) {
		h := NewAgentsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ChatAgentRequest{
			Message: "Hello!",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/agents/test-agent/chat", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_manager" {
			t.Error("Expected error code 'no_manager'")
		}
	})
}

// -----------------------------------------------------------------------------
// Helper Function Tests
// -----------------------------------------------------------------------------

func TestBuildMessageHistory(t *testing.T) {
	t.Run("empty history", func(t *testing.T) {
		messages := buildMessageHistory(nil, "Hello")
		if len(messages) != 1 {
			t.Errorf("Expected 1 message, got %d", len(messages))
		}
		if messages[0].Content != "Hello" {
			t.Errorf("Expected 'Hello', got %s", messages[0].Content)
		}
		if messages[0].Role != yarn.RoleUser {
			t.Errorf("Expected user role, got %s", messages[0].Role)
		}
	})

	t.Run("with history", func(t *testing.T) {
		history := []ChatHistoryMessage{
			{Role: "user", Content: "Hi"},
			{Role: "assistant", Content: "Hello!"},
		}
		messages := buildMessageHistory(history, "How are you?")
		if len(messages) != 3 {
			t.Errorf("Expected 3 messages, got %d", len(messages))
		}
		if messages[0].Content != "Hi" {
			t.Errorf("Expected 'Hi', got %s", messages[0].Content)
		}
		if messages[1].Content != "Hello!" {
			t.Errorf("Expected 'Hello!', got %s", messages[1].Content)
		}
		if messages[2].Content != "How are you?" {
			t.Errorf("Expected 'How are you?', got %s", messages[2].Content)
		}
	})

	t.Run("with unknown role defaults to user", func(t *testing.T) {
		history := []ChatHistoryMessage{
			{Role: "unknown_role", Content: "Test"},
		}
		messages := buildMessageHistory(history, "Hello")
		if messages[0].Role != yarn.RoleUser {
			t.Errorf("Expected unknown role to default to user, got %s", messages[0].Role)
		}
	})

	t.Run("with agent name", func(t *testing.T) {
		history := []ChatHistoryMessage{
			{Role: "assistant", Content: "Hi", Name: "senior"},
		}
		messages := buildMessageHistory(history, "Hello")
		if messages[0].AgentName != "senior" {
			t.Errorf("Expected agent name 'senior', got %s", messages[0].AgentName)
		}
	})
}

func TestJoinAgentNames(t *testing.T) {
	tests := []struct {
		names    []string
		expected string
	}{
		{nil, "(none)"},
		{[]string{}, "(none)"},
		{[]string{"agent1"}, "agent1"},
		{[]string{"agent1", "agent2"}, "agent1, agent2"},
		{[]string{"a", "b", "c"}, "a, b, c"},
	}

	for _, tt := range tests {
		result := joinAgentNames(tt.names)
		if result != tt.expected {
			t.Errorf("joinAgentNames(%v) = %q, expected %q", tt.names, result, tt.expected)
		}
	}
}

// -----------------------------------------------------------------------------
// Route Registration Tests
// -----------------------------------------------------------------------------

func TestAgentsHandler_RegisterRoutes(t *testing.T) {
	mockMgr := NewMockAgentManager()
	h := NewAgentsHandler(mockMgr)
	router := NewRouter()
	h.RegisterRoutes(router)

	// Test that routes are registered by checking if they respond
	tests := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/agents"},
		{http.MethodPost, "/api/agents/test/chat"},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			var body *bytes.Reader
			if tt.method == http.MethodPost {
				body = bytes.NewReader([]byte(`{"message":"test"}`))
			}

			var req *http.Request
			if body != nil {
				req = httptest.NewRequest(tt.method, tt.path, body)
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, tt.path, nil)
			}
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			// Should not get 404 (not found means route isn't registered)
			if rec.Code == http.StatusNotFound {
				resp := parseAPIResponse(t, rec.Body)
				// Only fail if this is the router's "not found" error, not our handler's error
				if resp.Error != nil && resp.Error.Code == "not_found" {
					t.Errorf("Route %s %s not found", tt.method, tt.path)
				}
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Mock Agent Manager Tests
// -----------------------------------------------------------------------------

func TestMockAgentManager(t *testing.T) {
	t.Run("add and get mock agent", func(t *testing.T) {
		mgr := NewMockAgentManager()
		agent := &MockAgent{
			name:         "test-agent",
			role:         "junior",
			backend:      "loom",
			model:        "test-model",
			ready:        true,
			chatResponse: "Mock response",
		}
		mgr.AddMockAgent(agent)

		retrieved, ok := mgr.Get("test-agent")
		if !ok {
			t.Error("Expected to find agent")
		}
		if retrieved.Name() != "test-agent" {
			t.Errorf("Expected name 'test-agent', got %s", retrieved.Name())
		}
	})

	t.Run("get nonexistent agent", func(t *testing.T) {
		mgr := NewMockAgentManager()
		_, ok := mgr.Get("nonexistent")
		if ok {
			t.Error("Expected not to find agent")
		}
	})

	t.Run("list agents", func(t *testing.T) {
		mgr := NewMockAgentManager()
		mgr.AddMockAgent(&MockAgent{name: "agent1"})
		mgr.AddMockAgent(&MockAgent{name: "agent2"})

		names := mgr.List()
		if len(names) != 2 {
			t.Errorf("Expected 2 agents, got %d", len(names))
		}
	})

	t.Run("status returns all agents", func(t *testing.T) {
		mgr := NewMockAgentManager()
		mgr.AddMockAgent(&MockAgent{
			name:    "agent1",
			backend: "loom",
			model:   "model1",
			ready:   true,
		})

		status := mgr.Status(context.Background())
		if len(status) != 1 {
			t.Errorf("Expected 1 agent status, got %d", len(status))
		}
		if status["agent1"].Ready != true {
			t.Error("Expected agent to be ready")
		}
	})

	t.Run("mock agent chat", func(t *testing.T) {
		agent := &MockAgent{
			name:         "test-agent",
			chatResponse: "Hello from mock!",
		}

		messages := []*yarn.Message{
			yarn.NewMessage(yarn.RoleUser, "Hi"),
		}

		response, err := agent.Chat(context.Background(), messages)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if response.Content != "Hello from mock!" {
			t.Errorf("Expected 'Hello from mock!', got %s", response.Content)
		}
	})

	t.Run("mock agent chat error", func(t *testing.T) {
		agent := &MockAgent{
			name:      "test-agent",
			chatError: errors.New("mock error"),
		}

		messages := []*yarn.Message{
			yarn.NewMessage(yarn.RoleUser, "Hi"),
		}

		_, err := agent.Chat(context.Background(), messages)
		if err == nil {
			t.Error("Expected error")
		}
	})
}

// -----------------------------------------------------------------------------
// Interface Verification Tests
// -----------------------------------------------------------------------------

func TestAgentManagerInterface(t *testing.T) {
	// Verify MockAgentManager implements AgentManager
	var _ AgentManager = (*MockAgentManager)(nil)
}

func TestChatableAgentInterface(t *testing.T) {
	// Verify MockAgent implements ChatableAgent
	var _ ChatableAgent = (*MockAgent)(nil)
}
