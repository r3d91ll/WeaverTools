// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"context"
	"net/http"
	"sync"

	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/yarn"
)

// AgentManager is the interface for managing agents.
// This interface allows for dependency injection and testing.
type AgentManager interface {
	// Status returns the status of all agents.
	Status(ctx context.Context) map[string]runtime.AgentStatus
	// Get retrieves an agent by name.
	Get(name string) (ChatableAgent, bool)
	// List returns all agent names.
	List() []string
}

// ChatableAgent is the interface for an agent that can chat.
type ChatableAgent interface {
	// Chat sends messages to the agent and returns the response.
	Chat(ctx context.Context, messages []*yarn.Message) (*yarn.Message, error)
	// Name returns the agent name.
	Name() string
	// ModelName returns the model name.
	ModelName() string
}

// AgentsHandler handles agent-related API requests.
type AgentsHandler struct {
	// manager is the agent manager (runtime.Manager or mock)
	manager AgentManager

	// mu protects concurrent access
	mu sync.RWMutex
}

// NewAgentsHandler creates a new AgentsHandler with the given agent manager.
func NewAgentsHandler(manager AgentManager) *AgentsHandler {
	return &AgentsHandler{
		manager: manager,
	}
}

// NewAgentsHandlerWithRuntime creates a new AgentsHandler with a runtime.Manager.
// This is a convenience function for production use.
func NewAgentsHandlerWithRuntime(manager *runtime.Manager) *AgentsHandler {
	return &AgentsHandler{
		manager: &runtimeManagerAdapter{manager: manager},
	}
}

// runtimeManagerAdapter adapts *runtime.Manager to the AgentManager interface.
type runtimeManagerAdapter struct {
	manager *runtime.Manager
}

func (a *runtimeManagerAdapter) Status(ctx context.Context) map[string]runtime.AgentStatus {
	return a.manager.Status(ctx)
}

func (a *runtimeManagerAdapter) Get(name string) (ChatableAgent, bool) {
	agent, ok := a.manager.Get(name)
	if !ok {
		return nil, false
	}
	return agent, true
}

func (a *runtimeManagerAdapter) List() []string {
	return a.manager.List()
}

// RegisterRoutes registers the agent API routes on the router.
func (h *AgentsHandler) RegisterRoutes(router *Router) {
	router.GET("/api/agents", h.ListAgents)
	router.POST("/api/agents/:name/chat", h.ChatWithAgent)
}

// -----------------------------------------------------------------------------
// API Response Types
// -----------------------------------------------------------------------------

// AgentListResponse is the JSON response for GET /api/agents.
type AgentListResponse struct {
	Agents []AgentInfo `json:"agents"`
}

// AgentInfo is the JSON representation of an agent's information.
type AgentInfo struct {
	Name         string `json:"name"`
	Role         string `json:"role"`
	Backend      string `json:"backend"`
	Model        string `json:"model,omitempty"`
	Ready        bool   `json:"ready"`
	HiddenStates bool   `json:"hiddenStates"`
}

// ChatRequest is the expected JSON body for POST /api/agents/:name/chat.
type ChatAgentRequest struct {
	Message string `json:"message"`
	// History contains previous messages in the conversation
	History []ChatHistoryMessage `json:"history,omitempty"`
}

// ChatHistoryMessage represents a single message in chat history.
type ChatHistoryMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// ChatAgentResponse is the JSON response for POST /api/agents/:name/chat.
type ChatAgentResponse struct {
	Content   string                 `json:"content"`
	Agent     string                 `json:"agent"`
	Model     string                 `json:"model,omitempty"`
	LatencyMs float64                `json:"latencyMs,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------

// ListAgents handles GET /api/agents.
// It returns a list of all registered agents with their status.
func (h *AgentsHandler) ListAgents(w http.ResponseWriter, r *http.Request) {
	h.mu.RLock()
	manager := h.manager
	h.mu.RUnlock()

	if manager == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_manager",
			"Agent manager is not available")
		return
	}

	// Get agent status from manager
	ctx := r.Context()
	statusMap := manager.Status(ctx)

	// Convert to response format
	agents := make([]AgentInfo, 0, len(statusMap))
	for _, status := range statusMap {
		agents = append(agents, AgentInfo{
			Name:         status.Name,
			Role:         string(status.Role),
			Backend:      status.Backend,
			Model:        status.Model,
			Ready:        status.Ready,
			HiddenStates: status.HiddenStates,
		})
	}

	response := AgentListResponse{
		Agents: agents,
	}

	WriteJSON(w, http.StatusOK, response)
}

// ChatWithAgent handles POST /api/agents/:name/chat.
// It sends a message to the specified agent and returns the response.
func (h *AgentsHandler) ChatWithAgent(w http.ResponseWriter, r *http.Request) {
	// Get agent name from path
	agentName := PathParam(r, "name")
	if agentName == "" {
		WriteError(w, http.StatusBadRequest, "missing_agent_name",
			"Agent name is required in the URL path")
		return
	}

	// Parse request body
	var req ChatAgentRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	// Validate message
	if req.Message == "" {
		WriteError(w, http.StatusBadRequest, "empty_message",
			"Message cannot be empty")
		return
	}

	h.mu.RLock()
	manager := h.manager
	h.mu.RUnlock()

	if manager == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_manager",
			"Agent manager is not available")
		return
	}

	// Get the agent
	agent, ok := manager.Get(agentName)
	if !ok {
		availableAgents := manager.List()
		WriteError(w, http.StatusNotFound, "agent_not_found",
			"Agent '"+agentName+"' not found. Available agents: "+joinAgentNames(availableAgents))
		return
	}

	// Build message history for the agent
	messages := buildMessageHistory(req.History, req.Message)

	// Send chat request to agent
	ctx := r.Context()
	response, err := agent.Chat(ctx, messages)
	if err != nil {
		WriteError(w, http.StatusInternalServerError, "chat_error",
			"Failed to chat with agent: "+err.Error())
		return
	}

	// Build response with metadata
	metadata := make(map[string]interface{})
	if response.Metadata != nil {
		for k, v := range response.Metadata {
			metadata[k] = v
		}
	}

	// Extract latency from metadata if available
	var latencyMs float64
	if latency, ok := metadata["latency_ms"].(float64); ok {
		latencyMs = latency
	}

	chatResponse := ChatAgentResponse{
		Content:   response.Content,
		Agent:     agentName,
		Model:     agent.ModelName(),
		LatencyMs: latencyMs,
		Metadata:  metadata,
	}

	WriteJSON(w, http.StatusOK, chatResponse)
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// buildMessageHistory converts chat history and new message to yarn.Message format.
func buildMessageHistory(history []ChatHistoryMessage, newMessage string) []*yarn.Message {
	messages := make([]*yarn.Message, 0, len(history)+1)

	// Add history messages
	for _, h := range history {
		role := yarn.MessageRole(h.Role)
		if role != yarn.RoleUser && role != yarn.RoleAssistant && role != yarn.RoleSystem {
			role = yarn.RoleUser // Default to user for unknown roles
		}
		msg := yarn.NewMessage(role, h.Content)
		if h.Name != "" {
			msg.AgentName = h.Name
		}
		messages = append(messages, msg)
	}

	// Add new user message
	userMsg := yarn.NewMessage(yarn.RoleUser, newMessage)
	messages = append(messages, userMsg)

	return messages
}

// joinAgentNames joins agent names into a comma-separated string.
func joinAgentNames(names []string) string {
	if len(names) == 0 {
		return "(none)"
	}
	result := ""
	for i, name := range names {
		if i > 0 {
			result += ", "
		}
		result += name
	}
	return result
}

// -----------------------------------------------------------------------------
// Mock Manager for Testing
// -----------------------------------------------------------------------------

// MockAgentManager is a mock implementation of AgentManager for testing.
// It allows tests to run without a real backend.
type MockAgentManager struct {
	agents map[string]*MockAgent
	mu     sync.RWMutex
}

// Ensure MockAgentManager implements AgentManager.
var _ AgentManager = (*MockAgentManager)(nil)

// MockAgent represents a mock agent for testing.
type MockAgent struct {
	name         string
	role         string
	backend      string
	model        string
	ready        bool
	hiddenStates bool
	chatResponse string
	chatError    error
}

// Ensure MockAgent implements ChatableAgent.
var _ ChatableAgent = (*MockAgent)(nil)

// NewMockAgentManager creates a new mock agent manager.
func NewMockAgentManager() *MockAgentManager {
	return &MockAgentManager{
		agents: make(map[string]*MockAgent),
	}
}

// AddMockAgent adds a mock agent to the manager.
func (m *MockAgentManager) AddMockAgent(agent *MockAgent) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.agents[agent.name] = agent
}

// Status returns status for all mock agents.
func (m *MockAgentManager) Status(ctx context.Context) map[string]runtime.AgentStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]runtime.AgentStatus)
	for name, agent := range m.agents {
		result[name] = runtime.AgentStatus{
			Name:         name,
			Role:         "junior", // Use a valid wool.Role string representation
			Backend:      agent.backend,
			Model:        agent.model,
			Ready:        agent.ready,
			HiddenStates: agent.hiddenStates,
		}
	}
	return result
}

// Get retrieves a mock agent by name.
// Returns the agent wrapped as ChatableAgent interface.
func (m *MockAgentManager) Get(name string) (ChatableAgent, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	agent, ok := m.agents[name]
	if !ok {
		return nil, false
	}
	return agent, true
}

// List returns all mock agent names.
func (m *MockAgentManager) List() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	names := make([]string, 0, len(m.agents))
	for name := range m.agents {
		names = append(names, name)
	}
	return names
}

// Chat simulates a chat with a mock agent.
func (a *MockAgent) Chat(ctx context.Context, messages []*yarn.Message) (*yarn.Message, error) {
	if a.chatError != nil {
		return nil, a.chatError
	}
	response := yarn.NewMessage(yarn.RoleAssistant, a.chatResponse)
	response.AgentName = a.name
	response.WithMetadata("model", a.model)
	response.WithMetadata("latency_ms", 100.0)
	return response, nil
}

// Name returns the mock agent's name.
func (a *MockAgent) Name() string {
	return a.name
}

// ModelName returns the mock agent's model name.
func (a *MockAgent) ModelName() string {
	return a.model
}
