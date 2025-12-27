// Package runtime provides live agent instances from Wool definitions.
package runtime

import (
	"context"
	"strings"
	"sync"

	"github.com/r3d91ll/weaver/pkg/backend"
	werrors "github.com/r3d91ll/weaver/pkg/errors"
	"github.com/r3d91ll/wool"
	"github.com/r3d91ll/yarn"
)

// Agent is a live agent instance with a backend connection.
type Agent struct {
	Definition wool.Agent
	Backend    backend.Backend
	mu         sync.RWMutex
}

// NewAgent creates a live agent from a Wool definition and backend.
func NewAgent(def wool.Agent, b backend.Backend) *Agent {
	return &Agent{
		Definition: def,
		Backend:    b,
	}
}

// Chat sends messages to the agent and returns the response with optional hidden state.
func (a *Agent) Chat(ctx context.Context, messages []*yarn.Message) (*yarn.Message, error) {
	// Copy fields under lock, then release before I/O to avoid lock contention
	a.mu.RLock()
	def := a.Definition // Copy the definition struct
	b := a.Backend      // Copy the backend pointer
	a.mu.RUnlock()

	// Convert Yarn messages to backend format
	chatMessages := make([]backend.ChatMessage, 0, len(messages)+1)

	// Add system prompt
	if def.SystemPrompt != "" {
		chatMessages = append(chatMessages, backend.ChatMessage{
			Role:    "system",
			Content: def.SystemPrompt,
		})
	}

	// Add conversation messages
	for _, msg := range messages {
		chatMessages = append(chatMessages, backend.ChatMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
			Name:    msg.AgentName,
		})
	}

	// Build request - always request hidden states if backend supports it
	// Convert GPU config to device string (e.g., "0" -> "cuda:0")
	device := def.GPU
	if device != "" && device != "auto" {
		device = "cuda:" + device
	}

	req := backend.ChatRequest{
		Model:              def.Model,
		Messages:           chatMessages,
		MaxTokens:          def.MaxTokens,
		Temperature:        def.Temperature,
		ReturnHiddenStates: b.Capabilities().SupportsHidden,
		Device:             device,
	}

	// Call backend (lock already released)
	resp, err := b.Chat(ctx, req)
	if err != nil {
		return nil, createAgentChatError(def.Name, def.Backend, err)
	}

	// Create response message
	result := yarn.NewAgentMessage(
		yarn.RoleAssistant,
		resp.Content,
		def.ID,
		def.Name,
	)

	// Attach hidden state if available (types now match, no conversion needed)
	result.HiddenState = resp.HiddenState

	// Add metadata
	result.WithMetadata("model", resp.Model)
	result.WithMetadata("latency_ms", resp.LatencyMS)
	result.WithMetadata("finish_reason", resp.FinishReason)

	return result, nil
}

// Name returns the agent name.
func (a *Agent) Name() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.Definition.Name
}

// Role returns the agent role.
func (a *Agent) Role() wool.Role {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.Definition.Role
}

// BackendName returns the configured backend name.
func (a *Agent) BackendName() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.Definition.Backend
}

// ModelName returns the configured model name.
func (a *Agent) ModelName() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.Definition.Model
}

// SupportsHiddenStates returns true if this agent can provide hidden states.
func (a *Agent) SupportsHiddenStates() bool {
	return a.Backend.Capabilities().SupportsHidden
}

// IsReady returns true if the agent's backend is available.
func (a *Agent) IsReady(ctx context.Context) bool {
	return a.Backend.IsAvailable(ctx)
}

// Manager manages live agent instances.
type Manager struct {
	agents   map[string]*Agent
	registry *backend.Registry
	mu       sync.RWMutex
}

// NewManager creates a new agent manager.
func NewManager(registry *backend.Registry) *Manager {
	return &Manager{
		agents:   make(map[string]*Agent),
		registry: registry,
	}
}

// Create creates a live agent from a Wool definition.
// Returns an error if an agent with the same name already exists.
func (m *Manager) Create(def wool.Agent) (*Agent, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check for duplicate
	if _, exists := m.agents[def.Name]; exists {
		return nil, createAgentAlreadyExistsError(def.Name, m.listAgentNamesLocked())
	}

	// Get backend
	b, ok := m.registry.Get(def.Backend)
	if !ok {
		return nil, createBackendNotFoundError(def.Name, def.Backend, m.registry.List())
	}

	// Create agent
	agent := NewAgent(def, b)
	m.agents[def.Name] = agent

	return agent, nil
}

// Get retrieves an agent by name.
func (m *Manager) Get(name string) (*Agent, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	agent, ok := m.agents[name]
	return agent, ok
}

// List returns all agent names.
func (m *Manager) List() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	names := make([]string, 0, len(m.agents))
	for name := range m.agents {
		names = append(names, name)
	}
	return names
}

// Status returns status for all agents.
func (m *Manager) Status(ctx context.Context) map[string]AgentStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]AgentStatus)
	for name, agent := range m.agents {
		result[name] = AgentStatus{
			Name:         name,
			Role:         agent.Role(),
			Backend:      agent.BackendName(),
			Model:        agent.ModelName(),
			Ready:        agent.IsReady(ctx),
			HiddenStates: agent.SupportsHiddenStates(),
		}
	}
	return result
}

// AgentStatus represents agent status.
type AgentStatus struct {
	Name         string    `json:"name"`
	Role         wool.Role `json:"role"`
	Backend      string    `json:"backend"`
	Model        string    `json:"model"`
	Ready        bool      `json:"ready"`
	HiddenStates bool      `json:"hidden_states"`
}

// CreateDefaults creates default senior and junior agents.
func (m *Manager) CreateDefaults(juniorModel string) error {
	// Create senior (Claude Code)
	senior := wool.DefaultSenior()
	senior.ID = "senior-001"
	if _, err := m.Create(senior); err != nil {
		return createDefaultAgentCreationError("senior", senior.Name, senior.Backend, err)
	}

	// Create junior (Loom)
	junior := wool.DefaultJunior()
	junior.ID = "junior-001"
	junior.Model = juniorModel
	if _, err := m.Create(junior); err != nil {
		return createDefaultAgentCreationError("junior", junior.Name, junior.Backend, err)
	}

	return nil
}

// listAgentNamesLocked returns all agent names without acquiring the lock.
// Must be called while holding the mutex.
func (m *Manager) listAgentNamesLocked() []string {
	names := make([]string, 0, len(m.agents))
	for name := range m.agents {
		names = append(names, name)
	}
	return names
}

// -----------------------------------------------------------------------------
// Error Creation Helpers
// -----------------------------------------------------------------------------

// createAgentChatError creates a structured error for chat failures.
// Detects specific failure modes and provides contextual suggestions.
func createAgentChatError(agentName, backendName string, cause error) *werrors.WeaverError {
	errStr := strings.ToLower(cause.Error())

	// Check if the cause is already a WeaverError (from backend)
	// In that case, wrap it to add agent context
	if we, ok := werrors.AsWeaverError(cause); ok {
		// Add agent context to the existing error
		return werrors.AgentWrap(we, werrors.ErrAgentChatFailed, "chat request failed").
			WithContext("agent", agentName).
			WithContext("backend", backendName)
	}

	// Detect specific error types
	switch {
	case isConnectionRefused(errStr):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed: backend connection refused").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("Ensure the backend service is running").
			WithSuggestion("Check your network configuration")

	case isTimeout(errStr):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed: request timed out").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("The backend is taking too long to respond").
			WithSuggestion("Try a simpler request or check backend status")

	case isAuthError(errStr):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed: authentication error").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("Check your API credentials or authentication status").
			WithSuggestion("For Claude Code: Run 'claude auth login'")

	case isRateLimit(errStr):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed: rate limit exceeded").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("Wait a moment before sending more requests").
			WithSuggestion("Check your API usage limits")

	case isModelNotFound(errStr):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed: model not found").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("Verify the model name in your agent configuration").
			WithSuggestion("Check available models with the backend")

	case isContextCanceled(errStr):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request was cancelled").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("The request was interrupted").
			WithSuggestion("Try sending the message again")

	default:
		// Generic chat failure
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed").
			WithContext("agent", agentName).
			WithContext("backend", backendName)
	}
}

// createAgentAlreadyExistsError creates a structured error when an agent name is already in use.
func createAgentAlreadyExistsError(name string, existingAgents []string) *werrors.WeaverError {
	err := werrors.Agent(werrors.ErrAgentAlreadyExists, "an agent with this name already exists").
		WithContext("agent", name)

	if len(existingAgents) > 0 {
		err.WithContext("existing_agents", strings.Join(existingAgents, ", "))
	}

	return err.
		WithSuggestion("Choose a different name for the new agent").
		WithSuggestion("Or modify the existing agent's configuration")
}

// createBackendNotFoundError creates a structured error when the backend is not registered.
func createBackendNotFoundError(agentName, backendName string, availableBackends []string) *werrors.WeaverError {
	err := werrors.Backend(werrors.ErrBackendNotFound, "the specified backend is not registered").
		WithContext("agent", agentName).
		WithContext("backend", backendName)

	if len(availableBackends) > 0 {
		err.WithContext("available_backends", strings.Join(availableBackends, ", "))
		err.WithSuggestion("Use one of the available backends: " + strings.Join(availableBackends, ", "))
	} else {
		err.WithSuggestion("No backends are currently registered")
		err.WithSuggestion("Check your configuration to enable backends")
	}

	return err.
		WithSuggestion("Verify the backend name in your agent configuration").
		WithSuggestion("Common backends: claudecode, loom")
}

// createDefaultAgentCreationError creates a structured error for default agent creation failures.
func createDefaultAgentCreationError(agentType, name, backend string, cause error) *werrors.WeaverError {
	// Check if cause is already a WeaverError
	if we, ok := werrors.AsWeaverError(cause); ok {
		// Wrap with additional context
		return werrors.AgentWrap(we, werrors.ErrAgentCreationFailed, "failed to create default "+agentType+" agent").
			WithContext("agent_type", agentType).
			WithContext("agent", name).
			WithContext("backend", backend)
	}

	return werrors.AgentWrap(cause, werrors.ErrAgentCreationFailed, "failed to create default "+agentType+" agent").
		WithContext("agent_type", agentType).
		WithContext("agent", name).
		WithContext("backend", backend).
		WithSuggestion("Check that the " + backend + " backend is available").
		WithSuggestion("Verify your configuration settings")
}

// -----------------------------------------------------------------------------
// Error Detection Helpers
// -----------------------------------------------------------------------------

// isConnectionRefused checks if the error indicates a connection refusal.
func isConnectionRefused(errStr string) bool {
	return strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "no such host") ||
		strings.Contains(errStr, "connection reset")
}

// isTimeout checks if the error indicates a timeout.
func isTimeout(errStr string) bool {
	return strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "deadline exceeded") ||
		strings.Contains(errStr, "timed out")
}

// isAuthError checks if the error indicates an authentication problem.
func isAuthError(errStr string) bool {
	return strings.Contains(errStr, "unauthorized") ||
		strings.Contains(errStr, "401") ||
		strings.Contains(errStr, "authentication") ||
		strings.Contains(errStr, "auth failed") ||
		strings.Contains(errStr, "invalid api key")
}

// isRateLimit checks if the error indicates rate limiting.
func isRateLimit(errStr string) bool {
	return strings.Contains(errStr, "rate limit") ||
		strings.Contains(errStr, "429") ||
		strings.Contains(errStr, "too many requests") ||
		strings.Contains(errStr, "quota exceeded")
}

// isModelNotFound checks if the error indicates a missing model.
func isModelNotFound(errStr string) bool {
	return strings.Contains(errStr, "model not found") ||
		strings.Contains(errStr, "unknown model") ||
		strings.Contains(errStr, "model does not exist")
}

// isContextCanceled checks if the error indicates a cancelled context.
func isContextCanceled(errStr string) bool {
	return strings.Contains(errStr, "context canceled") ||
		strings.Contains(errStr, "context cancelled") ||
		strings.Contains(errStr, "operation was canceled")
}
