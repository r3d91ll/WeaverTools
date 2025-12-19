// Package runtime provides live agent instances from Wool definitions.
package runtime

import (
	"context"
	"fmt"
	"sync"

	"github.com/r3d91ll/weaver/pkg/backend"
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
	a.mu.RLock()
	defer a.mu.RUnlock()

	// Convert Yarn messages to backend format
	chatMessages := make([]backend.ChatMessage, 0, len(messages)+1)

	// Add system prompt
	if a.Definition.SystemPrompt != "" {
		chatMessages = append(chatMessages, backend.ChatMessage{
			Role:    "system",
			Content: a.Definition.SystemPrompt,
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
	device := a.Definition.GPU
	if device != "" && device != "auto" {
		device = "cuda:" + device
	}

	req := backend.ChatRequest{
		Model:              a.Definition.Model,
		Messages:           chatMessages,
		MaxTokens:          a.Definition.MaxTokens,
		Temperature:        a.Definition.Temperature,
		ReturnHiddenStates: a.Backend.SupportsHiddenStates(),
		Device:             device,
	}

	// Call backend
	resp, err := a.Backend.Chat(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("agent %s: %w", a.Definition.Name, err)
	}

	// Create response message
	result := yarn.NewAgentMessage(
		yarn.RoleAssistant,
		resp.Content,
		a.Definition.ID,
		a.Definition.Name,
	)

	// Attach hidden state if available
	if resp.HiddenState != nil {
		result.HiddenState = &yarn.HiddenState{
			Vector: resp.HiddenState.Vector,
			Shape:  resp.HiddenState.Shape,
			Layer:  resp.HiddenState.Layer,
			DType:  resp.HiddenState.DType,
		}
	}

	// Add metadata
	result.WithMetadata("model", resp.Model)
	result.WithMetadata("latency_ms", resp.LatencyMS)
	result.WithMetadata("finish_reason", resp.FinishReason)

	return result, nil
}

// Name returns the agent name.
func (a *Agent) Name() string {
	return a.Definition.Name
}

// Role returns the agent role.
func (a *Agent) Role() wool.Role {
	return a.Definition.Role
}

// SupportsHiddenStates returns true if this agent can provide hidden states.
func (a *Agent) SupportsHiddenStates() bool {
	return a.Backend.SupportsHiddenStates()
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
func (m *Manager) Create(def wool.Agent) (*Agent, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Get backend
	b, ok := m.registry.Get(def.Backend)
	if !ok {
		return nil, fmt.Errorf("backend %q not found", def.Backend)
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
			Backend:      agent.Definition.Backend,
			Model:        agent.Definition.Model,
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
		return fmt.Errorf("failed to create senior: %w", err)
	}

	// Create junior (Loom)
	junior := wool.DefaultJunior()
	junior.ID = "junior-001"
	junior.Model = juniorModel
	if _, err := m.Create(junior); err != nil {
		return fmt.Errorf("failed to create junior: %w", err)
	}

	return nil
}
