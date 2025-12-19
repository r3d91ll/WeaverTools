package wool

// Agent defines an agent's identity and capabilities.
// This is the specification - Weaver creates runtime instances from these.
type Agent struct {
	ID           string   `json:"id" yaml:"id"`
	Name         string   `json:"name" yaml:"name"`
	Role         Role     `json:"role" yaml:"role"`
	Backend      string   `json:"backend" yaml:"backend"`           // "loom" or "claudecode"
	Model        string   `json:"model,omitempty" yaml:"model"`     // Model ID for the backend
	SystemPrompt string   `json:"system_prompt" yaml:"system_prompt"`
	Tools        []string `json:"tools,omitempty" yaml:"tools"`     // Tool names this agent can use
	ToolsEnabled bool     `json:"tools_enabled" yaml:"tools_enabled"`
	Active       bool     `json:"active" yaml:"active"`             // Whether agent is active for this session

	// Inference parameters (for Loom backend)
	MaxTokens     int     `json:"max_tokens,omitempty" yaml:"max_tokens"`
	Temperature   float64 `json:"temperature,omitempty" yaml:"temperature"`
	ContextLength int     `json:"context_length,omitempty" yaml:"context_length"`
	TopP          float64 `json:"top_p,omitempty" yaml:"top_p"`
	TopK          int     `json:"top_k,omitempty" yaml:"top_k"`

	// GPU assignment (for Loom backend)
	// "auto" = let Loom decide, "0" = cuda:0, "1" = cuda:1, etc.
	GPU string `json:"gpu,omitempty" yaml:"gpu"`
}

// Config is an alias for Agent for YAML configuration loading.
type Config = Agent

// Capability defines what an agent can do.
type Capability struct {
	Name        string `json:"name" yaml:"name"`
	Description string `json:"description" yaml:"description"`
	Enabled     bool   `json:"enabled" yaml:"enabled"`
}

// DefaultSenior returns a default Senior agent configuration.
func DefaultSenior() Agent {
	return Agent{
		Name:    "senior",
		Role:    RoleSenior,
		Backend: "claudecode",
		SystemPrompt: `You are the Senior Engineer in a multi-agent AI research system.
Your role is to handle complex reasoning, architecture decisions, and code review.
You can delegate routine tasks to Junior agents using @junior <task>.`,
		ToolsEnabled: true,
	}
}

// DefaultJunior returns a default Junior agent configuration.
func DefaultJunior() Agent {
	return Agent{
		Name:    "junior",
		Role:    RoleJunior,
		Backend: "loom",
		SystemPrompt: `You are the Junior Engineer in a multi-agent AI research system.
Your role is to handle implementation tasks, file operations, and routine work.
Report results back to Senior for review.`,
		ToolsEnabled: true,
	}
}

// DefaultSubject returns a default Subject agent configuration for experiments.
func DefaultSubject() Agent {
	return Agent{
		Name:    "subject",
		Role:    RoleSubject,
		Backend: "loom",
		SystemPrompt: `You are a Subject in a conveyance measurement experiment.
Respond naturally to prompts. Your hidden states will be analyzed.`,
		ToolsEnabled: false,
	}
}

// Validate checks if the agent configuration is valid.
func (a *Agent) Validate() error {
	if a.Name == "" {
		return &ValidationError{Field: "name", Message: "name is required"}
	}
	if !a.Role.IsValid() {
		return &ValidationError{Field: "role", Message: "invalid role"}
	}
	if a.Backend == "" {
		return &ValidationError{Field: "backend", Message: "backend is required"}
	}
	return nil
}

// ValidationError represents a validation failure.
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return e.Field + ": " + e.Message
}
