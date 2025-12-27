package wool

import (
	"testing"
)

// TestAgentValidate tests the Validate method for Agent.
func TestAgentValidate(t *testing.T) {
	tests := []struct {
		name        string
		agent       Agent
		wantErr     bool
		wantField   string
		wantMessage string
	}{
		{
			name: "valid senior agent with loom backend",
			agent: Agent{
				Name:         "test-senior",
				Role:         RoleSenior,
				Backend:      "loom",
				SystemPrompt: "You are a senior engineer.",
				ToolsEnabled: true,
				Active:       true,
			},
			wantErr: false,
		},
		{
			name: "valid senior agent with claudecode backend",
			agent: Agent{
				Name:         "test-senior",
				Role:         RoleSenior,
				Backend:      "claudecode",
				SystemPrompt: "You are a senior engineer.",
				ToolsEnabled: true,
				Active:       true,
			},
			wantErr: false,
		},
		{
			name: "valid junior agent",
			agent: Agent{
				Name:         "test-junior",
				Role:         RoleJunior,
				Backend:      "loom",
				SystemPrompt: "You are a junior engineer.",
				ToolsEnabled: true,
				Active:       true,
			},
			wantErr: false,
		},
		{
			name: "valid subject agent without tools",
			agent: Agent{
				Name:         "test-subject",
				Role:         RoleSubject,
				Backend:      "loom",
				SystemPrompt: "You are a subject.",
				ToolsEnabled: false,
				Active:       true,
			},
			wantErr: false,
		},
		{
			name: "valid conversant agent",
			agent: Agent{
				Name:         "test-conversant",
				Role:         RoleConversant,
				Backend:      "loom",
				SystemPrompt: "You are a conversant.",
				ToolsEnabled: false,
				Active:       true,
			},
			wantErr: false,
		},
		{
			name: "valid observer agent",
			agent: Agent{
				Name:         "test-observer",
				Role:         RoleObserver,
				Backend:      "loom",
				SystemPrompt: "You are an observer.",
				ToolsEnabled: false,
				Active:       false,
			},
			wantErr: false,
		},
		{
			name: "valid agent with inference parameters",
			agent: Agent{
				Name:        "test-agent",
				Role:        RoleJunior,
				Backend:     "loom",
				Temperature: 0.7,
				TopP:        0.9,
				MaxTokens:   2048,
				TopK:        40,
			},
			wantErr: false,
		},
		{
			name: "valid agent with zero temperature",
			agent: Agent{
				Name:        "test-agent",
				Role:        RoleJunior,
				Backend:     "loom",
				Temperature: 0,
			},
			wantErr: false,
		},
		{
			name: "valid agent with max temperature",
			agent: Agent{
				Name:        "test-agent",
				Role:        RoleJunior,
				Backend:     "loom",
				Temperature: 2.0,
			},
			wantErr: false,
		},
		{
			name: "valid agent with zero top_p",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleJunior,
				Backend: "loom",
				TopP:    0,
			},
			wantErr: false,
		},
		{
			name: "valid agent with max top_p",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleJunior,
				Backend: "loom",
				TopP:    1.0,
			},
			wantErr: false,
		},
		{
			name: "valid agent with capabilities",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleJunior,
				Backend: "loom",
				Capabilities: []Capability{
					{Name: "code-review", Description: "Review code", Enabled: true},
					{Name: "testing", Description: "Run tests", Enabled: false},
				},
			},
			wantErr: false,
		},
		{
			name: "valid agent with tools list",
			agent: Agent{
				Name:         "test-agent",
				Role:         RoleJunior,
				Backend:      "loom",
				ToolsEnabled: true,
				Tools:        []string{"read_file", "write_file", "execute_command"},
			},
			wantErr: false,
		},
		{
			name: "valid agent with gpu setting",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleJunior,
				Backend: "loom",
				GPU:     "auto",
			},
			wantErr: false,
		},
		{
			name: "valid agent with specific gpu",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleJunior,
				Backend: "loom",
				GPU:     "0",
			},
			wantErr: false,
		},
		{
			name: "missing name",
			agent: Agent{
				Name:    "",
				Role:    RoleSenior,
				Backend: "loom",
			},
			wantErr:     true,
			wantField:   "name",
			wantMessage: "name is required",
		},
		{
			name: "invalid role",
			agent: Agent{
				Name:    "test-agent",
				Role:    Role("invalid"),
				Backend: "loom",
			},
			wantErr:     true,
			wantField:   "role",
			wantMessage: "invalid role",
		},
		{
			name: "empty role",
			agent: Agent{
				Name:    "test-agent",
				Role:    Role(""),
				Backend: "loom",
			},
			wantErr:     true,
			wantField:   "role",
			wantMessage: "invalid role",
		},
		{
			name: "missing backend",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleSenior,
				Backend: "",
			},
			wantErr:     true,
			wantField:   "backend",
			wantMessage: "backend is required",
		},
		{
			name: "invalid backend",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleSenior,
				Backend: "invalid",
			},
			wantErr:     true,
			wantField:   "backend",
			wantMessage: "invalid backend",
		},
		{
			name: "typo in backend",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleSenior,
				Backend: "looom",
			},
			wantErr:     true,
			wantField:   "backend",
			wantMessage: "invalid backend",
		},
		{
			name: "capitalized backend is invalid",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleSenior,
				Backend: "Loom",
			},
			wantErr:     true,
			wantField:   "backend",
			wantMessage: "invalid backend",
		},
		{
			name: "tools enabled for role that does not support tools",
			agent: Agent{
				Name:         "test-subject",
				Role:         RoleSubject,
				Backend:      "loom",
				ToolsEnabled: true,
			},
			wantErr:     true,
			wantField:   "tools_enabled",
			wantMessage: "role 'subject' does not support tools",
		},
		{
			name: "tools enabled for conversant role",
			agent: Agent{
				Name:         "test-conversant",
				Role:         RoleConversant,
				Backend:      "loom",
				ToolsEnabled: true,
			},
			wantErr:     true,
			wantField:   "tools_enabled",
			wantMessage: "role 'conversant' does not support tools",
		},
		{
			name: "tools enabled for observer role",
			agent: Agent{
				Name:         "test-observer",
				Role:         RoleObserver,
				Backend:      "loom",
				ToolsEnabled: true,
			},
			wantErr:     true,
			wantField:   "tools_enabled",
			wantMessage: "role 'observer' does not support tools",
		},
		{
			name: "negative temperature",
			agent: Agent{
				Name:        "test-agent",
				Role:        RoleJunior,
				Backend:     "loom",
				Temperature: -0.1,
			},
			wantErr:     true,
			wantField:   "temperature",
			wantMessage: "temperature must be between 0.0 and 2.0",
		},
		{
			name: "temperature too high",
			agent: Agent{
				Name:        "test-agent",
				Role:        RoleJunior,
				Backend:     "loom",
				Temperature: 2.1,
			},
			wantErr:     true,
			wantField:   "temperature",
			wantMessage: "temperature must be between 0.0 and 2.0",
		},
		{
			name: "negative top_p",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleJunior,
				Backend: "loom",
				TopP:    -0.1,
			},
			wantErr:     true,
			wantField:   "top_p",
			wantMessage: "top_p must be between 0.0 and 1.0",
		},
		{
			name: "top_p too high",
			agent: Agent{
				Name:    "test-agent",
				Role:    RoleJunior,
				Backend: "loom",
				TopP:    1.1,
			},
			wantErr:     true,
			wantField:   "top_p",
			wantMessage: "top_p must be between 0.0 and 1.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.agent.Validate()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Agent.Validate() expected error, got nil")
					return
				}
				valErr, ok := err.(*ValidationError)
				if !ok {
					t.Errorf("Agent.Validate() error type = %T, want *ValidationError", err)
					return
				}
				if valErr.Field != tt.wantField {
					t.Errorf("Agent.Validate() error field = %q, want %q", valErr.Field, tt.wantField)
				}
				if valErr.Message != tt.wantMessage {
					t.Errorf("Agent.Validate() error message = %q, want %q", valErr.Message, tt.wantMessage)
				}
			} else {
				if err != nil {
					t.Errorf("Agent.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// TestValidationErrorError tests the Error method for ValidationError.
func TestValidationErrorError(t *testing.T) {
	tests := []struct {
		name    string
		err     ValidationError
		wantStr string
	}{
		{
			name:    "field and message",
			err:     ValidationError{Field: "name", Message: "name is required"},
			wantStr: "name: name is required",
		},
		{
			name:    "different field",
			err:     ValidationError{Field: "backend", Message: "backend must be 'loom' or 'claudecode'"},
			wantStr: "backend: backend must be 'loom' or 'claudecode'",
		},
		{
			name:    "empty field",
			err:     ValidationError{Field: "", Message: "some error"},
			wantStr: ": some error",
		},
		{
			name:    "empty message",
			err:     ValidationError{Field: "field", Message: ""},
			wantStr: "field: ",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.wantStr {
				t.Errorf("ValidationError.Error() = %q, want %q", got, tt.wantStr)
			}
		})
	}
}

// TestDefaultSenior tests the DefaultSenior function.
func TestDefaultSenior(t *testing.T) {
	agent := DefaultSenior()

	if agent.Name != "senior" {
		t.Errorf("DefaultSenior().Name = %q, want %q", agent.Name, "senior")
	}
	if agent.Role != RoleSenior {
		t.Errorf("DefaultSenior().Role = %q, want %q", agent.Role, RoleSenior)
	}
	if agent.Backend != "claudecode" {
		t.Errorf("DefaultSenior().Backend = %q, want %q", agent.Backend, "claudecode")
	}
	if !agent.ToolsEnabled {
		t.Errorf("DefaultSenior().ToolsEnabled = %v, want true", agent.ToolsEnabled)
	}
	if !agent.Active {
		t.Errorf("DefaultSenior().Active = %v, want true", agent.Active)
	}
	if agent.SystemPrompt == "" {
		t.Error("DefaultSenior().SystemPrompt should not be empty")
	}

	// Validate default agent should pass validation
	if err := agent.Validate(); err != nil {
		t.Errorf("DefaultSenior().Validate() unexpected error: %v", err)
	}
}

// TestDefaultJunior tests the DefaultJunior function.
func TestDefaultJunior(t *testing.T) {
	agent := DefaultJunior()

	if agent.Name != "junior" {
		t.Errorf("DefaultJunior().Name = %q, want %q", agent.Name, "junior")
	}
	if agent.Role != RoleJunior {
		t.Errorf("DefaultJunior().Role = %q, want %q", agent.Role, RoleJunior)
	}
	if agent.Backend != "loom" {
		t.Errorf("DefaultJunior().Backend = %q, want %q", agent.Backend, "loom")
	}
	if !agent.ToolsEnabled {
		t.Errorf("DefaultJunior().ToolsEnabled = %v, want true", agent.ToolsEnabled)
	}
	if !agent.Active {
		t.Errorf("DefaultJunior().Active = %v, want true", agent.Active)
	}
	if agent.SystemPrompt == "" {
		t.Error("DefaultJunior().SystemPrompt should not be empty")
	}

	// Validate default agent should pass validation
	if err := agent.Validate(); err != nil {
		t.Errorf("DefaultJunior().Validate() unexpected error: %v", err)
	}
}

// TestDefaultSubject tests the DefaultSubject function.
func TestDefaultSubject(t *testing.T) {
	agent := DefaultSubject()

	if agent.Name != "subject" {
		t.Errorf("DefaultSubject().Name = %q, want %q", agent.Name, "subject")
	}
	if agent.Role != RoleSubject {
		t.Errorf("DefaultSubject().Role = %q, want %q", agent.Role, RoleSubject)
	}
	if agent.Backend != "loom" {
		t.Errorf("DefaultSubject().Backend = %q, want %q", agent.Backend, "loom")
	}
	if agent.ToolsEnabled {
		t.Errorf("DefaultSubject().ToolsEnabled = %v, want false", agent.ToolsEnabled)
	}
	if !agent.Active {
		t.Errorf("DefaultSubject().Active = %v, want true", agent.Active)
	}
	if agent.SystemPrompt == "" {
		t.Error("DefaultSubject().SystemPrompt should not be empty")
	}

	// Validate default agent should pass validation
	if err := agent.Validate(); err != nil {
		t.Errorf("DefaultSubject().Validate() unexpected error: %v", err)
	}
}

// TestValidBackends tests the ValidBackends variable.
func TestValidBackends(t *testing.T) {
	expected := []string{"loom", "claudecode"}

	if len(ValidBackends) != len(expected) {
		t.Errorf("len(ValidBackends) = %d, want %d", len(ValidBackends), len(expected))
	}

	for i, backend := range expected {
		if i >= len(ValidBackends) || ValidBackends[i] != backend {
			t.Errorf("ValidBackends[%d] = %q, want %q", i, ValidBackends[i], backend)
		}
	}
}

// TestCapabilityStruct tests that Capability struct can be properly instantiated.
func TestCapabilityStruct(t *testing.T) {
	cap := Capability{
		Name:        "code-review",
		Description: "Review code for quality and best practices",
		Enabled:     true,
	}

	if cap.Name != "code-review" {
		t.Errorf("Capability.Name = %q, want %q", cap.Name, "code-review")
	}
	if cap.Description != "Review code for quality and best practices" {
		t.Errorf("Capability.Description = %q, want %q", cap.Description, "Review code for quality and best practices")
	}
	if !cap.Enabled {
		t.Errorf("Capability.Enabled = %v, want true", cap.Enabled)
	}
}

// TestAgentWithAllFields tests an agent with all optional fields populated.
func TestAgentWithAllFields(t *testing.T) {
	agent := Agent{
		ID:            "agent-123",
		Name:          "full-agent",
		Role:          RoleSenior,
		Backend:       "claudecode",
		Model:         "claude-3-opus",
		SystemPrompt:  "You are a helpful assistant.",
		Tools:         []string{"read", "write", "execute"},
		ToolsEnabled:  true,
		Active:        true,
		MaxTokens:     4096,
		Temperature:   0.7,
		ContextLength: 8192,
		TopP:          0.95,
		TopK:          50,
		GPU:           "auto",
		Capabilities: []Capability{
			{Name: "planning", Description: "Create plans", Enabled: true},
		},
	}

	if err := agent.Validate(); err != nil {
		t.Errorf("Agent.Validate() with all fields unexpected error: %v", err)
	}

	// Verify all fields are set correctly
	if agent.ID != "agent-123" {
		t.Errorf("Agent.ID = %q, want %q", agent.ID, "agent-123")
	}
	if agent.Model != "claude-3-opus" {
		t.Errorf("Agent.Model = %q, want %q", agent.Model, "claude-3-opus")
	}
	if len(agent.Tools) != 3 {
		t.Errorf("len(Agent.Tools) = %d, want 3", len(agent.Tools))
	}
	if agent.MaxTokens != 4096 {
		t.Errorf("Agent.MaxTokens = %d, want 4096", agent.MaxTokens)
	}
	if agent.ContextLength != 8192 {
		t.Errorf("Agent.ContextLength = %d, want 8192", agent.ContextLength)
	}
	if agent.TopK != 50 {
		t.Errorf("Agent.TopK = %d, want 50", agent.TopK)
	}
}

// TestConfigAlias tests that Config is an alias for Agent.
func TestConfigAlias(t *testing.T) {
	// Config should be usable exactly like Agent
	var config Config = Agent{
		Name:    "config-test",
		Role:    RoleJunior,
		Backend: "loom",
	}

	if err := config.Validate(); err != nil {
		t.Errorf("Config.Validate() unexpected error: %v", err)
	}

	if config.Name != "config-test" {
		t.Errorf("Config.Name = %q, want %q", config.Name, "config-test")
	}
}
