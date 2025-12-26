// Weaver - Multi-Agent Orchestrator for AI Research
//
// Weaver coordinates AI agents (Claude Code + local models via The Loom)
// enabling multi-agent conversations with conveyance measurement.
//
// Components:
//   - Wool: Agent roles and definitions
//   - Yarn: Conversations, measurements, and storage
//   - Loom: Model engine (The Loom - separate server)
//   - Weaver: This orchestrator
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	"github.com/r3d91ll/weaver/pkg/backend"
	"github.com/r3d91ll/weaver/pkg/config"
	werrors "github.com/r3d91ll/weaver/pkg/errors"
	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/weaver/pkg/shell"
	"github.com/r3d91ll/wool"
	"github.com/r3d91ll/yarn"
)

const version = "2.0.0-alpha"

func main() {
	// Parse flags
	configPath := flag.String("config", "", "Config file path (default: ~/.config/weaver/config.yaml)")
	sessionName := flag.String("session", "default", "Session name")
	initConfig := flag.Bool("init", false, "Initialize default config file")
	showVersion := flag.Bool("version", false, "Show version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Printf("Weaver %s\n", version)
		os.Exit(0)
	}

	// Determine config path
	cfgPath := *configPath
	if cfgPath == "" {
		cfgPath = config.DefaultConfigPath()
	}

	// Initialize config if requested
	if *initConfig {
		if err := config.InitConfig(cfgPath); err != nil {
			werrors.Display(createConfigInitError(cfgPath, err))
			os.Exit(1)
		}
		fmt.Printf("Config initialized at: %s\n", cfgPath)
		fmt.Println("Edit this file to configure agents and backends.")
		os.Exit(0)
	}

	// Load config
	cfg, err := config.LoadOrDefault(cfgPath)
	if err != nil {
		werrors.Display(createConfigLoadError(cfgPath, err))
		os.Exit(1)
	}

	// Setup context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\nShutting down...")
		cancel()
	}()

	// Display banner
	fmt.Println("╔═══════════════════════════════════════════════════════════╗")
	fmt.Println("║           Weaver - Multi-Agent Orchestrator               ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Show config location
	if _, err := os.Stat(cfgPath); err == nil {
		fmt.Printf("Config: %s\n", cfgPath)
	} else {
		fmt.Printf("Config: (using defaults, run --init to create)\n")
	}
	fmt.Println()

	// Initialize backend registry
	registry := backend.NewRegistry()

	if cfg.Backends.ClaudeCode.Enabled {
		claudeCode := backend.NewClaudeCode(backend.ClaudeCodeConfig{})
		registry.Register("claudecode", claudeCode)
	}

	if cfg.Backends.Loom.Enabled {
		loom := backend.NewLoom(backend.LoomConfig{
			URL: cfg.Backends.Loom.URL,
		})
		registry.Register("loom", loom)
	}

	// Check backend availability
	fmt.Println("Backends:")
	status := registry.Status(ctx)
	for name, s := range status {
		availStr := "✗"
		if s.Available {
			availStr = "✓"
		}
		hiddenStr := ""
		if s.Capabilities.SupportsHidden {
			hiddenStr = " [hidden states]"
		}
		fmt.Printf("  %s %-12s (%s)%s\n", availStr, name, s.Type, hiddenStr)
	}
	fmt.Println()

	// Check at least one backend is available
	available := registry.Available(ctx)
	if len(available) == 0 {
		werrors.Display(createBackendUnavailableError(cfg, status))
		os.Exit(1)
	}

	// Create agent manager
	agentMgr := runtime.NewManager(registry)

	// Create agents from config (only active agents)
	// Sort agent names for consistent output across runs
	agentNames := make([]string, 0, len(cfg.Agents))
	for name := range cfg.Agents {
		agentNames = append(agentNames, name)
	}
	sort.Strings(agentNames)

	fmt.Println("Agents:")
	for _, name := range agentNames {
		agentCfg := cfg.Agents[name]
		// Skip inactive agents
		if !agentCfg.Active {
			fmt.Printf("  - %-10s (inactive)\n", name)
			continue
		}

		// Dereference pointer fields (guaranteed non-nil after InferenceDefaults)
		temp := float64(0)
		if agentCfg.Temperature != nil {
			temp = *agentCfg.Temperature
		}
		topP := float64(0)
		if agentCfg.TopP != nil {
			topP = *agentCfg.TopP
		}

		def := wool.Agent{
			ID:            name + "-001",
			Name:          name,
			Role:          wool.Role(agentCfg.Role),
			Backend:       agentCfg.Backend,
			Model:         agentCfg.Model,
			SystemPrompt:  agentCfg.SystemPrompt,
			Tools:         agentCfg.Tools,
			ToolsEnabled:  agentCfg.ToolsEnabled,
			Active:        agentCfg.Active,
			MaxTokens:     agentCfg.MaxTokens,
			Temperature:   temp,
			ContextLength: agentCfg.ContextLength,
			TopP:          topP,
			TopK:          agentCfg.TopK,
			GPU:           agentCfg.GPU,
		}

		agent, err := agentMgr.Create(def)
		if err != nil {
			werrors.Display(createAgentCreationError(name, agentCfg, err))
			continue
		}

		ready := "✗"
		if agent.IsReady(ctx) {
			ready = "✓"
		}
		hidden := ""
		if agent.SupportsHiddenStates() {
			hidden = " [hidden states]"
		}
		model := ""
		if agentCfg.Model != "" {
			model = fmt.Sprintf(" (%s)", agentCfg.Model)
		}
		fmt.Printf("  %s %-10s %s, %s%s%s\n", ready, name, agentCfg.Role, agentCfg.Backend, model, hidden)
	}
	fmt.Println()

	// Create session
	session := yarn.NewSession(*sessionName, "Weaver interactive session")
	session.Config.MeasurementMode = yarn.MeasurementMode(cfg.Session.MeasurementMode)
	session.Config.AutoExport = cfg.Session.AutoExport
	session.Config.ExportPath = cfg.Session.ExportPath

	fmt.Printf("Session: %s (%s)\n", session.Name, session.ID[:8])
	fmt.Println()

	// Get history file path
	homeDir, _ := os.UserHomeDir()
	historyFile := filepath.Join(homeDir, ".weaver_history")

	// Determine default agent (sorted for deterministic fallback)
	defaultAgent := "senior"
	if _, ok := cfg.Agents["senior"]; !ok {
		// Use first active agent (sorted alphabetically for consistency)
		names := make([]string, 0, len(cfg.Agents))
		for name := range cfg.Agents {
			if cfg.Agents[name].Active {
				names = append(names, name)
			}
		}
		if len(names) > 0 {
			sort.Strings(names)
			defaultAgent = names[0]
		}
	}

	// Create and run shell
	sh, err := shell.New(agentMgr, session, shell.Config{
		HistoryFile:  historyFile,
		DefaultAgent: defaultAgent,
	})
	if err != nil {
		fmt.Printf("Failed to create shell: %v\n", err)
		os.Exit(1)
	}

	if err := sh.Run(ctx); err != nil && err != context.Canceled {
		fmt.Printf("Shell error: %v\n", err)
		os.Exit(1)
	}

	// Export session on exit
	if session.Stats().MessageCount > 0 && cfg.Session.AutoExport {
		session.End()
		if err := session.Export(); err != nil {
			fmt.Printf("Warning: Failed to export session: %v\n", err)
		} else {
			fmt.Printf("Session exported to %s/%s\n", session.Config.ExportPath, session.ID)
		}
	}

	fmt.Println("Goodbye!")
}

// createConfigLoadError creates a structured error for config loading failures.
// It analyzes the underlying error to provide specific guidance on how to fix it.
func createConfigLoadError(path string, err error) *werrors.WeaverError {
	errStr := err.Error()

	// Check for file not found
	if os.IsNotExist(err) || strings.Contains(errStr, "no such file") {
		return werrors.ConfigNotFound(path)
	}

	// Check for YAML parse errors
	if strings.Contains(errStr, "yaml") || strings.Contains(errStr, "unmarshal") ||
		strings.Contains(errStr, "parse") {
		return werrors.ConfigParseError(path, err)
	}

	// Check for permission errors
	if os.IsPermission(err) || strings.Contains(errStr, "permission denied") {
		return werrors.ConfigWrap(err, werrors.ErrConfigReadFailed, "permission denied reading config file").
			WithContext("path", path)
	}

	// Generic config read failure
	return werrors.ConfigWrap(err, werrors.ErrConfigReadFailed, "failed to read configuration").
		WithContext("path", path)
}

// createConfigInitError creates a structured error for config initialization failures.
// It provides guidance on directory creation and permissions.
func createConfigInitError(path string, err error) *werrors.WeaverError {
	errStr := err.Error()

	// Check for permission errors
	if os.IsPermission(err) || strings.Contains(errStr, "permission denied") {
		return werrors.ConfigWrap(err, werrors.ErrConfigInitFailed, "permission denied creating config file").
			WithContext("path", path).
			WithContext("directory", filepath.Dir(path))
	}

	// Check for directory not found
	if strings.Contains(errStr, "no such file or directory") ||
		strings.Contains(errStr, "directory") {
		return werrors.ConfigWrap(err, werrors.ErrConfigInitFailed, "config directory does not exist").
			WithContext("path", path).
			WithContext("directory", filepath.Dir(path)).
			WithSuggestion("Create the directory first: mkdir -p " + filepath.Dir(path))
	}

	// Check for disk full or write errors
	if strings.Contains(errStr, "no space") || strings.Contains(errStr, "disk full") {
		return werrors.ConfigWrap(err, werrors.ErrConfigWriteFailed, "disk is full").
			WithContext("path", path)
	}

	// Generic init failure
	return werrors.ConfigWrap(err, werrors.ErrConfigInitFailed, "failed to initialize configuration").
		WithContext("path", path)
}

// createBackendUnavailableError creates a structured error when no backends are available.
// It provides specific suggestions for each configured backend type.
func createBackendUnavailableError(cfg *config.Config, status map[string]backend.Status) *werrors.WeaverError {
	// Build the error with context about configured backends
	err := werrors.Backend(werrors.ErrBackendUnavailable, "no backends available")

	// Track which backends are configured and their status
	var configuredBackends []string

	// Add context for Claude Code backend
	if cfg.Backends.ClaudeCode.Enabled {
		configuredBackends = append(configuredBackends, "claudecode")
		if s, ok := status["claudecode"]; ok && !s.Available {
			err = err.WithContext("claudecode_status", "not available")
		}
	}

	// Add context for Loom backend
	if cfg.Backends.Loom.Enabled {
		configuredBackends = append(configuredBackends, "loom")
		if s, ok := status["loom"]; ok && !s.Available {
			err = err.WithContext("loom_status", "not available")
			err = err.WithContext("loom_url", cfg.Backends.Loom.URL)
		}
	}

	// Add configured backends summary
	if len(configuredBackends) > 0 {
		err = err.WithContext("configured_backends", strings.Join(configuredBackends, ", "))
	}

	// Add backend-specific suggestions
	// Claude Code suggestions
	if cfg.Backends.ClaudeCode.Enabled {
		err = err.WithSuggestion("For Claude Code: Ensure 'claude' CLI is installed and in your PATH")
		err = err.WithSuggestion("Install Claude CLI with: npm install -g @anthropic-ai/claude-cli")
		err = err.WithSuggestion("After installing, run 'claude auth login' to authenticate")
	}

	// Loom suggestions
	if cfg.Backends.Loom.Enabled {
		err = err.WithSuggestion(fmt.Sprintf("For Loom: Ensure The Loom server is running at %s", cfg.Backends.Loom.URL))
		err = err.WithSuggestion("Check Loom server status with: curl " + cfg.Backends.Loom.URL + "/health")
		err = err.WithSuggestion("Start the Loom server if it's not running")
	}

	// General suggestions
	if !cfg.Backends.ClaudeCode.Enabled && !cfg.Backends.Loom.Enabled {
		err = err.WithSuggestion("Enable at least one backend in your config file")
		err = err.WithSuggestion("Run 'weaver --init' to create a default configuration with backends enabled")
	}

	return err
}

// createAgentCreationError creates a structured error for agent creation failures.
// It analyzes the underlying error to provide specific guidance on invalid fields,
// valid options, and example configurations.
func createAgentCreationError(name string, agentCfg config.AgentConfig, err error) *werrors.WeaverError {
	errStr := err.Error()

	// Valid options for reference
	validRoles := []string{"senior", "junior", "conversant", "subject", "observer"}
	validBackends := []string{"loom", "claudecode"}

	// Check for "agent already exists" error
	if strings.Contains(errStr, "already exists") {
		return werrors.Agent(werrors.ErrAgentAlreadyExists, fmt.Sprintf("agent '%s' already exists", name)).
			WithContext("agent", name).
			WithSuggestion("Each agent must have a unique name in the configuration").
			WithSuggestion("Rename the duplicate agent or remove one of the definitions")
	}

	// Check for "backend not found" error
	if strings.Contains(errStr, "not found") && strings.Contains(errStr, "backend") {
		return werrors.Agent(werrors.ErrAgentInvalidConfig, fmt.Sprintf("invalid backend '%s' for agent '%s'", agentCfg.Backend, name)).
			WithContext("agent", name).
			WithContext("invalid_field", "backend").
			WithContext("invalid_value", agentCfg.Backend).
			WithContext("valid_options", strings.Join(validBackends, ", ")).
			WithSuggestion(fmt.Sprintf("Valid backends are: %s", strings.Join(validBackends, ", "))).
			WithSuggestion("Update your config.yaml with a valid backend value").
			WithSuggestion("Example: backend: claudecode  # for Claude Code CLI").
			WithSuggestion("Example: backend: loom       # for The Loom server")
	}

	// Check for role validation errors
	if strings.Contains(errStr, "role") {
		return werrors.Agent(werrors.ErrAgentInvalidConfig, fmt.Sprintf("invalid role '%s' for agent '%s'", agentCfg.Role, name)).
			WithContext("agent", name).
			WithContext("invalid_field", "role").
			WithContext("invalid_value", agentCfg.Role).
			WithContext("valid_options", strings.Join(validRoles, ", ")).
			WithSuggestion(fmt.Sprintf("Valid roles are: %s", strings.Join(validRoles, ", "))).
			WithSuggestion("senior: high-level reasoning and orchestration (uses Claude Code)").
			WithSuggestion("junior: implementation tasks and tool execution (uses Loom)").
			WithSuggestion("subject: experiment participant for conveyance measurement").
			WithSuggestion("Example configuration:").
			WithSuggestion("  myagent:").
			WithSuggestion("    role: junior").
			WithSuggestion("    backend: loom").
			WithSuggestion("    model: Qwen/Qwen2.5-Coder-7B-Instruct")
	}

	// Check for tools_enabled with incompatible role
	if strings.Contains(errStr, "tools") && strings.Contains(errStr, "role") {
		return werrors.Agent(werrors.ErrAgentInvalidConfig, fmt.Sprintf("tools_enabled not supported for role '%s' on agent '%s'", agentCfg.Role, name)).
			WithContext("agent", name).
			WithContext("invalid_field", "tools_enabled").
			WithContext("role", agentCfg.Role).
			WithSuggestion("Only 'senior' and 'junior' roles support tools").
			WithSuggestion("Either change the role to senior/junior or set tools_enabled: false").
			WithSuggestion("For measurement experiments, use role: subject with tools_enabled: false")
	}

	// Check for temperature out of range
	if strings.Contains(errStr, "temperature") {
		temp := float64(0)
		if agentCfg.Temperature != nil {
			temp = *agentCfg.Temperature
		}
		return werrors.Agent(werrors.ErrAgentInvalidConfig, fmt.Sprintf("invalid temperature %.2f for agent '%s'", temp, name)).
			WithContext("agent", name).
			WithContext("invalid_field", "temperature").
			WithContext("invalid_value", fmt.Sprintf("%.2f", temp)).
			WithContext("valid_range", "0.0 - 2.0").
			WithSuggestion("Temperature must be between 0.0 and 2.0").
			WithSuggestion("Lower values (0.0-0.5): more deterministic output").
			WithSuggestion("Higher values (0.8-1.2): more creative/varied output").
			WithSuggestion("Example: temperature: 0.7  # balanced default")
	}

	// Check for top_p out of range
	if strings.Contains(errStr, "top_p") {
		topP := float64(0)
		if agentCfg.TopP != nil {
			topP = *agentCfg.TopP
		}
		return werrors.Agent(werrors.ErrAgentInvalidConfig, fmt.Sprintf("invalid top_p %.2f for agent '%s'", topP, name)).
			WithContext("agent", name).
			WithContext("invalid_field", "top_p").
			WithContext("invalid_value", fmt.Sprintf("%.2f", topP)).
			WithContext("valid_range", "0.0 - 1.0").
			WithSuggestion("top_p must be between 0.0 and 1.0").
			WithSuggestion("Example: top_p: 0.9  # common default for balanced sampling")
	}

	// Check for missing required fields
	if strings.Contains(errStr, "required") || strings.Contains(errStr, "missing") {
		return werrors.Agent(werrors.ErrAgentInvalidConfig, fmt.Sprintf("missing required field for agent '%s'", name)).
			WithContext("agent", name).
			WithSuggestion("Required fields for each agent: name, role, backend").
			WithSuggestion("For Loom backend, also specify 'model'").
			WithSuggestion("Example minimal configuration:").
			WithSuggestion("  agents:").
			WithSuggestion("    myagent:").
			WithSuggestion("      role: junior").
			WithSuggestion("      backend: loom").
			WithSuggestion("      model: Qwen/Qwen2.5-Coder-7B-Instruct").
			WithSuggestion("      active: true")
	}

	// Generic agent creation error with helpful context
	return werrors.AgentWrap(err, werrors.ErrAgentCreationFailed, fmt.Sprintf("failed to create agent '%s'", name)).
		WithContext("agent", name).
		WithContext("role", agentCfg.Role).
		WithContext("backend", agentCfg.Backend).
		WithContext("model", agentCfg.Model).
		WithSuggestion("Check your config.yaml for agent configuration errors").
		WithSuggestion(fmt.Sprintf("Valid roles: %s", strings.Join(validRoles, ", "))).
		WithSuggestion(fmt.Sprintf("Valid backends: %s", strings.Join(validBackends, ", "))).
		WithSuggestion("Ensure the specified backend is enabled in the backends section").
		WithSuggestion("Run 'weaver --init' to see an example configuration")
}
