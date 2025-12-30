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
	"time"

	"github.com/r3d91ll/weaver/pkg/api"
	"github.com/r3d91ll/weaver/pkg/backend"
	"github.com/r3d91ll/weaver/pkg/config"
	werrors "github.com/r3d91ll/weaver/pkg/errors"
	"github.com/r3d91ll/weaver/pkg/loom"
	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/weaver/pkg/shell"
	"github.com/r3d91ll/wool"
	"github.com/r3d91ll/yarn"
)

const version = "1.0.0"

func main() {
	// Parse flags
	configPath := flag.String("config", "", "Config file path (default: ~/.config/weaver/config.yaml)")
	sessionName := flag.String("session", "default", "Session name")
	initConfig := flag.Bool("init", false, "Initialize default config file")
	showVersion := flag.Bool("version", false, "Show version and exit")
	serveMode := flag.Bool("serve", false, "Start HTTP/WebSocket server for web UI")
	servePort := flag.Int("port", 8081, "Port for HTTP server (only used with --serve)")
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

	// Initialize TheLoom manager (TheLoom runs as a systemd service)
	var loomMgr *loom.Manager
	if cfg.Backends.Loom.Enabled {
		loomMgr = loom.NewManager(loom.Config{
			URL:  cfg.Backends.Loom.URL,
			Port: cfg.Backends.Loom.Port,
		})

		// Check if TheLoom is running
		if err := loomMgr.EnsureRunning(ctx); err != nil {
			fmt.Printf("⚠ TheLoom not available: %v\n", err)
			// Continue - backend will show as unavailable
		}

		loomBackend := backend.NewLoom(backend.LoomConfig{
			URL: loomMgr.URL(),
		})
		registry.Register("loom", loomBackend)
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

	// Branch based on mode: serve (HTTP/WS) or interactive shell
	if *serveMode {
		// Start HTTP/WebSocket server for web UI
		serverConfig := api.DefaultServerConfig()
		serverConfig.Port = *servePort

		server := api.NewServer(serverConfig)
		router := server.Router()

		// Create WebSocket hub for real-time updates
		hub := api.NewHub()
		go hub.Run()

		// Register API handlers
		configHandler := api.NewConfigHandler(cfgPath)
		configHandler.RegisterRoutes(router)

		sessionStore := api.NewMemorySessionStore()
		sessionsHandler := api.NewSessionsHandler(sessionStore)
		sessionsHandler.RegisterRoutes(router)

		backendsHandler := api.NewBackendsHandlerWithRegistry(registry)
		backendsHandler.RegisterRoutes(router)

		agentsHandler := api.NewAgentsHandlerWithRuntime(agentMgr)
		agentsHandler.RegisterRoutes(router)

		exportHandler := api.NewExportHandler(sessionStore)
		exportHandler.RegisterRoutes(router)

		resourcesHandler := api.NewResourcesHandler()
		resourcesHandler.RegisterRoutes(router)

		// Register WebSocket handler
		wsHandler := api.NewWebSocketHandler(hub)
		router.GET("/ws", wsHandler.HandleFunc())

		if err := server.Start(); err != nil {
			fmt.Printf("Failed to start HTTP server: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("Weaver HTTP/WebSocket server running on http://%s\n", server.Address())
		fmt.Println("Press Ctrl+C to stop")

		// Wait for shutdown signal
		<-ctx.Done()

		// Graceful shutdown with timeout
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			fmt.Printf("Error during server shutdown: %v\n", err)
		}
	} else {
		// Interactive shell mode
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
			werrors.Display(createShellInitError(historyFile, err))
			os.Exit(1)
		}

		if err := sh.Run(ctx); err != nil && err != context.Canceled {
			werrors.Display(createShellRunError(err))
			os.Exit(1)
		}
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

	// Note: TheLoom is managed by systemd, not by this process.
	// To stop it: systemctl stop the-loom

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

// createShellInitError creates a structured error for shell initialization failures.
// It analyzes the underlying error to provide specific guidance on readline setup,
// history file permissions, and terminal configuration.
func createShellInitError(historyFile string, err error) *werrors.WeaverError {
	errStr := err.Error()

	// Check for history file permission issues
	if strings.Contains(errStr, "permission") ||
		strings.Contains(errStr, "Permission denied") ||
		os.IsPermission(err) {
		return werrors.Command(werrors.ErrShellHistoryFailed, "cannot access shell history file").
			WithCause(err).
			WithContext("history_file", historyFile).
			WithContext("error_type", "permission denied").
			WithSuggestion("Check permissions on history file: ls -la " + historyFile).
			WithSuggestion("Fix permissions with: chmod 600 " + historyFile).
			WithSuggestion("Or remove and let Weaver recreate it: rm " + historyFile)
	}

	// Check for history file directory not found
	historyDir := filepath.Dir(historyFile)
	if strings.Contains(errStr, "no such file or directory") ||
		strings.Contains(errStr, "directory") {
		return werrors.Command(werrors.ErrShellHistoryFailed, "history file directory does not exist").
			WithCause(err).
			WithContext("history_file", historyFile).
			WithContext("directory", historyDir).
			WithSuggestion("Create the directory: mkdir -p " + historyDir).
			WithSuggestion("Ensure your home directory is properly configured")
	}

	// Check for disk full or write errors related to history
	if strings.Contains(errStr, "no space") ||
		strings.Contains(errStr, "disk full") ||
		strings.Contains(errStr, "quota") {
		return werrors.Command(werrors.ErrShellHistoryFailed, "cannot write to history file - disk full").
			WithCause(err).
			WithContext("history_file", historyFile).
			WithSuggestion("Free up disk space on your system").
			WithSuggestion("Clear the history file: rm " + historyFile)
	}

	// Check for readline library issues
	if strings.Contains(errStr, "readline") ||
		strings.Contains(errStr, "terminal") ||
		strings.Contains(errStr, "tty") {
		return werrors.Command(werrors.ErrShellReadlineFailed, "failed to initialize readline/terminal").
			WithCause(err).
			WithContext("history_file", historyFile).
			WithContext("terminal", os.Getenv("TERM")).
			WithSuggestion("Ensure TERM environment variable is set correctly").
			WithSuggestion("Try setting: export TERM=xterm-256color").
			WithSuggestion("If running in a non-interactive context, Weaver requires a TTY")
	}

	// Check for invalid terminal or not a TTY
	if strings.Contains(errStr, "not a terminal") ||
		strings.Contains(errStr, "inappropriate ioctl") ||
		strings.Contains(errStr, "bad file descriptor") {
		return werrors.Command(werrors.ErrShellReadlineFailed, "not connected to a valid terminal").
			WithCause(err).
			WithContext("history_file", historyFile).
			WithSuggestion("Weaver requires an interactive terminal (TTY) to run").
			WithSuggestion("If running in a script, consider using pipes or the API directly").
			WithSuggestion("If running in Docker, use: docker run -it ...")
	}

	// Check for Ctrl-C or signal interruption during init
	if strings.Contains(errStr, "interrupt") ||
		strings.Contains(errStr, "signal") {
		return werrors.Command(werrors.ErrShellInitFailed, "shell initialization interrupted").
			WithCause(err).
			WithContext("history_file", historyFile).
			WithSuggestion("Try starting Weaver again").
			WithSuggestion("Allow initialization to complete before pressing Ctrl-C")
	}

	// Generic shell initialization error
	return werrors.CommandWrap(err, werrors.ErrShellInitFailed, "failed to initialize interactive shell").
		WithContext("history_file", historyFile).
		WithSuggestion("Check that your terminal supports interactive input").
		WithSuggestion("Verify the history file path is writable: touch " + historyFile).
		WithSuggestion("Ensure readline library is properly installed on your system")
}

// createShellRunError creates a structured error for shell runtime failures.
// It provides context about what went wrong during shell execution.
func createShellRunError(err error) *werrors.WeaverError {
	errStr := err.Error()

	// Check for EOF/input stream closed
	if strings.Contains(errStr, "EOF") ||
		strings.Contains(errStr, "closed pipe") ||
		strings.Contains(errStr, "broken pipe") {
		return werrors.Command(werrors.ErrShellReadlineFailed, "input stream closed unexpectedly").
			WithCause(err).
			WithSuggestion("The input stream was closed. This can happen when:").
			WithSuggestion("  - Running Weaver in a non-interactive script").
			WithSuggestion("  - The terminal connection was lost").
			WithSuggestion("  - Input was piped and reached end of file").
			WithSuggestion("For non-interactive use, consider using the API directly")
	}

	// Check for interrupt/signal during execution
	if strings.Contains(errStr, "interrupt") ||
		strings.Contains(errStr, "signal") {
		return werrors.Command(werrors.ErrShellInitFailed, "shell execution interrupted").
			WithCause(err).
			WithSuggestion("The shell was interrupted by a signal").
			WithSuggestion("Use /quit or /exit to gracefully exit Weaver")
	}

	// Check for readline-specific errors during execution
	if strings.Contains(errStr, "readline") {
		return werrors.Command(werrors.ErrShellReadlineFailed, "readline error during shell execution").
			WithCause(err).
			WithSuggestion("Try restarting Weaver").
			WithSuggestion("If the problem persists, check your terminal configuration").
			WithSuggestion("Ensure TERM environment variable is set correctly")
	}

	// Check for I/O errors
	if strings.Contains(errStr, "input/output error") ||
		strings.Contains(errStr, "I/O error") {
		return werrors.Command(werrors.ErrShellReadlineFailed, "terminal I/O error").
			WithCause(err).
			WithSuggestion("There was an error reading from or writing to the terminal").
			WithSuggestion("This can happen if the terminal connection was interrupted").
			WithSuggestion("Try reconnecting to your terminal session and restart Weaver")
	}

	// Generic shell run error
	return werrors.CommandWrap(err, werrors.ErrShellInitFailed, "shell encountered an error").
		WithSuggestion("Try restarting Weaver").
		WithSuggestion("Check system logs for any relevant error messages").
		WithSuggestion("If the problem persists, please report the issue with the error details above")
}
