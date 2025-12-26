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
			fmt.Printf("  ✗ %-10s - failed: %v\n", name, err)
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
