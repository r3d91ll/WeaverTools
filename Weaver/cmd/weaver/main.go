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
	"syscall"

	"github.com/r3d91ll/weaver/pkg/backend"
	"github.com/r3d91ll/weaver/pkg/config"
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
			fmt.Printf("Failed to initialize config: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Config initialized at: %s\n", cfgPath)
		fmt.Println("Edit this file to configure agents and backends.")
		os.Exit(0)
	}

	// Load config
	cfg, err := config.LoadOrDefault(cfgPath)
	if err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
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
		fmt.Println("⚠ No backends available!")
		fmt.Println("  • Ensure 'claude' CLI is installed for Claude Code")
		fmt.Println("  • Ensure The Loom is running at", cfg.Backends.Loom.URL)
		os.Exit(1)
	}

	// Create agent manager
	agentMgr := runtime.NewManager(registry)

	// Create agents from config (only active agents)
	fmt.Println("Agents:")
	for name, agentCfg := range cfg.Agents {
		// Skip inactive agents
		if !agentCfg.Active {
			fmt.Printf("  - %-10s (inactive)\n", name)
			continue
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
			Temperature:   agentCfg.Temperature,
			ContextLength: agentCfg.ContextLength,
			TopP:          agentCfg.TopP,
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

	// Determine default agent
	defaultAgent := "senior"
	if _, ok := cfg.Agents["senior"]; !ok {
		// Use first available agent
		for name := range cfg.Agents {
			defaultAgent = name
			break
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
