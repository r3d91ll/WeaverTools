// Package shell provides the interactive REPL for Weaver.
package shell

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/chzyer/readline"
	"github.com/r3d91ll/weaver/pkg/analysis"
	"github.com/r3d91ll/weaver/pkg/concepts"
	werrors "github.com/r3d91ll/weaver/pkg/errors"
	"github.com/r3d91ll/weaver/pkg/export"
	"github.com/r3d91ll/weaver/pkg/help"
	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/yarn"
)

// Shell is the interactive command-line interface.
type Shell struct {
	agents         *runtime.Manager
	session        *yarn.Session
	conv           *yarn.Conversation
	rl             *readline.Instance
	defaultAgent   string // Default agent to route messages to
	conceptStore   *concepts.Store
	analysisClient *analysis.Client
	Prompter       Prompter // Prompter for confirmation prompts (exported for testing)
}

// Config holds shell configuration.
type Config struct {
	HistoryFile  string
	DefaultAgent string
	LoomURL      string // URL for TheLoom analysis endpoints
}

// New creates a new interactive shell.
func New(agents *runtime.Manager, session *yarn.Session, cfg Config) (*Shell, error) {
	// Build prompt with agent indicator
	prompt := func() []byte {
		return []byte("\033[32mweaver>\033[0m ")
	}

	rl, err := readline.NewEx(&readline.Config{
		Prompt:          string(prompt()),
		HistoryFile:     cfg.HistoryFile,
		InterruptPrompt: "^C",
		EOFPrompt:       "exit",
	})
	if err != nil {
		return nil, err
	}

	defaultAgent := cfg.DefaultAgent
	if defaultAgent == "" {
		defaultAgent = "senior"
	}

	return &Shell{
		agents:         agents,
		session:        session,
		conv:           session.ActiveConversation(),
		rl:             rl,
		defaultAgent:   defaultAgent,
		conceptStore:   concepts.NewStore(),
		analysisClient: analysis.NewClient(cfg.LoomURL), // NewClient defaults to localhost:8080
		Prompter:       NewInteractivePrompter(),        // Default to interactive prompts
	}, nil
}

// Run starts the interactive loop.
func (s *Shell) Run(ctx context.Context) error {
	defer s.rl.Close()

	// Startup hint - concise message pointing to /help for details
	fmt.Println(help.Dim("Type a message to chat. ") +
		help.StyleCommand("/help") + help.Dim(" for commands, ") +
		help.StyleExample("@agent") + help.Dim(" to target agents."))
	fmt.Println()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := s.rl.Readline()
		if err != nil {
			if err == readline.ErrInterrupt {
				continue
			}
			if err == io.EOF {
				return nil
			}
			return err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Handle commands
		if strings.HasPrefix(line, "/") {
			if err := s.handleCommand(ctx, line); err != nil {
				if err == errQuit {
					return nil
				}
				werrors.Display(err)
			}
			continue
		}

		// Handle message
		if err := s.handleMessage(ctx, line); err != nil {
			werrors.Display(err)
		}
	}
}

var errQuit = fmt.Errorf("quit")

func (s *Shell) handleCommand(ctx context.Context, line string) error {
	parts := strings.Fields(line)
	cmd := parts[0]

	switch cmd {
	case "/quit", "/exit", "/q":
		return errQuit

	case "/help", "/h":
		if len(parts) > 1 {
			// Show detailed help for a specific command
			cmdName := parts[1]
			renderer := help.NewRenderer(os.Stdout)
			renderer.RenderCommand(cmdName)
		} else {
			s.printHelp()
		}

	case "/agents":
		s.printAgents(ctx)

	case "/session":
		s.printSession()

	case "/history":
		s.printHistory()

	case "/clear":
		s.conv = yarn.NewConversation(s.session.Name + "-conv")
		s.session.AddConversation(s.conv)
		fmt.Println("Conversation cleared.")

	case "/default":
		if len(parts) > 1 {
			s.defaultAgent = parts[1]
			fmt.Printf("Default agent set to: %s\n", s.defaultAgent)
		} else {
			fmt.Printf("Default agent: %s\n", s.defaultAgent)
		}

	// Concept extraction and analysis commands
	case "/extract":
		return s.handleExtract(ctx, parts[1:])

	case "/analyze":
		return s.handleAnalyze(ctx, parts[1:])

	case "/compare":
		return s.handleCompare(ctx, parts[1:])

	case "/validate":
		return s.handleValidate(ctx, parts[1:])

	case "/concepts":
		s.printConcepts()

	case "/metrics":
		return s.handleMetrics(ctx, parts[1:])

	case "/clear_concepts":
		count := s.conceptStore.ClearAll()
		fmt.Printf("Cleared %d concepts.\n", count)

	// Academic export commands
	case "/export_latex":
		return s.handleExportLaTeX(ctx, parts[1:])

	case "/export_csv":
		return s.handleExportCSV(ctx, parts[1:])

	case "/export_figures":
		return s.handleExportFigures(ctx, parts[1:])

	case "/export_bibtex":
		return s.handleExportBibTeX(ctx, parts[1:])

	case "/export_repro":
		return s.handleExportRepro(ctx, parts[1:])

	case "/export_all":
		return s.handleExportAll(ctx, parts[1:])

	default:
		fmt.Printf("Unknown command: %s\n", cmd)
	}

	return nil
}

func (s *Shell) handleMessage(ctx context.Context, line string) error {
	// Parse @agent prefix
	targetAgent := s.defaultAgent
	message := line

	if strings.HasPrefix(line, "@") {
		parts := strings.SplitN(line, " ", 2)
		targetAgent = strings.TrimPrefix(parts[0], "@")
		if len(parts) > 1 {
			message = parts[1]
		} else {
			return createMissingMessageError(targetAgent)
		}
	}

	// Get agent
	agent, ok := s.agents.Get(targetAgent)
	if !ok {
		return createAgentNotFoundError(targetAgent, s.agents.List())
	}

	// Add user message to conversation
	userMsg := yarn.NewAgentMessage(yarn.RoleUser, message, "user", "user")
	s.conv.Add(userMsg)

	// Show thinking indicator
	fmt.Printf("\033[33m[%s]\033[0m thinking...\n", agent.Name())

	// Get response
	resp, err := agent.Chat(ctx, s.conv.History(-1))
	if err != nil {
		return createChatError(agent.Name(), agent.BackendName(), err)
	}

	// Add response to conversation
	s.conv.Add(resp)

	// Display response
	fmt.Printf("\033[36m[%s]\033[0m %s\n", agent.Name(), resp.Content)

	// Show hidden state indicator if present
	if resp.HasHiddenState() {
		dim := resp.HiddenState.Dimension()
		fmt.Printf("\033[90m  └─ hidden state: %d dimensions\033[0m\n", dim)
	}

	fmt.Println()
	return nil
}

func (s *Shell) printHelp() {
	renderer := help.NewRenderer(os.Stdout)
	renderer.RenderFull()
}

func (s *Shell) printAgents(ctx context.Context) {
	fmt.Println("Agents:")
	for name, status := range s.agents.Status(ctx) {
		ready := "✗"
		if status.Ready {
			ready = "✓"
		}
		hidden := ""
		if status.HiddenStates {
			hidden = " [hidden states]"
		}
		defaultMark := ""
		if name == s.defaultAgent {
			defaultMark = " (default)"
		}
		fmt.Printf("  %s %-10s (%s, %s)%s%s\n", ready, name, status.Role, status.Backend, hidden, defaultMark)
	}
}

func (s *Shell) printSession() {
	stats := s.session.Stats()
	fmt.Printf("Session: %s\n", s.session.Name)
	fmt.Printf("  ID: %s\n", s.session.ID[:8])
	fmt.Printf("  Conversations: %d\n", stats.ConversationCount)
	fmt.Printf("  Messages: %d\n", stats.MessageCount)
	fmt.Printf("  Measurements: %d\n", stats.MeasurementCount)
}

func (s *Shell) printHistory() {
	messages := s.conv.History(10)
	if len(messages) == 0 {
		fmt.Println("No messages yet.")
		return
	}

	fmt.Printf("Last %d messages:\n", len(messages))
	for _, msg := range messages {
		role := string(msg.Role)
		if msg.AgentName != "" {
			role = msg.AgentName
		}
		content := msg.Content
		if len(content) > 80 {
			content = content[:80] + "..."
		}
		fmt.Printf("  [%s] %s\n", role, content)
	}
}

// handleExtract handles /extract <concept> <n> command.
func (s *Shell) handleExtract(ctx context.Context, args []string) error {
	if len(args) < 1 {
		fmt.Println("Usage: /extract <concept> [count]")
		fmt.Println("  Examples:")
		fmt.Println("    /extract honor 20")
		fmt.Println("    /extract love 15")
		fmt.Println("    /extract random 20  (baseline)")
		return nil
	}

	concept := args[0]
	count := 10 // default
	if len(args) > 1 {
		n, err := strconv.Atoi(args[1])
		if err != nil {
			return createInvalidCountError(args[1], concept)
		}
		if n <= 0 {
			return createCountOutOfRangeError(args[1], concept, "must be positive (1-100)")
		}
		if n > 100 {
			return createCountOutOfRangeError(args[1], concept, "exceeds maximum of 100")
		}
		count = n
	}

	// Find an agent with hidden state support
	extractAgent, err := s.findHiddenStateAgent(ctx)
	if err != nil {
		return createNoHiddenStateAgentError(s.agents.List())
	}

	fmt.Printf("\033[33mExtracting %d samples for '%s' using %s...\033[0m\n", count, concept, extractAgent.Name())

	// Create extractor and run
	extractor := concepts.NewExtractor(extractAgent.Backend, s.conceptStore)
	cfg := concepts.DefaultExtractionConfig(concept, count)

	result, err := extractor.Extract(ctx, cfg)
	if err != nil {
		return createExtractionError(concept, count, extractAgent.Name(), err)
	}

	// Display results
	fmt.Printf("\033[32m✓ Extracted %d samples\033[0m\n", result.SamplesAdded)
	fmt.Printf("  Concept: %s\n", result.Concept)
	fmt.Printf("  Total samples: %d\n", result.TotalSamples)
	fmt.Printf("  Dimension: %d\n", result.Dimension)
	fmt.Printf("  Time: %.1fms\n", result.DurationMs)
	if len(result.Errors) > 0 {
		fmt.Printf("  \033[31mErrors: %d\033[0m\n", len(result.Errors))
		for _, e := range result.Errors {
			fmt.Printf("    - %s\n", e)
		}
	}
	fmt.Println()

	return nil
}

// handleAnalyze handles /analyze <concept> command.
func (s *Shell) handleAnalyze(ctx context.Context, args []string) error {
	if len(args) < 1 {
		fmt.Println("Usage: /analyze <concept>")
		fmt.Println("  Runs Kakeya geometry analysis on stored concept vectors.")
		return nil
	}

	conceptName := args[0]
	concept, ok := s.conceptStore.Get(conceptName)
	if !ok {
		return createConceptNotFoundError(conceptName, "/analyze", s.conceptStore.List())
	}

	vectors := concept.VectorsAsFloat64()
	if len(vectors) < 3 {
		return createInsufficientSamplesError(conceptName, "/analyze", len(vectors), 3)
	}

	fmt.Printf("\033[33mAnalyzing '%s' (%d vectors, %d dimensions)...\033[0m\n",
		conceptName, len(vectors), concept.Dimension())

	result, err := s.analysisClient.AnalyzeGeometry(ctx, vectors)
	if err != nil {
		return createAnalysisError(conceptName, "/analyze", len(vectors), err)
	}

	// Display results
	fmt.Printf("\n\033[36m=== Kakeya Geometry Analysis: %s ===\033[0m\n", conceptName)
	fmt.Printf("Overall Health: %s\n", formatHealth(result.OverallHealth))
	fmt.Printf("Vectors: %d, Dimension: %d\n\n", result.NumVectors, result.AmbientDim)

	fmt.Println("Wolf Axiom (density concentration):")
	fmt.Printf("  Max Density Ratio: %.2f\n", result.WolfAxiom.MaxDensityRatio)
	fmt.Printf("  Mean Density Ratio: %.2f\n", result.WolfAxiom.MeanDensityRatio)
	fmt.Printf("  Uniformity p-value: %.4f\n", result.WolfAxiom.UniformityPValue)
	fmt.Printf("  Severity: %s\n\n", result.WolfAxiom.Severity)

	fmt.Println("Directional Coverage:")
	fmt.Printf("  Effective Dim: %d / %d (%.1f%%)\n",
		result.DirectionalCoverage.EffectiveDim,
		result.DirectionalCoverage.AmbientDim,
		result.DirectionalCoverage.CoverageRatio*100)
	fmt.Printf("  Coverage Quality: %s\n", result.DirectionalCoverage.CoverageQuality)
	fmt.Printf("  Spherical Uniformity: %.3f\n", result.DirectionalCoverage.SphericalUniformity)
	fmt.Printf("  Isotropy Score: %.3f\n\n", result.DirectionalCoverage.IsotropyScore)

	fmt.Println("Grain Analysis (clustering):")
	fmt.Printf("  Num Grains: %d\n", result.GrainAnalysis.NumGrains)
	fmt.Printf("  Grain Coverage: %.1f%%\n", result.GrainAnalysis.GrainCoverage*100)
	fmt.Printf("  Mean Grain Size: %.1f\n", result.GrainAnalysis.MeanGrainSize)
	fmt.Printf("  Mean Aspect Ratio: %.2f\n\n", result.GrainAnalysis.MeanAspectRatio)

	fmt.Printf("Analysis time: %.1fms\n\n", result.AnalysisTimeMs)

	return nil
}

// handleCompare handles /compare <concept1> <concept2> command.
func (s *Shell) handleCompare(ctx context.Context, args []string) error {
	if len(args) < 2 {
		fmt.Println("Usage: /compare <concept1> <concept2>")
		fmt.Println("  Compares geometric properties between two concepts.")
		return nil
	}

	name1, name2 := args[0], args[1]

	concept1, ok := s.conceptStore.Get(name1)
	if !ok {
		return createConceptNotFoundError(name1, "/compare", s.conceptStore.List())
	}
	concept2, ok := s.conceptStore.Get(name2)
	if !ok {
		return createConceptNotFoundError(name2, "/compare", s.conceptStore.List())
	}

	vectors1 := concept1.VectorsAsFloat64()
	vectors2 := concept2.VectorsAsFloat64()

	if len(vectors1) < 3 {
		return createInsufficientSamplesError(name1, "/compare", len(vectors1), 3)
	}
	if len(vectors2) < 3 {
		return createInsufficientSamplesError(name2, "/compare", len(vectors2), 3)
	}

	fmt.Printf("\033[33mComparing '%s' (%d) vs '%s' (%d)...\033[0m\n",
		name1, len(vectors1), name2, len(vectors2))

	result, err := s.analysisClient.CompareBilateral(ctx, vectors1, vectors2)
	if err != nil {
		return createComparisonError(name1, name2, len(vectors1), len(vectors2), err)
	}

	// Display results
	fmt.Printf("\n\033[36m=== Bilateral Comparison: %s ↔ %s ===\033[0m\n", name1, name2)
	fmt.Printf("Directional Alignment: %.3f\n", result.DirectionalAlignment)
	fmt.Printf("Subspace Overlap:      %.3f\n", result.SubspaceOverlap)
	fmt.Printf("Grain Alignment:       %.3f\n", result.GrainAlignment)
	fmt.Printf("Density Similarity:    %.3f\n", result.DensitySimilarity)
	fmt.Printf("Effective Dim Ratio:   %.3f\n", result.EffectiveDimRatio)
	fmt.Printf("\n\033[1mOverall Alignment:     %.3f\033[0m\n", result.OverallAlignment)
	fmt.Printf("\nAnalysis time: %.1fms\n\n", result.AnalysisTimeMs)

	return nil
}

// handleValidate handles /validate <concept> <n> command.
func (s *Shell) handleValidate(ctx context.Context, args []string) error {
	if len(args) < 1 {
		fmt.Println("Usage: /validate <concept> [iterations]")
		fmt.Println("  Extracts concept multiple times and checks consistency.")
		return nil
	}

	concept := args[0]
	iterations := 3 // default
	if len(args) > 1 {
		n, err := strconv.Atoi(args[1])
		if err != nil {
			return createInvalidIterationsError(args[1], concept)
		}
		if n <= 0 {
			return createIterationsOutOfRangeError(args[1], concept, "must be positive (1-20)")
		}
		if n > 20 {
			return createIterationsOutOfRangeError(args[1], concept, "exceeds maximum of 20")
		}
		iterations = n
	}

	// Find an agent with hidden state support
	extractAgent, err := s.findHiddenStateAgent(ctx)
	if err != nil {
		return createNoHiddenStateAgentError(s.agents.List())
	}

	fmt.Printf("\033[33mValidating '%s' with %d iterations...\033[0m\n\n", concept, iterations)

	// Store results for each iteration
	var results []*analysis.GeometryResult

	for i := 0; i < iterations; i++ {
		// Create a temporary store for this iteration
		tempStore := concepts.NewStore()
		extractor := concepts.NewExtractor(extractAgent.Backend, tempStore)
		cfg := concepts.DefaultExtractionConfig(concept, 10) // 10 samples per iteration

		fmt.Printf("Iteration %d: extracting...", i+1)
		_, err := extractor.Extract(ctx, cfg)
		if err != nil {
			fmt.Printf(" \033[31mfailed: %v\033[0m\n", err)
			continue
		}

		tempConcept, ok := tempStore.Get(concept)
		if !ok {
			fmt.Printf(" \033[31mfailed: concept not found after extraction\033[0m\n")
			continue
		}
		vectors := tempConcept.VectorsAsFloat64()

		fmt.Printf(" analyzing...")
		result, err := s.analysisClient.AnalyzeGeometry(ctx, vectors)
		if err != nil {
			fmt.Printf(" \033[31mfailed: %v\033[0m\n", err)
			continue
		}

		results = append(results, result)
		fmt.Printf(" \033[32mdone\033[0m (D_eff=%d, coverage=%.2f)\n",
			result.DirectionalCoverage.EffectiveDim,
			result.DirectionalCoverage.CoverageRatio)
	}

	if len(results) < 2 {
		return createInsufficientIterationsError(concept, len(results), iterations)
	}

	// Calculate consistency metrics
	fmt.Printf("\n\033[36m=== Consistency Report ===\033[0m\n")

	var sumDeff, sumCoverage, sumDensity float64
	var minDeff, maxDeff int = math.MaxInt, 0
	var minCoverage, maxCoverage float64 = 1.0, 0.0

	for _, r := range results {
		deff := r.DirectionalCoverage.EffectiveDim
		coverage := r.DirectionalCoverage.CoverageRatio

		sumDeff += float64(deff)
		sumCoverage += coverage
		sumDensity += r.WolfAxiom.MaxDensityRatio

		if deff < minDeff {
			minDeff = deff
		}
		if deff > maxDeff {
			maxDeff = deff
		}
		if coverage < minCoverage {
			minCoverage = coverage
		}
		if coverage > maxCoverage {
			maxCoverage = coverage
		}
	}

	n := float64(len(results))
	fmt.Printf("Effective Dimension: avg=%.1f, range=[%d, %d]\n", sumDeff/n, minDeff, maxDeff)
	fmt.Printf("Coverage Ratio:      avg=%.3f, range=[%.3f, %.3f]\n", sumCoverage/n, minCoverage, maxCoverage)
	fmt.Printf("Max Density Ratio:   avg=%.2f\n", sumDensity/n)

	// Consistency score (lower variance = more consistent)
	deffRange := float64(maxDeff - minDeff)
	coverageRange := maxCoverage - minCoverage

	if deffRange <= 5 && coverageRange <= 0.1 {
		fmt.Printf("\n\033[32m✓ High consistency\033[0m - geometric signature is stable\n")
	} else if deffRange <= 10 && coverageRange <= 0.2 {
		fmt.Printf("\n\033[33m~ Moderate consistency\033[0m - some variation in geometry\n")
	} else {
		fmt.Printf("\n\033[31m✗ Low consistency\033[0m - geometric signature unstable\n")
	}
	fmt.Println()

	return nil
}

// handleMetrics handles /metrics <concept> command.
func (s *Shell) handleMetrics(ctx context.Context, args []string) error {
	if len(args) < 1 {
		fmt.Println("Usage: /metrics <concept>")
		fmt.Println("  Shows raw metric values for a concept.")
		return nil
	}

	conceptName := args[0]
	concept, ok := s.conceptStore.Get(conceptName)
	if !ok {
		return createConceptNotFoundError(conceptName, "/metrics", s.conceptStore.List())
	}

	vectors := concept.VectorsAsFloat64()
	if len(vectors) < 3 {
		return createInsufficientSamplesError(conceptName, "/metrics", len(vectors), 3)
	}

	result, err := s.analysisClient.AnalyzeGeometry(ctx, vectors)
	if err != nil {
		return createMetricsAnalysisError(conceptName, len(vectors), err)
	}

	// Raw JSON-like output
	fmt.Printf("concept: %s\n", conceptName)
	fmt.Printf("num_samples: %d\n", len(vectors))
	fmt.Printf("dimension: %d\n", concept.Dimension())
	fmt.Printf("overall_health: %s\n", result.OverallHealth)
	fmt.Printf("wolf.max_density_ratio: %.4f\n", result.WolfAxiom.MaxDensityRatio)
	fmt.Printf("wolf.mean_density_ratio: %.4f\n", result.WolfAxiom.MeanDensityRatio)
	fmt.Printf("wolf.uniformity_p_value: %.6f\n", result.WolfAxiom.UniformityPValue)
	fmt.Printf("wolf.violation_count: %d\n", result.WolfAxiom.ViolationCount)
	fmt.Printf("wolf.severity: %s\n", result.WolfAxiom.Severity)
	fmt.Printf("coverage.effective_dim: %d\n", result.DirectionalCoverage.EffectiveDim)
	fmt.Printf("coverage.ambient_dim: %d\n", result.DirectionalCoverage.AmbientDim)
	fmt.Printf("coverage.ratio: %.6f\n", result.DirectionalCoverage.CoverageRatio)
	fmt.Printf("coverage.quality: %s\n", result.DirectionalCoverage.CoverageQuality)
	fmt.Printf("coverage.spherical_uniformity: %.6f\n", result.DirectionalCoverage.SphericalUniformity)
	fmt.Printf("coverage.isotropy_score: %.6f\n", result.DirectionalCoverage.IsotropyScore)
	fmt.Printf("grains.num_grains: %d\n", result.GrainAnalysis.NumGrains)
	fmt.Printf("grains.coverage: %.6f\n", result.GrainAnalysis.GrainCoverage)
	fmt.Printf("grains.mean_size: %.4f\n", result.GrainAnalysis.MeanGrainSize)
	fmt.Printf("grains.mean_aspect_ratio: %.4f\n", result.GrainAnalysis.MeanAspectRatio)
	fmt.Printf("analysis_time_ms: %.2f\n", result.AnalysisTimeMs)

	return nil
}

// printConcepts displays all stored concepts.
func (s *Shell) printConcepts() {
	concepts := s.conceptStore.List()
	if len(concepts) == 0 {
		fmt.Println("No concepts stored. Use /extract to add some.")
		return
	}

	fmt.Println("Stored concepts:")
	for name, count := range concepts {
		concept, ok := s.conceptStore.Get(name)
		if !ok {
			continue // Skip if somehow missing
		}
		dim := concept.Dimension()
		fmt.Printf("  %-15s %3d samples (%d-dim)\n", name, count, dim)
	}
	fmt.Println()
}

// formatHealth formats the health status with color.
func formatHealth(health string) string {
	switch {
	case strings.HasPrefix(health, "healthy"):
		return "\033[32m" + health + "\033[0m"
	case strings.HasPrefix(health, "warning"):
		return "\033[33m" + health + "\033[0m"
	default:
		return "\033[31m" + health + "\033[0m"
	}
}

// findHiddenStateAgent returns the first agent that supports hidden states,
// selected deterministically by sorted name. Requires agent to have model configured and be ready.
func (s *Shell) findHiddenStateAgent(ctx context.Context) (*runtime.Agent, error) {
	status := s.agents.Status(ctx)
	names := make([]string, 0, len(status))
	for name := range status {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		agent, ok := s.agents.Get(name)
		// Must have a model configured, support hidden states, and be ready
		if ok && agent.ModelName() != "" && agent.SupportsHiddenStates() && agent.IsReady(ctx) {
			return agent, nil
		}
	}
	return nil, fmt.Errorf("no agent with hidden state support available")
}

// -----------------------------------------------------------------------------
// Error Helper Functions
// -----------------------------------------------------------------------------

// createMissingMessageError creates a structured error when no message follows @agent prefix.
func createMissingMessageError(agentName string) *werrors.WeaverError {
	return werrors.Command(werrors.ErrCommandInvalidSyntax, "no message provided after @agent prefix").
		WithContext("agent", agentName).
		WithContext("input", "@"+agentName).
		WithSuggestion("Provide a message after the agent prefix: @" + agentName + " <your message>").
		WithSuggestion("Example: @" + agentName + " Hello, can you help me?").
		WithSuggestion("To list available agents, use the /agents command")
}

// createAgentNotFoundError creates a structured error when the specified agent doesn't exist.
func createAgentNotFoundError(agentName string, availableAgents []string) *werrors.WeaverError {
	err := werrors.AgentNotFound(agentName)

	// Add available agents to context
	if len(availableAgents) > 0 {
		sort.Strings(availableAgents)
		err.WithContext("available_agents", strings.Join(availableAgents, ", "))
		err.WithSuggestion("Available agents: " + strings.Join(availableAgents, ", "))
	}

	err.WithSuggestion("Check the agent name for typos")
	err.WithSuggestion("Use /agents to list all available agents")

	return err
}

// createChatError creates a structured error when chat with an agent fails.
func createChatError(agentName, backendName string, cause error) *werrors.WeaverError {
	errStr := cause.Error()

	// Detect specific error types and provide targeted suggestions
	switch {
	case strings.Contains(errStr, "connection refused") || strings.Contains(errStr, "connect:"):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed: backend connection error").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("Check if the " + backendName + " backend is running").
			WithSuggestion("Verify the backend connection settings in your configuration").
			WithSuggestion("Try using a different agent with /agents to see alternatives")

	case strings.Contains(errStr, "timeout") || strings.Contains(errStr, "deadline exceeded"):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request timed out").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("The backend is taking too long to respond").
			WithSuggestion("Try a shorter message or simpler request").
			WithSuggestion("Check if the backend service is overloaded")

	case strings.Contains(errStr, "unauthorized") || strings.Contains(errStr, "auth") || strings.Contains(errStr, "401"):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed: authentication error").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("Check your API credentials for the " + backendName + " backend").
			WithSuggestion("For Claude: Run 'claude auth login' to re-authenticate").
			WithSuggestion("For Loom: Verify your API token configuration")

	case strings.Contains(errStr, "rate limit") || strings.Contains(errStr, "429"):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed: rate limit exceeded").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("Wait a moment before trying again").
			WithSuggestion("Consider using a different agent or backend")

	case strings.Contains(errStr, "model") && (strings.Contains(errStr, "not found") || strings.Contains(errStr, "invalid")):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed: model configuration error").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("Check the model name in the agent configuration").
			WithSuggestion("Verify the model is available on the backend")

	case strings.Contains(errStr, "context canceled"):
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request was interrupted").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("The request was canceled before completion").
			WithSuggestion("Try the request again")

	default:
		// Generic chat failure with the original error
		return werrors.AgentWrap(cause, werrors.ErrAgentChatFailed, "chat request failed").
			WithContext("agent", agentName).
			WithContext("backend", backendName).
			WithSuggestion("Check the backend connection with /agents").
			WithSuggestion("Try the request again").
			WithSuggestion("If the problem persists, check the backend logs")
	}
}

// -----------------------------------------------------------------------------
// /extract Command Error Helpers
// -----------------------------------------------------------------------------

// createInvalidCountError creates a structured error when the count argument is not a valid number.
func createInvalidCountError(invalidValue, concept string) *werrors.WeaverError {
	return werrors.Command(werrors.ErrCommandInvalidArg, "invalid count value: not a number").
		WithContext("command", "/extract").
		WithContext("argument", "count").
		WithContext("invalid_value", invalidValue).
		WithContext("concept", concept).
		WithContext("valid_range", "1-100").
		WithSuggestion("Count must be a positive integer between 1 and 100").
		WithSuggestion("Example: /extract " + concept + " 20").
		WithSuggestion("Default count is 10 if not specified: /extract " + concept)
}

// createCountOutOfRangeError creates a structured error when the count is out of valid range.
func createCountOutOfRangeError(value, concept, reason string) *werrors.WeaverError {
	return werrors.Validation(werrors.ErrValidationOutOfRange, "count "+reason).
		WithContext("command", "/extract").
		WithContext("argument", "count").
		WithContext("value", value).
		WithContext("concept", concept).
		WithContext("valid_range", "1-100").
		WithSuggestion("Count must be between 1 and 100").
		WithSuggestion("Example: /extract " + concept + " 20").
		WithSuggestion("Recommended: 10-30 samples for quick tests, 50-100 for comprehensive analysis")
}

// createNoHiddenStateAgentError creates a structured error when no agent supports hidden states.
func createNoHiddenStateAgentError(availableAgents []string) *werrors.WeaverError {
	err := werrors.Agent(werrors.ErrAgentNoHiddenState, "no agent with hidden state support available")

	// Add context about available agents
	if len(availableAgents) > 0 {
		sort.Strings(availableAgents)
		err.WithContext("available_agents", strings.Join(availableAgents, ", "))
	}

	err.WithContext("command", "/extract").
		WithContext("required_capability", "hidden_states").
		WithSuggestion("Hidden state extraction requires a backend that returns embedding vectors").
		WithSuggestion("The Loom backend with local models typically supports hidden states").
		WithSuggestion("Use /agents to check which agents support hidden states (look for [hidden states] indicator)").
		WithSuggestion("Configure an agent with a Loom backend that has hidden state support enabled")

	return err
}

// createExtractionError creates a structured error when concept extraction fails.
func createExtractionError(concept string, count int, agentName string, cause error) *werrors.WeaverError {
	errStr := cause.Error()

	// Detect specific error types and provide targeted suggestions
	switch {
	case strings.Contains(errStr, "connection refused") || strings.Contains(errStr, "connect:"):
		return werrors.CommandWrap(cause, werrors.ErrConceptsExtractionFailed, "extraction failed: backend connection error").
			WithContext("command", "/extract").
			WithContext("concept", concept).
			WithContext("count", fmt.Sprintf("%d", count)).
			WithContext("agent", agentName).
			WithSuggestion("Check if the backend service is running").
			WithSuggestion("Verify the backend URL in your configuration").
			WithSuggestion("Use /agents to check agent status")

	case strings.Contains(errStr, "timeout") || strings.Contains(errStr, "deadline exceeded"):
		return werrors.CommandWrap(cause, werrors.ErrConceptsExtractionFailed, "extraction failed: request timed out").
			WithContext("command", "/extract").
			WithContext("concept", concept).
			WithContext("count", fmt.Sprintf("%d", count)).
			WithContext("agent", agentName).
			WithSuggestion("Try a smaller sample count (e.g., /extract " + concept + " 5)").
			WithSuggestion("The backend may be overloaded - try again later").
			WithSuggestion("Check backend logs for performance issues")

	case strings.Contains(errStr, "hidden state") || strings.Contains(errStr, "embedding"):
		return werrors.CommandWrap(cause, werrors.ErrConceptsExtractionFailed, "extraction failed: hidden states not available").
			WithContext("command", "/extract").
			WithContext("concept", concept).
			WithContext("agent", agentName).
			WithSuggestion("The backend may not support hidden state extraction").
			WithSuggestion("Verify the model supports returning embeddings/hidden states").
			WithSuggestion("Try a different agent with /agents")

	case strings.Contains(errStr, "context canceled"):
		return werrors.CommandWrap(cause, werrors.ErrConceptsExtractionFailed, "extraction was interrupted").
			WithContext("command", "/extract").
			WithContext("concept", concept).
			WithContext("count", fmt.Sprintf("%d", count)).
			WithContext("agent", agentName).
			WithSuggestion("The extraction was canceled before completion").
			WithSuggestion("Try running /extract again")

	default:
		// Generic extraction failure
		return werrors.CommandWrap(cause, werrors.ErrConceptsExtractionFailed, "concept extraction failed").
			WithContext("command", "/extract").
			WithContext("concept", concept).
			WithContext("count", fmt.Sprintf("%d", count)).
			WithContext("agent", agentName).
			WithSuggestion("Check the backend connection with /agents").
			WithSuggestion("Try with fewer samples: /extract " + concept + " 5").
			WithSuggestion("If the problem persists, check backend logs for details")
	}
}

// -----------------------------------------------------------------------------
// /analyze and /compare Command Error Helpers
// -----------------------------------------------------------------------------

// createConceptNotFoundError creates a structured error when a concept is not found in the store.
func createConceptNotFoundError(conceptName, command string, storedConcepts map[string]int) *werrors.WeaverError {
	err := werrors.Command(werrors.ErrConceptsNotFound, fmt.Sprintf("concept %q not found", conceptName)).
		WithContext("command", command).
		WithContext("concept", conceptName)

	// Add list of available concepts if any exist
	if len(storedConcepts) > 0 {
		names := make([]string, 0, len(storedConcepts))
		for name := range storedConcepts {
			names = append(names, name)
		}
		sort.Strings(names)
		err.WithContext("available_concepts", strings.Join(names, ", "))
		err.WithSuggestion("Available concepts: " + strings.Join(names, ", "))
	} else {
		err.WithContext("available_concepts", "none")
		err.WithSuggestion("No concepts have been extracted yet")
	}

	err.WithSuggestion("Use /extract to create the concept first: /extract " + conceptName + " 20").
		WithSuggestion("Use /concepts to list all stored concepts")

	return err
}

// createInsufficientSamplesError creates a structured error when a concept has too few samples.
func createInsufficientSamplesError(conceptName, command string, currentCount, requiredCount int) *werrors.WeaverError {
	return werrors.Command(werrors.ErrConceptsInsufficientSamples,
		fmt.Sprintf("%q has insufficient samples for analysis", conceptName)).
		WithContext("command", command).
		WithContext("concept", conceptName).
		WithContext("current_samples", fmt.Sprintf("%d", currentCount)).
		WithContext("required_samples", fmt.Sprintf("%d", requiredCount)).
		WithSuggestion(fmt.Sprintf("Need at least %d samples, but only have %d", requiredCount, currentCount)).
		WithSuggestion(fmt.Sprintf("Extract more samples: /extract %s %d", conceptName, requiredCount+5)).
		WithSuggestion("For reliable analysis, consider extracting 10-20 samples").
		WithSuggestion("Use /concepts to check current sample counts")
}

// createAnalysisError creates a structured error when geometry analysis fails.
func createAnalysisError(conceptName, command string, vectorCount int, cause error) *werrors.WeaverError {
	errStr := cause.Error()

	// Detect specific error types and provide targeted suggestions
	switch {
	case strings.Contains(errStr, "connection refused") || strings.Contains(errStr, "connect:"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisServerUnavailable, "analysis server connection failed").
			WithContext("command", command).
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("Check if TheLoom analysis server is running").
			WithSuggestion("Verify the server URL in your configuration").
			WithSuggestion("Default: http://localhost:8080 - ensure the server is started")

	case strings.Contains(errStr, "timeout") || strings.Contains(errStr, "deadline exceeded"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "analysis request timed out").
			WithContext("command", command).
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("The analysis is taking too long - try with fewer vectors").
			WithSuggestion("Extract a smaller sample: /extract " + conceptName + " 10 and analyze again").
			WithSuggestion("Check if the analysis server is overloaded")

	case strings.Contains(errStr, "EOF") || strings.Contains(errStr, "unexpected end"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisInvalidResponse, "analysis server returned incomplete response").
			WithContext("command", command).
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("The analysis server may have crashed or restarted").
			WithSuggestion("Check the analysis server logs for errors").
			WithSuggestion("Try the request again")

	case strings.Contains(errStr, "invalid") || strings.Contains(errStr, "parse") || strings.Contains(errStr, "unmarshal"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisInvalidResponse, "analysis server returned invalid data").
			WithContext("command", command).
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("The server response could not be parsed").
			WithSuggestion("Check for version mismatch between Weaver and TheLoom").
			WithSuggestion("Verify the analysis server is running the correct version")

	case strings.Contains(errStr, "context canceled"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "analysis was interrupted").
			WithContext("command", command).
			WithContext("concept", conceptName).
			WithSuggestion("The request was canceled before completion").
			WithSuggestion("Try running the command again")

	default:
		// Generic analysis failure
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "geometry analysis failed").
			WithContext("command", command).
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("Check the analysis server connection and logs").
			WithSuggestion("Verify your vectors have the expected dimension").
			WithSuggestion("Try again with /analyze " + conceptName)
	}
}

// createComparisonError creates a structured error when bilateral comparison fails.
func createComparisonError(concept1, concept2 string, count1, count2 int, cause error) *werrors.WeaverError {
	errStr := cause.Error()

	// Detect specific error types and provide targeted suggestions
	switch {
	case strings.Contains(errStr, "connection refused") || strings.Contains(errStr, "connect:"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisServerUnavailable, "analysis server connection failed").
			WithContext("command", "/compare").
			WithContext("concept1", concept1).
			WithContext("concept2", concept2).
			WithContext("vectors1", fmt.Sprintf("%d", count1)).
			WithContext("vectors2", fmt.Sprintf("%d", count2)).
			WithSuggestion("Check if TheLoom analysis server is running").
			WithSuggestion("Verify the server URL in your configuration").
			WithSuggestion("Default: http://localhost:8080 - ensure the server is started")

	case strings.Contains(errStr, "timeout") || strings.Contains(errStr, "deadline exceeded"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "comparison request timed out").
			WithContext("command", "/compare").
			WithContext("concept1", concept1).
			WithContext("concept2", concept2).
			WithContext("vectors1", fmt.Sprintf("%d", count1)).
			WithContext("vectors2", fmt.Sprintf("%d", count2)).
			WithSuggestion("The comparison is taking too long - try with fewer vectors").
			WithSuggestion("Extract smaller samples for both concepts and compare again").
			WithSuggestion("Check if the analysis server is overloaded")

	case strings.Contains(errStr, "dimension") || strings.Contains(errStr, "mismatch"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "comparison failed: vector dimension mismatch").
			WithContext("command", "/compare").
			WithContext("concept1", concept1).
			WithContext("concept2", concept2).
			WithContext("vectors1", fmt.Sprintf("%d", count1)).
			WithContext("vectors2", fmt.Sprintf("%d", count2)).
			WithSuggestion("Both concepts must have vectors of the same dimension").
			WithSuggestion("Re-extract both concepts from the same model/backend").
			WithSuggestion("Use /concepts to check dimensions of stored concepts")

	case strings.Contains(errStr, "EOF") || strings.Contains(errStr, "unexpected end"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisInvalidResponse, "analysis server returned incomplete response").
			WithContext("command", "/compare").
			WithContext("concept1", concept1).
			WithContext("concept2", concept2).
			WithSuggestion("The analysis server may have crashed or restarted").
			WithSuggestion("Check the analysis server logs for errors").
			WithSuggestion("Try the request again")

	case strings.Contains(errStr, "context canceled"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "comparison was interrupted").
			WithContext("command", "/compare").
			WithContext("concept1", concept1).
			WithContext("concept2", concept2).
			WithSuggestion("The request was canceled before completion").
			WithSuggestion("Try running the command again")

	default:
		// Generic comparison failure
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "bilateral comparison failed").
			WithContext("command", "/compare").
			WithContext("concept1", concept1).
			WithContext("concept2", concept2).
			WithContext("vectors1", fmt.Sprintf("%d", count1)).
			WithContext("vectors2", fmt.Sprintf("%d", count2)).
			WithSuggestion("Check the analysis server connection and logs").
			WithSuggestion("Verify both concepts have vectors of the same dimension").
			WithSuggestion("Try again with /compare " + concept1 + " " + concept2)
	}
}

// -----------------------------------------------------------------------------
// /validate Command Error Helpers
// -----------------------------------------------------------------------------

// createInvalidIterationsError creates a structured error when the iterations argument is not a valid number.
func createInvalidIterationsError(invalidValue, concept string) *werrors.WeaverError {
	return werrors.Command(werrors.ErrCommandInvalidArg, "invalid iterations value: not a number").
		WithContext("command", "/validate").
		WithContext("argument", "iterations").
		WithContext("invalid_value", invalidValue).
		WithContext("concept", concept).
		WithContext("valid_range", "1-20").
		WithSuggestion("Iterations must be a positive integer between 1 and 20").
		WithSuggestion("Example: /validate " + concept + " 5").
		WithSuggestion("Default iterations is 3 if not specified: /validate " + concept)
}

// createIterationsOutOfRangeError creates a structured error when iterations is out of valid range.
func createIterationsOutOfRangeError(value, concept, reason string) *werrors.WeaverError {
	return werrors.Validation(werrors.ErrValidationOutOfRange, "iterations "+reason).
		WithContext("command", "/validate").
		WithContext("argument", "iterations").
		WithContext("value", value).
		WithContext("concept", concept).
		WithContext("valid_range", "1-20").
		WithSuggestion("Iterations must be between 1 and 20").
		WithSuggestion("Example: /validate " + concept + " 5").
		WithSuggestion("Recommended: 3-5 iterations for quick validation, 10-20 for comprehensive testing")
}

// createInsufficientIterationsError creates a structured error when fewer than 2 iterations succeeded.
func createInsufficientIterationsError(concept string, successfulIterations, requestedIterations int) *werrors.WeaverError {
	return werrors.Command(werrors.ErrConceptsInsufficientSamples,
		fmt.Sprintf("validation requires at least 2 successful iterations, got %d", successfulIterations)).
		WithContext("command", "/validate").
		WithContext("concept", concept).
		WithContext("successful_iterations", fmt.Sprintf("%d", successfulIterations)).
		WithContext("requested_iterations", fmt.Sprintf("%d", requestedIterations)).
		WithContext("required_iterations", "2").
		WithSuggestion("Need at least 2 successful iterations to calculate consistency metrics").
		WithSuggestion("Check the iteration failure messages above for error details").
		WithSuggestion("Verify the backend is running and stable").
		WithSuggestion("Try running validation again: /validate " + concept)
}

// -----------------------------------------------------------------------------
// /metrics Command Error Helpers
// -----------------------------------------------------------------------------

// createMetricsAnalysisError creates a structured error when the metrics analysis fails.
func createMetricsAnalysisError(conceptName string, vectorCount int, cause error) *werrors.WeaverError {
	errStr := cause.Error()

	// Detect specific error types and provide targeted suggestions
	switch {
	case strings.Contains(errStr, "connection refused") || strings.Contains(errStr, "connect:"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisServerUnavailable, "metrics analysis failed: server connection error").
			WithContext("command", "/metrics").
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("Check if TheLoom analysis server is running").
			WithSuggestion("Verify the server URL in your configuration").
			WithSuggestion("Default: http://localhost:8080 - ensure the server is started")

	case strings.Contains(errStr, "timeout") || strings.Contains(errStr, "deadline exceeded"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "metrics analysis timed out").
			WithContext("command", "/metrics").
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("The analysis is taking too long - try with fewer samples").
			WithSuggestion("Check if the analysis server is overloaded").
			WithSuggestion("Try using /analyze for a more verbose analysis with status updates")

	case strings.Contains(errStr, "EOF") || strings.Contains(errStr, "unexpected end"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisInvalidResponse, "metrics analysis returned incomplete response").
			WithContext("command", "/metrics").
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("The analysis server may have crashed or restarted").
			WithSuggestion("Check the analysis server logs for errors").
			WithSuggestion("Try the request again")

	case strings.Contains(errStr, "invalid") || strings.Contains(errStr, "parse") || strings.Contains(errStr, "unmarshal"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisInvalidResponse, "metrics analysis returned invalid data").
			WithContext("command", "/metrics").
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("The server response could not be parsed").
			WithSuggestion("Check for version mismatch between Weaver and TheLoom").
			WithSuggestion("Verify the analysis server is running the correct version")

	case strings.Contains(errStr, "context canceled"):
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "metrics analysis was interrupted").
			WithContext("command", "/metrics").
			WithContext("concept", conceptName).
			WithSuggestion("The request was canceled before completion").
			WithSuggestion("Try running /metrics " + conceptName + " again")

	default:
		// Generic metrics failure
		return werrors.CommandWrap(cause, werrors.ErrAnalysisFailed, "metrics analysis failed").
			WithContext("command", "/metrics").
			WithContext("concept", conceptName).
			WithContext("vector_count", fmt.Sprintf("%d", vectorCount)).
			WithSuggestion("Check the analysis server connection and logs").
			WithSuggestion("Verify your vectors have the expected dimension").
			WithSuggestion("Try again with /metrics " + conceptName)
	}
}

// -----------------------------------------------------------------------------
// Export Command Handlers
// -----------------------------------------------------------------------------

// toolVersion returns the current tool version for exports.
const toolVersion = "0.1.0"

// handleExportLaTeX handles /export_latex command.
// Exports measurements as a LaTeX table.
func (s *Shell) handleExportLaTeX(_ context.Context, args []string) error {
	measurements := s.session.Measurements
	if len(measurements) == 0 {
		return createExportNoDataError("/export_latex", "latex")
	}

	// Create export directory
	exportDir := s.getExportDir("latex")
	if err := os.MkdirAll(exportDir, 0755); err != nil {
		return createExportDirError("/export_latex", "latex", exportDir, err)
	}

	// Convert measurements to export format
	rows := s.measurementsToRows()

	// Generate measurement table
	config := export.DefaultMeasurementTableConfig()
	config.Caption = fmt.Sprintf("Conveyance Measurements - Session %s", s.session.Name)
	config.Label = "tab:measurements"
	tableContent := export.GenerateMeasurementTable(rows, config)

	// Generate summary table
	stats := export.ComputeSummaryStats(rows)
	summaryConfig := export.DefaultSummaryTableConfig()
	summaryConfig.Caption = "Summary Statistics"
	summaryConfig.Label = "tab:summary"
	summaryContent := export.GenerateSummaryTable(stats, summaryConfig)

	// Write measurement table
	measurementPath := filepath.Join(exportDir, "measurements.tex")
	if err := os.WriteFile(measurementPath, []byte(tableContent), 0644); err != nil {
		return createExportError("/export_latex", "latex", measurementPath, err)
	}

	// Write summary table
	summaryPath := filepath.Join(exportDir, "summary.tex")
	if err := os.WriteFile(summaryPath, []byte(summaryContent), 0644); err != nil {
		return createExportError("/export_latex", "latex", summaryPath, err)
	}

	fmt.Printf("\033[32m✓ Exported LaTeX tables\033[0m\n")
	fmt.Printf("  Measurements: %s\n", measurementPath)
	fmt.Printf("  Summary: %s\n", summaryPath)
	fmt.Printf("  Records: %d\n", len(rows))
	fmt.Println()

	return nil
}

// handleExportCSV handles /export_csv command.
// Exports measurements as CSV with accompanying metadata JSON.
func (s *Shell) handleExportCSV(_ context.Context, args []string) error {
	measurements := s.session.Measurements
	if len(measurements) == 0 {
		return createExportNoDataError("/export_csv", "csv")
	}

	// Create export directory
	exportDir := s.getExportDir("csv")
	if err := os.MkdirAll(exportDir, 0755); err != nil {
		return createExportDirError("/export_csv", "csv", exportDir, err)
	}

	// Convert measurements to CSV format
	csvMeasurements := s.measurementsToCSV()

	// Write CSV file
	csvPath := filepath.Join(exportDir, "measurements.csv")
	csvFile, err := os.Create(csvPath)
	if err != nil {
		return createExportError("/export_csv", "csv", csvPath, err)
	}
	defer csvFile.Close()

	config := export.DefaultCSVConfig()
	if err := export.ExportMeasurementsToCSV(csvFile, csvMeasurements, config); err != nil {
		return createExportError("/export_csv", "csv", csvPath, err)
	}

	// Write metadata JSON
	metadataPath := filepath.Join(exportDir, "metadata.json")
	metadataFile, err := os.Create(metadataPath)
	if err != nil {
		return createExportError("/export_csv", "metadata", metadataPath, err)
	}
	defer metadataFile.Close()

	if err := export.ExportMetadataToJSON(metadataFile, config); err != nil {
		return createExportError("/export_csv", "metadata", metadataPath, err)
	}

	fmt.Printf("\033[32m✓ Exported CSV data\033[0m\n")
	fmt.Printf("  Data: %s\n", csvPath)
	fmt.Printf("  Metadata: %s\n", metadataPath)
	fmt.Printf("  Records: %d\n", len(csvMeasurements))
	fmt.Println()

	return nil
}

// handleExportFigures handles /export_figures command.
// Exports measurement plots as SVG and PDF figures.
func (s *Shell) handleExportFigures(_ context.Context, args []string) error {
	measurements := s.session.Measurements
	if len(measurements) == 0 {
		return createExportNoDataError("/export_figures", "figures")
	}

	// Create export directories
	svgDir := s.getExportDir("svg")
	pdfDir := s.getExportDir("pdf")
	if err := os.MkdirAll(svgDir, 0755); err != nil {
		return createExportDirError("/export_figures", "svg", svgDir, err)
	}
	if err := os.MkdirAll(pdfDir, 0755); err != nil {
		return createExportDirError("/export_figures", "pdf", pdfDir, err)
	}

	// Convert measurements to rows
	rows := s.measurementsToRows()

	metrics := []export.PlotMetric{
		export.MetricDEff,
		export.MetricBeta,
		export.MetricAlignment,
		export.MetricCPair,
	}

	// Export each metric as both SVG and PDF
	for _, metric := range metrics {
		metricName := string(metric)

		// SVG export
		svgConfig := export.DefaultSVGConfig()
		svgConfig.Title = fmt.Sprintf("%s Over Time - %s", export.GetMetricInfo(metric).Name, s.session.Name)
		svgConfig.ToolVersion = toolVersion
		svg := export.GenerateMeasurementPlot(rows, metric, svgConfig)

		svgPath := filepath.Join(svgDir, fmt.Sprintf("%s.svg", metricName))
		if err := os.WriteFile(svgPath, []byte(svg), 0644); err != nil {
			return createExportError("/export_figures", "svg", svgPath, err)
		}

		// PDF export
		pdfConfig := export.DefaultPDFConfig()
		pdfConfig.Title = fmt.Sprintf("%s Over Time - %s", export.GetMetricInfo(metric).Name, s.session.Name)
		pdfConfig.ToolVersion = toolVersion
		pdfConfig.Author = "WeaverTools"
		pdf := export.GenerateMeasurementPDFPlot(rows, metric, pdfConfig)

		pdfPath := filepath.Join(pdfDir, fmt.Sprintf("%s.pdf", metricName))
		if err := os.WriteFile(pdfPath, pdf, 0644); err != nil {
			return createExportError("/export_figures", "pdf", pdfPath, err)
		}
	}

	// Generate multi-metric comparison plot
	svgConfig := export.DefaultSVGConfig()
	svgConfig.Title = fmt.Sprintf("All Metrics Over Time - %s", s.session.Name)
	svgConfig.ToolVersion = toolVersion
	multiSVG := export.GenerateMultiMetricPlot(rows, metrics, svgConfig)

	multiSVGPath := filepath.Join(svgDir, "all_metrics.svg")
	if err := os.WriteFile(multiSVGPath, []byte(multiSVG), 0644); err != nil {
		return createExportError("/export_figures", "svg", multiSVGPath, err)
	}

	pdfConfig := export.DefaultPDFConfig()
	pdfConfig.Title = fmt.Sprintf("All Metrics Over Time - %s", s.session.Name)
	pdfConfig.ToolVersion = toolVersion
	pdfConfig.Author = "WeaverTools"
	multiPDF := export.GenerateMultiMetricPDFPlot(rows, metrics, pdfConfig)

	multiPDFPath := filepath.Join(pdfDir, "all_metrics.pdf")
	if err := os.WriteFile(multiPDFPath, multiPDF, 0644); err != nil {
		return createExportError("/export_figures", "pdf", multiPDFPath, err)
	}

	fmt.Printf("\033[32m✓ Exported figures\033[0m\n")
	fmt.Printf("  SVG: %s\n", svgDir)
	fmt.Printf("  PDF: %s\n", pdfDir)
	fmt.Printf("  Plots: %d metrics + multi-metric comparison\n", len(metrics))
	fmt.Println()

	return nil
}

// handleExportBibTeX handles /export_bibtex command.
// Exports a BibTeX citation for the analysis.
func (s *Shell) handleExportBibTeX(_ context.Context, args []string) error {
	// Create export directory
	exportDir := s.getExportDir("bibtex")
	if err := os.MkdirAll(exportDir, 0755); err != nil {
		return createExportDirError("/export_bibtex", "bibtex", exportDir, err)
	}

	// Compute experiment hash
	stats := s.session.Stats()
	hash := export.ComputeExperimentHash(
		toolVersion,
		string(s.session.Config.MeasurementMode),
		stats.MeasurementCount,
		stats.ConversationCount,
		s.session.StartedAt,
		s.session.EndedAt,
		nil,
	)

	// Generate citation
	entry := export.GenerateWeaverCitation(toolVersion, hash)

	// Write BibTeX file
	bibPath := filepath.Join(exportDir, "weaver.bib")
	entries := []*export.BibTeXEntry{entry}
	bibContent := export.GenerateBibTeXFile(entries, export.DefaultBibTeXConfig())

	if err := os.WriteFile(bibPath, []byte(bibContent), 0644); err != nil {
		return createExportError("/export_bibtex", "bibtex", bibPath, err)
	}

	fmt.Printf("\033[32m✓ Exported BibTeX citation\033[0m\n")
	fmt.Printf("  File: %s\n", bibPath)
	fmt.Printf("  Key: %s\n", entry.Key)
	fmt.Printf("  Hash: %s\n", hash.ShortHash())
	fmt.Println()

	// Show the citation
	fmt.Println("Citation entry:")
	fmt.Println(entry.String())

	return nil
}

// handleExportRepro handles /export_repro command.
// Exports a reproducibility report.
func (s *Shell) handleExportRepro(_ context.Context, args []string) error {
	// Create export directory
	exportDir := s.getExportDir("reproducibility")
	if err := os.MkdirAll(exportDir, 0755); err != nil {
		return createExportDirError("/export_repro", "reproducibility", exportDir, err)
	}

	stats := s.session.Stats()

	// Build reproducibility report
	builder := export.NewReportBuilder().
		WithToolVersion(toolVersion).
		WithSessionID(s.session.ID).
		WithSessionName(s.session.Name).
		WithSessionDescription(s.session.Description).
		WithMeasurementMode(string(s.session.Config.MeasurementMode)).
		WithMeasurementCount(stats.MeasurementCount).
		WithConversationCount(stats.ConversationCount).
		WithTimeRange(s.session.StartedAt, s.session.EndedAt)

	// Add session metadata as parameters
	if s.session.Metadata != nil {
		for k, v := range s.session.Metadata {
			if str, ok := v.(string); ok {
				builder.WithParameter(k, str)
			}
		}
	}

	report := builder.Build()

	// Export in all three formats
	formats := []struct {
		format   export.ReportFormat
		filename string
	}{
		{export.FormatMarkdown, "report.md"},
		{export.FormatLaTeX, "report.tex"},
		{export.FormatJSON, "report.json"},
	}

	for _, f := range formats {
		config := export.DefaultReproducibilityConfig()
		config.Format = f.format

		content := report.Format(config)
		path := filepath.Join(exportDir, f.filename)

		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return createExportError("/export_repro", string(f.format), path, err)
		}
	}

	fmt.Printf("\033[32m✓ Exported reproducibility report\033[0m\n")
	fmt.Printf("  Directory: %s\n", exportDir)
	fmt.Printf("  Formats: Markdown, LaTeX, JSON\n")
	fmt.Printf("  Hash: %s\n", report.ExperimentHash.ShortHash())
	fmt.Println()

	return nil
}

// handleExportAll handles /export_all command.
// Exports all formats at once.
func (s *Shell) handleExportAll(ctx context.Context, args []string) error {
	fmt.Println("\033[33mExporting all formats...\033[0m")
	fmt.Println()

	// Track errors and failed formats
	var errors []error
	var failedFormats []string

	// Export LaTeX
	if err := s.handleExportLaTeX(ctx, args); err != nil {
		errors = append(errors, err)
		failedFormats = append(failedFormats, "LaTeX")
		werrors.Display(err)
	}

	// Export CSV
	if err := s.handleExportCSV(ctx, args); err != nil {
		errors = append(errors, err)
		failedFormats = append(failedFormats, "CSV")
		werrors.Display(err)
	}

	// Export Figures
	if err := s.handleExportFigures(ctx, args); err != nil {
		errors = append(errors, err)
		failedFormats = append(failedFormats, "Figures")
		werrors.Display(err)
	}

	// Export BibTeX
	if err := s.handleExportBibTeX(ctx, args); err != nil {
		errors = append(errors, err)
		failedFormats = append(failedFormats, "BibTeX")
		werrors.Display(err)
	}

	// Export Reproducibility Report
	if err := s.handleExportRepro(ctx, args); err != nil {
		errors = append(errors, err)
		failedFormats = append(failedFormats, "Reproducibility")
		werrors.Display(err)
	}

	if len(errors) > 0 {
		fmt.Printf("\033[33mCompleted with %d error(s)\033[0m\n", len(errors))
		return createExportMultipleError(failedFormats, errors[0])
	}

	fmt.Printf("\033[32m✓ All exports complete\033[0m\n")
	fmt.Printf("  Export directory: %s\n", s.getExportBaseDir())
	fmt.Println()

	return nil
}

// -----------------------------------------------------------------------------
// Export Helper Functions
// -----------------------------------------------------------------------------

// getExportBaseDir returns the base export directory for the session.
func (s *Shell) getExportBaseDir() string {
	exportPath := s.session.Config.ExportPath
	if exportPath == "" {
		exportPath = "./experiments"
	}
	return filepath.Join(exportPath, s.session.ID)
}

// getExportDir returns the export directory for a specific format.
func (s *Shell) getExportDir(format string) string {
	return filepath.Join(s.getExportBaseDir(), format)
}

// measurementsToRows converts session measurements to export.MeasurementRow format.
func (s *Shell) measurementsToRows() []export.MeasurementRow {
	measurements := s.session.Measurements
	rows := make([]export.MeasurementRow, len(measurements))

	for i, m := range measurements {
		rows[i] = export.MeasurementRow{
			Turn:       m.TurnNumber,
			Sender:     m.SenderName,
			Receiver:   m.ReceiverName,
			DEff:       m.DEff,
			Beta:       m.Beta,
			Alignment:  m.Alignment,
			CPair:      m.CPair,
			BetaStatus: string(m.BetaStatus),
		}
	}

	return rows
}

// measurementsToCSV converts session measurements to export.CSVMeasurement format.
func (s *Shell) measurementsToCSV() []*export.CSVMeasurement {
	measurements := s.session.Measurements
	csvMeasurements := make([]*export.CSVMeasurement, len(measurements))

	for i, m := range measurements {
		csvMeasurements[i] = &export.CSVMeasurement{
			ID:             m.ID,
			Timestamp:      m.Timestamp,
			SessionID:      m.SessionID,
			ConversationID: m.ConversationID,
			TurnNumber:     m.TurnNumber,
			SenderID:       m.SenderID,
			SenderName:     m.SenderName,
			SenderRole:     m.SenderRole,
			ReceiverID:     m.ReceiverID,
			ReceiverName:   m.ReceiverName,
			ReceiverRole:   m.ReceiverRole,
			DEff:           m.DEff,
			Beta:           m.Beta,
			Alignment:      m.Alignment,
			CPair:          m.CPair,
			BetaStatus:     string(m.BetaStatus),
			IsUnilateral:   m.IsUnilateral,
			MessageContent: m.MessageContent,
			TokenCount:     m.TokenCount,
		}
	}

	return csvMeasurements
}

// createExportError creates a structured error for export failures.
// It analyzes the underlying error cause to provide specific guidance.
func createExportError(command, format, path string, cause error) *werrors.WeaverError {
	errStr := cause.Error()

	switch {
	case strings.Contains(errStr, "permission denied"):
		return werrors.IOWrap(cause, werrors.ErrExportPermissionDenied, "export failed: permission denied").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("Check write permissions for the export directory").
			WithSuggestion("Try exporting to a different location").
			WithSuggestion("Run 'ls -la' on the directory to check permissions")

	case strings.Contains(errStr, "no space left") || strings.Contains(errStr, "disk quota"):
		return werrors.IOWrap(cause, werrors.ErrExportDiskFull, "export failed: disk space full").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("Free up disk space").
			WithSuggestion("Try exporting to a different disk or location").
			WithSuggestion("Check disk usage with 'df -h'")

	case strings.Contains(errStr, "read-only"):
		return werrors.IOWrap(cause, werrors.ErrExportReadOnly, "export failed: read-only filesystem").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("The target location is on a read-only filesystem").
			WithSuggestion("Try exporting to a different location").
			WithSuggestion("Remount the filesystem with write permissions if applicable")

	case strings.Contains(errStr, "file name too long") || strings.Contains(errStr, "name too long"):
		return werrors.IOWrap(cause, werrors.ErrExportPathTooLong, "export failed: path too long").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithContext("path_length", fmt.Sprintf("%d", len(path))).
			WithSuggestion("Use a shorter export path").
			WithSuggestion("Configure a different export directory in session settings")

	case strings.Contains(errStr, "invalid argument") || strings.Contains(errStr, "invalid path"):
		return werrors.IOWrap(cause, werrors.ErrExportInvalidPath, "export failed: invalid path").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("Check the export path for invalid characters").
			WithSuggestion("Ensure the path is properly formatted")

	case strings.Contains(errStr, "no such file or directory"):
		return werrors.IOWrap(cause, werrors.ErrExportDirCreateFailed, "export failed: directory does not exist").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("Check if the parent directory exists").
			WithSuggestion("The export directory could not be created")

	case strings.Contains(errStr, "is a directory"):
		return werrors.IOWrap(cause, werrors.ErrExportWriteFailed, "export failed: path is a directory").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("The specified path is a directory, not a file").
			WithSuggestion("Check the export path configuration")

	case strings.Contains(errStr, "too many open files"):
		return werrors.IOWrap(cause, werrors.ErrExportWriteFailed, "export failed: too many open files").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("Too many files are currently open").
			WithSuggestion("Close some files or increase the system file descriptor limit")

	default:
		return werrors.IOWrap(cause, werrors.ErrExportWriteFailed, "export failed").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("Check if the export directory exists and is writable").
			WithSuggestion("Review the error details for more information").
			WithSuggestion("Try running the export command again")
	}
}

// createExportNoDataError creates a structured error when no data is available to export.
func createExportNoDataError(command, format string) *werrors.WeaverError {
	return werrors.IO(werrors.ErrExportNoData, "no data available to export").
		WithContext("command", command).
		WithContext("format", format).
		WithSuggestion("Run some analyses first to generate data").
		WithSuggestion("Use /extract and /analyze commands to create measurements").
		WithSuggestion("Use /session to check current session status")
}

// createExportDirError creates a structured error for directory creation failures.
func createExportDirError(command, format, path string, cause error) *werrors.WeaverError {
	errStr := cause.Error()

	switch {
	case strings.Contains(errStr, "permission denied"):
		return werrors.IOWrap(cause, werrors.ErrExportDirCreateFailed, "failed to create export directory: permission denied").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("Check write permissions for the parent directory").
			WithSuggestion("Try exporting to a different location")

	case strings.Contains(errStr, "no space left"):
		return werrors.IOWrap(cause, werrors.ErrExportDiskFull, "failed to create export directory: disk full").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("Free up disk space before exporting")

	case strings.Contains(errStr, "file exists"):
		return werrors.IOWrap(cause, werrors.ErrExportDirCreateFailed, "failed to create export directory: path exists as file").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("A file exists at the directory path").
			WithSuggestion("Remove the file or use a different export location")

	default:
		return werrors.IOWrap(cause, werrors.ErrExportDirCreateFailed, "failed to create export directory").
			WithContext("command", command).
			WithContext("format", format).
			WithContext("path", path).
			WithSuggestion("Check if the parent directory exists").
			WithSuggestion("Verify write permissions for the directory")
	}
}

// createExportMultipleError creates a structured error for /export_all failures.
func createExportMultipleError(failedFormats []string, firstError error) *werrors.WeaverError {
	return werrors.IOWrap(firstError, werrors.ErrExportFailed, "some exports failed").
		WithContext("command", "/export_all").
		WithContext("failed_formats", strings.Join(failedFormats, ", ")).
		WithSuggestion("Check individual export error messages above").
		WithSuggestion("Fix the issues and run /export_all again").
		WithSuggestion("Or run individual export commands for specific formats")
}

// Close closes the shell.
func (s *Shell) Close() error {
	return s.rl.Close()
}
