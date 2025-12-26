// Package shell provides the interactive REPL for Weaver.
package shell

import (
	"context"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/chzyer/readline"
	"github.com/r3d91ll/weaver/pkg/analysis"
	"github.com/r3d91ll/weaver/pkg/concepts"
	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/weaver/pkg/spinner"
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
	}, nil
}

// Run starts the interactive loop.
func (s *Shell) Run(ctx context.Context) error {
	defer s.rl.Close()

	fmt.Println("Type a message to chat. Use @agent to target a specific agent.")
	fmt.Println("Commands: /agents, /session, /history, /clear, /help, /quit")
	fmt.Println("Concepts: /extract, /analyze, /compare, /concepts, /metrics")
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
				fmt.Printf("Error: %v\n", err)
			}
			continue
		}

		// Handle message
		if err := s.handleMessage(ctx, line); err != nil {
			fmt.Printf("Error: %v\n", err)
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
		s.printHelp()

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
			return fmt.Errorf("no message after @%s", targetAgent)
		}
	}

	// Get agent
	agent, ok := s.agents.Get(targetAgent)
	if !ok {
		return fmt.Errorf("agent %q not found", targetAgent)
	}

	// Add user message to conversation
	userMsg := yarn.NewAgentMessage(yarn.RoleUser, message, "user", "user")
	s.conv.Add(userMsg)

	// Show thinking spinner with elapsed time
	spin := spinner.New(fmt.Sprintf("\033[33m[%s]\033[0m thinking...", agent.Name()))
	spin.Start()

	// Get response
	resp, err := agent.Chat(ctx, s.conv.History(-1))
	spin.Stop()
	if err != nil {
		return err
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
	fmt.Println("Commands:")
	fmt.Println("  /agents        - List available agents")
	fmt.Println("  /session       - Show session info")
	fmt.Println("  /history       - Show conversation history")
	fmt.Println("  /clear         - Start new conversation")
	fmt.Println("  /default <agent> - Set default agent")
	fmt.Println("  /quit          - Exit")
	fmt.Println()
	fmt.Println("Concept Extraction & Analysis:")
	fmt.Println("  /extract <concept> [n]   - Extract n samples (default 10)")
	fmt.Println("  /analyze <concept>       - Run Kakeya geometry analysis")
	fmt.Println("  /compare <c1> <c2>       - Compare two concepts")
	fmt.Println("  /validate <concept> [n]  - Test consistency over n iterations")
	fmt.Println("  /metrics <concept>       - Show raw metric values")
	fmt.Println("  /concepts                - List stored concepts")
	fmt.Println("  /clear_concepts          - Remove all concepts")
	fmt.Println()
	fmt.Println("Messages:")
	fmt.Println("  <text>         - Send to default agent")
	fmt.Println("  @senior <text> - Send to senior agent")
	fmt.Println("  @junior <text> - Send to junior agent")
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
			return fmt.Errorf("invalid count: %s", args[1])
		}
		if n <= 0 {
			return fmt.Errorf("count must be positive")
		}
		if n > 100 {
			return fmt.Errorf("count exceeds maximum (100)")
		}
		count = n
	}

	// Find an agent with hidden state support
	extractAgent, err := s.findHiddenStateAgent(ctx)
	if err != nil {
		return err
	}

	fmt.Printf("\033[33mExtracting %d samples for '%s' using %s...\033[0m\n", count, concept, extractAgent.Name())

	// Create extractor and run
	extractor := concepts.NewExtractor(extractAgent.Backend, s.conceptStore)
	cfg := concepts.DefaultExtractionConfig(concept, count)

	result, err := extractor.Extract(ctx, cfg)
	if err != nil {
		return err
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
		return fmt.Errorf("concept %q not found (use /extract first)", conceptName)
	}

	vectors := concept.VectorsAsFloat64()
	if len(vectors) < 3 {
		return fmt.Errorf("need at least 3 samples, have %d", len(vectors))
	}

	fmt.Printf("\033[33mAnalyzing '%s' (%d vectors, %d dimensions)...\033[0m\n",
		conceptName, len(vectors), concept.Dimension())

	result, err := s.analysisClient.AnalyzeGeometry(ctx, vectors)
	if err != nil {
		return err
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
		return fmt.Errorf("concept %q not found", name1)
	}
	concept2, ok := s.conceptStore.Get(name2)
	if !ok {
		return fmt.Errorf("concept %q not found", name2)
	}

	vectors1 := concept1.VectorsAsFloat64()
	vectors2 := concept2.VectorsAsFloat64()

	if len(vectors1) < 3 {
		return fmt.Errorf("%q needs at least 3 samples, has %d", name1, len(vectors1))
	}
	if len(vectors2) < 3 {
		return fmt.Errorf("%q needs at least 3 samples, has %d", name2, len(vectors2))
	}

	fmt.Printf("\033[33mComparing '%s' (%d) vs '%s' (%d)...\033[0m\n",
		name1, len(vectors1), name2, len(vectors2))

	result, err := s.analysisClient.CompareBilateral(ctx, vectors1, vectors2)
	if err != nil {
		return err
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
			return fmt.Errorf("invalid iterations: %s", args[1])
		}
		if n <= 0 {
			return fmt.Errorf("iterations must be positive")
		}
		if n > 20 {
			return fmt.Errorf("iterations exceeds maximum (20)")
		}
		iterations = n
	}

	// Find an agent with hidden state support
	extractAgent, err := s.findHiddenStateAgent(ctx)
	if err != nil {
		return err
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
		return fmt.Errorf("need at least 2 successful iterations for validation")
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
		return fmt.Errorf("concept %q not found", conceptName)
	}

	vectors := concept.VectorsAsFloat64()
	if len(vectors) < 3 {
		return fmt.Errorf("need at least 3 samples, have %d", len(vectors))
	}

	result, err := s.analysisClient.AnalyzeGeometry(ctx, vectors)
	if err != nil {
		return err
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
// selected deterministically by sorted name.
func (s *Shell) findHiddenStateAgent(ctx context.Context) (*runtime.Agent, error) {
	status := s.agents.Status(ctx)
	names := make([]string, 0, len(status))
	for name := range status {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		agent, ok := s.agents.Get(name)
		if ok && agent.SupportsHiddenStates() {
			return agent, nil
		}
	}
	return nil, fmt.Errorf("no agent with hidden state support available")
}

// Close closes the shell.
func (s *Shell) Close() error {
	return s.rl.Close()
}
