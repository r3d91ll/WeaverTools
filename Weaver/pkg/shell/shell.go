// Package shell provides the interactive REPL for Weaver.
package shell

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/chzyer/readline"
	"github.com/r3d91ll/weaver/pkg/runtime"
	"github.com/r3d91ll/yarn"
)

// Shell is the interactive command-line interface.
type Shell struct {
	agents  *runtime.Manager
	session *yarn.Session
	conv    *yarn.Conversation
	rl      *readline.Instance
	default_ string // Default agent to route messages to
}

// Config holds shell configuration.
type Config struct {
	HistoryFile  string
	DefaultAgent string
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
		agents:   agents,
		session:  session,
		conv:     session.ActiveConversation(),
		rl:       rl,
		default_: defaultAgent,
	}, nil
}

// Run starts the interactive loop.
func (s *Shell) Run(ctx context.Context) error {
	defer s.rl.Close()

	fmt.Println("Type a message to chat. Use @agent to target a specific agent.")
	fmt.Println("Commands: /agents, /session, /history, /clear, /help, /quit")
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
			s.default_ = parts[1]
			fmt.Printf("Default agent set to: %s\n", s.default_)
		} else {
			fmt.Printf("Default agent: %s\n", s.default_)
		}

	default:
		fmt.Printf("Unknown command: %s\n", cmd)
	}

	return nil
}

func (s *Shell) handleMessage(ctx context.Context, line string) error {
	// Parse @agent prefix
	targetAgent := s.default_
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

	// Show thinking indicator
	fmt.Printf("\033[33m[%s]\033[0m thinking...\n", agent.Name())

	// Get response
	resp, err := agent.Chat(ctx, s.conv.History(-1))
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
		if name == s.default_ {
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

// Close closes the shell.
func (s *Shell) Close() error {
	return s.rl.Close()
}
