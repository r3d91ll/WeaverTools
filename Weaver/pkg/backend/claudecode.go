package backend

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// ClaudeCode wraps the Claude Code CLI as a backend.
type ClaudeCode struct {
	name         string
	systemPrompt string
	contextLimit int
}

// ClaudeCodeConfig holds configuration for Claude Code backend.
type ClaudeCodeConfig struct {
	Name         string `yaml:"name"`
	SystemPrompt string `yaml:"system_prompt"`
	ContextLimit int    `yaml:"context_limit"`
}

// NewClaudeCode creates a new Claude Code backend.
func NewClaudeCode(cfg ClaudeCodeConfig) *ClaudeCode {
	name := cfg.Name
	if name == "" {
		name = "claude-code"
	}
	contextLimit := cfg.ContextLimit
	if contextLimit == 0 {
		contextLimit = 200000
	}
	return &ClaudeCode{
		name:         name,
		systemPrompt: cfg.SystemPrompt,
		contextLimit: contextLimit,
	}
}

func (c *ClaudeCode) Name() string { return c.name }
func (c *ClaudeCode) Type() Type   { return TypeClaudeCode }

func (c *ClaudeCode) IsAvailable(ctx context.Context) bool {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "claude", "--version")
	return cmd.Run() == nil
}

func (c *ClaudeCode) Capabilities() Capabilities {
	return Capabilities{
		ContextLimit:      c.contextLimit,
		SupportsTools:     true,
		SupportsStreaming: true,
		SupportsHidden:    false,
		MaxTokens:         8192,
	}
}

func (c *ClaudeCode) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	start := time.Now()
	prompt := c.buildPrompt(req.Messages)

	cmd := exec.CommandContext(ctx, "claude", "-p", "--output-format", "json")
	if c.systemPrompt != "" {
		cmd.Args = append(cmd.Args, "--system-prompt", c.systemPrompt)
	}
	cmd.Stdin = strings.NewReader(prompt)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("claude error: %s", string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to run claude: %w", err)
	}

	var resp struct {
		Result string `json:"result"`
	}
	content := ""
	if err := json.Unmarshal(output, &resp); err != nil {
		content = strings.TrimSpace(string(output))
	} else {
		content = resp.Result
	}

	return &ChatResponse{
		Content:      content,
		Model:        "claude-code",
		FinishReason: "stop",
		LatencyMS:    float64(time.Since(start).Milliseconds()),
		Usage: TokenUsage{
			PromptTokens:     len(prompt) / 4,
			CompletionTokens: len(content) / 4,
			TotalTokens:      (len(prompt) + len(content)) / 4,
		},
	}, nil
}

func (c *ClaudeCode) ChatStream(ctx context.Context, req ChatRequest) (<-chan StreamChunk, <-chan error) {
	chunks := make(chan StreamChunk, 100)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		prompt := c.buildPrompt(req.Messages)

		cmd := exec.CommandContext(ctx, "claude",
			"-p", "--verbose",
			"--output-format", "stream-json",
			"--dangerously-skip-permissions",
		)
		if c.systemPrompt != "" {
			cmd.Args = append(cmd.Args, "--system-prompt", c.systemPrompt)
		}

		stdin, err := cmd.StdinPipe()
		if err != nil {
			errs <- err
			return
		}
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			errs <- err
			return
		}
		if err := cmd.Start(); err != nil {
			errs <- err
			return
		}

		if _, err := stdin.Write([]byte(prompt)); err != nil {
			errs <- fmt.Errorf("failed to write prompt: %w", err)
			return
		}
		stdin.Close()

		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}

			var event struct {
				Type  string `json:"type"`
				Delta struct {
					Text string `json:"text"`
				} `json:"delta"`
				Result string `json:"result"`
			}

			if err := json.Unmarshal([]byte(line), &event); err != nil {
				chunks <- StreamChunk{Content: line}
				continue
			}

			switch event.Type {
			case "content_block_delta":
				if event.Delta.Text != "" {
					chunks <- StreamChunk{Content: event.Delta.Text}
				}
			case "message_delta":
				chunks <- StreamChunk{Done: true, FinishReason: "stop"}
			default:
				if event.Result != "" {
					chunks <- StreamChunk{Content: event.Result}
				}
			}
		}

		if err := cmd.Wait(); err != nil {
			// Only send error if channel isn't full
			select {
			case errs <- fmt.Errorf("command failed: %w", err):
			default:
			}
		}
	}()

	return chunks, errs
}

func (c *ClaudeCode) buildPrompt(messages []ChatMessage) string {
	var parts []string
	for _, msg := range messages {
		switch msg.Role {
		case "user":
			parts = append(parts, fmt.Sprintf("User: %s", msg.Content))
		case "assistant":
			parts = append(parts, fmt.Sprintf("Assistant: %s", msg.Content))
		}
	}
	parts = append(parts, "Assistant:")
	return strings.Join(parts, "\n\n")
}
