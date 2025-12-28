package backend

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
)

// ClaudeCode wraps the Claude Code CLI as a backend.
type ClaudeCode struct {
	name         string
	systemPrompt string
	contextLimit int
	maxTokens    int
}

// ClaudeCodeConfig holds configuration for Claude Code backend.
type ClaudeCodeConfig struct {
	Name         string `yaml:"name"`
	SystemPrompt string `yaml:"system_prompt"`
	ContextLimit int    `yaml:"context_limit"`
	MaxTokens    int    `yaml:"max_tokens"` // Default: 25000 (Claude CLI default)
}

// Validate checks if the ClaudeCodeConfig is valid.
// Returns nil for valid configs (including empty/default config).
// Returns *WeaverError for invalid configurations.
func (c ClaudeCodeConfig) Validate() *werrors.WeaverError {
	// Validate Name format if provided (no special characters that could cause issues)
	if c.Name != "" && strings.ContainsAny(c.Name, " \t\n\r@#$%^&*(){}[]|\\<>") {
		return werrors.ValidationInvalid("Name", c.Name, "contains invalid characters").
			WithSuggestion("Use alphanumeric characters, hyphens, or underscores only").
			WithSuggestion("Example: 'my-claude-backend' or 'claude_code_1'")
	}

	// Validate ContextLimit is non-negative
	if c.ContextLimit < 0 {
		return werrors.ValidationOutOfRange("ContextLimit", c.ContextLimit, 0, "unlimited").
			WithSuggestion("ContextLimit must be 0 or greater").
			WithSuggestion("Use 0 to apply the default context limit (200000)")
	}

	// Validate MaxTokens is non-negative
	if c.MaxTokens < 0 {
		return werrors.ValidationOutOfRange("MaxTokens", c.MaxTokens, 0, "unlimited").
			WithSuggestion("MaxTokens must be 0 or greater").
			WithSuggestion("Use 0 to apply the default max tokens (25000)")
	}

	return nil
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
	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = 25000 // Claude CLI default (configurable via MAX_MCP_OUTPUT_TOKENS)
	}
	return &ClaudeCode{
		name:         name,
		systemPrompt: cfg.SystemPrompt,
		contextLimit: contextLimit,
		maxTokens:    maxTokens,
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
		MaxTokens:         c.maxTokens,
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
		return nil, createClaudeChatError(ctx, err)
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

	// Token usage is estimated using a simple heuristic (chars/4)
	// since Claude CLI doesn't provide actual token counts in its output
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

		// --dangerously-skip-permissions is required for non-interactive streaming mode.
		// Without it, Claude CLI prompts for confirmation which blocks the subprocess.
		// This is safe in this context because Weaver is designed for automated agent
		// orchestration where the user has already consented to agent operations.
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
			errs <- createStreamSetupError(err, "stdin pipe")
			return
		}
		defer stdin.Close() // Ensure stdin is closed even on error

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			errs <- createStreamSetupError(err, "stdout pipe")
			return
		}
		if err := cmd.Start(); err != nil {
			errs <- createStreamStartError(ctx, err)
			return
		}

		if _, err := stdin.Write([]byte(prompt)); err != nil {
			errs <- createStreamWriteError(err)
			return
		}
		stdin.Close() // Close immediately to signal EOF to subprocess

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
			case errs <- createStreamCommandError(ctx, err):
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

// -----------------------------------------------------------------------------
// Error Helper Functions
// -----------------------------------------------------------------------------
// These functions create structured WeaverErrors for different Claude Code
// failure scenarios with appropriate context and suggestions.

// createClaudeChatError creates a structured error for Chat() failures.
// It distinguishes between: CLI not installed, authentication issues,
// API errors, timeout, and general execution failures.
func createClaudeChatError(ctx context.Context, err error) *werrors.WeaverError {
	errStr := err.Error()

	// Check for CLI not installed
	if isNotInstalledError(errStr) {
		return werrors.BackendWithContext(
			werrors.ErrBackendNotInstalled,
			"Claude CLI is not installed or not found in PATH",
			map[string]string{werrors.ContextBackend: werrors.BackendClaudeCode},
		).WithCause(err).
			WithSuggestion("Install Claude CLI: npm install -g @anthropic-ai/claude-cli").
			WithSuggestion("Or on macOS: brew install anthropic/tap/claude-cli").
			WithSuggestion("After installation, authenticate with: claude auth login")
	}

	// Check for context timeout/cancellation
	if ctx.Err() != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return werrors.BackendWithContext(
				werrors.ErrBackendTimeout,
				"Claude CLI request timed out",
				map[string]string{werrors.ContextBackend: werrors.BackendClaudeCode},
			).WithCause(err).
				WithSuggestion("The request took too long to complete").
				WithSuggestion("Try a shorter prompt or simpler request").
				WithSuggestion("Check your network connection")
		}
		return werrors.BackendWithContext(
			werrors.ErrBackendConnectionFailed,
			"Claude CLI request was cancelled",
			map[string]string{werrors.ContextBackend: werrors.BackendClaudeCode},
		).WithCause(err).
			WithSuggestion("The request was interrupted before completion")
	}

	// Handle exit errors with stderr content
	if exitErr, ok := err.(*exec.ExitError); ok {
		stderr := string(exitErr.Stderr)
		return createExitError(exitErr, stderr)
	}

	// Generic execution failure
	return werrors.BackendWrap(err, werrors.ErrBackendConnectionFailed,
		"failed to execute Claude CLI").
		WithContext("backend", "claudecode").
		WithSuggestion("Check that 'claude' is in your PATH").
		WithSuggestion("Try running 'claude --version' to verify installation")
}

// createExitError creates a structured error from Claude CLI exit errors.
// It parses stderr to determine the specific error type.
func createExitError(exitErr *exec.ExitError, stderr string) *werrors.WeaverError {
	stderrLower := strings.ToLower(stderr)

	// Authentication errors
	if isAuthError(stderrLower) {
		return werrors.BackendWithContext(
			werrors.ErrBackendAuthFailed,
			"Claude CLI authentication failed",
			map[string]string{
				werrors.ContextBackend: werrors.BackendClaudeCode,
				"exit_code":            fmt.Sprintf("%d", exitErr.ExitCode()),
			},
		).WithCause(exitErr).
			WithContext("details", truncateStderr(stderr)).
			WithSuggestion("Run 'claude auth login' to authenticate").
			WithSuggestion("Check that your API key is valid").
			WithSuggestion("Verify your Anthropic account is active")
	}

	// Rate limiting
	if isRateLimitError(stderrLower) {
		return werrors.BackendWithContext(
			werrors.ErrBackendAPIError,
			"Claude API rate limit exceeded",
			map[string]string{
				werrors.ContextBackend: werrors.BackendClaudeCode,
				"exit_code":            fmt.Sprintf("%d", exitErr.ExitCode()),
			},
		).WithCause(exitErr).
			WithContext("details", truncateStderr(stderr)).
			WithSuggestion("Wait a few minutes and try again").
			WithSuggestion("Consider reducing request frequency").
			WithSuggestion("Check your API usage limits at console.anthropic.com")
	}

	// API errors (model not found, invalid request, etc.)
	if isAPIError(stderrLower) {
		return werrors.BackendWithContext(
			werrors.ErrBackendAPIError,
			"Claude API returned an error",
			map[string]string{
				werrors.ContextBackend: werrors.BackendClaudeCode,
				"exit_code":            fmt.Sprintf("%d", exitErr.ExitCode()),
			},
		).WithCause(exitErr).
			WithContext("details", truncateStderr(stderr)).
			WithSuggestion("Check the error details above for more information").
			WithSuggestion("Verify your request parameters are valid").
			WithSuggestion("Check Anthropic status page for service issues")
	}

	// Network/connection errors
	if isNetworkError(stderrLower) {
		return werrors.BackendWithContext(
			werrors.ErrBackendConnectionFailed,
			"Claude CLI could not connect to API",
			map[string]string{
				werrors.ContextBackend: werrors.BackendClaudeCode,
				"exit_code":            fmt.Sprintf("%d", exitErr.ExitCode()),
			},
		).WithCause(exitErr).
			WithContext("details", truncateStderr(stderr)).
			WithSuggestion("Check your internet connection").
			WithSuggestion("Verify firewall settings allow outbound HTTPS").
			WithSuggestion("Try again in a few moments")
	}

	// Generic API error with stderr content
	return werrors.BackendWithContext(
		werrors.ErrBackendAPIError,
		"Claude CLI returned an error",
		map[string]string{
			werrors.ContextBackend: werrors.BackendClaudeCode,
			"exit_code":            fmt.Sprintf("%d", exitErr.ExitCode()),
		},
	).WithCause(exitErr).
		WithContext("details", truncateStderr(stderr)).
		WithSuggestion("Check the error details above").
		WithSuggestion("Run 'claude --help' for CLI usage information")
}

// createStreamSetupError creates a structured error for stream pipe setup failures.
func createStreamSetupError(err error, pipeType string) *werrors.WeaverError {
	return werrors.BackendWrap(err, werrors.ErrBackendStreamFailed,
		fmt.Sprintf("failed to create %s for Claude CLI", pipeType)).
		WithContext("backend", "claudecode").
		WithContext("pipe_type", pipeType).
		WithSuggestion("This is likely a system resource issue").
		WithSuggestion("Try closing some applications to free resources").
		WithSuggestion("Check system file descriptor limits")
}

// createStreamStartError creates a structured error for stream command start failures.
func createStreamStartError(ctx context.Context, err error) *werrors.WeaverError {
	errStr := err.Error()

	// Check for CLI not installed
	if isNotInstalledError(errStr) {
		return werrors.BackendWithContext(
			werrors.ErrBackendNotInstalled,
			"Claude CLI is not installed or not found in PATH",
			map[string]string{werrors.ContextBackend: werrors.BackendClaudeCode},
		).WithCause(err).
			WithSuggestion("Install Claude CLI: npm install -g @anthropic-ai/claude-cli").
			WithSuggestion("Or on macOS: brew install anthropic/tap/claude-cli").
			WithSuggestion("After installation, authenticate with: claude auth login")
	}

	return werrors.BackendWrap(err, werrors.ErrBackendStreamFailed,
		"failed to start Claude CLI for streaming").
		WithContext("backend", "claudecode").
		WithSuggestion("Check that 'claude' is in your PATH").
		WithSuggestion("Try running 'claude --version' to verify installation")
}

// createStreamWriteError creates a structured error for stream prompt write failures.
func createStreamWriteError(err error) *werrors.WeaverError {
	return werrors.BackendWrap(err, werrors.ErrBackendStreamFailed,
		"failed to send prompt to Claude CLI").
		WithContext("backend", "claudecode").
		WithSuggestion("The Claude CLI process may have terminated unexpectedly").
		WithSuggestion("Try the request again").
		WithSuggestion("Check system resources and memory availability")
}

// createStreamCommandError creates a structured error for streaming command failures.
func createStreamCommandError(ctx context.Context, err error) *werrors.WeaverError {
	// Check for context timeout/cancellation
	if ctx.Err() != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return werrors.BackendWithContext(
				werrors.ErrBackendTimeout,
				"Claude CLI streaming request timed out",
				map[string]string{werrors.ContextBackend: werrors.BackendClaudeCode},
			).WithCause(err).
				WithSuggestion("The streaming request took too long to complete").
				WithSuggestion("Try a shorter prompt or simpler request")
		}
		return werrors.BackendWithContext(
			werrors.ErrBackendStreamFailed,
			"Claude CLI streaming was cancelled",
			map[string]string{werrors.ContextBackend: werrors.BackendClaudeCode},
		).WithCause(err).
			WithSuggestion("The streaming request was interrupted")
	}

	// Handle exit errors
	if exitErr, ok := err.(*exec.ExitError); ok {
		stderr := string(exitErr.Stderr)
		if stderr != "" {
			return createExitError(exitErr, stderr)
		}
		// No stderr, generic stream failure
		return werrors.BackendWithContext(
			werrors.ErrBackendStreamFailed,
			"Claude CLI streaming command failed",
			map[string]string{
				werrors.ContextBackend: werrors.BackendClaudeCode,
				"exit_code":            fmt.Sprintf("%d", exitErr.ExitCode()),
			},
		).WithCause(err).
			WithSuggestion("The Claude CLI exited unexpectedly during streaming").
			WithSuggestion("Try the request again").
			WithSuggestion("Check 'claude auth status' for authentication issues")
	}

	// Generic stream failure
	return werrors.BackendWrap(err, werrors.ErrBackendStreamFailed,
		"Claude CLI streaming command failed").
		WithContext("backend", "claudecode").
		WithSuggestion("An error occurred during streaming response").
		WithSuggestion("Try the request again")
}

// -----------------------------------------------------------------------------
// Error Detection Helpers
// -----------------------------------------------------------------------------

// isNotInstalledError checks if the error indicates the CLI is not installed.
func isNotInstalledError(errStr string) bool {
	errLower := strings.ToLower(errStr)
	return strings.Contains(errLower, "executable file not found") ||
		strings.Contains(errLower, "no such file or directory") ||
		strings.Contains(errLower, "command not found") ||
		strings.Contains(errLower, "not found in path")
}

// isAuthError checks if stderr indicates an authentication failure.
func isAuthError(stderr string) bool {
	return strings.Contains(stderr, "unauthorized") ||
		strings.Contains(stderr, "authentication") ||
		strings.Contains(stderr, "auth") && strings.Contains(stderr, "fail") ||
		strings.Contains(stderr, "api key") ||
		strings.Contains(stderr, "invalid key") ||
		strings.Contains(stderr, "401") ||
		strings.Contains(stderr, "not authenticated") ||
		strings.Contains(stderr, "login required")
}

// isRateLimitError checks if stderr indicates rate limiting.
func isRateLimitError(stderr string) bool {
	return strings.Contains(stderr, "rate limit") ||
		strings.Contains(stderr, "too many requests") ||
		strings.Contains(stderr, "429") ||
		strings.Contains(stderr, "quota exceeded") ||
		strings.Contains(stderr, "throttl")
}

// isAPIError checks if stderr indicates a general API error.
func isAPIError(stderr string) bool {
	return strings.Contains(stderr, "api error") ||
		strings.Contains(stderr, "api_error") ||
		strings.Contains(stderr, "invalid request") ||
		strings.Contains(stderr, "bad request") ||
		strings.Contains(stderr, "400") ||
		strings.Contains(stderr, "500") ||
		strings.Contains(stderr, "502") ||
		strings.Contains(stderr, "503") ||
		strings.Contains(stderr, "model not found") ||
		strings.Contains(stderr, "invalid model")
}

// isNetworkError checks if stderr indicates a network error.
func isNetworkError(stderr string) bool {
	return strings.Contains(stderr, "network") ||
		strings.Contains(stderr, "connection") && strings.Contains(stderr, "fail") ||
		strings.Contains(stderr, "connection") && strings.Contains(stderr, "refuse") ||
		strings.Contains(stderr, "timeout") ||
		strings.Contains(stderr, "timed out") ||
		strings.Contains(stderr, "dns") ||
		strings.Contains(stderr, "unreachable") ||
		strings.Contains(stderr, "no route to host")
}

// truncateStderr truncates stderr to a reasonable length for error context.
func truncateStderr(stderr string) string {
	stderr = strings.TrimSpace(stderr)
	if len(stderr) > 200 {
		return stderr[:200] + "..."
	}
	return stderr
}
