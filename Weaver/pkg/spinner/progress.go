// Package spinner provides animated terminal spinners and progress bars for long-running operations.
// This file implements a progress bar component for tracking operations with known total counts.
package spinner

import (
	"io"
	"os"
	"sync"
	"time"
)

// ProgressConfig holds configuration options for a progress bar.
type ProgressConfig struct {
	// Total is the total number of items to process.
	// Must be > 0 for the progress bar to display meaningful progress.
	Total int

	// Message is the text displayed next to the progress bar.
	Message string

	// Width is the width of the progress bar in characters.
	// Defaults to 20 if not specified or <= 0.
	Width int

	// ShowPercentage displays the percentage complete (e.g., "40%").
	ShowPercentage bool

	// ShowCount displays current/total count (e.g., "(8/20)").
	ShowCount bool

	// ShowElapsed displays elapsed time since start (e.g., "(2.4s)").
	ShowElapsed bool

	// ShowETA displays estimated time remaining after enough samples.
	// Only shown after MinSamplesForETA items have been processed.
	ShowETA bool

	// MinSamplesForETA is the minimum number of samples needed before showing ETA.
	// Defaults to 2 if not specified or <= 0.
	MinSamplesForETA int

	// Writer is the output destination.
	// Defaults to os.Stderr if not specified.
	Writer io.Writer

	// IsTTY indicates whether the output is a terminal.
	// When false, progress bar falls back to simple status messages without animation.
	// If not explicitly set, it is auto-detected from the Writer.
	IsTTY *bool
}

// DefaultProgressConfig returns a progress bar configuration with sensible defaults.
func DefaultProgressConfig() ProgressConfig {
	return ProgressConfig{
		Total:            100,
		Message:          "Processing...",
		Width:            20,
		ShowPercentage:   true,
		ShowCount:        true,
		ShowElapsed:      true,
		ShowETA:          true,
		MinSamplesForETA: 2,
		Writer:           os.Stderr,
	}
}

// ProgressBar displays a visual progress bar in the terminal.
// It shows progress as a bar with optional percentage, counts, elapsed time, and ETA.
type ProgressBar struct {
	// mu protects all state fields for thread-safe access.
	mu sync.Mutex

	// config holds the progress bar configuration.
	config ProgressConfig

	// current is the current progress count (0 to Total).
	current int

	// startTime is when the progress bar was started.
	startTime time.Time

	// active indicates whether the progress bar is currently running.
	active bool

	// isTTY is the resolved TTY status (from config or auto-detected).
	isTTY bool

	// lastOutput stores the length of last printed line for clearing.
	lastOutput int
}

// NewProgress creates a new progress bar with the given total and message.
// Uses default configuration values for other options.
func NewProgress(total int, message string) *ProgressBar {
	cfg := DefaultProgressConfig()
	cfg.Total = total
	cfg.Message = message
	return NewProgressWithConfig(cfg)
}

// NewProgressWithConfig creates a new progress bar with custom configuration.
func NewProgressWithConfig(config ProgressConfig) *ProgressBar {
	// Apply defaults for unset values
	if config.Total <= 0 {
		config.Total = 100
	}
	if config.Width <= 0 {
		config.Width = 20
	}
	if config.MinSamplesForETA <= 0 {
		config.MinSamplesForETA = 2
	}
	if config.Writer == nil {
		config.Writer = os.Stderr
	}

	// Determine TTY status: use explicit config or auto-detect
	isTTY := isTerminalWriter(config.Writer)
	if config.IsTTY != nil {
		isTTY = *config.IsTTY
	}

	return &ProgressBar{
		config: config,
		isTTY:  isTTY,
	}
}

// Config returns the current progress bar configuration.
func (p *ProgressBar) Config() ProgressConfig {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.config
}

// Message returns the current progress bar message.
func (p *ProgressBar) Message() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.config.Message
}

// Total returns the total count for the progress bar.
func (p *ProgressBar) Total() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.config.Total
}

// Current returns the current progress count.
func (p *ProgressBar) Current() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.current
}

// IsActive returns true if the progress bar is currently running.
func (p *ProgressBar) IsActive() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.active
}

// Elapsed returns the duration since the progress bar started.
// Returns 0 if the progress bar has not been started.
func (p *ProgressBar) Elapsed() time.Duration {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.startTime.IsZero() {
		return 0
	}
	return time.Since(p.startTime)
}

// IsTTY returns whether the progress bar is outputting to a terminal.
// When false, the progress bar uses simple status messages without animation.
func (p *ProgressBar) IsTTY() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.isTTY
}

// Percentage returns the current progress as a percentage (0-100).
func (p *ProgressBar) Percentage() float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.config.Total <= 0 {
		return 0
	}
	return float64(p.current) / float64(p.config.Total) * 100
}
