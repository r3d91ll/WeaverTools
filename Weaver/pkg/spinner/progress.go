// Package spinner provides animated terminal spinners and progress bars for long-running operations.
// This file implements a progress bar component for tracking operations with known total counts.
package spinner

import (
	"fmt"
	"io"
	"os"
	"strings"
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

// Unicode box-drawing characters for the progress bar.
const (
	// barFilled is the character for completed progress.
	barFilled = "█"
	// barEmpty is the character for remaining progress.
	barEmpty = "░"
)

// render writes the current progress bar state to the output.
// In TTY mode, it clears the previous line and displays an updated bar.
// In non-TTY mode, it does nothing (updates are printed on new lines instead).
// Thread-safe: acquires mutex to read state.
func (p *ProgressBar) render() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.active {
		return
	}

	// Non-TTY mode: do not render inline updates (handled separately)
	if !p.isTTY {
		return
	}

	// Build and display the progress bar
	output := p.buildOutput()
	p.clearAndWrite(output)
}

// renderNonTTY prints a progress update on a new line for non-TTY environments.
// Caller must hold the mutex.
func (p *ProgressBar) renderNonTTY() {
	if p.isTTY {
		return
	}

	output := p.buildOutput()
	fmt.Fprintln(p.config.Writer, output)
}

// buildOutput constructs the progress bar string from current state.
// Caller must hold the mutex.
// Output format: Message [████████░░░░░░░░░░░░] 40% (8/20) (2.4s)
func (p *ProgressBar) buildOutput() string {
	var parts []string

	// Add message prefix
	if p.config.Message != "" {
		parts = append(parts, p.config.Message)
	}

	// Build the visual bar
	bar := p.buildBar()
	parts = append(parts, bar)

	// Add percentage if enabled
	if p.config.ShowPercentage {
		pct := 0.0
		if p.config.Total > 0 {
			pct = float64(p.current) / float64(p.config.Total) * 100
		}
		parts = append(parts, fmt.Sprintf("%.0f%%", pct))
	}

	// Add count if enabled
	if p.config.ShowCount {
		parts = append(parts, fmt.Sprintf("(%d/%d)", p.current, p.config.Total))
	}

	// Add elapsed time if enabled
	if p.config.ShowElapsed && !p.startTime.IsZero() {
		elapsed := time.Since(p.startTime)
		parts = append(parts, p.formatElapsed(elapsed))
	}

	// Add ETA if enabled and we have enough samples
	if p.config.ShowETA && p.current >= p.config.MinSamplesForETA {
		if eta := p.calculateETA(); eta > 0 {
			parts = append(parts, p.formatETA(eta))
		}
	}

	return strings.Join(parts, " ")
}

// buildBar constructs the visual progress bar portion.
// Returns a string like "[████████░░░░░░░░░░░░]"
// Caller must hold the mutex.
func (p *ProgressBar) buildBar() string {
	width := p.config.Width
	if width <= 0 {
		width = 20
	}

	// Calculate filled portion
	filled := 0
	if p.config.Total > 0 {
		filled = (p.current * width) / p.config.Total
		if filled > width {
			filled = width
		}
		if filled < 0 {
			filled = 0
		}
	}
	empty := width - filled

	// Build the bar using unicode characters
	var bar strings.Builder
	bar.WriteString("[")
	for i := 0; i < filled; i++ {
		bar.WriteString(barFilled)
	}
	for i := 0; i < empty; i++ {
		bar.WriteString(barEmpty)
	}
	bar.WriteString("]")

	return bar.String()
}

// formatElapsed formats a duration for display.
// Short durations show as "(1.2s)", longer as "(1m 30s)".
// Caller must hold the mutex.
func (p *ProgressBar) formatElapsed(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("(%.1fs)", d.Seconds())
	}
	minutes := int(d.Minutes())
	seconds := int(d.Seconds()) % 60
	return fmt.Sprintf("(%dm %ds)", minutes, seconds)
}

// calculateETA estimates the remaining time based on average time per item.
// Uses simple division: avgTimePerItem = elapsed / current, then ETA = avgTimePerItem * remaining.
// Returns 0 if ETA cannot be calculated (no progress or already complete).
// Caller must hold the mutex.
func (p *ProgressBar) calculateETA() time.Duration {
	// Cannot calculate ETA without progress or elapsed time
	if p.current <= 0 || p.startTime.IsZero() {
		return 0
	}

	// No remaining work
	remaining := p.config.Total - p.current
	if remaining <= 0 {
		return 0
	}

	elapsed := time.Since(p.startTime)

	// Calculate average time per item and estimate remaining time
	avgTimePerItem := elapsed / time.Duration(p.current)
	eta := avgTimePerItem * time.Duration(remaining)

	return eta
}

// formatETA formats estimated time remaining for display.
// Short durations show as "ETA: 30s", longer as "ETA: 1m 15s", very long as "ETA: 2h 30m".
// Caller must hold the mutex.
func (p *ProgressBar) formatETA(d time.Duration) string {
	if d < time.Minute {
		// Round to nearest second for cleaner display
		seconds := int(d.Seconds() + 0.5)
		if seconds < 1 {
			seconds = 1
		}
		return fmt.Sprintf("ETA: %ds", seconds)
	}
	if d < time.Hour {
		minutes := int(d.Minutes())
		seconds := int(d.Seconds()) % 60
		if seconds > 0 {
			return fmt.Sprintf("ETA: %dm %ds", minutes, seconds)
		}
		return fmt.Sprintf("ETA: %dm", minutes)
	}
	// For durations >= 1 hour
	hours := int(d.Hours())
	minutes := int(d.Minutes()) % 60
	if minutes > 0 {
		return fmt.Sprintf("ETA: %dh %dm", hours, minutes)
	}
	return fmt.Sprintf("ETA: %dh", hours)
}

// clearAndWrite clears the current line and writes new content.
// Uses carriage return + spaces for cross-platform compatibility.
// Caller must hold the mutex.
func (p *ProgressBar) clearAndWrite(output string) {
	// Clear previous output by overwriting with spaces
	if p.lastOutput > 0 {
		spaces := strings.Repeat(" ", p.lastOutput)
		fmt.Fprint(p.config.Writer, carriageReturn+spaces+carriageReturn)
	}

	fmt.Fprint(p.config.Writer, output)
	p.lastOutput = len(output)
}

// clearLine clears the current progress bar line from the terminal.
// Caller must hold the mutex.
func (p *ProgressBar) clearLine() {
	if p.lastOutput > 0 {
		spaces := strings.Repeat(" ", p.lastOutput)
		fmt.Fprint(p.config.Writer, carriageReturn+spaces+carriageReturn)
		p.lastOutput = 0
	}
}

// Start begins progress tracking and displays the initial progress bar.
// It is safe to call Start on an already running progress bar (no-op).
// In non-TTY mode, prints the initial status on a new line.
// Thread-safe: uses mutex to protect state changes.
func (p *ProgressBar) Start() {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Guard against double-start: if already active, do nothing
	if p.active {
		return
	}

	// Initialize state
	p.active = true
	p.startTime = time.Now()
	p.current = 0

	// Display initial progress
	if p.isTTY {
		// TTY mode: hide cursor and render inline
		fmt.Fprint(p.config.Writer, hideCursor)
		output := p.buildOutput()
		p.clearAndWrite(output)
	} else {
		// Non-TTY mode: print on new line
		p.renderNonTTY()
	}
}

// Increment advances the progress by 1.
// Does nothing if the progress bar is not active.
// Thread-safe: uses mutex to protect state changes.
func (p *ProgressBar) Increment() {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Do nothing if not active
	if !p.active {
		return
	}

	// Increment but don't exceed total
	if p.current < p.config.Total {
		p.current++
	}

	// Update display
	if p.isTTY {
		output := p.buildOutput()
		p.clearAndWrite(output)
	} else {
		// Non-TTY: only output on significant progress or completion
		// Output every 10% or on completion
		if p.current == p.config.Total || (p.current%(p.config.Total/10+1)) == 0 {
			p.renderNonTTY()
		}
	}
}

// Set sets the progress to a specific value.
// If n is negative, it is clamped to 0. If n exceeds total, it is clamped to total.
// Does nothing if the progress bar is not active.
// Thread-safe: uses mutex to protect state changes.
func (p *ProgressBar) Set(n int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Do nothing if not active
	if !p.active {
		return
	}

	// Clamp value to valid range
	if n < 0 {
		n = 0
	}
	if n > p.config.Total {
		n = p.config.Total
	}

	// Track if we made significant progress for non-TTY output
	oldCurrent := p.current
	p.current = n

	// Update display
	if p.isTTY {
		output := p.buildOutput()
		p.clearAndWrite(output)
	} else {
		// Non-TTY: output if we've made meaningful progress (crossed 10% threshold)
		oldPct := 0
		newPct := 0
		if p.config.Total > 0 {
			oldPct = (oldCurrent * 10) / p.config.Total
			newPct = (p.current * 10) / p.config.Total
		}
		if newPct > oldPct || p.current == p.config.Total {
			p.renderNonTTY()
		}
	}
}

// Complete stops the progress bar and displays a success message.
// If message is empty, a default completion message is used.
// Displays: ✓ message (elapsed) in green.
// Does nothing if called on an inactive progress bar (but still shows message).
// Thread-safe: uses mutex to protect state changes.
func (p *ProgressBar) Complete(message string) {
	p.complete(message, symbolSuccess, colorGreen)
}

// Fail stops the progress bar and displays a failure message.
// If message is empty, a default failure message is used.
// Displays: ✗ message (elapsed) in red.
// Does nothing if called on an inactive progress bar (but still shows message).
// Thread-safe: uses mutex to protect state changes.
func (p *ProgressBar) Fail(message string) {
	p.complete(message, symbolFailure, colorRed)
}

// complete is the internal implementation for Complete and Fail.
// It stops the progress bar and displays a final status with the given symbol and color.
// In non-TTY mode, displays a simple status message without ANSI codes.
func (p *ProgressBar) complete(message, symbol, color string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Use default message if empty
	if message == "" {
		message = fmt.Sprintf("%s complete", p.config.Message)
	}

	// Capture elapsed time
	var elapsed time.Duration
	if !p.startTime.IsZero() {
		elapsed = time.Since(p.startTime)
	}

	// If not active, just display the final message
	if !p.active {
		var output string
		if p.isTTY {
			// TTY mode: use colors
			if p.config.ShowElapsed && elapsed > 0 {
				output = fmt.Sprintf("%s%s%s %s %s\n", color, symbol, colorReset, message, p.formatElapsed(elapsed))
			} else {
				output = fmt.Sprintf("%s%s%s %s\n", color, symbol, colorReset, message)
			}
		} else {
			// Non-TTY mode: plain text without colors
			if p.config.ShowElapsed && elapsed > 0 {
				output = fmt.Sprintf("%s %s %s\n", symbol, message, p.formatElapsed(elapsed))
			} else {
				output = fmt.Sprintf("%s %s\n", symbol, message)
			}
		}
		fmt.Fprint(p.config.Writer, output)
		return
	}

	// Mark as inactive
	p.active = false

	// Build final output
	var output string
	if p.isTTY {
		// Clear the progress bar line first
		p.clearLine()
		fmt.Fprint(p.config.Writer, showCursor)

		// TTY mode: use colors and elapsed time
		if p.config.ShowElapsed {
			output = fmt.Sprintf("%s%s%s %s %s\n", color, symbol, colorReset, message, p.formatElapsed(elapsed))
		} else {
			output = fmt.Sprintf("%s%s%s %s\n", color, symbol, colorReset, message)
		}
	} else {
		// Non-TTY mode: plain text without colors
		if p.config.ShowElapsed {
			output = fmt.Sprintf("%s %s %s\n", symbol, message, p.formatElapsed(elapsed))
		} else {
			output = fmt.Sprintf("%s %s\n", symbol, message)
		}
	}
	fmt.Fprint(p.config.Writer, output)
}
