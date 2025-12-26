// Package spinner provides animated terminal spinners for long-running operations.
// It displays visual feedback with configurable character sets and elapsed time.
package spinner

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/term"
)

// ANSI escape sequences for terminal control.
const (
	// hideCursor hides the terminal cursor.
	hideCursor = "\033[?25l"
	// showCursor shows the terminal cursor.
	showCursor = "\033[?25h"
	// clearLine clears the current line and moves cursor to beginning.
	clearLine = "\r\033[K"
	// carriageReturn moves cursor to beginning of line.
	carriageReturn = "\r"

	// Color codes for status indicators.
	colorGreen = "\033[32m"
	colorRed   = "\033[31m"
	colorReset = "\033[0m"

	// Status indicator symbols.
	symbolSuccess = "✓"
	symbolFailure = "✗"
)

// CharSet defines a set of characters for spinner animation.
type CharSet []string

// Common spinner character sets for different visual styles.
var (
	// Braille provides smooth animation using braille characters.
	// Best for modern terminals with Unicode support.
	Braille = CharSet{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

	// Dots provides a simple dot animation.
	// Good fallback for terminals with limited Unicode support.
	Dots = CharSet{"⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"}

	// Line provides a rotating line animation.
	// Classic spinner style, works in most terminals.
	Line = CharSet{"|", "/", "-", "\\"}

	// Arc provides a rotating arc animation.
	// Smooth visual effect for modern terminals.
	Arc = CharSet{"◜", "◠", "◝", "◞", "◡", "◟"}
)

// Config holds configuration options for a spinner.
type Config struct {
	// CharSet defines the animation characters to cycle through.
	// Defaults to Braille if not specified.
	CharSet CharSet

	// Message is the text displayed next to the spinner.
	Message string

	// RefreshRate controls how fast the spinner animates.
	// Defaults to 80ms for smooth animation.
	RefreshRate time.Duration

	// ShowElapsed displays elapsed time next to the message.
	// Format: "message (1.2s)" or "message (1m 30s)"
	ShowElapsed bool

	// Writer is the output destination.
	// Defaults to os.Stderr if not specified.
	Writer io.Writer

	// HideCursor hides the terminal cursor while spinning.
	// Defaults to true for cleaner visual appearance.
	HideCursor bool

	// IsTTY indicates whether the output is a terminal.
	// When false, spinner falls back to static messages without animation.
	// If not explicitly set, it is auto-detected from the Writer.
	IsTTY *bool
}

// DefaultConfig returns a configuration with sensible defaults.
func DefaultConfig() Config {
	return Config{
		CharSet:     Braille,
		Message:     "Loading...",
		RefreshRate: 80 * time.Millisecond,
		ShowElapsed: true,
		Writer:      os.Stderr,
		HideCursor:  true,
	}
}

// Spinner displays an animated spinner in the terminal.
type Spinner struct {
	mu sync.Mutex

	config    Config
	active    bool
	startTime time.Time
	stopCh    chan struct{}
	doneCh    chan struct{}
	frame     int
	isTTY     bool // resolved TTY status (from config or auto-detected)

	// lastOutput stores the length of last printed line for clearing.
	lastOutput int
}

// New creates a new spinner with the given message.
// Uses default configuration values.
func New(message string) *Spinner {
	cfg := DefaultConfig()
	cfg.Message = message
	return NewWithConfig(cfg)
}

// NewWithConfig creates a new spinner with custom configuration.
func NewWithConfig(config Config) *Spinner {
	// Apply defaults for unset values
	if len(config.CharSet) == 0 {
		config.CharSet = Braille
	}
	if config.RefreshRate == 0 {
		config.RefreshRate = 80 * time.Millisecond
	}
	if config.Writer == nil {
		config.Writer = os.Stderr
	}

	// Determine TTY status: use explicit config or auto-detect
	isTTY := isTerminalWriter(config.Writer)
	if config.IsTTY != nil {
		isTTY = *config.IsTTY
	}

	return &Spinner{
		config: config,
		isTTY:  isTTY,
	}
}

// isTerminalWriter checks if the given writer is a terminal.
// Returns true if the writer is an *os.File pointing to a terminal.
func isTerminalWriter(w io.Writer) bool {
	if f, ok := w.(*os.File); ok {
		return term.IsTerminal(int(f.Fd()))
	}
	return false
}

// Message returns the current spinner message.
func (s *Spinner) Message() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.config.Message
}

// IsActive returns true if the spinner is currently running.
func (s *Spinner) IsActive() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active
}

// Elapsed returns the duration since the spinner started.
// Returns 0 if the spinner has not been started.
func (s *Spinner) Elapsed() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.startTime.IsZero() {
		return 0
	}
	return time.Since(s.startTime)
}

// IsTTY returns whether the spinner is outputting to a terminal.
// When false, the spinner uses static messages without animation.
func (s *Spinner) IsTTY() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.isTTY
}

// spin is the core animation loop that runs in a goroutine.
// It renders frames at the configured refresh rate until stopped.
func (s *Spinner) spin() {
	ticker := time.NewTicker(s.config.RefreshRate)
	defer ticker.Stop()

	// Render initial frame immediately
	s.render()

	for {
		select {
		case <-s.stopCh:
			// Signal that we're done
			close(s.doneCh)
			return
		case <-ticker.C:
			s.render()
		}
	}
}

// render writes the current spinner frame to the output.
// Thread-safe: acquires mutex to read state.
func (s *Spinner) render() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.active {
		return
	}

	// Get the current animation frame character
	char := s.config.CharSet[s.frame%len(s.config.CharSet)]
	s.frame++

	// Build the output line
	var output string
	if s.config.ShowElapsed {
		elapsed := time.Since(s.startTime)
		output = fmt.Sprintf("%s %s %s", char, s.config.Message, s.formatElapsed(elapsed))
	} else {
		output = fmt.Sprintf("%s %s", char, s.config.Message)
	}

	// Clear the previous line and write the new output
	s.clearAndWrite(output)
}

// clearAndWrite clears the current line and writes new content.
// Caller must hold the mutex.
func (s *Spinner) clearAndWrite(output string) {
	// Use carriage return + spaces to clear (more compatible than ANSI clear)
	// Then write the new content
	if s.lastOutput > 0 {
		// Clear the previous output by overwriting with spaces
		spaces := strings.Repeat(" ", s.lastOutput)
		fmt.Fprint(s.config.Writer, carriageReturn+spaces+carriageReturn)
	}

	fmt.Fprint(s.config.Writer, output)
	s.lastOutput = len(output)
}

// clearLine clears the current spinner line from the terminal.
// Caller must hold the mutex.
func (s *Spinner) clearLine() {
	if s.lastOutput > 0 {
		spaces := strings.Repeat(" ", s.lastOutput)
		fmt.Fprint(s.config.Writer, carriageReturn+spaces+carriageReturn)
		s.lastOutput = 0
	}
}

// formatElapsed formats a duration for display.
// Short durations show as "(1.2s)", longer as "(1m 30s)".
func (s *Spinner) formatElapsed(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("(%.1fs)", d.Seconds())
	}
	minutes := int(d.Minutes())
	seconds := int(d.Seconds()) % 60
	return fmt.Sprintf("(%dm %ds)", minutes, seconds)
}

// hideCursorIfEnabled writes the hide cursor escape sequence if configured.
// Caller must hold the mutex.
func (s *Spinner) hideCursorIfEnabled() {
	if s.config.HideCursor {
		fmt.Fprint(s.config.Writer, hideCursor)
	}
}

// showCursorIfEnabled writes the show cursor escape sequence if configured.
// Caller must hold the mutex.
func (s *Spinner) showCursorIfEnabled() {
	if s.config.HideCursor {
		fmt.Fprint(s.config.Writer, showCursor)
	}
}

// Start begins the spinner animation.
// It is safe to call Start on an already running spinner (no-op).
// In non-TTY mode, prints a static message without animation.
// Thread-safe: uses mutex to protect state changes.
func (s *Spinner) Start() {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Guard against double-start: if already active, do nothing
	if s.active {
		return
	}

	// Initialize state
	s.active = true
	s.startTime = time.Now()
	s.frame = 0
	s.stopCh = make(chan struct{})
	s.doneCh = make(chan struct{})

	// Non-TTY mode: print a static message and return
	if !s.isTTY {
		// Print static message without animation
		fmt.Fprintf(s.config.Writer, "%s...\n", s.config.Message)
		return
	}

	// Hide cursor for cleaner appearance
	s.hideCursorIfEnabled()

	// Start the animation goroutine
	go s.spin()
}

// Stop halts the spinner animation and cleans up.
// It is safe to call Stop on an already stopped or never-started spinner (no-op).
// Stop blocks until the animation goroutine has fully terminated.
// In non-TTY mode, just marks the spinner as inactive.
// Thread-safe: uses mutex to protect state changes.
func (s *Spinner) Stop() {
	s.mu.Lock()

	// Guard against stop-before-start or double-stop: if not active, do nothing
	if !s.active {
		s.mu.Unlock()
		return
	}

	// Mark as inactive first to prevent render() from writing
	s.active = false

	// Non-TTY mode: no goroutine was started, just mark as inactive
	if !s.isTTY {
		s.mu.Unlock()
		return
	}

	// Get references to channels before unlocking
	stopCh := s.stopCh
	doneCh := s.doneCh

	s.mu.Unlock()

	// Signal the goroutine to stop (non-blocking)
	close(stopCh)

	// Wait for the goroutine to acknowledge and exit
	<-doneCh

	// Clean up the terminal output
	s.mu.Lock()
	s.clearLine()
	s.showCursorIfEnabled()
	s.mu.Unlock()
}

// Update changes the spinner message while it is running.
// If the spinner is not active, it updates the message for the next Start().
// Thread-safe: uses mutex to protect state changes.
func (s *Spinner) Update(message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.config.Message = message
}

// Success stops the spinner and displays a success indicator.
// If message is empty, the current spinner message is used.
// Displays: ✓ message (elapsed) in green.
// Thread-safe: uses mutex to protect state changes.
func (s *Spinner) Success(message string) {
	s.complete(message, symbolSuccess, colorGreen)
}

// Fail stops the spinner and displays a failure indicator.
// If message is empty, the current spinner message is used.
// Displays: ✗ message (elapsed) in red.
// Thread-safe: uses mutex to protect state changes.
func (s *Spinner) Fail(message string) {
	s.complete(message, symbolFailure, colorRed)
}

// complete is the internal implementation for Success and Fail.
// It stops the spinner and displays a final status with the given symbol and color.
// In non-TTY mode, displays a simple status message without ANSI codes.
func (s *Spinner) complete(message, symbol, color string) {
	s.mu.Lock()

	// If not active, just display the final message without stopping anything
	if !s.active {
		if message == "" {
			message = s.config.Message
		}
		isTTY := s.isTTY
		showElapsed := s.config.ShowElapsed
		startTime := s.startTime
		s.mu.Unlock()

		// Display final status even if spinner wasn't running
		s.mu.Lock()
		var output string
		if isTTY {
			// TTY mode: use colors
			output = fmt.Sprintf("%s%s%s %s\n", color, symbol, colorReset, message)
		} else {
			// Non-TTY mode: plain text without colors
			if showElapsed && !startTime.IsZero() {
				output = fmt.Sprintf("%s %s %s\n", symbol, message, s.formatElapsed(time.Since(startTime)))
			} else {
				output = fmt.Sprintf("%s %s\n", symbol, message)
			}
		}
		fmt.Fprint(s.config.Writer, output)
		s.mu.Unlock()
		return
	}

	// Capture elapsed time and message before stopping
	elapsed := time.Since(s.startTime)
	if message == "" {
		message = s.config.Message
	}
	showElapsed := s.config.ShowElapsed
	isTTY := s.isTTY

	// Mark as inactive to prevent render() from writing
	s.active = false

	// Non-TTY mode: no goroutine was started, just display final status
	if !isTTY {
		s.mu.Unlock()

		// Display final status (plain text without ANSI codes)
		s.mu.Lock()
		var output string
		if showElapsed {
			output = fmt.Sprintf("%s %s %s\n", symbol, message, s.formatElapsed(elapsed))
		} else {
			output = fmt.Sprintf("%s %s\n", symbol, message)
		}
		fmt.Fprint(s.config.Writer, output)
		s.mu.Unlock()
		return
	}

	// Get references to channels before unlocking
	stopCh := s.stopCh
	doneCh := s.doneCh

	s.mu.Unlock()

	// Signal the goroutine to stop (non-blocking)
	close(stopCh)

	// Wait for the goroutine to acknowledge and exit
	<-doneCh

	// Clean up and display final status
	s.mu.Lock()
	s.clearLine()
	s.showCursorIfEnabled()

	// Build the final output with color
	var output string
	if showElapsed {
		output = fmt.Sprintf("%s%s%s %s %s\n", color, symbol, colorReset, message, s.formatElapsed(elapsed))
	} else {
		output = fmt.Sprintf("%s%s%s %s\n", color, symbol, colorReset, message)
	}
	fmt.Fprint(s.config.Writer, output)
	s.mu.Unlock()
}
