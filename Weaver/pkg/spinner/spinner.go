// Package spinner provides animated terminal spinners for long-running operations.
// It displays visual feedback with configurable character sets and elapsed time.
package spinner

import (
	"io"
	"os"
	"sync"
	"time"
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

	return &Spinner{
		config: config,
	}
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
