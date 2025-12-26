package spinner

import (
	"bytes"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestNew verifies that New creates a spinner with default configuration.
func TestNew(t *testing.T) {
	s := New("test message")
	if s == nil {
		t.Fatal("New returned nil")
	}
	if s.Message() != "test message" {
		t.Errorf("expected message 'test message', got %q", s.Message())
	}
	if s.IsActive() {
		t.Error("spinner should not be active before Start()")
	}
}

// TestNewWithConfig verifies that NewWithConfig applies custom configuration.
func TestNewWithConfig(t *testing.T) {
	var buf bytes.Buffer
	cfg := Config{
		CharSet:     Line,
		Message:     "custom message",
		RefreshRate: 50 * time.Millisecond,
		ShowElapsed: false,
		Writer:      &buf,
		HideCursor:  false,
	}

	s := NewWithConfig(cfg)
	if s == nil {
		t.Fatal("NewWithConfig returned nil")
	}
	if s.Message() != "custom message" {
		t.Errorf("expected message 'custom message', got %q", s.Message())
	}
}

// TestNewWithConfigDefaults verifies that missing config values get defaults.
func TestNewWithConfigDefaults(t *testing.T) {
	cfg := Config{Message: "test"}
	s := NewWithConfig(cfg)

	// CharSet should default to Braille
	if len(s.config.CharSet) != len(Braille) {
		t.Errorf("expected CharSet to default to Braille (len %d), got len %d", len(Braille), len(s.config.CharSet))
	}

	// RefreshRate should default to 80ms
	if s.config.RefreshRate != 80*time.Millisecond {
		t.Errorf("expected RefreshRate to default to 80ms, got %v", s.config.RefreshRate)
	}

	// Writer should default to os.Stderr (not nil)
	if s.config.Writer == nil {
		t.Error("expected Writer to default to os.Stderr, got nil")
	}
}

// TestBasicStartStop verifies basic start and stop functionality.
func TestBasicStartStop(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "loading",
		Writer:     &buf,
		HideCursor: false,
	})

	// Should not be active before start
	if s.IsActive() {
		t.Error("spinner should not be active before Start()")
	}

	// Start the spinner
	s.Start()

	// Give the goroutine time to start and render
	time.Sleep(50 * time.Millisecond)

	if !s.IsActive() {
		t.Error("spinner should be active after Start()")
	}

	// Stop the spinner
	s.Stop()

	if s.IsActive() {
		t.Error("spinner should not be active after Stop()")
	}
}

// TestDoubleStart verifies that double-start is a no-op.
func TestDoubleStart(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "loading",
		Writer:     &buf,
		HideCursor: false,
	})

	s.Start()
	startTime := s.startTime

	// Wait a bit and start again
	time.Sleep(20 * time.Millisecond)
	s.Start()

	// Start time should not have changed
	if s.startTime != startTime {
		t.Error("double-start should be a no-op")
	}

	s.Stop()
}

// TestStopBeforeStart verifies that stopping an unstarted spinner is safe.
func TestStopBeforeStart(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "loading",
		Writer:     &buf,
		HideCursor: false,
	})

	// This should not panic or block
	s.Stop()

	if s.IsActive() {
		t.Error("spinner should not be active")
	}
}

// TestDoubleStop verifies that double-stop is safe.
func TestDoubleStop(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "loading",
		Writer:     &buf,
		HideCursor: false,
	})

	s.Start()
	time.Sleep(20 * time.Millisecond)
	s.Stop()

	// This should not panic or block
	s.Stop()

	if s.IsActive() {
		t.Error("spinner should not be active")
	}
}

// TestElapsedTimeTracking verifies that elapsed time is tracked correctly.
func TestElapsedTimeTracking(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "loading",
		Writer:     &buf,
		HideCursor: false,
	})

	// Elapsed should be 0 before start
	if s.Elapsed() != 0 {
		t.Errorf("expected elapsed to be 0 before start, got %v", s.Elapsed())
	}

	s.Start()
	time.Sleep(100 * time.Millisecond)

	elapsed := s.Elapsed()
	if elapsed < 100*time.Millisecond {
		t.Errorf("expected elapsed >= 100ms, got %v", elapsed)
	}

	s.Stop()
}

// TestElapsedTimeFormatting verifies the formatElapsed helper.
func TestElapsedTimeFormatting(t *testing.T) {
	s := New("test")

	tests := []struct {
		duration time.Duration
		expected string
	}{
		{500 * time.Millisecond, "(0.5s)"},
		{1 * time.Second, "(1.0s)"},
		{1500 * time.Millisecond, "(1.5s)"},
		{30 * time.Second, "(30.0s)"},
		{59*time.Second + 900*time.Millisecond, "(59.9s)"},
		{60 * time.Second, "(1m 0s)"},
		{61 * time.Second, "(1m 1s)"},
		{90 * time.Second, "(1m 30s)"},
		{120 * time.Second, "(2m 0s)"},
		{5*time.Minute + 30*time.Second, "(5m 30s)"},
	}

	for _, tc := range tests {
		result := s.formatElapsed(tc.duration)
		if result != tc.expected {
			t.Errorf("formatElapsed(%v): expected %q, got %q", tc.duration, tc.expected, result)
		}
	}
}

// TestUpdateMessage verifies that Update() changes the message.
func TestUpdateMessage(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "initial",
		Writer:     &buf,
		HideCursor: false,
	})

	if s.Message() != "initial" {
		t.Errorf("expected message 'initial', got %q", s.Message())
	}

	// Update before start
	s.Update("before start")
	if s.Message() != "before start" {
		t.Errorf("expected message 'before start', got %q", s.Message())
	}

	// Start and update
	s.Start()
	time.Sleep(20 * time.Millisecond)

	s.Update("during spin")
	if s.Message() != "during spin" {
		t.Errorf("expected message 'during spin', got %q", s.Message())
	}

	s.Stop()

	// Update after stop
	s.Update("after stop")
	if s.Message() != "after stop" {
		t.Errorf("expected message 'after stop', got %q", s.Message())
	}
}

// TestSuccess verifies the Success method stops and displays correctly.
func TestSuccess(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "processing",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: true,
		IsTTY:       boolPtr(true), // Force TTY mode to get color output
	})

	s.Start()
	time.Sleep(50 * time.Millisecond)
	s.Success("completed successfully")

	if s.IsActive() {
		t.Error("spinner should not be active after Success()")
	}

	output := buf.String()
	if !strings.Contains(output, symbolSuccess) {
		t.Error("Success output should contain success symbol")
	}
	if !strings.Contains(output, "completed successfully") {
		t.Error("Success output should contain the message")
	}
	if !strings.Contains(output, colorGreen) {
		t.Error("Success output should contain green color code")
	}
}

// TestSuccessDefaultMessage verifies Success uses current message if empty.
func TestSuccessDefaultMessage(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "original message",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: false,
	})

	s.Start()
	time.Sleep(20 * time.Millisecond)
	s.Success("")

	output := buf.String()
	if !strings.Contains(output, "original message") {
		t.Error("Success with empty message should use original message")
	}
}

// TestFail verifies the Fail method stops and displays correctly.
func TestFail(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "processing",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: true,
		IsTTY:       boolPtr(true), // Force TTY mode to get color output
	})

	s.Start()
	time.Sleep(50 * time.Millisecond)
	s.Fail("operation failed")

	if s.IsActive() {
		t.Error("spinner should not be active after Fail()")
	}

	output := buf.String()
	if !strings.Contains(output, symbolFailure) {
		t.Error("Fail output should contain failure symbol")
	}
	if !strings.Contains(output, "operation failed") {
		t.Error("Fail output should contain the message")
	}
	if !strings.Contains(output, colorRed) {
		t.Error("Fail output should contain red color code")
	}
}

// TestFailDefaultMessage verifies Fail uses current message if empty.
func TestFailDefaultMessage(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "original message",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: false,
	})

	s.Start()
	time.Sleep(20 * time.Millisecond)
	s.Fail("")

	output := buf.String()
	if !strings.Contains(output, "original message") {
		t.Error("Fail with empty message should use original message")
	}
}

// TestSuccessWithoutStart verifies Success works even if spinner wasn't started.
func TestSuccessWithoutStart(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "test",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: false,
	})

	// Should not panic
	s.Success("done")

	output := buf.String()
	if !strings.Contains(output, symbolSuccess) {
		t.Error("Success output should contain success symbol")
	}
	if !strings.Contains(output, "done") {
		t.Error("Success output should contain the message")
	}
}

// TestFailWithoutStart verifies Fail works even if spinner wasn't started.
func TestFailWithoutStart(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "test",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: false,
	})

	// Should not panic
	s.Fail("error")

	output := buf.String()
	if !strings.Contains(output, symbolFailure) {
		t.Error("Fail output should contain failure symbol")
	}
	if !strings.Contains(output, "error") {
		t.Error("Fail output should contain the message")
	}
}

// TestThreadSafetyConcurrentStartStop tests concurrent start/stop operations.
func TestThreadSafetyConcurrentStartStop(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "concurrent test",
		Writer:     &buf,
		HideCursor: false,
	})

	var wg sync.WaitGroup
	iterations := 100

	// Launch goroutines that rapidly start and stop
	for i := 0; i < iterations; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			s.Start()
		}()
		go func() {
			defer wg.Done()
			s.Stop()
		}()
	}

	wg.Wait()

	// Ensure we end in a clean state
	s.Stop()
	if s.IsActive() {
		t.Error("spinner should not be active after all operations complete")
	}
}

// TestThreadSafetyConcurrentUpdate tests concurrent Update operations.
func TestThreadSafetyConcurrentUpdate(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "initial",
		Writer:     &buf,
		HideCursor: false,
	})

	s.Start()
	defer s.Stop()

	var wg sync.WaitGroup
	iterations := 100

	// Launch goroutines that rapidly update the message
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			s.Update(strings.Repeat("x", n%20))
		}(i)
	}

	wg.Wait()

	// Ensure spinner is still in a valid state
	if !s.IsActive() {
		t.Error("spinner should still be active")
	}
}

// TestThreadSafetyConcurrentMixedOperations tests mixed concurrent operations.
func TestThreadSafetyConcurrentMixedOperations(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "mixed test",
		Writer:     &buf,
		HideCursor: false,
	})

	var wg sync.WaitGroup
	iterations := 50

	// Mix of Start, Stop, Update, Message, IsActive, and Elapsed calls
	for i := 0; i < iterations; i++ {
		wg.Add(6)
		go func() {
			defer wg.Done()
			s.Start()
		}()
		go func() {
			defer wg.Done()
			s.Stop()
		}()
		go func() {
			defer wg.Done()
			s.Update("updated")
		}()
		go func() {
			defer wg.Done()
			_ = s.Message()
		}()
		go func() {
			defer wg.Done()
			_ = s.IsActive()
		}()
		go func() {
			defer wg.Done()
			_ = s.Elapsed()
		}()
	}

	wg.Wait()

	// Clean up
	s.Stop()
}

// TestRenderOutput verifies that the spinner produces output with the expected format.
func TestRenderOutput(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		CharSet:     Line, // Use Line for predictable characters
		Message:     "loading",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: false,
		RefreshRate: 20 * time.Millisecond,
	})

	s.Start()
	time.Sleep(50 * time.Millisecond)
	s.Stop()

	output := buf.String()
	if !strings.Contains(output, "loading") {
		t.Errorf("output should contain message 'loading', got: %q", output)
	}
}

// TestRenderWithElapsedTime verifies that elapsed time is shown when configured.
func TestRenderWithElapsedTime(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "working",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: true,
		RefreshRate: 20 * time.Millisecond,
		IsTTY:       boolPtr(true), // Force TTY mode to get animated output with elapsed time
	})

	s.Start()
	time.Sleep(50 * time.Millisecond)
	s.Stop()

	output := buf.String()
	// Should contain elapsed time format like "(0.0s)"
	if !strings.Contains(output, "(") || !strings.Contains(output, "s)") {
		t.Errorf("output should contain elapsed time, got: %q", output)
	}
}

// TestCharSets verifies that all predefined character sets are valid.
func TestCharSets(t *testing.T) {
	charSets := []struct {
		name    string
		charSet CharSet
	}{
		{"Braille", Braille},
		{"Dots", Dots},
		{"Line", Line},
		{"Arc", Arc},
	}

	for _, cs := range charSets {
		t.Run(cs.name, func(t *testing.T) {
			if len(cs.charSet) == 0 {
				t.Errorf("%s CharSet should not be empty", cs.name)
			}
			for i, char := range cs.charSet {
				if char == "" {
					t.Errorf("%s CharSet[%d] should not be empty", cs.name, i)
				}
			}
		})
	}
}

// TestDefaultConfig verifies the default configuration values.
func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if len(cfg.CharSet) != len(Braille) {
		t.Errorf("expected default CharSet to be Braille (len %d), got len %d", len(Braille), len(cfg.CharSet))
	}
	if cfg.Message != "Loading..." {
		t.Errorf("expected default Message 'Loading...', got %q", cfg.Message)
	}
	if cfg.RefreshRate != 80*time.Millisecond {
		t.Errorf("expected default RefreshRate 80ms, got %v", cfg.RefreshRate)
	}
	if !cfg.ShowElapsed {
		t.Error("expected default ShowElapsed to be true")
	}
	if !cfg.HideCursor {
		t.Error("expected default HideCursor to be true")
	}
}

// boolPtr is a helper to get a pointer to a bool value.
func boolPtr(b bool) *bool {
	return &b
}

// TestNonTTYDetection verifies that non-TTY is detected when using bytes.Buffer.
func TestNonTTYDetection(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message: "test",
		Writer:  &buf,
	})

	// bytes.Buffer is not a TTY
	if s.IsTTY() {
		t.Error("bytes.Buffer should not be detected as TTY")
	}
}

// TestNonTTYExplicitConfig verifies that IsTTY can be explicitly configured.
func TestNonTTYExplicitConfig(t *testing.T) {
	var buf bytes.Buffer

	// Force IsTTY to true even though Writer is not a TTY
	s := NewWithConfig(Config{
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	if !s.IsTTY() {
		t.Error("IsTTY should be true when explicitly set")
	}

	// Force IsTTY to false
	s2 := NewWithConfig(Config{
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(false),
	})

	if s2.IsTTY() {
		t.Error("IsTTY should be false when explicitly set")
	}
}

// TestNonTTYStaticOutput verifies that non-TTY mode produces static output.
func TestNonTTYStaticOutput(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:    "processing",
		Writer:     &buf,
		HideCursor: false,
		IsTTY:      boolPtr(false),
	})

	s.Start()
	time.Sleep(50 * time.Millisecond)
	s.Stop()

	output := buf.String()

	// Should contain the message with "..."
	if !strings.Contains(output, "processing...") {
		t.Errorf("non-TTY output should contain static message, got: %q", output)
	}

	// Should NOT contain ANSI escape sequences
	if strings.Contains(output, "\033[") {
		t.Errorf("non-TTY output should not contain ANSI escape sequences, got: %q", output)
	}

	// Should NOT contain spinner characters
	for _, char := range Braille {
		if strings.Contains(output, char) {
			t.Errorf("non-TTY output should not contain spinner characters, got: %q", output)
		}
	}
}

// TestNonTTYSuccessOutput verifies Success output in non-TTY mode.
func TestNonTTYSuccessOutput(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "processing",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: true,
		IsTTY:       boolPtr(false),
	})

	s.Start()
	time.Sleep(50 * time.Millisecond)
	s.Success("completed")

	output := buf.String()

	// Should contain the success symbol and message
	if !strings.Contains(output, symbolSuccess) {
		t.Errorf("non-TTY success output should contain success symbol, got: %q", output)
	}
	if !strings.Contains(output, "completed") {
		t.Errorf("non-TTY success output should contain message, got: %q", output)
	}

	// Should NOT contain color codes
	if strings.Contains(output, colorGreen) || strings.Contains(output, colorReset) {
		t.Errorf("non-TTY success output should not contain color codes, got: %q", output)
	}
}

// TestNonTTYFailOutput verifies Fail output in non-TTY mode.
func TestNonTTYFailOutput(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "processing",
		Writer:      &buf,
		HideCursor:  false,
		ShowElapsed: true,
		IsTTY:       boolPtr(false),
	})

	s.Start()
	time.Sleep(50 * time.Millisecond)
	s.Fail("error occurred")

	output := buf.String()

	// Should contain the failure symbol and message
	if !strings.Contains(output, symbolFailure) {
		t.Errorf("non-TTY fail output should contain failure symbol, got: %q", output)
	}
	if !strings.Contains(output, "error occurred") {
		t.Errorf("non-TTY fail output should contain message, got: %q", output)
	}

	// Should NOT contain color codes
	if strings.Contains(output, colorRed) || strings.Contains(output, colorReset) {
		t.Errorf("non-TTY fail output should not contain color codes, got: %q", output)
	}
}

// TestNonTTYNoAnimation verifies that non-TTY mode doesn't animate.
func TestNonTTYNoAnimation(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "loading",
		Writer:      &buf,
		HideCursor:  false,
		RefreshRate: 20 * time.Millisecond,
		IsTTY:       boolPtr(false),
	})

	s.Start()
	initialOutput := buf.String()

	// Wait for what would be multiple animation frames
	time.Sleep(100 * time.Millisecond)

	finalOutput := buf.String()

	// Output should not change (no animation)
	if initialOutput != finalOutput {
		t.Errorf("non-TTY output should not animate, initial: %q, final: %q", initialOutput, finalOutput)
	}

	s.Stop()
}

// TestNonTTYDoubleStartStop verifies start/stop edge cases in non-TTY mode.
func TestNonTTYDoubleStartStop(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(false),
	})

	// Double start should be safe
	s.Start()
	s.Start()

	// Double stop should be safe
	s.Stop()
	s.Stop()

	if s.IsActive() {
		t.Error("spinner should not be active after stop")
	}
}

// TestNonTTYSuccessWithoutStart verifies Success works in non-TTY mode without Start.
func TestNonTTYSuccessWithoutStart(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "test",
		Writer:      &buf,
		ShowElapsed: false,
		IsTTY:       boolPtr(false),
	})

	// Should not panic
	s.Success("done")

	output := buf.String()
	if !strings.Contains(output, symbolSuccess) {
		t.Error("Success output should contain success symbol")
	}
	if !strings.Contains(output, "done") {
		t.Error("Success output should contain the message")
	}

	// Should NOT contain color codes in non-TTY mode
	if strings.Contains(output, colorGreen) {
		t.Error("non-TTY Success should not contain color codes")
	}
}

// TestNonTTYFailWithoutStart verifies Fail works in non-TTY mode without Start.
func TestNonTTYFailWithoutStart(t *testing.T) {
	var buf bytes.Buffer
	s := NewWithConfig(Config{
		Message:     "test",
		Writer:      &buf,
		ShowElapsed: false,
		IsTTY:       boolPtr(false),
	})

	// Should not panic
	s.Fail("error")

	output := buf.String()
	if !strings.Contains(output, symbolFailure) {
		t.Error("Fail output should contain failure symbol")
	}
	if !strings.Contains(output, "error") {
		t.Error("Fail output should contain the message")
	}

	// Should NOT contain color codes in non-TTY mode
	if strings.Contains(output, colorRed) {
		t.Error("non-TTY Fail should not contain color codes")
	}
}
