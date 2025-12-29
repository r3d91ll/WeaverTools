package spinner

import (
	"bytes"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestNewProgress verifies that NewProgress creates a progress bar with default configuration.
func TestNewProgress(t *testing.T) {
	p := NewProgress(50, "test message")
	if p == nil {
		t.Fatal("NewProgress returned nil")
	}
	if p.Message() != "test message" {
		t.Errorf("expected message 'test message', got %q", p.Message())
	}
	if p.Total() != 50 {
		t.Errorf("expected total 50, got %d", p.Total())
	}
	if p.Current() != 0 {
		t.Errorf("expected current 0, got %d", p.Current())
	}
	if p.IsActive() {
		t.Error("progress bar should not be active before Start()")
	}
}

// TestNewProgressWithConfig verifies that NewProgressWithConfig applies custom configuration.
func TestNewProgressWithConfig(t *testing.T) {
	var buf bytes.Buffer
	cfg := ProgressConfig{
		Total:            100,
		Message:          "custom message",
		Width:            30,
		ShowPercentage:   true,
		ShowCount:        true,
		ShowElapsed:      true,
		ShowETA:          true,
		MinSamplesForETA: 3,
		Writer:           &buf,
	}

	p := NewProgressWithConfig(cfg)
	if p == nil {
		t.Fatal("NewProgressWithConfig returned nil")
	}
	if p.Message() != "custom message" {
		t.Errorf("expected message 'custom message', got %q", p.Message())
	}
	if p.Total() != 100 {
		t.Errorf("expected total 100, got %d", p.Total())
	}

	config := p.Config()
	if config.Width != 30 {
		t.Errorf("expected width 30, got %d", config.Width)
	}
	if config.MinSamplesForETA != 3 {
		t.Errorf("expected MinSamplesForETA 3, got %d", config.MinSamplesForETA)
	}
}

// TestNewProgressWithConfigDefaults verifies that missing config values get defaults.
func TestNewProgressWithConfigDefaults(t *testing.T) {
	cfg := ProgressConfig{Message: "test"}
	p := NewProgressWithConfig(cfg)

	config := p.Config()

	// Total should default to 100
	if config.Total != 100 {
		t.Errorf("expected Total to default to 100, got %d", config.Total)
	}

	// Width should default to 20
	if config.Width != 20 {
		t.Errorf("expected Width to default to 20, got %d", config.Width)
	}

	// MinSamplesForETA should default to 2
	if config.MinSamplesForETA != 2 {
		t.Errorf("expected MinSamplesForETA to default to 2, got %d", config.MinSamplesForETA)
	}

	// Writer should default to os.Stderr (not nil)
	if config.Writer == nil {
		t.Error("expected Writer to default to os.Stderr, got nil")
	}
}

// TestDefaultProgressConfig verifies the default configuration values.
func TestDefaultProgressConfig(t *testing.T) {
	cfg := DefaultProgressConfig()

	if cfg.Total != 100 {
		t.Errorf("expected default Total 100, got %d", cfg.Total)
	}
	if cfg.Message != "Processing..." {
		t.Errorf("expected default Message 'Processing...', got %q", cfg.Message)
	}
	if cfg.Width != 20 {
		t.Errorf("expected default Width 20, got %d", cfg.Width)
	}
	if !cfg.ShowPercentage {
		t.Error("expected default ShowPercentage to be true")
	}
	if !cfg.ShowCount {
		t.Error("expected default ShowCount to be true")
	}
	if !cfg.ShowElapsed {
		t.Error("expected default ShowElapsed to be true")
	}
	if !cfg.ShowETA {
		t.Error("expected default ShowETA to be true")
	}
	if cfg.MinSamplesForETA != 2 {
		t.Errorf("expected default MinSamplesForETA 2, got %d", cfg.MinSamplesForETA)
	}
}

// TestBasicStartIncrementComplete verifies basic start, increment, and complete functionality.
func TestBasicStartIncrementComplete(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "processing",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	// Should not be active before start
	if p.IsActive() {
		t.Error("progress bar should not be active before Start()")
	}

	// Start the progress bar
	p.Start()

	if !p.IsActive() {
		t.Error("progress bar should be active after Start()")
	}
	if p.Current() != 0 {
		t.Errorf("expected current 0 after start, got %d", p.Current())
	}

	// Increment several times
	for i := 0; i < 5; i++ {
		p.Increment()
	}

	if p.Current() != 5 {
		t.Errorf("expected current 5 after 5 increments, got %d", p.Current())
	}

	// Complete the progress bar
	p.Complete("done")

	if p.IsActive() {
		t.Error("progress bar should not be active after Complete()")
	}
}

// TestSetProgress verifies the Set method.
func TestSetProgress(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   100,
		Message: "loading",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()

	// Set to 50
	p.Set(50)
	if p.Current() != 50 {
		t.Errorf("expected current 50, got %d", p.Current())
	}

	// Set to 75
	p.Set(75)
	if p.Current() != 75 {
		t.Errorf("expected current 75, got %d", p.Current())
	}

	p.Complete("done")
}

// TestDoubleStart verifies that double-start is a no-op.
func TestProgressDoubleStart(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "loading",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()

	// Increment a few times
	p.Increment()
	p.Increment()

	// Double start should be a no-op - should not reset current
	p.Start()

	if p.Current() != 2 {
		t.Errorf("double-start should be a no-op, expected current 2, got %d", p.Current())
	}

	p.Complete("done")
}

// TestCompleteBeforeStart verifies that completing before starting is safe.
func TestCompleteBeforeStart(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:       10,
		Message:     "test",
		Writer:      &buf,
		ShowElapsed: false,
		IsTTY:       boolPtr(true),
	})

	// This should not panic
	p.Complete("done")

	output := buf.String()
	if !strings.Contains(output, symbolSuccess) {
		t.Error("Complete output should contain success symbol")
	}
	if !strings.Contains(output, "done") {
		t.Error("Complete output should contain the message")
	}
}

// TestFailBeforeStart verifies that failing before starting is safe.
func TestFailBeforeStart(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:       10,
		Message:     "test",
		Writer:      &buf,
		ShowElapsed: false,
		IsTTY:       boolPtr(true),
	})

	// This should not panic
	p.Fail("error")

	output := buf.String()
	if !strings.Contains(output, symbolFailure) {
		t.Error("Fail output should contain failure symbol")
	}
	if !strings.Contains(output, "error") {
		t.Error("Fail output should contain the message")
	}
}

// TestIncrementBeforeStart verifies that incrementing before starting does nothing.
func TestIncrementBeforeStart(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	// Increment before start should be a no-op
	p.Increment()
	p.Increment()

	if p.Current() != 0 {
		t.Errorf("expected current 0 before start, got %d", p.Current())
	}
}

// TestSetBeforeStart verifies that setting before starting does nothing.
func TestSetBeforeStart(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	// Set before start should be a no-op
	p.Set(5)

	if p.Current() != 0 {
		t.Errorf("expected current 0 before start, got %d", p.Current())
	}
}

// TestIncrementPastTotal verifies that incrementing past total clamps to total.
func TestIncrementPastTotal(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   5,
		Message: "loading",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()

	// Increment 10 times (past total of 5)
	for i := 0; i < 10; i++ {
		p.Increment()
	}

	if p.Current() != 5 {
		t.Errorf("expected current to be clamped to total 5, got %d", p.Current())
	}

	p.Complete("done")
}

// TestSetPastTotal verifies that setting past total clamps to total.
func TestSetPastTotal(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "loading",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()

	// Set past total
	p.Set(100)

	if p.Current() != 10 {
		t.Errorf("expected current to be clamped to total 10, got %d", p.Current())
	}

	p.Complete("done")
}

// TestSetNegative verifies that setting negative value clamps to 0.
func TestSetNegative(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "loading",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()
	p.Set(5)

	// Set negative
	p.Set(-10)

	if p.Current() != 0 {
		t.Errorf("expected current to be clamped to 0, got %d", p.Current())
	}

	p.Complete("done")
}

// TestPercentageCalculation verifies that percentage is calculated correctly.
func TestPercentageCalculation(t *testing.T) {
	p := NewProgress(100, "test")

	// Before start, should be 0%
	if p.Percentage() != 0 {
		t.Errorf("expected 0%% before start, got %.2f%%", p.Percentage())
	}

	p.Start()

	// At 0/100, should be 0%
	if p.Percentage() != 0 {
		t.Errorf("expected 0%% at 0/100, got %.2f%%", p.Percentage())
	}

	// At 50/100, should be 50%
	p.Set(50)
	if p.Percentage() != 50 {
		t.Errorf("expected 50%% at 50/100, got %.2f%%", p.Percentage())
	}

	// At 100/100, should be 100%
	p.Set(100)
	if p.Percentage() != 100 {
		t.Errorf("expected 100%% at 100/100, got %.2f%%", p.Percentage())
	}

	p.Complete("")
}

// TestElapsedTimeTracking verifies that elapsed time is tracked correctly.
func TestProgressElapsedTimeTracking(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "loading",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	// Elapsed should be 0 before start
	if p.Elapsed() != 0 {
		t.Errorf("expected elapsed to be 0 before start, got %v", p.Elapsed())
	}

	p.Start()
	time.Sleep(100 * time.Millisecond)

	elapsed := p.Elapsed()
	if elapsed < 100*time.Millisecond {
		t.Errorf("expected elapsed >= 100ms, got %v", elapsed)
	}

	p.Complete("done")
}

// TestSuccessOutput verifies the Complete method stops and displays correctly.
func TestSuccessOutput(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:       10,
		Message:     "processing",
		Writer:      &buf,
		ShowElapsed: true,
		IsTTY:       boolPtr(true),
	})

	p.Start()
	time.Sleep(50 * time.Millisecond)
	p.Complete("completed successfully")

	if p.IsActive() {
		t.Error("progress bar should not be active after Complete()")
	}

	output := buf.String()
	if !strings.Contains(output, symbolSuccess) {
		t.Error("Complete output should contain success symbol")
	}
	if !strings.Contains(output, "completed successfully") {
		t.Error("Complete output should contain the message")
	}
	if !strings.Contains(output, colorGreen) {
		t.Error("Complete output should contain green color code")
	}
}

// TestCompleteDefaultMessage verifies Complete uses message from config if empty.
func TestCompleteDefaultMessage(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:       10,
		Message:     "original message",
		Writer:      &buf,
		ShowElapsed: false,
		IsTTY:       boolPtr(true),
	})

	p.Start()
	time.Sleep(20 * time.Millisecond)
	p.Complete("")

	output := buf.String()
	if !strings.Contains(output, "original message complete") {
		t.Errorf("Complete with empty message should use 'original message complete', got: %q", output)
	}
}

// TestFailOutput verifies the Fail method stops and displays correctly.
func TestFailOutput(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:       10,
		Message:     "processing",
		Writer:      &buf,
		ShowElapsed: true,
		IsTTY:       boolPtr(true),
	})

	p.Start()
	time.Sleep(50 * time.Millisecond)
	p.Fail("operation failed")

	if p.IsActive() {
		t.Error("progress bar should not be active after Fail()")
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

// TestFailDefaultMessage verifies Fail uses message from config if empty.
func TestFailDefaultMessage(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:       10,
		Message:     "original message",
		Writer:      &buf,
		ShowElapsed: false,
		IsTTY:       boolPtr(true),
	})

	p.Start()
	time.Sleep(20 * time.Millisecond)
	p.Fail("")

	output := buf.String()
	if !strings.Contains(output, "original message complete") {
		t.Errorf("Fail with empty message should use 'original message complete', got: %q", output)
	}
}

// TestNonTTYDetection verifies that non-TTY is detected when using bytes.Buffer.
func TestProgressNonTTYDetection(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "test",
		Writer:  &buf,
	})

	// bytes.Buffer is not a TTY
	if p.IsTTY() {
		t.Error("bytes.Buffer should not be detected as TTY")
	}
}

// TestNonTTYExplicitConfig verifies that IsTTY can be explicitly configured.
func TestProgressNonTTYExplicitConfig(t *testing.T) {
	var buf bytes.Buffer

	// Force IsTTY to true even though Writer is not a TTY
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	if !p.IsTTY() {
		t.Error("IsTTY should be true when explicitly set")
	}

	// Force IsTTY to false
	p2 := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(false),
	})

	if p2.IsTTY() {
		t.Error("IsTTY should be false when explicitly set")
	}
}

// TestNonTTYOutput verifies that non-TTY mode produces appropriate output.
func TestNonTTYOutput(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:          10,
		Message:        "processing",
		Writer:         &buf,
		ShowPercentage: true,
		ShowCount:      true,
		ShowElapsed:    false,
		IsTTY:          boolPtr(false),
	})

	p.Start()

	// Increment all the way to trigger non-TTY output
	for i := 0; i < 10; i++ {
		p.Increment()
	}

	p.Complete("done")

	output := buf.String()

	// Should contain the message
	if !strings.Contains(output, "processing") {
		t.Errorf("non-TTY output should contain message, got: %q", output)
	}

	// Should contain progress bar characters
	if !strings.Contains(output, barFilled) && !strings.Contains(output, barEmpty) {
		t.Errorf("non-TTY output should contain progress bar characters, got: %q", output)
	}

	// Should NOT contain carriage return (which would be used for inline updates in TTY mode)
	// But in non-TTY mode, we use newlines instead
	if strings.Contains(output, "\r") {
		t.Errorf("non-TTY output should not contain carriage return, got: %q", output)
	}
}

// TestNonTTYSuccessOutput verifies Success output in non-TTY mode.
func TestNonTTYSuccessOutput(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:       10,
		Message:     "processing",
		Writer:      &buf,
		ShowElapsed: true,
		IsTTY:       boolPtr(false),
	})

	p.Start()
	time.Sleep(50 * time.Millisecond)
	p.Complete("completed")

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
	p := NewProgressWithConfig(ProgressConfig{
		Total:       10,
		Message:     "processing",
		Writer:      &buf,
		ShowElapsed: true,
		IsTTY:       boolPtr(false),
	})

	p.Start()
	time.Sleep(50 * time.Millisecond)
	p.Fail("error occurred")

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

// TestThreadSafetyConcurrentIncrement tests concurrent Increment calls.
func TestThreadSafetyConcurrentIncrement(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   1000,
		Message: "concurrent test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()

	var wg sync.WaitGroup
	goroutines := 100
	incrementsPerGoroutine := 10

	// Launch goroutines that concurrently increment
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < incrementsPerGoroutine; j++ {
				p.Increment()
			}
		}()
	}

	wg.Wait()

	// Expected: 100 * 10 = 1000 increments (clamped to total of 1000)
	expected := goroutines * incrementsPerGoroutine
	if expected > 1000 {
		expected = 1000
	}

	if p.Current() != expected {
		t.Errorf("expected current %d after concurrent increments, got %d", expected, p.Current())
	}

	p.Complete("done")
}

// TestThreadSafetyConcurrentSet tests concurrent Set calls.
func TestThreadSafetyConcurrentSet(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   100,
		Message: "concurrent set test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()

	var wg sync.WaitGroup
	iterations := 100

	// Launch goroutines that concurrently set values
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			p.Set(n)
		}(i)
	}

	wg.Wait()

	// Final value should be within valid range
	current := p.Current()
	if current < 0 || current > 100 {
		t.Errorf("current should be in range [0, 100], got %d", current)
	}

	p.Complete("done")
}

// TestThreadSafetyConcurrentMixedOperations tests mixed concurrent operations.
func TestThreadSafetyConcurrentMixedOperations(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   100,
		Message: "mixed test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()

	var wg sync.WaitGroup
	iterations := 50

	// Mix of Increment, Set, Current, Percentage, IsActive, Elapsed calls
	for i := 0; i < iterations; i++ {
		wg.Add(6)
		go func() {
			defer wg.Done()
			p.Increment()
		}()
		go func(n int) {
			defer wg.Done()
			p.Set(n)
		}(i)
		go func() {
			defer wg.Done()
			_ = p.Current()
		}()
		go func() {
			defer wg.Done()
			_ = p.Percentage()
		}()
		go func() {
			defer wg.Done()
			_ = p.IsActive()
		}()
		go func() {
			defer wg.Done()
			_ = p.Elapsed()
		}()
	}

	wg.Wait()

	// Should still be in a valid state
	if !p.IsActive() {
		t.Error("progress bar should still be active")
	}

	p.Complete("done")
}

// TestETACalculation verifies ETA calculation accuracy.
func TestETACalculation(t *testing.T) {
	// Create a progress bar with known timing
	p := &ProgressBar{
		config: ProgressConfig{
			Total:            10,
			MinSamplesForETA: 2,
		},
		current:   5, // 50% complete
		startTime: time.Now().Add(-10 * time.Second), // Started 10 seconds ago
		active:    true,
	}

	// With 5 items completed in 10 seconds, avg is 2 seconds per item
	// With 5 remaining, ETA should be ~10 seconds
	eta := p.calculateETA()

	// Allow some tolerance for timing variations
	expectedETA := 10 * time.Second
	tolerance := 500 * time.Millisecond

	if eta < expectedETA-tolerance || eta > expectedETA+tolerance {
		t.Errorf("expected ETA ~%v, got %v", expectedETA, eta)
	}
}

// TestETANotShownBeforeMinSamples verifies ETA is not shown before MinSamplesForETA.
func TestETANotShownBeforeMinSamples(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:            10,
		Message:          "loading",
		Writer:           &buf,
		ShowETA:          true,
		MinSamplesForETA: 3,
		IsTTY:            boolPtr(true),
	})

	p.Start()

	// After 1 sample
	p.Increment()
	output1 := buf.String()
	if strings.Contains(output1, "ETA:") {
		t.Error("ETA should not be shown after only 1 sample (MinSamplesForETA=3)")
	}

	// After 2 samples
	p.Increment()
	output2 := buf.String()
	if strings.Contains(output2, "ETA:") {
		t.Error("ETA should not be shown after only 2 samples (MinSamplesForETA=3)")
	}

	p.Complete("done")
}

// TestETAShownAfterMinSamples verifies ETA is shown after MinSamplesForETA.
func TestETAShownAfterMinSamples(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:            10,
		Message:          "loading",
		Writer:           &buf,
		ShowETA:          true,
		MinSamplesForETA: 2,
		IsTTY:            boolPtr(true),
	})

	p.Start()

	// Wait a bit to ensure elapsed time is measured
	time.Sleep(20 * time.Millisecond)

	// Increment past MinSamplesForETA
	for i := 0; i < 5; i++ {
		p.Increment()
	}

	output := buf.String()
	// ETA should be shown after 5 samples (>= MinSamplesForETA=2)
	if !strings.Contains(output, "ETA:") {
		t.Errorf("ETA should be shown after reaching MinSamplesForETA, got: %q", output)
	}

	p.Complete("done")
}

// TestETAZeroAtCompletion verifies ETA is 0 when progress is complete.
func TestETAZeroAtCompletion(t *testing.T) {
	p := &ProgressBar{
		config: ProgressConfig{
			Total:            10,
			MinSamplesForETA: 2,
		},
		current:   10, // 100% complete
		startTime: time.Now().Add(-10 * time.Second),
		active:    true,
	}

	eta := p.calculateETA()
	if eta != 0 {
		t.Errorf("expected ETA 0 when complete, got %v", eta)
	}
}

// TestETAZeroBeforeStart verifies ETA is 0 when progress hasn't started.
func TestETAZeroBeforeStart(t *testing.T) {
	p := &ProgressBar{
		config: ProgressConfig{
			Total:            10,
			MinSamplesForETA: 2,
		},
		current: 0,
		active:  true,
	}

	eta := p.calculateETA()
	if eta != 0 {
		t.Errorf("expected ETA 0 when no progress, got %v", eta)
	}
}

// TestFormatETAShortDuration verifies ETA formatting for seconds.
func TestFormatETAShortDuration(t *testing.T) {
	p := &ProgressBar{}

	tests := []struct {
		duration time.Duration
		expected string
	}{
		{500 * time.Millisecond, "ETA: 1s"},
		{1 * time.Second, "ETA: 1s"},
		{30 * time.Second, "ETA: 30s"},
		{59 * time.Second, "ETA: 59s"},
	}

	for _, tc := range tests {
		result := p.formatETA(tc.duration)
		if result != tc.expected {
			t.Errorf("formatETA(%v): expected %q, got %q", tc.duration, tc.expected, result)
		}
	}
}

// TestFormatETAMinutes verifies ETA formatting for minutes.
func TestFormatETAMinutes(t *testing.T) {
	p := &ProgressBar{}

	tests := []struct {
		duration time.Duration
		expected string
	}{
		{60 * time.Second, "ETA: 1m"},
		{90 * time.Second, "ETA: 1m 30s"},
		{5 * time.Minute, "ETA: 5m"},
		{5*time.Minute + 30*time.Second, "ETA: 5m 30s"},
	}

	for _, tc := range tests {
		result := p.formatETA(tc.duration)
		if result != tc.expected {
			t.Errorf("formatETA(%v): expected %q, got %q", tc.duration, tc.expected, result)
		}
	}
}

// TestFormatETAHours verifies ETA formatting for hours.
func TestFormatETAHours(t *testing.T) {
	p := &ProgressBar{}

	tests := []struct {
		duration time.Duration
		expected string
	}{
		{1 * time.Hour, "ETA: 1h"},
		{1*time.Hour + 30*time.Minute, "ETA: 1h 30m"},
		{2 * time.Hour, "ETA: 2h"},
		{2*time.Hour + 45*time.Minute, "ETA: 2h 45m"},
	}

	for _, tc := range tests {
		result := p.formatETA(tc.duration)
		if result != tc.expected {
			t.Errorf("formatETA(%v): expected %q, got %q", tc.duration, tc.expected, result)
		}
	}
}

// TestBuildBarEmptyProgress verifies the progress bar display at 0%.
func TestBuildBarEmptyProgress(t *testing.T) {
	p := &ProgressBar{
		config: ProgressConfig{
			Total: 10,
			Width: 10,
		},
		current: 0,
	}

	bar := p.buildBar()

	// Should have opening bracket, 10 empty chars, closing bracket
	expectedEmpty := strings.Repeat(barEmpty, 10)
	expected := "[" + expectedEmpty + "]"

	if bar != expected {
		t.Errorf("expected bar %q, got %q", expected, bar)
	}
}

// TestBuildBarFullProgress verifies the progress bar display at 100%.
func TestBuildBarFullProgress(t *testing.T) {
	p := &ProgressBar{
		config: ProgressConfig{
			Total: 10,
			Width: 10,
		},
		current: 10,
	}

	bar := p.buildBar()

	// Should have opening bracket, 10 filled chars, closing bracket
	expectedFilled := strings.Repeat(barFilled, 10)
	expected := "[" + expectedFilled + "]"

	if bar != expected {
		t.Errorf("expected bar %q, got %q", expected, bar)
	}
}

// TestBuildBarHalfProgress verifies the progress bar display at 50%.
func TestBuildBarHalfProgress(t *testing.T) {
	p := &ProgressBar{
		config: ProgressConfig{
			Total: 10,
			Width: 10,
		},
		current: 5,
	}

	bar := p.buildBar()

	// Should have 5 filled and 5 empty
	expected := "[" + strings.Repeat(barFilled, 5) + strings.Repeat(barEmpty, 5) + "]"

	if bar != expected {
		t.Errorf("expected bar %q, got %q", expected, bar)
	}
}

// TestBuildOutput verifies the complete output string format.
func TestBuildOutput(t *testing.T) {
	p := &ProgressBar{
		config: ProgressConfig{
			Total:          10,
			Width:          10,
			Message:        "Loading",
			ShowPercentage: true,
			ShowCount:      true,
			ShowElapsed:    false,
			ShowETA:        false,
		},
		current: 5,
	}

	output := p.buildOutput()

	// Should contain message
	if !strings.Contains(output, "Loading") {
		t.Errorf("output should contain message, got: %q", output)
	}

	// Should contain percentage
	if !strings.Contains(output, "50%") {
		t.Errorf("output should contain '50%%', got: %q", output)
	}

	// Should contain count
	if !strings.Contains(output, "(5/10)") {
		t.Errorf("output should contain '(5/10)', got: %q", output)
	}

	// Should contain progress bar
	if !strings.Contains(output, "[") || !strings.Contains(output, "]") {
		t.Errorf("output should contain progress bar brackets, got: %q", output)
	}
}

// TestProgressFormatElapsed verifies elapsed time formatting for progress bar.
func TestProgressFormatElapsed(t *testing.T) {
	p := &ProgressBar{}

	tests := []struct {
		duration time.Duration
		expected string
	}{
		{500 * time.Millisecond, "(0.5s)"},
		{1 * time.Second, "(1.0s)"},
		{30 * time.Second, "(30.0s)"},
		{60 * time.Second, "(1m 0s)"},
		{90 * time.Second, "(1m 30s)"},
		{5*time.Minute + 30*time.Second, "(5m 30s)"},
	}

	for _, tc := range tests {
		result := p.formatElapsed(tc.duration)
		if result != tc.expected {
			t.Errorf("formatElapsed(%v): expected %q, got %q", tc.duration, tc.expected, result)
		}
	}
}

// TestProgressBarWithZeroTotal verifies behavior with zero total (edge case).
func TestProgressBarWithZeroTotal(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   0, // Zero total should be defaulted to 100
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	// Total should have been defaulted to 100
	if p.Total() != 100 {
		t.Errorf("expected Total to default to 100 when 0 is provided, got %d", p.Total())
	}

	// Percentage should be 0 at start
	if p.Percentage() != 0 {
		t.Errorf("expected 0%% at start, got %.2f%%", p.Percentage())
	}
}

// TestProgressBarWithNegativeTotal verifies behavior with negative total (edge case).
func TestProgressBarWithNegativeTotal(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   -10, // Negative total should be defaulted to 100
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	// Total should have been defaulted to 100
	if p.Total() != 100 {
		t.Errorf("expected Total to default to 100 when negative is provided, got %d", p.Total())
	}
}

// TestIncrementAfterComplete verifies increment after complete is a no-op.
func TestIncrementAfterComplete(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()
	p.Set(5)
	p.Complete("done")

	// Increment after complete should be a no-op (not active)
	oldCurrent := p.Current()
	p.Increment()

	if p.Current() != oldCurrent {
		t.Errorf("increment after complete should be no-op, expected %d, got %d", oldCurrent, p.Current())
	}
}

// TestSetAfterComplete verifies set after complete is a no-op.
func TestSetAfterComplete(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgressWithConfig(ProgressConfig{
		Total:   10,
		Message: "test",
		Writer:  &buf,
		IsTTY:   boolPtr(true),
	})

	p.Start()
	p.Set(5)
	p.Complete("done")

	// Set after complete should be a no-op (not active)
	oldCurrent := p.Current()
	p.Set(8)

	if p.Current() != oldCurrent {
		t.Errorf("set after complete should be no-op, expected %d, got %d", oldCurrent, p.Current())
	}
}
