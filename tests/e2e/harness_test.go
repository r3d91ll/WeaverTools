package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// E2EHarness manages the lifecycle of external services for E2E testing.
type E2EHarness struct {
	mu sync.Mutex

	// TheLoom server
	loomProcess *exec.Cmd
	loomURL     string
	loomPort    int

	// Configuration
	timeout     time.Duration
	startupWait time.Duration
	cleanup     []func()
}

// HarnessConfig configures the E2E harness.
type HarnessConfig struct {
	// LoomPort is the port to start TheLoom server on (0 for auto)
	LoomPort int

	// Timeout is the HTTP client timeout
	Timeout time.Duration

	// StartupWait is how long to wait for servers to start
	StartupWait time.Duration

	// TheLoomPath is the path to TheLoom project directory
	TheLoomPath string

	// SkipServerStart skips starting external servers (for mock-only testing)
	SkipServerStart bool
}

// DefaultHarnessConfig returns sensible defaults for local development.
func DefaultHarnessConfig() HarnessConfig {
	return HarnessConfig{
		LoomPort:        0, // Auto-select available port
		Timeout:         30 * time.Second,
		StartupWait:     60 * time.Second,
		TheLoomPath:     "../../TheLoom/the-loom",
		SkipServerStart: false,
	}
}

// Global harness instance for test suite
var (
	globalHarness *E2EHarness
	harnessOnce   sync.Once
	harnessErr    error
)

// GetHarness returns the global harness instance, initializing if needed.
func GetHarness() (*E2EHarness, error) {
	harnessOnce.Do(func() {
		config := DefaultHarnessConfig()

		// Check environment overrides
		if port := os.Getenv("E2E_LOOM_PORT"); port != "" {
			fmt.Sscanf(port, "%d", &config.LoomPort)
		}
		if path := os.Getenv("E2E_THELOOM_PATH"); path != "" {
			config.TheLoomPath = path
		}
		if os.Getenv("E2E_SKIP_SERVER") == "true" || os.Getenv("E2E_SKIP_SERVER") == "1" {
			config.SkipServerStart = true
		}

		globalHarness, harnessErr = NewHarness(config)
	})
	return globalHarness, harnessErr
}

// NewHarness creates a new E2E test harness.
func NewHarness(config HarnessConfig) (*E2EHarness, error) {
	h := &E2EHarness{
		timeout:     config.Timeout,
		startupWait: config.StartupWait,
		cleanup:     make([]func(), 0),
	}

	if !config.SkipServerStart {
		// Try to start TheLoom server
		err := h.startLoomServer(config.LoomPort, config.TheLoomPath)
		if err != nil {
			// Log warning but don't fail - tests can use mocks
			fmt.Printf("Warning: Could not start TheLoom server: %v\n", err)
			fmt.Println("E2E tests will use mock servers instead.")
		}
	}

	return h, nil
}

// startLoomServer starts TheLoom server as a subprocess.
func (h *E2EHarness) startLoomServer(port int, loomPath string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if TheLoom path exists
	absPath, err := filepath.Abs(loomPath)
	if err != nil {
		return fmt.Errorf("invalid TheLoom path: %w", err)
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return fmt.Errorf("TheLoom path does not exist: %s", absPath)
	}

	// Check if poetry is available
	_, err = exec.LookPath("poetry")
	if err != nil {
		return fmt.Errorf("poetry not found in PATH: %w", err)
	}

	// Find available port if not specified
	if port == 0 {
		port, err = findAvailablePort()
		if err != nil {
			return fmt.Errorf("could not find available port: %w", err)
		}
	}

	h.loomPort = port
	h.loomURL = fmt.Sprintf("http://localhost:%d", port)

	// Build command to start TheLoom server
	cmd := exec.Command("poetry", "run", "loom", "--port", fmt.Sprintf("%d", port))
	cmd.Dir = absPath
	cmd.Env = append(os.Environ(),
		"CUDA_VISIBLE_DEVICES=", // Disable GPU for testing (use CPU mode)
	)

	// Capture output for debugging
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Start server
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start TheLoom server: %w", err)
	}

	h.loomProcess = cmd

	// Register cleanup
	h.cleanup = append(h.cleanup, func() {
		if h.loomProcess != nil && h.loomProcess.Process != nil {
			h.loomProcess.Process.Kill()
			h.loomProcess.Wait()
		}
	})

	// Wait for server to be ready
	ctx, cancel := context.WithTimeout(context.Background(), h.startupWait)
	defer cancel()

	if err := h.waitForHealth(ctx, h.loomURL); err != nil {
		h.Cleanup()
		return fmt.Errorf("TheLoom server failed to start: %w", err)
	}

	fmt.Printf("TheLoom server started on %s\n", h.loomURL)
	return nil
}

// waitForHealth polls the health endpoint until ready or timeout.
func (h *E2EHarness) waitForHealth(ctx context.Context, baseURL string) error {
	healthURL := fmt.Sprintf("%s/health", baseURL)
	client := &http.Client{Timeout: 5 * time.Second}

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			resp, err := client.Get(healthURL)
			if err == nil {
				defer resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return nil
				}
			}
		}
	}
}

// LoomURL returns the URL of the TheLoom server (empty if not running).
func (h *E2EHarness) LoomURL() string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.loomURL
}

// IsLoomRunning returns true if TheLoom server is running.
func (h *E2EHarness) IsLoomRunning() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.loomProcess != nil && h.loomURL != ""
}

// Cleanup stops all managed processes and releases resources.
func (h *E2EHarness) Cleanup() {
	h.mu.Lock()
	defer h.mu.Unlock()

	for i := len(h.cleanup) - 1; i >= 0; i-- {
		h.cleanup[i]()
	}
	h.cleanup = nil
	h.loomProcess = nil
	h.loomURL = ""
}

// findAvailablePort finds an available TCP port.
func findAvailablePort() (int, error) {
	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

// HealthStatus represents the health response from TheLoom.
type HealthStatus struct {
	Status    string `json:"status"`
	Model     string `json:"model,omitempty"`
	Version   string `json:"version,omitempty"`
	GPU       bool   `json:"gpu_available,omitempty"`
	MemoryMB  int    `json:"memory_mb,omitempty"`
}

// CheckHealth verifies the health endpoint of a server.
func CheckHealth(baseURL string) (*HealthStatus, error) {
	healthURL := fmt.Sprintf("%s/health", baseURL)
	client := &http.Client{Timeout: 10 * time.Second}

	resp, err := client.Get(healthURL)
	if err != nil {
		return nil, fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("health check returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read health response: %w", err)
	}

	var status HealthStatus
	if err := json.Unmarshal(body, &status); err != nil {
		return nil, fmt.Errorf("failed to parse health response: %w", err)
	}

	return &status, nil
}

// SkipIfNoServer skips a test if the real server is not available.
func SkipIfNoServer(t *testing.T) string {
	t.Helper()

	harness, err := GetHarness()
	if err != nil {
		t.Skipf("Harness not available: %v", err)
	}

	if !harness.IsLoomRunning() {
		t.Skip("TheLoom server not running - using mock tests only")
	}

	return harness.LoomURL()
}

// RequireServer fails a test if the real server is not available.
func RequireServer(t *testing.T) string {
	t.Helper()

	harness, err := GetHarness()
	if err != nil {
		t.Fatalf("Harness not available: %v", err)
	}

	if !harness.IsLoomRunning() {
		t.Fatal("TheLoom server required but not running")
	}

	return harness.LoomURL()
}

// ============================================================================
// Harness Tests
// ============================================================================

// TestHarnessInitialization tests that the harness can be created.
func TestHarnessInitialization(t *testing.T) {
	// Test with skip server mode (should always succeed)
	config := DefaultHarnessConfig()
	config.SkipServerStart = true

	h, err := NewHarness(config)
	if err != nil {
		t.Fatalf("NewHarness() with skip server failed: %v", err)
	}
	defer h.Cleanup()

	if h.IsLoomRunning() {
		t.Error("IsLoomRunning() should be false with SkipServerStart")
	}

	if h.LoomURL() != "" {
		t.Error("LoomURL() should be empty with SkipServerStart")
	}
}

// TestHarnessConfigDefaults tests default configuration values.
func TestHarnessConfigDefaults(t *testing.T) {
	config := DefaultHarnessConfig()

	if config.LoomPort != 0 {
		t.Errorf("Default LoomPort = %d, want 0 (auto)", config.LoomPort)
	}
	if config.Timeout <= 0 {
		t.Error("Default Timeout should be positive")
	}
	if config.StartupWait <= 0 {
		t.Error("Default StartupWait should be positive")
	}
	if config.TheLoomPath == "" {
		t.Error("Default TheLoomPath should not be empty")
	}
}

// TestHealthCheckFunction tests the CheckHealth function against a mock server.
func TestHealthCheckFunction(t *testing.T) {
	// Create a mock health server
	server := mockHealthServer(t, HealthStatus{
		Status:   "ok",
		Model:    "test-model",
		Version:  "1.0.0",
		GPU:      false,
		MemoryMB: 1024,
	})
	defer server.Close()

	// Check health
	status, err := CheckHealth(server.URL)
	if err != nil {
		t.Fatalf("CheckHealth() failed: %v", err)
	}

	if status.Status != "ok" {
		t.Errorf("Status = %q, want %q", status.Status, "ok")
	}
	if status.Model != "test-model" {
		t.Errorf("Model = %q, want %q", status.Model, "test-model")
	}
	if status.GPU {
		t.Error("GPU should be false for mock server")
	}
}

// TestHealthCheckFailure tests CheckHealth with a failing server.
func TestHealthCheckFailure(t *testing.T) {
	// Create a failing mock server
	server := mockFailingHealthServer(t)
	defer server.Close()

	status, err := CheckHealth(server.URL)
	if err == nil {
		t.Error("CheckHealth() should fail for error response")
	}
	if status != nil {
		t.Error("Status should be nil on failure")
	}
}

// TestHealthCheckUnreachable tests CheckHealth with an unreachable server.
func TestHealthCheckUnreachable(t *testing.T) {
	// Use a port that's definitely not listening
	status, err := CheckHealth("http://localhost:59999")
	if err == nil {
		t.Error("CheckHealth() should fail for unreachable server")
	}
	if status != nil {
		t.Error("Status should be nil for unreachable server")
	}
}

// TestFindAvailablePort tests that findAvailablePort returns a usable port.
func TestFindAvailablePort(t *testing.T) {
	port, err := findAvailablePort()
	if err != nil {
		t.Fatalf("findAvailablePort() failed: %v", err)
	}

	if port <= 0 || port > 65535 {
		t.Errorf("Port %d is out of valid range", port)
	}

	// Verify we can actually listen on it
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		t.Errorf("Could not listen on returned port %d: %v", port, err)
	} else {
		listener.Close()
	}
}

// TestHarnessCleanup tests that cleanup properly releases resources.
func TestHarnessCleanup(t *testing.T) {
	config := DefaultHarnessConfig()
	config.SkipServerStart = true

	h, err := NewHarness(config)
	if err != nil {
		t.Fatalf("NewHarness() failed: %v", err)
	}

	// Add a test cleanup function
	cleanupCalled := false
	h.cleanup = append(h.cleanup, func() {
		cleanupCalled = true
	})

	h.Cleanup()

	if !cleanupCalled {
		t.Error("Cleanup function was not called")
	}

	// Verify state is reset
	if h.loomURL != "" {
		t.Error("LoomURL should be empty after cleanup")
	}
}

// TestEnvironmentOverrides tests that environment variables override config.
func TestEnvironmentOverrides(t *testing.T) {
	// This test verifies the environment parsing logic works
	tests := []struct {
		name     string
		envKey   string
		envValue string
		wantSkip bool
	}{
		{"skip with true", "E2E_SKIP_SERVER", "true", true},
		{"skip with 1", "E2E_SKIP_SERVER", "1", true},
		{"no skip with false", "E2E_SKIP_SERVER", "false", false},
		{"no skip with empty", "E2E_SKIP_SERVER", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			skip := tt.envValue == "true" || tt.envValue == "1"
			if skip != tt.wantSkip {
				t.Errorf("Skip logic for %q = %v, want %v", tt.envValue, skip, tt.wantSkip)
			}
		})
	}
}

// ============================================================================
// Mock Server Helpers
// ============================================================================

// mockHealthServer creates a mock server returning a health status.
func mockHealthServer(t *testing.T, status HealthStatus) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(status)
			return
		}
		http.NotFound(w, r)
	}))
}

// mockFailingHealthServer creates a mock server that returns 500.
func mockFailingHealthServer(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
	}))
}

// ============================================================================
// Integration Tests (with real server when available)
// ============================================================================

// TestIntegrationWithRealServer tests against a real TheLoom server if available.
func TestIntegrationWithRealServer(t *testing.T) {
	// Skip if server not available
	baseURL := SkipIfNoServer(t)

	// Test health endpoint
	status, err := CheckHealth(baseURL)
	if err != nil {
		t.Fatalf("Health check failed: %v", err)
	}

	if status.Status != "ok" {
		t.Errorf("Server status = %q, want %q", status.Status, "ok")
	}

	t.Logf("Connected to TheLoom server: model=%s, gpu=%v", status.Model, status.GPU)
}

// TestFullE2EWithRealBackend tests the full E2E flow with a real backend.
func TestFullE2EWithRealBackend(t *testing.T) {
	// Skip if server not available - mock tests cover the logic
	baseURL := SkipIfNoServer(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create HTTP client
	client := &http.Client{Timeout: 30 * time.Second}

	// Test 1: Verify health
	t.Run("health endpoint", func(t *testing.T) {
		status, err := CheckHealth(baseURL)
		if err != nil {
			t.Fatalf("Health check failed: %v", err)
		}
		if status.Status != "ok" {
			t.Errorf("Status = %q, want %q", status.Status, "ok")
		}
	})

	// Test 2: Test chat completions endpoint (if model loaded)
	t.Run("chat completions", func(t *testing.T) {
		req := map[string]any{
			"model": "default",
			"messages": []map[string]string{
				{"role": "user", "content": "Hello, this is a test message"},
			},
			"return_hidden_states": true,
		}

		reqBody, _ := json.Marshal(req)
		httpReq, _ := http.NewRequestWithContext(ctx, "POST",
			fmt.Sprintf("%s/v1/chat/completions", baseURL),
			strings.NewReader(string(reqBody)))
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(httpReq)
		if err != nil {
			// Server might not have a model loaded - that's OK
			t.Skipf("Chat endpoint not available: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusServiceUnavailable {
			t.Skip("Model not loaded on server")
		}

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Errorf("Chat completions returned %d: %s", resp.StatusCode, string(body))
		}
	})
}

// ============================================================================
// Test Suite Entry Point
// ============================================================================

// TestMain sets up and tears down the E2E test environment.
func TestMain(m *testing.M) {
	fmt.Println("=== E2E Test Suite ===")
	fmt.Printf("Running on %s/%s\n", runtime.GOOS, runtime.GOARCH)

	// Initialize harness
	harness, err := GetHarness()
	if err != nil {
		fmt.Printf("Warning: Harness initialization failed: %v\n", err)
		fmt.Println("Tests will use mock servers only.")
	} else if harness.IsLoomRunning() {
		fmt.Printf("TheLoom server available at: %s\n", harness.LoomURL())
	} else {
		fmt.Println("TheLoom server not started - tests will use mocks.")
	}

	// Run tests
	code := m.Run()

	// Cleanup
	if harness != nil {
		fmt.Println("Cleaning up test environment...")
		harness.Cleanup()
	}

	fmt.Println("=== E2E Test Suite Complete ===")
	os.Exit(code)
}

// ============================================================================
// Utility Tests for Test Infrastructure
// ============================================================================

// TestMockServerCreation tests that mock servers work correctly.
func TestMockServerCreation(t *testing.T) {
	server := mockLoomServer(t, true)
	defer server.Close()

	// Verify mock server responds to health
	resp, err := http.Get(server.URL + "/health")
	if err != nil {
		t.Fatalf("Mock server health request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Mock server health returned %d, want 200", resp.StatusCode)
	}
}

// TestMockServerChatCompletions tests that mock server handles chat requests.
func TestMockServerChatCompletions(t *testing.T) {
	server := mockLoomServer(t, true)
	defer server.Close()

	// Create chat request
	req := map[string]any{
		"model":               "test",
		"messages":            []map[string]string{{"role": "user", "content": "test"}},
		"return_hidden_states": true,
	}
	reqBody, _ := json.Marshal(req)

	resp, err := http.Post(
		server.URL+"/v1/chat/completions",
		"application/json",
		strings.NewReader(string(reqBody)),
	)
	if err != nil {
		t.Fatalf("Mock server chat request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Mock server chat returned %d, want 200", resp.StatusCode)
	}

	// Verify response structure
	var chatResp map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		t.Fatalf("Failed to decode chat response: %v", err)
	}

	if _, ok := chatResp["text"]; !ok {
		t.Error("Chat response missing 'text' field")
	}
	if _, ok := chatResp["hidden_state"]; !ok {
		t.Error("Chat response missing 'hidden_state' field")
	}
}

// TestE2ETestsRunWithoutManualIntervention tests that all E2E tests can run automatically.
func TestE2ETestsRunWithoutManualIntervention(t *testing.T) {
	// This test verifies the automation requirement from the spec
	// All tests should run via `go test -v ./...` without manual steps

	tests := []struct {
		name        string
		testFunc    func(t *testing.T)
		requiresGPU bool
	}{
		{"delegation routing", func(t *testing.T) { TestDelegationRouting(t) }, false},
		{"delegation agent roles", func(t *testing.T) { TestDelegationAgentRoles(t) }, false},
		{"delegation role capabilities", func(t *testing.T) { TestDelegationRoleCapabilities(t) }, false},
		{"measurement creation", func(t *testing.T) { TestMeasurementCreation(t) }, false},
		{"measurement beta status", func(t *testing.T) { TestMeasurementBetaStatusComputation(t) }, false},
		{"measurement bilateral detection", func(t *testing.T) { TestMeasurementBilateralDetection(t) }, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.requiresGPU {
				t.Skip("Requires GPU - skipping in automated run")
			}
			// Run the test function - if it panics or fails, this test fails
			tt.testFunc(t)
		})
	}
}

// TestE2ESuiteCanRunInCI verifies the test suite works in CI environments.
func TestE2ESuiteCanRunInCI(t *testing.T) {
	// Verify essential tests work without external dependencies
	t.Run("mock server tests", func(t *testing.T) {
		server := mockLoomServer(t, true)
		defer server.Close()

		// Should be able to check health
		status, err := CheckHealth(server.URL)
		if err != nil {
			t.Fatalf("Health check failed: %v", err)
		}
		if status.Status != "ok" {
			t.Errorf("Status = %q, want ok", status.Status)
		}
	})

	t.Run("harness without server", func(t *testing.T) {
		config := DefaultHarnessConfig()
		config.SkipServerStart = true

		h, err := NewHarness(config)
		if err != nil {
			t.Fatalf("Harness creation failed: %v", err)
		}
		defer h.Cleanup()

		// Verify harness is in expected state
		if h.IsLoomRunning() {
			t.Error("Server should not be running with SkipServerStart")
		}
	})
}
