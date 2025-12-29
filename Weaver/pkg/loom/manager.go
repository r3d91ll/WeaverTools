// Package loom manages TheLoom server as a subprocess.
// It handles starting, health checking, and stopping TheLoom.
package loom

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Manager handles TheLoom server lifecycle.
type Manager struct {
	config     Config
	cmd        *exec.Cmd
	httpClient *http.Client
	mu         sync.Mutex
	running    bool
	stdout     io.ReadCloser
	stderr     io.ReadCloser
}

// Config holds TheLoom manager configuration.
type Config struct {
	// Path to TheLoom directory (containing pyproject.toml)
	Path string `yaml:"path"`
	// Port for TheLoom server
	Port int `yaml:"port"`
	// PreloadModel to load on startup (optional)
	PreloadModel string `yaml:"preload_model"`
	// AutoStart if true, start TheLoom if not running
	AutoStart bool `yaml:"auto_start"`
	// StartupTimeout is how long to wait for TheLoom to be ready
	StartupTimeout time.Duration `yaml:"startup_timeout"`
	// GPUs is a list of GPU device IDs to use (e.g., [0, 1]). Empty = use all available.
	GPUs []int `yaml:"gpus"`
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		Port:           8080,
		AutoStart:      true,
		StartupTimeout: 60 * time.Second,
	}
}

// NewManager creates a new TheLoom manager.
func NewManager(cfg Config) *Manager {
	if cfg.Port == 0 {
		cfg.Port = 8080
	}
	if cfg.StartupTimeout == 0 {
		cfg.StartupTimeout = 60 * time.Second
	}
	return &Manager{
		config: cfg,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

// URL returns the TheLoom server URL.
func (m *Manager) URL() string {
	return fmt.Sprintf("http://localhost:%d", m.config.Port)
}

// IsRunning checks if TheLoom is responding to health checks.
func (m *Manager) IsRunning(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, "GET", m.URL()+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// EnsureRunning checks if TheLoom is running, and starts it if not.
// Returns nil if TheLoom is ready, error otherwise.
func (m *Manager) EnsureRunning(ctx context.Context) error {
	// Already running externally?
	if m.IsRunning(ctx) {
		return nil
	}

	// Auto-start disabled?
	if !m.config.AutoStart {
		return fmt.Errorf("TheLoom not running at %s and auto_start is disabled", m.URL())
	}

	// Need path to start
	if m.config.Path == "" {
		return fmt.Errorf("TheLoom not running and no path configured for auto-start")
	}

	return m.Start(ctx)
}

// Start launches TheLoom as a subprocess.
func (m *Manager) Start(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.running {
		return nil
	}

	// Verify TheLoom directory exists
	pyproject := filepath.Join(m.config.Path, "pyproject.toml")
	if _, err := os.Stat(pyproject); os.IsNotExist(err) {
		return fmt.Errorf("TheLoom not found at %s (missing pyproject.toml)", m.config.Path)
	}

	// Build command arguments
	args := []string{"run", "loom", "--port", fmt.Sprintf("%d", m.config.Port)}
	if m.config.PreloadModel != "" {
		args = append(args, "--preload", m.config.PreloadModel)
	}
	// Add GPU specification if configured
	if len(m.config.GPUs) > 0 {
		gpuStrs := make([]string, len(m.config.GPUs))
		for i, gpu := range m.config.GPUs {
			gpuStrs[i] = strconv.Itoa(gpu)
		}
		args = append(args, "--gpus", strings.Join(gpuStrs, ","))
	}

	// Create command
	m.cmd = exec.CommandContext(ctx, "poetry", args...)
	m.cmd.Dir = m.config.Path

	// Capture output for logging
	var err error
	m.stdout, err = m.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	m.stderr, err = m.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}

	// Start the process
	if err := m.cmd.Start(); err != nil {
		return fmt.Errorf("start TheLoom: %w", err)
	}
	m.running = true

	// Log output in background
	go m.logOutput("loom", m.stdout)
	go m.logOutput("loom", m.stderr)

	// Wait for health check
	if err := m.waitForReady(ctx); err != nil {
		m.Stop() // Clean up on failure
		return err
	}

	return nil
}

// waitForReady polls health endpoint until ready or timeout.
func (m *Manager) waitForReady(ctx context.Context) error {
	deadline := time.Now().Add(m.config.StartupTimeout)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if m.IsRunning(ctx) {
				return nil
			}
			if time.Now().After(deadline) {
				return fmt.Errorf("TheLoom failed to start within %v", m.config.StartupTimeout)
			}
			// Check if process died
			if m.cmd.ProcessState != nil && m.cmd.ProcessState.Exited() {
				return fmt.Errorf("TheLoom process exited unexpectedly")
			}
		}
	}
}

// Stop gracefully shuts down TheLoom.
func (m *Manager) Stop() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.running || m.cmd == nil || m.cmd.Process == nil {
		return nil
	}

	// Send interrupt signal for graceful shutdown
	if err := m.cmd.Process.Signal(os.Interrupt); err != nil {
		// If interrupt fails, force kill
		m.cmd.Process.Kill()
	}

	// Wait for process to exit (with timeout)
	done := make(chan error, 1)
	go func() {
		done <- m.cmd.Wait()
	}()

	select {
	case <-done:
		// Process exited
	case <-time.After(5 * time.Second):
		// Force kill if graceful shutdown takes too long
		m.cmd.Process.Kill()
		<-done
	}

	m.running = false
	m.cmd = nil
	return nil
}

// logOutput reads from reader and logs with prefix.
func (m *Manager) logOutput(prefix string, r io.Reader) {
	buf := make([]byte, 1024)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			// For now, just discard output to avoid noise
			// Could add verbose flag to print: fmt.Printf("[%s] %s", prefix, buf[:n])
			_ = prefix
		}
		if err != nil {
			return
		}
	}
}

// IsManaged returns true if this manager started TheLoom (vs external).
func (m *Manager) IsManaged() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.running
}
