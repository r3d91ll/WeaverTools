// Package loom provides a client for TheLoom hidden state extraction server.
// TheLoom is expected to run as a systemd service (see TheLoom/the-loom/the-loom.service).
package loom

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// Manager provides health checking for an external TheLoom server.
// It no longer manages TheLoom as a subprocess - TheLoom should be run
// via systemd: `systemctl start the-loom`
type Manager struct {
	config     Config
	httpClient *http.Client
}

// Config holds TheLoom client configuration.
type Config struct {
	// URL of the TheLoom server (e.g., "http://localhost:8080")
	// If empty, defaults to http://localhost:{Port}
	URL string `yaml:"url"`
	// Port for TheLoom server (used if URL is empty)
	Port int `yaml:"port"`
	// HealthTimeout is how long to wait for health check response
	HealthTimeout time.Duration `yaml:"health_timeout"`
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		Port:          8080,
		HealthTimeout: 5 * time.Second,
	}
}

// NewManager creates a new TheLoom manager.
func NewManager(cfg Config) *Manager {
	if cfg.Port == 0 {
		cfg.Port = 8080
	}
	if cfg.HealthTimeout == 0 {
		cfg.HealthTimeout = 5 * time.Second
	}
	return &Manager{
		config: cfg,
		httpClient: &http.Client{
			Timeout: cfg.HealthTimeout,
		},
	}
}

// URL returns the TheLoom server URL.
func (m *Manager) URL() string {
	if m.config.URL != "" {
		return m.config.URL
	}
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

// EnsureRunning checks if TheLoom is running.
// Returns nil if TheLoom is available, error otherwise.
// TheLoom should be running via systemd: `systemctl start the-loom`
func (m *Manager) EnsureRunning(ctx context.Context) error {
	if m.IsRunning(ctx) {
		return nil
	}
	return fmt.Errorf("TheLoom not running at %s. Start it with: systemctl start the-loom", m.URL())
}

// Stop is a no-op since TheLoom is managed by systemd.
// Kept for interface compatibility.
func (m *Manager) Stop() error {
	return nil
}

// IsManaged returns false since TheLoom is managed by systemd, not this process.
func (m *Manager) IsManaged() bool {
	return false
}
