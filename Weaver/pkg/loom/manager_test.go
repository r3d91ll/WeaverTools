// Package loom provides a client for TheLoom hidden state extraction server.
package loom

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNewManager(t *testing.T) {
	tests := []struct {
		name     string
		cfg      Config
		wantPort int
	}{
		{
			name:     "default port when zero",
			cfg:      Config{},
			wantPort: 8080,
		},
		{
			name:     "custom port",
			cfg:      Config{Port: 9000},
			wantPort: 9000,
		},
		{
			name:     "with URL",
			cfg:      Config{URL: "http://localhost:8888", Port: 0},
			wantPort: 8080, // Port still gets default even with URL
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewManager(tt.cfg)
			if m == nil {
				t.Fatal("NewManager returned nil")
			}
			if m.config.Port != tt.wantPort {
				t.Errorf("Port = %d, want %d", m.config.Port, tt.wantPort)
			}
			if m.httpClient == nil {
				t.Error("httpClient should not be nil")
			}
		})
	}
}

func TestManager_URL(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantURL string
	}{
		{
			name:    "uses URL when provided",
			cfg:     Config{URL: "http://custom:9000"},
			wantURL: "http://custom:9000",
		},
		{
			name:    "falls back to port when URL empty",
			cfg:     Config{Port: 8080},
			wantURL: "http://localhost:8080",
		},
		{
			name:    "default port fallback",
			cfg:     Config{},
			wantURL: "http://localhost:8080",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewManager(tt.cfg)
			if got := m.URL(); got != tt.wantURL {
				t.Errorf("URL() = %q, want %q", got, tt.wantURL)
			}
		})
	}
}

func TestManager_IsRunning(t *testing.T) {
	// Create a test server that returns 200 OK
	healthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer healthyServer.Close()

	// Create a test server that returns 500
	unhealthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer unhealthyServer.Close()

	tests := []struct {
		name    string
		url     string
		want    bool
	}{
		{
			name: "healthy server",
			url:  healthyServer.URL,
			want: true,
		},
		{
			name: "unhealthy server",
			url:  unhealthyServer.URL,
			want: false,
		},
		{
			name: "no server",
			url:  "http://localhost:59999", // Unlikely to be running
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewManager(Config{
				URL:           tt.url,
				HealthTimeout: 1 * time.Second,
			})
			ctx := context.Background()
			if got := m.IsRunning(ctx); got != tt.want {
				t.Errorf("IsRunning() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestManager_EnsureRunning(t *testing.T) {
	// Create a test server that returns 200 OK
	healthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer healthyServer.Close()

	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{
			name:    "healthy server returns nil",
			url:     healthyServer.URL,
			wantErr: false,
		},
		{
			name:    "no server returns error",
			url:     "http://localhost:59999",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewManager(Config{
				URL:           tt.url,
				HealthTimeout: 1 * time.Second,
			})
			ctx := context.Background()
			err := m.EnsureRunning(ctx)
			if (err != nil) != tt.wantErr {
				t.Errorf("EnsureRunning() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil && tt.wantErr {
				// Verify error message contains helpful info
				if err.Error() == "" {
					t.Error("EnsureRunning() error message should not be empty")
				}
			}
		})
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Port != 8080 {
		t.Errorf("DefaultConfig().Port = %d, want 8080", cfg.Port)
	}
	if cfg.HealthTimeout != 5*time.Second {
		t.Errorf("DefaultConfig().HealthTimeout = %v, want 5s", cfg.HealthTimeout)
	}
}

func TestManager_ContextCancellation(t *testing.T) {
	// Create a test server with delay
	slowServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer slowServer.Close()

	m := NewManager(Config{
		URL:           slowServer.URL,
		HealthTimeout: 10 * time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Should return false quickly due to context cancellation
	start := time.Now()
	running := m.IsRunning(ctx)
	elapsed := time.Since(start)

	if running {
		t.Error("IsRunning() should return false when context is cancelled")
	}
	if elapsed > 1*time.Second {
		t.Errorf("IsRunning() took %v, expected quick return due to context cancellation", elapsed)
	}
}
