// Package api provides the HTTP/WebSocket server for the Weaver web UI.
// It exposes REST endpoints for configuration, sessions, agents, and exports.
package api

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

// Server represents the HTTP API server for Weaver.
type Server struct {
	httpServer *http.Server
	router     *Router
	config     *ServerConfig

	// mu protects server state
	mu      sync.RWMutex
	running bool
}

// ServerConfig holds configuration for the API server.
type ServerConfig struct {
	// Host is the interface to bind to (default: "localhost")
	Host string `yaml:"host" json:"host"`

	// Port is the port to listen on (default: 8081)
	Port int `yaml:"port" json:"port"`

	// ReadTimeout is the maximum duration for reading the entire request
	ReadTimeout time.Duration `yaml:"read_timeout" json:"readTimeout"`

	// WriteTimeout is the maximum duration before timing out writes of the response
	WriteTimeout time.Duration `yaml:"write_timeout" json:"writeTimeout"`

	// IdleTimeout is the maximum amount of time to wait for the next request
	IdleTimeout time.Duration `yaml:"idle_timeout" json:"idleTimeout"`

	// CORSOrigins is a list of allowed origins for CORS requests
	CORSOrigins []string `yaml:"cors_origins" json:"corsOrigins"`

	// EnableLogging enables request logging middleware
	EnableLogging bool `yaml:"enable_logging" json:"enableLogging"`
}

// DefaultServerConfig returns sensible defaults for the API server.
func DefaultServerConfig() *ServerConfig {
	return &ServerConfig{
		Host:          "localhost",
		Port:          8081,
		ReadTimeout:   15 * time.Second,
		WriteTimeout:  15 * time.Second,
		IdleTimeout:   60 * time.Second,
		CORSOrigins:   []string{"http://localhost:5173"}, // Vite dev server
		EnableLogging: true,
	}
}

// NewServer creates a new API server with the given configuration.
func NewServer(config *ServerConfig) *Server {
	if config == nil {
		config = DefaultServerConfig()
	}

	// Apply defaults for zero values
	if config.Host == "" {
		config.Host = "localhost"
	}
	if config.Port == 0 {
		config.Port = 8081
	}
	if config.ReadTimeout == 0 {
		config.ReadTimeout = 15 * time.Second
	}
	if config.WriteTimeout == 0 {
		config.WriteTimeout = 15 * time.Second
	}
	if config.IdleTimeout == 0 {
		config.IdleTimeout = 60 * time.Second
	}

	router := NewRouter()

	s := &Server{
		router: router,
		config: config,
	}

	return s
}

// Address returns the server address in host:port format.
func (s *Server) Address() string {
	return fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
}

// Router returns the underlying router for registering handlers.
func (s *Server) Router() *Router {
	return s.router
}

// Config returns the server configuration.
func (s *Server) Config() *ServerConfig {
	return s.config
}

// Start starts the HTTP server in a goroutine.
// It returns immediately after starting. Use Wait() or Shutdown() to manage lifecycle.
func (s *Server) Start() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return fmt.Errorf("server is already running")
	}

	// Build handler chain with middleware
	var handler http.Handler = s.router

	// Apply middleware in reverse order (last added is outermost)
	if len(s.config.CORSOrigins) > 0 {
		handler = CORSMiddleware(s.config.CORSOrigins)(handler)
		// Configure WebSocket upgrader to use the same CORS origins
		SetUpgraderCheckOrigin(makeOriginChecker(s.config.CORSOrigins))
	}
	if s.config.EnableLogging {
		handler = LoggingMiddleware(handler)
	}
	handler = RecoveryMiddleware(handler)

	s.httpServer = &http.Server{
		Addr:         s.Address(),
		Handler:      handler,
		ReadTimeout:  s.config.ReadTimeout,
		WriteTimeout: s.config.WriteTimeout,
		IdleTimeout:  s.config.IdleTimeout,
	}

	s.running = true

	// Use error channel to detect binding failures
	errCh := make(chan error, 1)
	go func() {
		log.Printf("[api] Starting server on %s", s.Address())
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("[api] Server error: %v", err)
			errCh <- err
		}
		close(errCh)
	}()

	// Wait briefly to catch immediate binding errors (e.g., port in use)
	select {
	case err := <-errCh:
		s.running = false
		return fmt.Errorf("server failed to start: %w", err)
	case <-time.After(100 * time.Millisecond):
		// Server likely started successfully
		return nil
	}
}

// Shutdown gracefully shuts down the server with a timeout.
func (s *Server) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return nil
	}

	log.Printf("[api] Shutting down server...")
	s.running = false

	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// IsRunning returns true if the server is currently running.
func (s *Server) IsRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.running
}

// makeOriginChecker creates a function that validates WebSocket origins
// against the configured CORS origins list.
func makeOriginChecker(allowedOrigins []string) func(*http.Request) bool {
	// Build a set for O(1) lookup
	allowed := make(map[string]bool)
	for _, origin := range allowedOrigins {
		allowed[origin] = true
		// Also allow wildcard
		if origin == "*" {
			return func(r *http.Request) bool {
				return true
			}
		}
	}

	return func(r *http.Request) bool {
		origin := r.Header.Get("Origin")
		if origin == "" {
			// No origin header (same-origin request) - allow
			return true
		}
		return allowed[origin]
	}
}
