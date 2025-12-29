// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"bufio"
	"log"
	"net"
	"net/http"
	"runtime/debug"
	"strings"
	"time"
)

// Middleware is a function that wraps an http.Handler.
type Middleware func(http.Handler) http.Handler

// -----------------------------------------------------------------------------
// CORS Middleware
// -----------------------------------------------------------------------------

// CORSMiddleware creates middleware that handles Cross-Origin Resource Sharing.
// It allows requests from the specified origins and handles preflight OPTIONS requests.
func CORSMiddleware(allowedOrigins []string) Middleware {
	originSet := make(map[string]bool)
	for _, origin := range allowedOrigins {
		originSet[origin] = true
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")

			// Check if origin is allowed
			if origin != "" && originSet[origin] {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Access-Control-Allow-Credentials", "true")
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, PATCH, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
				w.Header().Set("Access-Control-Max-Age", "86400") // 24 hours
			}

			// Handle preflight requests
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// -----------------------------------------------------------------------------
// Logging Middleware
// -----------------------------------------------------------------------------

// responseWriter wraps http.ResponseWriter to capture the status code.
type responseWriter struct {
	http.ResponseWriter
	statusCode int
	written    int64
}

// WriteHeader captures the status code before writing.
func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Write captures the number of bytes written.
func (rw *responseWriter) Write(b []byte) (int, error) {
	n, err := rw.ResponseWriter.Write(b)
	rw.written += int64(n)
	return n, err
}

// Hijack implements http.Hijacker to support WebSocket upgrades.
// It delegates to the underlying ResponseWriter if it supports hijacking.
func (rw *responseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if hijacker, ok := rw.ResponseWriter.(http.Hijacker); ok {
		return hijacker.Hijack()
	}
	return nil, nil, http.ErrNotSupported
}

// Flush implements http.Flusher to support streaming responses.
func (rw *responseWriter) Flush() {
	if flusher, ok := rw.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// LoggingMiddleware creates middleware that logs HTTP requests.
// Format: [api] METHOD /path status latency bytes
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap response writer to capture status code
		wrapped := &responseWriter{
			ResponseWriter: w,
			statusCode:     http.StatusOK,
		}

		// Process request
		next.ServeHTTP(wrapped, r)

		// Calculate latency
		latency := time.Since(start)

		// Format latency for display
		latencyStr := formatLatency(latency)

		// Log the request
		log.Printf("[api] %s %s %d %s %d bytes",
			r.Method,
			r.URL.Path,
			wrapped.statusCode,
			latencyStr,
			wrapped.written,
		)
	})
}

// formatLatency formats a duration for human-readable display.
func formatLatency(d time.Duration) string {
	if d < time.Millisecond {
		return d.Round(time.Microsecond).String()
	}
	if d < time.Second {
		return d.Round(time.Millisecond).String()
	}
	return d.Round(10 * time.Millisecond).String()
}

// -----------------------------------------------------------------------------
// Recovery Middleware
// -----------------------------------------------------------------------------

// RecoveryMiddleware creates middleware that recovers from panics.
// It logs the panic and returns a 500 Internal Server Error.
func RecoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				// Log the panic with stack trace
				log.Printf("[api] PANIC: %v\n%s", err, debug.Stack())

				// Return 500 error
				WriteError(w, http.StatusInternalServerError,
					"internal_error",
					"An unexpected error occurred")
			}
		}()

		next.ServeHTTP(w, r)
	})
}

// -----------------------------------------------------------------------------
// Content-Type Middleware
// -----------------------------------------------------------------------------

// ContentTypeMiddleware creates middleware that validates Content-Type for requests with bodies.
// It ensures POST, PUT, and PATCH requests have application/json Content-Type.
func ContentTypeMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only check for methods that typically have request bodies
		if r.Method == http.MethodPost || r.Method == http.MethodPut || r.Method == http.MethodPatch {
			contentType := r.Header.Get("Content-Type")

			// Allow requests without body (Content-Length: 0)
			if r.ContentLength > 0 {
				if !strings.HasPrefix(contentType, "application/json") {
					WriteError(w, http.StatusUnsupportedMediaType,
						"unsupported_media_type",
						"Content-Type must be application/json")
					return
				}
			}
		}

		next.ServeHTTP(w, r)
	})
}

// -----------------------------------------------------------------------------
// Request ID Middleware
// -----------------------------------------------------------------------------

// RequestIDMiddleware creates middleware that adds a unique request ID to each request.
// The request ID is added to the response headers as X-Request-ID.
func RequestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if request already has an ID (from upstream proxy)
		requestID := r.Header.Get("X-Request-ID")

		if requestID == "" {
			// Generate a simple timestamp-based ID
			requestID = generateRequestID()
		}

		// Add to response headers
		w.Header().Set("X-Request-ID", requestID)

		next.ServeHTTP(w, r)
	})
}

// generateRequestID generates a simple unique request ID.
func generateRequestID() string {
	// Use timestamp + random suffix for uniqueness
	// This is a simple implementation; for production, consider using UUID
	return time.Now().Format("20060102150405.000000000")
}

// -----------------------------------------------------------------------------
// Middleware Chain Helper
// -----------------------------------------------------------------------------

// Chain applies multiple middleware to a handler in the order provided.
// The first middleware in the slice is the outermost (first to receive request).
func Chain(handler http.Handler, middlewares ...Middleware) http.Handler {
	// Apply in reverse order so first middleware wraps outermost
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	return handler
}
