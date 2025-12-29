// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
)

// HandlerFunc is the function signature for API handlers.
type HandlerFunc func(w http.ResponseWriter, r *http.Request)

// Route represents a registered route with its handler.
type Route struct {
	Method  string
	Pattern string
	Handler HandlerFunc
}

// Router is a simple HTTP router that supports path parameters.
// It provides a lightweight alternative to external routing libraries.
type Router struct {
	routes []Route
	mu     sync.RWMutex

	// NotFound is called when no route matches
	NotFound http.Handler
}

// NewRouter creates a new Router instance.
func NewRouter() *Router {
	return &Router{
		routes: make([]Route, 0),
		NotFound: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			WriteError(w, http.StatusNotFound, "not_found", "The requested resource was not found")
		}),
	}
}

// Handle registers a handler for the given method and pattern.
// Patterns support path parameters with :param syntax (e.g., /api/users/:id).
func (rt *Router) Handle(method, pattern string, handler HandlerFunc) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	rt.routes = append(rt.routes, Route{
		Method:  method,
		Pattern: pattern,
		Handler: handler,
	})
}

// GET registers a handler for GET requests.
func (rt *Router) GET(pattern string, handler HandlerFunc) {
	rt.Handle(http.MethodGet, pattern, handler)
}

// POST registers a handler for POST requests.
func (rt *Router) POST(pattern string, handler HandlerFunc) {
	rt.Handle(http.MethodPost, pattern, handler)
}

// PUT registers a handler for PUT requests.
func (rt *Router) PUT(pattern string, handler HandlerFunc) {
	rt.Handle(http.MethodPut, pattern, handler)
}

// DELETE registers a handler for DELETE requests.
func (rt *Router) DELETE(pattern string, handler HandlerFunc) {
	rt.Handle(http.MethodDelete, pattern, handler)
}

// PATCH registers a handler for PATCH requests.
func (rt *Router) PATCH(pattern string, handler HandlerFunc) {
	rt.Handle(http.MethodPatch, pattern, handler)
}

// ServeHTTP implements the http.Handler interface.
func (rt *Router) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	path := r.URL.Path

	for _, route := range rt.routes {
		// Check method first
		if route.Method != r.Method {
			continue
		}

		// Try to match the pattern
		params, matched := matchPath(route.Pattern, path)
		if matched {
			// Store params in request context
			if len(params) > 0 {
				r = setPathParams(r, params)
			}
			route.Handler(w, r)
			return
		}
	}

	// No route matched
	rt.NotFound.ServeHTTP(w, r)
}

// matchPath matches a URL path against a pattern and extracts path parameters.
// Pattern syntax: /api/users/:id matches /api/users/123 with id=123
func matchPath(pattern, path string) (map[string]string, bool) {
	patternParts := strings.Split(strings.Trim(pattern, "/"), "/")
	pathParts := strings.Split(strings.Trim(path, "/"), "/")

	if len(patternParts) != len(pathParts) {
		return nil, false
	}

	params := make(map[string]string)

	for i, patternPart := range patternParts {
		if strings.HasPrefix(patternPart, ":") {
			// This is a parameter
			paramName := patternPart[1:]
			params[paramName] = pathParts[i]
		} else if patternPart != pathParts[i] {
			// Literal part doesn't match
			return nil, false
		}
	}

	return params, true
}

// contextKey is a type for context keys to avoid collisions.
type contextKey string

const pathParamsKey contextKey = "pathParams"

// setPathParams stores path parameters in the request context.
func setPathParams(r *http.Request, params map[string]string) *http.Request {
	ctx := context.WithValue(r.Context(), pathParamsKey, params)
	return r.WithContext(ctx)
}

// PathParam extracts a path parameter from the request.
func PathParam(r *http.Request, name string) string {
	params, ok := r.Context().Value(pathParamsKey).(map[string]string)
	if !ok {
		return ""
	}
	return params[name]
}

// -----------------------------------------------------------------------------
// Response Helpers
// -----------------------------------------------------------------------------

// APIResponse is the standard response wrapper for API endpoints.
type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   *APIError   `json:"error,omitempty"`
}

// APIError represents an error response.
type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// WriteJSON writes a JSON response with the given status code.
func WriteJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	response := APIResponse{
		Success: status >= 200 && status < 300,
		Data:    data,
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		// If encoding fails, log it but there's nothing we can do
		// since headers are already sent
		return
	}
}

// WriteError writes a JSON error response.
func WriteError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	response := APIResponse{
		Success: false,
		Error: &APIError{
			Code:    code,
			Message: message,
		},
	}

	json.NewEncoder(w).Encode(response)
}

// ReadJSON reads and decodes a JSON request body into the given target.
func ReadJSON(r *http.Request, target interface{}) error {
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(target)
}
