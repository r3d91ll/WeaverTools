// Package analysis provides HTTP client for TheLoom analysis endpoints.
// Used for Kakeya geometry analysis of concept hidden states.
package analysis

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
)

// Client connects to TheLoom analysis endpoints.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a new analysis client.
func NewClient(baseURL string) *Client {
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// WolfAxiomResult contains Wolf-inspired density analysis.
type WolfAxiomResult struct {
	MaxDensityRatio  float64 `json:"max_density_ratio"`
	MeanDensityRatio float64 `json:"mean_density_ratio"`
	UniformityPValue float64 `json:"uniformity_p_value"`
	ViolationCount   int     `json:"violation_count"`
	Severity         string  `json:"severity"`
}

// DirectionalCoverageResult contains coverage analysis.
type DirectionalCoverageResult struct {
	AmbientDim         int     `json:"ambient_dim"`
	EffectiveDim       int     `json:"effective_dim"`
	CoverageRatio      float64 `json:"coverage_ratio"`
	CoverageQuality    string  `json:"coverage_quality"`
	SphericalUniformity float64 `json:"spherical_uniformity"`
	IsotropyScore      float64 `json:"isotropy_score"`
}

// GrainAnalysisResult contains grain/cluster detection.
type GrainAnalysisResult struct {
	NumGrains       int     `json:"num_grains"`
	GrainCoverage   float64 `json:"grain_coverage"`
	MeanGrainSize   float64 `json:"mean_grain_size"`
	MeanAspectRatio float64 `json:"mean_aspect_ratio"`
}

// GeometryResult contains full Kakeya-inspired geometry analysis.
type GeometryResult struct {
	OverallHealth       string                    `json:"overall_health"`
	NumVectors          int                       `json:"num_vectors"`
	AmbientDim          int                       `json:"ambient_dim"`
	WolfAxiom           WolfAxiomResult           `json:"wolf_axiom"`
	DirectionalCoverage DirectionalCoverageResult `json:"directional_coverage"`
	GrainAnalysis       GrainAnalysisResult       `json:"grain_analysis"`
	AnalysisTimeMs      float64                   `json:"analysis_time_ms"`
}

// BilateralResult contains sender/receiver geometry comparison.
type BilateralResult struct {
	DirectionalAlignment float64 `json:"directional_alignment"`
	SubspaceOverlap      float64 `json:"subspace_overlap"`
	GrainAlignment       float64 `json:"grain_alignment"`
	DensitySimilarity    float64 `json:"density_similarity"`
	EffectiveDimRatio    float64 `json:"effective_dim_ratio"`
	OverallAlignment     float64 `json:"overall_alignment"`
	AnalysisTimeMs       float64 `json:"analysis_time_ms"`
}

// AnalyzeGeometry performs Kakeya-inspired analysis on a set of vectors.
func (c *Client) AnalyzeGeometry(ctx context.Context, vectors [][]float64) (*GeometryResult, error) {
	if len(vectors) < 3 {
		return nil, createInsufficientVectorsError("geometry analysis", len(vectors), 3)
	}

	reqBody := map[string]any{
		"vectors": vectors,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, createMarshalRequestError(err, "geometry", c.baseURL)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/analyze/geometry", bytes.NewReader(body))
	if err != nil {
		return nil, createRequestCreationError(err, "geometry", c.baseURL)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, createConnectionError(err, "geometry", c.baseURL)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, createResponseReadError(err, "geometry", c.baseURL)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, createHTTPError(resp.StatusCode, string(respBody), "geometry", c.baseURL)
	}

	var result GeometryResult
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, createUnmarshalResponseError(err, "geometry", c.baseURL, string(respBody))
	}

	return &result, nil
}

// CompareBilateral compares geometric properties between two vector sets.
func (c *Client) CompareBilateral(ctx context.Context, sender, receiver [][]float64) (*BilateralResult, error) {
	if len(sender) < 3 {
		return nil, createInsufficientVectorsError("bilateral comparison (sender)", len(sender), 3)
	}
	if len(receiver) < 3 {
		return nil, createInsufficientVectorsError("bilateral comparison (receiver)", len(receiver), 3)
	}

	reqBody := map[string]any{
		"sender_vectors":   sender,
		"receiver_vectors": receiver,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, createMarshalRequestError(err, "bilateral", c.baseURL)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/analyze/bilateral", bytes.NewReader(body))
	if err != nil {
		return nil, createRequestCreationError(err, "bilateral", c.baseURL)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, createConnectionError(err, "bilateral", c.baseURL)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, createResponseReadError(err, "bilateral", c.baseURL)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, createHTTPError(resp.StatusCode, string(respBody), "bilateral", c.baseURL)
	}

	var result BilateralResult
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, createUnmarshalResponseError(err, "bilateral", c.baseURL, string(respBody))
	}

	return &result, nil
}

// IsAvailable checks if the TheLoom server is reachable.
func (c *Client) IsAvailable(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// -----------------------------------------------------------------------------
// Error Helper Functions
// -----------------------------------------------------------------------------
// These functions create structured WeaverErrors for various analysis client
// failure scenarios with appropriate context and suggestions.

// createInsufficientVectorsError creates a structured error when not enough
// vectors are provided for analysis.
func createInsufficientVectorsError(operation string, provided, required int) *werrors.WeaverError {
	return werrors.New(werrors.ErrConceptsInsufficientSamples,
		fmt.Sprintf("%s requires at least %d vectors, but only %d provided", operation, required, provided)).
		WithContext("operation", operation).
		WithContext("provided_vectors", fmt.Sprintf("%d", provided)).
		WithContext("required_vectors", fmt.Sprintf("%d", required)).
		WithSuggestion(fmt.Sprintf("Extract more concept samples using '/extract <concept> %d'", required)).
		WithSuggestion("Use '/concepts' to see current sample counts for each concept").
		WithSuggestion("For meaningful analysis, consider extracting 10+ samples per concept")
}

// createMarshalRequestError creates a structured error when JSON marshaling
// of the request body fails.
func createMarshalRequestError(cause error, analysisType, serverURL string) *werrors.WeaverError {
	return werrors.IOWrap(cause, werrors.ErrIOMarshalFailed,
		"failed to serialize analysis request").
		WithContext("analysis_type", analysisType).
		WithContext("server_url", serverURL).
		WithContext("operation", "request marshaling").
		WithSuggestion("This is an internal error - the vector data may be corrupted").
		WithSuggestion("Try re-extracting the concept samples with '/extract'").
		WithSuggestion("If the problem persists, report this issue")
}

// createRequestCreationError creates a structured error when HTTP request
// creation fails.
func createRequestCreationError(cause error, analysisType, serverURL string) *werrors.WeaverError {
	return werrors.NetworkWrap(cause, werrors.ErrAnalysisFailed,
		"failed to create analysis request").
		WithContext("analysis_type", analysisType).
		WithContext("server_url", serverURL).
		WithContext("operation", "request creation").
		WithSuggestion("Check that the server URL is valid in your configuration").
		WithSuggestion("Verify the URL format: http://hostname:port").
		WithSuggestion("Current URL: " + serverURL)
}

// createConnectionError creates a structured error when the HTTP request
// to the analysis server fails. Detects timeout, connection refused, DNS, etc.
func createConnectionError(cause error, analysisType, serverURL string) *werrors.WeaverError {
	errStr := strings.ToLower(cause.Error())

	// Detect specific connection error types
	if isAnalysisTimeout(errStr) {
		return werrors.NetworkWrap(cause, werrors.ErrAnalysisServerUnavailable,
			"analysis server request timed out").
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("issue", "request timeout").
			WithSuggestion("Check if TheLoom server is running and responsive").
			WithSuggestion("Verify the server URL: " + serverURL).
			WithSuggestion("Try the health check: curl " + serverURL + "/health").
			WithSuggestion("Consider reducing the number of samples for faster analysis")
	}

	if isAnalysisConnectionRefused(errStr) {
		return werrors.NetworkWrap(cause, werrors.ErrAnalysisServerUnavailable,
			"cannot connect to analysis server - connection refused").
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("issue", "connection refused").
			WithSuggestion("Start TheLoom server: cd TheLoom && python -m loom.server").
			WithSuggestion("Verify the server is running on the configured port").
			WithSuggestion("Check if a firewall is blocking the connection").
			WithSuggestion("Verify the URL in your configuration: " + serverURL)
	}

	if isAnalysisDNSError(errStr) {
		return werrors.NetworkWrap(cause, werrors.ErrAnalysisServerUnavailable,
			"cannot resolve analysis server hostname").
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("issue", "DNS resolution failed").
			WithSuggestion("Check that the hostname is correct in your configuration").
			WithSuggestion("Try using an IP address instead of hostname").
			WithSuggestion("Verify network connectivity and DNS settings")
	}

	// Generic connection error
	return werrors.NetworkWrap(cause, werrors.ErrAnalysisServerUnavailable,
		"failed to connect to analysis server").
		WithContext("analysis_type", analysisType).
		WithContext("server_url", serverURL).
		WithSuggestion("Check that TheLoom server is running").
		WithSuggestion("Verify the server URL: " + serverURL).
		WithSuggestion("Test connectivity: curl " + serverURL + "/health")
}

// createResponseReadError creates a structured error when reading the
// response body fails.
func createResponseReadError(cause error, analysisType, serverURL string) *werrors.WeaverError {
	errStr := strings.ToLower(cause.Error())

	if isAnalysisEOF(errStr) {
		return werrors.IOWrap(cause, werrors.ErrAnalysisInvalidResponse,
			"analysis server closed connection unexpectedly").
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("issue", "incomplete response").
			WithSuggestion("The analysis server may have crashed during processing").
			WithSuggestion("Check TheLoom server logs for errors").
			WithSuggestion("Try the analysis again with fewer samples").
			WithSuggestion("If persistent, restart TheLoom server")
	}

	return werrors.IOWrap(cause, werrors.ErrAnalysisInvalidResponse,
		"failed to read analysis response").
		WithContext("analysis_type", analysisType).
		WithContext("server_url", serverURL).
		WithContext("operation", "response reading").
		WithSuggestion("The analysis server may have encountered an error").
		WithSuggestion("Check TheLoom server logs for details").
		WithSuggestion("Try the analysis again")
}

// createHTTPError creates a structured error based on HTTP status code.
// Provides specific guidance for common error codes.
func createHTTPError(statusCode int, responseBody, analysisType, serverURL string) *werrors.WeaverError {
	respLower := strings.ToLower(responseBody)

	switch {
	case statusCode == 400:
		// Bad request - usually validation error
		return werrors.New(werrors.ErrAnalysisInvalidResponse,
			fmt.Sprintf("analysis server rejected request (HTTP %d)", statusCode)).
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("status_code", fmt.Sprintf("%d", statusCode)).
			WithContext("response", truncateResponse(responseBody)).
			WithSuggestion("The request format may be invalid or unsupported").
			WithSuggestion("Check that vector dimensions are consistent").
			WithSuggestion("Verify all vectors have the same length")

	case statusCode == 404:
		// Endpoint not found
		return werrors.New(werrors.ErrAnalysisServerUnavailable,
			fmt.Sprintf("analysis endpoint not found (HTTP %d)", statusCode)).
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("status_code", fmt.Sprintf("%d", statusCode)).
			WithSuggestion("The analysis endpoint may not be available on this server").
			WithSuggestion("Check that TheLoom server has analysis capabilities enabled").
			WithSuggestion("Verify the server version supports Kakeya analysis")

	case statusCode == 422:
		// Validation error
		return werrors.New(werrors.ErrAnalysisInvalidResponse,
			fmt.Sprintf("analysis validation failed (HTTP %d)", statusCode)).
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("status_code", fmt.Sprintf("%d", statusCode)).
			WithContext("response", truncateResponse(responseBody)).
			WithSuggestion("The analysis input did not pass server validation").
			WithSuggestion("Ensure vectors have consistent dimensions").
			WithSuggestion("Re-extract concept samples to get fresh data")

	case statusCode == 500:
		// Internal server error
		if isAnalysisGPUError(respLower) {
			return werrors.New(werrors.ErrAnalysisFailed,
				"analysis server GPU error").
				WithContext("analysis_type", analysisType).
				WithContext("server_url", serverURL).
				WithContext("status_code", "500").
				WithContext("issue", "GPU/CUDA error").
				WithSuggestion("The server may be out of GPU memory").
				WithSuggestion("Try analyzing fewer samples").
				WithSuggestion("Restart TheLoom server to reset GPU state").
				WithSuggestion("Consider using CPU-only mode if available")
		}
		return werrors.New(werrors.ErrAnalysisFailed,
			fmt.Sprintf("analysis server internal error (HTTP %d)", statusCode)).
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("status_code", fmt.Sprintf("%d", statusCode)).
			WithContext("response", truncateResponse(responseBody)).
			WithSuggestion("Check TheLoom server logs for error details").
			WithSuggestion("Try the analysis again - it may be a transient error").
			WithSuggestion("If persistent, restart TheLoom server")

	case statusCode == 502 || statusCode == 503 || statusCode == 504:
		// Gateway/service unavailable errors
		return werrors.New(werrors.ErrAnalysisServerUnavailable,
			fmt.Sprintf("analysis server unavailable (HTTP %d)", statusCode)).
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("status_code", fmt.Sprintf("%d", statusCode)).
			WithSuggestion("The analysis server may be overloaded or restarting").
			WithSuggestion("Wait a moment and try again").
			WithSuggestion("Check if TheLoom server is running properly")

	default:
		// Generic HTTP error
		return werrors.New(werrors.ErrAnalysisFailed,
			fmt.Sprintf("analysis failed with HTTP status %d", statusCode)).
			WithContext("analysis_type", analysisType).
			WithContext("server_url", serverURL).
			WithContext("status_code", fmt.Sprintf("%d", statusCode)).
			WithContext("response", truncateResponse(responseBody)).
			WithSuggestion("Check TheLoom server logs for more details").
			WithSuggestion("Verify the server is functioning correctly")
	}
}

// createUnmarshalResponseError creates a structured error when JSON
// unmarshaling of the response fails.
func createUnmarshalResponseError(cause error, analysisType, serverURL, responseBody string) *werrors.WeaverError {
	return werrors.IOWrap(cause, werrors.ErrAnalysisInvalidResponse,
		"failed to parse analysis response").
		WithContext("analysis_type", analysisType).
		WithContext("server_url", serverURL).
		WithContext("operation", "response parsing").
		WithContext("response_preview", truncateResponse(responseBody)).
		WithSuggestion("The server response format may be incompatible").
		WithSuggestion("Check TheLoom server version matches the expected API").
		WithSuggestion("Server response may indicate an error - check logs")
}

// -----------------------------------------------------------------------------
// Error Detection Helpers
// -----------------------------------------------------------------------------

// isAnalysisTimeout checks if the error indicates a timeout.
func isAnalysisTimeout(errStr string) bool {
	patterns := []string{
		"timeout",
		"deadline exceeded",
		"timed out",
		"context deadline",
		"i/o timeout",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isAnalysisConnectionRefused checks if the error indicates connection refused.
func isAnalysisConnectionRefused(errStr string) bool {
	patterns := []string{
		"connection refused",
		"connect: connection refused",
		"no connection could be made",
		"actively refused",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isAnalysisDNSError checks if the error indicates DNS resolution failure.
func isAnalysisDNSError(errStr string) bool {
	patterns := []string{
		"no such host",
		"dns",
		"lookup",
		"name resolution",
		"getaddrinfo",
		"could not resolve host",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isAnalysisEOF checks if the error indicates EOF/incomplete response.
func isAnalysisEOF(errStr string) bool {
	patterns := []string{
		"eof",
		"unexpected end",
		"connection reset",
		"broken pipe",
	}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// isAnalysisGPUError checks if the response indicates a GPU/CUDA error.
func isAnalysisGPUError(respLower string) bool {
	patterns := []string{
		"cuda",
		"gpu",
		"out of memory",
		"oom",
		"memory allocation",
		"cublas",
		"cudnn",
	}
	for _, p := range patterns {
		if strings.Contains(respLower, p) {
			return true
		}
	}
	return false
}

// truncateResponse truncates long response bodies for error context.
func truncateResponse(resp string) string {
	const maxLen = 200
	resp = strings.TrimSpace(resp)
	if len(resp) > maxLen {
		return resp[:maxLen] + "..."
	}
	return resp
}
