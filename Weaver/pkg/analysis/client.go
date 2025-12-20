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
	"time"
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
		return nil, fmt.Errorf("need at least 3 vectors, got %d", len(vectors))
	}

	reqBody := map[string]any{
		"vectors": vectors,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/analyze/geometry", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("analysis failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var result GeometryResult
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}

// CompareBilateral compares geometric properties between two vector sets.
func (c *Client) CompareBilateral(ctx context.Context, sender, receiver [][]float64) (*BilateralResult, error) {
	if len(sender) < 3 {
		return nil, fmt.Errorf("need at least 3 sender vectors, got %d", len(sender))
	}
	if len(receiver) < 3 {
		return nil, fmt.Errorf("need at least 3 receiver vectors, got %d", len(receiver))
	}

	reqBody := map[string]any{
		"sender_vectors":   sender,
		"receiver_vectors": receiver,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/analyze/bilateral", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bilateral analysis failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var result BilateralResult
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
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
