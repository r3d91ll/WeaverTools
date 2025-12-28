// Package export provides academic export format utilities.
// Generates LaTeX tables, CSV data, and other publication-ready outputs.
package export

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// HashAlgorithm identifies the hashing algorithm used for experiment hashes.
const HashAlgorithm = "SHA-256"

// ExperimentConfig holds the configuration parameters for hashing.
// These fields determine the experiment's reproducibility signature.
type ExperimentConfig struct {
	// ToolVersion is the version of the tool used.
	ToolVersion string `json:"tool_version"`

	// MeasurementMode indicates how measurements were captured.
	// Examples: "passive", "active", "triggered"
	MeasurementMode string `json:"measurement_mode"`

	// MeasurementCount is the total number of measurements in the experiment.
	MeasurementCount int `json:"measurement_count"`

	// ConversationCount is the number of conversations in the session.
	ConversationCount int `json:"conversation_count"`

	// StartTime is when the experiment started (UTC).
	StartTime time.Time `json:"start_time"`

	// EndTime is when the experiment ended (UTC), if available.
	EndTime *time.Time `json:"end_time,omitempty"`

	// Parameters holds additional configuration parameters as key-value pairs.
	// Keys are sorted alphabetically during hashing for determinism.
	Parameters map[string]string `json:"parameters,omitempty"`
}

// ExperimentHash represents the computed hash and its metadata.
type ExperimentHash struct {
	// Hash is the hex-encoded SHA-256 hash of the experiment configuration.
	Hash string `json:"hash"`

	// Algorithm identifies the hashing algorithm used.
	Algorithm string `json:"algorithm"`

	// ComputedAt is when the hash was computed.
	ComputedAt time.Time `json:"computed_at"`

	// Config is the configuration that was hashed.
	Config *ExperimentConfig `json:"config"`
}

// HashBuilder constructs experiment hashes from configuration data.
type HashBuilder struct {
	config *ExperimentConfig
}

// NewHashBuilder creates a new HashBuilder with default configuration.
func NewHashBuilder() *HashBuilder {
	return &HashBuilder{
		config: &ExperimentConfig{
			Parameters: make(map[string]string),
		},
	}
}

// WithToolVersion sets the tool version.
func (hb *HashBuilder) WithToolVersion(version string) *HashBuilder {
	hb.config.ToolVersion = version
	return hb
}

// WithMeasurementMode sets the measurement mode.
func (hb *HashBuilder) WithMeasurementMode(mode string) *HashBuilder {
	hb.config.MeasurementMode = mode
	return hb
}

// WithMeasurementCount sets the measurement count.
func (hb *HashBuilder) WithMeasurementCount(count int) *HashBuilder {
	hb.config.MeasurementCount = count
	return hb
}

// WithConversationCount sets the conversation count.
func (hb *HashBuilder) WithConversationCount(count int) *HashBuilder {
	hb.config.ConversationCount = count
	return hb
}

// WithTimeRange sets the start and end times.
// If endTime is nil, only the start time is used.
func (hb *HashBuilder) WithTimeRange(startTime time.Time, endTime *time.Time) *HashBuilder {
	hb.config.StartTime = startTime
	hb.config.EndTime = endTime
	return hb
}

// WithParameter adds a configuration parameter.
// Parameters are sorted by key during hashing for determinism.
func (hb *HashBuilder) WithParameter(key, value string) *HashBuilder {
	if hb.config.Parameters == nil {
		hb.config.Parameters = make(map[string]string)
	}
	hb.config.Parameters[key] = value
	return hb
}

// WithParameters adds multiple configuration parameters.
func (hb *HashBuilder) WithParameters(params map[string]string) *HashBuilder {
	if hb.config.Parameters == nil {
		hb.config.Parameters = make(map[string]string)
	}
	for k, v := range params {
		hb.config.Parameters[k] = v
	}
	return hb
}

// Build computes the experiment hash and returns the ExperimentHash.
// The hash is deterministic: identical configurations produce identical hashes.
func (hb *HashBuilder) Build() *ExperimentHash {
	hash := computeHash(hb.config)
	return &ExperimentHash{
		Hash:       hash,
		Algorithm:  HashAlgorithm,
		ComputedAt: time.Now(),
		Config:     hb.config,
	}
}

// computeHash generates a deterministic SHA-256 hash from the configuration.
func computeHash(config *ExperimentConfig) string {
	// Build a canonical string representation for hashing.
	// Order is fixed to ensure determinism.
	var sb strings.Builder

	// Version
	sb.WriteString("version:")
	sb.WriteString(config.ToolVersion)
	sb.WriteString("|")

	// Measurement mode
	sb.WriteString("mode:")
	sb.WriteString(config.MeasurementMode)
	sb.WriteString("|")

	// Counts
	sb.WriteString("measurements:")
	sb.WriteString(fmt.Sprintf("%d", config.MeasurementCount))
	sb.WriteString("|")
	sb.WriteString("conversations:")
	sb.WriteString(fmt.Sprintf("%d", config.ConversationCount))
	sb.WriteString("|")

	// Time range (UTC, RFC3339 format for consistency)
	sb.WriteString("start:")
	sb.WriteString(config.StartTime.UTC().Format(time.RFC3339))
	sb.WriteString("|")

	if config.EndTime != nil {
		sb.WriteString("end:")
		sb.WriteString(config.EndTime.UTC().Format(time.RFC3339))
		sb.WriteString("|")
	}

	// Parameters (sorted by key for determinism)
	if len(config.Parameters) > 0 {
		sb.WriteString("params:")
		keys := make([]string, 0, len(config.Parameters))
		for k := range config.Parameters {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		for i, k := range keys {
			if i > 0 {
				sb.WriteString(",")
			}
			sb.WriteString(k)
			sb.WriteString("=")
			sb.WriteString(config.Parameters[k])
		}
		sb.WriteString("|")
	}

	// Compute SHA-256 hash
	hasher := sha256.New()
	hasher.Write([]byte(sb.String()))
	return hex.EncodeToString(hasher.Sum(nil))
}

// ShortHash returns the first 8 characters of the full hash.
// This is suitable for display in reports and citations.
func (eh *ExperimentHash) ShortHash() string {
	if len(eh.Hash) >= 8 {
		return eh.Hash[:8]
	}
	return eh.Hash
}

// Verify recomputes the hash and checks if it matches the stored hash.
// Returns true if the hash is valid, false otherwise.
func (eh *ExperimentHash) Verify() bool {
	if eh.Config == nil {
		return false
	}
	recomputed := computeHash(eh.Config)
	return recomputed == eh.Hash
}

// ToJSON returns the experiment hash as a JSON string.
func (eh *ExperimentHash) ToJSON() (string, error) {
	data, err := json.MarshalIndent(eh, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal experiment hash: %w", err)
	}
	return string(data), nil
}

// ComputeExperimentHash is a convenience function to compute a hash from
// session-like data. This is the primary entry point for hash generation.
func ComputeExperimentHash(
	toolVersion string,
	measurementMode string,
	measurementCount int,
	conversationCount int,
	startTime time.Time,
	endTime *time.Time,
	params map[string]string,
) *ExperimentHash {
	builder := NewHashBuilder().
		WithToolVersion(toolVersion).
		WithMeasurementMode(measurementMode).
		WithMeasurementCount(measurementCount).
		WithConversationCount(conversationCount).
		WithTimeRange(startTime, endTime)

	if params != nil {
		builder.WithParameters(params)
	}

	return builder.Build()
}

// HashFromConfig computes an experiment hash directly from an ExperimentConfig.
func HashFromConfig(config *ExperimentConfig) *ExperimentHash {
	if config == nil {
		config = &ExperimentConfig{}
	}

	hash := computeHash(config)
	return &ExperimentHash{
		Hash:       hash,
		Algorithm:  HashAlgorithm,
		ComputedAt: time.Now(),
		Config:     config,
	}
}
