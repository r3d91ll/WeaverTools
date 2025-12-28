package export

import (
	"encoding/json"
	"testing"
	"time"
)

// TestHashDeterminism verifies that identical inputs produce the same hash.
func TestHashDeterminism(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)
	endTime := time.Date(2024, 6, 15, 11, 45, 0, 0, time.UTC)

	// Compute hash multiple times with identical inputs
	hash1 := ComputeExperimentHash(
		"1.0.0",
		"active",
		100,
		5,
		startTime,
		&endTime,
		map[string]string{"model": "gpt-4", "temperature": "0.7"},
	)

	hash2 := ComputeExperimentHash(
		"1.0.0",
		"active",
		100,
		5,
		startTime,
		&endTime,
		map[string]string{"model": "gpt-4", "temperature": "0.7"},
	)

	hash3 := ComputeExperimentHash(
		"1.0.0",
		"active",
		100,
		5,
		startTime,
		&endTime,
		map[string]string{"temperature": "0.7", "model": "gpt-4"}, // Different order
	)

	if hash1.Hash != hash2.Hash {
		t.Errorf("Hash mismatch for identical inputs: %s != %s", hash1.Hash, hash2.Hash)
	}

	if hash1.Hash != hash3.Hash {
		t.Errorf("Hash should be order-independent for params: %s != %s", hash1.Hash, hash3.Hash)
	}

	// Verify the hash is 64 characters (SHA-256 hex)
	if len(hash1.Hash) != 64 {
		t.Errorf("Expected hash length 64, got %d", len(hash1.Hash))
	}

	// Verify algorithm is set correctly
	if hash1.Algorithm != HashAlgorithm {
		t.Errorf("Expected algorithm %s, got %s", HashAlgorithm, hash1.Algorithm)
	}
}

// TestHashDeterminismNoEndTime verifies determinism when end time is nil.
func TestHashDeterminismNoEndTime(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)

	hash1 := ComputeExperimentHash(
		"1.0.0",
		"passive",
		50,
		3,
		startTime,
		nil,
		nil,
	)

	hash2 := ComputeExperimentHash(
		"1.0.0",
		"passive",
		50,
		3,
		startTime,
		nil,
		nil,
	)

	if hash1.Hash != hash2.Hash {
		t.Errorf("Hash mismatch for identical inputs (no end time): %s != %s", hash1.Hash, hash2.Hash)
	}
}

// TestHashDifference verifies that different inputs produce different hashes.
func TestHashDifference(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)

	baseHash := ComputeExperimentHash(
		"1.0.0",
		"active",
		100,
		5,
		startTime,
		nil,
		nil,
	)

	tests := []struct {
		name          string
		version       string
		mode          string
		measurements  int
		conversations int
		startTime     time.Time
		params        map[string]string
	}{
		{"different version", "2.0.0", "active", 100, 5, startTime, nil},
		{"different mode", "1.0.0", "passive", 100, 5, startTime, nil},
		{"different measurement count", "1.0.0", "active", 101, 5, startTime, nil},
		{"different conversation count", "1.0.0", "active", 100, 6, startTime, nil},
		{"different start time", "1.0.0", "active", 100, 5, startTime.Add(time.Hour), nil},
		{"with parameters", "1.0.0", "active", 100, 5, startTime, map[string]string{"key": "value"}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hash := ComputeExperimentHash(
				tc.version,
				tc.mode,
				tc.measurements,
				tc.conversations,
				tc.startTime,
				nil,
				tc.params,
			)

			if hash.Hash == baseHash.Hash {
				t.Errorf("Expected different hash for %s, got same: %s", tc.name, hash.Hash)
			}
		})
	}
}

// TestHashBuilder verifies the builder pattern works correctly.
func TestHashBuilder(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)
	endTime := time.Date(2024, 6, 15, 11, 45, 0, 0, time.UTC)

	hash := NewHashBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(100).
		WithConversationCount(5).
		WithTimeRange(startTime, &endTime).
		WithParameter("model", "gpt-4").
		WithParameter("temperature", "0.7").
		Build()

	// Verify config is set correctly
	if hash.Config.ToolVersion != "1.0.0" {
		t.Errorf("Expected ToolVersion 1.0.0, got %s", hash.Config.ToolVersion)
	}
	if hash.Config.MeasurementMode != "active" {
		t.Errorf("Expected MeasurementMode active, got %s", hash.Config.MeasurementMode)
	}
	if hash.Config.MeasurementCount != 100 {
		t.Errorf("Expected MeasurementCount 100, got %d", hash.Config.MeasurementCount)
	}
	if hash.Config.ConversationCount != 5 {
		t.Errorf("Expected ConversationCount 5, got %d", hash.Config.ConversationCount)
	}
	if !hash.Config.StartTime.Equal(startTime) {
		t.Errorf("Expected StartTime %v, got %v", startTime, hash.Config.StartTime)
	}
	if hash.Config.EndTime == nil || !hash.Config.EndTime.Equal(endTime) {
		t.Errorf("Expected EndTime %v, got %v", endTime, hash.Config.EndTime)
	}
	if hash.Config.Parameters["model"] != "gpt-4" {
		t.Errorf("Expected parameter model=gpt-4, got %s", hash.Config.Parameters["model"])
	}
	if hash.Config.Parameters["temperature"] != "0.7" {
		t.Errorf("Expected parameter temperature=0.7, got %s", hash.Config.Parameters["temperature"])
	}

	// Verify hash was computed
	if hash.Hash == "" {
		t.Error("Expected hash to be computed, got empty string")
	}
}

// TestHashBuilderWithParameters verifies bulk parameter setting.
func TestHashBuilderWithParameters(t *testing.T) {
	params := map[string]string{
		"model":       "gpt-4",
		"temperature": "0.7",
		"max_tokens":  "1000",
	}

	hash := NewHashBuilder().
		WithToolVersion("1.0.0").
		WithParameters(params).
		Build()

	if len(hash.Config.Parameters) != 3 {
		t.Errorf("Expected 3 parameters, got %d", len(hash.Config.Parameters))
	}

	for k, v := range params {
		if hash.Config.Parameters[k] != v {
			t.Errorf("Expected parameter %s=%s, got %s", k, v, hash.Config.Parameters[k])
		}
	}
}

// TestShortHash verifies the short hash function.
func TestShortHash(t *testing.T) {
	hash := NewHashBuilder().
		WithToolVersion("1.0.0").
		Build()

	shortHash := hash.ShortHash()

	if len(shortHash) != 8 {
		t.Errorf("Expected short hash length 8, got %d", len(shortHash))
	}

	if shortHash != hash.Hash[:8] {
		t.Errorf("Short hash should be first 8 chars: %s != %s", shortHash, hash.Hash[:8])
	}
}

// TestVerify verifies the hash verification function.
func TestVerify(t *testing.T) {
	hash := NewHashBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(100).
		Build()

	// Valid hash should verify
	if !hash.Verify() {
		t.Error("Expected valid hash to verify successfully")
	}

	// Tampered hash should not verify
	tamperedHash := &ExperimentHash{
		Hash:      "0000000000000000000000000000000000000000000000000000000000000000",
		Algorithm: HashAlgorithm,
		Config:    hash.Config,
	}
	if tamperedHash.Verify() {
		t.Error("Expected tampered hash to fail verification")
	}

	// Nil config should not verify
	nilConfigHash := &ExperimentHash{
		Hash:   hash.Hash,
		Config: nil,
	}
	if nilConfigHash.Verify() {
		t.Error("Expected nil config hash to fail verification")
	}
}

// TestToJSON verifies JSON serialization.
func TestToJSON(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)

	hash := ComputeExperimentHash(
		"1.0.0",
		"active",
		100,
		5,
		startTime,
		nil,
		map[string]string{"model": "gpt-4"},
	)

	jsonStr, err := hash.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON failed: %v", err)
	}

	// Verify it's valid JSON
	var parsed ExperimentHash
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		t.Fatalf("Failed to parse JSON: %v", err)
	}

	// Verify fields were serialized correctly
	if parsed.Hash != hash.Hash {
		t.Errorf("Hash mismatch after JSON round-trip: %s != %s", parsed.Hash, hash.Hash)
	}
	if parsed.Algorithm != hash.Algorithm {
		t.Errorf("Algorithm mismatch after JSON round-trip: %s != %s", parsed.Algorithm, hash.Algorithm)
	}
	if parsed.Config.ToolVersion != hash.Config.ToolVersion {
		t.Errorf("ToolVersion mismatch after JSON round-trip: %s != %s",
			parsed.Config.ToolVersion, hash.Config.ToolVersion)
	}
}

// TestHashFromConfig verifies hash generation from config struct.
func TestHashFromConfig(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)

	config := &ExperimentConfig{
		ToolVersion:       "1.0.0",
		MeasurementMode:   "active",
		MeasurementCount:  100,
		ConversationCount: 5,
		StartTime:         startTime,
		Parameters:        map[string]string{"model": "gpt-4"},
	}

	hash1 := HashFromConfig(config)
	hash2 := HashFromConfig(config)

	if hash1.Hash != hash2.Hash {
		t.Errorf("HashFromConfig should be deterministic: %s != %s", hash1.Hash, hash2.Hash)
	}

	// Verify config is stored
	if hash1.Config.ToolVersion != "1.0.0" {
		t.Errorf("Expected ToolVersion 1.0.0, got %s", hash1.Config.ToolVersion)
	}
}

// TestHashFromNilConfig verifies nil config handling.
func TestHashFromNilConfig(t *testing.T) {
	hash := HashFromConfig(nil)

	if hash.Hash == "" {
		t.Error("Expected hash to be computed even for nil config")
	}
	if hash.Config == nil {
		t.Error("Expected config to be non-nil after HashFromConfig")
	}
}

// TestEmptyConfig verifies hashing of empty configuration.
func TestEmptyConfig(t *testing.T) {
	hash1 := NewHashBuilder().Build()
	hash2 := NewHashBuilder().Build()

	if hash1.Hash != hash2.Hash {
		t.Errorf("Empty config hash should be deterministic: %s != %s", hash1.Hash, hash2.Hash)
	}

	if hash1.Hash == "" {
		t.Error("Expected non-empty hash for empty config")
	}
}

// TestParameterOrderIndependence verifies that parameter order doesn't affect hash.
func TestParameterOrderIndependence(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)

	// Add parameters in different orders
	hash1 := NewHashBuilder().
		WithToolVersion("1.0.0").
		WithTimeRange(startTime, nil).
		WithParameter("a", "1").
		WithParameter("b", "2").
		WithParameter("c", "3").
		Build()

	hash2 := NewHashBuilder().
		WithToolVersion("1.0.0").
		WithTimeRange(startTime, nil).
		WithParameter("c", "3").
		WithParameter("a", "1").
		WithParameter("b", "2").
		Build()

	hash3 := NewHashBuilder().
		WithToolVersion("1.0.0").
		WithTimeRange(startTime, nil).
		WithParameter("b", "2").
		WithParameter("c", "3").
		WithParameter("a", "1").
		Build()

	if hash1.Hash != hash2.Hash {
		t.Errorf("Parameter order should not affect hash: %s != %s", hash1.Hash, hash2.Hash)
	}
	if hash1.Hash != hash3.Hash {
		t.Errorf("Parameter order should not affect hash: %s != %s", hash1.Hash, hash3.Hash)
	}
}

// TestTimeZoneIndependence verifies that timezone doesn't affect hash.
func TestTimeZoneIndependence(t *testing.T) {
	// Same instant in different timezones
	utcTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)
	estLocation, _ := time.LoadLocation("America/New_York")
	pstLocation, _ := time.LoadLocation("America/Los_Angeles")

	// EST is UTC-4 in June (DST), PST is UTC-7 in June (PDT, UTC-7)
	estTime := utcTime.In(estLocation) // Same instant, different representation
	pstTime := utcTime.In(pstLocation) // Same instant, different representation

	hash1 := ComputeExperimentHash("1.0.0", "active", 100, 5, utcTime, nil, nil)
	hash2 := ComputeExperimentHash("1.0.0", "active", 100, 5, estTime, nil, nil)
	hash3 := ComputeExperimentHash("1.0.0", "active", 100, 5, pstTime, nil, nil)

	if hash1.Hash != hash2.Hash {
		t.Errorf("Timezone should not affect hash (UTC vs EST): %s != %s", hash1.Hash, hash2.Hash)
	}
	if hash1.Hash != hash3.Hash {
		t.Errorf("Timezone should not affect hash (UTC vs PST): %s != %s", hash1.Hash, hash3.Hash)
	}
}

// TestHashAlgorithmConstant verifies the algorithm constant is correct.
func TestHashAlgorithmConstant(t *testing.T) {
	if HashAlgorithm != "SHA-256" {
		t.Errorf("Expected HashAlgorithm to be SHA-256, got %s", HashAlgorithm)
	}
}

// TestComputedAtIsSet verifies that ComputedAt is set to current time.
func TestComputedAtIsSet(t *testing.T) {
	before := time.Now()
	hash := NewHashBuilder().Build()
	after := time.Now()

	if hash.ComputedAt.Before(before) || hash.ComputedAt.After(after) {
		t.Errorf("ComputedAt should be between test start and end: %v (expected between %v and %v)",
			hash.ComputedAt, before, after)
	}
}
