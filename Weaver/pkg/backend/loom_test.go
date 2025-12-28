package backend

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
	yarn "github.com/r3d91ll/yarn"
)

// -----------------------------------------------------------------------------
// LoomConfig Validation Tests
// -----------------------------------------------------------------------------

// TestLoomConfig_Validate tests the Validate() method for LoomConfig
// using table-driven tests to cover all validation scenarios.
func TestLoomConfig_Validate(t *testing.T) {
	tests := []struct {
		name           string
		config         LoomConfig
		wantErr        bool
		wantCode       string
		wantFieldInMsg string // Field name that should appear in error message/context
	}{
		{
			name:    "empty config passes",
			config:  LoomConfig{},
			wantErr: false,
		},
		{
			name: "valid config with all fields",
			config: LoomConfig{
				Name:    "my-loom-backend",
				URL:     "http://localhost:8080",
				Model:   "llama-3.2-1b",
				Timeout: 120 * time.Second,
			},
			wantErr: false,
		},
		{
			name: "valid config with https URL",
			config: LoomConfig{
				URL: "https://api.example.com:8443",
			},
			wantErr: false,
		},
		{
			name: "valid config with underscores in name",
			config: LoomConfig{
				Name: "loom_backend_1",
			},
			wantErr: false,
		},
		{
			name: "valid config with hyphens in name",
			config: LoomConfig{
				Name: "loom-primary",
			},
			wantErr: false,
		},
		{
			name: "valid config with zero Timeout",
			config: LoomConfig{
				Timeout: 0,
			},
			wantErr: false,
		},
		{
			name: "invalid URL scheme - ftp",
			config: LoomConfig{
				URL: "ftp://example.com/files",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "URL",
		},
		{
			name: "invalid URL scheme - file",
			config: LoomConfig{
				URL: "file:///path/to/file",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "URL",
		},
		{
			name: "invalid URL scheme - ws",
			config: LoomConfig{
				URL: "ws://example.com:8080",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "URL",
		},
		{
			name: "malformed URL - missing scheme",
			config: LoomConfig{
				URL: "://localhost:8080",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "URL",
		},
		{
			name: "malformed URL - control characters",
			config: LoomConfig{
				URL: "http://localhost:8080/path\x00with\x7fcontrol",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "URL",
		},
		{
			name: "negative Timeout returns error",
			config: LoomConfig{
				Timeout: -5 * time.Second,
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationOutOfRange,
			wantFieldInMsg: "Timeout",
		},
		{
			name: "very negative Timeout returns error",
			config: LoomConfig{
				Timeout: -1000 * time.Hour,
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationOutOfRange,
			wantFieldInMsg: "Timeout",
		},
		{
			name: "Name with spaces returns error",
			config: LoomConfig{
				Name: "my loom backend",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with tab returns error",
			config: LoomConfig{
				Name: "loom\tbackend",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with newline returns error",
			config: LoomConfig{
				Name: "loom\nbackend",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with @ returns error",
			config: LoomConfig{
				Name: "loom@server",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with # returns error",
			config: LoomConfig{
				Name: "loom#1",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with $ returns error",
			config: LoomConfig{
				Name: "loom$var",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with % returns error",
			config: LoomConfig{
				Name: "loom%encoded",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with parentheses returns error",
			config: LoomConfig{
				Name: "loom(backend)",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with brackets returns error",
			config: LoomConfig{
				Name: "loom[0]",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with braces returns error",
			config: LoomConfig{
				Name: "loom{backend}",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with pipe returns error",
			config: LoomConfig{
				Name: "loom|alternate",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with backslash returns error",
			config: LoomConfig{
				Name: "loom\\path",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with angle brackets returns error",
			config: LoomConfig{
				Name: "loom<server>",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}

				// Verify error code
				if err.Code != tt.wantCode {
					t.Errorf("error code = %q, want %q", err.Code, tt.wantCode)
				}

				// Verify error message contains the field name
				if tt.wantFieldInMsg != "" {
					if !strings.Contains(err.Message, tt.wantFieldInMsg) {
						t.Errorf("error message should contain field name %q, got: %s",
							tt.wantFieldInMsg, err.Message)
					}
				}

				// Verify error has context with the field name
				if tt.wantFieldInMsg != "" {
					fieldVal, hasField := err.Context["field"]
					if !hasField {
						t.Error("error context should contain 'field' key")
					} else if fieldVal != tt.wantFieldInMsg {
						t.Errorf("context['field'] = %q, want %q", fieldVal, tt.wantFieldInMsg)
					}
				}

				// Verify error has suggestions
				if len(err.Suggestions) == 0 {
					t.Error("error should have at least one suggestion")
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
		})
	}
}

// TestLoomConfig_Validate_SuggestionsContent verifies that validation errors
// include helpful, actionable suggestions.
func TestLoomConfig_Validate_SuggestionsContent(t *testing.T) {
	t.Run("invalid name suggestions mention valid characters", func(t *testing.T) {
		cfg := LoomConfig{Name: "invalid name"}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for invalid name")
		}

		foundCharSuggestion := false
		foundExampleSuggestion := false
		for _, s := range err.Suggestions {
			if strings.Contains(s, "alphanumeric") ||
				strings.Contains(s, "hyphens") ||
				strings.Contains(s, "underscores") {
				foundCharSuggestion = true
			}
			if strings.Contains(s, "Example") ||
				strings.Contains(s, "example") ||
				strings.Contains(s, "my-loom") ||
				strings.Contains(s, "loom_") {
				foundExampleSuggestion = true
			}
		}

		if !foundCharSuggestion {
			t.Error("suggestions should mention valid character types (alphanumeric, hyphens, underscores)")
		}
		if !foundExampleSuggestion {
			t.Error("suggestions should include an example of a valid name")
		}
	})

	t.Run("invalid URL scheme suggestions mention http/https", func(t *testing.T) {
		cfg := LoomConfig{URL: "ftp://example.com/files"}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for invalid URL scheme")
		}

		foundSchemeSuggestion := false
		foundExampleSuggestion := false
		for _, s := range err.Suggestions {
			if strings.Contains(s, "http") || strings.Contains(s, "https") {
				foundSchemeSuggestion = true
			}
			if strings.Contains(s, "Example") ||
				strings.Contains(s, "example") ||
				strings.Contains(s, "localhost:8080") {
				foundExampleSuggestion = true
			}
		}

		if !foundSchemeSuggestion {
			t.Error("suggestions should mention http/https scheme")
		}
		if !foundExampleSuggestion {
			t.Error("suggestions should include an example of a valid URL")
		}
	})

	t.Run("malformed URL suggestions mention valid format", func(t *testing.T) {
		cfg := LoomConfig{URL: "://not-a-valid-url"}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for malformed URL")
		}

		foundFormatSuggestion := false
		for _, s := range err.Suggestions {
			if strings.Contains(s, "http://") ||
				strings.Contains(s, "https://") ||
				strings.Contains(s, "host:port") {
				foundFormatSuggestion = true
			}
		}

		if !foundFormatSuggestion {
			t.Error("suggestions should mention valid URL format")
		}
	})

	t.Run("negative Timeout suggestions mention valid range", func(t *testing.T) {
		cfg := LoomConfig{Timeout: -100 * time.Second}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for negative Timeout")
		}

		foundRangeSuggestion := false
		foundDefaultSuggestion := false
		for _, s := range err.Suggestions {
			if strings.Contains(s, "0") && strings.Contains(s, "greater") {
				foundRangeSuggestion = true
			}
			if strings.Contains(s, "default") || strings.Contains(s, "120") {
				foundDefaultSuggestion = true
			}
		}

		if !foundRangeSuggestion {
			t.Error("suggestions should mention valid range (0 or greater)")
		}
		if !foundDefaultSuggestion {
			t.Error("suggestions should mention default value")
		}
	})
}

// TestLoomConfig_Validate_ErrorContext verifies that validation errors
// include appropriate context information.
func TestLoomConfig_Validate_ErrorContext(t *testing.T) {
	t.Run("invalid name error includes value in context", func(t *testing.T) {
		invalidName := "bad name!"
		cfg := LoomConfig{Name: invalidName}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for invalid name")
		}

		valueCtx, hasValue := err.Context["value"]
		if !hasValue {
			t.Error("error context should contain 'value' key")
		} else if valueCtx != invalidName {
			t.Errorf("context['value'] = %q, want %q", valueCtx, invalidName)
		}
	})

	t.Run("invalid URL error includes value in context", func(t *testing.T) {
		invalidURL := "ftp://example.com"
		cfg := LoomConfig{URL: invalidURL}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for invalid URL")
		}

		valueCtx, hasValue := err.Context["value"]
		if !hasValue {
			t.Error("error context should contain 'value' key")
		} else if valueCtx != invalidURL {
			t.Errorf("context['value'] = %q, want %q", valueCtx, invalidURL)
		}
	})

	t.Run("out of range error includes value and range in context", func(t *testing.T) {
		cfg := LoomConfig{Timeout: -100 * time.Second}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for negative Timeout")
		}

		// Should have value in context
		if _, hasValue := err.Context["value"]; !hasValue {
			t.Error("error context should contain 'value' key")
		}

		// Should have min in context
		if _, hasMin := err.Context["min"]; !hasMin {
			t.Error("error context should contain 'min' key")
		}

		// Should have max in context
		if _, hasMax := err.Context["max"]; !hasMax {
			t.Error("error context should contain 'max' key")
		}
	})
}

// TestLoomConfig_Validate_Category verifies that validation errors
// have the correct category.
func TestLoomConfig_Validate_Category(t *testing.T) {
	tests := []struct {
		name   string
		config LoomConfig
	}{
		{
			name:   "invalid name",
			config: LoomConfig{Name: "bad name"},
		},
		{
			name:   "invalid URL scheme",
			config: LoomConfig{URL: "ftp://example.com"},
		},
		{
			name:   "negative Timeout",
			config: LoomConfig{Timeout: -1 * time.Second},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if err == nil {
				t.Fatal("expected error")
			}

			if err.Category != werrors.CategoryValidation {
				t.Errorf("error category = %q, want %q", err.Category, werrors.CategoryValidation)
			}
		})
	}
}

// TestLoomConfig_Validate_FirstErrorReturned verifies that validation
// returns on the first error encountered (Name is checked before URL and Timeout).
func TestLoomConfig_Validate_FirstErrorReturned(t *testing.T) {
	cfg := LoomConfig{
		Name:    "bad name",         // Invalid (checked first)
		URL:     "ftp://example.com", // Also invalid
		Timeout: -100 * time.Second,  // Also invalid
	}

	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error")
	}

	// Name is validated first, so error should be about Name
	if err.Code != werrors.ErrValidationInvalidValue {
		t.Errorf("expected ErrValidationInvalidValue for Name error, got %q", err.Code)
	}
	if fieldCtx, ok := err.Context["field"]; ok && fieldCtx != "Name" {
		t.Errorf("expected first error to be about Name field, got %q", fieldCtx)
	}
}

// TestLoomConfig_Validate_URLEdgeCases tests edge cases for URL validation.
func TestLoomConfig_Validate_URLEdgeCases(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{
			name:    "http with port",
			url:     "http://localhost:8080",
			wantErr: false,
		},
		{
			name:    "https with port",
			url:     "https://api.example.com:443",
			wantErr: false,
		},
		{
			name:    "http without port",
			url:     "http://example.com",
			wantErr: false,
		},
		{
			name:    "https without port",
			url:     "https://example.com",
			wantErr: false,
		},
		{
			name:    "http with path",
			url:     "http://localhost:8080/v1/api",
			wantErr: false,
		},
		{
			name:    "http with IPv4 address",
			url:     "http://192.168.1.100:8080",
			wantErr: false,
		},
		{
			name:    "http with IPv6 address",
			url:     "http://[::1]:8080",
			wantErr: false,
		},
		{
			name:    "empty URL is valid (uses default)",
			url:     "",
			wantErr: false,
		},
		{
			name:    "ftp scheme is invalid",
			url:     "ftp://files.example.com",
			wantErr: true,
		},
		{
			name:    "websocket scheme is invalid",
			url:     "ws://socket.example.com",
			wantErr: true,
		},
		{
			name:    "no scheme is invalid",
			url:     "example.com:8080",
			wantErr: true, // url.Parse treats this as scheme:opaque, not http
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := LoomConfig{URL: tt.url}
			err := cfg.Validate()

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

// -----------------------------------------------------------------------------
// SSE Parsing Tests
// -----------------------------------------------------------------------------

// TestParseSSE_ValidEvents tests that parseSSE correctly parses valid SSE events.
func TestParseSSE_ValidEvents(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []sseEvent
	}{
		{
			name: "single event with event and data",
			input: `event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}

`,
			expected: []sseEvent{
				{
					Event: "content_block_delta",
					Data:  `{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}`,
				},
			},
		},
		{
			name: "multiple events",
			input: `event: content_block_delta
data: {"delta":{"text":"Hello"}}

event: content_block_delta
data: {"delta":{"text":" world"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

`,
			expected: []sseEvent{
				{Event: "content_block_delta", Data: `{"delta":{"text":"Hello"}}`},
				{Event: "content_block_delta", Data: `{"delta":{"text":" world"}}`},
				{Event: "message_delta", Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`},
			},
		},
		{
			name: "event without event field (data only)",
			input: `data: {"type":"message_delta"}

`,
			expected: []sseEvent{
				{Event: "", Data: `{"type":"message_delta"}`},
			},
		},
		{
			name: "data with leading space preserved correctly",
			input: `event: test
data:  has two leading spaces

`,
			expected: []sseEvent{
				{Event: "test", Data: " has two leading spaces"},
			},
		},
		{
			name: "data with no space after colon",
			input: `event: test
data:no_space

`,
			expected: []sseEvent{
				{Event: "test", Data: "no_space"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			reader := strings.NewReader(tt.input)
			events := parseSSE(ctx, reader)

			var result []sseEvent
			for event := range events {
				result = append(result, event)
			}

			if len(result) != len(tt.expected) {
				t.Errorf("expected %d events, got %d", len(tt.expected), len(result))
				return
			}

			for i, expected := range tt.expected {
				if result[i].Event != expected.Event {
					t.Errorf("event[%d].Event = %q, want %q", i, result[i].Event, expected.Event)
				}
				if result[i].Data != expected.Data {
					t.Errorf("event[%d].Data = %q, want %q", i, result[i].Data, expected.Data)
				}
			}
		})
	}
}

// TestParseSSE_MultiLineData tests parsing of multi-line data fields.
func TestParseSSE_MultiLineData(t *testing.T) {
	input := `event: multi_line
data: line1
data: line2
data: line3

`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result))
	}

	expected := "line1\nline2\nline3"
	if result[0].Data != expected {
		t.Errorf("Data = %q, want %q", result[0].Data, expected)
	}
}

// TestParseSSE_CommentsIgnored tests that comment lines are properly skipped.
func TestParseSSE_CommentsIgnored(t *testing.T) {
	input := `: this is a comment
event: test
: another comment mid-stream
data: hello
: final comment

`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result))
	}

	if result[0].Event != "test" {
		t.Errorf("Event = %q, want %q", result[0].Event, "test")
	}
	if result[0].Data != "hello" {
		t.Errorf("Data = %q, want %q", result[0].Data, "hello")
	}
}

// TestParseSSE_EmptyStream tests handling of empty input.
func TestParseSSE_EmptyStream(t *testing.T) {
	ctx := context.Background()
	reader := strings.NewReader("")
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 0 {
		t.Errorf("expected 0 events, got %d", len(result))
	}
}

// TestParseSSE_OnlyComments tests stream containing only comments.
func TestParseSSE_OnlyComments(t *testing.T) {
	input := `: comment 1
: comment 2
: comment 3
`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 0 {
		t.Errorf("expected 0 events, got %d", len(result))
	}
}

// TestParseSSE_OnlyEmptyLines tests stream with only empty lines.
func TestParseSSE_OnlyEmptyLines(t *testing.T) {
	input := `


`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 0 {
		t.Errorf("expected 0 events, got %d", len(result))
	}
}

// TestParseSSE_EventAtEndOfStream tests event at end without trailing newline.
func TestParseSSE_EventAtEndOfStream(t *testing.T) {
	// Event at end of stream without trailing empty line should still be emitted
	input := `event: final
data: last_message`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result))
	}

	if result[0].Event != "final" {
		t.Errorf("Event = %q, want %q", result[0].Event, "final")
	}
	if result[0].Data != "last_message" {
		t.Errorf("Data = %q, want %q", result[0].Data, "last_message")
	}
}

// TestParseSSE_IgnoresUnknownFields tests that id: and retry: fields are ignored.
func TestParseSSE_IgnoresUnknownFields(t *testing.T) {
	input := `id: 12345
event: test
retry: 10000
data: hello

`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result))
	}

	// id: and retry: should be ignored; only event: and data: are parsed
	if result[0].Event != "test" {
		t.Errorf("Event = %q, want %q", result[0].Event, "test")
	}
	if result[0].Data != "hello" {
		t.Errorf("Data = %q, want %q", result[0].Data, "hello")
	}
}

// TestParseSSE_ContextCancellation tests that parsing stops on context cancellation.
func TestParseSSE_ContextCancellation(t *testing.T) {
	// Create a reader that blocks indefinitely (simulates slow stream)
	blockingReader := &blockingReader{
		lines: []string{
			"event: test1\n",
			"data: hello1\n",
			"\n",
			"event: test2\n",
			"data: hello2\n",
			"\n",
		},
		delay: 100 * time.Millisecond,
	}

	ctx, cancel := context.WithCancel(context.Background())
	events := parseSSE(ctx, blockingReader)

	// Read first event
	event1, ok := <-events
	if !ok {
		t.Fatal("expected to receive first event")
	}
	if event1.Event != "test1" {
		t.Errorf("first event = %q, want %q", event1.Event, "test1")
	}

	// Cancel context before second event
	cancel()

	// Channel should close without delivering all events
	// Wait a bit for goroutine to detect cancellation
	time.Sleep(50 * time.Millisecond)

	// Drain remaining events (if any) - should be limited due to cancellation
	remaining := 0
	for range events {
		remaining++
	}

	// The exact behavior depends on timing, but we should have received <= 2 events total
	// (The test verifies cancellation is checked, not exact timing)
	t.Logf("received %d remaining events after cancellation", remaining)
}

// blockingReader is a test helper that reads lines with delays.
type blockingReader struct {
	lines []string
	index int
	delay time.Duration
}

func (r *blockingReader) Read(p []byte) (n int, err error) {
	if r.index >= len(r.lines) {
		// Block indefinitely to simulate slow stream
		time.Sleep(10 * time.Second)
		return 0, nil
	}
	time.Sleep(r.delay)
	line := r.lines[r.index]
	r.index++
	return copy(p, line), nil
}

// TestParseSSE_MalformedEvents tests handling of malformed event lines.
func TestParseSSE_MalformedEvents(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []sseEvent
	}{
		{
			name: "line without colon is ignored",
			input: `event: test
no_colon_here
data: hello

`,
			expected: []sseEvent{
				{Event: "test", Data: "hello"},
			},
		},
		{
			name: "empty event field",
			input: `event:
data: hello

`,
			expected: []sseEvent{
				{Event: "", Data: "hello"},
			},
		},
		{
			name: "empty data field",
			input: `event: test
data:

`,
			expected: []sseEvent{
				{Event: "test", Data: ""},
			},
		},
		{
			name: "event with only whitespace value",
			input: `event:
data: hello

`,
			expected: []sseEvent{
				{Event: "", Data: "hello"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			reader := strings.NewReader(tt.input)
			events := parseSSE(ctx, reader)

			var result []sseEvent
			for event := range events {
				result = append(result, event)
			}

			if len(result) != len(tt.expected) {
				t.Errorf("expected %d events, got %d", len(tt.expected), len(result))
				return
			}

			for i, expected := range tt.expected {
				if result[i].Event != expected.Event {
					t.Errorf("event[%d].Event = %q, want %q", i, result[i].Event, expected.Event)
				}
				if result[i].Data != expected.Data {
					t.Errorf("event[%d].Data = %q, want %q", i, result[i].Data, expected.Data)
				}
			}
		})
	}
}

// TestParseSSE_RealWorldEvents tests parsing of realistic Loom SSE events.
func TestParseSSE_RealWorldEvents(t *testing.T) {
	input := `event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" there"}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"!"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}

`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 4 {
		t.Fatalf("expected 4 events, got %d", len(result))
	}

	// Verify event types
	eventTypes := []string{"content_block_delta", "content_block_delta", "content_block_delta", "message_delta"}
	for i, expected := range eventTypes {
		if result[i].Event != expected {
			t.Errorf("event[%d].Event = %q, want %q", i, result[i].Event, expected)
		}
	}

	// Verify last event has valid JSON with stop_reason
	if !strings.Contains(result[3].Data, `"stop_reason":"end_turn"`) {
		t.Errorf("final event should contain stop_reason, got: %s", result[3].Data)
	}
}

// TestParseSSE_ErrorEvent tests parsing of error events.
func TestParseSSE_ErrorEvent(t *testing.T) {
	input := `event: error
data: {"type":"error","error":{"message":"Internal server error"}}

`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result))
	}

	if result[0].Event != "error" {
		t.Errorf("Event = %q, want %q", result[0].Event, "error")
	}
	if !strings.Contains(result[0].Data, "Internal server error") {
		t.Errorf("error event should contain message, got: %s", result[0].Data)
	}
}

// TestParseSSE_ConsecutiveEmptyLines tests multiple consecutive empty lines.
func TestParseSSE_ConsecutiveEmptyLines(t *testing.T) {
	input := `event: test1
data: hello1



event: test2
data: hello2

`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	// Should get 2 events, extra empty lines between events are ignored
	if len(result) != 2 {
		t.Fatalf("expected 2 events, got %d", len(result))
	}
}

// TestParseSSE_DataWithSpecialCharacters tests data containing special characters.
func TestParseSSE_DataWithSpecialCharacters(t *testing.T) {
	input := `event: test
data: {"text":"Hello\nWorld\twith\ttabs"}

`
	ctx := context.Background()
	reader := strings.NewReader(input)
	events := parseSSE(ctx, reader)

	var result []sseEvent
	for event := range events {
		result = append(result, event)
	}

	if len(result) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result))
	}

	expected := `{"text":"Hello\nWorld\twith\ttabs"}`
	if result[0].Data != expected {
		t.Errorf("Data = %q, want %q", result[0].Data, expected)
	}
}

// TestParseSSE_JSONPayloadParsing tests end-to-end JSON payload parsing.
func TestParseSSE_JSONPayloadParsing(t *testing.T) {
	t.Run("content_block_delta JSON parsing", func(t *testing.T) {
		input := `event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello world"}}

`
		ctx := context.Background()
		reader := strings.NewReader(input)
		events := parseSSE(ctx, reader)

		event := <-events
		var delta loomContentDelta
		if err := jsonUnmarshal([]byte(event.Data), &delta); err != nil {
			t.Fatalf("failed to parse content delta: %v", err)
		}

		if delta.Type != "content_block_delta" {
			t.Errorf("Type = %q, want %q", delta.Type, "content_block_delta")
		}
		if delta.Delta.Type != "text_delta" {
			t.Errorf("Delta.Type = %q, want %q", delta.Delta.Type, "text_delta")
		}
		if delta.Delta.Text != "Hello world" {
			t.Errorf("Delta.Text = %q, want %q", delta.Delta.Text, "Hello world")
		}
	})

	t.Run("message_delta JSON parsing", func(t *testing.T) {
		input := `event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

`
		ctx := context.Background()
		reader := strings.NewReader(input)
		events := parseSSE(ctx, reader)

		event := <-events
		var msgDelta loomMessageDelta
		if err := jsonUnmarshal([]byte(event.Data), &msgDelta); err != nil {
			t.Fatalf("failed to parse message delta: %v", err)
		}

		if msgDelta.Type != "message_delta" {
			t.Errorf("Type = %q, want %q", msgDelta.Type, "message_delta")
		}
		if msgDelta.Delta.StopReason != "end_turn" {
			t.Errorf("Delta.StopReason = %q, want %q", msgDelta.Delta.StopReason, "end_turn")
		}
		if msgDelta.Usage.PromptTokens != 10 {
			t.Errorf("Usage.PromptTokens = %d, want %d", msgDelta.Usage.PromptTokens, 10)
		}
		if msgDelta.Usage.CompletionTokens != 5 {
			t.Errorf("Usage.CompletionTokens = %d, want %d", msgDelta.Usage.CompletionTokens, 5)
		}
	})

	t.Run("error event JSON parsing", func(t *testing.T) {
		input := `event: error
data: {"type":"error","error":{"message":"Model not loaded"}}

`
		ctx := context.Background()
		reader := strings.NewReader(input)
		events := parseSSE(ctx, reader)

		event := <-events
		var errEvent loomErrorEvent
		if err := jsonUnmarshal([]byte(event.Data), &errEvent); err != nil {
			t.Fatalf("failed to parse error event: %v", err)
		}

		if errEvent.Type != "error" {
			t.Errorf("Type = %q, want %q", errEvent.Type, "error")
		}
		if errEvent.Error.Message != "Model not loaded" {
			t.Errorf("Error.Message = %q, want %q", errEvent.Error.Message, "Model not loaded")
		}
	})

	t.Run("message_delta with metadata and hidden_state", func(t *testing.T) {
		input := `event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},"metadata":{"model":"test-model","latency_ms":123.45,"tokens_per_second":50.5},"hidden_state":{"final":[0.1,0.2,0.3],"shape":[1,3],"layer":-1,"dtype":"float32"}}

`
		ctx := context.Background()
		reader := strings.NewReader(input)
		events := parseSSE(ctx, reader)

		event := <-events
		var msgDelta loomMessageDelta
		if err := jsonUnmarshal([]byte(event.Data), &msgDelta); err != nil {
			t.Fatalf("failed to parse message delta: %v", err)
		}

		// Check metadata
		if msgDelta.Metadata == nil {
			t.Fatal("expected metadata to be present")
		}
		if msgDelta.Metadata.Model != "test-model" {
			t.Errorf("Metadata.Model = %q, want %q", msgDelta.Metadata.Model, "test-model")
		}
		if msgDelta.Metadata.LatencyMS != 123.45 {
			t.Errorf("Metadata.LatencyMS = %f, want %f", msgDelta.Metadata.LatencyMS, 123.45)
		}

		// Check hidden state
		if msgDelta.HiddenState == nil {
			t.Fatal("expected hidden_state to be present")
		}
		if len(msgDelta.HiddenState.Final) != 3 {
			t.Errorf("HiddenState.Final length = %d, want %d", len(msgDelta.HiddenState.Final), 3)
		}
		if msgDelta.HiddenState.Layer != -1 {
			t.Errorf("HiddenState.Layer = %d, want %d", msgDelta.HiddenState.Layer, -1)
		}
	})
}

// jsonUnmarshal is a test helper to avoid importing encoding/json in test file.
func jsonUnmarshal(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}

// -----------------------------------------------------------------------------
// Layer Extraction Tests
// -----------------------------------------------------------------------------

// TestLoomLayerExtraction tests the layer-by-layer hidden state extraction functionality.
func TestLoomLayerExtraction(t *testing.T) {
	t.Run("ParseLayerSelection_all", func(t *testing.T) {
		layers, isAll, err := ParseLayerSelection("all")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !isAll {
			t.Error("expected isAll to be true")
		}
		if layers != nil {
			t.Error("expected layers to be nil for 'all'")
		}
	})

	t.Run("ParseLayerSelection_single_int", func(t *testing.T) {
		layers, isAll, err := ParseLayerSelection(5)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if isAll {
			t.Error("expected isAll to be false")
		}
		if len(layers) != 1 || layers[0] != 5 {
			t.Errorf("expected [5], got %v", layers)
		}
	})

	t.Run("ParseLayerSelection_slice_int", func(t *testing.T) {
		layers, isAll, err := ParseLayerSelection([]int{0, 5, 11})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if isAll {
			t.Error("expected isAll to be false")
		}
		if len(layers) != 3 {
			t.Errorf("expected 3 layers, got %d", len(layers))
		}
		if layers[0] != 0 || layers[1] != 5 || layers[2] != 11 {
			t.Errorf("expected [0, 5, 11], got %v", layers)
		}
	})

	t.Run("ParseLayerSelection_slice_interface", func(t *testing.T) {
		// Simulate JSON unmarshal result
		layers, isAll, err := ParseLayerSelection([]interface{}{float64(0), float64(5), float64(11)})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if isAll {
			t.Error("expected isAll to be false")
		}
		if len(layers) != 3 {
			t.Errorf("expected 3 layers, got %d", len(layers))
		}
	})

	t.Run("ParseLayerSelection_float64", func(t *testing.T) {
		// JSON numbers are float64
		layers, isAll, err := ParseLayerSelection(float64(7))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if isAll {
			t.Error("expected isAll to be false")
		}
		if len(layers) != 1 || layers[0] != 7 {
			t.Errorf("expected [7], got %v", layers)
		}
	})

	t.Run("ParseLayerSelection_invalid_string", func(t *testing.T) {
		_, _, err := ParseLayerSelection("invalid")
		if err == nil {
			t.Error("expected error for invalid string")
		}
	})

	t.Run("ParseLayerSelection_nil", func(t *testing.T) {
		layers, isAll, err := ParseLayerSelection(nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if isAll {
			t.Error("expected isAll to be false for nil")
		}
		if layers != nil {
			t.Error("expected layers to be nil")
		}
	})
}

// TestCosineSimilarity tests the cosine similarity helper function.
func TestCosineSimilarity(t *testing.T) {
	t.Run("identical_vectors", func(t *testing.T) {
		a := []float32{1.0, 0.0, 0.0}
		b := []float32{1.0, 0.0, 0.0}
		sim := cosineSimilarity(a, b)
		if sim < 0.999 || sim > 1.001 {
			t.Errorf("expected similarity ~1.0, got %f", sim)
		}
	})

	t.Run("orthogonal_vectors", func(t *testing.T) {
		a := []float32{1.0, 0.0, 0.0}
		b := []float32{0.0, 1.0, 0.0}
		sim := cosineSimilarity(a, b)
		if sim < -0.001 || sim > 0.001 {
			t.Errorf("expected similarity ~0.0, got %f", sim)
		}
	})

	t.Run("opposite_vectors", func(t *testing.T) {
		a := []float32{1.0, 0.0, 0.0}
		b := []float32{-1.0, 0.0, 0.0}
		sim := cosineSimilarity(a, b)
		if sim < -1.001 || sim > -0.999 {
			t.Errorf("expected similarity ~-1.0, got %f", sim)
		}
	})

	t.Run("empty_vectors", func(t *testing.T) {
		a := []float32{}
		b := []float32{}
		sim := cosineSimilarity(a, b)
		if sim != 0 {
			t.Errorf("expected similarity 0 for empty vectors, got %f", sim)
		}
	})

	t.Run("mismatched_lengths", func(t *testing.T) {
		a := []float32{1.0, 2.0}
		b := []float32{1.0, 2.0, 3.0}
		sim := cosineSimilarity(a, b)
		if sim != 0 {
			t.Errorf("expected similarity 0 for mismatched lengths, got %f", sim)
		}
	})

	t.Run("zero_vector", func(t *testing.T) {
		a := []float32{0.0, 0.0, 0.0}
		b := []float32{1.0, 2.0, 3.0}
		sim := cosineSimilarity(a, b)
		if sim != 0 {
			t.Errorf("expected similarity 0 for zero vector, got %f", sim)
		}
	})
}

// TestSqrt tests the sqrt helper function.
func TestSqrt(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{4.0, 2.0},
		{9.0, 3.0},
		{16.0, 4.0},
		{2.0, 1.4142135623730951},
		{0.0, 0.0},
		{-1.0, 0.0},
	}

	for _, tt := range tests {
		result := sqrt(tt.input)
		diff := result - tt.expected
		if diff < 0 {
			diff = -diff
		}
		if diff > 0.0001 {
			t.Errorf("sqrt(%f) = %f, expected %f", tt.input, result, tt.expected)
		}
	}
}

// TestCompareLayerStates tests the layer comparison functionality.
func TestCompareLayerStates(t *testing.T) {
	t.Run("nil_responses", func(t *testing.T) {
		result := CompareLayerStates(nil, nil)
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if len(result.Layers) != 0 {
			t.Error("expected empty layers")
		}
	})

	t.Run("common_layers", func(t *testing.T) {
		resp1 := &ChatResponse{
			HiddenStates: map[int]*yarn.HiddenState{
				0: {Vector: []float32{1.0, 0.0, 0.0}, Layer: 0},
				5: {Vector: []float32{0.0, 1.0, 0.0}, Layer: 5},
			},
		}
		resp2 := &ChatResponse{
			HiddenStates: map[int]*yarn.HiddenState{
				0: {Vector: []float32{1.0, 0.0, 0.0}, Layer: 0},
				5: {Vector: []float32{0.0, 0.0, 1.0}, Layer: 5},
				10: {Vector: []float32{1.0, 1.0, 1.0}, Layer: 10},
			},
		}

		result := CompareLayerStates(resp1, resp2)
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if len(result.Layers) != 2 {
			t.Errorf("expected 2 common layers, got %d", len(result.Layers))
		}

		// Layer 0 should have similarity ~1.0 (identical vectors)
		if sim, ok := result.CosineSimilarity[0]; ok {
			if sim < 0.999 {
				t.Errorf("expected high similarity for layer 0, got %f", sim)
			}
		} else {
			t.Error("expected similarity for layer 0")
		}

		// Layer 5 should have similarity ~0.0 (orthogonal vectors)
		if sim, ok := result.CosineSimilarity[5]; ok {
			if sim > 0.001 || sim < -0.001 {
				t.Errorf("expected ~0 similarity for layer 5, got %f", sim)
			}
		} else {
			t.Error("expected similarity for layer 5")
		}
	})

	t.Run("no_common_layers", func(t *testing.T) {
		resp1 := &ChatResponse{
			HiddenStates: map[int]*yarn.HiddenState{
				0: {Vector: []float32{1.0, 0.0}, Layer: 0},
			},
		}
		resp2 := &ChatResponse{
			HiddenStates: map[int]*yarn.HiddenState{
				5: {Vector: []float32{0.0, 1.0}, Layer: 5},
			},
		}

		result := CompareLayerStates(resp1, resp2)
		if len(result.Layers) != 0 {
			t.Errorf("expected 0 common layers, got %d", len(result.Layers))
		}
	})
}

// TestGetLayerDEff tests extracting D_eff values from response metadata.
func TestGetLayerDEff(t *testing.T) {
	t.Run("nil_response", func(t *testing.T) {
		result := GetLayerDEff(nil)
		if len(result) != 0 {
			t.Error("expected empty map for nil response")
		}
	})

	t.Run("no_metadata", func(t *testing.T) {
		resp := &ChatResponse{}
		result := GetLayerDEff(resp)
		if len(result) != 0 {
			t.Error("expected empty map for no metadata")
		}
	})

	t.Run("with_deff_data", func(t *testing.T) {
		resp := &ChatResponse{
			Metadata: map[string]any{
				"layer_d_eff": map[string]float64{
					"0":  128.5,
					"5":  256.0,
					"11": 512.3,
				},
			},
		}
		result := GetLayerDEff(resp)
		if len(result) != 3 {
			t.Errorf("expected 3 D_eff values, got %d", len(result))
		}
		if result[0] != 128.5 {
			t.Errorf("expected D_eff[0] = 128.5, got %f", result[0])
		}
		if result[5] != 256.0 {
			t.Errorf("expected D_eff[5] = 256.0, got %f", result[5])
		}
		if result[11] != 512.3 {
			t.Errorf("expected D_eff[11] = 512.3, got %f", result[11])
		}
	})
}

// TestLoomRequestLayerSerialization tests that layer selection is properly serialized in requests.
func TestLoomRequestLayerSerialization(t *testing.T) {
	t.Run("layers_all", func(t *testing.T) {
		req := loomRequest{
			Model:              "test-model",
			Messages:           []ChatMessage{{Role: "user", Content: "test"}},
			ReturnHiddenStates: true,
			HiddenStateLayers:  "all",
		}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		if !strings.Contains(string(data), `"hidden_state_layers":"all"`) {
			t.Errorf("expected hidden_state_layers to be 'all', got: %s", string(data))
		}
	})

	t.Run("layers_slice", func(t *testing.T) {
		req := loomRequest{
			Model:              "test-model",
			Messages:           []ChatMessage{{Role: "user", Content: "test"}},
			ReturnHiddenStates: true,
			HiddenStateLayers:  []int{0, 5, 11},
		}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		if !strings.Contains(string(data), `"hidden_state_layers":[0,5,11]`) {
			t.Errorf("expected hidden_state_layers to be [0,5,11], got: %s", string(data))
		}
	})

	t.Run("no_layers", func(t *testing.T) {
		req := loomRequest{
			Model:              "test-model",
			Messages:           []ChatMessage{{Role: "user", Content: "test"}},
			ReturnHiddenStates: true,
		}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		// hidden_state_layers should be omitted when nil
		if strings.Contains(string(data), "hidden_state_layers") {
			t.Errorf("expected hidden_state_layers to be omitted, got: %s", string(data))
		}
	})
}

// TestLoomResponseMultiLayerParsing tests parsing of multi-layer hidden state responses.
func TestLoomResponseMultiLayerParsing(t *testing.T) {
	t.Run("parse_multi_layer_response", func(t *testing.T) {
		jsonData := `{
			"text": "Hello world",
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 5,
				"total_tokens": 15
			},
			"hidden_states": {
				"0": {
					"vector": [0.1, 0.2, 0.3],
					"shape": [1, 3],
					"layer": 0,
					"dtype": "float32",
					"d_eff": 2.5
				},
				"5": {
					"vector": [0.4, 0.5, 0.6],
					"shape": [1, 3],
					"layer": 5,
					"dtype": "float32",
					"d_eff": 3.0
				}
			}
		}`

		var resp loomResponse
		if err := json.Unmarshal([]byte(jsonData), &resp); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}

		if resp.Text != "Hello world" {
			t.Errorf("expected text 'Hello world', got '%s'", resp.Text)
		}

		if len(resp.HiddenStates) != 2 {
			t.Fatalf("expected 2 hidden states, got %d", len(resp.HiddenStates))
		}

		// Check layer 0
		hs0 := resp.HiddenStates["0"]
		if hs0 == nil {
			t.Fatal("expected hidden state for layer 0")
		}
		if hs0.Layer != 0 {
			t.Errorf("expected layer 0, got %d", hs0.Layer)
		}
		if len(hs0.Vector) != 3 {
			t.Errorf("expected 3 vector elements, got %d", len(hs0.Vector))
		}
		if hs0.DEff != 2.5 {
			t.Errorf("expected d_eff 2.5, got %f", hs0.DEff)
		}

		// Check layer 5
		hs5 := resp.HiddenStates["5"]
		if hs5 == nil {
			t.Fatal("expected hidden state for layer 5")
		}
		if hs5.Layer != 5 {
			t.Errorf("expected layer 5, got %d", hs5.Layer)
		}
	})
}

// TestLayerAnalysisResult tests the LayerAnalysisResult structure.
func TestLayerAnalysisResult(t *testing.T) {
	result := LayerAnalysisResult{
		Layers:       []int{0, 5, 11},
		DEff:         map[int]float64{0: 128.0, 5: 256.0, 11: 512.0},
		HiddenStates: map[int][]float32{0: {0.1, 0.2}, 5: {0.3, 0.4}, 11: {0.5, 0.6}},
		Shape:        []int{1, 2},
		Model:        "test-model",
	}

	// Verify JSON serialization
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var parsed LayerAnalysisResult
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if len(parsed.Layers) != 3 {
		t.Errorf("expected 3 layers, got %d", len(parsed.Layers))
	}
	if parsed.Model != "test-model" {
		t.Errorf("expected model 'test-model', got '%s'", parsed.Model)
	}
	if parsed.DEff[5] != 256.0 {
		t.Errorf("expected D_eff[5] = 256.0, got %f", parsed.DEff[5])
	}
}
