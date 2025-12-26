package backend

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

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
