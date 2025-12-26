# Streaming Integration Verification Guide

This document describes how to verify that Weaver can stream from a running Loom server.

## Prerequisites

1. **GPU with sufficient VRAM** (8GB+ recommended for TinyLlama)
2. **The Loom server** with a model loaded
3. **Weaver** built and ready to use

## Quick Verification

### 1. Start The Loom Server

```bash
cd TheLoom/the-loom
python -m uvicorn src.transport.http:create_http_app --factory --host 0.0.0.0 --port 8080
```

### 2. Load a Model

```bash
curl -X POST http://localhost:8080/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "dtype": "float16"}'
```

### 3. Test Streaming with curl

This command tests the SSE streaming endpoint directly:

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "max_tokens": 50,
    "temperature": 0.7,
    "stream": true
  }'
```

You should see SSE events appearing incrementally:

```
event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"1"}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":","}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" "}}

...

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{...}}
```

**Key verification points:**
- Tokens appear one at a time, not all at once
- Each token has its own `content_block_delta` event
- The stream ends with a `message_delta` event

## Running the Integration Tests

### Python Integration Tests (Server-side)

```bash
cd TheLoom/the-loom
pytest tests/test_integration.py::TestStreamingChatCompletionsIntegration -v -s
```

This runs three tests:
1. `test_streaming_produces_incremental_tokens` - Verifies tokens arrive incrementally
2. `test_streaming_with_hidden_states` - Verifies hidden states are returned in final event
3. `test_streaming_vs_nonstreaming_equivalence` - Compares streaming vs non-streaming output

### Go Unit Tests (Client-side)

```bash
cd Weaver
go test ./pkg/backend/... -v -run TestParseSSE
```

This tests the SSE parser logic used by the Weaver Loom backend.

## Manual End-to-End Test with Weaver

### 1. Configure Weaver to Use Loom Backend

Ensure your Weaver configuration has a Loom backend configured:

```yaml
backends:
  - name: loom
    type: loom
    url: http://localhost:8080
    model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 2. Use Weaver to Chat

When using Weaver with the Loom backend, you should observe:
- Text appearing incrementally in the terminal
- Real-time token display, not buffered output
- Smooth streaming experience similar to Claude Code

## Acceptance Criteria

The streaming implementation is verified when:

1. **Streaming works with real Loom server** - SSE events are properly parsed
2. **Tokens appear incrementally** - Not all at once, but one-by-one as generated
3. **Hidden states are returned** - In the final `message_delta` event when requested
4. **Error handling works** - Errors are properly propagated as `error` events

## Troubleshooting

### No streaming output
- Check the server is running: `curl http://localhost:8080/health`
- Verify the model is loaded: `curl http://localhost:8080/models`
- Check for firewall/port issues

### All tokens appear at once
- This could indicate buffering in the network stack
- Try disabling nginx/proxy buffering (`X-Accel-Buffering: no` header)
- Verify the client is reading the stream correctly

### Hidden states not appearing
- Ensure `return_hidden_states: true` is set in the request
- Hidden states only appear in the final `message_delta` event
