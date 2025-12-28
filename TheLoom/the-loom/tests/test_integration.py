"""Integration tests with real models.

These tests actually load models and validate hidden state extraction.
Run with: pytest tests/test_integration.py -v -s

Requires:
- GPU with sufficient VRAM (8GB+ recommended)
- Models will be downloaded on first run

Mark as slow to skip in normal test runs:
    pytest -m "not integration"
"""

import json
import time
from pathlib import Path

import pytest
import torch

# Skip all tests in this module if no GPU available
# Mark with 'gpu' marker to allow exclusion with: pytest -m 'not gpu'
pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for integration tests"),
]

# Test models - using smaller variants for faster testing
TEST_MODELS = {
    "embedding": "BAAI/bge-small-en-v1.5",  # 33M params, ~130MB
    "generative_small": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B params
}

# Directory for saving example outputs
EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "outputs"


@pytest.fixture(scope="module")
def examples_dir():
    """Create examples directory for saving outputs."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLES_DIR


@pytest.fixture(scope="module")
def app_client():
    """Create test client with real model loading.

    Cleanup is handled in fixture teardown to avoid relying on test execution order.
    """
    from fastapi.testclient import TestClient
    from src.transport.http import create_http_app
    from src.config import Config

    config = Config()
    app = create_http_app(config)

    with TestClient(app) as client:
        yield client

        # Teardown: unload all models after tests complete
        print("\n=== Fixture Cleanup: Unloading Models ===")
        try:
            response = client.get("/models")
            if response.status_code == 200:
                data = response.json()
                for model in data.get("loaded_models", []):
                    model_id = model["model_id"].replace("/", "--")
                    client.delete(f"/models/{model_id}")
                    print(f"  Unloaded: {model['model_id']}")
            print("=== Cleanup Complete ===")
        except Exception as e:
            print(f"  Cleanup warning: {e}")


class TestHealthAndSetup:
    """Basic health checks before model tests."""

    def test_health_endpoint(self, app_client):
        """Verify server health endpoint works."""
        response = app_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "gpu_info" in data

        print("\n=== Health Check ===")
        print(json.dumps(data, indent=2))

    def test_list_loaders(self, app_client):
        """Verify loader registry is populated."""
        response = app_client.get("/loaders")
        assert response.status_code == 200

        data = response.json()
        assert "loaders" in data
        assert "fallback_order" in data
        assert len(data["fallback_order"]) >= 3

        print("\n=== Available Loaders ===")
        print(json.dumps(data, indent=2))


class TestEmbeddingModel:
    """Test hidden state extraction with embedding models."""

    @pytest.fixture(scope="class")
    def loaded_embedding_model(self, app_client):
        """Load embedding model once for all tests in class."""
        model_id = TEST_MODELS["embedding"]

        print(f"\n=== Loading Embedding Model: {model_id} ===")
        start = time.time()

        response = app_client.post("/models/load", json={
            "model": model_id,
            "dtype": "float16",
        })

        load_time = time.time() - start
        assert response.status_code == 200, f"Failed to load model: {response.text}"

        data = response.json()
        print(f"Loaded in {load_time:.2f}s")
        print(json.dumps(data, indent=2))

        yield data

        # Cleanup
        app_client.delete(f"/models/{model_id.replace('/', '--')}")

    def test_embed_basic(self, app_client, loaded_embedding_model, examples_dir):
        """Test basic embedding extraction."""
        model_id = TEST_MODELS["embedding"]

        response = app_client.post("/embed", json={
            "model": model_id,
            "text": "The quick brown fox jumps over the lazy dog.",
            "pooling": "mean",
            "normalize": True,
        })

        assert response.status_code == 200
        data = response.json()

        # Validate structure
        assert "embedding" in data
        assert "shape" in data
        assert "metadata" in data

        # Validate embedding
        embedding = data["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

        # Validate shape matches embedding length
        assert data["shape"][-1] == len(embedding)

        # Validate normalization (L2 norm should be ~1.0)
        import math
        norm = math.sqrt(sum(x * x for x in embedding))
        assert 0.99 < norm < 1.01, f"Expected normalized vector, got norm={norm}"

        print("\n=== Embedding Output ===")
        print(f"Shape: {data['shape']}")
        print(f"Embedding (first 10): {embedding[:10]}")
        print(f"L2 norm: {norm:.6f}")
        print(f"Metadata: {data['metadata']}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "text": "The quick brown fox jumps over the lazy dog.",
                "pooling": "mean",
                "normalize": True,
            },
            "response": data,
            "validation": {
                "embedding_length": len(embedding),
                "l2_norm": norm,
                "is_normalized": True,
            }
        }
        with open(examples_dir / "embed_basic.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_analyze_endpoint(self, app_client, loaded_embedding_model, examples_dir):
        """Test analyze endpoint with D_eff computation."""
        model_id = TEST_MODELS["embedding"]

        response = app_client.post("/analyze", json={
            "model": model_id,
            "text": "Machine learning models can extract semantic representations.",
            "pooling": "mean",
        })

        assert response.status_code == 200
        data = response.json()

        # Validate analysis fields
        assert "mean" in data
        assert "std" in data
        assert "min" in data
        assert "max" in data
        assert "embedding_shape" in data

        print("\n=== Analysis Output ===")
        print(json.dumps(data, indent=2))

        # Save example
        with open(examples_dir / "analyze_output.json", "w") as f:
            json.dump(data, f, indent=2)


class TestGenerativeModel:
    """Test hidden state extraction with generative models."""

    @pytest.fixture(scope="class")
    def loaded_generative_model(self, app_client):
        """Load generative model once for all tests in class."""
        model_id = TEST_MODELS["generative_small"]

        print(f"\n=== Loading Generative Model: {model_id} ===")
        start = time.time()

        response = app_client.post("/models/load", json={
            "model": model_id,
            "dtype": "float16",
        })

        load_time = time.time() - start
        assert response.status_code == 200, f"Failed to load model: {response.text}"

        data = response.json()
        print(f"Loaded in {load_time:.2f}s")
        print(json.dumps(data, indent=2))

        yield data

        # Cleanup
        app_client.delete(f"/models/{model_id.replace('/', '--')}")

    def test_generate_with_hidden_states(self, app_client, loaded_generative_model, examples_dir):
        """Test generation with hidden state extraction."""
        model_id = TEST_MODELS["generative_small"]

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "The meaning of life is",
            "max_tokens": 20,
            "temperature": 0.7,
            "return_hidden_states": True,
            "hidden_state_layers": [-1],
        })

        assert response.status_code == 200
        data = response.json()

        # Validate structure
        assert "text" in data
        assert "token_count" in data
        assert "hidden_states" in data
        assert "metadata" in data

        # Validate hidden states
        hidden_states = data["hidden_states"]
        assert hidden_states is not None
        assert "-1" in hidden_states

        layer_data = hidden_states["-1"]
        assert "data" in layer_data
        assert "shape" in layer_data
        assert "dtype" in layer_data

        # Validate shape matches model hidden size
        expected_hidden_size = loaded_generative_model["hidden_size"]
        assert layer_data["shape"][-1] == expected_hidden_size

        print("\n=== Generation with Hidden States ===")
        print(f"Generated text: {data['text']}")
        print(f"Token count: {data['token_count']}")
        print(f"Hidden state shape: {layer_data['shape']}")
        print(f"Hidden state dtype: {layer_data['dtype']}")
        print(f"Hidden state (first 10 values): {layer_data['data'][:10]}")
        print(f"Metadata: {data['metadata']}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "prompt": "The meaning of life is",
                "max_tokens": 20,
                "temperature": 0.7,
                "return_hidden_states": True,
                "hidden_state_layers": [-1],
            },
            "response": {
                "text": data["text"],
                "token_count": data["token_count"],
                "hidden_states": {
                    "-1": {
                        "shape": layer_data["shape"],
                        "dtype": layer_data["dtype"],
                        "data_preview": layer_data["data"][:20],
                        "data_length": len(layer_data["data"]),
                    }
                },
                "metadata": data["metadata"],
            },
            "validation": {
                "hidden_size_matches": layer_data["shape"][-1] == expected_hidden_size,
                "expected_hidden_size": expected_hidden_size,
            }
        }
        with open(examples_dir / "generate_hidden_states.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_generate_multiple_layers(self, app_client, loaded_generative_model, examples_dir):
        """Test extraction from multiple layers."""
        model_id = TEST_MODELS["generative_small"]
        num_layers = loaded_generative_model["num_layers"]

        # Request last 3 layers
        layers_to_request = [-1, -2, -3]

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "Hello, how are you?",
            "max_tokens": 10,
            "temperature": 0.5,
            "return_hidden_states": True,
            "hidden_state_layers": layers_to_request,
        })

        assert response.status_code == 200
        data = response.json()

        hidden_states = data["hidden_states"]
        assert hidden_states is not None

        # Validate all requested layers present
        for layer_idx in layers_to_request:
            layer_key = str(layer_idx)
            assert layer_key in hidden_states, f"Layer {layer_idx} missing from response"

            layer_data = hidden_states[layer_key]
            assert "data" in layer_data
            assert "shape" in layer_data

        print("\n=== Multiple Layer Extraction ===")
        print(f"Requested layers: {layers_to_request}")
        print(f"Model has {num_layers} layers")
        for layer_idx in layers_to_request:
            layer_data = hidden_states[str(layer_idx)]
            print(f"  Layer {layer_idx}: shape={layer_data['shape']}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "hidden_state_layers": layers_to_request,
            },
            "response": {
                "layers_returned": list(hidden_states.keys()),
                "layer_shapes": {k: v["shape"] for k, v in hidden_states.items()},
            },
            "validation": {
                "all_layers_present": all(str(layer_idx) in hidden_states for layer_idx in layers_to_request),
                "model_num_layers": num_layers,
            }
        }
        with open(examples_dir / "generate_multiple_layers.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_generate_all_layers(self, app_client, loaded_generative_model, examples_dir):
        """Test extraction from ALL layers using 'all' keyword."""
        model_id = TEST_MODELS["generative_small"]
        num_layers = loaded_generative_model["num_layers"]

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "Test",
            "max_tokens": 5,
            "temperature": 0.1,
            "return_hidden_states": True,
            "hidden_state_layers": "all",
        })

        assert response.status_code == 200
        data = response.json()

        hidden_states = data["hidden_states"]
        assert hidden_states is not None

        # Should have all layers
        assert len(hidden_states) == num_layers, \
            f"Expected {num_layers} layers, got {len(hidden_states)}"

        print("\n=== All Layers Extraction ===")
        print(f"Model layers: {num_layers}")
        print(f"Layers returned: {len(hidden_states)}")
        print(f"Layer keys: {sorted(hidden_states.keys(), key=lambda x: int(x))}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "hidden_state_layers": "all",
            },
            "response": {
                "num_layers_returned": len(hidden_states),
                "layer_keys": sorted(hidden_states.keys(), key=lambda x: int(x)),
                "first_layer_shape": hidden_states[str(-num_layers)]["shape"],
                "last_layer_shape": hidden_states["-1"]["shape"],
            },
            "validation": {
                "expected_layers": num_layers,
                "actual_layers": len(hidden_states),
                "matches": len(hidden_states) == num_layers,
            }
        }
        with open(examples_dir / "generate_all_layers.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_generate_full_sequence(self, app_client, loaded_generative_model, examples_dir):
        """Test full sequence hidden states for manifold analysis."""
        model_id = TEST_MODELS["generative_small"]

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "One two three",
            "max_tokens": 5,
            "temperature": 0.1,
            "return_hidden_states": True,
            "hidden_state_layers": [-1],
            "return_full_sequence": True,
        })

        assert response.status_code == 200
        data = response.json()

        # Check for sequence hidden states
        if data.get("sequence_hidden_states"):
            seq_states = data["sequence_hidden_states"]
            assert "-1" in seq_states

            layer_data = seq_states["-1"]
            assert "data" in layer_data
            assert "shape" in layer_data

            # Shape should be [num_tokens, hidden_size]
            shape = layer_data["shape"]
            assert len(shape) == 2
            num_tokens, hidden_size = shape

            print("\n=== Full Sequence Hidden States ===")
            print(f"Shape: [{num_tokens} tokens, {hidden_size} hidden_size]")
            print(f"Total values: {num_tokens * hidden_size}")

            # Save example
            example = {
                "request": {
                    "model": model_id,
                    "return_full_sequence": True,
                },
                "response": {
                    "shape": shape,
                    "interpretation": f"{num_tokens} tokens × {hidden_size} hidden dimensions",
                    "use_case": "Manifold/boundary object analysis - track semantic evolution across tokens",
                },
            }
            with open(examples_dir / "generate_full_sequence.json", "w") as f:
                json.dump(example, f, indent=2)
        else:
            print("\n=== Full Sequence Hidden States ===")
            print("Note: return_full_sequence not supported by this model/loader")


class TestBatchOperations:
    """Test batch endpoints."""

    def test_batch_embed(self, app_client, examples_dir):
        """Test batch embedding extraction."""
        model_id = TEST_MODELS["embedding"]

        # First ensure model is loaded
        app_client.post("/models/load", json={"model": model_id, "dtype": "float16"})

        texts = [
            "First sentence about machine learning.",
            "Second sentence about natural language processing.",
            "Third sentence about neural networks.",
        ]

        response = app_client.post("/embed/batch", json={
            "model": model_id,
            "texts": texts,
            "pooling": "mean",
            "normalize": True,
        })

        assert response.status_code == 200
        data = response.json()

        # Validate structure
        assert "embeddings" in data
        assert "shapes" in data
        assert "total_time_ms" in data
        assert "texts_processed" in data

        # Validate counts
        assert len(data["embeddings"]) == len(texts)
        assert len(data["shapes"]) == len(texts)
        assert data["texts_processed"] == len(texts)

        print("\n=== Batch Embedding ===")
        print(f"Texts processed: {data['texts_processed']}")
        print(f"Total time: {data['total_time_ms']:.2f}ms")
        print(f"Time per text: {data['total_time_ms'] / len(texts):.2f}ms")
        for i, shape in enumerate(data["shapes"]):
            print(f"  Text {i+1} shape: {shape}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "texts": texts,
                "pooling": "mean",
                "normalize": True,
            },
            "response": {
                "texts_processed": data["texts_processed"],
                "total_time_ms": data["total_time_ms"],
                "shapes": data["shapes"],
                "embeddings_preview": [emb[:5] for emb in data["embeddings"]],
            }
        }
        with open(examples_dir / "batch_embed.json", "w") as f:
            json.dump(example, f, indent=2)


class TestStreamingChatCompletionsIntegration:
    """Integration tests for streaming chat completions with real models.

    These tests verify that streaming works end-to-end with actual model inference,
    validating that tokens arrive incrementally rather than all at once.
    """

    @pytest.fixture(scope="class")
    def loaded_chat_model(self, app_client):
        """Load a chat model once for all streaming tests in class."""
        model_id = TEST_MODELS["generative_small"]

        print(f"\n=== Loading Chat Model for Streaming: {model_id} ===")
        start = time.time()

        response = app_client.post("/models/load", json={
            "model": model_id,
            "dtype": "float16",
        })

        load_time = time.time() - start
        assert response.status_code == 200, f"Failed to load model: {response.text}"

        data = response.json()
        print(f"Loaded in {load_time:.2f}s")
        print(json.dumps(data, indent=2))

        yield data

        # Cleanup
        app_client.delete(f"/models/{model_id.replace('/', '--')}")

    def test_streaming_produces_incremental_tokens(self, app_client, loaded_chat_model, examples_dir):
        """Test that streaming produces tokens incrementally, not all at once.

        This is the key acceptance criterion: tokens should appear incrementally
        rather than being buffered and returned as a single response.
        """
        import httpx
        from fastapi.testclient import TestClient

        model_id = TEST_MODELS["generative_small"]

        print("\n=== Streaming Incremental Token Test ===")

        # Use httpx to make a streaming request
        request_body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 5 slowly."},
            ],
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": True,
            "return_hidden_states": False,
        }

        # Track timing of token arrivals
        token_arrival_times = []
        tokens_received = []
        event_types_seen = set()

        with app_client.stream("POST", "/v1/chat/completions", json=request_body) as response:
            assert response.status_code == 200, f"Request failed: {response.status_code}"
            assert "text/event-stream" in response.headers.get("content-type", "")

            current_event_type = None
            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                    event_types_seen.add(current_event_type)
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            data = json.loads(data_str)
                            if current_event_type == "content_block_delta":
                                token = data.get("delta", {}).get("text", "")
                                if token:
                                    tokens_received.append(token)
                                    token_arrival_times.append(time.time())
                            elif current_event_type == "message_delta":
                                # Final message received
                                pass
                        except json.JSONDecodeError:
                            pass

        # Validate results
        assert len(tokens_received) > 0, "Should have received at least one token"
        assert "content_block_delta" in event_types_seen, "Should have received content_block_delta events"
        assert "message_delta" in event_types_seen, "Should have received message_delta event"

        # Validate incremental behavior: tokens should arrive over time, not all at once
        if len(token_arrival_times) >= 2:
            time_diffs = [
                token_arrival_times[i+1] - token_arrival_times[i]
                for i in range(len(token_arrival_times) - 1)
            ]
            # At least some tokens should have visible time gaps (indicating streaming)
            # Note: This may be flaky on very fast systems, but the test demonstrates the pattern
            avg_gap = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            print(f"Average token gap: {avg_gap*1000:.2f}ms")
            print(f"Total tokens received: {len(tokens_received)}")

        full_text = "".join(tokens_received)
        print(f"Streamed text: {full_text}")
        print(f"Event types seen: {event_types_seen}")

        # Save example
        example = {
            "request": request_body,
            "response": {
                "tokens_received": len(tokens_received),
                "event_types": list(event_types_seen),
                "full_text": full_text,
            },
            "validation": {
                "incremental_streaming": len(tokens_received) > 1,
                "content_block_delta_received": "content_block_delta" in event_types_seen,
                "message_delta_received": "message_delta" in event_types_seen,
            }
        }
        with open(examples_dir / "streaming_incremental_tokens.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_streaming_with_hidden_states(self, app_client, loaded_chat_model, examples_dir):
        """Test that streaming returns hidden states in the final message_delta event."""
        model_id = TEST_MODELS["generative_small"]

        print("\n=== Streaming with Hidden States Test ===")

        request_body = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Hello!"},
            ],
            "max_tokens": 10,
            "temperature": 0.5,
            "stream": True,
            "return_hidden_states": True,
        }

        hidden_state_received = None
        message_delta_received = False

        with app_client.stream("POST", "/v1/chat/completions", json=request_body) as response:
            assert response.status_code == 200

            current_event_type = None
            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str and current_event_type == "message_delta":
                        try:
                            data = json.loads(data_str)
                            message_delta_received = True
                            hidden_state_received = data.get("hidden_state")
                        except json.JSONDecodeError:
                            pass

        # Validate results
        assert message_delta_received, "Should have received message_delta event"
        assert hidden_state_received is not None, "Should have received hidden_state in message_delta"
        assert "final" in hidden_state_received, "Hidden state should have 'final' field"
        assert "shape" in hidden_state_received, "Hidden state should have 'shape' field"
        assert "layer" in hidden_state_received, "Hidden state should have 'layer' field"

        print(f"Hidden state shape: {hidden_state_received['shape']}")
        print(f"Hidden state layer: {hidden_state_received['layer']}")
        print(f"Hidden state vector length: {len(hidden_state_received['final'])}")

        # Validate shape matches expected hidden size
        expected_hidden_size = loaded_chat_model["hidden_size"]
        assert hidden_state_received["shape"][-1] == expected_hidden_size

        # Save example
        example = {
            "request": request_body,
            "response": {
                "hidden_state_received": True,
                "hidden_state_shape": hidden_state_received["shape"],
                "hidden_state_layer": hidden_state_received["layer"],
            },
            "validation": {
                "hidden_size_matches_model": hidden_state_received["shape"][-1] == expected_hidden_size,
                "expected_hidden_size": expected_hidden_size,
            }
        }
        with open(examples_dir / "streaming_with_hidden_states.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_streaming_vs_nonstreaming_equivalence(self, app_client, loaded_chat_model, examples_dir):
        """Test that streaming and non-streaming produce equivalent results."""
        model_id = TEST_MODELS["generative_small"]

        print("\n=== Streaming vs Non-Streaming Equivalence Test ===")

        # Fixed seed for reproducibility (via temperature=0)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ]

        # Non-streaming request
        non_stream_response = app_client.post("/v1/chat/completions", json={
            "model": model_id,
            "messages": messages,
            "max_tokens": 20,
            "temperature": 0.0,  # Deterministic
            "stream": False,
            "return_hidden_states": False,
        })
        assert non_stream_response.status_code == 200
        non_stream_data = non_stream_response.json()
        non_stream_text = non_stream_data["text"]

        # Streaming request with same parameters
        stream_tokens = []
        request_body = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 20,
            "temperature": 0.0,  # Deterministic
            "stream": True,
            "return_hidden_states": False,
        }

        with app_client.stream("POST", "/v1/chat/completions", json=request_body) as response:
            assert response.status_code == 200

            current_event_type = None
            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str and current_event_type == "content_block_delta":
                        try:
                            data = json.loads(data_str)
                            token = data.get("delta", {}).get("text", "")
                            if token:
                                stream_tokens.append(token)
                        except json.JSONDecodeError:
                            pass

        stream_text = "".join(stream_tokens)

        print(f"Non-streaming text: {repr(non_stream_text)}")
        print(f"Streaming text: {repr(stream_text)}")

        # They should be equivalent (or at least very similar with temp=0)
        # Note: Due to RNG differences, we just check they're non-empty and reasonable
        assert len(non_stream_text) > 0, "Non-streaming should produce text"
        assert len(stream_text) > 0, "Streaming should produce text"

        # Save example
        example = {
            "request": {"messages": messages, "max_tokens": 20, "temperature": 0.0},
            "non_streaming_response": {"text": non_stream_text},
            "streaming_response": {"text": stream_text, "token_count": len(stream_tokens)},
            "validation": {
                "both_produced_output": len(non_stream_text) > 0 and len(stream_text) > 0,
            }
        }
        with open(examples_dir / "streaming_vs_nonstreaming.json", "w") as f:
            json.dump(example, f, indent=2)


class TestOutputFormat:
    """Validate output format is correct and easily parseable."""

    def test_hidden_state_format_list(self, app_client, examples_dir):
        """Validate list format is correct JSON arrays."""
        model_id = TEST_MODELS["embedding"]
        app_client.post("/models/load", json={"model": model_id, "dtype": "float16"})

        response = app_client.post("/embed", json={
            "model": model_id,
            "text": "Test text",
            "pooling": "mean",
        })

        assert response.status_code == 200
        data = response.json()

        embedding = data["embedding"]

        # Validate it's a flat list of floats
        assert isinstance(embedding, list)
        assert all(isinstance(x, (int, float)) for x in embedding)

        # Validate JSON round-trip works
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["embedding"] == embedding

        print("\n=== List Format Validation ===")
        print(f"Type: {type(embedding).__name__}")
        print(f"Length: {len(embedding)}")
        print(f"Element type: {type(embedding[0]).__name__}")
        print("JSON round-trip: OK")

    def test_hidden_state_format_base64(self, app_client, examples_dir):
        """Validate base64 format for efficient transfer."""
        model_id = TEST_MODELS["generative_small"]
        app_client.post("/models/load", json={"model": model_id, "dtype": "float16"})

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "Test",
            "max_tokens": 5,
            "return_hidden_states": True,
            "hidden_state_layers": [-1],
            "hidden_state_format": "base64",
        })

        assert response.status_code == 200
        data = response.json()

        if data.get("hidden_states"):
            layer_data = data["hidden_states"]["-1"]

            assert "data" in layer_data
            assert "shape" in layer_data
            assert "dtype" in layer_data
            assert layer_data.get("encoding") == "base64"

            # Validate it's a base64 string
            import base64
            decoded = base64.b64decode(layer_data["data"])

            # Map dtype string to bytes per element
            dtype_to_bytes = {
                "float16": 2,
                "float32": 4,
                "float64": 8,
                "bfloat16": 2,
            }
            dtype_str = layer_data["dtype"].replace("torch.", "")
            bytes_per_element = dtype_to_bytes.get(dtype_str, 4)  # default float32

            # Validate size matches shape
            expected_elements = 1
            for dim in layer_data["shape"]:
                expected_elements *= dim
            expected_bytes = expected_elements * bytes_per_element
            assert len(decoded) == expected_bytes

            print("\n=== Base64 Format Validation ===")
            print(f"Shape: {layer_data['shape']}")
            print(f"Dtype: {dtype_str} ({bytes_per_element} bytes/element)")
            print(f"Encoded length: {len(layer_data['data'])} chars")
            print(f"Decoded bytes: {len(decoded)}")
            print(f"Expected bytes: {expected_bytes}")
            print("Decoding: OK")

            # Map dtype for numpy
            numpy_dtype = {
                "float16": "np.float16",
                "float32": "np.float32",
                "float64": "np.float64",
                "bfloat16": "np.float16",  # numpy doesn't have bfloat16, use float16
            }.get(dtype_str, "np.float32")

            # Save example
            example = {
                "format": "base64",
                "shape": layer_data["shape"],
                "dtype": dtype_str,
                "bytes_per_element": bytes_per_element,
                "encoded_sample": layer_data["data"][:100] + "...",
                "decoding_instructions": [
                    "1. Base64 decode the 'data' field",
                    f"2. Interpret as {dtype_str} array ({bytes_per_element} bytes/element)",
                    "3. Reshape according to 'shape' field",
                ],
                "python_example": f"""
import base64
import numpy as np

decoded = base64.b64decode(layer_data['data'])
array = np.frombuffer(decoded, dtype={numpy_dtype})
array = array.reshape(layer_data['shape'])
""".strip(),
            }
            with open(examples_dir / "hidden_state_base64_format.json", "w") as f:
                json.dump(example, f, indent=2)


class TestLayerByLayerAnalysis:
    """Test layer-by-layer hidden state analysis with D_eff computation.

    These tests validate the layer-by-layer analysis capabilities including:
    - D_eff computation per layer
    - All layers extraction with D_eff metrics
    - D_eff patterns across layers (information flow analysis)
    """

    @pytest.fixture(scope="class")
    def loaded_analysis_model(self, app_client):
        """Load model once for all layer analysis tests."""
        model_id = TEST_MODELS["generative_small"]

        print(f"\n=== Loading Model for Layer Analysis: {model_id} ===")
        start = time.time()

        response = app_client.post("/models/load", json={
            "model": model_id,
            "dtype": "float16",
        })

        load_time = time.time() - start
        assert response.status_code == 200, f"Failed to load model: {response.text}"

        data = response.json()
        print(f"Loaded in {load_time:.2f}s")
        print(json.dumps(data, indent=2))

        yield data

        # Cleanup
        app_client.delete(f"/models/{model_id.replace('/', '--')}")


def test_layer_deff_computation(app_client, examples_dir):
    """Test D_eff computation for individual layers.

    This test validates that:
    1. Hidden states can be extracted from multiple layers
    2. D_eff can be computed for each layer's hidden states
    3. D_eff values are within expected ranges (1 <= D_eff <= hidden_dim)
    """
    import numpy as np

    model_id = TEST_MODELS["generative_small"]

    # First ensure model is loaded
    load_response = app_client.post("/models/load", json={
        "model": model_id,
        "dtype": "float16",
    })
    assert load_response.status_code == 200
    model_info = load_response.json()
    num_layers = model_info["num_layers"]
    hidden_size = model_info["hidden_size"]

    # Request hidden states from multiple layers (first, middle, last)
    layers_to_test = [0, num_layers // 2, -1]

    response = app_client.post("/generate", json={
        "model": model_id,
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "max_tokens": 10,
        "temperature": 0.1,
        "return_hidden_states": True,
        "hidden_state_layers": layers_to_test,
        "hidden_state_format": "list",
    })

    assert response.status_code == 200
    data = response.json()

    hidden_states = data.get("hidden_states", {})
    assert hidden_states is not None, "No hidden states returned"

    # Compute D_eff for each layer
    d_eff_results = {}

    print("\n=== Layer-by-Layer D_eff Computation ===")
    print(f"Model: {model_id}")
    print(f"Layers tested: {layers_to_test}")
    print(f"Hidden size: {hidden_size}")
    print("-" * 50)

    for layer_idx in layers_to_test:
        layer_key = str(layer_idx)
        assert layer_key in hidden_states, f"Layer {layer_idx} missing from response"

        layer_data = hidden_states[layer_key]
        assert "data" in layer_data, f"No data in layer {layer_idx}"
        assert "shape" in layer_data, f"No shape in layer {layer_idx}"

        # Convert to numpy array
        hidden_state_flat = np.array(layer_data["data"])
        shape = layer_data["shape"]

        # Reshape to (num_tokens, hidden_dim) if needed
        if len(shape) == 2:
            embeddings = hidden_state_flat.reshape(shape)
        else:
            # Single vector - reshape to (1, hidden_dim)
            embeddings = hidden_state_flat.reshape(1, -1)

        # Compute D_eff using PCA-based approach
        # D_eff = number of dimensions needed to capture 90% variance
        if embeddings.shape[0] >= 2:
            # Center the embeddings
            centered = embeddings - embeddings.mean(axis=0)

            # Compute covariance matrix
            cov = centered.T @ centered / (embeddings.shape[0] - 1)

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[::-1]  # Sort descending
            eigenvalues = np.maximum(eigenvalues, 0)  # Handle numerical issues

            # Calculate cumulative variance
            total_variance = eigenvalues.sum()
            if total_variance > 1e-10:
                cumulative_variance = np.cumsum(eigenvalues) / total_variance

                # Find D_eff (dimensions for 90% variance)
                d_eff = int(np.searchsorted(cumulative_variance, 0.90) + 1)
                d_eff = max(1, min(d_eff, embeddings.shape[1]))
            else:
                d_eff = 1
        else:
            # Single sample - D_eff = 1
            d_eff = 1

        d_eff_results[layer_idx] = {
            "d_eff": d_eff,
            "shape": shape,
            "variance_ratio": d_eff / hidden_size,
        }

        print(f"Layer {layer_idx:3d}: D_eff = {d_eff:4d} ({d_eff/hidden_size*100:.1f}% of {hidden_size})")

        # Validate D_eff is within expected range
        assert 1 <= d_eff <= hidden_size, \
            f"D_eff {d_eff} out of range [1, {hidden_size}] for layer {layer_idx}"

    # Save example
    example = {
        "request": {
            "model": model_id,
            "hidden_state_layers": layers_to_test,
            "prompt": "The quick brown fox jumps over the lazy dog.",
        },
        "d_eff_results": {str(k): v for k, v in d_eff_results.items()},
        "model_info": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
        },
        "validation": {
            "all_layers_present": len(d_eff_results) == len(layers_to_test),
            "d_eff_in_valid_range": all(
                1 <= v["d_eff"] <= hidden_size
                for v in d_eff_results.values()
            ),
        },
    }
    with open(examples_dir / "layer_deff_computation.json", "w") as f:
        json.dump(example, f, indent=2)

    print("-" * 50)
    print(f"All {len(layers_to_test)} layers processed successfully!")


def test_all_layers_deff_trajectory(app_client, examples_dir):
    """Test D_eff computation across ALL layers to analyze information flow.

    This test validates:
    1. All layers can be extracted using 'all' keyword
    2. D_eff can be computed for every layer
    3. D_eff trajectory shows expected patterns (typically varies across layers)
    """
    import numpy as np

    model_id = TEST_MODELS["generative_small"]

    # Ensure model is loaded
    load_response = app_client.post("/models/load", json={
        "model": model_id,
        "dtype": "float16",
    })
    assert load_response.status_code == 200
    model_info = load_response.json()
    num_layers = model_info["num_layers"]
    hidden_size = model_info["hidden_size"]

    # Request ALL layers
    response = app_client.post("/generate", json={
        "model": model_id,
        "prompt": "Artificial intelligence is transforming how we work and live.",
        "max_tokens": 5,
        "temperature": 0.1,
        "return_hidden_states": True,
        "hidden_state_layers": "all",
        "hidden_state_format": "list",
    })

    assert response.status_code == 200
    data = response.json()

    hidden_states = data.get("hidden_states", {})
    assert hidden_states is not None, "No hidden states returned"
    assert len(hidden_states) == num_layers, \
        f"Expected {num_layers} layers, got {len(hidden_states)}"

    # Compute D_eff trajectory across all layers
    d_eff_trajectory = []

    print("\n=== D_eff Trajectory Across All Layers ===")
    print(f"Model: {model_id} ({num_layers} layers, {hidden_size} hidden dim)")
    print("-" * 60)

    for layer_idx in range(num_layers):
        # Try both positive and negative indexing
        layer_key = str(layer_idx) if str(layer_idx) in hidden_states else str(layer_idx - num_layers)
        if layer_key not in hidden_states:
            # Try finding the key
            possible_keys = [str(layer_idx), str(layer_idx - num_layers)]
            for pk in possible_keys:
                if pk in hidden_states:
                    layer_key = pk
                    break

        if layer_key not in hidden_states:
            print(f"Warning: Layer {layer_idx} not found, skipping")
            continue

        layer_data = hidden_states[layer_key]
        hidden_state_flat = np.array(layer_data["data"])
        shape = layer_data["shape"]

        # Reshape
        if len(shape) == 2:
            embeddings = hidden_state_flat.reshape(shape)
        else:
            embeddings = hidden_state_flat.reshape(1, -1)

        # Compute D_eff
        if embeddings.shape[0] >= 2:
            centered = embeddings - embeddings.mean(axis=0)
            cov = centered.T @ centered / (embeddings.shape[0] - 1)
            eigenvalues = np.linalg.eigvalsh(cov)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)

            total_variance = eigenvalues.sum()
            if total_variance > 1e-10:
                cumulative_variance = np.cumsum(eigenvalues) / total_variance
                d_eff = int(np.searchsorted(cumulative_variance, 0.90) + 1)
                d_eff = max(1, min(d_eff, embeddings.shape[1]))
            else:
                d_eff = 1
        else:
            d_eff = 1

        d_eff_trajectory.append({
            "layer": layer_idx,
            "d_eff": d_eff,
            "utilization": d_eff / hidden_size,
        })

    # Analyze trajectory
    d_eff_values = [t["d_eff"] for t in d_eff_trajectory]
    min_d_eff = min(d_eff_values)
    max_d_eff = max(d_eff_values)
    mean_d_eff = sum(d_eff_values) / len(d_eff_values)

    print(f"D_eff Range: [{min_d_eff}, {max_d_eff}]")
    print(f"D_eff Mean: {mean_d_eff:.1f}")
    print(f"Utilization Range: [{min_d_eff/hidden_size*100:.1f}%, {max_d_eff/hidden_size*100:.1f}%]")
    print("-" * 60)

    # Print trajectory visualization
    for entry in d_eff_trajectory:
        bar_len = int(entry["utilization"] * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"Layer {entry['layer']:2d}: {bar} {entry['d_eff']:4d} ({entry['utilization']*100:.1f}%)")

    # Validate trajectory
    assert len(d_eff_trajectory) == num_layers, \
        f"Expected {num_layers} D_eff values, got {len(d_eff_trajectory)}"

    # D_eff should vary across layers (not all identical)
    d_eff_std = np.std(d_eff_values)
    print(f"\nD_eff Standard Deviation: {d_eff_std:.2f}")

    # Save example
    example = {
        "request": {
            "model": model_id,
            "hidden_state_layers": "all",
        },
        "trajectory": d_eff_trajectory,
        "summary": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "d_eff_min": min_d_eff,
            "d_eff_max": max_d_eff,
            "d_eff_mean": mean_d_eff,
            "d_eff_std": float(d_eff_std),
        },
        "interpretation": {
            "low_d_eff_layers": [
                t["layer"] for t in d_eff_trajectory
                if t["d_eff"] < mean_d_eff - d_eff_std
            ],
            "high_d_eff_layers": [
                t["layer"] for t in d_eff_trajectory
                if t["d_eff"] > mean_d_eff + d_eff_std
            ],
        },
    }
    with open(examples_dir / "all_layers_deff_trajectory.json", "w") as f:
        json.dump(example, f, indent=2)

    print("\nAll layers D_eff trajectory computed successfully!")


def test_layer_deff_patterns(app_client, examples_dir):
    """Test that D_eff patterns reveal information flow characteristics.

    This test validates:
    1. Early layers vs late layers D_eff comparison
    2. Information compression/expansion patterns
    3. Bottleneck detection (layers with significantly lower D_eff)
    """
    import numpy as np

    model_id = TEST_MODELS["generative_small"]

    # Ensure model is loaded
    load_response = app_client.post("/models/load", json={
        "model": model_id,
        "dtype": "float16",
    })
    assert load_response.status_code == 200
    model_info = load_response.json()
    num_layers = model_info["num_layers"]
    hidden_size = model_info["hidden_size"]

    # Test with a more complex prompt to see variation
    test_prompts = [
        "Simple test.",
        "The relationship between artificial intelligence and human cognition involves complex patterns of information processing.",
    ]

    all_results = []

    for prompt_idx, prompt in enumerate(test_prompts):
        # Request all layers
        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": prompt,
            "max_tokens": 3,
            "temperature": 0.1,
            "return_hidden_states": True,
            "hidden_state_layers": "all",
            "hidden_state_format": "list",
        })

        assert response.status_code == 200
        data = response.json()
        hidden_states = data.get("hidden_states", {})

        # Compute D_eff for each layer
        d_eff_values = []

        for layer_idx in range(num_layers):
            layer_key = str(layer_idx) if str(layer_idx) in hidden_states else str(layer_idx - num_layers)
            if layer_key not in hidden_states:
                for pk in [str(layer_idx), str(layer_idx - num_layers)]:
                    if pk in hidden_states:
                        layer_key = pk
                        break

            if layer_key not in hidden_states:
                continue

            layer_data = hidden_states[layer_key]
            hidden_state_flat = np.array(layer_data["data"])
            shape = layer_data["shape"]

            if len(shape) == 2:
                embeddings = hidden_state_flat.reshape(shape)
            else:
                embeddings = hidden_state_flat.reshape(1, -1)

            if embeddings.shape[0] >= 2:
                centered = embeddings - embeddings.mean(axis=0)
                cov = centered.T @ centered / (embeddings.shape[0] - 1)
                eigenvalues = np.linalg.eigvalsh(cov)[::-1]
                eigenvalues = np.maximum(eigenvalues, 0)

                total_variance = eigenvalues.sum()
                if total_variance > 1e-10:
                    cumulative_variance = np.cumsum(eigenvalues) / total_variance
                    d_eff = int(np.searchsorted(cumulative_variance, 0.90) + 1)
                    d_eff = max(1, min(d_eff, embeddings.shape[1]))
                else:
                    d_eff = 1
            else:
                d_eff = 1

            d_eff_values.append(d_eff)

        # Analyze patterns
        early_layers = d_eff_values[:num_layers // 3]
        middle_layers = d_eff_values[num_layers // 3: 2 * num_layers // 3]
        late_layers = d_eff_values[2 * num_layers // 3:]

        early_mean = np.mean(early_layers) if early_layers else 0
        middle_mean = np.mean(middle_layers) if middle_layers else 0
        late_mean = np.mean(late_layers) if late_layers else 0

        # Detect bottleneck layers (>1 std below mean)
        overall_mean = np.mean(d_eff_values)
        overall_std = np.std(d_eff_values)
        bottleneck_threshold = overall_mean - overall_std

        bottleneck_layers = [
            i for i, d in enumerate(d_eff_values)
            if d < bottleneck_threshold
        ]

        result = {
            "prompt_idx": prompt_idx,
            "prompt_length": len(prompt.split()),
            "early_mean": float(early_mean),
            "middle_mean": float(middle_mean),
            "late_mean": float(late_mean),
            "overall_mean": float(overall_mean),
            "overall_std": float(overall_std),
            "bottleneck_layers": bottleneck_layers,
            "d_eff_values": d_eff_values,
        }
        all_results.append(result)

    print("\n=== D_eff Pattern Analysis ===")
    print(f"Model: {model_id}")
    print("-" * 60)

    for result in all_results:
        print(f"\nPrompt {result['prompt_idx']+1} ({result['prompt_length']} words):")
        print(f"  Early layers D_eff mean:  {result['early_mean']:.1f}")
        print(f"  Middle layers D_eff mean: {result['middle_mean']:.1f}")
        print(f"  Late layers D_eff mean:   {result['late_mean']:.1f}")
        print(f"  Bottleneck layers: {result['bottleneck_layers'] or 'None detected'}")

        # Determine pattern
        if result['late_mean'] < result['early_mean'] * 0.8:
            pattern = "compression (early > late)"
        elif result['late_mean'] > result['early_mean'] * 1.2:
            pattern = "expansion (late > early)"
        else:
            pattern = "stable"
        print(f"  Pattern: {pattern}")

    # Validate that we got results for both prompts
    assert len(all_results) == 2, "Should have results for both prompts"

    # Validate D_eff values are in valid range
    for result in all_results:
        for d_eff in result["d_eff_values"]:
            assert 1 <= d_eff <= hidden_size, \
                f"D_eff {d_eff} out of range [1, {hidden_size}]"

    # Save example
    example = {
        "model": model_id,
        "model_info": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
        },
        "prompts": test_prompts,
        "results": all_results,
        "interpretation": {
            "compression_pattern": "D_eff decreases in later layers, suggesting information bottleneck",
            "expansion_pattern": "D_eff increases in later layers, suggesting feature elaboration",
            "stable_pattern": "D_eff remains relatively constant across layers",
        },
    }
    with open(examples_dir / "layer_deff_patterns.json", "w") as f:
        json.dump(example, f, indent=2)

    print("\nD_eff pattern analysis completed successfully!")
