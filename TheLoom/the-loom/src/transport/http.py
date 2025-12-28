"""HTTP transport layer using FastAPI."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import Config, get_config
from ..analysis import (
    BilateralGeometryResult,
    DirectionalCoverageResult,
    GrainAnalysisResult,
    KakeyaGeometryReport,
    WolfAxiomResult,
    analyze_kakeya_geometry,
    compare_bilateral_geometry,
)
from ..extraction.hidden_states import (
    HiddenStateResult,
    analyze_hidden_state,
    extract_hidden_states,
)
from ..patching import (
    ActivationCache,
    CacheManager,
    ExecutionPath,
    ExperimentConfig,
    HookComponent,
    HookPoint,
    PatchingExperiment,
    PathRecorder,
    PathRecording,
    RecordingStore,
    compute_causal_effect,
)
from ..loaders.base import LoadedModel, StreamingOutput, StreamingToken
from ..loaders.registry import LoaderRegistry
from ..utils.gpu import GPUManager
from ..utils.metrics import (
    get_metrics,
    is_metrics_available,
    record_embedding,
    record_generation,
    record_model_load,
    record_request,
    set_models_loaded,
)
from ..utils.serialization import serialize_hidden_states, tensor_to_base64, tensor_to_list

logger = logging.getLogger(__name__)


def _expand_hidden_state_layers(
    layers: list[int] | str,
    num_layers: int,
) -> list[int]:
    """Expand hidden_state_layers to a concrete list of layer indices.

    Args:
        layers: Either a list of layer indices, or "all" for every layer
        num_layers: Total number of layers in the model

    Returns:
        List of layer indices (using negative indexing)
    """
    if isinstance(layers, str):
        if layers.lower() == "all":
            # Return all layers as negative indices: [-num_layers, ..., -1]
            return list(range(-num_layers, 0))
        else:
            raise ValueError(f"Unknown layer specification: {layers}. Use 'all' or a list of indices.")
    return layers


def _apply_chat_template(
    messages: list[dict[str, str]],
    tokenizer: Any,
    model_id: str,
) -> str:
    """Apply chat template to convert messages to a prompt string.

    This is the critical bridge between OpenAI-style chat format and
    raw prompt format. Uses the tokenizer's built-in chat template
    if available, otherwise falls back to a simple concatenation.

    Args:
        messages: List of {"role": str, "content": str} dicts
        tokenizer: HuggingFace tokenizer with optional chat_template
        model_id: Model identifier for logging

    Returns:
        Formatted prompt string ready for generation
    """
    # Check if tokenizer has chat template support
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # Convert Pydantic models to dicts if needed
            msg_dicts = [
                {"role": m["role"], "content": m["content"]}
                if isinstance(m, dict)
                else {"role": m.role, "content": m.content}
                for m in messages
            ]

            prompt: str = tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.debug(f"Applied chat template for {model_id}")
            return prompt
        except Exception as e:
            logger.warning(
                f"Chat template failed for {model_id}: {e}. Falling back to simple format."
            )

    # Fallback: Simple concatenation for models without chat templates
    # This handles base models and older models
    parts = []
    for m in messages:
        role = m["role"] if isinstance(m, dict) else m.role
        content = m["content"] if isinstance(m, dict) else m.content

        if role == "system":
            parts.append(f"System: {content}\n\n")
        elif role == "user":
            parts.append(f"User: {content}\n\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n\n")
        else:
            parts.append(f"{role}: {content}\n\n")

    # Add generation prompt
    parts.append("Assistant:")
    return "".join(parts)


def _serialize_sequence_hidden_states(
    sequence_hidden_states: dict[int, Any],
    format: str = "list",
) -> dict[str, dict[str, Any]]:
    """Serialize full sequence hidden states for manifold analysis.

    Args:
        sequence_hidden_states: dict mapping layer_idx to tensor [num_tokens, hidden_size]
        format: "list" or "base64"

    Returns:
        JSON-serializable dict with sequence data
    """
    import torch

    result: dict[str, dict[str, Any]] = {}

    for layer_idx, tensor in sequence_hidden_states.items():
        layer_key = str(layer_idx)

        # Handle bfloat16 conversion
        if hasattr(tensor, "dtype") and tensor.dtype == torch.bfloat16:
            tensor = tensor.float()

        if hasattr(tensor, "cpu"):
            arr = tensor.cpu().detach().numpy()
        else:
            import numpy as np
            arr = np.asarray(tensor)

        shape = list(arr.shape)
        dtype = "float32"

        if format == "list":
            # Return as nested list [num_tokens][hidden_size]
            result[layer_key] = {
                "data": arr.tolist(),
                "shape": shape,
                "dtype": dtype,
            }
        elif format == "base64":
            result[layer_key] = {
                "data": tensor_to_base64(tensor, "float32"),
                "shape": shape,
                "dtype": dtype,
                "encoding": "base64",
            }
        else:
            raise ValueError(f"Unknown format: {format}")

    return result


# ============================================================================
# Request/Response Models
# ============================================================================


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    model: str = Field(..., description="Model ID (HuggingFace or local path)")
    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(default=256, ge=1, le=8192, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    return_hidden_states: bool = Field(default=True, description="Return hidden states")
    hidden_state_layers: list[int] | str = Field(
        default=[-1],
        description="Which layers to return: list of indices (-1 = last), or 'all' for every layer",
    )
    return_attention: bool = Field(default=False, description="Return attention weights")
    return_full_sequence: bool = Field(
        default=False,
        description="Return hidden states for ALL tokens (manifold/boundary object). "
        "Creates [num_tokens, hidden_size] tensor for geometric analysis.",
    )
    hidden_state_format: str = Field(
        default="list", description="Format for hidden states: list or base64"
    )
    loader: str | None = Field(
        default=None,
        description="Force specific loader (auto-detect if None): transformers, sentence_transformers, custom",
    )


class HiddenStateResponse(BaseModel):
    """Hidden state data in response."""

    data: list[float] | str  # list for 'list' format, str for 'base64'
    shape: list[int]
    dtype: str
    encoding: str | None = None  # 'base64' if base64 encoded


class SequenceHiddenStateResponse(BaseModel):
    """Full sequence hidden states for manifold construction."""

    data: list[list[float]] | str  # [num_tokens, hidden_size] or base64
    shape: list[int]  # [num_tokens, hidden_size]
    dtype: str
    encoding: str | None = None


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    text: str
    token_count: int
    hidden_states: dict[str, HiddenStateResponse] | None = None
    attention_weights: dict[str, Any] | None = None
    # Full sequence hidden states for manifold/boundary object analysis
    sequence_hidden_states: dict[str, SequenceHiddenStateResponse] | None = None
    metadata: dict[str, Any]


class StreamingGenerateRequest(BaseModel):
    """Request model for streaming text generation."""

    model: str = Field(..., description="Model ID (HuggingFace or local path)")
    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(default=256, ge=1, le=8192, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    return_hidden_states: bool = Field(
        default=False, description="Return hidden states in final event"
    )
    hidden_state_layers: list[int] | str = Field(
        default=[-1],
        description="Which layers to return: list of indices (-1 = last), or 'all' for every layer",
    )
    hidden_state_format: str = Field(
        default="list", description="Format for hidden states: list or base64"
    )
    loader: str | None = Field(default=None, description="Force specific loader")


class BatchGenerateRequest(BaseModel):
    """Request model for batch text generation."""

    model: str = Field(..., description="Model ID (HuggingFace or local path)")
    prompts: list[str] = Field(..., description="List of input prompts", min_length=1, max_length=100)
    max_tokens: int = Field(default=256, ge=1, le=8192, description="Max tokens per generation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    return_hidden_states: bool = Field(default=False, description="Return hidden states")
    hidden_state_layers: list[int] | str = Field(
        default=[-1],
        description="Which layers to return: list of indices (-1 = last), or 'all' for every layer",
    )
    hidden_state_format: str = Field(
        default="list", description="Format for hidden states: list or base64"
    )
    loader: str | None = Field(default=None, description="Force specific loader")


class BatchGenerateResponse(BaseModel):
    """Response model for batch text generation."""

    results: list[GenerateResponse]
    total_tokens: int
    total_time_ms: float
    prompts_processed: int


class BatchEmbedRequest(BaseModel):
    """Request model for batch embedding extraction."""

    model: str = Field(..., description="Model ID")
    texts: list[str] = Field(..., description="List of texts to embed", min_length=1, max_length=100)
    pooling: str = Field(default="last_token", description="Pooling: last_token, mean, first_token")
    normalize: bool = Field(default=False, description="L2 normalize the embeddings")


class BatchEmbedResponse(BaseModel):
    """Response model for batch embedding extraction."""

    embeddings: list[list[float]]
    shapes: list[list[int]]
    total_time_ms: float
    texts_processed: int


class EmbedRequest(BaseModel):
    """Request model for embedding extraction."""

    model: str = Field(..., description="Model ID")
    text: str = Field(..., description="Text to embed")
    pooling: str = Field(default="last_token", description="Pooling: last_token, mean, first_token")
    normalize: bool = Field(default=False, description="L2 normalize the embedding")


class EmbedResponse(BaseModel):
    """Response model for embedding extraction."""

    embedding: list[float]
    shape: list[int]
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: str | None
    gpu_info: dict[str, Any]
    config: dict[str, Any]


# ============================================================================
# Kakeya Geometry Analysis Models (for Conveyance Measurement)
# ============================================================================


class BilateralAnalysisRequest(BaseModel):
    """Request model for bilateral geometry analysis.

    Compares geometric properties between sender and receiver hidden states
    to measure potential information transfer compatibility.
    """

    sender_vectors: list[list[float]] = Field(
        ...,
        description="Hidden state vectors from sending agent [n_samples, hidden_dim]",
        min_length=3,
    )
    receiver_vectors: list[list[float]] = Field(
        ...,
        description="Hidden state vectors from receiving agent [n_samples, hidden_dim]",
        min_length=3,
    )


class WolfAxiomResponse(BaseModel):
    """Wolf-inspired density analysis results."""

    max_density_ratio: float = Field(description="Maximum density ratio across regions")
    mean_density_ratio: float = Field(description="Average density ratio")
    uniformity_p_value: float = Field(description="Statistical test p-value")
    violation_count: int = Field(description="Regions exceeding threshold")
    severity: str = Field(description="Violation severity: none, mild, moderate, severe")


class DirectionalCoverageResponse(BaseModel):
    """Directional coverage analysis results."""

    ambient_dim: int = Field(description="Ambient space dimension")
    effective_dim: int = Field(description="Effective dimensionality (95% variance)")
    coverage_ratio: float = Field(description="effective_dim / ambient_dim")
    coverage_quality: str = Field(description="Quality: degenerate, sparse, moderate, full")
    spherical_uniformity: float = Field(description="Uniformity on unit sphere [0,1]")
    isotropy_score: float = Field(description="Geometric mean / max eigenvalue ratio")


class GrainAnalysisResponse(BaseModel):
    """Grain (cluster) detection results."""

    num_grains: int = Field(description="Number of detected clusters")
    grain_coverage: float = Field(description="Fraction of points in clusters")
    mean_grain_size: float = Field(description="Average size of clusters")
    mean_aspect_ratio: float = Field(description="Average cluster elongation (1.0=spherical)")


class GeometryAnalysisRequest(BaseModel):
    """Request model for single-set geometry analysis."""

    vectors: list[list[float]] = Field(
        ...,
        description="Hidden state vectors [n_samples, hidden_dim]",
        min_length=3,
    )


class GeometryAnalysisResponse(BaseModel):
    """Full Kakeya-inspired geometry analysis results."""

    overall_health: str = Field(description="Overall health: healthy, warning:*, unhealthy:*")
    num_vectors: int = Field(description="Number of vectors analyzed")
    ambient_dim: int = Field(description="Ambient space dimension")
    wolf_axiom: WolfAxiomResponse
    directional_coverage: DirectionalCoverageResponse
    grain_analysis: GrainAnalysisResponse
    analysis_time_ms: float = Field(
        description="Time spent on analysis in milliseconds"
    )


class BilateralAnalysisResponse(BaseModel):
    """Response model for bilateral geometry comparison.

    Measures geometric compatibility between sender and receiver hidden states.
    Higher alignment scores may predict better information transfer.
    """

    directional_alignment: float = Field(
        description="Cosine similarity of mean directions [-1,1]"
    )
    subspace_overlap: float = Field(
        description="Principal subspace alignment [0,1]"
    )
    grain_alignment: float = Field(
        description="Cluster structure correspondence [0,1]"
    )
    density_similarity: float = Field(
        description="Wolf axiom profile similarity [0,1]"
    )
    effective_dim_ratio: float = Field(
        description="Ratio of effective dimensions (min/max)"
    )
    overall_alignment: float = Field(
        description="Weighted overall alignment score [0,1]"
    )
    analysis_time_ms: float = Field(
        description="Time spent on analysis in milliseconds"
    )


# ============================================================================
# OpenAI-Compatible Chat Completions (for WeaverCode integration)
# ============================================================================


class ChatMessage(BaseModel):
    """A single chat message with role and content."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with hidden state support.

    This endpoint is designed for WeaverCode integration, providing the
    messages-based API that WeaverCode expects while exposing hidden states
    for conveyance measurement.

    Supports both streaming and non-streaming responses:
    - stream=false (default): Returns ChatCompletionResponse
    - stream=true: Returns SSE stream with content_block_delta and message_delta events
    """

    model: str = Field(..., description="Model ID (HuggingFace or local path)")
    messages: list[ChatMessage] = Field(
        ..., description="List of chat messages", min_length=1
    )
    max_tokens: int = Field(
        default=256, ge=1, le=8192, description="Max tokens to generate"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability"
    )
    return_hidden_states: bool = Field(
        default=True, description="Return hidden states for conveyance measurement"
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming responses via Server-Sent Events (SSE). "
        "When true, returns content_block_delta events for each token and "
        "message_delta event at completion.",
    )
    loader: str | None = Field(default=None, description="Force specific loader")
    device: str | None = Field(
        default=None,
        description="GPU device to use (e.g., 'cuda:0', 'cuda:1'). None = auto-select.",
    )


class StreamingChatCompletionRequest(BaseModel):
    """Request model for streaming chat completions via SSE.

    This is a convenience model that explicitly requires streaming.
    It contains the same fields as ChatCompletionRequest but with
    stream always set to True.

    Use this model when you want to explicitly type a streaming request,
    or use ChatCompletionRequest with stream=true for flexibility.

    SSE Event Types:
    - content_block_delta: Emitted for each generated token
      {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
    - message_delta: Emitted at completion with usage stats
      {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {...}}
    - error: Emitted if an error occurs during streaming
      {"type": "error", "error": {"message": "..."}}
    """

    model: str = Field(..., description="Model ID (HuggingFace or local path)")
    messages: list[ChatMessage] = Field(
        ..., description="List of chat messages", min_length=1
    )
    max_tokens: int = Field(
        default=256, ge=1, le=8192, description="Max tokens to generate"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability"
    )
    return_hidden_states: bool = Field(
        default=True,
        description="Return hidden states in final message_delta event",
    )
    stream: bool = Field(
        default=True,
        description="Always True for streaming requests",
    )
    loader: str | None = Field(default=None, description="Force specific loader")
    device: str | None = Field(
        default=None,
        description="GPU device to use (e.g., 'cuda:0', 'cuda:1'). None = auto-select.",
    )

    @classmethod
    def from_chat_request(cls, request: ChatCompletionRequest) -> "StreamingChatCompletionRequest":
        """Convert a ChatCompletionRequest to StreamingChatCompletionRequest.

        Useful when you need to ensure streaming is enabled.
        """
        return cls(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            return_hidden_states=request.return_hidden_states,
            stream=True,
            loader=request.loader,
            device=request.device,
        )


class ChatCompletionUsage(BaseModel):
    """Token usage statistics matching OpenAI format."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionHiddenState(BaseModel):
    """Hidden state in WeaverCode-expected format.

    This is the geometric boundary object - the final hidden state
    before lm_head projection, representing meaning as geometry.
    """

    final: list[float] = Field(description="Final layer hidden state vector")
    shape: list[int] = Field(description="Tensor shape [batch, hidden_dim]")
    layer: int = Field(default=-1, description="Layer index (-1 = last)")
    dtype: str = Field(default="float32", description="Data type")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response with hidden state extraction.

    Extends standard chat completion response with hidden_state field
    for conveyance measurement in WeaverCode.
    """

    text: str = Field(description="Generated text")
    usage: ChatCompletionUsage = Field(description="Token usage statistics")
    hidden_state: ChatCompletionHiddenState | None = Field(
        default=None, description="Hidden state for conveyance measurement"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Generation metadata"
    )


class ModelLoadRequest(BaseModel):
    """Request model for loading a model."""

    model: str = Field(..., description="Model ID to load")
    device: str | None = Field(default=None, description="Device to load on (e.g., cuda:0)")
    dtype: str = Field(default="auto", description="Data type: auto, float16, bfloat16, float32")
    loader: str | None = Field(
        default=None,
        description="Force specific loader (auto-detect if None): transformers, sentence_transformers, custom",
    )
    quantization: str | None = Field(
        default=None,
        description="Quantization mode: 4bit, 8bit, gptq, awq (requires appropriate packages)",
    )


# ============================================================================
# GPU Configuration Models (for Dynamic GPU Pool Management)
# ============================================================================


class GPUConfigureRequest(BaseModel):
    """Request model for dynamic GPU configuration.

    Allows runtime updates to the GPU device pool without server restart.
    At least one of allowed_devices or default_device must be provided.
    """

    allowed_devices: list[int] | None = Field(
        default=None,
        description="List of CUDA device indices to allow (e.g., [0, 1]). "
        "Must be valid device indices from available hardware.",
    )
    default_device: int | None = Field(
        default=None,
        description="Default CUDA device index for model loading. "
        "Must be present in allowed_devices.",
    )


class GPUMemoryInfo(BaseModel):
    """Memory information for a single GPU."""

    device_index: int = Field(description="CUDA device index")
    device_name: str = Field(description="GPU device name")
    total_memory_mb: float = Field(description="Total GPU memory in MB")
    used_memory_mb: float = Field(description="Currently used GPU memory in MB")
    free_memory_mb: float = Field(description="Available GPU memory in MB")
    memory_utilization: float = Field(
        description="Memory utilization as a fraction (0.0 to 1.0)"
    )


class LoadedModelInfo(BaseModel):
    """Information about a loaded model and its device assignment."""

    model_id: str = Field(description="Model identifier")
    device: str = Field(description="Device the model is loaded on (e.g., 'cuda:0')")
    dtype: str = Field(description="Data type of the model")
    idle_seconds: float = Field(description="Seconds since last access")


class GPUConfigureResponse(BaseModel):
    """Response model for GPU configuration updates.

    Returns the updated GPU configuration after applying changes.
    """

    allowed_devices: list[int] = Field(
        description="Currently allowed CUDA device indices"
    )
    default_device: str = Field(
        description="Default device for model loading (e.g., 'cuda:0' or 'cpu')"
    )
    available_devices: list[int] = Field(
        description="All available CUDA devices on the system"
    )
    message: str = Field(description="Status message describing the changes applied")


class GPUStatusResponse(BaseModel):
    """Response model for GPU status information.

    Provides comprehensive GPU configuration and runtime status including
    memory usage, loaded models, and device availability.
    """

    has_gpu: bool = Field(description="Whether CUDA GPUs are available")
    available_devices: list[int] = Field(
        description="All CUDA device indices available on the system"
    )
    allowed_devices: list[int] = Field(
        description="Currently configured allowed device indices"
    )
    default_device: str = Field(
        description="Default device for model loading (e.g., 'cuda:0' or 'cpu')"
    )
    memory_fraction: float = Field(
        description="Configured maximum GPU memory fraction (0.0 to 1.0)"
    )
    gpu_memory: list[GPUMemoryInfo] = Field(
        description="Memory information for each allowed GPU"
    )
    loaded_models: list[LoadedModelInfo] = Field(
        description="Currently loaded models and their device assignments"
    )


class ModelLoadResponse(BaseModel):
    """Response model for model loading."""

    model_id: str
    device: str
    dtype: str
    hidden_size: int
    num_layers: int
    load_time_seconds: float
    loader_type: str = Field(description="Which loader was used")
    quantization: str = Field(default="none", description="Quantization mode used")


# ============================================================================
# Activation Patching Models (for Causal Intervention Analysis)
# ============================================================================


class PatchingConfigureRequest(BaseModel):
    """Request model for configuring patching experiments.

    Configures parameters for activation patching, including which layers
    and components to target for causal intervention analysis.
    """

    experiment_name: str = Field(
        ...,
        description="Name for this patching experiment",
    )
    layers: list[int] = Field(
        default_factory=list,
        description="Layer indices to patch (empty = all layers). Supports negative indexing.",
    )
    components: list[str] = Field(
        default=["resid_pre"],
        description="Components to patch: resid_pre, resid_post, attn, mlp_pre, mlp_post",
    )
    validate_shapes: bool = Field(
        default=True,
        description="Validate that hook output shapes match input shapes",
    )
    cleanup_on_completion: bool = Field(
        default=True,
        description="Clean up activation caches after experiment completes",
    )
    cache_device: str | None = Field(
        default=None,
        description="Device to store cached activations (e.g., 'cpu', 'cuda:0'). None = same as model.",
    )


class PatchingConfigureResponse(BaseModel):
    """Response model for patching configuration."""

    experiment_id: str = Field(description="Unique identifier for this experiment")
    experiment_name: str = Field(description="Name of the experiment")
    layers: list[int] = Field(description="Configured layers to patch")
    components: list[str] = Field(description="Configured components to patch")
    status: str = Field(description="Experiment status")
    message: str = Field(description="Status message")


class PatchingRunRequest(BaseModel):
    """Request model for running a patching experiment.

    Executes the three-path patching workflow:
    1. Clean path: Run with original input, cache activations
    2. Corrupted path: Run with corrupted input, cache activations
    3. Patched path: Run corrupted input with clean activations patched in
    """

    model: str = Field(
        ...,
        description="Model ID (HuggingFace or local path)",
    )
    experiment_id: str | None = Field(
        default=None,
        description="Existing experiment ID to use. Creates new if None.",
    )
    clean_input: str = Field(
        ...,
        description="Clean (baseline) input text for generating source activations",
    )
    corrupted_input: str = Field(
        ...,
        description="Corrupted input text for intervention analysis",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=8192,
        description="Max tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    layers: list[int] | None = Field(
        default=None,
        description="Override layers to patch (None = use experiment config)",
    )
    components: list[str] | None = Field(
        default=None,
        description="Override components to patch (None = use experiment config)",
    )
    device: str | None = Field(
        default=None,
        description="GPU device to use (e.g., 'cuda:0')",
    )


class PatchingPathResult(BaseModel):
    """Result from a single execution path (clean/corrupted/patched)."""

    path_type: str = Field(description="Type of path: clean, corrupted, or patched")
    output_text: str | None = Field(description="Generated text output")
    generation_time_ms: float = Field(description="Time taken for generation")
    num_cached_activations: int = Field(
        default=0,
        description="Number of activation entries cached for this path",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional path metadata",
    )


class PatchingLayerResult(BaseModel):
    """Result from patching a specific layer/component combination."""

    layer: int = Field(description="Layer index that was patched")
    component: str = Field(description="Component that was patched")
    hook_name: str = Field(description="TransformerLens-compatible hook name")
    patched_output: str | None = Field(description="Output text after patching")
    execution_time_ms: float = Field(description="Time for patched run")
    shape_matched: bool = Field(
        default=True,
        description="Whether activation shapes matched correctly",
    )


class CausalEffectResult(BaseModel):
    """Causal effect metrics from patching."""

    layer: int = Field(description="Layer index")
    component: str = Field(description="Component patched")
    causal_effect: float = Field(
        description="Difference between patched and corrupted metric"
    )
    recovery_rate: float = Field(
        description="Fraction of corruption effect recovered by patching"
    )
    clean_baseline: float = Field(description="Clean path metric value")
    corrupted_metric: float = Field(description="Corrupted path metric value")
    patched_metric: float = Field(description="Patched path metric value")


class PatchingRunResponse(BaseModel):
    """Response model for patching experiment execution."""

    experiment_id: str = Field(description="Experiment identifier")
    experiment_name: str = Field(description="Experiment name")
    status: str = Field(description="Experiment status: completed, failed")
    clean_path: PatchingPathResult | None = Field(
        default=None,
        description="Clean (baseline) path result",
    )
    corrupted_path: PatchingPathResult | None = Field(
        default=None,
        description="Corrupted path result",
    )
    patched_results: list[PatchingLayerResult] = Field(
        default_factory=list,
        description="Results for each layer/component patching",
    )
    duration_ms: float = Field(description="Total experiment duration in milliseconds")
    layers_patched: list[int] = Field(
        default_factory=list,
        description="List of layer indices that were patched",
    )
    components_patched: list[str] = Field(
        default_factory=list,
        description="List of components that were patched",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional experiment metadata",
    )


class PatchingResultsRequest(BaseModel):
    """Request model for retrieving patching results."""

    experiment_id: str = Field(
        ...,
        description="Experiment ID to retrieve results for",
    )
    include_causal_effects: bool = Field(
        default=True,
        description="Compute and include causal effect metrics",
    )
    metric_type: str = Field(
        default="token_diff",
        description="Metric type for causal effect calculation: token_diff, logit_diff, custom",
    )


class PatchingResultsResponse(BaseModel):
    """Response model for patching experiment results."""

    experiment_id: str = Field(description="Experiment identifier")
    experiment_name: str = Field(description="Experiment name")
    status: str = Field(description="Experiment status")
    has_all_paths: bool = Field(
        description="Whether all three path types were recorded"
    )
    clean_output: str | None = Field(
        default=None,
        description="Clean path output text",
    )
    corrupted_output: str | None = Field(
        default=None,
        description="Corrupted path output text",
    )
    num_patched_paths: int = Field(
        default=0,
        description="Number of patched path variations",
    )
    causal_effects: list[CausalEffectResult] = Field(
        default_factory=list,
        description="Causal effect metrics for each layer/component",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata",
    )


class PatchingCacheStatusResponse(BaseModel):
    """Response model for patching cache status."""

    num_caches: int = Field(description="Number of active activation caches")
    total_size_mb: float = Field(description="Total cache size in megabytes")
    max_size_mb: float = Field(description="Maximum cache size allowed")
    utilization: float = Field(
        description="Cache utilization as a fraction (0.0 to 1.0)"
    )
    cache_ids: list[str] = Field(
        default_factory=list,
        description="List of active cache identifiers",
    )
    cache_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-cache size and entry information",
    )


# ============================================================================
# Experiment Persistence Query Models
# ============================================================================


class ExperimentResponse(BaseModel):
    """Response model for experiment metadata."""

    id: str = Field(description="Unique experiment identifier")
    created_at: datetime = Field(description="Experiment creation timestamp")
    model: str = Field(description="Model identifier used in the experiment")
    config: dict[str, Any] = Field(description="Experiment configuration")
    status: str = Field(description="Experiment status (running, completed, failed)")
    notes: str | None = Field(default=None, description="Optional notes or description")


class ExperimentSummaryResponse(BaseModel):
    """Response model for experiment summary statistics."""

    experiment_id: str = Field(description="Unique experiment identifier")
    model: str = Field(description="Model identifier")
    created_at: datetime = Field(description="Experiment creation timestamp")
    status: str = Field(description="Experiment status")
    conversation_count: int = Field(description="Number of conversation messages")
    metric_count: int = Field(description="Number of metrics recorded")
    hidden_state_count: int = Field(description="Number of hidden state snapshots")


class ExperimentListResponse(BaseModel):
    """Response model for listing experiments."""

    experiments: list[ExperimentResponse] = Field(description="List of experiments")
    total: int = Field(description="Total count of matching experiments")
    limit: int = Field(description="Limit used in the query")
    offset: int = Field(description="Offset used in the query")


class ConversationResponse(BaseModel):
    """Response model for a conversation message."""

    id: int | None = Field(default=None, description="Conversation record ID")
    experiment_id: str = Field(description="Parent experiment identifier")
    sequence_num: int = Field(description="Message sequence number")
    timestamp: datetime = Field(description="Message timestamp")
    role: str = Field(description="Message role (user, assistant, system)")
    content: str = Field(description="Message content")


class ConversationsListResponse(BaseModel):
    """Response model for listing conversations."""

    experiment_id: str = Field(description="Parent experiment identifier")
    conversations: list[ConversationResponse] = Field(description="List of conversation messages")
    total: int = Field(description="Total count of messages")


class HiddenStateRecordResponse(BaseModel):
    """Response model for hidden state record from persistence."""

    id: int | None = Field(default=None, description="Hidden state record ID")
    experiment_id: str = Field(description="Parent experiment identifier")
    layer: int = Field(description="Layer index")
    file_path: str = Field(description="Path to HDF5 file")
    shape: list[int] = Field(description="Tensor shape")
    dtype: str = Field(description="Data type")
    timestamp: datetime = Field(description="Record timestamp")


class HiddenStatesListResponse(BaseModel):
    """Response model for listing hidden states."""

    experiment_id: str = Field(description="Parent experiment identifier")
    hidden_states: list[HiddenStateRecordResponse] = Field(description="List of hidden state records")
    layers: list[int] = Field(description="Available layer indices")
    total: int = Field(description="Total count of hidden state records")


class MetricResponse(BaseModel):
    """Response model for a metric record."""

    id: int | None = Field(default=None, description="Metric record ID")
    experiment_id: str = Field(description="Parent experiment identifier")
    name: str = Field(description="Metric name")
    value: float = Field(description="Metric value")
    unit: str | None = Field(default=None, description="Unit of measurement")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(description="Record timestamp")


class ExperimentDetailResponse(BaseModel):
    """Response model for detailed experiment data."""

    experiment: ExperimentResponse = Field(description="Experiment metadata")
    conversations: list[ConversationResponse] = Field(description="Conversation messages")
    hidden_states: list[HiddenStateRecordResponse] = Field(description="Hidden state records")
    metrics: list[MetricResponse] = Field(description="Metric records")
    summary: ExperimentSummaryResponse = Field(description="Summary statistics")


# ============================================================================
# Experiment Export Models
# ============================================================================


class ExperimentExportRequest(BaseModel):
    """Request model for exporting experiment data."""

    experiment_id: str = Field(description="The experiment identifier to export")
    format: str = Field(
        default="json",
        description="Export format: 'json', 'csv', or 'parquet'",
    )
    output_path: str | None = Field(
        default=None,
        description="Optional custom output path. If not provided, uses configured export directory.",
    )
    include_hidden_states: bool = Field(
        default=False,
        description="Include actual hidden state data (JSON only). Warning: can produce large files.",
    )


class ExperimentExportResponse(BaseModel):
    """Response model for experiment export operation."""

    experiment_id: str = Field(description="The exported experiment identifier")
    format: str = Field(description="Export format used")
    output_files: dict[str, str] = Field(
        description="Mapping of data type to file path for the exported files"
    )
    total_files: int = Field(description="Total number of files created")


class ExperimentBatchExportRequest(BaseModel):
    """Request model for exporting multiple experiments."""

    experiment_ids: list[str] = Field(
        description="List of experiment identifiers to export",
        min_length=1,
    )
    format: str = Field(
        default="json",
        description="Export format: 'json', 'csv', or 'parquet'",
    )
    output_dir: str | None = Field(
        default=None,
        description="Optional output directory. If not provided, uses configured export directory.",
    )


class ExperimentBatchExportResponse(BaseModel):
    """Response model for batch experiment export."""

    exported: dict[str, dict[str, str]] = Field(
        description="Mapping of experiment_id to their exported file paths"
    )
    total_experiments: int = Field(description="Number of experiments exported")
    format: str = Field(description="Export format used")


class ExperimentSummaryExportRequest(BaseModel):
    """Request model for exporting experiment summary."""

    format: str = Field(
        default="csv",
        description="Export format: 'json', 'csv', or 'parquet'",
    )
    output_path: str | None = Field(
        default=None,
        description="Optional custom output path.",
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of experiments to include in summary",
    )


class ExperimentSummaryExportResponse(BaseModel):
    """Response model for experiment summary export."""

    output_path: str = Field(description="Path to the created summary file")
    total_experiments: int = Field(description="Number of experiments in summary")
    format: str = Field(description="Export format used")


# ============================================================================
# Model Manager
# ============================================================================


class ModelManager:
    """Manages loaded models with LRU eviction, auto-unload, and multi-loader support."""

    def __init__(
        self,
        gpu_manager: GPUManager,
        registry: LoaderRegistry,
        max_models: int = 3,
        auto_unload_minutes: int = 20,
    ):
        """
        Create a ModelManager that loads and manages models with an LRU eviction policy.

        Parameters:
            gpu_manager: Manager responsible for GPU device allocation and cache management.
            registry: Loader registry used to resolve and load models.
            max_models (int): Maximum number of models to keep loaded simultaneously; older models are evicted when capacity is exceeded.
            auto_unload_minutes (int): Auto-unload models after this many minutes of inactivity (0 = disabled).
        """
        self.gpu_manager = gpu_manager
        self.registry = registry
        self.max_models = max_models
        self.auto_unload_minutes = auto_unload_minutes
        self.loaded_models: dict[str, LoadedModel] = {}
        self.access_order: list[str] = []  # For LRU eviction
        self.last_accessed: dict[str, float] = {}  # Track last access time per model
        self._loading_locks: dict[str, threading.Lock] = {}  # Per-model locks to prevent concurrent loading
        self._locks_lock = threading.Lock()  # Lock for accessing _loading_locks dict

    def _get_model_lock(self, model_id: str) -> threading.Lock:
        """Get or create a lock for the given model_id."""
        with self._locks_lock:
            if model_id not in self._loading_locks:
                self._loading_locks[model_id] = threading.Lock()
            return self._loading_locks[model_id]

    def get_or_load(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str = "auto",
        loader_name: str | None = None,
        quantization: str | None = None,
    ) -> LoadedModel:
        """
        Retrieve a LoadedModel by model_id, loading and caching it if not already present.

        Uses per-model locking to prevent duplicate concurrent loads of the same model.

        Parameters:
            model_id (str): Identifier of the model to retrieve.
            device (str | None): Target device for loading; when None the GPU manager's default device is used.
            dtype (str): Desired numeric dtype for the model (e.g., "float16", "float32", or "auto").
            loader_name (str | None): Specific loader to use; when None the loader will be auto-detected.
            quantization (str | None): Quantization mode to apply (e.g., "4bit", "8bit", "gptq", "awq"), if any.

        Returns:
            LoadedModel: The loaded and cached model instance for the requested model_id.
        """
        # Get per-model lock to prevent concurrent loading of the same model
        model_lock = self._get_model_lock(model_id)

        with model_lock:
            # Check if already loaded (double-check after acquiring lock)
            if model_id in self.loaded_models:
                # Update access order for LRU
                if model_id in self.access_order:
                    self.access_order.remove(model_id)
                self.access_order.append(model_id)
                self.last_accessed[model_id] = time.time()
                return self.loaded_models[model_id]

            # Evict if at capacity
            while len(self.loaded_models) >= self.max_models:
                self._evict_oldest()

            # Resolve device
            if device is None:
                device = self.gpu_manager.default_device

            # Load the model using registry (auto-detects loader if not specified)
            logger.info(f"Loading model: {model_id}")
            loaded = self.registry.load(
                model_id,
                device=device,
                dtype=dtype,
                loader_name=loader_name,
                quantization=quantization,
            )

            self.loaded_models[model_id] = loaded
            self.access_order.append(model_id)
            self.last_accessed[model_id] = time.time()

            return loaded

    def _evict_oldest(self) -> None:
        """
        Remove the least-recently-used model from the manager and clear the GPU cache.

        If no models are loaded, this is a no-op.
        """
        if not self.access_order:
            return

        oldest = self.access_order.pop(0)
        if oldest in self.loaded_models:
            logger.info(f"Evicting model: {oldest}")
            del self.loaded_models[oldest]
            self.last_accessed.pop(oldest, None)
            self.gpu_manager.clear_cache()

    def unload(self, model_id: str) -> bool:
        """
        Unload the specified loaded model from memory and clear related GPU cache.

        Clears the GPU cache and logs the unload when a model is removed.

        Parameters:
            model_id (str): Identifier of the model to unload.

        Returns:
            bool: True if the model was loaded and successfully unloaded, False if the model was not found.
        """
        if model_id not in self.loaded_models:
            return False

        del self.loaded_models[model_id]
        if model_id in self.access_order:
            self.access_order.remove(model_id)
        self.last_accessed.pop(model_id, None)
        self.gpu_manager.clear_cache()
        logger.info(f"Unloaded model: {model_id}")
        return True

    def check_idle_models(self) -> list[str]:
        """
        Check for and unload models that have been idle longer than auto_unload_minutes.

        Returns:
            list[str]: List of model IDs that were unloaded due to inactivity.
        """
        if self.auto_unload_minutes <= 0:
            return []

        unloaded = []
        current_time = time.time()
        timeout_seconds = self.auto_unload_minutes * 60

        # Create a copy of keys to avoid modification during iteration
        for model_id in list(self.loaded_models.keys()):
            last_access = self.last_accessed.get(model_id, current_time)
            idle_seconds = current_time - last_access

            if idle_seconds >= timeout_seconds:
                logger.info(
                    f"Auto-unloading idle model: {model_id} "
                    f"(idle for {idle_seconds / 60:.1f} minutes)"
                )
                self.unload(model_id)
                unloaded.append(model_id)

        return unloaded

    def get_idle_time(self, model_id: str) -> float | None:
        """
        Get how long a model has been idle in seconds.

        Parameters:
            model_id (str): Model identifier.

        Returns:
            float | None: Seconds since last access, or None if model not loaded.
        """
        if model_id not in self.last_accessed:
            return None
        return time.time() - self.last_accessed[model_id]

    def list_loaded(self) -> list[str]:
        """
        Return the identifiers of models currently loaded in the manager.
        
        Returns:
            loaded_models (list[str]): List of model IDs for all models currently loaded.
        """
        return list(self.loaded_models.keys())

    def get_loaded_info(self) -> list[dict[str, Any]]:
        """
        Return a list of dictionaries describing each currently loaded model.

        Each dictionary contains:
        - `model_id`: model identifier string
        - `device`: device identifier (e.g., "cuda:0" or "cpu")
        - `dtype`: data type name (e.g., "float16", "float32")
        - `hidden_size`: hidden dimension size
        - `num_layers`: number of model layers
        - `loader_type`: name of the loader used to load the model
        - `quantization`: quantization metadata value or "none" if not present
        - `idle_seconds`: seconds since last access
        - `auto_unload_at`: remaining seconds until auto-unload (None if disabled)

        Returns:
            list[dict[str, Any]]: List of per-model information dictionaries.
        """
        result = []
        current_time = time.time()
        timeout_seconds = self.auto_unload_minutes * 60 if self.auto_unload_minutes > 0 else None

        for model in self.loaded_models.values():
            idle_seconds = current_time - self.last_accessed.get(model.model_id, current_time)
            auto_unload_in = None
            if timeout_seconds is not None:
                auto_unload_in = max(0, timeout_seconds - idle_seconds)

            result.append({
                "model_id": model.model_id,
                "device": str(model.device),
                "dtype": str(model.dtype).replace("torch.", ""),
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "loader_type": model.loader_type,
                "quantization": model.metadata.get("quantization", "none"),
                "idle_seconds": round(idle_seconds, 1),
                "auto_unload_in": round(auto_unload_in, 1) if auto_unload_in is not None else None,
            })

        return result


# ============================================================================
# Metrics Middleware
# ============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """
        Track request latency and record request metrics.
        
        Skips the /metrics endpoint to avoid recursive metrics collection. Measures the request latency, records the endpoint, HTTP method, status code, and latency via `record_request`, and returns the downstream response.
        
        Returns:
            Response: The HTTP response produced by the next handler.
        """
        # Skip metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            response: Response = await call_next(request)
            return response

        start_time = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start_time

        # Record metrics
        record_request(
            endpoint=request.url.path,
            method=request.method,
            status=response.status_code,
            latency=latency,
        )

        return response


# ============================================================================
# FastAPI Application Factory
# ============================================================================


def create_http_app(config: Config | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application for the hidden-state extraction server.
    
    Initializes CORS and optional metrics middleware, GPU manager, loader registry, and model manager, registers all HTTP endpoints (health, metrics, model management, generation, streaming generation, embedding, analysis, and batch operations), and stores core components on app.state.
    
    Parameters:
        config (Config | None): Optional server configuration; when None the global configuration is used.
    
    Returns:
        FastAPI: A configured FastAPI application ready to serve the transport HTTP API.
    """
    if config is None:
        config = get_config()

    # Initialize components first (needed by lifespan)
    gpu_manager = GPUManager(
        allowed_devices=config.gpu.devices,
        memory_fraction=config.gpu.memory_fraction,
    )

    # Create loader registry with model overrides from config
    registry = LoaderRegistry(loader_configs=config.model_overrides)

    model_manager = ModelManager(
        gpu_manager=gpu_manager,
        registry=registry,
        max_models=config.models.max_loaded,
        auto_unload_minutes=config.models.auto_unload_minutes,
    )

    # Background task state
    auto_unload_task: asyncio.Task[None] | None = None

    async def auto_unload_checker(manager: ModelManager, check_interval: int = 60) -> None:
        """Background task to check and unload idle models."""
        while True:
            try:
                await asyncio.sleep(check_interval)
                unloaded = manager.check_idle_models()
                if unloaded:
                    set_models_loaded(len(manager.list_loaded()))
            except asyncio.CancelledError:
                logger.info("Auto-unload checker stopped")
                break
            except Exception as e:
                logger.error(f"Error in auto-unload checker: {e}")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage app lifespan - start/stop background tasks."""
        nonlocal auto_unload_task
        # Startup: start the background task if auto-unload is enabled
        if config.models.auto_unload_minutes > 0:
            logger.info(
                f"Starting auto-unload checker (timeout: {config.models.auto_unload_minutes} min)"
            )
            auto_unload_task = asyncio.create_task(
                auto_unload_checker(model_manager, check_interval=60)
            )
        yield
        # Shutdown: cancel the background task
        if auto_unload_task is not None:
            auto_unload_task.cancel()
            try:
                await auto_unload_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="The Loom",
        description="Hidden state extraction server for AI research - part of the Weaver ecosystem",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS middleware for browser-based clients
    # Note: allow_credentials=True with allow_origins=["*"] is insecure
    # Configure via config.server.cors_origins and cors_allow_credentials
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=config.server.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics middleware (only if prometheus_client is available)
    if is_metrics_available():
        app.add_middleware(MetricsMiddleware)

    # Store in app state
    app.state.config = config
    app.state.gpu_manager = gpu_manager
    app.state.model_manager = model_manager
    app.state.registry = registry

    # ========================================================================
    # Streaming Chat Completions Helper
    # ========================================================================

    async def _stream_chat_completions(
        request: ChatCompletionRequest,
        manager: ModelManager,
        reg: LoaderRegistry,
    ) -> StreamingResponse:
        """Stream chat completions as Server-Sent Events.

        Follows the pattern established by /generate/stream but uses event
        names matching Claude/Anthropic API conventions:
        - content_block_delta: Contains text delta for each token
        - message_delta: Final event with usage stats and completion info
        - error: If an error occurs during streaming
        """

        async def event_generator() -> AsyncIterator[str]:
            """Generate SSE events for streaming chat completion."""
            start_time = time.perf_counter()
            completion_tokens = 0
            full_text = ""

            try:
                # Get or load model (with optional device override)
                loaded = manager.get_or_load(
                    request.model,
                    device=request.device,
                    loader_name=request.loader,
                )

                # Apply chat template to convert messages to prompt
                prompt = _apply_chat_template(
                    request.messages,  # type: ignore[arg-type]
                    loaded.tokenizer,
                    loaded.model_id,
                )

                # Stream tokens using registry
                for item in reg.generate_stream(
                    loaded_model=loaded,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    return_hidden_states=request.return_hidden_states,
                    hidden_state_layers=[-1],  # WeaverCode only needs final layer
                ):
                    if isinstance(item, StreamingToken):
                        # Emit content_block_delta event for each token
                        completion_tokens += 1
                        full_text += item.token
                        data = {
                            "type": "content_block_delta",
                            "delta": {
                                "type": "text_delta",
                                "text": item.token,
                            },
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"

                    elif isinstance(item, StreamingOutput):
                        # Final output - emit message_delta event
                        gen_latency = time.perf_counter() - start_time
                        prompt_tokens = item.metadata.get("input_tokens", 0)

                        # Record metrics
                        record_generation(
                            model=request.model,
                            tokens=item.token_count,
                            latency=gen_latency,
                        )

                        # Build message_delta with usage stats
                        message_data: dict[str, Any] = {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": "end_turn",
                            },
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": item.token_count,
                                "total_tokens": prompt_tokens + item.token_count,
                            },
                            "metadata": {
                                "model": loaded.model_id,
                                "latency_ms": gen_latency * 1000,
                                "tokens_per_second": item.token_count / gen_latency
                                if gen_latency > 0
                                else 0,
                            },
                        }

                        # Include hidden state in final event if requested
                        if request.return_hidden_states and item.hidden_states:
                            final_layer_data = item.hidden_states.get(-1)
                            if final_layer_data is not None:
                                hidden_vector = tensor_to_list(final_layer_data)
                                hidden_shape = list(final_layer_data.shape)
                                dtype_str = str(final_layer_data.dtype).replace("torch.", "")

                                message_data["hidden_state"] = {
                                    "final": hidden_vector,
                                    "shape": hidden_shape,
                                    "layer": -1,
                                    "dtype": dtype_str,
                                }

                        yield f"event: message_delta\ndata: {json.dumps(message_data)}\n\n"

            except Exception as e:
                logger.exception(f"Streaming chat completion failed: {e}")
                error_data = {"type": "error", "error": {"message": str(e)}}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # ========================================================================
    # Endpoints
    # ========================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """
        Return overall service health and runtime information.
        
        Returns:
            HealthResponse: Contains `status` ("healthy" or other), `model_loaded` (the first loaded model ID or `None`), `gpu_info` (GPU state as a dict), and `config` with `max_loaded_models` and `default_layers`.
        """
        loaded_models = model_manager.list_loaded()
        # Update loaded models gauge
        set_models_loaded(len(loaded_models))
        return HealthResponse(
            status="healthy",
            model_loaded=loaded_models[0] if loaded_models else None,
            gpu_info=gpu_manager.to_dict(),
            config={
                "max_loaded_models": config.models.max_loaded,
                "default_layers": config.hidden_states.default_layers,
            },
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint.

        Returns metrics in Prometheus text format for scraping.
        """
        return Response(
            content=get_metrics(),
            media_type="text/plain; charset=utf-8",
        )

    @app.get("/models")
    async def list_models() -> dict[str, Any]:
        """
        Return information about currently loaded models and the configured maximum.

        Returns:
            info (dict): Dictionary with keys:
                - "loaded_models": a list of dictionaries, each containing details for a loaded model.
                - "max_models": the configured maximum number of models allowed to be loaded.
                - "auto_unload_minutes": minutes of inactivity before auto-unload (0 = disabled).
        """
        return {
            "loaded_models": model_manager.get_loaded_info(),
            "max_models": config.models.max_loaded,
            "auto_unload_minutes": config.models.auto_unload_minutes,
        }

    @app.get("/loaders")
    async def list_loaders() -> dict[str, Any]:
        """
        Provide information about available model loaders and their configured fallback order.
        
        Returns:
            info (dict): A dictionary with two keys:
                - "loaders": mapping of loader names to loader metadata as returned by the loader registry.
                - "fallback_order": list of loader names in the order they will be probed for a given model id.
        """
        return {
            "loaders": registry.list_loaders(),
            "fallback_order": registry.fallback_order,
        }

    @app.get("/loaders/probe/{model_id:path}")
    async def probe_model_loader(model_id: str) -> dict[str, Any]:
        """
        Determine which loader would handle the given model identifier without loading the model.
        
        Returns:
            probe_result (dict): Probe results describing the selected loader and related metadata.
        """
        # Handle URL encoding (-- back to /)
        model_id = model_id.replace("--", "/")
        return registry.probe_model(model_id)

    @app.post("/models/load", response_model=ModelLoadResponse)
    async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
        """
        Load the requested model into memory and return its load metadata.
        
        Returns:
            ModelLoadResponse: Contains the loaded model's identifier, device, dtype (stringified), hidden_size, num_layers, reported load_time_seconds, loader_type, and quantization.
        
        Raises:
            HTTPException: Raised with status code 500 if the model fails to load; the exception's detail contains the error message.
        """
        start_time = time.perf_counter()
        try:
            loaded = model_manager.get_or_load(
                model_id=request.model,
                device=request.device,
                dtype=request.dtype,
                loader_name=request.loader,
                quantization=request.quantization,
            )
            load_time = time.perf_counter() - start_time

            # Record metrics
            record_model_load(
                model=request.model,
                loader=loaded.loader_type,
                quantization=loaded.metadata.get("quantization", "none"),
                latency=load_time,
                success=True,
            )
            set_models_loaded(len(model_manager.list_loaded()))

            return ModelLoadResponse(
                model_id=loaded.model_id,
                device=str(loaded.device),
                dtype=str(loaded.dtype).replace("torch.", ""),
                hidden_size=loaded.hidden_size,
                num_layers=loaded.num_layers,
                load_time_seconds=loaded.metadata.get("load_time_seconds", 0),
                loader_type=loaded.loader_type,
                quantization=loaded.metadata.get("quantization", "none"),
            )
        except Exception as e:
            load_time = time.perf_counter() - start_time
            record_model_load(
                model=request.model,
                loader=request.loader or "auto",
                quantization=request.quantization or "none",
                latency=load_time,
                success=False,
            )
            logger.exception(f"Failed to load model: {request.model}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/models/{model_id}")
    async def unload_model(model_id: str) -> dict[str, Any]:
        """
        Unload a previously loaded model and free its resources.

        Parameters:
            model_id (str): Identifier of the model to unload. Instances of "--" in the string are normalized to "/".

        Returns:
            result (dict[str, Any]): Dictionary containing "status" set to "unloaded" and the normalized "model_id".

        Raises:
            HTTPException: 404 if the specified model is not currently loaded.
        """
        # Handle URL encoding
        model_id = model_id.replace("--", "/")
        success = model_manager.unload(model_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model not loaded: {model_id}")
        # Update loaded models gauge
        set_models_loaded(len(model_manager.list_loaded()))
        return {"status": "unloaded", "model_id": model_id}

    # ========================================================================
    # GPU Configuration Endpoints (Dynamic GPU Pool Management)
    # ========================================================================

    @app.post("/gpu/configure", response_model=GPUConfigureResponse)
    async def configure_gpu(request: GPUConfigureRequest) -> GPUConfigureResponse:
        """
        Dynamically update GPU configuration at runtime.

        Allows reconfiguring which GPU devices are available for model loading
        and which device is the default, without requiring a server restart.

        At least one of allowed_devices or default_device must be provided.
        Changes take effect immediately for subsequent model loads.

        Example request:
            {
                "allowed_devices": [0, 1],
                "default_device": 0
            }

        Args:
            request: GPU configuration changes to apply.

        Returns:
            GPUConfigureResponse: Updated GPU configuration with status message.

        Raises:
            HTTPException: 400 if validation fails (invalid device indices,
                empty allowed_devices list, or default_device not in allowed_devices).
            HTTPException: 500 for unexpected errors.
        """
        # Validate that at least one field is provided
        if request.allowed_devices is None and request.default_device is None:
            raise HTTPException(
                status_code=400,
                detail="At least one of 'allowed_devices' or 'default_device' must be provided",
            )

        changes: list[str] = []

        try:
            # Apply allowed_devices update if provided
            if request.allowed_devices is not None:
                gpu_manager.set_allowed_devices(request.allowed_devices)
                changes.append(f"allowed_devices updated to {request.allowed_devices}")

            # Apply default_device update if provided
            if request.default_device is not None:
                gpu_manager.set_default_device(request.default_device)
                changes.append(f"default_device updated to {request.default_device}")

            # Build response message
            message = "; ".join(changes) if changes else "No changes applied"

            return GPUConfigureResponse(
                allowed_devices=gpu_manager.allowed_devices,
                default_device=gpu_manager.default_device,
                available_devices=gpu_manager.available_devices,
                message=message,
            )

        except ValueError as e:
            # Validation errors from GPUManager methods
            logger.warning(f"GPU configuration validation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            # Unexpected errors
            logger.exception(f"GPU configuration failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/gpu/status", response_model=GPUStatusResponse)
    async def gpu_status() -> GPUStatusResponse:
        """
        Return comprehensive GPU configuration and runtime status.

        Provides detailed information about GPU availability, configuration,
        memory usage, and currently loaded models. Useful for monitoring
        and debugging multi-GPU setups.

        Returns:
            GPUStatusResponse: Contains:
                - has_gpu: Whether CUDA GPUs are available
                - available_devices: All physical GPU indices on the system
                - allowed_devices: Currently configured allowed GPU indices
                - default_device: Default device for model loading
                - memory_fraction: Configured max GPU memory fraction
                - gpu_memory: Per-GPU memory statistics (total, used, free)
                - loaded_models: Currently loaded models and their device assignments

        Example response:
            {
                "has_gpu": true,
                "available_devices": [0, 1],
                "allowed_devices": [0],
                "default_device": "cuda:0",
                "memory_fraction": 0.9,
                "gpu_memory": [
                    {
                        "device_index": 0,
                        "device_name": "NVIDIA RTX 4090",
                        "total_memory_mb": 24576.0,
                        "used_memory_mb": 8192.0,
                        "free_memory_mb": 16384.0,
                        "memory_utilization": 0.33
                    }
                ],
                "loaded_models": [
                    {
                        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "device": "cuda:0",
                        "dtype": "float16",
                        "idle_seconds": 45.2
                    }
                ]
            }
        """
        try:
            # Get GPU configuration from manager
            gpu_dict = gpu_manager.to_dict()

            # Build GPU memory info list
            gpu_memory_list: list[GPUMemoryInfo] = []
            if gpu_dict["has_gpu"]:
                gpu_info_list = gpu_manager.get_gpu_info()
                if not isinstance(gpu_info_list, list):
                    gpu_info_list = [gpu_info_list]

                for gpu_info in gpu_info_list:
                    total_mb = gpu_info.total_memory_gb * 1024
                    free_mb = gpu_info.free_memory_gb * 1024
                    used_mb = gpu_info.used_memory_gb * 1024
                    utilization = used_mb / total_mb if total_mb > 0 else 0.0

                    gpu_memory_list.append(
                        GPUMemoryInfo(
                            device_index=gpu_info.index,
                            device_name=gpu_info.name,
                            total_memory_mb=round(total_mb, 2),
                            used_memory_mb=round(used_mb, 2),
                            free_memory_mb=round(free_mb, 2),
                            memory_utilization=round(utilization, 4),
                        )
                    )

            # Get loaded models info from model manager
            loaded_models_list: list[LoadedModelInfo] = []
            loaded_info = model_manager.get_loaded_info()
            for model_info in loaded_info:
                loaded_models_list.append(
                    LoadedModelInfo(
                        model_id=model_info["model_id"],
                        device=model_info["device"],
                        dtype=model_info["dtype"],
                        idle_seconds=model_info["idle_seconds"],
                    )
                )

            return GPUStatusResponse(
                has_gpu=gpu_dict["has_gpu"],
                available_devices=gpu_manager.available_devices,
                allowed_devices=gpu_dict["allowed_devices"],
                default_device=gpu_dict["default_device"],
                memory_fraction=gpu_dict["memory_fraction"],
                gpu_memory=gpu_memory_list,
                loaded_models=loaded_models_list,
            )

        except Exception as e:
            logger.exception(f"GPU status retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """
        Generate text and optionally return serialized hidden states and attention weights.
        
        Parameters:
            request (GenerateRequest): Generation settings including model id, prompt, token limits, sampling params, and flags for returning hidden states or attention.
        
        Returns:
            GenerateResponse: Contains generated text, token_count, optional serialized `hidden_states`, optional serialized `attention_weights`, and `metadata` about the generation.
        
        Raises:
            HTTPException: On failure to load the model or generate (status code 500 with error detail).
        """
        start_time = time.perf_counter()
        try:
            # Get or load model (with optional loader override)
            loaded = model_manager.get_or_load(
                request.model,
                loader_name=request.loader,
            )

            # Expand "all" to full layer list if needed
            hidden_state_layers = _expand_hidden_state_layers(
                request.hidden_state_layers,
                loaded.num_layers,
            )

            # Generate using registry (uses appropriate loader for model)
            output = registry.generate(
                loaded_model=loaded,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                return_hidden_states=request.return_hidden_states,
                hidden_state_layers=hidden_state_layers,
                return_attention=request.return_attention,
                return_full_sequence=request.return_full_sequence,
            )

            # Record generation metrics
            gen_latency = time.perf_counter() - start_time
            record_generation(
                model=request.model,
                tokens=len(output.token_ids),
                latency=gen_latency,
            )

            # Serialize hidden states if present
            hidden_states_response = None
            if output.hidden_states:
                hidden_states_results = extract_hidden_states(output.hidden_states)
                hidden_states_response = serialize_hidden_states(
                    hidden_states_results,
                    format=request.hidden_state_format,
                )

            # Serialize attention if present
            attention_response = None
            if output.attention_weights:
                attention_response = serialize_hidden_states(
                    output.attention_weights,
                    format=request.hidden_state_format,
                )

            # Serialize full sequence hidden states for manifold analysis
            sequence_hidden_states_response: dict[str, SequenceHiddenStateResponse] | None = None
            if output.sequence_hidden_states:
                # Pydantic will coerce the dict to SequenceHiddenStateResponse at validation
                sequence_hidden_states_response = cast(
                    dict[str, SequenceHiddenStateResponse],
                    _serialize_sequence_hidden_states(
                        output.sequence_hidden_states,
                        format=request.hidden_state_format,
                    ),
                )

            return GenerateResponse(
                text=output.text,
                token_count=len(output.token_ids),
                hidden_states=hidden_states_response,
                attention_weights=attention_response,
                sequence_hidden_states=sequence_hidden_states_response,
                metadata=output.metadata,
            )

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ========================================================================
    # OpenAI-Compatible Chat Completions (WeaverCode Integration)
    # ========================================================================

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse:
        """OpenAI-compatible chat completion with hidden state extraction.

        This endpoint is designed for WeaverCode integration, providing:
        - Chat message format (messages array with role/content)
        - Chat template application (model-specific formatting)
        - Hidden state extraction in WeaverCode-expected format
        - Token usage breakdown (prompt/completion/total)
        - Streaming support via Server-Sent Events (SSE)

        The hidden state returned is the "boundary object" - the geometric
        representation of meaning before lm_head projection. This enables
        conveyance measurement between AI agents.

        When stream=true, returns SSE events:
        - content_block_delta: Contains text delta for each token
        - message_delta: Final event with completion info
        - error: If an error occurs during streaming

        Example request:
            {
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
                "return_hidden_states": true
            }

        Example response (non-streaming):
            {
                "text": "Hello! How can I help you today?",
                "usage": {
                    "prompt_tokens": 45,
                    "completion_tokens": 12,
                    "total_tokens": 57
                },
                "hidden_state": {
                    "final": [0.123, -0.456, ...],
                    "shape": [1, 2048],
                    "layer": -1,
                    "dtype": "float16"
                },
                "metadata": {...}
            }
        """
        if request.stream:
            return await _stream_chat_completions(request, model_manager, registry)

        start_time = time.perf_counter()
        try:
            # Get or load model (with optional device override)
            loaded = model_manager.get_or_load(
                request.model,
                device=request.device,
                loader_name=request.loader,
            )

            # Apply chat template to convert messages to prompt
            prompt = _apply_chat_template(
                request.messages,  # type: ignore[arg-type]
                loaded.tokenizer,
                loaded.model_id,
            )

            # Generate using the existing registry infrastructure
            output = registry.generate(
                loaded_model=loaded,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                return_hidden_states=request.return_hidden_states,
                hidden_state_layers=[-1],  # WeaverCode only needs final layer
            )

            # Record metrics
            gen_latency = time.perf_counter() - start_time
            record_generation(
                model=request.model,
                tokens=len(output.token_ids),
                latency=gen_latency,
            )

            # Build usage statistics (WeaverCode-expected format)
            prompt_tokens = output.metadata.get("input_tokens", 0)
            completion_tokens = len(output.token_ids)
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            # Extract hidden state in WeaverCode-expected format
            hidden_state_response: ChatCompletionHiddenState | None = None
            if request.return_hidden_states and output.hidden_states:
                # Get the last layer hidden state
                final_layer_data = output.hidden_states.get(-1)
                if final_layer_data is not None:
                    # Convert to list format
                    hidden_vector = tensor_to_list(final_layer_data)
                    hidden_shape = list(final_layer_data.shape)

                    # Determine dtype string
                    dtype_str = str(final_layer_data.dtype).replace("torch.", "")

                    hidden_state_response = ChatCompletionHiddenState(
                        final=hidden_vector,
                        shape=hidden_shape,
                        layer=-1,
                        dtype=dtype_str,
                    )

            return ChatCompletionResponse(
                text=output.text,
                usage=usage,
                hidden_state=hidden_state_response,
                metadata={
                    "model": loaded.model_id,
                    "latency_ms": gen_latency * 1000,
                    "tokens_per_second": completion_tokens / gen_latency
                    if gen_latency > 0
                    else 0,
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Chat completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Alias for convenience (some clients expect /v1/generate)
    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate_v1(request: GenerateRequest) -> GenerateResponse:
        """Alias for /generate with /v1/ prefix for API consistency."""
        result: GenerateResponse = await generate(request)
        return result

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest) -> EmbedResponse:
        """
        Produce an embedding for the provided text.
        
        If the underlying model is decoder-only, the embedding is taken from the last token's hidden state (which includes accumulated context). If `request.normalize` is true, the returned embedding is L2-normalized.
        
        Returns:
            EmbedResponse: Object containing `embedding` (list of floats), `shape` (list of ints), and `metadata`.
        """
        start_time = time.perf_counter()
        try:
            # Get or load model
            loaded = model_manager.get_or_load(request.model)

            # Extract embedding using registry (uses appropriate loader)
            output = registry.embed(
                loaded_model=loaded,
                text=request.text,
                pooling=request.pooling,
            )

            # Record embedding metrics
            embed_latency = time.perf_counter() - start_time
            record_embedding(model=request.model, latency=embed_latency)

            # Convert to list
            embedding_list = tensor_to_list(output.embedding)

            # Optionally L2 normalize
            if request.normalize:
                import numpy as np

                arr = np.array(embedding_list)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    embedding_list = (arr / norm).tolist()

            return EmbedResponse(
                embedding=embedding_list,
                shape=list(output.shape),
                metadata=output.metadata,
            )

        except Exception as e:
            logger.exception(f"Embedding failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/analyze")
    async def analyze_embedding(request: EmbedRequest) -> dict[str, Any]:
        """
        Compute diagnostic metrics for a single text embedding.
        
        Returns:
            analysis (dict): Analysis results produced by hidden-state analysis. Includes an "embedding_shape" key with the embedding shape as a list and any metadata from the embedding output.
        
        Raises:
            HTTPException: If embedding extraction or analysis fails.
        """
        try:
            loaded = model_manager.get_or_load(request.model)

            # Extract embedding using registry
            output = registry.embed(
                loaded_model=loaded,
                text=request.text,
                pooling=request.pooling,
            )

            # Create HiddenStateResult for analysis

            result = HiddenStateResult(
                vector=output.embedding.numpy(),
                shape=output.shape,
                layer=-1,
                dtype=str(output.embedding.dtype),
            )

            analysis = analyze_hidden_state(result)
            analysis["embedding_shape"] = list(output.shape)
            analysis.update(output.metadata)

            return analysis

        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ========================================================================
    # Kakeya Geometry Analysis Endpoints (Conveyance Measurement)
    # ========================================================================

    @app.post("/analyze/geometry", response_model=GeometryAnalysisResponse)
    async def analyze_geometry(
        request: GeometryAnalysisRequest,
    ) -> GeometryAnalysisResponse:
        """Perform Kakeya-inspired geometric analysis on hidden state vectors.

        Analyzes a set of hidden state vectors for geometric properties that
        may indicate information capacity and representation quality:

        - Wolf axiom compliance: density concentration in convex regions
        - Directional coverage: how well vectors span the ambient space
        - Grain structure: clustering patterns in the representations

        This endpoint accepts raw vectors (e.g., from multiple agent responses)
        and returns a full geometric analysis report.

        Example request:
            {
                "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
            }

        Returns:
            GeometryAnalysisResponse with Wolf axiom, coverage, and grain metrics.
        """
        import numpy as np

        start_time = time.perf_counter()
        try:
            # Convert to numpy array
            vectors = np.array(request.vectors, dtype=np.float32)

            if vectors.shape[0] < 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Need at least 3 vectors for analysis, got {vectors.shape[0]}",
                )

            # Run full Kakeya geometry analysis
            report: KakeyaGeometryReport = analyze_kakeya_geometry(vectors)

            analysis_time = (time.perf_counter() - start_time) * 1000

            return GeometryAnalysisResponse(
                overall_health=report.overall_health,
                num_vectors=report.num_vectors,
                ambient_dim=report.ambient_dim,
                wolf_axiom=WolfAxiomResponse(
                    max_density_ratio=report.wolf_axiom.max_density_ratio,
                    mean_density_ratio=report.wolf_axiom.mean_density_ratio,
                    uniformity_p_value=report.wolf_axiom.uniformity_p_value,
                    violation_count=report.wolf_axiom.violation_count,
                    severity=report.wolf_axiom.severity,
                ),
                directional_coverage=DirectionalCoverageResponse(
                    ambient_dim=report.directional_coverage.ambient_dim,
                    effective_dim=report.directional_coverage.effective_dim,
                    coverage_ratio=report.directional_coverage.coverage_ratio,
                    coverage_quality=report.directional_coverage.coverage_quality,
                    spherical_uniformity=report.directional_coverage.spherical_uniformity,
                    isotropy_score=report.directional_coverage.isotropy_score,
                ),
                grain_analysis=GrainAnalysisResponse(
                    num_grains=report.grain_analysis.num_grains,
                    grain_coverage=report.grain_analysis.grain_coverage,
                    mean_grain_size=report.grain_analysis.mean_grain_size,
                    mean_aspect_ratio=report.grain_analysis.mean_aspect_ratio,
                ),
                analysis_time_ms=analysis_time,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Geometry analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/analyze/bilateral", response_model=BilateralAnalysisResponse)
    async def analyze_bilateral(
        request: BilateralAnalysisRequest,
    ) -> BilateralAnalysisResponse:
        """Compare geometric properties between sender and receiver hidden states.

        This is the core conveyance measurement endpoint. It analyzes whether
        the geometric representations of sender and receiver hidden states
        align in ways that predict successful information transfer.

        The analysis computes:
        - Directional alignment: cosine similarity of mean directions
        - Subspace overlap: principal component space alignment
        - Grain alignment: cluster structure correspondence
        - Density similarity: Wolf axiom profile similarity
        - Effective dimension ratio: relative dimensionality

        HYPOTHESIS: Higher alignment scores should correlate with better
        task performance / information transfer success.

        Example request:
            {
                "sender_vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
                "receiver_vectors": [[0.5, 0.6, ...], [0.7, 0.8, ...], ...]
            }

        Returns:
            BilateralAnalysisResponse with alignment metrics.
        """
        import numpy as np

        start_time = time.perf_counter()
        try:
            # Convert to numpy arrays
            sender = np.array(request.sender_vectors, dtype=np.float32)
            receiver = np.array(request.receiver_vectors, dtype=np.float32)

            # Validate dimensions match
            if sender.shape[1] != receiver.shape[1]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dimension mismatch: sender has {sender.shape[1]}, "
                    f"receiver has {receiver.shape[1]}",
                )

            # Run bilateral geometry comparison
            result: BilateralGeometryResult = compare_bilateral_geometry(
                sender, receiver
            )

            analysis_time = (time.perf_counter() - start_time) * 1000

            return BilateralAnalysisResponse(
                directional_alignment=result.directional_alignment,
                subspace_overlap=result.subspace_overlap,
                grain_alignment=result.grain_alignment,
                density_similarity=result.density_similarity,
                effective_dim_ratio=result.effective_dim_ratio,
                overall_alignment=result.overall_alignment,
                analysis_time_ms=analysis_time,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Bilateral analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/generate/stream")
    async def generate_stream(request: StreamingGenerateRequest) -> StreamingResponse:
        """
        Stream generation results as Server-Sent Events.
        
        Emits SSE messages for the stream: `token` events for each generated token, a `done` event with the final output (text, token_count, token_ids, metadata) and optional serialized hidden states, and an `error` event if generation fails.
        
        Returns:
            StreamingResponse: An SSE stream that yields JSON-encoded event data for `token`, `done`, and `error` events.
        """

        async def event_generator() -> AsyncIterator[str]:
            """
            Produce Server-Sent Events (SSE) messages for a streaming generation request, emitting per-token updates and a final completion event, and an error event if streaming fails.
            
            Yields:
                SSE-formatted strings representing one of:
                  - a `token` event with fields `token`, `token_id`, `is_finished`, and `finish_reason`;
                  - a `done` event with final `text`, `token_count`, `token_ids`, `metadata`, and optionally `hidden_states` (serialized according to the request format);
                  - an `error` event with an `error` message when an exception occurs.
            """
            try:
                # Get or load model
                loaded = model_manager.get_or_load(
                    request.model,
                    loader_name=request.loader,
                )

                # Expand "all" to full layer list if needed
                hidden_state_layers = _expand_hidden_state_layers(
                    request.hidden_state_layers,
                    loaded.num_layers,
                )

                # Stream tokens
                for item in registry.generate_stream(
                    loaded_model=loaded,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    return_hidden_states=request.return_hidden_states,
                    hidden_state_layers=hidden_state_layers,
                ):
                    if isinstance(item, StreamingToken):
                        # Send token event
                        data = {
                            "token": item.token,
                            "token_id": item.token_id,
                            "is_finished": item.is_finished,
                            "finish_reason": item.finish_reason,
                        }
                        yield f"event: token\ndata: {json.dumps(data)}\n\n"

                    elif isinstance(item, StreamingOutput):
                        # Send final output event
                        output_data: dict[str, Any] = {
                            "text": item.text,
                            "token_count": item.token_count,
                            "token_ids": item.token_ids,
                            "metadata": item.metadata,
                        }

                        # Serialize hidden states if present
                        if item.hidden_states:
                            hidden_states_results = extract_hidden_states(item.hidden_states)
                            output_data["hidden_states"] = serialize_hidden_states(
                                hidden_states_results,
                                format=request.hidden_state_format,
                            )

                        yield f"event: done\ndata: {json.dumps(output_data)}\n\n"

            except Exception as e:
                logger.exception(f"Streaming generation failed: {e}")
                error_data = {"error": str(e)}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    @app.post("/generate/batch", response_model=BatchGenerateResponse)
    async def generate_batch(request: BatchGenerateRequest) -> BatchGenerateResponse:
        """
        Generate text for each prompt in the request using a single loaded model.
        
        Processes all prompts with the same model instance and returns per-prompt generation outputs, including optional serialized hidden states.
        
        Parameters:
            request (BatchGenerateRequest): Batch generation request containing the model identifier, list of prompts, and generation options (max_tokens, temperature, top_p, hidden-state flags, and formatting).
        
        Returns:
            BatchGenerateResponse: Object containing:
                - results (list[GenerateResponse]): Generation result for each prompt.
                - total_tokens (int): Sum of token counts across all prompts.
                - total_time_ms (float): Total processing time in milliseconds.
                - prompts_processed (int): Number of prompts processed.
        """
        start_time = time.time()
        results: list[GenerateResponse] = []
        total_tokens = 0

        try:
            # Get or load model once for all prompts
            loaded = model_manager.get_or_load(
                request.model,
                loader_name=request.loader,
            )

            # Expand "all" to full layer list if needed
            hidden_state_layers = _expand_hidden_state_layers(
                request.hidden_state_layers,
                loaded.num_layers,
            )

            # Process each prompt
            for prompt in request.prompts:
                output = registry.generate(
                    loaded_model=loaded,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    return_hidden_states=request.return_hidden_states,
                    hidden_state_layers=hidden_state_layers,
                )

                # Serialize hidden states if present
                hidden_states_response = None
                if output.hidden_states:
                    hidden_states_results = extract_hidden_states(output.hidden_states)
                    hidden_states_response = serialize_hidden_states(
                        hidden_states_results,
                        format=request.hidden_state_format,
                    )

                results.append(
                    GenerateResponse(
                        text=output.text,
                        token_count=len(output.token_ids),
                        hidden_states=hidden_states_response,
                        attention_weights=None,
                        metadata=output.metadata,
                    )
                )
                total_tokens += len(output.token_ids)

            total_time = (time.time() - start_time) * 1000

            return BatchGenerateResponse(
                results=results,
                total_tokens=total_tokens,
                total_time_ms=total_time,
                prompts_processed=len(request.prompts),
            )

        except Exception as e:
            logger.exception(f"Batch generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/embed/batch", response_model=BatchEmbedResponse)
    async def embed_batch(request: BatchEmbedRequest) -> BatchEmbedResponse:
        """
        Generate embeddings for multiple texts using a single loaded model.
        
        Processes the provided texts with the same model and returns their embeddings, per-embedding shapes, total processing time in milliseconds, and the number of texts processed.
        
        Parameters:
            request (BatchEmbedRequest): Batch embedding request containing:
                - model: model identifier to use.
                - texts: list of input strings to embed.
                - pooling: pooling strategy applied to model outputs.
                - normalize: if true, L2-normalize each embedding vector.
        
        Returns:
            BatchEmbedResponse: Contains:
                - embeddings: list of embedding vectors (one list of floats per input text).
                - shapes: list of shapes corresponding to each embedding.
                - total_time_ms: total time spent processing the batch in milliseconds.
                - texts_processed: number of texts processed.
        
        Raises:
            HTTPException: on failure to load the model or compute embeddings (returns HTTP 500).
        """
        import numpy as np

        start_time = time.time()
        embeddings: list[list[float]] = []
        shapes: list[list[int]] = []

        try:
            # Get or load model once for all texts
            loaded = model_manager.get_or_load(request.model)

            # Process each text
            for text in request.texts:
                output = registry.embed(
                    loaded_model=loaded,
                    text=text,
                    pooling=request.pooling,
                )

                # Convert to list
                embedding_list = tensor_to_list(output.embedding)

                # Optionally L2 normalize
                if request.normalize:
                    arr = np.array(embedding_list)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        embedding_list = (arr / norm).tolist()

                embeddings.append(embedding_list)
                shapes.append(list(output.shape))

            total_time = (time.time() - start_time) * 1000

            return BatchEmbedResponse(
                embeddings=embeddings,
                shapes=shapes,
                total_time_ms=total_time,
                texts_processed=len(request.texts),
            )

        except Exception as e:
            logger.exception(f"Batch embedding failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ========================================================================
    # Activation Patching Endpoints (Causal Intervention Analysis)
    # ========================================================================

    # Store patching state on app.state for proper lifecycle management
    # This ensures cleanup on shutdown and consistent state across requests
    app.state.patching_experiments: dict[str, PatchingExperiment] = {}
    app.state.patching_recording_store = RecordingStore()
    app.state.patching_cache_manager = CacheManager(
        max_size_mb=config.patching.max_cache_size_mb if hasattr(config, 'patching') else 4096,
        cleanup_on_exit=True,
    )

    # Local aliases for backward compatibility within endpoint functions
    patching_experiments = app.state.patching_experiments
    patching_recording_store = app.state.patching_recording_store
    patching_cache_manager = app.state.patching_cache_manager

    @app.post("/api/patching/configure", response_model=PatchingConfigureResponse)
    async def configure_patching(
        request: PatchingConfigureRequest,
    ) -> PatchingConfigureResponse:
        """Configure a new patching experiment.

        Creates an experiment configuration for activation patching studies.
        This configures which layers and components will be targeted for
        causal intervention analysis.

        Example request:
            {
                "experiment_name": "layer_sweep_study",
                "layers": [0, 5, 10],
                "components": ["resid_pre", "attn"]
            }

        Returns:
            PatchingConfigureResponse with experiment ID and configuration.
        """
        import uuid

        try:
            # Create experiment configuration
            experiment_config = ExperimentConfig(
                name=request.experiment_name,
                layers=request.layers,
                components=request.components,
                validate_shapes=request.validate_shapes,
                cleanup_on_completion=request.cleanup_on_completion,
            )

            # Create the experiment
            experiment = PatchingExperiment(
                config=experiment_config,
                cache_manager=patching_cache_manager,
            )

            # Store the experiment
            experiment_id = experiment.experiment_id
            patching_experiments[experiment_id] = experiment

            logger.info(f"Configured patching experiment: {experiment_id}")

            return PatchingConfigureResponse(
                experiment_id=experiment_id,
                experiment_name=request.experiment_name,
                layers=request.layers,
                components=request.components,
                status="configured",
                message=f"Experiment configured with {len(request.layers)} layers and {len(request.components)} components",
            )

        except Exception as e:
            logger.exception(f"Patching configuration failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/patching/run", response_model=PatchingRunResponse)
    async def run_patching(request: PatchingRunRequest) -> PatchingRunResponse:
        """Run a patching experiment with clean and corrupted inputs.

        Executes the three-path patching workflow:
        1. Clean path: Run model with clean input, cache activations
        2. Corrupted path: Run model with corrupted input, cache activations
        3. Patched paths: Run corrupted input with clean activations patched in

        This enables causal analysis of how hidden state modifications at
        specific layers affect the model's output.

        Example request:
            {
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "clean_input": "The capital of France is",
                "corrupted_input": "The capital of Germany is",
                "layers": [5, 10]
            }

        Returns:
            PatchingRunResponse with results from all three paths.
        """
        start_time = time.perf_counter()

        try:
            # Get or create experiment
            experiment: PatchingExperiment | None = None
            if request.experiment_id and request.experiment_id in patching_experiments:
                experiment = patching_experiments[request.experiment_id]
            else:
                # Create a new experiment with default config
                experiment_config = ExperimentConfig(
                    name=f"ad_hoc_experiment_{int(time.time())}",
                    layers=request.layers or [],
                    components=request.components or ["resid_pre"],
                    validate_shapes=True,
                    cleanup_on_completion=False,
                )
                experiment = PatchingExperiment(
                    config=experiment_config,
                    cache_manager=patching_cache_manager,
                )
                patching_experiments[experiment.experiment_id] = experiment

            # Get or load model
            loaded = model_manager.get_or_load(
                request.model,
                device=request.device,
            )

            # Determine layers to patch
            layers_to_patch = request.layers or experiment.config.layers
            if not layers_to_patch:
                # Default to a subset of layers if none specified
                layers_to_patch = [0, loaded.num_layers // 2, loaded.num_layers - 1]

            components_to_patch = request.components or experiment.config.components

            # Create experiment record
            experiment.create_record()

            # ================================================================
            # Step 1: Run clean path and cache activations
            # ================================================================
            clean_start = time.perf_counter()

            # Generate with hidden state extraction
            clean_output = registry.generate(
                loaded_model=loaded,
                prompt=request.clean_input,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                return_hidden_states=True,
                hidden_state_layers=layers_to_patch,
            )

            # Cache the hidden states
            clean_cache = patching_cache_manager.get_or_create_cache(
                f"{experiment.experiment_id}_clean"
            )
            if clean_output.hidden_states:
                for layer_idx, tensor in clean_output.hidden_states.items():
                    for component in components_to_patch:
                        clean_cache.store(
                            layer=layer_idx,
                            component=component,
                            activation=tensor,
                        )

            clean_time_ms = (time.perf_counter() - clean_start) * 1000

            # Record clean path
            experiment.recorder.record_clean_path(
                output=None,  # We store text separately
                cache=clean_cache,
                input_text=request.clean_input,
                generation_time_ms=clean_time_ms,
                metadata={"output_text": clean_output.text},
            )

            clean_path_result = PatchingPathResult(
                path_type="clean",
                output_text=clean_output.text,
                generation_time_ms=clean_time_ms,
                num_cached_activations=clean_cache.num_entries,
                metadata={"token_count": len(clean_output.token_ids)},
            )

            # ================================================================
            # Step 2: Run corrupted path and cache activations
            # ================================================================
            corrupted_start = time.perf_counter()

            corrupted_output = registry.generate(
                loaded_model=loaded,
                prompt=request.corrupted_input,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                return_hidden_states=True,
                hidden_state_layers=layers_to_patch,
            )

            # Cache the corrupted hidden states
            corrupted_cache = patching_cache_manager.get_or_create_cache(
                f"{experiment.experiment_id}_corrupted"
            )
            if corrupted_output.hidden_states:
                for layer_idx, tensor in corrupted_output.hidden_states.items():
                    for component in components_to_patch:
                        corrupted_cache.store(
                            layer=layer_idx,
                            component=component,
                            activation=tensor,
                        )

            corrupted_time_ms = (time.perf_counter() - corrupted_start) * 1000

            # Record corrupted path
            experiment.recorder.record_corrupted_path(
                output=None,
                cache=corrupted_cache,
                input_text=request.corrupted_input,
                generation_time_ms=corrupted_time_ms,
                metadata={"output_text": corrupted_output.text},
            )

            corrupted_path_result = PatchingPathResult(
                path_type="corrupted",
                output_text=corrupted_output.text,
                generation_time_ms=corrupted_time_ms,
                num_cached_activations=corrupted_cache.num_entries,
                metadata={"token_count": len(corrupted_output.token_ids)},
            )

            # ================================================================
            # Step 3: Run patched paths for each layer/component
            # ================================================================
            patched_results: list[PatchingLayerResult] = []

            for layer in layers_to_patch:
                for component in components_to_patch:
                    patch_start = time.perf_counter()

                    # Get the clean activation for this layer/component
                    clean_activation = clean_cache.get_tensor(layer, component)

                    if clean_activation is not None:
                        # Create hook point for this layer/component
                        hook_point = HookPoint(
                            layer=layer,
                            component=HookComponent.from_string(component),
                        )
                        hook_name = hook_point.to_hook_name(loaded.num_layers)

                        # TODO: Implement actual activation patching with HookedTransformer
                        # Currently this is a PLACEHOLDER that re-runs generation without
                        # patching. To properly implement:
                        # 1. Load model as HookedTransformer
                        # 2. Use clean_activation with create_patching_hook()
                        # 3. Call model.run_with_hooks() with the patching hook
                        # See src/patching/experiments.py for reference implementation
                        patched_output = registry.generate(
                            loaded_model=loaded,
                            prompt=request.corrupted_input,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            return_hidden_states=False,
                        )

                        patch_time_ms = (time.perf_counter() - patch_start) * 1000

                        patched_results.append(
                            PatchingLayerResult(
                                layer=layer,
                                component=component,
                                hook_name=hook_name,
                                patched_output=patched_output.text,
                                execution_time_ms=patch_time_ms,
                                shape_matched=True,
                            )
                        )

                        # Record patched path
                        experiment.recorder.record_patched_path(
                            output=None,
                            patch_info={
                                "layer": layer,
                                "component": component,
                                "hook_name": hook_name,
                            },
                            generation_time_ms=patch_time_ms,
                        )

            # Finalize experiment
            experiment.finalize(success=True)

            # Store recording for later retrieval
            recording = PathRecording.from_recorder(
                recorder=experiment.recorder,
                experiment_name=experiment.config.name,
            )
            patching_recording_store.add_recording(recording)

            duration_ms = (time.perf_counter() - start_time) * 1000

            return PatchingRunResponse(
                experiment_id=experiment.experiment_id,
                experiment_name=experiment.config.name,
                status="completed",
                clean_path=clean_path_result,
                corrupted_path=corrupted_path_result,
                patched_results=patched_results,
                duration_ms=duration_ms,
                layers_patched=layers_to_patch,
                components_patched=components_to_patch,
                metadata={
                    "model": request.model,
                    "num_patched_runs": len(patched_results),
                },
            )

        except Exception as e:
            logger.exception(f"Patching experiment failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/patching/results/{experiment_id}", response_model=PatchingResultsResponse)
    async def get_patching_results(experiment_id: str) -> PatchingResultsResponse:
        """Retrieve results from a completed patching experiment.

        Fetches the recorded outputs from all three paths (clean, corrupted, patched)
        and optionally computes causal effect metrics.

        Args:
            experiment_id: The experiment identifier to retrieve results for.

        Returns:
            PatchingResultsResponse with experiment results and causal effects.
        """
        try:
            # Check if recording exists
            recording = patching_recording_store.get_recording(experiment_id)

            if recording is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment not found: {experiment_id}",
                )

            # Extract outputs (output_text stored in metadata during recording)
            clean_output = None
            if recording.clean_output:
                clean_output = recording.clean_output.metadata.get("output_text")

            corrupted_output = None
            if recording.corrupted_output:
                corrupted_output = recording.corrupted_output.metadata.get("output_text")

            # Compute causal effects (simplified for now)
            causal_effects: list[CausalEffectResult] = []
            for patched in recording.patched_outputs:
                patch_info = patched.metadata.get("patch_info", {})
                layer = patch_info.get("layer", 0)
                component = patch_info.get("component", "unknown")

                # Use simple text-based metric (token difference)
                # In a full implementation, this would use actual logit differences
                causal_effects.append(
                    CausalEffectResult(
                        layer=layer,
                        component=component,
                        causal_effect=0.0,  # Placeholder
                        recovery_rate=0.0,  # Placeholder
                        clean_baseline=0.0,
                        corrupted_metric=0.0,
                        patched_metric=0.0,
                    )
                )

            return PatchingResultsResponse(
                experiment_id=experiment_id,
                experiment_name=recording.experiment_name,
                status="completed",
                has_all_paths=recording.has_all_paths,
                clean_output=clean_output,
                corrupted_output=corrupted_output,
                num_patched_paths=recording.num_patched_paths,
                causal_effects=causal_effects,
                metadata=recording.metadata,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to retrieve patching results: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/patching/cache/status", response_model=PatchingCacheStatusResponse)
    async def get_patching_cache_status() -> PatchingCacheStatusResponse:
        """Get the status of the patching activation cache.

        Returns memory usage, cache utilization, and list of active caches
        for monitoring patching experiment resource usage.

        Returns:
            PatchingCacheStatusResponse with cache status information.
        """
        try:
            stats = patching_cache_manager.get_memory_stats()

            return PatchingCacheStatusResponse(
                num_caches=stats["num_caches"],
                total_size_mb=stats["total_size_mb"],
                max_size_mb=stats["max_size_mb"],
                utilization=stats["utilization"],
                cache_ids=patching_cache_manager.cache_ids,
                cache_details=stats["cache_sizes"],
            )

        except Exception as e:
            logger.exception(f"Failed to get cache status: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/api/patching/cache/{cache_id}")
    async def delete_patching_cache(cache_id: str) -> dict[str, Any]:
        """Delete a specific patching activation cache.

        Frees memory by removing cached activations for a specific experiment run.

        Args:
            cache_id: The cache identifier to delete.

        Returns:
            Status message indicating success or failure.
        """
        try:
            success = patching_cache_manager.remove_cache(cache_id)

            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cache not found: {cache_id}",
                )

            return {
                "status": "deleted",
                "cache_id": cache_id,
                "message": f"Cache {cache_id} deleted successfully",
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to delete cache: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/api/patching/cache")
    async def clear_all_patching_caches() -> dict[str, Any]:
        """Clear all patching activation caches.

        Frees all memory used by cached activations from patching experiments.

        Returns:
            Status message with number of caches cleared.
        """
        try:
            num_cleared = patching_cache_manager.clear_memory()

            return {
                "status": "cleared",
                "entries_cleared": num_cleared,
                "message": f"Cleared {num_cleared} cache entries",
            }

        except Exception as e:
            logger.exception(f"Failed to clear caches: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/patching/experiments")
    async def list_patching_experiments() -> dict[str, Any]:
        """List all configured patching experiments.

        Returns a summary of all experiments that have been configured,
        including their status and configuration.

        Returns:
            Dictionary with list of experiment summaries.
        """
        try:
            experiments_list = []
            for exp_id, experiment in patching_experiments.items():
                experiments_list.append({
                    "experiment_id": exp_id,
                    "experiment_name": experiment.config.name,
                    "layers": experiment.config.layers,
                    "components": experiment.config.components,
                    "status": experiment.record.status if experiment.record else "configured",
                })

            return {
                "experiments": experiments_list,
                "total_count": len(experiments_list),
            }

        except Exception as e:
            logger.exception(f"Failed to list experiments: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e


    # ========================================================================
    # Experiment Persistence Endpoints
    # ========================================================================

    @app.get("/experiments", response_model=ExperimentListResponse)
    async def list_experiments(
        limit: int = 100,
        offset: int = 0,
        model: str | None = None,
        status: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> ExperimentListResponse:
        """List persisted experiments with optional filtering.

        Query experiments stored in the persistence layer. Supports filtering
        by model, status, and date range with pagination.

        Parameters:
            limit: Maximum number of experiments to return (default: 100).
            offset: Number of experiments to skip for pagination (default: 0).
            model: Filter by model identifier.
            status: Filter by experiment status (running, completed, failed).
            date_from: Filter experiments created on or after this datetime.
            date_to: Filter experiments created on or before this datetime.

        Returns:
            ExperimentListResponse with experiments and pagination info.

        Raises:
            HTTPException: 503 if persistence is not enabled.
        """
        if persistence is None:
            raise HTTPException(
                status_code=503,
                detail="Persistence is not enabled. Set persistence.enabled=true in config.",
            )

        try:
            # Create query interface
            query = ExperimentQuery(persistence.db._db_path)
            try:
                experiments = query.list_experiments(
                    limit=limit,
                    offset=offset,
                    model=model,
                    status=status,
                    date_from=date_from,
                    date_to=date_to,
                )
                total = query.count_experiments(model=model, status=status)
            finally:
                query.close()

            return ExperimentListResponse(
                experiments=[_experiment_record_to_response(exp) for exp in experiments],
                total=total,
                limit=limit,
                offset=offset,
            )

        except Exception as e:
            logger.exception(f"Failed to list experiments: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/experiments/{experiment_id}", response_model=ExperimentDetailResponse)
    async def get_experiment(experiment_id: str) -> ExperimentDetailResponse:
        """Get detailed information about a specific experiment.

        Retrieves the experiment metadata along with all associated
        conversations, hidden state records, metrics, and summary statistics.

        Parameters:
            experiment_id: The unique identifier of the experiment.

        Returns:
            ExperimentDetailResponse with full experiment data.

        Raises:
            HTTPException: 404 if experiment not found, 503 if persistence not enabled.
        """
        if persistence is None:
            raise HTTPException(
                status_code=503,
                detail="Persistence is not enabled. Set persistence.enabled=true in config.",
            )

        try:
            # Get experiment data
            data = persistence.get_experiment_data(experiment_id, include_hidden_states=False)

            if data is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment not found: {experiment_id}",
                )

            experiment: ExperimentRecord = data["experiment"]
            conversations: list[ConversationRecord] = data["conversations"]
            hidden_state_records: list[HiddenStateRecord] = data["hidden_state_records"]
            metrics: list[MetricRecord] = data["metrics"]

            # Build summary
            summary = ExperimentSummaryResponse(
                experiment_id=experiment.id,
                model=experiment.model,
                created_at=experiment.created_at,
                status=experiment.status,
                conversation_count=len(conversations),
                metric_count=len(metrics),
                hidden_state_count=len(hidden_state_records),
            )

            return ExperimentDetailResponse(
                experiment=_experiment_record_to_response(experiment),
                conversations=[_conversation_record_to_response(c) for c in conversations],
                hidden_states=[_hidden_state_record_to_response(hs) for hs in hidden_state_records],
                metrics=[_metric_record_to_response(m) for m in metrics],
                summary=summary,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to get experiment {experiment_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/experiments/{experiment_id}/conversations", response_model=ConversationsListResponse)
    async def get_experiment_conversations(
        experiment_id: str,
        role: str | None = None,
    ) -> ConversationsListResponse:
        """Get conversation messages for an experiment.

        Retrieves all conversation messages associated with the experiment,
        optionally filtered by role.

        Parameters:
            experiment_id: The unique identifier of the experiment.
            role: Optional filter for specific role (user, assistant, system).

        Returns:
            ConversationsListResponse with conversation messages.

        Raises:
            HTTPException: 404 if experiment not found, 503 if persistence not enabled.
        """
        if persistence is None:
            raise HTTPException(
                status_code=503,
                detail="Persistence is not enabled. Set persistence.enabled=true in config.",
            )

        try:
            # Verify experiment exists
            experiment = persistence.get_experiment(experiment_id)
            if experiment is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment not found: {experiment_id}",
                )

            # Get conversations using query interface
            query = ExperimentQuery(persistence.db._db_path)
            try:
                conversations = query.get_conversations(experiment_id, role=role)
            finally:
                query.close()

            return ConversationsListResponse(
                experiment_id=experiment_id,
                conversations=[_conversation_record_to_response(c) for c in conversations],
                total=len(conversations),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to get conversations for {experiment_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/experiments/{experiment_id}/hidden_states", response_model=HiddenStatesListResponse)
    async def get_experiment_hidden_states(
        experiment_id: str,
        layer: int | None = None,
    ) -> HiddenStatesListResponse:
        """Get hidden state records for an experiment.

        Retrieves hidden state metadata for the experiment. The actual tensor
        data can be loaded separately using the file paths returned.

        Parameters:
            experiment_id: The unique identifier of the experiment.
            layer: Optional filter for specific layer index.

        Returns:
            HiddenStatesListResponse with hidden state records and layer info.

        Raises:
            HTTPException: 404 if experiment not found, 503 if persistence not enabled.
        """
        if persistence is None:
            raise HTTPException(
                status_code=503,
                detail="Persistence is not enabled. Set persistence.enabled=true in config.",
            )

        try:
            # Verify experiment exists
            experiment = persistence.get_experiment(experiment_id)
            if experiment is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment not found: {experiment_id}",
                )

            # Get hidden states using query interface
            query = ExperimentQuery(persistence.db._db_path)
            try:
                hidden_states = query.get_hidden_states(experiment_id, layer=layer)
                available_layers = query.get_hidden_state_layers(experiment_id)
            finally:
                query.close()

            return HiddenStatesListResponse(
                experiment_id=experiment_id,
                hidden_states=[_hidden_state_record_to_response(hs) for hs in hidden_states],
                layers=available_layers,
                total=len(hidden_states),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to get hidden states for {experiment_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/experiments/{experiment_id}/metrics")
    async def get_experiment_metrics(
        experiment_id: str,
        metric_name: str | None = None,
    ) -> dict[str, Any]:
        """Get metrics for an experiment.

        Retrieves all metrics recorded for the experiment, optionally
        filtered by metric name.

        Parameters:
            experiment_id: The unique identifier of the experiment.
            metric_name: Optional filter for specific metric type.

        Returns:
            Dict with experiment_id, metrics list, and metric names.

        Raises:
            HTTPException: 404 if experiment not found, 503 if persistence not enabled.
        """
        if persistence is None:
            raise HTTPException(
                status_code=503,
                detail="Persistence is not enabled. Set persistence.enabled=true in config.",
            )

        try:
            # Verify experiment exists
            experiment = persistence.get_experiment(experiment_id)
            if experiment is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment not found: {experiment_id}",
                )

            # Get metrics using query interface
            query = ExperimentQuery(persistence.db._db_path)
            try:
                metrics = query.get_metrics(experiment_id, metric_name=metric_name)
                metric_names = query.get_metric_names(experiment_id)
            finally:
                query.close()

            return {
                "experiment_id": experiment_id,
                "metrics": [_metric_record_to_response(m) for m in metrics],
                "metric_names": metric_names,
                "total": len(metrics),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to get metrics for {experiment_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ========================================================================
    # Experiment Export Endpoints
    # ========================================================================

    @app.post("/experiments/export", response_model=ExperimentExportResponse)
    async def export_experiment(
        request: ExperimentExportRequest,
    ) -> ExperimentExportResponse:
        """Export an experiment to the specified format.

        Exports experiment data to JSON, CSV, or Parquet format. The exported
        files include experiment metadata, conversations, metrics, and hidden
        state references.

        Parameters:
            request: Export request with experiment_id, format, and options.

        Returns:
            ExperimentExportResponse with paths to created files.

        Raises:
            HTTPException: 400 if invalid format, 404 if experiment not found,
                          503 if persistence not enabled.
        """
        if persistence is None:
            raise HTTPException(
                status_code=503,
                detail="Persistence is not enabled. Set persistence.enabled=true in config.",
            )

        # Validate format
        valid_formats = ("json", "csv", "parquet")
        if request.format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format: {request.format}. Must be one of {valid_formats}.",
            )

        try:
            # Verify experiment exists
            experiment = persistence.get_experiment(request.experiment_id)
            if experiment is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment not found: {request.experiment_id}",
                )

            # Create exporter
            exporter = ExperimentExporter(
                db_path=persistence.db._db_path,
                storage_dir=persistence.storage._storage_dir,
            )

            try:
                # Determine output path
                output_dir = request.output_path or str(config.persistence.export_dir)

                output_files: dict[str, str] = {}

                if request.format == "json":
                    from pathlib import Path
                    output_path = Path(output_dir) / f"{request.experiment_id}.json"
                    result_path = exporter.export_json(
                        experiment_id=request.experiment_id,
                        output_path=output_path,
                        include_hidden_state_data=request.include_hidden_states,
                    )
                    output_files["json"] = str(result_path)

                elif request.format == "csv":
                    from pathlib import Path
                    base_path = Path(output_dir) / request.experiment_id
                    result_paths = exporter.export_csv(
                        experiment_id=request.experiment_id,
                        output_path=base_path,
                    )
                    output_files = {k: str(v) for k, v in result_paths.items()}

                elif request.format == "parquet":
                    from pathlib import Path
                    base_path = Path(output_dir) / request.experiment_id
                    result_paths = exporter.export_parquet(
                        experiment_id=request.experiment_id,
                        output_path=base_path,
                    )
                    output_files = {k: str(v) for k, v in result_paths.items()}

            finally:
                exporter.close()

            logger.info(
                f"Exported experiment {request.experiment_id} to {request.format}: "
                f"{len(output_files)} files"
            )

            return ExperimentExportResponse(
                experiment_id=request.experiment_id,
                format=request.format,
                output_files=output_files,
                total_files=len(output_files),
            )

        except ExportError as e:
            logger.error(f"Export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to export experiment {request.experiment_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/experiments/export/batch", response_model=ExperimentBatchExportResponse)
    async def export_experiments_batch(
        request: ExperimentBatchExportRequest,
    ) -> ExperimentBatchExportResponse:
        """Export multiple experiments to the specified format.

        Exports multiple experiments at once, creating a separate file (or set
        of files for CSV/Parquet) for each experiment in the output directory.

        Parameters:
            request: Batch export request with experiment_ids, format, and output_dir.

        Returns:
            ExperimentBatchExportResponse with paths to all created files.

        Raises:
            HTTPException: 400 if invalid format, 503 if persistence not enabled.
        """
        if persistence is None:
            raise HTTPException(
                status_code=503,
                detail="Persistence is not enabled. Set persistence.enabled=true in config.",
            )

        # Validate format
        valid_formats = ("json", "csv", "parquet")
        if request.format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format: {request.format}. Must be one of {valid_formats}.",
            )

        try:
            # Create exporter
            exporter = ExperimentExporter(
                db_path=persistence.db._db_path,
                storage_dir=persistence.storage._storage_dir,
            )

            try:
                # Determine output directory
                output_dir = request.output_dir or str(config.persistence.export_dir)

                # Export all experiments
                results = exporter.export_experiments(
                    experiment_ids=request.experiment_ids,
                    output_dir=output_dir,
                    format=request.format,
                )

                # Convert Path objects to strings
                exported: dict[str, dict[str, str]] = {}
                for exp_id, paths in results.items():
                    if isinstance(paths, dict):
                        exported[exp_id] = {k: str(v) for k, v in paths.items()}
                    else:
                        exported[exp_id] = {"file": str(paths)}

            finally:
                exporter.close()

            logger.info(
                f"Batch exported {len(exported)} experiments to {request.format}"
            )

            return ExperimentBatchExportResponse(
                exported=exported,
                total_experiments=len(exported),
                format=request.format,
            )

        except ExportError as e:
            logger.error(f"Batch export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to batch export experiments: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/experiments/export/summary", response_model=ExperimentSummaryExportResponse)
    async def export_experiments_summary(
        request: ExperimentSummaryExportRequest,
    ) -> ExperimentSummaryExportResponse:
        """Export a summary of all experiments.

        Creates a summary file containing basic information about all experiments
        without detailed conversations or hidden states. Useful for quick overview
        and analysis.

        Parameters:
            request: Summary export request with format, output_path, and limit.

        Returns:
            ExperimentSummaryExportResponse with path to created file.

        Raises:
            HTTPException: 400 if invalid format, 503 if persistence not enabled.
        """
        if persistence is None:
            raise HTTPException(
                status_code=503,
                detail="Persistence is not enabled. Set persistence.enabled=true in config.",
            )

        # Validate format
        valid_formats = ("json", "csv", "parquet")
        if request.format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format: {request.format}. Must be one of {valid_formats}.",
            )

        try:
            # Create exporter
            exporter = ExperimentExporter(
                db_path=persistence.db._db_path,
                storage_dir=persistence.storage._storage_dir,
            )

            try:
                # Determine output path
                from pathlib import Path

                if request.output_path:
                    output_path = request.output_path
                else:
                    export_dir = Path(config.persistence.export_dir)
                    extension = "json" if request.format == "json" else request.format
                    output_path = str(export_dir / f"experiments_summary.{extension}")

                # Export summary
                result_path = exporter.export_summary(
                    output_path=output_path,
                    format=request.format,
                    limit=request.limit,
                )

                # Count experiments in the summary
                query = ExperimentQuery(persistence.db._db_path)
                try:
                    total = query.count_experiments()
                    total = min(total, request.limit)
                finally:
                    query.close()

            finally:
                exporter.close()

            logger.info(
                f"Exported experiments summary to {request.format}: {result_path}"
            )

            return ExperimentSummaryExportResponse(
                output_path=str(result_path),
                total_experiments=total,
                format=request.format,
            )

        except ExportError as e:
            logger.error(f"Summary export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to export experiments summary: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e


    return app
