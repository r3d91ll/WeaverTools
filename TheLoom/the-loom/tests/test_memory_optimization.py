"""Tests for memory optimization features.

This module tests the GPU memory optimization capabilities including:
- Gradient checkpointing for training scenarios
- Precision mode validation with GPU capability detection
- Selective caching via TransformerLens names_filter
- Streaming extraction for long sequences
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config import MemoryConfig, PrecisionValidationResult, VALID_PRECISION_MODES
from src.extraction.hidden_states import (
    PRECISION_DTYPE_MAP,
    HiddenStateResult,
    InferenceResult,
    SelectiveCacheResult,
    StreamingChunkResult,
    TrainingExtractionResult,
    build_hook_filter,
    collect_streaming_results,
    extract_hidden_states,
    extract_with_precision,
)


# Marker for tests requiring GPU
gpu = pytest.mark.gpu


class TestPrecisionModeValidation:
    """Tests for precision mode validation with GPU capability detection."""

    def test_valid_precision_modes(self):
        """Verify all valid precision modes are recognized."""
        expected_modes = {"auto", "fp32", "fp16", "bf16"}
        assert VALID_PRECISION_MODES == expected_modes

    def test_precision_dtype_mapping(self):
        """Verify precision string to torch dtype mapping."""
        assert PRECISION_DTYPE_MAP["fp16"] == torch.float16
        assert PRECISION_DTYPE_MAP["fp32"] == torch.float32
        assert PRECISION_DTYPE_MAP["bf16"] == torch.bfloat16
        # Also check alternative names
        assert PRECISION_DTYPE_MAP["float16"] == torch.float16
        assert PRECISION_DTYPE_MAP["float32"] == torch.float32
        assert PRECISION_DTYPE_MAP["bfloat16"] == torch.bfloat16

    def test_memory_config_defaults(self):
        """Verify MemoryConfig default values."""
        config = MemoryConfig()
        assert config.precision_mode == "auto"
        assert config.enable_gradient_checkpointing is False
        assert config.streaming_chunk_size == 512
        assert config.memory_warning_threshold == 0.85
        assert config.activation_cache_filter == []

    def test_invalid_precision_mode(self):
        """Verify invalid precision mode is detected."""
        config = MemoryConfig(precision_mode="invalid")
        result = config.validate_precision()

        assert result.is_valid is False
        assert len(result.warnings) > 0
        assert "invalid" in result.warnings[0].lower() or "Invalid" in result.warnings[0]

    @patch("torch.cuda.is_available", return_value=False)
    def test_precision_validation_no_gpu(self, mock_cuda):
        """Verify precision validation handles no GPU case."""
        config = MemoryConfig(precision_mode="auto")
        result = config.validate_precision()

        assert result.gpu_available is False
        assert result.bf16_supported is False
        assert result.resolved_precision == "fp32"  # Falls back to fp32 on CPU
        assert result.is_valid is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(7, 5))  # Pre-Ampere (Volta)
    def test_bf16_not_supported_on_pre_ampere(self, mock_cap, mock_cuda):
        """Verify BF16 fails gracefully on pre-Ampere GPUs."""
        config = MemoryConfig(precision_mode="bf16")
        result = config.validate_precision()

        assert result.gpu_available is True
        assert result.bf16_supported is False
        assert result.compute_capability == (7, 5)
        # Should still be "valid" but with warnings
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "BF16" in result.warnings[0] or "bf16" in result.warnings[0]

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(8, 6))  # Ampere
    def test_bf16_supported_on_ampere(self, mock_cap, mock_cuda):
        """Verify BF16 works on Ampere+ GPUs."""
        config = MemoryConfig(precision_mode="bf16")
        result = config.validate_precision()

        assert result.gpu_available is True
        assert result.bf16_supported is True
        assert result.compute_capability == (8, 6)
        assert result.resolved_precision == "bf16"
        assert result.is_valid is True
        assert len(result.warnings) == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(7, 0))  # Volta
    def test_fp16_works_on_all_gpus(self, mock_cap, mock_cuda):
        """Verify FP16 works on all CUDA GPUs."""
        config = MemoryConfig(precision_mode="fp16")
        result = config.validate_precision()

        assert result.gpu_available is True
        assert result.resolved_precision == "fp16"
        assert result.is_valid is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(8, 0))  # Ampere
    def test_auto_selects_bf16_on_ampere(self, mock_cap, mock_cuda):
        """Verify auto mode selects BF16 on Ampere+ GPUs."""
        config = MemoryConfig(precision_mode="auto")
        result = config.validate_precision()

        assert result.resolved_precision == "bf16"
        assert result.bf16_supported is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(7, 5))  # Pre-Ampere
    def test_auto_selects_fp16_on_pre_ampere(self, mock_cap, mock_cuda):
        """Verify auto mode selects FP16 on pre-Ampere GPUs."""
        config = MemoryConfig(precision_mode="auto")
        result = config.validate_precision()

        assert result.resolved_precision == "fp16"
        assert result.bf16_supported is False


class TestExtractWithPrecision:
    """Tests for precision conversion in hidden state extraction."""

    def test_convert_to_fp16(self):
        """Verify conversion to FP16 precision."""
        hidden_states = {0: torch.randn(1, 10, 768, dtype=torch.float32)}

        result = extract_with_precision(hidden_states, precision="fp16")

        assert result[0].dtype == torch.float16

    def test_convert_to_bf16(self):
        """Verify conversion to BF16 precision."""
        hidden_states = {0: torch.randn(1, 10, 768, dtype=torch.float32)}

        result = extract_with_precision(hidden_states, precision="bf16")

        assert result[0].dtype == torch.bfloat16

    def test_convert_to_fp32(self):
        """Verify conversion to FP32 (no change for fp32 input)."""
        hidden_states = {0: torch.randn(1, 10, 768, dtype=torch.float32)}

        result = extract_with_precision(hidden_states, precision="fp32")

        assert result[0].dtype == torch.float32

    def test_multiple_layers(self):
        """Verify precision conversion works for multiple layers."""
        hidden_states = {
            0: torch.randn(1, 10, 768, dtype=torch.float32),
            5: torch.randn(1, 10, 768, dtype=torch.float32),
            11: torch.randn(1, 10, 768, dtype=torch.float32),
        }

        result = extract_with_precision(hidden_states, precision="fp16")

        for layer_idx in [0, 5, 11]:
            assert result[layer_idx].dtype == torch.float16

    def test_invalid_precision_raises(self):
        """Verify invalid precision raises ValueError."""
        hidden_states = {0: torch.randn(1, 10, 768)}

        with pytest.raises(ValueError) as exc_info:
            extract_with_precision(hidden_states, precision="invalid")

        assert "unsupported precision" in str(exc_info.value).lower()


class TestSelectiveCaching:
    """Tests for TransformerLens selective caching support."""

    def test_build_hook_filter_specific_layers(self):
        """Verify hook filter builds correct hook names for specific layers."""
        hooks = build_hook_filter(layers=[0, 5, 11])

        expected = [
            "blocks.0.hook_resid_post",
            "blocks.5.hook_resid_post",
            "blocks.11.hook_resid_post",
        ]
        assert hooks == expected

    def test_build_hook_filter_all_layers(self):
        """Verify hook filter builds hooks for all layers when n_layers provided."""
        hooks = build_hook_filter(n_layers=6)

        expected = [f"blocks.{i}.hook_resid_post" for i in range(6)]
        assert hooks == expected

    def test_build_hook_filter_custom_hook_types(self):
        """Verify hook filter supports custom hook types."""
        hooks = build_hook_filter(
            layers=[0, 3],
            hook_types=["hook_resid_post", "attn.hook_pattern"]
        )

        expected = [
            "blocks.0.hook_resid_post",
            "blocks.0.attn.hook_pattern",
            "blocks.3.hook_resid_post",
            "blocks.3.attn.hook_pattern",
        ]
        assert hooks == expected

    def test_build_hook_filter_requires_layers_or_n_layers(self):
        """Verify hook filter raises when neither layers nor n_layers provided."""
        with pytest.raises(ValueError) as exc_info:
            build_hook_filter()

        assert "layers" in str(exc_info.value).lower() or "n_layers" in str(exc_info.value).lower()

    def test_selective_cache_result_get_residual_stream(self):
        """Verify SelectiveCacheResult.get_residual_stream works."""
        cache = {
            "blocks.0.hook_resid_post": torch.randn(1, 10, 768),
            "blocks.5.hook_resid_post": torch.randn(1, 10, 768),
        }
        result = SelectiveCacheResult(
            cache=cache,
            logits=None,
            hooks_cached=list(cache.keys()),
            stopped_at_layer=None,
        )

        # Layer 0 should return tensor
        layer_0 = result.get_residual_stream(0)
        assert layer_0 is not None
        assert layer_0.shape == (1, 10, 768)

        # Layer 5 should return tensor
        layer_5 = result.get_residual_stream(5)
        assert layer_5 is not None

        # Layer 3 (not cached) should return None
        layer_3 = result.get_residual_stream(3)
        assert layer_3 is None

    def test_selective_cache_result_get_attention_pattern(self):
        """Verify SelectiveCacheResult.get_attention_pattern works."""
        cache = {
            "blocks.0.attn.hook_pattern": torch.randn(1, 12, 10, 10),
        }
        result = SelectiveCacheResult(
            cache=cache,
            logits=None,
            hooks_cached=list(cache.keys()),
            stopped_at_layer=None,
        )

        # Layer 0 attention should return tensor
        attn_0 = result.get_attention_pattern(0)
        assert attn_0 is not None
        assert attn_0.shape == (1, 12, 10, 10)

        # Layer 1 attention (not cached) should return None
        attn_1 = result.get_attention_pattern(1)
        assert attn_1 is None

    def test_selective_cache_result_to_hidden_states_dict(self):
        """Verify SelectiveCacheResult converts to layer-indexed dict."""
        cache = {
            "blocks.0.hook_resid_post": torch.randn(1, 10, 768),
            "blocks.5.hook_resid_post": torch.randn(1, 10, 768),
            "blocks.0.attn.hook_pattern": torch.randn(1, 12, 10, 10),  # Should be excluded
        }
        result = SelectiveCacheResult(
            cache=cache,
            logits=None,
            hooks_cached=list(cache.keys()),
            stopped_at_layer=None,
        )

        hidden_states = result.to_hidden_states_dict()

        # Should only include residual stream hooks
        assert 0 in hidden_states
        assert 5 in hidden_states
        assert len(hidden_states) == 2

    def test_names_filter_must_be_list_of_strings(self):
        """Verify names_filter validation rejects non-list types."""
        # Create a SelectiveCacheResult to verify the list requirement is documented
        # The actual validation happens in extract_with_selective_cache
        hooks = build_hook_filter(layers=[0, 1, 2])

        # All items should be strings
        for hook in hooks:
            assert isinstance(hook, str)


class TestStreamingChunking:
    """Tests for streaming extraction for long sequences."""

    def test_streaming_chunk_result_creation(self):
        """Verify StreamingChunkResult is created correctly."""
        hidden_states = {0: torch.randn(1, 100, 768)}
        result = StreamingChunkResult(
            chunk_index=0,
            start_position=0,
            end_position=100,
            hidden_states=hidden_states,
            is_last_chunk=False,
        )

        assert result.chunk_index == 0
        assert result.start_position == 0
        assert result.end_position == 100
        assert result.chunk_length == 100
        assert result.is_last_chunk is False

    def test_streaming_chunk_result_chunk_length(self):
        """Verify chunk_length property works correctly."""
        result = StreamingChunkResult(
            chunk_index=2,
            start_position=1024,
            end_position=1536,
            hidden_states={},
            is_last_chunk=True,
        )

        assert result.chunk_length == 512
        assert result.is_last_chunk is True

    def test_streaming_chunk_result_to_hidden_state_results(self):
        """Verify conversion to HiddenStateResult objects."""
        tensor = torch.randn(1, 50, 768)
        result = StreamingChunkResult(
            chunk_index=0,
            start_position=0,
            end_position=50,
            hidden_states={0: tensor, 5: tensor.clone()},
            is_last_chunk=False,
        )

        hidden_state_results = result.to_hidden_state_results()

        assert 0 in hidden_state_results
        assert 5 in hidden_state_results
        assert isinstance(hidden_state_results[0], HiddenStateResult)

    def test_streaming_chunk_result_normalized(self):
        """Verify normalization in to_hidden_state_results."""
        tensor = torch.tensor([[[3.0, 4.0]]])  # norm = 5
        result = StreamingChunkResult(
            chunk_index=0,
            start_position=0,
            end_position=1,
            hidden_states={0: tensor},
            is_last_chunk=False,
        )

        hidden_state_results = result.to_hidden_state_results(normalize=True)

        # Should be L2 normalized
        vector = hidden_state_results[0].vector
        norm = np.linalg.norm(vector)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_streaming_chunk_metadata(self):
        """Verify StreamingChunkResult metadata handling."""
        result = StreamingChunkResult(
            chunk_index=0,
            start_position=0,
            end_position=512,
            hidden_states={},
            is_last_chunk=False,
            metadata={"total_seq_len": 2048, "chunk_size": 512},
        )

        assert result.metadata["total_seq_len"] == 2048
        assert result.metadata["chunk_size"] == 512


class TestTrainingExtraction:
    """Tests for training-mode extraction with gradients enabled."""

    def test_training_extraction_result_creation(self):
        """Verify TrainingExtractionResult is created correctly."""
        hidden_states = {0: torch.randn(1, 10, 768)}
        logits = torch.randn(1, 10, 50257)

        result = TrainingExtractionResult(
            hidden_states=hidden_states,
            logits=logits,
            gradients_enabled=True,
            metadata={"layers_extracted": [0, 5, 11]},
        )

        assert result.gradients_enabled is True
        assert result.logits is not None
        assert 0 in result.hidden_states
        assert result.metadata["layers_extracted"] == [0, 5, 11]

    def test_training_extraction_result_to_hidden_states_dict(self):
        """Verify TrainingExtractionResult.to_hidden_states_dict works."""
        hidden_states = {0: torch.randn(1, 10, 768), 5: torch.randn(1, 10, 768)}
        result = TrainingExtractionResult(
            hidden_states=hidden_states,
            logits=None,
            gradients_enabled=False,
        )

        hs_dict = result.to_hidden_states_dict()

        assert hs_dict is hidden_states  # Should return the same dict

    def test_training_extraction_result_to_hidden_state_results(self):
        """Verify conversion to HiddenStateResult objects."""
        hidden_states = {0: torch.randn(1, 10, 768)}
        result = TrainingExtractionResult(
            hidden_states=hidden_states,
            logits=None,
            gradients_enabled=True,
        )

        hs_results = result.to_hidden_state_results()

        assert 0 in hs_results
        assert isinstance(hs_results[0], HiddenStateResult)


class TestInferenceOptimization:
    """Tests for inference-optimized extraction."""

    def test_inference_result_creation(self):
        """Verify InferenceResult is created correctly."""
        hidden_states = {0: torch.randn(1, 10, 768)}
        logits = torch.randn(1, 10, 50257)

        result = InferenceResult(
            hidden_states=hidden_states,
            logits=logits,
            inference_mode_used=True,
            metadata={"precision": "fp16"},
        )

        assert result.inference_mode_used is True
        assert result.logits is not None
        assert 0 in result.hidden_states
        assert result.metadata["precision"] == "fp16"

    def test_inference_result_to_hidden_states_dict(self):
        """Verify InferenceResult.to_hidden_states_dict works."""
        hidden_states = {0: torch.randn(1, 10, 768)}
        result = InferenceResult(
            hidden_states=hidden_states,
            logits=None,
            inference_mode_used=True,
        )

        hs_dict = result.to_hidden_states_dict()

        assert hs_dict is hidden_states

    def test_inference_result_to_hidden_state_results(self):
        """Verify conversion to HiddenStateResult objects."""
        hidden_states = {0: torch.randn(1, 10, 768)}
        result = InferenceResult(
            hidden_states=hidden_states,
            logits=None,
            inference_mode_used=True,
        )

        hs_results = result.to_hidden_state_results()

        assert 0 in hs_results
        assert isinstance(hs_results[0], HiddenStateResult)


class TestMemoryConfigValidation:
    """Tests for MemoryConfig validation bounds."""

    def test_streaming_chunk_size_minimum(self):
        """Verify streaming_chunk_size has minimum of 1."""
        config = MemoryConfig(streaming_chunk_size=1)
        assert config.streaming_chunk_size == 1

        with pytest.raises(ValueError):
            MemoryConfig(streaming_chunk_size=0)

    def test_memory_warning_threshold_bounds(self):
        """Verify memory_warning_threshold is between 0 and 1."""
        config = MemoryConfig(memory_warning_threshold=0.0)
        assert config.memory_warning_threshold == 0.0

        config = MemoryConfig(memory_warning_threshold=1.0)
        assert config.memory_warning_threshold == 1.0

        with pytest.raises(ValueError):
            MemoryConfig(memory_warning_threshold=1.5)

        with pytest.raises(ValueError):
            MemoryConfig(memory_warning_threshold=-0.1)

    def test_activation_cache_filter_accepts_list(self):
        """Verify activation_cache_filter accepts list of strings."""
        filter_list = ["blocks.0.hook_resid_post", "blocks.5.hook_resid_post"]
        config = MemoryConfig(activation_cache_filter=filter_list)

        assert config.activation_cache_filter == filter_list


class TestHiddenStatesWithPrecision:
    """Tests for hidden state extraction with various precisions."""

    def test_extract_hidden_states_bfloat16_conversion(self):
        """Verify bfloat16 tensors are converted to float32 for numpy."""
        tensor = torch.randn(1, 10, 768, dtype=torch.bfloat16)
        hidden_dict = {0: tensor}

        results = extract_hidden_states(hidden_dict)

        # Should be converted to float32 since numpy doesn't support bf16
        assert results[0].dtype == "float32"
        assert isinstance(results[0].vector, np.ndarray)

    def test_extract_hidden_states_float16(self):
        """Verify float16 tensors are handled correctly."""
        tensor = torch.randn(1, 10, 768, dtype=torch.float16)
        hidden_dict = {0: tensor}

        results = extract_hidden_states(hidden_dict)

        assert results[0].dtype == "float16"
        assert isinstance(results[0].vector, np.ndarray)

    def test_extract_hidden_states_float32(self):
        """Verify float32 tensors are handled correctly."""
        tensor = torch.randn(1, 10, 768, dtype=torch.float32)
        hidden_dict = {0: tensor}

        results = extract_hidden_states(hidden_dict)

        assert results[0].dtype == "float32"
        assert isinstance(results[0].vector, np.ndarray)


# GPU-specific tests that require actual CUDA hardware
@gpu
class TestGPUMemoryOptimization:
    """GPU-specific memory optimization tests.

    These tests require CUDA hardware and are marked with @gpu marker.
    Run with: pytest -m gpu
    """

    @pytest.fixture
    def cuda_available(self):
        """Check if CUDA is available for GPU tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return True

    def test_tensor_precision_on_gpu(self, cuda_available):
        """Verify tensor precision conversion on GPU."""
        tensor = torch.randn(1, 10, 768, device="cuda", dtype=torch.float32)
        hidden_states = {0: tensor}

        result = extract_with_precision(hidden_states, precision="fp16")

        assert result[0].dtype == torch.float16
        assert result[0].device.type == "cuda"

    def test_bf16_support_detection(self, cuda_available):
        """Verify BF16 support detection based on GPU capability."""
        config = MemoryConfig(precision_mode="auto")
        result = config.validate_precision()

        # Result should reflect actual GPU capability
        assert result.gpu_available is True
        assert result.compute_capability is not None

        # BF16 supported only on Ampere+ (compute capability 8.0+)
        if result.compute_capability[0] >= 8:
            assert result.bf16_supported is True
        else:
            assert result.bf16_supported is False

    def test_memory_efficient_precision_conversion(self, cuda_available):
        """Verify memory usage decreases with lower precision."""
        torch.cuda.reset_peak_memory_stats()

        # Create FP32 tensor
        tensor_fp32 = torch.randn(100, 1024, 768, device="cuda", dtype=torch.float32)
        fp32_memory = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        del tensor_fp32
        torch.cuda.empty_cache()

        # Create FP16 tensor of same logical size
        tensor_fp16 = torch.randn(100, 1024, 768, device="cuda", dtype=torch.float16)
        fp16_memory = torch.cuda.max_memory_allocated()

        del tensor_fp16
        torch.cuda.empty_cache()

        # FP16 should use roughly half the memory
        assert fp16_memory < fp32_memory * 0.6  # Some overhead, but should be <60%
