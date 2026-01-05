"""Tests for TNT Olympian loader and memory analysis.

This module tests:
1. TNTOlympianLoader - checkpoint loading and inference
2. TNT Memory Analysis - D_eff, beta, weight norms for Conveyance Hypothesis
3. ShakespeareBPETokenizer - tokenizer wrapper

Markers:
    - synthetic: Tests using synthetic data (no real checkpoints)
    - integration: Tests requiring actual model weights
    - slow: Tests with longer execution time
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.loaders.base import EmbeddingOutput, GenerationOutput, LoadedModel
from src.loaders.tnt_olympian_loader import (
    MODEL_CONFIG_KEYS,
    REQUIRED_CHECKPOINT_KEYS,
    TNTOlympianLoader,
)


# ============================================================================
# TNTOlympianLoader Tests
# ============================================================================


class TestTNTOlympianLoaderBasics:
    """Tests for TNTOlympianLoader basic properties."""

    def test_name(self) -> None:
        loader = TNTOlympianLoader()
        assert loader.name == "tnt_olympian"

    def test_can_load_tnt_olympian_paths(self) -> None:
        loader = TNTOlympianLoader()

        # Should handle TNT Olympian-specific patterns
        assert loader.can_load("/home/user/models/Atlas/tnt_olympian.pt")
        assert loader.can_load("/path/to/olympian/model.pt")
        assert loader.can_load("tnt_model.pt")
        assert loader.can_load("/runs/tnt_olympian_overnight/stage2_final.pt")
        assert loader.can_load("/models/tnt-experiment/checkpoint.pt")

    def test_can_load_excludes_huggingface_models(self) -> None:
        loader = TNTOlympianLoader()

        # Should NOT handle HuggingFace model IDs
        assert not loader.can_load("meta-llama/Llama-3.1-8B-Instruct")
        assert not loader.can_load("mistralai/Mistral-7B-Instruct-v0.2")
        assert not loader.can_load("sentence-transformers/all-MiniLM-L6-v2")

    def test_can_load_excludes_non_pt_files(self) -> None:
        loader = TNTOlympianLoader()

        # Should NOT handle non-.pt files
        assert not loader.can_load("/path/to/olympian/model.safetensors")
        assert not loader.can_load("olympian_model.pth")
        assert not loader.can_load("tnt_model.bin")

    def test_can_load_excludes_random_paths(self) -> None:
        loader = TNTOlympianLoader()

        # Should NOT handle paths without tnt/olympian in name
        assert not loader.can_load("random-model-id")
        assert not loader.can_load("/home/user/models/llama.pt")


class TestTNTOlympianConfigFiltering:
    """Tests for checkpoint config filtering."""

    def test_filter_model_config_keeps_model_keys(self) -> None:
        loader = TNTOlympianLoader()

        full_config = {
            # Model keys (should keep)
            "vocab_size": 16384,
            "d_model": 512,
            "n_layers": 4,
            "n_heads": 8,
            "d_ff": 2048,
            "window_size": 64,
            "dropout": 0.1,
            "max_seq_len": 512,
            "global_chunk_size": 256,
            "local_chunk_sizes": [8, 16, 32, 64],
            # Training keys (should remove)
            "learning_rate": 0.001,
            "batch_size": 32,
            "warmup_steps": 1000,
            "total_steps": 100000,
            "optimizer": "adamw",
        }

        filtered = loader._filter_model_config(full_config)

        # Should keep model keys
        assert filtered["vocab_size"] == 16384
        assert filtered["d_model"] == 512
        assert filtered["n_layers"] == 4

        # Should remove training keys
        assert "learning_rate" not in filtered
        assert "batch_size" not in filtered
        assert "optimizer" not in filtered

    def test_filter_model_config_empty(self) -> None:
        loader = TNTOlympianLoader()

        filtered = loader._filter_model_config({})
        assert filtered == {}


class TestTNTOlympianCheckpointValidation:
    """Tests for checkpoint validation."""

    @pytest.fixture
    def valid_checkpoint(self) -> dict[str, Any]:
        """Create a valid TNT Olympian checkpoint dict."""
        return {
            "model_state_dict": {
                "blocks.0.weight": torch.randn(512, 512),
                "blocks.0.bias": torch.randn(512),
            },
            "global_step": 10000,
            "current_stage": 2,
            "config": {
                "vocab_size": 16384,
                "d_model": 512,
                "n_layers": 4,
                "n_heads": 8,
                "d_ff": 2048,
            },
        }

    @pytest.fixture
    def checkpoint_with_memory(self, valid_checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Create checkpoint with memory state."""
        checkpoint = valid_checkpoint.copy()
        checkpoint["memory_state"] = {
            "block_0": {
                "global": {
                    "W1": torch.randn(512, 512),
                    "W2": torch.randn(512, 512),
                    "k_buffer": torch.randn(1, 256, 64),
                    "v_buffer": torch.randn(1, 256, 64),
                    "buffer_len": 100,
                },
                "local": [
                    {
                        "W1": torch.randn(512, 512),
                        "W2": torch.randn(512, 512),
                        "k_buffer": torch.randn(1, 64, 64),
                        "v_buffer": torch.randn(1, 64, 64),
                        "buffer_len": 50,
                    }
                    for _ in range(4)
                ],
                "position": 1000,
            },
        }
        return checkpoint

    @pytest.mark.synthetic
    def test_validate_valid_checkpoint(self, valid_checkpoint: dict[str, Any]) -> None:
        loader = TNTOlympianLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(valid_checkpoint, path)

            result = loader._validate_checkpoint(valid_checkpoint, str(path))

            assert result["valid"] is True
            assert result["global_step"] == 10000
            assert result["current_stage"] == 2
            assert result["has_memory_state"] is False

    @pytest.mark.synthetic
    def test_validate_checkpoint_with_memory(
        self, checkpoint_with_memory: dict[str, Any]
    ) -> None:
        loader = TNTOlympianLoader()

        result = loader._validate_checkpoint(checkpoint_with_memory, "test.pt")

        assert result["valid"] is True
        assert result["has_memory_state"] is True

    @pytest.mark.synthetic
    def test_validate_missing_required_keys(self) -> None:
        loader = TNTOlympianLoader()

        incomplete_checkpoint = {
            "global_step": 1000,
            # Missing: model_state_dict, config
        }

        with pytest.raises(ValueError) as excinfo:
            loader._validate_checkpoint(incomplete_checkpoint, "test.pt")

        assert "missing required keys" in str(excinfo.value).lower()

    @pytest.mark.synthetic
    def test_validate_missing_model_params(self) -> None:
        loader = TNTOlympianLoader()

        checkpoint = {
            "model_state_dict": {},
            "global_step": 1000,
            "config": {},  # Missing d_model, vocab_size
        }

        with pytest.raises(ValueError) as excinfo:
            loader._validate_checkpoint(checkpoint, "test.pt")

        assert "missing model parameters" in str(excinfo.value).lower()


class TestTNTOlympianStateDictCleaning:
    """Tests for state dict cleaning (removing DDP/compile prefixes)."""

    def test_clean_ddp_prefix(self) -> None:
        loader = TNTOlympianLoader()

        state_dict = {
            "module.blocks.0.weight": torch.randn(512, 512),
            "module.blocks.0.bias": torch.randn(512),
        }

        cleaned = loader._clean_state_dict(state_dict)

        assert "blocks.0.weight" in cleaned
        assert "blocks.0.bias" in cleaned
        assert "module.blocks.0.weight" not in cleaned

    def test_clean_compile_prefix(self) -> None:
        loader = TNTOlympianLoader()

        state_dict = {
            "_orig_mod.blocks.0.weight": torch.randn(512, 512),
            "_orig_mod.blocks.0.bias": torch.randn(512),
        }

        cleaned = loader._clean_state_dict(state_dict)

        assert "blocks.0.weight" in cleaned
        assert "_orig_mod.blocks.0.weight" not in cleaned

    def test_clean_both_prefixes(self) -> None:
        loader = TNTOlympianLoader()

        # Combined prefixes (DDP wrapping a compiled model)
        state_dict = {
            "module._orig_mod.blocks.0.weight": torch.randn(512, 512),
        }

        cleaned = loader._clean_state_dict(state_dict)

        assert "blocks.0.weight" in cleaned
        assert "module._orig_mod.blocks.0.weight" not in cleaned

    def test_clean_no_prefix(self) -> None:
        loader = TNTOlympianLoader()

        state_dict = {
            "blocks.0.weight": torch.randn(512, 512),
            "blocks.0.bias": torch.randn(512),
        }

        cleaned = loader._clean_state_dict(state_dict)

        assert "blocks.0.weight" in cleaned
        assert "blocks.0.bias" in cleaned


class TestTNTOlympianLoaderIntegration:
    """Integration tests for TNTOlympianLoader with mocked models."""

    @pytest.fixture
    def mock_tnt_loaded_model(self) -> LoadedModel:
        """Create a LoadedModel stub for TNT Olympian tests."""
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.get_memory_state.return_value = {}

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer.vocab_size = 16384

        return LoadedModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            model_id="test-tnt-olympian",
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=512,
            num_layers=4,
            loader_type="tnt_olympian",
            metadata={
                "global_step": 10000,
                "current_stage": 2,
                "memory_state": {},
            },
        )

    @pytest.mark.synthetic
    def test_generate_returns_correct_output_type(
        self, mock_tnt_loaded_model: LoadedModel
    ) -> None:
        loader = TNTOlympianLoader()

        # Mock the model's generate method
        mock_tnt_loaded_model.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tnt_loaded_model.model._get_embeddings.return_value = torch.randn(1, 5, 512)

        # Mock blocks for hidden state extraction
        mock_block = MagicMock()
        mock_block.return_value = (torch.randn(1, 5, 512), None)
        mock_tnt_loaded_model.model.blocks = [mock_block] * 4

        result = loader.generate(
            mock_tnt_loaded_model,
            "Test prompt",
            max_tokens=10,
            temperature=0.7,
        )

        assert isinstance(result, GenerationOutput)
        assert isinstance(result.text, str)
        assert isinstance(result.token_ids, list)
        assert "memory_before" in result.metadata
        assert "memory_after" in result.metadata

    @pytest.mark.synthetic
    def test_embed_returns_correct_output_type(
        self, mock_tnt_loaded_model: LoadedModel
    ) -> None:
        loader = TNTOlympianLoader()

        # Mock model forward pass
        mock_tnt_loaded_model.model._get_embeddings.return_value = torch.randn(1, 5, 512)

        mock_block = MagicMock()
        mock_block.return_value = (torch.randn(1, 5, 512), None)
        mock_tnt_loaded_model.model.blocks = [mock_block] * 4

        result = loader.embed(
            mock_tnt_loaded_model,
            "Test text",
            pooling="last_token",
        )

        assert isinstance(result, EmbeddingOutput)
        assert isinstance(result.embedding, torch.Tensor)
        assert result.metadata["pooling"] == "last_token"

    @pytest.mark.synthetic
    def test_embed_pooling_strategies(
        self, mock_tnt_loaded_model: LoadedModel
    ) -> None:
        loader = TNTOlympianLoader()

        # Mock model forward pass
        mock_tnt_loaded_model.model._get_embeddings.return_value = torch.randn(1, 5, 512)

        mock_block = MagicMock()
        mock_block.return_value = (torch.randn(1, 5, 512), None)
        mock_tnt_loaded_model.model.blocks = [mock_block] * 4

        for pooling in ["last_token", "mean", "first"]:
            result = loader.embed(
                mock_tnt_loaded_model,
                "Test text",
                pooling=pooling,
            )
            assert result.metadata["pooling"] == pooling

    @pytest.mark.synthetic
    def test_embed_invalid_pooling_raises(
        self, mock_tnt_loaded_model: LoadedModel
    ) -> None:
        loader = TNTOlympianLoader()

        # Mock model forward pass
        mock_tnt_loaded_model.model._get_embeddings.return_value = torch.randn(1, 5, 512)

        mock_block = MagicMock()
        mock_block.return_value = (torch.randn(1, 5, 512), None)
        mock_tnt_loaded_model.model.blocks = [mock_block] * 4

        with pytest.raises(ValueError) as excinfo:
            loader.embed(
                mock_tnt_loaded_model,
                "Test text",
                pooling="invalid_pooling",
            )

        assert "unknown pooling" in str(excinfo.value).lower()


class TestTNTOlympianMemoryOperations:
    """Tests for memory extraction and reset operations."""

    @pytest.fixture
    def mock_model_with_memory(self) -> MagicMock:
        """Create a mock model with memory methods."""
        mock_model = MagicMock()
        mock_model.get_memory_state.return_value = {
            "block_0": {
                "global": {"W1": torch.randn(512, 512)},
                "local": [{"W1": torch.randn(256, 256)} for _ in range(4)],
            },
        }
        return mock_model

    def test_extract_memory_state(self, mock_model_with_memory: MagicMock) -> None:
        loader = TNTOlympianLoader()

        loaded_model = LoadedModel(
            model=mock_model_with_memory,
            tokenizer=MagicMock(),
            model_id="test-model",
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=512,
            num_layers=4,
            loader_type="tnt_olympian",
            metadata={},
        )

        state = loader.extract_memory_state(loaded_model)

        assert "block_0" in state
        mock_model_with_memory.get_memory_state.assert_called_once()

    def test_reset_memory_all(self, mock_model_with_memory: MagicMock) -> None:
        loader = TNTOlympianLoader()

        loaded_model = LoadedModel(
            model=mock_model_with_memory,
            tokenizer=MagicMock(),
            model_id="test-model",
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=512,
            num_layers=4,
            loader_type="tnt_olympian",
            metadata={},
        )

        loader.reset_memory(loaded_model, reset_type="all")

        mock_model_with_memory.reset_all_memories.assert_called_once()

    def test_reset_memory_local(self, mock_model_with_memory: MagicMock) -> None:
        loader = TNTOlympianLoader()

        loaded_model = LoadedModel(
            model=mock_model_with_memory,
            tokenizer=MagicMock(),
            model_id="test-model",
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=512,
            num_layers=4,
            loader_type="tnt_olympian",
            metadata={},
        )

        loader.reset_memory(loaded_model, reset_type="local")

        mock_model_with_memory.reset_local_memories.assert_called_once()

    def test_reset_memory_invalid_type_raises(
        self, mock_model_with_memory: MagicMock
    ) -> None:
        loader = TNTOlympianLoader()

        loaded_model = LoadedModel(
            model=mock_model_with_memory,
            tokenizer=MagicMock(),
            model_id="test-model",
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=512,
            num_layers=4,
            loader_type="tnt_olympian",
            metadata={},
        )

        with pytest.raises(ValueError) as excinfo:
            loader.reset_memory(loaded_model, reset_type="invalid")

        assert "unknown reset_type" in str(excinfo.value).lower()


class TestTNTOlympianRequiredKeys:
    """Tests for required key definitions."""

    def test_required_checkpoint_keys_defined(self) -> None:
        assert "model_state_dict" in REQUIRED_CHECKPOINT_KEYS
        assert "global_step" in REQUIRED_CHECKPOINT_KEYS
        assert "config" in REQUIRED_CHECKPOINT_KEYS

    def test_model_config_keys_defined(self) -> None:
        assert "vocab_size" in MODEL_CONFIG_KEYS
        assert "d_model" in MODEL_CONFIG_KEYS
        assert "n_layers" in MODEL_CONFIG_KEYS
        assert "n_heads" in MODEL_CONFIG_KEYS
        assert "global_chunk_size" in MODEL_CONFIG_KEYS
        assert "local_chunk_sizes" in MODEL_CONFIG_KEYS


# ============================================================================
# TNT Memory Analysis Tests
# ============================================================================


class TestMemoryAnalysisMetrics:
    """Tests for D_eff, beta, and other memory metrics."""

    def test_compute_d_eff_healthy_weights(self) -> None:
        """Test D_eff on weights with good dimensional spread."""
        from src.analysis.tnt_memory_analysis import compute_d_eff

        # Create weight matrix with full rank (healthy)
        weight = torch.randn(512, 512)

        d_eff = compute_d_eff(weight, variance_threshold=0.90)

        # For random normal, D_eff should be high
        assert d_eff > 100  # Well-spread weights should have high D_eff

    def test_compute_d_eff_collapsed_weights(self) -> None:
        """Test D_eff on collapsed/rank-deficient weights."""
        from src.analysis.tnt_memory_analysis import compute_d_eff

        # Create low-rank weight matrix (collapsed)
        u = torch.randn(512, 10)
        v = torch.randn(10, 512)
        weight = u @ v  # Rank 10 matrix

        d_eff = compute_d_eff(weight, variance_threshold=0.90)

        # Low-rank matrix should have low D_eff
        assert d_eff <= 15  # At most slightly above the true rank

    def test_compute_d_eff_identity_like(self) -> None:
        """Test D_eff on identity-like matrix."""
        from src.analysis.tnt_memory_analysis import compute_d_eff

        weight = torch.eye(256)

        d_eff = compute_d_eff(weight, variance_threshold=0.90)

        # Identity should have D_eff close to 256
        assert d_eff >= 200

    def test_compute_beta_healthy_weights(self) -> None:
        """Test beta on healthy weights (should be low)."""
        from src.analysis.tnt_memory_analysis import compute_beta

        # Random weights should have relatively uniform singular values
        weight = torch.randn(256, 256)

        beta = compute_beta(weight)

        # For random normal, beta should be moderate (not too high)
        assert beta < 5.0

    def test_compute_beta_collapsed_weights(self) -> None:
        """Test beta on collapsed weights (should be high)."""
        from src.analysis.tnt_memory_analysis import compute_beta

        # Create weight dominated by single direction
        u = torch.randn(256, 1)
        v = torch.randn(1, 256)
        weight = 10 * (u @ v) + 0.01 * torch.randn(256, 256)  # Rank-1 dominant

        beta = compute_beta(weight)

        # Collapsed weights should have high beta
        assert beta > 10.0

    def test_compute_beta_identity(self) -> None:
        """Test beta on identity matrix (should be ~1)."""
        from src.analysis.tnt_memory_analysis import compute_beta

        weight = torch.eye(128)

        beta = compute_beta(weight)

        # Identity has equal singular values, so beta ≈ 1
        assert 0.9 < beta < 1.1


class TestLocalMemoryAnalysis:
    """Tests for analyzing LocalMemoryModule state."""

    @pytest.fixture
    def mock_local_memory_state(self) -> dict[str, Any]:
        """Create mock LocalMemoryModule state."""
        return {
            "W1": torch.randn(512, 512),
            "W2": torch.randn(512, 512),
            "W_init_W1": torch.randn(512, 512),
            "W_init_W2": torch.randn(512, 512),
            "k_buffer": torch.randn(1, 64, 64),
            "v_buffer": torch.randn(1, 64, 64),
            "buffer_idx": 50,
            "buffer_len": 50,
            "momentum_W1": torch.randn(512, 512),
            "momentum_W2": torch.randn(512, 512),
        }

    def test_analyze_local_memory(
        self, mock_local_memory_state: dict[str, Any]
    ) -> None:
        from src.analysis.tnt_memory_analysis import (
            LocalMemoryMetrics,
            analyze_local_memory,
        )

        # Set chunk_size and context_size for the analysis
        metrics = analyze_local_memory(
            mock_local_memory_state,
            chunk_size=64,
            context_size=64,
        )

        assert isinstance(metrics, LocalMemoryMetrics)
        assert metrics.chunk_size == 64
        assert metrics.weight_norm_w1 > 0
        assert metrics.weight_norm_w2 > 0
        assert metrics.d_eff > 0
        assert metrics.beta > 0
        assert 0 <= metrics.buffer_utilization <= 1

    def test_analyze_local_memory_buffer_utilization(self) -> None:
        from src.analysis.tnt_memory_analysis import analyze_local_memory

        state = {
            "W1": torch.randn(256, 256),
            "W2": torch.randn(256, 256),
            "k_buffer": torch.randn(1, 100, 64),
            "v_buffer": torch.randn(1, 100, 64),
            "buffer_len": 75,  # 75% full
        }

        metrics = analyze_local_memory(state, chunk_size=32, context_size=100)

        assert metrics.buffer_utilization == pytest.approx(0.75, rel=0.01)


class TestBlockMemoryAnalysis:
    """Tests for analyzing TNTHierarchicalMemory block state."""

    @pytest.fixture
    def mock_block_state(self) -> dict[str, Any]:
        """Create mock block memory state."""

        def make_local_state() -> dict[str, Any]:
            return {
                "W1": torch.randn(256, 256),
                "W2": torch.randn(256, 256),
                "k_buffer": torch.randn(1, 64, 64),
                "v_buffer": torch.randn(1, 64, 64),
                "buffer_len": 32,
            }

        return {
            "global": make_local_state(),
            "local": [make_local_state() for _ in range(4)],
            "position": 1000,
        }

    def test_analyze_block_memory(self, mock_block_state: dict[str, Any]) -> None:
        from src.analysis.tnt_memory_analysis import (
            BlockMemoryMetrics,
            analyze_block_memory,
        )

        metrics = analyze_block_memory(
            mock_block_state,
            block_idx=0,
            local_chunk_sizes=[8, 16, 32, 64],
            context_size=64,
        )

        assert isinstance(metrics, BlockMemoryMetrics)
        assert metrics.position == 1000
        assert metrics.global_memory is not None
        assert len(metrics.local_memories) == 4


class TestFullMemoryAnalysis:
    """Tests for full TNT memory state analysis."""

    @pytest.fixture
    def mock_full_memory_state(self) -> dict[str, Any]:
        """Create mock full memory state."""

        def make_local_state() -> dict[str, Any]:
            return {
                "W1": torch.randn(256, 256),
                "W2": torch.randn(256, 256),
                "k_buffer": torch.randn(1, 64, 64),
                "v_buffer": torch.randn(1, 64, 64),
                "buffer_len": 32,
            }

        def make_block_state() -> dict[str, Any]:
            return {
                "global": make_local_state(),
                "local": [make_local_state() for _ in range(4)],
                "position": 1000,
            }

        return {
            "block_0": make_block_state(),
            "block_1": make_block_state(),
            "block_2": make_block_state(),
            "block_3": make_block_state(),
        }

    def test_analyze_tnt_memory(self, mock_full_memory_state: dict[str, Any]) -> None:
        from src.analysis.tnt_memory_analysis import TNTMemoryAnalysis, analyze_tnt_memory

        analysis = analyze_tnt_memory(
            mock_full_memory_state,
            config={
                "local_chunk_sizes": [8, 16, 32, 64],
                "context_size": 64,
            },
        )

        assert isinstance(analysis, TNTMemoryAnalysis)
        assert len(analysis.blocks) == 4
        assert analysis.aggregate_d_eff > 0
        assert analysis.aggregate_beta > 0
        assert analysis.total_weight_norm > 0


class TestMemoryStateComparison:
    """Tests for comparing memory states."""

    @pytest.fixture
    def create_memory_state(self) -> Any:
        """Factory for creating memory states with different characteristics."""

        def _create(scale: float = 1.0) -> dict[str, Any]:
            def make_local_state() -> dict[str, Any]:
                return {
                    "W1": scale * torch.randn(256, 256),
                    "W2": scale * torch.randn(256, 256),
                    "k_buffer": torch.randn(1, 64, 64),
                    "v_buffer": torch.randn(1, 64, 64),
                    "buffer_len": 32,
                }

            return {
                "block_0": {
                    "global": make_local_state(),
                    "local": [make_local_state() for _ in range(4)],
                    "position": 1000,
                },
            }

        return _create

    def test_compare_memory_states(self, create_memory_state: Any) -> None:
        from src.analysis.tnt_memory_analysis import compare_memory_states

        before = create_memory_state(scale=1.0)
        after = create_memory_state(scale=1.5)

        delta = compare_memory_states(
            before,
            after,
            config={
                "local_chunk_sizes": [8, 16, 32, 64],
                "context_size": 64,
            },
        )

        assert "before" in delta
        assert "after" in delta
        assert "d_eff_delta" in delta
        assert "beta_delta" in delta
        assert "weight_norm_delta" in delta


# ============================================================================
# Shakespeare BPE Tokenizer Tests
# ============================================================================


class TestShakespeareBPETokenizer:
    """Tests for Shakespeare BPE tokenizer wrapper."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create mock tokenizer."""
        mock = MagicMock()
        mock.get_vocab_size.return_value = 16384
        mock.token_to_id.side_effect = lambda x: {
            "[PAD]": 0,
            "[EOS]": 1,
            "[BOS]": 2,
            "[UNK]": 3,
        }.get(x, None)

        mock_encoding = MagicMock()
        mock_encoding.ids = [2, 100, 200, 300, 1]  # BOS, tokens, EOS
        mock.encode.return_value = mock_encoding
        mock.decode.return_value = "To be or not"

        return mock

    def test_tokenizer_encode_decode_mock(self, mock_tokenizer: MagicMock) -> None:
        """Test tokenizer using mocks."""
        # Test encode
        encoding = mock_tokenizer.encode("To be or not")
        assert isinstance(encoding.ids, list)
        assert len(encoding.ids) == 5

        # Test decode
        decoded = mock_tokenizer.decode([100, 200, 300])
        assert decoded == "To be or not"

        # Check vocabulary size
        assert mock_tokenizer.get_vocab_size() == 16384

    @pytest.mark.integration
    def test_tokenizer_real_file(self) -> None:
        """Test with real tokenizer file if available."""
        from src.loaders.tnt_olympian_tokenizer import (
            DEFAULT_TOKENIZER_PATH,
            ShakespeareBPETokenizer,
        )

        if not (DEFAULT_TOKENIZER_PATH / "tokenizer.json").exists():
            pytest.skip("Real tokenizer file not available")

        tokenizer = ShakespeareBPETokenizer()

        # Test basic encode/decode
        text = "To be, or not to be"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)

        assert len(ids) > 0
        assert isinstance(decoded, str)

        # Vocabulary should be ~16k
        assert 10000 < tokenizer.vocab_size < 20000


# ============================================================================
# Registry Integration Tests
# ============================================================================


class TestTNTOlympianRegistryIntegration:
    """Tests for TNT Olympian loader in the registry."""

    def test_registry_has_tnt_olympian_loader(self) -> None:
        from src.loaders.registry import LoaderRegistry

        registry = LoaderRegistry()

        assert "tnt_olympian" in registry.loaders
        assert isinstance(registry.loaders["tnt_olympian"], TNTOlympianLoader)

    def test_registry_fallback_order_includes_tnt_olympian(self) -> None:
        from src.loaders.registry import LoaderRegistry

        registry = LoaderRegistry()

        assert "tnt_olympian" in registry.fallback_order
        # Should be first in the order (highest priority for .pt files)
        assert registry.fallback_order[0] == "tnt_olympian"

    def test_registry_detects_tnt_olympian_pattern(self) -> None:
        from src.loaders.registry import LoaderRegistry

        registry = LoaderRegistry()

        # Should auto-detect TNT Olympian patterns
        name, loader = registry.get_loader("/path/to/tnt_olympian_model.pt")
        assert name == "tnt_olympian"
        assert isinstance(loader, TNTOlympianLoader)

        name, loader = registry.get_loader("/models/olympian/checkpoint.pt")
        assert name == "tnt_olympian"

    def test_registry_fallback_for_standard_models(self) -> None:
        from src.loaders.registry import LoaderRegistry

        registry = LoaderRegistry()

        # Standard HuggingFace model should NOT match TNT Olympian
        name, loader = registry.get_loader("meta-llama/Llama-3.1-8B")
        assert name != "tnt_olympian"
        assert name == "transformers"

    def test_probe_tnt_olympian_model(self) -> None:
        from src.loaders.registry import LoaderRegistry

        registry = LoaderRegistry()

        result = registry.probe_model("/home/user/olympian_model.pt")

        assert result["detected_loader"] == "tnt_olympian"
        assert result["selected_loader"] == "tnt_olympian"
        assert result["can_load"]["tnt_olympian"] is True
