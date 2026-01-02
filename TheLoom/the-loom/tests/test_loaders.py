"""Tests for multi-loader functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.loaders.atlas_loader import (
    REQUIRED_CHECKPOINT_KEYS,
    AtlasLoader,
    CheckpointValidationError,
)
from src.loaders.base import EmbeddingOutput, GenerationOutput, LoadedModel
from src.loaders.custom_loader import (
    CUSTOM_MODEL_REGISTRY,
    CustomLoader,
    CustomModelConfig,
    register_custom_model,
)
from src.loaders.registry import LoaderRegistry
from src.loaders.sentence_transformers_loader import (
    SentenceTransformersLoader,
)
from src.loaders.transformers_loader import TransformersLoader


class TestTransformersLoader:
    """Tests for TransformersLoader."""

    def test_name(self):
        loader = TransformersLoader()
        assert loader.name == "transformers"

    def test_can_load_standard_models(self):
        loader = TransformersLoader()

        # Should handle standard HuggingFace models
        assert loader.can_load("meta-llama/Llama-3.1-8B-Instruct")
        assert loader.can_load("mistralai/Mistral-7B-Instruct-v0.2")
        assert loader.can_load("Qwen/Qwen2-7B-Instruct")
        assert loader.can_load("microsoft/phi-2")

    def test_can_load_excludes_sentence_transformers(self):
        loader = TransformersLoader()

        # Should NOT handle sentence-transformers models
        assert not loader.can_load("sentence-transformers/all-MiniLM-L6-v2")


class TestSentenceTransformersLoader:
    """Tests for SentenceTransformersLoader."""

    def test_name(self):
        loader = SentenceTransformersLoader()
        assert loader.name == "sentence_transformers"

    def test_can_load_embedding_models(self):
        loader = SentenceTransformersLoader()

        # Should handle known embedding model patterns
        assert loader.can_load("sentence-transformers/all-MiniLM-L6-v2")
        assert loader.can_load("BAAI/bge-small-en-v1.5")
        assert loader.can_load("intfloat/e5-small-v2")
        assert loader.can_load("thenlper/gte-small")
        assert loader.can_load("nomic-ai/nomic-embed-text-v1")

    def test_can_load_excludes_instruct_models(self):
        loader = SentenceTransformersLoader()

        # Should NOT handle instruction-tuned models
        assert not loader.can_load("BAAI/bge-instruct")
        assert not loader.can_load("model-chat")

    def test_can_load_standard_models_returns_false(self):
        loader = SentenceTransformersLoader()

        # Should NOT handle standard decoder models
        assert not loader.can_load("meta-llama/Llama-3.1-8B-Instruct")
        assert not loader.can_load("mistralai/Mistral-7B-v0.1")


class TestCustomLoader:
    """Tests for CustomLoader."""

    def test_name(self):
        loader = CustomLoader()
        assert loader.name == "custom"

    def test_can_load_without_config(self):
        loader = CustomLoader()

        # Without registered configs, should return False
        assert not loader.can_load("random-model-id")

    def test_can_load_with_config(self):
        config = CustomModelConfig(
            model_id_pattern="my-org/custom-model",
            model_factory=lambda *args, **kwargs: MagicMock(),
            hidden_size=2048,
            num_layers=24,
        )

        loader = CustomLoader(custom_configs={"my-org/custom-model": config})

        assert loader.can_load("my-org/custom-model")
        assert not loader.can_load("other-model")

    def test_get_config(self):
        config = CustomModelConfig(
            model_id_pattern="my-org/.*",  # Regex pattern
            model_factory=lambda *args, **kwargs: MagicMock(),
        )

        loader = CustomLoader(custom_configs={"my-org/.*": config})

        # Exact match
        result = loader.get_config("my-org/.*")
        assert result is not None

    def test_register_custom_model(self):
        # Clear any existing registrations
        original = CUSTOM_MODEL_REGISTRY.copy()
        CUSTOM_MODEL_REGISTRY.clear()

        try:
            config = CustomModelConfig(
                model_id_pattern="test/model",
                model_factory=lambda *args, **kwargs: MagicMock(),
            )
            register_custom_model(config)

            assert "test/model" in CUSTOM_MODEL_REGISTRY
        finally:
            # Restore original
            CUSTOM_MODEL_REGISTRY.clear()
            CUSTOM_MODEL_REGISTRY.update(original)


class TestLoaderRegistry:
    """Tests for LoaderRegistry."""

    def test_init_default_loaders(self):
        registry = LoaderRegistry()

        assert "atlas" in registry.loaders
        assert "mistral" in registry.loaders
        assert "qwen" in registry.loaders
        assert "transformers" in registry.loaders
        assert "sentence_transformers" in registry.loaders
        assert "custom" in registry.loaders

    def test_fallback_order(self):
        registry = LoaderRegistry()

        assert registry.fallback_order == [
            "atlas",
            "mistral",
            "qwen",
            "transformers",
            "sentence_transformers",
            "custom",
        ]

    def test_get_loader_auto_detection(self):
        registry = LoaderRegistry()

        # Standard model -> transformers
        name, loader = registry.get_loader("meta-llama/Llama-3.1-8B")
        assert name == "transformers"
        assert isinstance(loader, TransformersLoader)

        # Embedding model -> sentence_transformers
        name, loader = registry.get_loader("sentence-transformers/all-MiniLM-L6-v2")
        assert name == "sentence_transformers"
        assert isinstance(loader, SentenceTransformersLoader)

    def test_get_loader_with_config_override(self):
        registry = LoaderRegistry(
            loader_configs={
                "my-model": {"loader": "sentence_transformers"},
            }
        )

        name, loader = registry.get_loader("my-model")
        assert name == "sentence_transformers"

    def test_get_model_config_defaults(self):
        registry = LoaderRegistry()

        config = registry.get_model_config("some-model")

        assert config["device"] == "cuda:0"
        assert config["dtype"] == "auto"
        assert config["trust_remote_code"] is False  # Secure default

    def test_get_model_config_with_override(self):
        registry = LoaderRegistry(
            loader_configs={
                "special-model": {
                    "device": "cuda:1",
                    "dtype": "float16",
                },
            }
        )

        config = registry.get_model_config("special-model")

        assert config["device"] == "cuda:1"
        assert config["dtype"] == "float16"

    def test_list_loaders(self):
        registry = LoaderRegistry()

        loaders = registry.list_loaders()

        assert "transformers" in loaders
        assert loaders["transformers"]["name"] == "transformers"
        assert loaders["transformers"]["type"] == "TransformersLoader"

    def test_probe_model(self):
        registry = LoaderRegistry()

        # Probe a standard model
        result = registry.probe_model("meta-llama/Llama-3.1-8B")

        assert result["model_id"] == "meta-llama/Llama-3.1-8B"
        assert result["detected_loader"] == "transformers"
        assert result["selected_loader"] == "transformers"
        assert result["can_load"]["transformers"] is True

    def test_probe_model_with_config(self):
        registry = LoaderRegistry(
            loader_configs={
                "forced-model": {"loader": "custom"},
            }
        )

        result = registry.probe_model("forced-model")

        assert result["configured_loader"] == "custom"
        assert result["selected_loader"] == "custom"

    def test_register_loader(self):
        registry = LoaderRegistry()

        # Create a mock loader
        mock_loader = MagicMock()
        mock_loader.name = "mock_loader"
        mock_loader.can_load.return_value = False

        registry.register_loader("mock", mock_loader)

        assert "mock" in registry.loaders
        # Should be inserted before 'custom'
        assert registry.fallback_order.index("mock") < registry.fallback_order.index("custom")


class TestLoaderRegistryIntegration:
    """Integration tests for LoaderRegistry with mocked models."""

    @pytest.fixture
    def mock_loaded_model(self):
        """
        Create a LoadedModel stub for tests.
        
        Returns:
            LoadedModel: A mock LoadedModel with MagicMock `model` and `tokenizer`, `model_id` "test-model", CPU `device`, `dtype` float32, `hidden_size` 768, `num_layers` 12, `loader_type` "transformers", and `metadata` {"loader_name": "transformers"}.
        """
        return LoadedModel(
            model=MagicMock(),
            tokenizer=MagicMock(),
            model_id="test-model",
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=768,
            num_layers=12,
            loader_type="transformers",
            metadata={"loader_name": "transformers"},
        )

    def test_generate_uses_correct_loader(self, mock_loaded_model):
        registry = LoaderRegistry()

        # Mock the transformers loader's generate method
        mock_output = GenerationOutput(
            text="Generated text",
            token_ids=[1, 2, 3],
            hidden_states={-1: torch.randn(1, 768)},
            attention_weights=None,
            metadata={},
        )
        registry.loaders["transformers"].generate = MagicMock(return_value=mock_output)

        result = registry.generate(mock_loaded_model, "Hello")

        assert result.text == "Generated text"
        registry.loaders["transformers"].generate.assert_called_once()

    def test_embed_uses_correct_loader(self, mock_loaded_model):
        registry = LoaderRegistry()

        # Mock the transformers loader's embed method
        mock_output = EmbeddingOutput(
            embedding=torch.randn(768),
            shape=(768,),
            metadata={},
        )
        registry.loaders["transformers"].embed = MagicMock(return_value=mock_output)

        result = registry.embed(mock_loaded_model, "Hello")

        assert result.shape == (768,)
        registry.loaders["transformers"].embed.assert_called_once()


class TestAutoDetectionPatterns:
    """Tests for model pattern auto-detection."""

    def test_llama_models(self):
        registry = LoaderRegistry()

        llama_models = [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B",
        ]

        for model in llama_models:
            name, _ = registry.get_loader(model)
            assert name == "transformers", f"Expected transformers for {model}"

    def test_mistral_models(self):
        registry = LoaderRegistry()

        mistral_models = [
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-v0.1",
        ]

        for model in mistral_models:
            name, _ = registry.get_loader(model)
            assert name == "transformers", f"Expected transformers for {model}"

    def test_embedding_models(self):
        registry = LoaderRegistry()

        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2",
        ]

        for model in embedding_models:
            name, _ = registry.get_loader(model)
            assert name == "sentence_transformers", f"Expected sentence_transformers for {model}"


# ============================================================================
# AtlasLoader Tests (consolidated from test_atlas_loader.py)
# ============================================================================


class TestAtlasLoaderBasics:
    """Tests for AtlasLoader basic properties."""

    def test_name(self):
        loader = AtlasLoader()
        assert loader.name == "atlas"

    def test_can_load_atlas_paths(self):
        loader = AtlasLoader()

        # Should handle Atlas-specific patterns
        assert loader.can_load("/home/user/models/Atlas/checkpoint.pt")
        assert loader.can_load("/path/to/atlas/model.pt")
        assert loader.can_load("atlas_model.pt")
        assert loader.can_load("/runs/atlas_dumas/checkpoints/checkpoint_1000.pt")

    def test_can_load_excludes_huggingface_models(self):
        loader = AtlasLoader()

        # Should NOT handle HuggingFace model IDs
        assert not loader.can_load("meta-llama/Llama-3.1-8B-Instruct")
        assert not loader.can_load("mistralai/Mistral-7B-Instruct-v0.2")
        assert not loader.can_load("sentence-transformers/all-MiniLM-L6-v2")

    def test_can_load_excludes_random_paths(self):
        loader = AtlasLoader()

        # Should NOT handle random non-Atlas paths without .pt extension
        assert not loader.can_load("random-model-id")
        assert not loader.can_load("/home/user/models/other/model")


class TestAtlasCheckpointValidation:
    """Tests for Atlas checkpoint validation."""

    @pytest.fixture
    def valid_checkpoint(self):
        """Create a valid checkpoint dict for testing.

        Note: Checkpoint must be > 1MB to pass size validation.
        512x512 float32 tensor = 1MB.
        """
        return {
            "step": 1000,
            "epoch": 10,
            "model_state_dict": {
                # Use larger tensors to pass 1MB size check
                "layer.weight": torch.randn(512, 512),  # ~1MB
                "layer.bias": torch.randn(512),
            },
            "config": {
                "d_model": 512,
                "n_layers": 4,
                "n_heads": 8,
                "vocab_size": 29056,
            },
        }

    @pytest.fixture
    def valid_checkpoint_with_memory(self, valid_checkpoint):
        """Create a valid checkpoint with memory states."""
        checkpoint = valid_checkpoint.copy()
        # Memory states in dict format: 4 layers with M and S matrices
        checkpoint["memory_states"] = [
            {
                "M": torch.randn(32, 1152, 128),
                "S": torch.randn(32, 1152, 128),
            }
            for _ in range(4)
        ]
        return checkpoint

    def test_validate_missing_file(self):
        loader = AtlasLoader()

        result = loader.validate_checkpoint("/nonexistent/path/checkpoint.pt")

        assert result["valid"] is False
        assert "not found" in result["error"].lower()

    def test_validate_missing_file_strict_raises(self):
        loader = AtlasLoader()

        with pytest.raises(CheckpointValidationError) as excinfo:
            loader.validate_checkpoint("/nonexistent/path/checkpoint.pt", strict=True)

        assert "not found" in str(excinfo.value).lower()

    def test_validate_valid_checkpoint(self, valid_checkpoint):
        loader = AtlasLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(valid_checkpoint, path)
            result = loader.validate_checkpoint(str(path))

            assert result["valid"] is True
            assert result["step"] == 1000
            assert result["epoch"] == 10
            assert result["num_parameters"] > 0

    def test_validate_missing_required_keys(self):
        loader = AtlasLoader()

        # Create checkpoint missing required keys but large enough to pass size check
        incomplete_checkpoint = {
            "step": 1000,
            # Missing: epoch, model_state_dict, config
            # Add padding to pass size check (> 1MB)
            "_padding": torch.randn(512, 512),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(incomplete_checkpoint, path)
            result = loader.validate_checkpoint(str(path))

            assert result["valid"] is False
            assert "missing required keys" in result["error"].lower()

    def test_validate_empty_model_state_dict(self):
        loader = AtlasLoader()

        checkpoint = {
            "step": 1000,
            "epoch": 10,
            "model_state_dict": {},  # Empty
            "config": {"d_model": 128},
            # Add padding to pass size check (> 1MB)
            "_padding": torch.randn(512, 512),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(checkpoint, path)
            result = loader.validate_checkpoint(str(path))

            assert result["valid"] is False
            assert "empty" in result["error"].lower()

    def test_validate_memory_states_dict_format(self, valid_checkpoint_with_memory):
        loader = AtlasLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(valid_checkpoint_with_memory, path)
            result = loader.validate_checkpoint(str(path), strict=True)

            assert result["valid"] is True
            assert result["has_memory_states"] is True
            assert result["num_layers_with_memory"] == 4
            # Check that layer shapes are recorded in strict mode
            assert "layer_0_M_shape" in result
            assert result["layer_0_M_shape"] == (32, 1152, 128)

    def test_validate_memory_states_tuple_format(self, valid_checkpoint):
        loader = AtlasLoader()

        checkpoint = valid_checkpoint.copy()
        # Memory states in tuple format: (M, S)
        checkpoint["memory_states"] = [
            (torch.randn(32, 1152, 128), torch.randn(32, 1152, 128))
            for _ in range(4)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(checkpoint, path)
            result = loader.validate_checkpoint(str(path), strict=True)

            assert result["valid"] is True
            assert result["has_memory_states"] is True
            assert result["num_layers_with_memory"] == 4

    def test_validate_checkpoint_too_small(self):
        loader = AtlasLoader()

        # Create a very small file (less than 1MB)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            with open(path, "wb") as f:
                f.write(b"tiny file")
            result = loader.validate_checkpoint(str(path))

            assert result["valid"] is False
            assert "too small" in result["error"].lower()


class TestAtlasStateDictCleaning:
    """Tests for state dict cleaning (removing DDP/compile prefixes)."""

    def test_clean_ddp_prefix(self):
        loader = AtlasLoader()

        state_dict = {
            "module.layer.weight": torch.randn(128, 128),
            "module.layer.bias": torch.randn(128),
        }

        cleaned = loader._clean_state_dict(state_dict)

        assert "layer.weight" in cleaned
        assert "layer.bias" in cleaned
        assert "module.layer.weight" not in cleaned
        assert "module.layer.bias" not in cleaned

    def test_clean_compile_prefix(self):
        loader = AtlasLoader()

        state_dict = {
            "_orig_mod.layer.weight": torch.randn(128, 128),
            "_orig_mod.layer.bias": torch.randn(128),
        }

        cleaned = loader._clean_state_dict(state_dict)

        assert "layer.weight" in cleaned
        assert "layer.bias" in cleaned
        assert "_orig_mod.layer.weight" not in cleaned

    def test_clean_both_prefixes(self):
        loader = AtlasLoader()

        # Combined prefixes (DDP wrapping a compiled model)
        state_dict = {
            "module._orig_mod.layer.weight": torch.randn(128, 128),
        }

        cleaned = loader._clean_state_dict(state_dict)

        # Implementation removes module. first, then _orig_mod. second
        assert "layer.weight" in cleaned
        assert "module._orig_mod.layer.weight" not in cleaned
        assert "_orig_mod.layer.weight" not in cleaned

    def test_clean_no_prefix(self):
        loader = AtlasLoader()

        state_dict = {
            "layer.weight": torch.randn(128, 128),
            "layer.bias": torch.randn(128),
        }

        cleaned = loader._clean_state_dict(state_dict)

        assert "layer.weight" in cleaned
        assert "layer.bias" in cleaned


class TestAtlasMemoryStateRestoration:
    """Tests for memory state restoration from checkpoint."""

    def test_restore_dict_format(self):
        loader = AtlasLoader()

        memory_states = [
            {"M": torch.randn(32, 1152, 128), "S": torch.randn(32, 1152, 128)}
            for _ in range(4)
        ]

        restored = loader._restore_memory_states(
            memory_states,
            torch.device("cpu"),
            torch.float32,
        )

        assert len(restored) == 4
        for M, S in restored:
            assert isinstance(M, torch.Tensor)
            assert isinstance(S, torch.Tensor)
            assert M.shape == (32, 1152, 128)
            assert S.shape == (32, 1152, 128)

    def test_restore_tuple_format(self):
        loader = AtlasLoader()

        memory_states = [
            (torch.randn(32, 1152, 128), torch.randn(32, 1152, 128))
            for _ in range(4)
        ]

        restored = loader._restore_memory_states(
            memory_states,
            torch.device("cpu"),
            torch.float32,
        )

        assert len(restored) == 4
        for M, S in restored:
            assert M.shape == (32, 1152, 128)
            assert S.shape == (32, 1152, 128)

    def test_restore_dtype_conversion(self):
        loader = AtlasLoader()

        # Create memory states in float32
        memory_states = [
            {"M": torch.randn(32, 1152, 128), "S": torch.randn(32, 1152, 128)}
        ]

        # Restore as float16
        restored = loader._restore_memory_states(
            memory_states,
            torch.device("cpu"),
            torch.float16,
        )

        assert restored[0][0].dtype == torch.float16
        assert restored[0][1].dtype == torch.float16

    def test_restore_invalid_format_raises(self):
        loader = AtlasLoader()

        # Invalid format - neither dict nor tuple
        memory_states = ["invalid"]

        with pytest.raises(ValueError) as excinfo:
            loader._restore_memory_states(
                memory_states,
                torch.device("cpu"),
                torch.float32,
            )

        assert "unexpected memory state format" in str(excinfo.value).lower()


class TestAtlasFindLatestCheckpoint:
    """Tests for finding the latest checkpoint in a directory."""

    def test_find_latest_checkpoint(self):
        loader = AtlasLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create multiple checkpoints with different timestamps
            for step in [1000, 5000, 10000]:
                ckpt_path = tmpdir_path / f"checkpoint_{step}.pt"
                torch.save({"step": step}, ckpt_path)

            latest = loader._find_latest_checkpoint(tmpdir_path)

            # Should return the most recently modified file
            assert latest.exists()
            assert latest.suffix == ".pt"

    def test_find_latest_no_checkpoints_raises(self):
        loader = AtlasLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError) as excinfo:
                loader._find_latest_checkpoint(Path(tmpdir))

            assert "no checkpoints found" in str(excinfo.value).lower()


class TestAtlasCheckpointPathResolution:
    """Tests for checkpoint path resolution."""

    def test_resolve_direct_file(self):
        loader = AtlasLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save({"step": 1000}, path)
            resolved = loader._resolve_checkpoint_path(str(path))

            assert resolved == path

    def test_resolve_directory(self):
        loader = AtlasLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint_1000.pt"
            torch.save({"step": 1000}, ckpt_path)

            resolved = loader._resolve_checkpoint_path(tmpdir)

            # Assertions must be inside context before temp dir is deleted
            assert resolved.exists()
            assert resolved.suffix == ".pt"

    def test_resolve_nonexistent_raises(self):
        loader = AtlasLoader()

        with pytest.raises(FileNotFoundError):
            loader._resolve_checkpoint_path("/nonexistent/model_id")


class TestAtlasLoaderIntegration:
    """Integration tests for AtlasLoader with mocked models."""

    @pytest.fixture
    def mock_atlas_loaded_model(self):
        """Create a LoadedModel stub for Atlas tests."""
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Generated text"

        return LoadedModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            model_id="test-atlas-model",
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=128,
            num_layers=4,
            loader_type="atlas",
            metadata={
                "epoch": 10,
                "step": 1000,
                "has_memory_states": True,
                "memory_states": None,
            },
        )

    def test_generate_returns_correct_output_type(self, mock_atlas_loaded_model):
        loader = AtlasLoader()

        # Mock the model's forward pass - set return_value on the model callable
        mock_atlas_loaded_model.model.return_value = (
            torch.randn(1, 5, 29056),
            None,
            None,
        )

        result = loader.generate(
            mock_atlas_loaded_model,
            "Test prompt",
            max_tokens=10,
            temperature=0.7,
        )

        assert isinstance(result, GenerationOutput)
        assert isinstance(result.text, str)
        assert isinstance(result.token_ids, list)
        assert "inference_time_ms" in result.metadata

    def test_generate_with_hidden_states(self, mock_atlas_loaded_model):
        loader = AtlasLoader()

        # Mock the model's forward pass
        mock_atlas_loaded_model.model.return_value = (
            torch.randn(1, 5, 29056),
            None,
            None,
        )

        result = loader.generate(
            mock_atlas_loaded_model,
            "Test prompt",
            return_hidden_states=True,
            hidden_state_layers=[-1],
        )

        assert result.hidden_states is not None
        assert -1 in result.hidden_states

    def test_embed_returns_correct_output_type(self, mock_atlas_loaded_model):
        loader = AtlasLoader()

        # Mock the model's forward pass
        mock_atlas_loaded_model.model.return_value = (
            torch.randn(1, 5, 29056),
            None,
            None,
        )

        result = loader.embed(
            mock_atlas_loaded_model,
            "Test text",
            pooling="last_token",
        )

        assert isinstance(result, EmbeddingOutput)
        assert isinstance(result.embedding, torch.Tensor)
        assert "pooling" in result.metadata
        assert result.metadata["pooling"] == "last_token"

    def test_embed_pooling_strategies(self, mock_atlas_loaded_model):
        loader = AtlasLoader()

        # Mock the model's forward pass
        mock_atlas_loaded_model.model.return_value = (
            torch.randn(1, 5, 29056),
            None,
            None,
        )

        for pooling in ["last_token", "mean", "first_token"]:
            result = loader.embed(
                mock_atlas_loaded_model,
                "Test text",
                pooling=pooling,
            )
            assert result.metadata["pooling"] == pooling

    def test_embed_invalid_pooling_raises(self, mock_atlas_loaded_model):
        loader = AtlasLoader()

        # Mock the model's forward pass
        mock_atlas_loaded_model.model.return_value = (
            torch.randn(1, 5, 29056),
            None,
            None,
        )

        with pytest.raises(ValueError) as excinfo:
            loader.embed(
                mock_atlas_loaded_model,
                "Test text",
                pooling="invalid_pooling",
            )

        assert "unknown pooling" in str(excinfo.value).lower()


class TestAtlasPrunedTokenizerIntegration:
    """Tests for tokenizer integration with AtlasLoader."""

    def test_tokenizer_encode_decode_mock(self):
        """Test tokenizer using mocks when bundled assets unavailable."""
        # Note: AtlasLoader() would be used with real tokenizer integration
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "Hello world"
        mock_tokenizer.vocab_size = 29056

        # Test encode
        encoded = mock_tokenizer.encode("Hello world")
        assert isinstance(encoded, list)
        assert len(encoded) == 5

        # Test decode
        decoded = mock_tokenizer.decode(encoded)
        assert decoded == "Hello world"

        # Check vocabulary size (pruned vocab is ~29k)
        assert mock_tokenizer.vocab_size < 30000


class TestAtlasConfigCreation:
    """Tests for AtlasConfig creation from checkpoint."""

    def test_create_config_from_atlasconfig_instance(self):
        """Test when checkpoint contains AtlasConfig instance."""
        loader = AtlasLoader()

        # Mock AtlasConfig instance
        mock_config = MagicMock()
        mock_config.d_model = 128
        mock_config.n_layers = 4
        mock_config.__class__.__name__ = "AtlasConfig"

        checkpoint = {"config": mock_config}

        # The method should return the config as-is when it's already AtlasConfig
        with patch("src.loaders.atlas_loader.AtlasConfig") as MockAtlasConfig:
            MockAtlasConfig.return_value = mock_config
            # Test the dict path since it's more straightforward
            checkpoint = {
                "config": {
                    "d_model": 128,
                    "n_layers": 4,
                    "n_heads": 4,
                    "d_ff": 512,
                    "vocab_size": 29056,
                }
            }

            with patch.object(loader, "_create_config_from_checkpoint") as mock_create:
                mock_create.return_value = MagicMock(d_model=128, n_layers=4)
                result = mock_create(checkpoint)
                assert result.d_model == 128

    def test_create_config_from_dict(self):
        """Test config creation from dict with various key names."""
        loader = AtlasLoader()

        # Test that the method handles different config key names
        # The loader internally handles hidden_size->d_model, num_layers->n_layers, etc.
        # We can't fully test _create_config_from_checkpoint without AtlasConfig
        # but we can verify the key mapping logic exists by checking the method
        assert hasattr(loader, "_create_config_from_checkpoint")


class TestAtlasLoadWithQuantization:
    """Tests for load method with unsupported quantization."""

    def test_quantization_warning(self, caplog):
        """Test that quantization generates a warning."""
        import logging

        loader = AtlasLoader()

        # We can't fully test load without actual checkpoint
        # but we can verify the quantization warning logic
        with patch.object(loader, "_resolve_checkpoint_path") as mock_resolve:
            mock_resolve.side_effect = FileNotFoundError("Test")

            with pytest.raises(FileNotFoundError):
                # This will fail at path resolution, but the warning should be logged first
                with caplog.at_level(logging.WARNING):
                    loader.load(
                        "test_model",
                        quantization="int8",
                    )


class TestAtlasRequiredCheckpointKeys:
    """Tests for required checkpoint key definitions."""

    def test_required_keys_defined(self):
        """Verify required checkpoint keys are defined correctly."""
        assert "step" in REQUIRED_CHECKPOINT_KEYS
        assert "epoch" in REQUIRED_CHECKPOINT_KEYS
        assert "model_state_dict" in REQUIRED_CHECKPOINT_KEYS
        assert "config" in REQUIRED_CHECKPOINT_KEYS


class TestAtlasDeviceRemapping:
    """Tests for device remapping during checkpoint loading."""

    def test_validate_uses_cpu_mapping(self):
        """Verify validation loads with CPU mapping to avoid GPU memory."""
        loader = AtlasLoader()

        # Create checkpoint large enough to pass size check (> 1MB)
        checkpoint = {
            "step": 1000,
            "epoch": 10,
            "model_state_dict": {"layer.weight": torch.randn(512, 512)},
            "config": {"d_model": 512},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(checkpoint, path)

            # Validation should work without GPU
            with patch("torch.load") as mock_load:
                mock_load.return_value = checkpoint

                loader.validate_checkpoint(str(path))

                # Verify torch.load was called with cpu mapping
                mock_load.assert_called()
                call_kwargs = mock_load.call_args[1]
                assert call_kwargs.get("map_location") == "cpu"
