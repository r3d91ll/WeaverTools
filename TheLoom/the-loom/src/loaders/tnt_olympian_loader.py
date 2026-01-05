"""TNT Olympian Model Loader for memory analysis.

This loader supports loading TNT Olympian checkpoints for analysis of
hierarchical memory states. The focus is on extracting and analyzing
memory metrics (D_eff, beta, weight norms) for Conveyance Hypothesis validation.

TNT Olympian Memory Architecture:
    - TNTOlympian model with N blocks
    - Each block contains TNTHierarchicalMemory:
        - 1 GlobalMemoryModule (large chunks, no reset)
        - 4 LocalMemoryModules (small chunks, periodic reset)
    - Each LocalMemoryModule has:
        - W1, W2: Memory MLP weights
        - k_buffer, v_buffer: Context window buffers
        - momentum_W1, momentum_W2: Newton-Schulz momentum

References:
    - TNT (arXiv:2511.07343): Hierarchical memory architecture
    - Titans (arXiv:2501.00663): MAG architecture, gradient-based memory
    - Atlas (arXiv:2505.23735): Omega rule optimization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .base import (
    EmbeddingOutput,
    GenerationOutput,
    LoadedModel,
    ModelLoader,
    resolve_dtype,
)
from .tnt_olympian_tokenizer import ShakespeareBPETokenizer

logger = logging.getLogger(__name__)

# Model config keys (filter out training params from checkpoint config)
MODEL_CONFIG_KEYS = {
    "vocab_size",
    "d_model",
    "n_layers",
    "n_heads",
    "d_ff",
    "window_size",
    "dropout",
    "max_seq_len",
    "tie_embeddings",
    "global_chunk_size",
    "local_chunk_sizes",
    "shard_length",
    "context_size",
    "poly_degree",
    "use_qk_projection",
}

# Required checkpoint keys
REQUIRED_CHECKPOINT_KEYS = {"model_state_dict", "global_step", "config"}


class TNTOlympianLoader(ModelLoader):
    """Loader for TNT Olympian models with hierarchical memory.

    This loader is designed for memory analysis rather than production inference.
    It loads checkpoints, extracts memory states, and provides access to internal
    model components for computing D_eff, beta, and other Conveyance metrics.

    Attributes:
        name: Loader identifier ("tnt_olympian").

    Example:
        ```python
        loader = TNTOlympianLoader()
        loaded = loader.load("/path/to/checkpoint.pt")

        # Access memory state for analysis
        memory_state = loaded.metadata["memory_state"]
        ```
    """

    @property
    def name(self) -> str:
        """Return loader identifier."""
        return "tnt_olympian"

    def can_load(self, model_id: str) -> bool:
        """Check if this loader can handle the given model.

        Matches paths containing 'olympian' or 'tnt' and ending with .pt.
        Does not match HuggingFace model IDs.

        Args:
            model_id: Path to checkpoint or model identifier.

        Returns:
            True if this loader can handle the model.
        """
        model_lower = model_id.lower()

        # Must be a file path ending in .pt
        if not model_lower.endswith(".pt"):
            return False

        # Match TNT Olympian patterns
        patterns = ["olympian", "tnt_", "/tnt/", "tnt-"]
        return any(p in model_lower for p in patterns)

    def _filter_model_config(self, checkpoint_config: dict[str, Any]) -> dict[str, Any]:
        """Extract only model config keys from checkpoint config.

        Checkpoint config often contains training parameters (learning_rate,
        batch_size, etc.) that should not be passed to TNTOlympianConfig.

        Args:
            checkpoint_config: Full config from checkpoint.

        Returns:
            Filtered config with only model parameters.
        """
        return {k: v for k, v in checkpoint_config.items() if k in MODEL_CONFIG_KEYS}

    def _validate_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: str,
    ) -> dict[str, Any]:
        """Validate checkpoint structure.

        Args:
            checkpoint: Loaded checkpoint dict.
            path: Path to checkpoint (for error messages).

        Returns:
            Validation result with status and details.

        Raises:
            ValueError: If checkpoint is invalid.
        """
        missing = REQUIRED_CHECKPOINT_KEYS - set(checkpoint.keys())
        if missing:
            raise ValueError(
                f"Invalid TNT Olympian checkpoint at {path}. "
                f"Missing required keys: {missing}"
            )

        config = checkpoint.get("config", {})
        if "d_model" not in config and "vocab_size" not in config:
            raise ValueError(
                f"Checkpoint config missing model parameters at {path}. "
                f"Expected 'd_model' or 'vocab_size' in config."
            )

        return {
            "valid": True,
            "has_memory_state": "memory_state" in checkpoint,
            "global_step": checkpoint.get("global_step", 0),
            "current_stage": checkpoint.get("current_stage", 0),
            "model_config": self._filter_model_config(config),
        }

    def load(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: str = "auto",
        tokenizer_path: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load TNT Olympian checkpoint for memory analysis.

        Args:
            model_id: Path to checkpoint file (.pt).
            device: Device for model ("cpu" recommended for analysis).
            dtype: Data type ("auto", "float32", "float16", "bfloat16").
            tokenizer_path: Path to tokenizer directory. If None, uses default.
            **kwargs: Additional options (unused).

        Returns:
            LoadedModel with model, tokenizer, and memory state in metadata.

        Raises:
            FileNotFoundError: If checkpoint not found.
            ValueError: If checkpoint is invalid.
        """
        # Import olympian package (installed from Todd_Atlas)
        try:
            from olympian import TNTOlympian, TNTOlympianConfig
        except ImportError as e:
            raise ImportError(
                "olympian package required. Install with: "
                "poetry add git+https://github.com/r3d91ll/Todd_Atlas.git"
            ) from e

        # Resolve paths
        checkpoint_path = Path(model_id)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Resolve device and dtype
        torch_device = torch.device(device)
        torch_dtype = resolve_dtype(dtype, torch_device)

        logger.info(f"Loading TNT Olympian checkpoint from {checkpoint_path}")

        # Load checkpoint (to CPU first, then move)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        # Validate checkpoint
        validation = self._validate_checkpoint(checkpoint, str(checkpoint_path))
        model_config = validation["model_config"]

        # Create model config and model
        logger.info(f"Creating TNTOlympian with config: {model_config}")
        config = TNTOlympianConfig(**model_config)
        model = TNTOlympian(config)

        # Load state dict (strict=False for dynamic memory buffers)
        state_dict = checkpoint["model_state_dict"]

        # Clean up DDP/compiled model prefixes
        state_dict = self._clean_state_dict(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys in state dict: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys in state dict: {unexpected[:5]}...")

        # Move to device and dtype
        model = model.to(device=torch_device, dtype=torch_dtype)
        model.eval()

        # Extract memory state if present
        memory_state = checkpoint.get("memory_state", None)
        if memory_state is None:
            # Extract from model
            memory_state = model.get_memory_state()
            logger.info("Memory state extracted from model (not in checkpoint)")
        else:
            # Restore memory state to model
            model.set_memory_state(memory_state)
            logger.info("Memory state restored from checkpoint")

        # Load tokenizer
        tokenizer = ShakespeareBPETokenizer(tokenizer_path)
        logger.info(f"Loaded tokenizer with vocab_size={tokenizer.vocab_size}")

        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            model_id=str(checkpoint_path),
            device=torch_device,
            dtype=torch_dtype,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            loader_type=self.name,
            metadata={
                "global_step": checkpoint.get("global_step", 0),
                "current_stage": checkpoint.get("current_stage", 0),
                "memory_state": memory_state,
                "config": model_config,
                "validation": validation,
            },
        )

    def _clean_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Clean state dict by removing DDP/compiled model prefixes.

        Args:
            state_dict: Original state dict from checkpoint.

        Returns:
            Cleaned state dict.
        """
        cleaned = {}
        prefixes = ["module.", "_orig_mod."]

        for key, value in state_dict.items():
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            cleaned[new_key] = value

        return cleaned

    def generate(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        return_hidden_states: bool = True,
        hidden_state_layers: list[int] | None = None,
        return_attention: bool = False,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate text and extract memory states.

        This method is designed for analysis - it runs inference and captures
        memory states before and after generation for comparison.

        Args:
            loaded_model: Loaded TNT Olympian model.
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            return_hidden_states: Include hidden states in output.
            hidden_state_layers: Layer indices for hidden states (-1 = last).
            return_attention: Include attention weights (not implemented).
            **kwargs: Additional generation options (top_k, top_p).

        Returns:
            GenerationOutput with text, token IDs, and hidden states.
        """
        model = loaded_model.model  
        tokenizer = loaded_model.tokenizer
        device = loaded_model.device

        # Capture memory state before generation
        memory_before = model.get_memory_state()  # type: ignore[operator]

        # Encode prompt
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            device=device,
        )

        # Generate
        with torch.no_grad():
            output_ids = model.generate(  # type: ignore[operator]
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=kwargs.get("top_k"),
                top_p=kwargs.get("top_p"),
            )

        # Capture memory state after generation
        memory_after = model.get_memory_state()  # type: ignore[operator]

        # Decode output
        generated_ids = output_ids[0].tolist()
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract hidden states if requested
        hidden_states = None
        if return_hidden_states:
            # Run forward pass to get hidden states
            with torch.no_grad():
                x = model._get_embeddings(output_ids)  # type: ignore[operator]
                layers = hidden_state_layers or [-1]

                hidden_states = {}
                for i, block in enumerate(model.blocks):  # type: ignore[arg-type]
                    x, _ = block(x, update_memory=False)
                    if i in layers or -1 in layers:
                        layer_idx = i if i in layers else -1
                        # Take last token's hidden state
                        hidden_states[layer_idx] = x[0, -1, :].cpu()

        return GenerationOutput(
            text=text,
            token_ids=generated_ids[len(input_ids[0]) :],  # Only new tokens
            hidden_states=hidden_states,
            attention_weights=None,  # Not implemented for TNT Olympian
            metadata={
                "memory_before": memory_before,
                "memory_after": memory_after,
                "prompt_length": len(input_ids[0]),
            },
        )

    def embed(
        self,
        loaded_model: LoadedModel,
        text: str,
        pooling: str = "last_token",
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """Extract embedding from text.

        Args:
            loaded_model: Loaded TNT Olympian model.
            text: Input text.
            pooling: Pooling strategy ("last_token", "mean", "first").
            **kwargs: Additional options.

        Returns:
            EmbeddingOutput with embedding tensor.
        """
        model = loaded_model.model  
        tokenizer = loaded_model.tokenizer
        device = loaded_model.device

        # Encode text
        input_ids = torch.tensor(
            [tokenizer.encode(text)],
            device=device,
        )

        # Forward pass
        with torch.no_grad():
            x = model._get_embeddings(input_ids)  # type: ignore[operator]
            for block in model.blocks:  # type: ignore[union-attr]
                x, _ = block(x, update_memory=False)

        # Apply pooling
        if pooling == "last_token":
            embedding = x[0, -1, :]
        elif pooling == "mean":
            embedding = x[0].mean(dim=0)
        elif pooling == "first":
            embedding = x[0, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return EmbeddingOutput(
            embedding=embedding.cpu(),
            shape=tuple(embedding.shape),
            metadata={
                "pooling": pooling,
                "sequence_length": input_ids.shape[1],
            },
        )

    def extract_memory_state(
        self,
        loaded_model: LoadedModel,
    ) -> dict[str, Any]:
        """Extract current memory state from loaded model.

        Convenience method for memory analysis workflows.

        Args:
            loaded_model: Loaded TNT Olympian model.

        Returns:
            Full memory state dict with block_0, block_1, etc.
        """
        model = loaded_model.model  
        result: dict[str, Any] = model.get_memory_state()  # type: ignore[operator]
        return result

    def reset_memory(
        self,
        loaded_model: LoadedModel,
        reset_type: str = "all",
    ) -> None:
        """Reset model memory.

        Args:
            loaded_model: Loaded TNT Olympian model.
            reset_type: "all" to reset everything, "local" for local memories only.
        """
        model = loaded_model.model  
        if reset_type == "all":
            model.reset_all_memories()  # type: ignore[operator]
        elif reset_type == "local":
            model.reset_local_memories()  # type: ignore[operator]
        else:
            raise ValueError(f"Unknown reset_type: {reset_type}")
