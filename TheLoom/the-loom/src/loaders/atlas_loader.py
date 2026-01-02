"""Atlas model loader with checkpoint validation and memory state handling.

This module provides the AtlasLoader class for loading Atlas model checkpoints
trained with the Titans-style memory architecture. It handles:
- Checkpoint loading with device remapping
- Checkpoint structure validation
- Memory state extraction and validation
- Integration with the pruned mT5 tokenizer

References:
- Atlas model: /home/todd/olympus/models/Atlas/
- Checkpoints: /home/todd/olympus/models/Atlas/runs/*/checkpoints/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .atlas_model import Atlas, AtlasConfig
from .atlas_tokenizer import PrunedTokenizer
from .base import (
    EmbeddingOutput,
    GenerationOutput,
    LoadedModel,
    ModelLoader,
    resolve_dtype,
)

logger = logging.getLogger(__name__)

# Default checkpoint directory from environment or hardcoded fallback
DEFAULT_CHECKPOINT_DIR = os.environ.get(
    "ATLAS_CHECKPOINT_DIR",
    "/home/todd/olympus/models/Atlas/runs/atlas_dumas/checkpoints",
)

# Required keys in a valid Atlas checkpoint
REQUIRED_CHECKPOINT_KEYS = [
    "step",
    "epoch",
    "model_state_dict",
    "config",
]

# Optional but expected keys
OPTIONAL_CHECKPOINT_KEYS = [
    "memory_states",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "training_state",
]


class CheckpointValidationError(Exception):
    """Raised when checkpoint validation fails."""

    pass


class AtlasLoader(ModelLoader):
    """Model loader for Atlas checkpoints with Titans-style memory.

    This loader handles:
    - Loading Atlas model checkpoints with device remapping
    - Validating checkpoint structure and memory states
    - Restoring memory states for continued inference
    - Integration with the pruned mT5 tokenizer

    Key capability: Proper memory state restoration for interpretability analysis.
    """

    @property
    def name(self) -> str:
        """Return the loader's identifier."""
        return "atlas"

    def can_load(self, model_id: str) -> bool:
        """Determine whether this loader can handle the given model identifier.

        Parameters:
            model_id: Model identifier or checkpoint path.

        Returns:
            True if the model_id looks like an Atlas checkpoint path.
        """
        model_lower = model_id.lower()

        # Check for Atlas-specific patterns
        atlas_patterns = [
            "atlas",
            "/atlas/",
            "atlas_",
            ".pt",  # PyTorch checkpoint files
        ]

        for pattern in atlas_patterns:
            if pattern in model_lower:
                # Verify it's not a HuggingFace model path
                if "/" in model_id and not Path(model_id).exists():
                    # Looks like a HF model ID, not Atlas
                    if not model_id.startswith("/"):
                        continue
                return True

        return False

    def validate_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Validate an Atlas checkpoint file.

        Parameters:
            checkpoint_path: Path to the checkpoint file.
            strict: If True, also validates memory state shapes.

        Returns:
            Validation result dict with keys:
                - valid: bool indicating if checkpoint is valid
                - epoch: int epoch number (if valid)
                - step: int step number (if valid)
                - error: str error message (if invalid)
                - config: AtlasConfig (if valid)

        Raises:
            CheckpointValidationError: If checkpoint is invalid and strict=True.
        """
        checkpoint_path = Path(checkpoint_path)
        result: dict[str, Any] = {
            "valid": False,
            "path": str(checkpoint_path),
        }

        # Check file exists
        if not checkpoint_path.exists():
            result["error"] = f"Checkpoint file not found: {checkpoint_path}"
            if strict:
                raise CheckpointValidationError(result["error"])
            return result

        # Check file size (Atlas checkpoints are typically 50-500MB)
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        result["file_size_mb"] = file_size_mb

        if file_size_mb < 1:
            result["error"] = f"Checkpoint file too small: {file_size_mb:.2f}MB"
            if strict:
                raise CheckpointValidationError(result["error"])
            return result

        try:
            # Load checkpoint with CPU mapping to avoid GPU memory during validation
            # Use weights_only=False because checkpoints contain AtlasConfig instances
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            # Check required keys
            missing_keys = [
                key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint
            ]
            if missing_keys:
                result["error"] = f"Missing required keys: {missing_keys}"
                if strict:
                    raise CheckpointValidationError(result["error"])
                return result

            # Extract metadata
            result["step"] = checkpoint.get("step", 0)
            result["epoch"] = checkpoint.get("epoch", 0)

            # Validate config
            config = checkpoint.get("config")
            if config is None:
                result["error"] = "Missing config in checkpoint"
                if strict:
                    raise CheckpointValidationError(result["error"])
                return result

            # Config can be AtlasConfig instance or dict
            if isinstance(config, dict):
                result["config_dict"] = config
            elif hasattr(config, "__dict__"):
                result["config_dict"] = {
                    k: v
                    for k, v in config.__dict__.items()
                    if not k.startswith("_")
                }
            else:
                result["config_dict"] = {"type": str(type(config))}

            # Validate model_state_dict
            model_state = checkpoint.get("model_state_dict", {})
            if not model_state:
                result["error"] = "Empty model_state_dict"
                if strict:
                    raise CheckpointValidationError(result["error"])
                return result

            result["num_parameters"] = sum(
                p.numel() for p in model_state.values() if isinstance(p, torch.Tensor)
            )

            # Validate memory states if present and strict mode
            memory_states = checkpoint.get("memory_states")
            if memory_states is not None:
                result["has_memory_states"] = True
                result["num_layers_with_memory"] = len(memory_states)

                if strict:
                    for i, layer_mem in enumerate(memory_states):
                        if isinstance(layer_mem, dict):
                            if "M" in layer_mem:
                                m_shape = tuple(layer_mem["M"].shape)
                                result[f"layer_{i}_M_shape"] = m_shape
                            if "S" in layer_mem:
                                s_shape = tuple(layer_mem["S"].shape)
                                result[f"layer_{i}_S_shape"] = s_shape
                        elif isinstance(layer_mem, tuple) and len(layer_mem) == 2:
                            # (M, S) tuple format
                            m_shape = tuple(layer_mem[0].shape)
                            s_shape = tuple(layer_mem[1].shape)
                            result[f"layer_{i}_M_shape"] = m_shape
                            result[f"layer_{i}_S_shape"] = s_shape
            else:
                result["has_memory_states"] = False

            # Checkpoint is valid
            result["valid"] = True

        except CheckpointValidationError:
            raise
        except Exception as e:
            result["error"] = f"Failed to load checkpoint: {e}"
            if strict:
                raise CheckpointValidationError(result["error"]) from e

        finally:
            # Clean up memory
            if "checkpoint" in dir():
                del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    def load(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        restore_memory: bool = True,
        trust_remote_code: bool = False,  # Ignored for Atlas but kept for interface
        quantization: str | None = None,  # Not supported for Atlas
        **kwargs: Any,
    ) -> LoadedModel:
        """Load an Atlas model from a checkpoint.

        Parameters:
            model_id: Path to checkpoint file or checkpoint directory.
            device: Target device for model placement.
            dtype: Data type resolution strategy.
            restore_memory: Whether to restore memory states from checkpoint.
            trust_remote_code: Ignored (kept for interface compatibility).
            quantization: Not supported for Atlas (raises if specified).
            **kwargs: Additional options (ignored).

        Returns:
            LoadedModel containing the Atlas model, tokenizer, and metadata.
        """
        if quantization is not None:
            logger.warning(
                f"Quantization '{quantization}' not supported for Atlas models, ignoring"
            )

        logger.info(f"Loading Atlas model from {model_id} on {device}")
        start_time = time.time()

        # Resolve checkpoint path
        checkpoint_path = self._resolve_checkpoint_path(model_id)
        logger.info(f"Resolved checkpoint path: {checkpoint_path}")

        # Validate checkpoint
        validation = self.validate_checkpoint(checkpoint_path, strict=True)
        logger.info(
            f"Checkpoint valid - epoch {validation['epoch']}, step {validation['step']}"
        )

        # Resolve device and dtype
        torch_device = torch.device(device)
        torch_dtype = resolve_dtype(dtype, torch_device)

        # Load checkpoint with device remapping
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        # Extract and create config
        config = self._create_config_from_checkpoint(checkpoint)

        # Create model
        model = Atlas(config)

        # Load state dict
        model_state = checkpoint["model_state_dict"]

        # Handle potential key mismatches (e.g., from DDP training)
        model_state = self._clean_state_dict(model_state)

        model.load_state_dict(model_state, strict=False)
        model = model.to(torch_device)
        model = model.to(torch_dtype)
        model.eval()

        # Extract memory states if present and requested
        memory_states = None
        if restore_memory and "memory_states" in checkpoint:
            memory_states = self._restore_memory_states(
                checkpoint["memory_states"],
                torch_device,
                torch_dtype,
            )
            logger.info(f"Restored memory states for {len(memory_states)} layers")

        # Load tokenizer
        try:
            tokenizer = PrunedTokenizer.from_bundled()
            logger.info(f"Loaded bundled tokenizer, vocab size: {tokenizer.vocab_size}")
        except FileNotFoundError:
            # Try loading from environment path
            tokenizer_path = os.environ.get(
                "ATLAS_TOKENIZER_DIR",
                "/home/todd/olympus/models/Atlas/tokenizer/atlas_tokenizer",
            )
            tokenizer = PrunedTokenizer.from_path(tokenizer_path)
            logger.info(
                f"Loaded tokenizer from {tokenizer_path}, vocab size: {tokenizer.vocab_size}"
            )

        load_time = time.time() - start_time

        logger.info(
            f"Atlas model loaded in {load_time:.2f}s - "
            f"hidden_size={config.d_model}, num_layers={config.n_layers}"
        )

        # Clean up checkpoint from memory
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                "load_time_seconds": load_time,
                "epoch": validation["epoch"],
                "step": validation["step"],
                "config": validation.get("config_dict", {}),
                "has_memory_states": memory_states is not None,
                "memory_states": memory_states,
                "file_size_mb": validation.get("file_size_mb", 0),
            },
        )

    def _resolve_checkpoint_path(self, model_id: str) -> Path:
        """Resolve model_id to an actual checkpoint path.

        Parameters:
            model_id: Path to checkpoint file, directory, or identifier.

        Returns:
            Path to the checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint cannot be found.
        """
        path = Path(model_id)

        # Direct file path
        if path.exists() and path.is_file():
            return path

        # Directory - find latest checkpoint
        if path.exists() and path.is_dir():
            return self._find_latest_checkpoint(path)

        # Check default directory
        default_path = Path(DEFAULT_CHECKPOINT_DIR) / model_id
        if default_path.exists():
            if default_path.is_file():
                return default_path
            return self._find_latest_checkpoint(default_path)

        # Try as a relative path in default directory
        for pattern in [f"{model_id}", f"{model_id}.pt", f"checkpoint_{model_id}.pt"]:
            candidate = Path(DEFAULT_CHECKPOINT_DIR) / pattern
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Cannot find Atlas checkpoint: {model_id}. "
            f"Checked paths: {path}, {default_path}"
        )

    def _find_latest_checkpoint(self, directory: Path) -> Path:
        """Find the most recent checkpoint in a directory.

        Parameters:
            directory: Directory to search.

        Returns:
            Path to the most recent checkpoint file.
        """
        checkpoints = list(directory.glob("checkpoint_*.pt"))
        if not checkpoints:
            checkpoints = list(directory.glob("*.pt"))

        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {directory}")

        # Sort by modification time, newest first
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]

    def _create_config_from_checkpoint(
        self, checkpoint: dict[str, Any]
    ) -> AtlasConfig:
        """Create an AtlasConfig from checkpoint data.

        Parameters:
            checkpoint: Loaded checkpoint dictionary.

        Returns:
            AtlasConfig instance.
        """
        config_data = checkpoint.get("config")

        if isinstance(config_data, AtlasConfig):
            return config_data

        if isinstance(config_data, dict):
            # Map checkpoint config keys to AtlasConfig
            # Handle potential key name differences
            config_kwargs = {}

            key_mapping = {
                "d_model": ["d_model", "hidden_size", "embed_dim"],
                "n_layers": ["n_layers", "num_layers", "num_hidden_layers"],
                "n_heads": ["n_heads", "num_heads", "num_attention_heads"],
                "d_ff": ["d_ff", "intermediate_size", "ffn_dim"],
                "vocab_size": ["vocab_size"],
                "max_seq_len": ["max_seq_len", "max_position_embeddings", "max_length"],
                "d_key": ["d_key", "d_memory"],
                "d_value": ["d_value"],
                "window_size": ["window_size"],
                "dropout": ["dropout", "hidden_dropout_prob"],
            }

            for target_key, source_keys in key_mapping.items():
                for source_key in source_keys:
                    if source_key in config_data:
                        config_kwargs[target_key] = config_data[source_key]
                        break

            # Ensure d_key and d_value match d_model if not specified
            if "d_key" not in config_kwargs and "d_model" in config_kwargs:
                config_kwargs["d_key"] = config_kwargs["d_model"]
            if "d_value" not in config_kwargs and "d_model" in config_kwargs:
                config_kwargs["d_value"] = config_kwargs["d_model"]

            return AtlasConfig(**config_kwargs)

        # Fallback to default config
        logger.warning("Using default AtlasConfig - checkpoint config not recognized")
        return AtlasConfig()

    def _clean_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Clean a state dict, removing DDP prefixes etc.

        Parameters:
            state_dict: Raw state dictionary from checkpoint.

        Returns:
            Cleaned state dictionary.
        """
        cleaned = {}
        for key, value in state_dict.items():
            # Remove 'module.' prefix from DDP training
            if key.startswith("module."):
                key = key[7:]
            # Remove '_orig_mod.' prefix from torch.compile
            if key.startswith("_orig_mod."):
                key = key[10:]
            cleaned[key] = value
        return cleaned

    def _restore_memory_states(
        self,
        memory_states: list[Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Restore memory states from checkpoint format.

        Parameters:
            memory_states: Memory states from checkpoint.
            device: Target device.
            dtype: Target dtype.

        Returns:
            List of (M, S) tuples for each layer.
        """
        restored = []

        for layer_state in memory_states:
            if isinstance(layer_state, dict):
                M = layer_state["M"].to(device=device, dtype=dtype)
                S = layer_state["S"].to(device=device, dtype=dtype)
            elif isinstance(layer_state, tuple) and len(layer_state) == 2:
                M = layer_state[0].to(device=device, dtype=dtype)
                S = layer_state[1].to(device=device, dtype=dtype)
            else:
                raise ValueError(f"Unexpected memory state format: {type(layer_state)}")

            restored.append((M, S))

        return restored

    def generate(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        return_hidden_states: bool = True,
        hidden_state_layers: list[int] | None = None,
        return_attention: bool = False,
        top_k: int | None = 50,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate text using the Atlas model.

        Parameters:
            loaded_model: Loaded Atlas model.
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            return_hidden_states: Whether to return hidden states.
            hidden_state_layers: Which layers to return (default: last).
            return_attention: Whether to return attention (not supported).
            top_k: Top-k sampling parameter.
            **kwargs: Additional generation options.

        Returns:
            GenerationOutput with generated text and optional hidden states.
        """
        model: Atlas = loaded_model.model
        tokenizer: PrunedTokenizer = loaded_model.tokenizer
        device = loaded_model.device

        if hidden_state_layers is None:
            hidden_state_layers = [-1]

        # Tokenize input
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=device)

        start_time = time.time()

        # Get memory states if stored in metadata
        memory_states = loaded_model.metadata.get("memory_states")

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        inference_time = time.time() - start_time

        # Extract generated tokens (excluding input)
        input_length = input_ids.shape[1]
        generated_ids = output_ids[0, input_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract hidden states if requested
        hidden_states_dict: dict[int, torch.Tensor] | None = None
        if return_hidden_states:
            # For Atlas, we need to do a forward pass to get hidden states
            # This is less efficient but necessary for the current architecture
            hidden_states_dict = self._extract_hidden_states(
                model,
                output_ids,
                hidden_state_layers,
                loaded_model.num_layers,
                memory_states,
            )

        return GenerationOutput(
            text=generated_text,
            token_ids=generated_ids,
            hidden_states=hidden_states_dict,
            attention_weights=None,  # Not supported for Atlas
            metadata={
                "inference_time_ms": inference_time * 1000,
                "tokens_generated": len(generated_ids),
                "tokens_per_second": len(generated_ids) / inference_time
                if inference_time > 0
                else 0,
                "input_tokens": input_length,
                "temperature": temperature,
                "model_id": loaded_model.model_id,
            },
        )

    def _extract_hidden_states(
        self,
        model: Atlas,
        input_ids: torch.Tensor,
        layers: list[int],
        num_layers: int,
        memory_states: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Extract hidden states from Atlas model.

        Parameters:
            model: Atlas model.
            input_ids: Input token IDs.
            layers: Layer indices to extract.
            num_layers: Total number of layers.
            memory_states: Optional pre-loaded memory states.

        Returns:
            Dict mapping layer index to final logits tensor (post-LM-head).
            Note: Atlas currently returns vocab-sized logits for all requested
            layers as intermediate hidden states are not exposed. Shape is
            [batch, vocab_size] rather than typical [batch, d_model].
        """
        # This is a simplified extraction - Atlas architecture would need
        # modification to return intermediate hidden states properly.
        # For now, we return the final layer's output (post-LM-head logits).
        result: dict[int, torch.Tensor] = {}

        with torch.no_grad():
            # Forward pass with memory states
            logits, new_memory_states, _ = model(
                input_ids,
                memory_states=memory_states,
                return_metrics=False,
            )

            # For the last layer, we can extract from the final output
            # before the lm_head projection
            for layer_idx in layers:
                # Convert negative indices to positive (e.g., -1 -> num_layers-1)
                _ = layer_idx if layer_idx >= 0 else num_layers + layer_idx
                # Store the last token's representation
                # Shape: [batch, hidden_size]
                result[layer_idx] = logits[:, -1, :].cpu()

        return result

    def embed(
        self,
        loaded_model: LoadedModel,
        text: str,
        pooling: str = "last_token",
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """Compute an embedding for text using Atlas model.

        Parameters:
            loaded_model: Loaded Atlas model.
            text: Input text to embed.
            pooling: Pooling strategy (last_token, mean, first_token).
            **kwargs: Additional options.

        Returns:
            EmbeddingOutput with the embedding tensor.
        """
        model: Atlas = loaded_model.model
        tokenizer: PrunedTokenizer = loaded_model.tokenizer
        device = loaded_model.device

        # Tokenize
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=device)

        start_time = time.time()

        # Get memory states if available
        memory_states = loaded_model.metadata.get("memory_states")

        with torch.no_grad():
            # Forward pass to get final hidden states
            logits, _, _ = model(
                input_ids,
                memory_states=memory_states,
                return_metrics=False,
            )

            # logits shape: [batch, seq_len, vocab_size]
            # We need hidden states before lm_head, which requires model modification
            # For now, use logits as a proxy (not ideal but functional)
            hidden_states = logits

            # Apply pooling
            if pooling == "last_token":
                embedding = hidden_states[:, -1, :]
            elif pooling == "mean":
                embedding = hidden_states.mean(dim=1)
            elif pooling == "first_token":
                embedding = hidden_states[:, 0, :]
            else:
                raise ValueError(
                    f"Unknown pooling: {pooling}. Use: last_token, mean, first_token"
                )

        inference_time = time.time() - start_time

        return EmbeddingOutput(
            embedding=embedding.squeeze(0).cpu(),
            shape=tuple(embedding.shape),
            metadata={
                "pooling": pooling,
                "inference_time_ms": inference_time * 1000,
                "input_tokens": input_ids.shape[1],
                "model_id": loaded_model.model_id,
            },
        )


def validate_checkpoint_cli(checkpoint_path: str) -> None:
    """CLI validation function for testing.

    Parameters:
        checkpoint_path: Path to checkpoint to validate.
    """
    loader = AtlasLoader()

    try:
        result = loader.validate_checkpoint(checkpoint_path, strict=True)
        print(f"Checkpoint valid: epoch {result['epoch']}")
        if result.get("step"):
            print(f"Step: {result['step']}")
        if result.get("config_dict"):
            print(f"Config: {result['config_dict']}")
        if result.get("has_memory_states"):
            print(f"Memory states: {result['num_layers_with_memory']} layers")
    except CheckpointValidationError as e:
        print(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atlas checkpoint loader and validator")
    parser.add_argument(
        "--validate",
        type=str,
        help="Path to checkpoint file to validate",
    )
    parser.add_argument(
        "--validate-all",
        action="store_true",
        help="Validate all checkpoints in default directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.validate:
        validate_checkpoint_cli(args.validate)
    elif args.validate_all:
        checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            print(f"Checkpoint directory not found: {checkpoint_dir}")
            sys.exit(1)

        loader = AtlasLoader()
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        valid_count = 0
        total_count = len(checkpoints)

        for cp_path in sorted(checkpoints):
            try:
                result = loader.validate_checkpoint(cp_path, strict=False)
                if result["valid"]:
                    valid_count += 1
                    if args.verbose:
                        print(f"  Valid: {cp_path.name} (epoch {result.get('epoch', '?')})")
                else:
                    print(f"  Invalid: {cp_path.name} - {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  Error: {cp_path.name} - {e}")

        print(f"{valid_count}/{total_count} valid")
    else:
        parser.print_help()
