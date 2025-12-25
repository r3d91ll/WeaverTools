"""Qwen-specific model loader for multimodal and MoE models.

This loader handles Qwen models that require special model classes:
1. Qwen3-Omni (qwen3_omni_moe) - Multimodal MoE models
2. Qwen2.5-Omni - Multimodal models
3. Qwen3-VL / Qwen3-VL-MoE - Vision-language models

Standard Qwen text models (Qwen3, Qwen3-MoE, Qwen2.5) work with
the TransformersLoader and don't need this loader.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any, ClassVar

import torch
from transformers import AutoConfig, AutoTokenizer

from .base import (
    EmbeddingOutput,
    GenerationOutput,
    LoadedModel,
    ModelLoader,
    StreamingOutput,
    StreamingToken,
    resolve_dtype,
)

logger = logging.getLogger(__name__)


# Model type to class mapping for Qwen multimodal models
QWEN_MODEL_CLASSES = {
    "qwen3_omni_moe": "Qwen3OmniMoeForConditionalGeneration",
    "qwen2_5_omni": "Qwen2_5OmniForConditionalGeneration",
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
    "qwen3_vl_moe": "Qwen3VLMoeForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
}


class QwenLoader(ModelLoader):
    """Model loader for Qwen multimodal and specialized models.

    This loader handles Qwen models that require specific model classes
    instead of AutoModelForCausalLM:
    - Qwen3-Omni-* (multimodal MoE)
    - Qwen2.5-Omni-* (multimodal)
    - Qwen3-VL-* (vision-language)

    Standard Qwen text models work with TransformersLoader.
    """

    # Patterns that indicate a model needs the Qwen loader
    QWEN_MULTIMODAL_PATTERNS: ClassVar[list[str]] = [
        "qwen/qwen3-omni",
        "qwen/qwen2.5-omni",
        "qwen/qwen2_5-omni",
        "qwen/qwen3-vl",
        "qwen/qwen2.5-vl",
        "qwen/qwen2_5-vl",
        "qwen/qwen2-vl",
    ]

    @property
    def name(self) -> str:
        return "qwen"

    def can_load(self, model_id: str) -> bool:
        """Check if this loader should handle the model.

        Returns True for Qwen multimodal models that need special handling.
        """
        model_lower = model_id.lower()

        for pattern in self.QWEN_MULTIMODAL_PATTERNS:
            if pattern in model_lower:
                return True

        return False

    def _get_model_class(self, model_type: str) -> Any:
        """Get the appropriate model class for a Qwen model type."""
        import transformers

        class_name = QWEN_MODEL_CLASSES.get(model_type)
        if class_name is None:
            raise ValueError(
                f"Unknown Qwen model type: {model_type}. "
                f"Supported types: {list(QWEN_MODEL_CLASSES.keys())}"
            )

        if not hasattr(transformers, class_name):
            raise ImportError(
                f"Model class {class_name} not found in transformers. "
                "You may need to update transformers: pip install -U transformers"
            )

        return getattr(transformers, class_name)

    def _get_hidden_size_and_layers(self, config: Any) -> tuple[int, int]:
        """Extract hidden size and num layers from nested Qwen config.

        Qwen multimodal models have nested configs:
        - qwen3_omni_moe: config.thinker_config.text_config
        - qwen2_5_omni: config.thinker_config.text_config
        - qwen3_vl: config.text_config
        """
        # Try nested paths in order of specificity
        config_paths = [
            ("thinker_config", "text_config"),  # Omni models
            ("text_config",),  # VL models
            (),  # Direct config
        ]

        for path in config_paths:
            cfg = config
            try:
                for attr in path:
                    cfg = getattr(cfg, attr)
                hidden_size = getattr(cfg, "hidden_size", None)
                num_layers = getattr(cfg, "num_hidden_layers", None)
                if hidden_size is not None and num_layers is not None:
                    return hidden_size, num_layers
            except AttributeError:
                continue

        # Fallback defaults
        logger.warning("Could not find hidden_size/num_layers in config, using defaults")
        return 4096, 32

    def load(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        quantization: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a Qwen multimodal model.

        Args:
            model_id: HuggingFace model ID
            device: Device to load on (only cuda:0 supported for large models)
            dtype: Data type (auto, float16, bfloat16, float32)
            trust_remote_code: Allow remote code
            quantization: Quantization mode (4bit, 8bit)
            **kwargs: Additional arguments

        Returns:
            LoadedModel with model and tokenizer
        """
        logger.info(f"Loading Qwen model {model_id} on {device} with dtype={dtype}")
        start_time = time.time()

        # Resolve device and dtype
        torch_device = torch.device(device)
        torch_dtype = resolve_dtype(dtype, torch_device)

        # Load config to determine model type
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model_type = getattr(config, "model_type", "unknown")

        logger.info(f"Detected Qwen model type: {model_type}")

        # Get the appropriate model class
        model_class = self._get_model_class(model_type)

        # Load tokenizer (standard AutoTokenizer works for Qwen)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

        # Ensure padding token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build model loading kwargs
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        # Handle quantization
        if quantization in ("4bit", "8bit"):
            model_kwargs.update(self._get_bitsandbytes_config(quantization, torch_dtype))
            model_kwargs["device_map"] = "auto"
        else:
            # Force single GPU to avoid spreading across multiple GPUs
            model_kwargs["device_map"] = device

        # Merge with additional kwargs
        model_kwargs.update(kwargs)

        # Load the model with the specific class
        logger.info(f"Loading with {model_class.__name__}")
        model = model_class.from_pretrained(model_id, **model_kwargs)

        model.eval()

        # Extract hidden size and num layers from nested config
        hidden_size, num_layers = self._get_hidden_size_and_layers(config)

        load_time = time.time() - start_time

        # Determine actual device after loading (device_map="auto" may distribute model)
        device_map = getattr(model, "hf_device_map", None)
        if device_map and isinstance(device_map, dict):
            # Model is distributed; use first device as primary
            first_device = next(iter(device_map.values()))
            actual_device = torch.device(first_device)
            logger.debug(f"Model distributed across devices: {device_map}")
        else:
            actual_device = torch_device

        logger.info(
            f"Qwen model loaded in {load_time:.2f}s - "
            f"hidden_size={hidden_size}, num_layers={num_layers}, "
            f"model_type={model_type}, device={actual_device}"
        )

        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            device=actual_device,
            dtype=torch_dtype,
            hidden_size=hidden_size,
            num_layers=num_layers,
            loader_type=self.name,
            metadata={
                "load_time_seconds": load_time,
                "trust_remote_code": trust_remote_code,
                "model_type": model_type,
                "model_class": model_class.__name__,
                "device_map": getattr(model, "hf_device_map", None),
            },
        )

    def _get_bitsandbytes_config(
        self,
        mode: str,
        compute_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Get BitsAndBytes configuration for quantization."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for 4bit/8bit quantization. "
                "Install with: pip install bitsandbytes"
            ) from e

        if mode == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization with NF4")
        elif mode == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")
        else:
            raise ValueError(f"Unknown quantization mode: {mode}")

        return {"quantization_config": quantization_config}

    def generate(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        return_hidden_states: bool = True,
        hidden_state_layers: list[int] | None = None,
        return_attention: bool = False,
        return_full_sequence: bool = False,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate text with hidden state extraction.

        Args:
            loaded_model: Previously loaded model
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_hidden_states: Extract hidden states
            hidden_state_layers: Which layers (-1 = last)
            return_attention: Extract attention weights
            return_full_sequence: Return hidden states for all tokens
            top_p: Nucleus sampling probability
            do_sample: Use sampling vs greedy
            **kwargs: Additional generation args

        Returns:
            GenerationOutput with text and hidden states
        """
        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        device = loaded_model.device

        if hidden_state_layers is None:
            hidden_state_layers = [-1]

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_length = inputs.input_ids.shape[1]

        # Generation config
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "do_sample": do_sample and temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "output_hidden_states": return_hidden_states,
            "output_attentions": return_attention,
            "return_dict_in_generate": True,
            **kwargs,
        }

        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)  # type: ignore[operator]

        inference_time = time.time() - start_time

        # Extract generated tokens
        generated_ids = outputs.sequences[0, input_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract hidden states if requested
        hidden_states_dict: dict[int, torch.Tensor] | None = None
        sequence_hidden_states_dict: dict[int, torch.Tensor] | None = None

        if return_hidden_states and hasattr(outputs, "hidden_states"):
            hidden_states_dict = self._extract_hidden_states(
                outputs.hidden_states,
                hidden_state_layers,
                loaded_model.num_layers,
            )

            if return_full_sequence:
                sequence_hidden_states_dict = self._extract_sequence_hidden_states(
                    outputs.hidden_states,
                    hidden_state_layers,
                    loaded_model.num_layers,
                )

        # Extract attention if requested
        attention_dict: dict[int, torch.Tensor] | None = None
        if return_attention and hasattr(outputs, "attentions"):
            attention_dict = self._extract_attention(
                outputs.attentions,
                hidden_state_layers,
                loaded_model.num_layers,
            )

        return GenerationOutput(
            text=generated_text,
            token_ids=generated_ids,
            hidden_states=hidden_states_dict,
            attention_weights=attention_dict,
            sequence_hidden_states=sequence_hidden_states_dict,
            metadata={
                "inference_time_ms": inference_time * 1000,
                "tokens_generated": len(generated_ids),
                "tokens_per_second": len(generated_ids) / inference_time
                if inference_time > 0
                else 0,
                "input_tokens": input_length,
                "temperature": temperature,
                "model_id": loaded_model.model_id,
                "full_sequence": return_full_sequence,
                "loader": self.name,
            },
        )

    def _extract_hidden_states(
        self,
        hidden_states: tuple,
        layers: list[int],
        num_layers: int,
    ) -> dict[int, torch.Tensor]:
        """
        Extract the last-token hidden-state vectors for specified layers from generation output.

        Qwen multimodal models follow the standard HuggingFace hidden_states format,
        identical to TransformersLoader. The hidden_states tuple structure is:
        - Outer tuple: one entry per generation step (each new token)
        - Inner tuple: (num_layers + 1) tensors - embedding layer at index 0,
          then transformer layers 1 through num_layers
        - Each tensor: shape [batch, seq_len, hidden_size]

        Parameters:
            hidden_states: Generation output hidden states: tuple[step][layer]
            layers: List of layer indices to extract (negative indices supported)
            num_layers: Number of model layers (for negative index resolution)

        Returns:
            dict mapping layer index to tensor of shape [batch, hidden_size]
        """
        result: dict[int, torch.Tensor] = {}

        if not hidden_states:
            return result

        # Get the final generation step's hidden states.
        # The last step (-1) contains hidden states after generating the final token.
        final_step = hidden_states[-1]

        for layer_idx in layers:
            # Convert negative indices: use num_layers + 1 (not num_layers) because
            # the tuple includes the embedding layer at index 0. So for a 32-layer model:
            #   - Index 0 = embedding output
            #   - Index 1-32 = transformer layer outputs
            #   - Index -1 resolves to 32 (last transformer layer)
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + 1 + layer_idx

            if 0 <= actual_idx < len(final_step):
                # Extract last token's hidden state from this layer.
                # Tensor shape: [batch, seq_len, hidden_size]
                # Slice [:, -1, :] selects all batches, last sequence position (the
                # newly generated token), and all hidden dimensions.
                # Result shape: [batch, hidden_size]
                # .cpu() transfers tensor from GPU to CPU for serialization/storage.
                layer_hidden = final_step[actual_idx][:, -1, :].cpu()
                result[layer_idx] = layer_hidden

        return result

    def _extract_attention(
        self,
        attentions: tuple,
        layers: list[int],
        num_layers: int,
    ) -> dict[int, torch.Tensor]:
        """
        Extract the attention weights for the requested layers from generation output.

        CRITICAL INDEXING DIFFERENCE from _extract_hidden_states():
        - hidden_states tuple has (num_layers + 1) entries: embedding layer at index 0,
          then transformer layers 1 through num_layers
        - attention tuple has only num_layers entries: NO embedding layer, just
          transformer layers 0 through (num_layers - 1)

        Therefore negative index resolution differs:
        - hidden_states: actual_idx = num_layers + 1 + layer_idx  (e.g., -1 -> 32 for 32-layer model)
        - attention:     actual_idx = num_layers + layer_idx      (e.g., -1 -> 31 for 32-layer model)

        The attentions tuple structure is:
        - Outer tuple: one entry per generation step (each new token)
        - Inner tuple: num_layers tensors (NO embedding layer, unlike hidden_states)
        - Each tensor: shape [batch, num_heads, query_seq_len, key_seq_len]

        Parameters:
            attentions: Generation output attention weights: tuple[step][layer]
            layers: List of layer indices to extract (negative indices supported)
            num_layers: Number of model layers (for negative index resolution)

        Returns:
            dict mapping layer index to tensor of shape [batch, num_heads, key_seq_len]
        """
        result: dict[int, torch.Tensor] = {}

        if not attentions:
            return result

        # Get the final generation step's attention weights
        final_step = attentions[-1]

        for layer_idx in layers:
            # Convert negative indices: use num_layers (NOT num_layers + 1) because
            # attention tuple has no embedding layer entry. For a 32-layer model:
            #   - Index 0-31 = attention weights for transformer layers 1-32
            #   - Index -1 resolves to 31 (last transformer layer's attention)
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx

            if 0 <= actual_idx < len(final_step):
                # Attention tensor shape: [batch, num_heads, query_seq_len, key_seq_len]
                # Slice [:, :, -1, :] selects all batches, all heads, last query position
                # (the newly generated token attending to all previous positions),
                # and all key positions.
                # Result shape: [batch, num_heads, key_seq_len]
                # .cpu() transfers tensor from GPU to CPU for serialization/storage.
                layer_attn = final_step[actual_idx][:, :, -1, :].cpu()
                result[layer_idx] = layer_attn

        return result

    def _extract_sequence_hidden_states(
        self,
        hidden_states: tuple,
        layers: list[int],
        num_layers: int,
    ) -> dict[int, torch.Tensor]:
        """Extract hidden states for ALL generated tokens (manifold construction).

        This creates the full geometric representation of the generation -
        each token's position in the model's semantic space, forming the
        "boundary object" manifold. This is valuable for analyzing the
        trajectory of meaning as generation unfolds.

        The hidden_states tuple from generate() is structured as:
        - Tuple of generation steps (one per token generated)
        - Each step has tuple of (num_layers + 1) tensors
        - Each tensor is [batch, seq_len, hidden_size]

        We extract the last token position from each step, giving us
        the newly generated token's representation at each step.

        Parameters:
            hidden_states: Generation output hidden states: tuple[step][layer]
            layers: List of layer indices to extract (negative indices supported)
            num_layers: Number of model layers (for negative index resolution)

        Returns:
            dict mapping layer_idx to tensor of shape [num_tokens, hidden_size]
        """
        result: dict[int, torch.Tensor] = {}

        if not hidden_states:
            return result

        for layer_idx in layers:
            # Convert negative indices using num_layers + 1 (same as _extract_hidden_states)
            # because hidden_states tuple includes embedding layer at index 0.
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + 1 + layer_idx

            # Build manifold: collect one hidden vector per generation step.
            # Each step represents one newly generated token's position in
            # the model's semantic space. Together they trace out the
            # "boundary object" geometry of the full generation.
            step_vectors = []
            for step in hidden_states:
                if 0 <= actual_idx < len(step):
                    # Extract the newly generated token's hidden state from this step.
                    # Slice [0, -1, :] selects:
                    #   - 0: first batch (single input assumption)
                    #   - -1: last sequence position (the token just generated)
                    #   - :: all hidden dimensions
                    # Result shape: [hidden_size] (1D vector for this token)
                    # .cpu() transfers to CPU for collection/serialization.
                    token_hidden = step[actual_idx][0, -1, :].cpu()
                    step_vectors.append(token_hidden)

            if step_vectors:
                # Stack all token vectors into a single matrix.
                # Final shape: [num_tokens, hidden_size] where num_tokens = number
                # of generation steps = number of tokens generated.
                # This matrix represents the manifold/trajectory through semantic space.
                sequence_tensor = torch.stack(step_vectors, dim=0)
                result[layer_idx] = sequence_tensor

        return result

    def embed(
        self,
        loaded_model: LoadedModel,
        text: str,
        pooling: str = "last_token",
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """
        Compute an embedding for the given text using the provided loaded model and pooling strategy.

        The default "last_token" pooling is recommended for decoder-only models because the final
        token's hidden state accumulates context from all previous tokens via causal attention.

        Parameters:
            loaded_model (LoadedModel): Loaded model container with `.model`, `.tokenizer`, `.device`.
            text (str): Input text to embed.
            pooling (str): Pooling strategy to reduce token-level hidden states to a single vector:
                - "last_token": Use the last non-padding token's hidden state (recommended).
                - "mean": Mean-pool hidden states across non-padding tokens.
                - "first_token": Use the first token's hidden state.

        Returns:
            EmbeddingOutput: Contains:
                - embedding: CPU tensor of the resulting embedding (batch dimension removed).
                - shape: Shape of the embedding tensor.
                - metadata: Dict with keys including "pooling", "inference_time_ms", etc.

        Raises:
            ValueError: If an unknown pooling strategy is provided.
        """
        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        device = loaded_model.device

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        inference_time = time.time() - start_time

        # Get last layer hidden states: [batch, seq_len, hidden_size]
        # Index -1 retrieves the final transformer layer's output (best for embeddings).
        last_hidden = outputs.hidden_states[-1]

        # Apply pooling strategy to reduce token-level representations to a single vector.
        # For decoder-only models, last_token is preferred because the final position
        # has "seen" all previous tokens via causal attention, accumulating full context.
        attention_mask = inputs.attention_mask
        if pooling == "last_token":
            # Use attention_mask to find the actual last token (not padding).
            # attention_mask is 1 for real tokens, 0 for padding.
            # sum(dim=1) gives sequence lengths; subtract 1 to get 0-indexed position.
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            # Advanced indexing: select [batch_i, seq_lengths[batch_i], :] for each batch.
            # This correctly handles variable-length sequences in the same batch.
            embedding = last_hidden[torch.arange(batch_size, device=device), seq_lengths]
        elif pooling == "mean":
            # Mean pooling: average all non-padding token representations.
            # Expand mask to [batch, seq, 1] for broadcasting with hidden states.
            mask = attention_mask.unsqueeze(-1)
            # Sum hidden states weighted by mask, divide by number of real tokens.
            # clamp(min=1) prevents division by zero for edge cases.
            mask_sum = mask.sum(dim=1).clamp(min=1)
            embedding = (last_hidden * mask).sum(dim=1) / mask_sum
        elif pooling == "first_token":
            # First token (often [CLS] or BOS) - mainly for encoder-style models.
            embedding = last_hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}. Use: last_token, mean, first_token")

        # Transfer embedding to CPU for storage/serialization.
        embedding = embedding.cpu()

        return EmbeddingOutput(
            # squeeze(0) removes the batch dimension [1, hidden_size] -> [hidden_size]
            # for single-input API consistency. The caller expects a 1D embedding vector.
            embedding=embedding.squeeze(0),
            shape=tuple(embedding.shape),
            metadata={
                "pooling": pooling,
                "inference_time_ms": inference_time * 1000,
                "input_tokens": inputs.input_ids.shape[1],
                "model_id": loaded_model.model_id,
                "loader": self.name,
            },
        )

    def generate_stream(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        return_hidden_states: bool = False,
        hidden_state_layers: list[int] | None = None,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> Iterator[StreamingToken | StreamingOutput]:
        """Generate text with streaming output.

        Falls back to non-streaming generation and yields tokens.
        Note: This provides buffered pseudo-streaming, not true real-time
        token streaming like TransformersLoader.generate_stream.
        """
        logger.debug("Qwen generate_stream uses buffered output, not true streaming")
        # Use non-streaming generate and yield tokens
        output = self.generate(
            loaded_model=loaded_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            return_hidden_states=return_hidden_states,
            hidden_state_layers=hidden_state_layers,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs,
        )

        # Yield tokens one at a time
        for i, token_id in enumerate(output.token_ids):
            token_text = loaded_model.tokenizer.decode([token_id])
            is_last = i == len(output.token_ids) - 1
            yield StreamingToken(
                token=token_text,
                token_id=token_id,
                is_finished=is_last,
                finish_reason="stop" if is_last else None,
            )

        # Yield final output
        yield StreamingOutput(
            text=output.text,
            token_ids=output.token_ids,
            token_count=len(output.token_ids),
            hidden_states=output.hidden_states,
            metadata=output.metadata,
        )
