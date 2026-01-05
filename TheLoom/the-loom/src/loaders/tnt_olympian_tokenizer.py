"""Shakespeare BPE Tokenizer for TNT Olympian models.

This module provides a wrapper for the Shakespeare BPE tokenizer used by
TNT Olympian models. The tokenizer is trained on Shakespeare corpus and
provides a 16k vocabulary optimized for Elizabethan English.

The tokenizer files are stored in the Todd_Atlas repository under
`tokenizers/shakespeare/`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# Default path to Shakespeare tokenizer in the Atlas repository
# This is used when no explicit path is provided
DEFAULT_TOKENIZER_PATH = Path("/home/todd/olympus/models/Atlas/tokenizers/shakespeare")


class ShakespeareBPETokenizer:
    """Wrapper for Shakespeare BPE tokenizer from Todd_Atlas.

    Provides a HuggingFace-compatible interface for the custom BPE tokenizer
    trained on Shakespeare corpus. Used by TNT Olympian models for
    tokenization during inference and analysis.

    Attributes:
        tokenizer: Underlying HuggingFace tokenizers.Tokenizer instance.
        vocab_size: Size of the vocabulary.
        pad_token_id: ID of the padding token (if defined).
        eos_token_id: ID of the end-of-sequence token (if defined).
        bos_token_id: ID of the beginning-of-sequence token (if defined).

    Example:
        ```python
        tokenizer = ShakespeareBPETokenizer()
        ids = tokenizer.encode("To be, or not to be")
        text = tokenizer.decode(ids)
        ```
    """

    def __init__(
        self,
        tokenizer_path: str | Path | None = None,
    ) -> None:
        """Initialize Shakespeare BPE tokenizer.

        Args:
            tokenizer_path: Path to tokenizer directory containing tokenizer.json.
                           If None, uses DEFAULT_TOKENIZER_PATH.

        Raises:
            FileNotFoundError: If tokenizer.json not found at specified path.
            ImportError: If tokenizers library not installed.
        """
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "tokenizers library required for ShakespeareBPETokenizer. "
                "Install with: pip install tokenizers"
            ) from e

        if tokenizer_path is None:
            tokenizer_path = DEFAULT_TOKENIZER_PATH

        tokenizer_path = Path(tokenizer_path)
        tokenizer_file = tokenizer_path / "tokenizer.json"

        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"Shakespeare tokenizer not found at {tokenizer_file}. "
                f"Ensure Todd_Atlas repo has tokenizers/shakespeare/ directory."
            )

        logger.info(f"Loading Shakespeare BPE tokenizer from {tokenizer_file}")
        self.tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_file))

        # Cache special token IDs
        self._pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self._eos_token_id = self.tokenizer.token_to_id("[EOS]")
        self._bos_token_id = self.tokenizer.token_to_id("[BOS]")
        self._unk_token_id = self.tokenizer.token_to_id("[UNK]")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode.
            add_special_tokens: Whether to include BOS/EOS tokens.

        Returns:
            List of token IDs.
        """
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        if not add_special_tokens:
            # Filter out special tokens if requested
            special_ids = {self._bos_token_id, self._eos_token_id, self._pad_token_id}
            ids = [t for t in ids if t not in special_ids]

        return list(ids)

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens))

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[list[int]]]:
        """HuggingFace-compatible tokenization interface.

        Args:
            text: Text or list of texts to tokenize.
            return_tensors: If "pt", return PyTorch tensors.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.
            max_length: Maximum sequence length.

        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'.
        """
        if isinstance(text, str):
            text = [text]

        all_ids = [self.encode(t) for t in text]

        # Handle truncation
        if truncation and max_length:
            all_ids = [ids[:max_length] for ids in all_ids]

        # Handle padding
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            if max_length:
                max_len = min(max_len, max_length)
            pad_id = self._pad_token_id or 0
            all_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]

        result: dict[str, list[list[int]]] = {"input_ids": all_ids}

        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch

            result = {k: torch.tensor(v) for k, v in result.items()}  # type: ignore

        return result

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return int(self.tokenizer.get_vocab_size())

    @property
    def pad_token_id(self) -> int | None:
        """Return padding token ID."""
        result = self._pad_token_id
        return int(result) if result is not None else None

    @property
    def eos_token_id(self) -> int | None:
        """Return end-of-sequence token ID."""
        result = self._eos_token_id
        return int(result) if result is not None else None

    @property
    def bos_token_id(self) -> int | None:
        """Return beginning-of-sequence token ID."""
        result = self._bos_token_id
        return int(result) if result is not None else None

    @property
    def unk_token_id(self) -> int | None:
        """Return unknown token ID."""
        result = self._unk_token_id
        return int(result) if result is not None else None
