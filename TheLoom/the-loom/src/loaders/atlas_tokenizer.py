"""Pruned mT5 tokenizer wrapper for Atlas model.

This module provides a tokenizer wrapper for the pruned mT5 tokenizer used
with the Atlas model. The vocabulary has been reduced from 250k to ~29k tokens
for efficient training on Shakespeare/de Vega corpora.

The tokenizer wraps HuggingFace AutoTokenizer with vocabulary remapping support
for the pruned token set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Path to bundled tokenizer assets
BUNDLED_TOKENIZER_DIR = (
    Path(__file__).parent.parent / "assets" / "tokenizers" / "mt5_shakespeare_pruned"
)


class PrunedTokenizer:
    """Wrapper for pruned mT5 tokenizer with vocabulary remapping.

    This class wraps a HuggingFace tokenizer that has been pruned from the
    original mT5 vocabulary (~250k tokens) to a much smaller set (~29k tokens)
    optimized for Shakespeare and classical literature training.

    The vocab_mapping.json contains:
    - original_vocab_size: Size of original mT5 vocabulary (250100)
    - pruned_vocab_size: Size of pruned vocabulary (~29k)
    - old_to_new_mapping: Dict mapping original token IDs to new IDs

    Attributes:
        tokenizer: The underlying HuggingFace tokenizer.
        vocab_mapping: Dict containing vocabulary remapping information.
        original_vocab_size: Size of original mT5 vocabulary.
        pruned_vocab_size: Size of pruned vocabulary.
        old_to_new: Mapping from original token IDs to pruned IDs.
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        vocab_mapping_path: str | Path | None = None,
    ) -> None:
        """Initialize the pruned tokenizer.

        Args:
            tokenizer_path: Path to the HuggingFace tokenizer directory.
            vocab_mapping_path: Optional path to vocab_mapping.json file.
                If not provided, looks for it in the tokenizer_path directory.
        """
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=False,
        )

        # Load vocabulary mapping
        if vocab_mapping_path is None:
            vocab_mapping_path = Path(tokenizer_path) / "vocab_mapping.json"

        vocab_mapping_path = Path(vocab_mapping_path)
        if vocab_mapping_path.exists():
            with open(vocab_mapping_path) as f:
                self.vocab_mapping: dict[str, Any] = json.load(f)

            self.original_vocab_size: int = self.vocab_mapping.get(
                "original_vocab_size", 250100
            )
            self.pruned_vocab_size: int = self.vocab_mapping.get(
                "pruned_vocab_size", len(self.tokenizer)
            )
            # Old ID (string) -> New ID (int) mapping
            self.old_to_new: dict[str, int] = self.vocab_mapping.get(
                "old_to_new_mapping", {}
            )
            logger.debug(
                f"Loaded vocab mapping: {self.original_vocab_size} -> "
                f"{self.pruned_vocab_size} tokens"
            )
        else:
            self.vocab_mapping = {}
            self.original_vocab_size = len(self.tokenizer)
            self.pruned_vocab_size = len(self.tokenizer)
            self.old_to_new = {}
            logger.warning(
                f"No vocab_mapping.json found at {vocab_mapping_path}, "
                "using tokenizer vocabulary as-is"
            )

    @classmethod
    def from_bundled(cls) -> PrunedTokenizer:
        """Load the bundled pruned mT5 tokenizer.

        This method loads the tokenizer from the bundled assets directory
        included with the TheLoom package.

        Returns:
            PrunedTokenizer: Initialized tokenizer with bundled assets.

        Raises:
            FileNotFoundError: If bundled tokenizer assets are not found.
        """
        if not BUNDLED_TOKENIZER_DIR.exists():
            raise FileNotFoundError(
                f"Bundled tokenizer not found at {BUNDLED_TOKENIZER_DIR}. "
                "Ensure tokenizer assets are properly installed."
            )

        vocab_mapping_path = BUNDLED_TOKENIZER_DIR / "vocab_mapping.json"

        return cls(
            tokenizer_path=BUNDLED_TOKENIZER_DIR,
            vocab_mapping_path=vocab_mapping_path if vocab_mapping_path.exists() else None,
        )

    @classmethod
    def from_path(cls, tokenizer_path: str | Path) -> PrunedTokenizer:
        """Load tokenizer from a custom path.

        Args:
            tokenizer_path: Path to tokenizer directory containing
                tokenizer files and optionally vocab_mapping.json.

        Returns:
            PrunedTokenizer: Initialized tokenizer.
        """
        return cls(tokenizer_path=tokenizer_path)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> list[int]:
        """Encode text to token IDs using the pruned vocabulary.

        Args:
            text: Input text to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.
            **kwargs: Additional arguments passed to tokenizer.encode().

        Returns:
            List of token IDs in the pruned vocabulary space.
        """
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.
            **kwargs: Additional arguments passed to tokenizer.decode().

        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def __call__(
        self,
        text: str | list[str],
        padding: bool | str = False,
        truncation: bool | str = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Tokenize text with full options (mirrors HuggingFace tokenizer).

        Args:
            text: Input text or list of texts to tokenize.
            padding: Padding strategy ('longest', 'max_length', or bool).
            truncation: Truncation strategy (True, 'longest_first', etc.).
            max_length: Maximum sequence length.
            return_tensors: Return type ('pt' for PyTorch, 'np' for NumPy, None for list).
            **kwargs: Additional arguments passed to tokenizer.

        Returns:
            Tokenized output (BatchEncoding if return_tensors specified).
        """
        return self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

    def remap_to_pruned(self, original_ids: list[int]) -> list[int]:
        """Remap token IDs from original vocabulary to pruned vocabulary.

        This is useful when processing data that was tokenized with the
        original mT5 tokenizer but needs to be used with the pruned model.

        Args:
            original_ids: Token IDs in the original (unpruned) vocabulary.

        Returns:
            Token IDs remapped to the pruned vocabulary.
            Tokens not in the pruned vocabulary are mapped to UNK.
        """
        if not self.old_to_new:
            return original_ids

        unk_id = self.tokenizer.unk_token_id or 2  # Default UNK ID
        return [
            self.old_to_new.get(str(orig_id), unk_id)
            for orig_id in original_ids
        ]

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int | None:
        """Return the padding token ID."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token ID."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        """Return the beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id

    @property
    def unk_token_id(self) -> int | None:
        """Return the unknown token ID."""
        return self.tokenizer.unk_token_id

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.tokenizer)
