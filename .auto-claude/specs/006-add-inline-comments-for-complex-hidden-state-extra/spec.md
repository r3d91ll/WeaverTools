# Add inline comments for complex hidden state extraction logic

## Overview

The hidden state extraction in TheLoom loaders (transformers_loader.py, qwen_loader.py, mistral_loader.py) involves complex logic for extracting tensors from different model architectures. Key algorithms lack inline comments explaining: tensor indexing, layer selection, bfloat16 handling, and batch dimension squeezing.

## Rationale

Hidden state extraction is the core value proposition of TheLoom. Different model architectures store hidden states differently, requiring loader-specific logic. Current code handles edge cases (bfloat16→float32 conversion, empty tensors, shape variations) but doesn't explain why. Contributors adding new loaders need to understand these patterns.

---
*This spec was created from ideation and is pending detailed specification.*
