# Memory Requirements for TheLoom

This document provides GPU memory requirements for running transformer models with TheLoom's hidden state extraction, along with optimization strategies to reduce memory usage.

## Overview

TheLoom's memory usage depends on several factors:
- **Model size** (number of parameters)
- **Precision mode** (FP32, FP16, BF16)
- **Activation caching** (full vs. selective)
- **Sequence length** (context window usage)
- **Batch size** (number of concurrent sequences)

TheLoom provides multiple optimization strategies to reduce memory usage:
1. **Mixed precision** - Use FP16/BF16 for 50% memory reduction per parameter
2. **Selective caching** - Cache only required activations instead of all layers
3. **Streaming extraction** - Process long sequences in chunks
4. **Memory monitoring** - Track usage and get warnings before OOM

## Memory Requirements by Model Size

The following table shows GPU VRAM requirements for different model sizes and precision modes. These estimates include model weights, KV cache for 2048 tokens, and activation memory for hidden state extraction.

### Model Weights Only

| Model Size | Parameters | FP32 (GB) | FP16 (GB) | BF16 (GB) | INT8 (GB) | INT4 (GB) |
|------------|------------|-----------|-----------|-----------|-----------|-----------|
| **3B**     | 3 billion  | 12        | 6         | 6         | 3         | 1.5       |
| **7B**     | 7 billion  | 28        | 14        | 14        | 7         | 3.5       |
| **13B**    | 13 billion | 52        | 26        | 26        | 13        | 6.5       |
| **30B**    | 30 billion | 120       | 60        | 60        | 30        | 15        |
| **70B**    | 70 billion | 280       | 140       | 140       | 70        | 35        |

### Total VRAM with Hidden State Extraction

These estimates include model weights + KV cache (2048 tokens) + activation cache for hidden state extraction:

| Model Size | FP32 Total | FP16 Total | BF16 Total | Recommended GPU |
|------------|------------|------------|------------|-----------------|
| **3B**     | 16 GB      | 8 GB       | 8 GB       | RTX 3080/4080 (10-16 GB) |
| **7B**     | 36 GB      | 18 GB      | 18 GB      | RTX 4090 (24 GB) / A10 |
| **13B**    | 68 GB      | 34 GB      | 34 GB      | A100 (40 GB) / 2x RTX 4090 |
| **30B**    | 160 GB     | 80 GB      | 80 GB      | A100 (80 GB) / H100 |
| **70B**    | 360 GB     | 180 GB     | 180 GB     | Multi-GPU (A100 80GB x3+) |

### Memory Breakdown

For a typical 7B model in FP16:

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| Model weights | 14 | 7B params × 2 bytes/param |
| KV cache (2K tokens) | 2 | Scales with sequence length |
| Activation cache (full) | 4-8 | All layer activations |
| Activation cache (selective) | 0.5-1 | Only requested hooks |
| Workspace/overhead | 1-2 | CUDA allocator overhead |
| **Total (full cache)** | **21-26** | Requires 24+ GB GPU |
| **Total (selective cache)** | **17-19** | Fits on 24 GB GPU |

## Precision Modes

TheLoom supports three precision modes for memory optimization:

### FP32 (Full Precision)
- **Memory**: Baseline (100%)
- **Compatibility**: All GPUs
- **Use case**: Maximum numerical accuracy, debugging
- **Tradeoff**: Highest memory usage

### FP16 (Half Precision)
- **Memory**: 50% of FP32
- **Compatibility**: All modern GPUs (CUDA compute capability 5.3+)
- **Use case**: Most inference workloads
- **Tradeoff**: Minor numerical precision loss, rarely affects results

### BF16 (Brain Float 16)
- **Memory**: 50% of FP32
- **Compatibility**: Ampere+ GPUs only (RTX 30xx, A100, H100) - compute capability 8.0+
- **Use case**: Training and inference with better numerical stability than FP16
- **Tradeoff**: Not available on older GPUs

### Precision Fallback Chain

When `precision_mode: auto` (default), TheLoom automatically selects the best precision:

```text
Request BF16 → Check GPU capability ≥ 8.0?
                    ├── Yes → Use BF16
                    └── No → Fallback to FP16 (with warning)

Request FP16 → Check GPU capability ≥ 5.3?
                    ├── Yes → Use FP16
                    └── No → Fallback to FP32 (with warning)
```

### GPU Compute Capability Reference

| GPU Family | Compute Capability | BF16 Support | FP16 Support |
|------------|-------------------|--------------|--------------|
| RTX 40xx (Ada) | 8.9 | Yes | Yes |
| RTX 30xx (Ampere) | 8.6 | Yes | Yes |
| A100 | 8.0 | Yes | Yes |
| H100 | 9.0 | Yes | Yes |
| RTX 20xx (Turing) | 7.5 | No | Yes |
| GTX 10xx (Pascal) | 6.1 | No | Yes |
| GTX 9xx (Maxwell) | 5.2 | No | Limited |

## Hardware Recommendations

### Entry-Level Research (7B models)

- **GPU**: RTX 4090 (24 GB) or RTX 3090 (24 GB)
- **Configuration**: FP16 + selective caching
- **Limits**: Single 7B model, moderate sequence lengths (4K tokens)

### Mid-Range Research (13B models)

- **GPU**: A100 40GB or 2x RTX 4090 with quantization
- **Configuration**: FP16/BF16 + selective caching
- **Limits**: 13B models, longer sequences (8K tokens)

### Large Model Research (30B+ models)

- **GPU**: A100 80GB, H100, or multi-GPU setup
- **Configuration**: BF16 + quantization + streaming
- **Limits**: 30B-70B models with full functionality

### Cloud GPU Options

| Provider | GPU | VRAM | Best For |
|----------|-----|------|----------|
| RunPod | RTX 4090 | 24 GB | 7B models |
| RunPod | A100-40G | 40 GB | 13B models |
| Lambda Labs | A100-80G | 80 GB | 30B models |
| AWS | p4d.24xlarge (8x A100) | 320 GB | 70B models |

## Configuration Guide

### Basic Memory Configuration

Add the following to your `config/default.yaml`:

```yaml
memory:
  # Precision mode: auto, fp32, fp16, bf16
  precision_mode: auto

  # Enable gradient checkpointing for training (not needed for inference)
  enable_gradient_checkpointing: false

  # Chunk size for streaming extraction (tokens)
  streaming_chunk_size: 512

  # Warn when GPU memory exceeds this threshold (0-1)
  memory_warning_threshold: 0.85

  # Selective caching: only cache these hook patterns
  # Empty list = cache all (default TransformerLens behavior)
  # Populated list = cache only specified hooks (memory efficient)
  activation_cache_filter: []
```

### Selective Activation Caching

Reduce memory by caching only the activations you need:

```yaml
memory:
  activation_cache_filter:
    # Only cache residual stream at specific layers
    - "blocks.0.hook_resid_post"
    - "blocks.5.hook_resid_post"
    - "blocks.11.hook_resid_post"
```

Common hook patterns:
- `blocks.{layer}.hook_resid_post` - Residual stream after layer
- `blocks.{layer}.hook_resid_pre` - Residual stream before layer
- `blocks.{layer}.attn.hook_pattern` - Attention patterns
- `blocks.{layer}.hook_mlp_out` - MLP output

### Precision Mode Selection

```yaml
# For maximum compatibility (works on any GPU)
memory:
  precision_mode: fp16

# For Ampere+ GPUs with better numerical stability
memory:
  precision_mode: bf16

# For maximum precision (debugging, validation)
memory:
  precision_mode: fp32

# Let TheLoom choose based on GPU capability
memory:
  precision_mode: auto
```

### Streaming for Long Sequences

Process sequences that exceed GPU memory:

```yaml
memory:
  streaming_chunk_size: 512  # Process 512 tokens at a time
```

Using streaming in code:

```python
from src.extraction.hidden_states import extract_streaming, collect_streaming_results

# Option 1: Process chunks one at a time
for chunk in extract_streaming(model, long_tokens, chunk_size=512):
    process_chunk(chunk.hidden_states)
    # Memory is freed after each chunk

# Option 2: Collect all chunks (uses more memory)
full_hidden_states = collect_streaming_results(
    extract_streaming(model, long_tokens, chunk_size=512)
)
```

### Memory Monitoring

Enable proactive warnings:

```yaml
memory:
  memory_warning_threshold: 0.85  # Warn at 85% usage
```

Using the GPUManager:

```python
from src.utils.gpu import GPUManager

gpu = GPUManager()

# Check if allocation is safe before large operations
if gpu.can_allocate(required_memory_gb=10.0, device=0):
    # Safe to proceed
    result = extract_hidden_states(model, tokens)
else:
    # Use streaming or reduce batch size
    result = extract_streaming(model, tokens, chunk_size=256)

# Monitor memory during extraction
status = gpu.check_memory_threshold(threshold=0.85)
for device_warning in status["warnings"]:
    print(f"Warning: GPU {device_warning['device']} at {device_warning['usage_percent']:.1f}%")
```

## Inference vs. Training Optimizations

Different optimizations apply to different workflows:

| Optimization | Inference | Training | Memory Savings |
|--------------|-----------|----------|----------------|
| Mixed precision (FP16/BF16) | Yes | Yes | 50% |
| Selective caching | Yes | Yes | 70-90% |
| Streaming extraction | Yes | Limited | Unbounded |
| Gradient checkpointing | No | Yes | 40-50% |
| `torch.inference_mode()` | Yes | No | 5-10% |

### Inference-Only Workflows

For pure inference (no backward pass), use these optimizations:

```python
from src.extraction.hidden_states import extract_inference_optimized

# Optimized for inference: disables gradients, uses inference mode
result = extract_inference_optimized(
    model=model,
    tokens=tokens,
    layers=[-1],  # Only extract last layer
    precision='fp16',
    clear_cache=True  # Clear CUDA cache after extraction
)
```

### Training Workflows

For training with backward passes, gradient checkpointing can help:

```python
from src.extraction.hidden_states import extract_with_checkpointing

# Trade 30% compute for 50% memory savings
result = extract_with_checkpointing(
    model=model,
    tokens=tokens,
    use_reentrant=True  # Required for some model architectures
)
```

Note: Gradient checkpointing only saves memory when there is a backward pass. For inference-only workloads, it adds overhead without benefit.

## Troubleshooting

### Common OOM Scenarios and Solutions

#### "CUDA out of memory" during model loading

**Cause**: Model weights exceed available GPU memory.

**Solutions**:
1. Use FP16/BF16 precision: `precision_mode: fp16`
2. Use quantization: `quantization: "4bit"` or `"8bit"`
3. Use a smaller model
4. Use a GPU with more VRAM

#### "CUDA out of memory" during hidden state extraction

**Cause**: Activation cache exceeds available memory.

**Solutions**:
1. Enable selective caching:
   ```yaml
   memory:
     activation_cache_filter:
       - "blocks.11.hook_resid_post"  # Only cache what you need
   ```
2. Reduce batch size
3. Use streaming for long sequences

#### "CUDA out of memory" with long sequences

**Cause**: KV cache and activations scale with sequence length.

**Solutions**:
1. Enable streaming:
   ```yaml
   memory:
     streaming_chunk_size: 512
   ```
2. Reduce `max_tokens` in generation
3. Process shorter sequences

#### Memory fragmentation (allocation fails despite available memory)

**Cause**: CUDA memory is fragmented from many small allocations.

**Solutions**:
1. Clear cache before large operations:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
2. Use `GPUManager.reset_peak_memory()` between operations
3. Restart the process if fragmentation persists

#### BF16 not working on GPU

**Cause**: GPU doesn't support BF16 (requires Ampere+).

**Solutions**:
1. Use `precision_mode: auto` for automatic fallback
2. Explicitly use FP16: `precision_mode: fp16`
3. Upgrade to an Ampere or newer GPU

#### vLLM using too much memory, leaving none for TheLoom

**Cause**: vLLM defaults to 90% GPU utilization.

**Solutions**:
1. Configure vLLM with lower memory fraction:
   ```python
   llm = LLM(model="...", gpu_memory_utilization=0.7)
   ```
2. Use `GPUManager.can_allocate()` to check available memory before TheLoom operations
3. Run TheLoom and vLLM on separate GPUs

### Checking Memory Status

```bash
# Check GPU memory usage
nvidia-smi

# In Python
python -c "
from src.utils.gpu import GPUManager
gpu = GPUManager()
gpu_info = gpu.get_gpu_info()
for info in (gpu_info if isinstance(gpu_info, list) else [gpu_info]):
    print(f'GPU {info.index}: {info.used_memory_gb:.1f}/{info.total_memory_gb:.1f} GB')
    print(f'  Peak: {info.peak_memory_gb:.1f} GB')
"
```

### Memory Debugging

Enable memory profiling (development only):

```python
import torch

# Track memory allocations
torch.cuda.memory._record_memory_history(enabled=True)

# Run your code
result = extract_hidden_states(model, tokens)

# Get memory snapshot
snapshot = torch.cuda.memory._snapshot()
# Analyze snapshot for leaks or unexpected allocations
```

Warning: Memory profiling has significant overhead. Disable in production.

## Best Practices

1. **Start with `precision_mode: auto`** - Let TheLoom choose the best precision for your GPU.

2. **Always use selective caching** for production - Only cache the hooks you actually need.

3. **Monitor memory during development** - Use `GPUManager.check_memory_threshold()` to catch issues early.

4. **Use streaming for unknown sequence lengths** - Prevents OOM with unexpectedly long inputs.

5. **Reserve 10-15% memory headroom** - Prevents fragmentation issues.

6. **Profile before optimizing** - Measure actual memory usage before applying optimizations.

7. **Test with production-sized inputs** - Memory issues often only appear with realistic data.

## References

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [TransformerLens Activation Caching](https://neelnanda-io.github.io/TransformerLens/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [CUDA Memory Fragmentation](https://pytorch.org/docs/stable/notes/cuda.html#memory-fragmentation)
