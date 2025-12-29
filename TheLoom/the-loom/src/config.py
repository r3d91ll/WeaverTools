"""Configuration management for Research Model Server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Valid precision modes for MemoryConfig
VALID_PRECISION_MODES = frozenset({"auto", "fp32", "fp16", "bf16"})


@dataclass
class PrecisionValidationResult:
    """Result from precision validation with GPU capability detection."""

    is_valid: bool
    resolved_precision: str
    gpu_available: bool
    bf16_supported: bool
    compute_capability: tuple[int, int] | None
    warnings: list[str]


class ServerConfig(BaseModel):
    """Server transport configuration."""

    transport: str = Field(default="http", description="Transport type: http, unix, or both")
    http_host: str = Field(default="0.0.0.0", description="HTTP server host")
    http_port: int = Field(default=8080, description="HTTP server port")
    unix_socket: str = Field(default="/tmp/loom.sock", description="Unix socket path")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Use ['*'] for development, specific origins for production",
    )
    cors_allow_credentials: bool = Field(
        default=False,
        description="Allow credentials in CORS requests. Should be False when origins is ['*']",
    )


class GPUConfig(BaseModel):
    """GPU device configuration."""

    devices: list[int] = Field(default=[0], description="Available CUDA device indices")
    default_device: int = Field(default=0, description="Default CUDA device")
    memory_fraction: float = Field(
        default=0.9, ge=0.1, le=1.0, description="Max GPU memory fraction to use"
    )


class HiddenStatesConfig(BaseModel):
    """Hidden state extraction configuration."""

    default_layers: list[int] = Field(
        default=[-1], description="Which layers to return by default (-1 = last)"
    )
    include_attention: bool = Field(
        default=False, description="Include attention weights in response"
    )
    precision: str = Field(
        default="float32", description="Precision for hidden states: float16, float32, bfloat16"
    )


class ModelsConfig(BaseModel):
    """Model loading and caching configuration."""

    cache_dir: str = Field(
        default="~/.cache/huggingface", description="HuggingFace cache directory"
    )
    preload: list[str] = Field(default=[], description="Models to load at startup")
    max_loaded: int = Field(default=3, ge=1, description="Maximum concurrent models in memory")
    default_dtype: str = Field(
        default="auto", description="Default dtype: auto, float16, bfloat16, float32"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow remote code execution for custom architectures (security risk - enable only for trusted models)",
    )
    auto_unload_minutes: int = Field(
        default=20,
        ge=0,
        description="Auto-unload models after this many minutes of inactivity (0 = disabled)",
    )


class LoadersConfig(BaseModel):
    """Loader-specific configuration."""

    # Loader priority order for auto-detection
    fallback_order: list[str] = Field(
        default=["transformers", "sentence_transformers", "custom"],
        description="Order to try loaders during auto-detection",
    )

    # TransformersLoader settings
    transformers_default_kwargs: dict[str, Any] = Field(
        default={},
        description="Default kwargs passed to TransformersLoader.load()",
    )

    # SentenceTransformersLoader settings
    sentence_transformers_default_kwargs: dict[str, Any] = Field(
        default={},
        description="Default kwargs passed to SentenceTransformersLoader.load()",
    )

    # CustomLoader settings
    custom_default_kwargs: dict[str, Any] = Field(
        default={},
        description="Default kwargs passed to CustomLoader.load()",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    file: str | None = Field(default=None, description="Log file path (None for stdout only)")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )


class PatchingConfig(BaseModel):
    """Activation patching configuration for causal intervention studies."""

    enabled: bool = Field(
        default=True,
        description="Enable activation patching functionality",
    )
    default_layers: list[int] = Field(
        default=[],
        description="Default layers to target for patching (empty = all layers)",
    )
    default_components: list[str] = Field(
        default=["resid_pre"],
        description="Default activation components to patch: resid_pre, resid_post, attn, mlp",
    )
    cache_dir: str = Field(
        default="~/.cache/loom/activations",
        description="Directory for storing activation caches",
    )
    max_cache_size_mb: int = Field(
        default=4096,
        ge=256,
        description="Maximum activation cache size in megabytes",
    )
    fold_layer_norm: bool = Field(
        default=False,
        description="Fold LayerNorm into weights (False preserves exact model behavior)",
    )
    cleanup_on_completion: bool = Field(
        default=True,
        description="Automatically clean up activation caches after experiments",
    )
    validate_hook_shapes: bool = Field(
        default=True,
        description="Validate that hook outputs match expected shapes",
    )
    stream_activations: bool = Field(
        default=False,
        description="Stream activations to disk for large models (reduces memory usage)",
    )


class PersistenceConfig(BaseModel):
    """Experiment result persistence configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable automatic persistence of experiment results",
    )
    db_path: str = Field(
        default="~/.local/share/loom/experiments.db",
        description="Path to SQLite database for experiment metadata",
    )
    hidden_states_dir: str = Field(
        default="~/.local/share/loom/hidden_states",
        description="Directory for storing hidden state snapshots in HDF5 format",
    )
    export_dir: str = Field(
        default="~/.local/share/loom/exports",
        description="Directory for exported experiment data (CSV, JSON, Parquet)",
    )
    compression: str | None = Field(
        default="gzip",
        description="Compression algorithm for hidden states: 'gzip', 'lzf', or None (no compression)",
    )
    compression_level: int = Field(
        default=4,
        ge=0,
        le=9,
        description="Compression level (0-9, higher = better compression but slower; 0 = no compression for gzip)",
    )
    auto_persist: bool = Field(
        default=True,
        description="Automatically persist results after each generation request",
    )
    chunk_size: int = Field(
        default=1024,
        ge=64,
        description="Chunk size for HDF5 dataset storage (affects compression efficiency)",
    )


class MemoryConfig(BaseModel):
    """Memory optimization configuration for GPU memory management.

    This configuration controls memory-efficient inference and training options
    including precision modes, gradient checkpointing, streaming extraction,
    and TransformerLens selective activation caching.
    """

    precision_mode: str = Field(
        default="auto",
        description="Precision mode for inference: auto, fp32, fp16, bf16. "
        "Auto detects GPU capability and selects optimal precision. "
        "BF16 requires Ampere+ GPU (compute capability 8.0+).",
    )
    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing for training scenarios. "
        "Only beneficial when backward passes are performed (training). "
        "Disabled by default for inference-only workloads.",
    )
    streaming_chunk_size: int = Field(
        default=512,
        ge=1,
        description="Chunk size for streaming hidden state extraction. "
        "Smaller values reduce peak memory but increase latency.",
    )
    memory_warning_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="GPU memory utilization threshold for proactive warnings. "
        "Default 85% triggers warning before OOM conditions.",
    )
    activation_cache_filter: list[str] = Field(
        default=[],
        description="List of TransformerLens hook names to cache. "
        "Empty list caches all activations (default behavior, high memory). "
        "Example: ['blocks.0.hook_resid_post', 'blocks.0.attn.hook_pattern']. "
        "Use selective caching to reduce memory from 2-3x overhead to <20%.",
    )

    def validate_precision(self, device: int | None = None) -> PrecisionValidationResult:
        """
        Validate precision mode against GPU capabilities.

        Checks if the configured precision_mode is valid and supported by the
        available GPU hardware. BF16 requires Ampere+ GPU (compute capability 8.0+).

        Parameters:
            device (int | None): CUDA device index to check. Defaults to 0 if CUDA
                is available, None otherwise.

        Returns:
            PrecisionValidationResult: Validation result containing:
                - is_valid: True if precision mode is valid and supported
                - resolved_precision: The actual precision that will be used
                - gpu_available: Whether CUDA is available
                - bf16_supported: Whether bf16 is supported on the GPU
                - compute_capability: GPU compute capability tuple or None
                - warnings: List of warning messages about precision selection
        """
        warnings: list[str] = []
        gpu_available = torch.cuda.is_available()
        bf16_supported = False
        compute_capability: tuple[int, int] | None = None

        # Validate precision_mode is a known value
        if self.precision_mode not in VALID_PRECISION_MODES:
            return PrecisionValidationResult(
                is_valid=False,
                resolved_precision=self.precision_mode,
                gpu_available=gpu_available,
                bf16_supported=bf16_supported,
                compute_capability=compute_capability,
                warnings=[
                    f"Invalid precision_mode '{self.precision_mode}'. "
                    f"Valid options: {sorted(VALID_PRECISION_MODES)}"
                ],
            )

        # Detect GPU capabilities
        if gpu_available:
            device_idx = device if device is not None else 0
            try:
                compute_capability = torch.cuda.get_device_capability(device_idx)
                # BF16 requires Ampere+ (compute capability 8.0+)
                bf16_supported = compute_capability[0] >= 8
            except (RuntimeError, AssertionError):
                # Device not available or invalid index
                compute_capability = None
                bf16_supported = False

        # Resolve "auto" precision based on GPU capability
        if self.precision_mode == "auto":
            if gpu_available and bf16_supported:
                resolved_precision = "bf16"
            elif gpu_available:
                resolved_precision = "fp16"
            else:
                resolved_precision = "fp32"
        else:
            resolved_precision = self.precision_mode

        # Check if requested precision is supported
        is_valid = True
        if resolved_precision == "bf16" and not bf16_supported:
            if gpu_available:
                warnings.append(
                    f"BF16 requested but GPU compute capability {compute_capability} < 8.0. "
                    "BF16 requires Ampere+ GPU. Consider using 'fp16' or 'auto'."
                )
            else:
                warnings.append(
                    "BF16 requested but no CUDA GPU available. "
                    "Consider using 'fp32' for CPU or 'auto' for automatic selection."
                )
            # Still valid as torch will handle the fallback, but warn
            is_valid = True

        if resolved_precision in ("fp16", "bf16") and not gpu_available:
            warnings.append(
                f"'{resolved_precision}' precision works best on GPU. "
                "No CUDA device detected."
            )

        return PrecisionValidationResult(
            is_valid=is_valid,
            resolved_precision=resolved_precision,
            gpu_available=gpu_available,
            bf16_supported=bf16_supported,
            compute_capability=compute_capability,
            warnings=warnings,
        )

class Config(BaseSettings):
    """Main configuration for The Loom."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    hidden_states: HiddenStatesConfig = Field(default_factory=HiddenStatesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    loaders: LoadersConfig = Field(default_factory=LoadersConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    patching: PatchingConfig = Field(default_factory=PatchingConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # Model-specific overrides (loader, dtype, device, etc.)
    model_overrides: dict[str, dict[str, Any]] = Field(
        default={},
        description="Per-model configuration overrides. Keys are model IDs, values are dicts with: loader, dtype, device, etc.",
    )

    model_config = SettingsConfigDict(
        env_prefix="LOOM_",
        env_nested_delimiter="__",
    )


def find_config_file() -> Path | None:
    """
    Locate a Loom configuration file by searching common filesystem locations in order.
    
    Searches these locations (in order) and returns the first path that exists:
    - ./config.yaml
    - ./config.yml
    - ./loom.yaml
    - ~/.config/loom/config.yaml
    - ~/.config/loom/config.yml
    - /etc/loom/config.yaml
    
    Returns:
        Path | None: The path to the first existing configuration file, or `None` if no file is found.
    """
    search_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path.cwd() / "loom.yaml",
        Path.home() / ".config" / "loom" / "config.yaml",
        Path.home() / ".config" / "loom" / "config.yml",
        Path("/etc/loom/config.yaml"),
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from file and environment.

    Priority (highest to lowest):
    1. Environment variables (LOOM_*)
    2. Specified config file
    3. Auto-discovered config file
    4. Default values
    """
    config_data: dict[str, Any] = {}

    # Find config file
    if config_path is None:
        config_path = find_config_file()
    elif isinstance(config_path, str):
        config_path = Path(config_path)

    # Load from YAML if found
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)
            if yaml_data:
                config_data = yaml_data

    # Create config with loaded data (env vars override via pydantic-settings)
    return Config(**config_data)


def get_model_config(config: Config, model_id: str) -> dict[str, Any]:
    """
    Compute the effective configuration for a given model by merging global defaults with any model-specific overrides.
    
    Parameters:
        config (Config): Global server configuration.
        model_id (str): Identifier of the model whose overrides should be applied.
    
    Returns:
        dict[str, Any]: Effective model configuration. Contains default keys such as `dtype`, `device`, and `loader`, with values replaced or extended by any entries from `config.model_overrides[model_id]` when present.
    """
    base_config = {
        "dtype": config.models.default_dtype,
        "device": f"cuda:{config.gpu.default_device}",
        "loader": "auto",
    }

    # Apply model-specific overrides
    if model_id in config.model_overrides:
        base_config.update(config.model_overrides[model_id])

    return base_config


# Global config instance (initialized lazily)
_config: Config | None = None


def get_config() -> Config:
    """
    Return the application's global configuration singleton.
    
    Returns:
        Config: The global Config instance; created and loaded on first access.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """
    Replace the module-level global configuration with the provided Config instance.
    
    This updates the internal singleton used by get_config() so subsequent calls return `config`.
    
    Parameters:
        config (Config): Configuration to install as the module-global instance.
    """
    global _config
    _config = config
