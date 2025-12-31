"""Training module for Atlas model with checkpoint saving and live monitoring.

This module provides training infrastructure for the Atlas model:
- AtlasTrainer: Full training loop with checkpoint saving
- Training utilities: Data loading, metric logging, resume capability

Phase 3 of Atlas Model Interpretability Integration.
"""

from .atlas_trainer import (
    AtlasTrainer,
    TrainingConfig,
    TrainingState,
    TrainingMetrics,
    CheckpointManager,
)

__all__ = [
    "AtlasTrainer",
    "TrainingConfig",
    "TrainingState",
    "TrainingMetrics",
    "CheckpointManager",
]
