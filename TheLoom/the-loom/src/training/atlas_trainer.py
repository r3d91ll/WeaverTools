"""Atlas model training module with checkpoint saving and live monitoring.

This module implements the full Atlas training pipeline with:
- Data loading from text files or synthetic generation
- Forward/backward passes with gradient accumulation
- AdamW optimizer with learning rate scheduling
- Checkpoint saving every epoch with memory state preservation
- Metric logging for dashboard integration
- Resume capability from any checkpoint

Usage:
    poetry run python -m src.training.atlas_trainer --epochs 2 --checkpoint-dir /tmp/test_training/

References:
- Atlas model: TheLoom/the-loom/src/loaders/atlas_model.py
- Checkpoint format: step, epoch, model_state_dict, memory_states, config
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add parent paths for imports when running as module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.loaders.atlas_model import Atlas, AtlasConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for Atlas training.

    This dataclass encapsulates all training hyperparameters including
    model configuration, optimization settings, and checkpoint behavior.

    Attributes:
        d_model: Model hidden dimension (default: 128)
        n_layers: Number of transformer layers (default: 4)
        n_heads: Number of attention heads (default: 4)
        d_ff: Feed-forward dimension (default: 512)
        vocab_size: Vocabulary size (default: 29056 for pruned mT5)
        max_seq_len: Maximum sequence length (default: 512)
        window_size: Sliding window attention size (default: 512)
        dropout: Dropout probability (default: 0.1)
        batch_size: Training batch size (default: 4)
        learning_rate: Peak learning rate (default: 1e-4)
        weight_decay: AdamW weight decay (default: 0.01)
        warmup_steps: LR warmup steps (default: 100)
        max_epochs: Maximum training epochs (default: 10)
        grad_accumulation_steps: Gradient accumulation (default: 1)
        max_grad_norm: Gradient clipping norm (default: 1.0)
        checkpoint_dir: Directory for checkpoints (default: ./checkpoints)
        save_every_epoch: Save checkpoint every N epochs (default: 1)
        log_every_steps: Log metrics every N steps (default: 10)
        resume_from: Path to checkpoint to resume from (default: None)
        seed: Random seed for reproducibility (default: 42)
        device: Target device (default: cuda if available)
        metrics_file: Path to metrics JSON file for dashboard (default: None)
    """
    # Model configuration
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    vocab_size: int = 29056  # Pruned mT5 vocabulary
    max_seq_len: int = 512
    window_size: int = 512
    dropout: float = 0.1

    # Training configuration
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_epochs: int = 10
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints"
    save_every_epoch: int = 1
    log_every_steps: int = 10
    resume_from: Optional[str] = None

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Metrics file for dashboard integration
    metrics_file: Optional[str] = None

    def to_atlas_config(self) -> AtlasConfig:
        """Convert to AtlasConfig for model instantiation."""
        return AtlasConfig(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            d_key=self.d_model,
            d_value=self.d_model,
            window_size=self.window_size,
            dropout=self.dropout,
        )


@dataclass
class TrainingState:
    """Mutable training state for checkpointing and resume.

    Tracks the current position in training for checkpoint/resume.

    Attributes:
        epoch: Current epoch (0-indexed)
        global_step: Total steps across all epochs
        best_loss: Best validation loss seen
        learning_rate: Current learning rate
    """
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    learning_rate: float = 0.0


@dataclass
class TrainingMetrics:
    """Training metrics for a single step or epoch.

    Captures all relevant metrics for logging and dashboard display.

    Attributes:
        loss: Cross-entropy loss
        perplexity: Exp of loss
        learning_rate: Current LR
        epoch: Current epoch
        step: Current step
        tokens_per_second: Training throughput
        gpu_memory_mb: GPU memory usage in MB
        memory_norm: Average memory matrix norm across layers
        timestamp: ISO timestamp
    """
    loss: float = 0.0
    perplexity: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0
    step: int = 0
    tokens_per_second: float = 0.0
    gpu_memory_mb: float = 0.0
    memory_norm: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ============================================================================
# Checkpoint Manager
# ============================================================================


class CheckpointManager:
    """Manages checkpoint saving and loading for Atlas training.

    Handles:
    - Saving checkpoints with model state, optimizer state, and memory states
    - Loading checkpoints for training resume
    - Tracking best checkpoints
    - Memory-efficient checkpoint format

    Checkpoint format:
        - step: int - Global step number
        - epoch: int - Current epoch
        - model_state_dict: dict - Model weights
        - optimizer_state_dict: dict - Optimizer state
        - scheduler_state_dict: dict - LR scheduler state
        - memory_states: list - Per-layer memory states [(M, S), ...]
        - config: AtlasConfig - Model configuration
        - training_state: dict - TrainingState as dict
        - metrics_history: list - Training metrics history
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_every_epoch: int = 1,
    ):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every_epoch: Save every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_epoch = save_every_epoch
        self._metrics_history: list[dict] = []

    def should_save(self, epoch: int) -> bool:
        """Determine if we should save a checkpoint at this epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            True if checkpoint should be saved
        """
        return (epoch + 1) % self.save_every_epoch == 0

    def save_checkpoint(
        self,
        model: Atlas,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        training_state: TrainingState,
        config: TrainingConfig,
        memory_states: Optional[list] = None,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            model: Atlas model
            optimizer: Optimizer
            scheduler: LR scheduler
            training_state: Current training state
            config: Training configuration
            memory_states: Optional memory states per layer

        Returns:
            Path to saved checkpoint
        """
        epoch = training_state.epoch
        step = training_state.global_step

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"

        # Extract memory states if not provided
        if memory_states is None:
            memory_states = []

        # Format memory states for saving
        formatted_memory = []
        for layer_state in memory_states:
            if isinstance(layer_state, tuple) and len(layer_state) == 2:
                M, S = layer_state
                formatted_memory.append({
                    "M": M.detach().cpu(),
                    "S": S.detach().cpu(),
                })
            else:
                formatted_memory.append(layer_state)

        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "memory_states": formatted_memory,
            "config": config.to_atlas_config(),
            "training_state": asdict(training_state),
            "metrics_history": self._metrics_history[-100:],  # Keep last 100
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Also save latest symlink
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path.name)

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: Atlas,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> tuple[TrainingState, list]:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Atlas model to load weights into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            device: Target device

        Returns:
            Tuple of (TrainingState, memory_states)
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        state_dict = checkpoint.get("training_state", {})
        training_state = TrainingState(
            epoch=state_dict.get("epoch", checkpoint.get("epoch", 0)),
            global_step=state_dict.get("global_step", checkpoint.get("step", 0)),
            best_loss=state_dict.get("best_loss", float("inf")),
            learning_rate=state_dict.get("learning_rate", 0.0),
        )

        # Restore memory states
        memory_states = []
        raw_memory = checkpoint.get("memory_states", [])
        for layer_mem in raw_memory:
            if isinstance(layer_mem, dict) and "M" in layer_mem:
                M = layer_mem["M"].to(device)
                S = layer_mem["S"].to(device)
                memory_states.append((M, S))
            elif isinstance(layer_mem, tuple):
                M = layer_mem[0].to(device)
                S = layer_mem[1].to(device)
                memory_states.append((M, S))

        # Restore metrics history
        self._metrics_history = checkpoint.get("metrics_history", [])

        logger.info(
            f"Resumed from epoch {training_state.epoch}, step {training_state.global_step}"
        )

        return training_state, memory_states

    def add_metrics(self, metrics: TrainingMetrics) -> None:
        """Add metrics to history for checkpointing.

        Args:
            metrics: Training metrics to record
        """
        self._metrics_history.append(metrics.to_dict())

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint if exists.

        Returns:
            Path to latest checkpoint or None
        """
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            # Resolve symlink
            return latest_path.resolve()

        # Find by epoch number
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(
                key=lambda p: int(p.stem.split("epoch")[1].split("_")[0]),
                reverse=True,
            )
            return checkpoints[0]

        return None


# ============================================================================
# Data Generation
# ============================================================================


class SyntheticDataLoader:
    """Generate synthetic training data for testing.

    Creates batches of random token sequences for training loop testing
    when real data is not available.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_length: int,
        batch_size: int,
        num_batches: int,
        device: str = "cpu",
    ):
        """Initialize the synthetic data loader.

        Args:
            vocab_size: Vocabulary size for random tokens
            seq_length: Sequence length per sample
            batch_size: Batch size
            num_batches: Number of batches per epoch
            device: Target device
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Generate batches of random token sequences.

        Yields:
            Tuple of (input_ids, labels) tensors
        """
        for _ in range(self.num_batches):
            # Generate random token IDs
            input_ids = torch.randint(
                0,
                self.vocab_size,
                (self.batch_size, self.seq_length),
                device=self.device,
            )
            # Labels are same as inputs for language modeling
            labels = input_ids.clone()

            yield input_ids, labels

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches


# ============================================================================
# Learning Rate Scheduler
# ============================================================================


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine annealing schedule with linear warmup.

    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of peak LR

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Main Trainer Class
# ============================================================================


class AtlasTrainer:
    """Full training pipeline for Atlas model.

    Handles the complete training workflow:
    - Model initialization or loading from checkpoint
    - Data iteration with batching
    - Forward pass, loss computation, backward pass
    - Gradient accumulation and clipping
    - Optimizer and scheduler steps
    - Checkpoint saving with memory states
    - Metrics logging for dashboard

    Example:
        config = TrainingConfig(max_epochs=10, checkpoint_dir="/tmp/training")
        trainer = AtlasTrainer(config)
        trainer.train()
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Initialize model
        atlas_config = config.to_atlas_config()
        self.model = Atlas(atlas_config).to(self.device)

        logger.info(f"Model initialized with {self.model.n_params:,} parameters")

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler will be initialized in train()
        self.scheduler: Optional[LambdaLR] = None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            save_every_epoch=config.save_every_epoch,
        )

        # Training state
        self.training_state = TrainingState()

        # Memory states (per layer)
        self.memory_states: Optional[list] = None

        # Metrics file for dashboard
        self._metrics_file = Path(config.metrics_file) if config.metrics_file else None

        # Resume if specified
        if config.resume_from:
            self._resume_from_checkpoint(config.resume_from)

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.training_state, self.memory_states = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.device,
        )

        # Increment epoch to start from next
        self.training_state.epoch += 1

    def _get_data_loader(self, num_batches: int) -> SyntheticDataLoader:
        """Get a data loader for training.

        For now uses synthetic data. Can be extended for real data.

        Args:
            num_batches: Number of batches per epoch

        Returns:
            Data loader iterator
        """
        return SyntheticDataLoader(
            vocab_size=self.config.vocab_size,
            seq_length=self.config.max_seq_len,
            batch_size=self.config.batch_size,
            num_batches=num_batches,
            device=str(self.device),
        )

    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss and metrics for a batch.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target token IDs [batch, seq_len]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Forward pass with memory states
        loss, new_memory_states, metrics = self.model.compute_loss(
            input_ids,
            labels,
            memory_states=self.memory_states,
            return_metrics=True,
        )

        # Detach memory states to prevent backprop through time
        # This is required because memory states are reused across batches
        if new_memory_states:
            self.memory_states = []
            for state in new_memory_states:
                if isinstance(state, tuple) and len(state) == 2:
                    M, S = state
                    self.memory_states.append((M.detach(), S.detach()))
                elif isinstance(state, dict):
                    self.memory_states.append({
                        k: v.detach() if isinstance(v, torch.Tensor) else v
                        for k, v in state.items()
                    })
                elif isinstance(state, torch.Tensor):
                    self.memory_states.append(state.detach())
                else:
                    self.memory_states.append(state)

        return loss, metrics or {}

    def _get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB.

        Returns:
            GPU memory in MB, or 0 if not using CUDA
        """
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0

    def _get_memory_norm(self) -> float:
        """Get average memory matrix norm across layers.

        Returns:
            Average memory norm
        """
        if not self.memory_states:
            return 0.0

        norms = []
        for layer_state in self.memory_states:
            if isinstance(layer_state, tuple) and len(layer_state) == 2:
                M, S = layer_state
                norms.append(M.norm().item())

        return sum(norms) / len(norms) if norms else 0.0

    def _write_metrics(self, metrics: TrainingMetrics) -> None:
        """Write metrics to file for dashboard.

        Args:
            metrics: Training metrics
        """
        if self._metrics_file is None:
            return

        self._metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to metrics file
        with open(self._metrics_file, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")

    def train_epoch(
        self,
        epoch: int,
        data_loader: SyntheticDataLoader,
    ) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number
            data_loader: Data loader for this epoch

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        start_time = time.time()
        tokens_processed = 0

        for batch_idx, (input_ids, labels) in enumerate(data_loader):
            # Forward and loss
            loss, model_metrics = self._compute_loss(input_ids, labels)

            # Scale loss for gradient accumulation
            loss = loss / self.config.grad_accumulation_steps

            # Backward
            loss.backward()

            # Step optimizer if accumulation complete
            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                self.training_state.global_step += 1

            # Track metrics
            step_loss = loss.item() * self.config.grad_accumulation_steps
            total_loss += step_loss
            num_batches += 1
            tokens_processed += input_ids.numel()

            # Log periodically
            if (batch_idx + 1) % self.config.log_every_steps == 0:
                elapsed = time.time() - start_time
                tokens_per_second = tokens_processed / elapsed if elapsed > 0 else 0
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate

                metrics = TrainingMetrics(
                    loss=step_loss,
                    perplexity=math.exp(min(step_loss, 20)),  # Clamp to avoid overflow
                    learning_rate=current_lr,
                    epoch=epoch,
                    step=self.training_state.global_step,
                    tokens_per_second=tokens_per_second,
                    gpu_memory_mb=self._get_gpu_memory_mb(),
                    memory_norm=self._get_memory_norm(),
                )

                self.checkpoint_manager.add_metrics(metrics)
                self._write_metrics(metrics)

                logger.info(
                    f"Epoch {epoch} | Step {self.training_state.global_step} | "
                    f"Loss: {step_loss:.4f} | PPL: {metrics.perplexity:.2f} | "
                    f"LR: {current_lr:.2e} | Tokens/s: {tokens_per_second:.1f}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(
        self,
        num_batches_per_epoch: int = 50,
    ) -> TrainingState:
        """Run the full training loop.

        Args:
            num_batches_per_epoch: Number of batches per epoch

        Returns:
            Final training state
        """
        # Calculate total training steps
        total_steps = num_batches_per_epoch * self.config.max_epochs

        # Initialize scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # If resuming, advance scheduler
        for _ in range(self.training_state.global_step):
            self.scheduler.step()

        logger.info(
            f"Starting training: {self.config.max_epochs} epochs, "
            f"{num_batches_per_epoch} batches/epoch, {total_steps} total steps"
        )

        start_epoch = self.training_state.epoch
        checkpoints_saved = 0

        for epoch in range(start_epoch, self.config.max_epochs):
            self.training_state.epoch = epoch
            epoch_start = time.time()

            # Get data loader
            data_loader = self._get_data_loader(num_batches_per_epoch)

            # Train epoch
            avg_loss = self.train_epoch(epoch, data_loader)

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Update best loss
            if avg_loss < self.training_state.best_loss:
                self.training_state.best_loss = avg_loss

            # Save checkpoint
            if self.checkpoint_manager.should_save(epoch):
                self.training_state.learning_rate = (
                    self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
                )
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.training_state,
                    self.config,
                    self.memory_states,
                )
                checkpoints_saved += 1

        logger.info(
            f"Training complete | Best Loss: {self.training_state.best_loss:.4f} | "
            f"Checkpoints saved: {checkpoints_saved}"
        )

        return self.training_state


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point for Atlas training."""
    parser = argparse.ArgumentParser(
        description="Train Atlas model with checkpoint saving",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to save checkpoints",
    )

    # Optional training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=50,
        help="Number of batches per epoch",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Model configuration
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model hidden dimension",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=29056,
        help="Vocabulary size",
    )

    # Other options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to write metrics JSON for dashboard",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create config
    config = TrainingConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device=args.device,
        seed=args.seed,
        metrics_file=args.metrics_file,
    )

    # Create trainer
    trainer = AtlasTrainer(config)

    # Train
    try:
        final_state = trainer.train(num_batches_per_epoch=args.batches_per_epoch)

        # Report results
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch*.pt"))

        print(f"\nTraining completed, {len(checkpoints)} checkpoints saved")
        print(f"Best loss: {final_state.best_loss:.4f}")
        print(f"Final step: {final_state.global_step}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
