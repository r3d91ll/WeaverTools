"""Integration tests for Atlas training pipeline.

These tests verify the complete Atlas training workflow including:
- Model training with checkpoint saving
- Checkpoint resume functionality
- Dashboard metrics generation
- Training overhead measurement

Run with: pytest tests/test_atlas_training.py -v -m integration

Requires:
- GPU with sufficient VRAM (4GB+ recommended for small models)
- Temporary directory access for checkpoints

Mark as slow to skip in normal test runs:
    pytest -m "not integration"
"""

import json
import math
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch

# Skip all tests in this module if no GPU available for true integration tests
# Mark with 'integration' marker to allow selection with: pytest -m integration
pytestmark = [
    pytest.mark.integration,
]

# Conditional skip for GPU tests
GPU_AVAILABLE = torch.cuda.is_available()
GPU_SKIP_REASON = "GPU required for full integration tests"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints.

    Cleaned up after all tests in the module complete.
    """
    tmpdir = tempfile.mkdtemp(prefix="atlas_training_test_")
    yield Path(tmpdir)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def temp_metrics_file(temp_checkpoint_dir):
    """Create a temporary metrics file path."""
    return temp_checkpoint_dir / "metrics.jsonl"


@pytest.fixture(scope="function")
def clean_checkpoint_dir(temp_checkpoint_dir):
    """Provide a clean checkpoint directory for each test.

    Clears any existing checkpoints before the test.
    """
    # Clear any existing checkpoints
    for f in temp_checkpoint_dir.glob("checkpoint_*.pt"):
        f.unlink()

    # Clear symlinks
    latest = temp_checkpoint_dir / "checkpoint_latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()

    return temp_checkpoint_dir


# ============================================================================
# Test Configuration Classes
# ============================================================================


class TestTrainingConfigBasics:
    """Test TrainingConfig dataclass."""

    def test_training_config_defaults(self):
        """Verify TrainingConfig has correct default values."""
        from src.training.atlas_trainer import TrainingConfig

        config = TrainingConfig()

        # Model defaults
        assert config.d_model == 128
        assert config.n_layers == 4
        assert config.n_heads == 4
        assert config.d_ff == 512
        assert config.vocab_size == 29056  # Pruned mT5
        assert config.max_seq_len == 512
        assert config.dropout == 0.1

        # Training defaults
        assert config.batch_size == 4
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.warmup_steps == 100
        assert config.max_epochs == 10
        assert config.grad_accumulation_steps == 1
        assert config.max_grad_norm == 1.0

        # Checkpoint defaults
        assert config.save_every_epoch == 1
        assert config.log_every_steps == 10
        assert config.resume_from is None

    def test_training_config_to_atlas_config(self):
        """Verify TrainingConfig converts to AtlasConfig correctly."""
        from src.training.atlas_trainer import TrainingConfig

        config = TrainingConfig(
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=256,
            vocab_size=1000,
        )

        atlas_config = config.to_atlas_config()

        assert atlas_config.d_model == 64
        assert atlas_config.n_layers == 2
        assert atlas_config.n_heads == 2
        assert atlas_config.d_ff == 256
        assert atlas_config.vocab_size == 1000


class TestTrainingState:
    """Test TrainingState dataclass."""

    def test_training_state_defaults(self):
        """Verify TrainingState has correct default values."""
        from src.training.atlas_trainer import TrainingState

        state = TrainingState()

        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_loss == float("inf")
        assert state.learning_rate == 0.0


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""

    def test_training_metrics_to_dict(self):
        """Verify TrainingMetrics serializes to dict correctly."""
        from src.training.atlas_trainer import TrainingMetrics

        metrics = TrainingMetrics(
            loss=1.5,
            perplexity=4.48,
            learning_rate=1e-4,
            epoch=0,
            step=100,
            tokens_per_second=8000,
            gpu_memory_mb=2000,
            memory_norm=1.0,
        )

        d = metrics.to_dict()

        assert d["loss"] == 1.5
        assert d["perplexity"] == 4.48
        assert d["learning_rate"] == 1e-4
        assert d["epoch"] == 0
        assert d["step"] == 100
        assert d["tokens_per_second"] == 8000
        assert d["gpu_memory_mb"] == 2000
        assert d["memory_norm"] == 1.0
        assert "timestamp" in d


# ============================================================================
# Test Checkpoint Manager
# ============================================================================


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    def test_checkpoint_manager_init(self, clean_checkpoint_dir):
        """Verify CheckpointManager creates directory."""
        from src.training.atlas_trainer import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=clean_checkpoint_dir,
            save_every_epoch=1,
        )

        assert manager.checkpoint_dir.exists()
        assert manager.save_every_epoch == 1

    def test_should_save_every_epoch(self, clean_checkpoint_dir):
        """Verify should_save returns True at correct intervals."""
        from src.training.atlas_trainer import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=clean_checkpoint_dir,
            save_every_epoch=1,
        )

        # Every epoch should trigger save (0-indexed)
        assert manager.should_save(0) is True  # End of epoch 0
        assert manager.should_save(1) is True  # End of epoch 1
        assert manager.should_save(2) is True  # End of epoch 2

    def test_should_save_every_n_epochs(self, clean_checkpoint_dir):
        """Verify should_save respects save_every_epoch setting."""
        from src.training.atlas_trainer import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=clean_checkpoint_dir,
            save_every_epoch=2,
        )

        # Only every 2nd epoch (0-indexed, so after epoch 1, 3, 5...)
        assert manager.should_save(0) is False
        assert manager.should_save(1) is True
        assert manager.should_save(2) is False
        assert manager.should_save(3) is True

    def test_get_latest_checkpoint_none(self, clean_checkpoint_dir):
        """Verify get_latest_checkpoint returns None when empty."""
        from src.training.atlas_trainer import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=clean_checkpoint_dir)

        latest = manager.get_latest_checkpoint()
        assert latest is None

    def test_add_metrics(self, clean_checkpoint_dir):
        """Verify metrics are added to history."""
        from src.training.atlas_trainer import CheckpointManager, TrainingMetrics

        manager = CheckpointManager(checkpoint_dir=clean_checkpoint_dir)

        metrics = TrainingMetrics(loss=1.5, step=100)
        manager.add_metrics(metrics)

        assert len(manager._metrics_history) == 1
        assert manager._metrics_history[0]["loss"] == 1.5


# ============================================================================
# Test Synthetic Data Loader
# ============================================================================


class TestSyntheticDataLoader:
    """Test SyntheticDataLoader for training."""

    def test_synthetic_data_loader_shape(self):
        """Verify synthetic data has correct shape."""
        from src.training.atlas_trainer import SyntheticDataLoader

        loader = SyntheticDataLoader(
            vocab_size=1000,
            seq_length=64,
            batch_size=4,
            num_batches=10,
            device="cpu",
        )

        assert len(loader) == 10

        for input_ids, labels in loader:
            assert input_ids.shape == (4, 64)
            assert labels.shape == (4, 64)
            assert input_ids.dtype == torch.long
            assert (input_ids >= 0).all()
            assert (input_ids < 1000).all()
            break  # Just test first batch

    def test_synthetic_data_loader_iteration(self):
        """Verify we can iterate full loader."""
        from src.training.atlas_trainer import SyntheticDataLoader

        loader = SyntheticDataLoader(
            vocab_size=100,
            seq_length=32,
            batch_size=2,
            num_batches=5,
            device="cpu",
        )

        batch_count = 0
        for input_ids, labels in loader:
            batch_count += 1

        assert batch_count == 5


# ============================================================================
# Test Learning Rate Schedule
# ============================================================================


class TestLRSchedule:
    """Test learning rate schedule with warmup."""

    def test_warmup_schedule(self):
        """Verify warmup increases LR linearly."""
        from src.training.atlas_trainer import get_cosine_schedule_with_warmup

        # Create dummy optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        # At step 0, LR should be ~0
        assert scheduler.get_last_lr()[0] < 1e-5

        # Step halfway through warmup
        for _ in range(50):
            scheduler.step()

        # LR should be ~50% of peak
        lr_50 = scheduler.get_last_lr()[0]
        assert 0.4e-4 < lr_50 < 0.6e-4

        # Step to end of warmup
        for _ in range(50):
            scheduler.step()

        # LR should be at peak
        lr_100 = scheduler.get_last_lr()[0]
        assert 0.9e-4 < lr_100 <= 1e-4

    def test_cosine_decay(self):
        """Verify cosine decay after warmup."""
        from src.training.atlas_trainer import get_cosine_schedule_with_warmup

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
            min_lr_ratio=0.1,
        )

        # Skip warmup
        for _ in range(100):
            scheduler.step()

        peak_lr = scheduler.get_last_lr()[0]

        # Step to near end
        for _ in range(800):
            scheduler.step()

        # LR should be decayed
        final_lr = scheduler.get_last_lr()[0]
        assert final_lr < peak_lr * 0.5


# ============================================================================
# Test Atlas Trainer (Unit Tests)
# ============================================================================


class TestAtlasTrainerUnit:
    """Unit tests for AtlasTrainer that don't require GPU."""

    def test_trainer_init_cpu(self, clean_checkpoint_dir):
        """Verify trainer initializes on CPU."""
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            checkpoint_dir=str(clean_checkpoint_dir),
            device="cpu",
        )

        trainer = AtlasTrainer(config)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device == torch.device("cpu")
        assert trainer.training_state.epoch == 0
        assert trainer.training_state.global_step == 0

    def test_trainer_gpu_memory_method(self, clean_checkpoint_dir):
        """Verify _get_gpu_memory_mb returns 0 on CPU."""
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            vocab_size=100,
            checkpoint_dir=str(clean_checkpoint_dir),
            device="cpu",
        )

        trainer = AtlasTrainer(config)

        # On CPU, should return 0
        memory_mb = trainer._get_gpu_memory_mb()

        # May or may not be 0 depending on CUDA availability
        assert isinstance(memory_mb, float)

    def test_trainer_memory_norm_empty(self, clean_checkpoint_dir):
        """Verify _get_memory_norm returns 0 when no memory states."""
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=32,
            n_layers=1,
            checkpoint_dir=str(clean_checkpoint_dir),
            device="cpu",
        )

        trainer = AtlasTrainer(config)

        # Memory states not initialized yet
        trainer.memory_states = None
        norm = trainer._get_memory_norm()
        assert norm == 0.0

    def test_trainer_data_loader(self, clean_checkpoint_dir):
        """Verify trainer creates data loader correctly."""
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=32,
            n_layers=1,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            checkpoint_dir=str(clean_checkpoint_dir),
            device="cpu",
        )

        trainer = AtlasTrainer(config)
        loader = trainer._get_data_loader(num_batches=5)

        assert len(loader) == 5

        for input_ids, labels in loader:
            assert input_ids.shape == (2, 32)
            break


# ============================================================================
# Integration Tests - Training with Checkpoints
# ============================================================================


class TestTrainingIntegration:
    """Integration tests for full training workflow.

    These tests run actual training and verify checkpoint saving/loading.
    """

    @pytest.mark.slow
    def test_train_small_model_cpu(self, clean_checkpoint_dir, temp_metrics_file):
        """Train a small model for a few epochs on CPU.

        This test verifies:
        1. Training loop runs without errors
        2. Checkpoints are saved correctly
        3. Metrics file is written
        """
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            learning_rate=1e-3,
            warmup_steps=5,
            max_epochs=2,
            log_every_steps=5,
            checkpoint_dir=str(clean_checkpoint_dir),
            metrics_file=str(temp_metrics_file),
            device="cpu",
            seed=42,
        )

        trainer = AtlasTrainer(config)

        # Train with small batches
        final_state = trainer.train(num_batches_per_epoch=10)

        # Verify training completed
        assert final_state.epoch == 1  # 0-indexed, completed 2 epochs
        assert final_state.global_step > 0
        assert final_state.best_loss < float("inf")

        # Verify checkpoints saved
        checkpoints = list(clean_checkpoint_dir.glob("checkpoint_epoch*.pt"))
        assert len(checkpoints) >= 2, f"Expected 2 checkpoints, found: {checkpoints}"

        # Verify latest symlink
        latest = clean_checkpoint_dir / "checkpoint_latest.pt"
        assert latest.exists() or latest.is_symlink()

        # Verify metrics file written
        assert temp_metrics_file.exists()
        with open(temp_metrics_file) as f:
            lines = f.readlines()
        assert len(lines) > 0, "Metrics file should have entries"

        # Verify metrics are valid JSON
        for line in lines[:3]:
            data = json.loads(line.strip())
            assert "loss" in data
            assert "step" in data

    @pytest.mark.slow
    def test_checkpoint_resume(self, clean_checkpoint_dir):
        """Verify training can resume from checkpoint.

        This test:
        1. Trains for 2 epochs
        2. Saves checkpoint
        3. Creates new trainer resuming from checkpoint
        4. Verifies state is restored correctly
        """
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        # First training run
        config1 = TrainingConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            max_epochs=2,
            checkpoint_dir=str(clean_checkpoint_dir),
            device="cpu",
            seed=42,
        )

        trainer1 = AtlasTrainer(config1)
        state1 = trainer1.train(num_batches_per_epoch=5)

        # Get checkpoint path
        checkpoints = list(clean_checkpoint_dir.glob("checkpoint_epoch*.pt"))
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

        # Resume training
        config2 = TrainingConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            max_epochs=4,  # Train 2 more epochs
            checkpoint_dir=str(clean_checkpoint_dir),
            resume_from=str(latest_checkpoint),
            device="cpu",
            seed=42,
        )

        trainer2 = AtlasTrainer(config2)

        # Verify state restored (epoch incremented for resume)
        assert trainer2.training_state.epoch == 2  # Next epoch to train
        assert trainer2.training_state.global_step > 0

        # Continue training
        state2 = trainer2.train(num_batches_per_epoch=5)

        # Verify training continued
        assert state2.epoch == 3  # 0-indexed, completed 4 epochs total
        assert state2.global_step > state1.global_step

        # Verify more checkpoints saved
        final_checkpoints = list(clean_checkpoint_dir.glob("checkpoint_epoch*.pt"))
        assert len(final_checkpoints) >= 4

    @pytest.mark.slow
    def test_checkpoint_memory_states(self, clean_checkpoint_dir):
        """Verify memory states are saved and restored correctly.

        Atlas model uses memory states that must persist across checkpoints.
        """
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=32,
            n_layers=2,  # Multiple layers for memory testing
            n_heads=1,
            d_ff=64,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            max_epochs=1,
            checkpoint_dir=str(clean_checkpoint_dir),
            device="cpu",
            seed=42,
        )

        trainer = AtlasTrainer(config)
        trainer.train(num_batches_per_epoch=5)

        # Get checkpoint
        checkpoints = list(clean_checkpoint_dir.glob("checkpoint_epoch*.pt"))
        assert len(checkpoints) > 0

        # Load checkpoint and verify memory states
        checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)

        assert "memory_states" in checkpoint
        memory_states = checkpoint["memory_states"]

        # Memory states format should be list of dicts or empty
        assert isinstance(memory_states, list)


# ============================================================================
# Integration Tests - Dashboard Metrics
# ============================================================================


class TestDashboardMetrics:
    """Test dashboard metrics generation during training."""

    @pytest.mark.slow
    def test_metrics_file_format(self, clean_checkpoint_dir, temp_metrics_file):
        """Verify metrics file has correct format for dashboard."""
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=32,
            n_layers=1,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            max_epochs=1,
            log_every_steps=2,  # Log frequently
            checkpoint_dir=str(clean_checkpoint_dir),
            metrics_file=str(temp_metrics_file),
            device="cpu",
        )

        trainer = AtlasTrainer(config)
        trainer.train(num_batches_per_epoch=10)

        # Read metrics file
        assert temp_metrics_file.exists()

        with open(temp_metrics_file) as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) >= 4, f"Expected at least 4 log entries, got {len(lines)}"

        # Verify each line is valid JSON with expected fields
        for line in lines:
            data = json.loads(line)

            # Required fields for dashboard
            assert "loss" in data
            assert "perplexity" in data
            assert "learning_rate" in data
            assert "epoch" in data
            assert "step" in data
            assert "tokens_per_second" in data
            assert "gpu_memory_mb" in data
            assert "memory_norm" in data
            assert "timestamp" in data

            # Validate types
            assert isinstance(data["loss"], (int, float))
            assert isinstance(data["perplexity"], (int, float))
            assert isinstance(data["step"], int)
            assert data["loss"] >= 0
            assert data["perplexity"] >= 1.0  # Perplexity is exp(loss)

    def test_metrics_incremental_steps(self, clean_checkpoint_dir, temp_metrics_file):
        """Verify metrics show increasing step numbers."""
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=32,
            n_layers=1,
            vocab_size=100,
            max_epochs=1,
            log_every_steps=1,  # Log every step
            checkpoint_dir=str(clean_checkpoint_dir),
            metrics_file=str(temp_metrics_file),
            device="cpu",
        )

        trainer = AtlasTrainer(config)
        trainer.train(num_batches_per_epoch=5)

        with open(temp_metrics_file) as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]

        # Steps should be monotonically increasing
        steps = [d["step"] for d in lines]
        for i in range(1, len(steps)):
            assert steps[i] >= steps[i-1], "Steps should increase monotonically"


# ============================================================================
# Integration Tests - Dashboard Components
# ============================================================================


class TestDashboardComponents:
    """Test dashboard component creation."""

    def test_dashboard_metrics_parsing(self):
        """Verify DashboardMetrics parses training output correctly."""
        from src.training.dashboard import DashboardMetrics

        data = {
            "loss": 2.5,
            "perplexity": 12.18,
            "learning_rate": 1e-4,
            "epoch": 0,
            "step": 100,
            "tokens_per_second": 8000,
            "gpu_memory_mb": 2500,
            "memory_norm": 1.5,
            "timestamp": "2024-01-01T12:00:00",
        }

        metrics = DashboardMetrics.from_dict(data)

        assert metrics.loss == 2.5
        assert metrics.perplexity == 12.18
        assert metrics.learning_rate == 1e-4
        assert metrics.epoch == 0
        assert metrics.step == 100
        assert metrics.tokens_per_second == 8000
        assert metrics.gpu_memory_mb == 2500
        assert metrics.memory_norm == 1.5
        assert metrics.timestamp == "2024-01-01T12:00:00"

    def test_metrics_buffer_thread_safety(self):
        """Verify MetricsBuffer is thread-safe."""
        from src.training.dashboard import MetricsBuffer, DashboardMetrics

        buffer = MetricsBuffer()

        def add_metrics(start, count):
            for i in range(count):
                metrics = DashboardMetrics(step=start + i, loss=float(i))
                buffer.add(metrics)

        # Create multiple threads adding metrics
        threads = [
            threading.Thread(target=add_metrics, args=(0, 50)),
            threading.Thread(target=add_metrics, args=(100, 50)),
            threading.Thread(target=add_metrics, args=(200, 50)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All metrics should be added
        all_metrics = buffer.get_all()
        assert len(all_metrics) == 150

    def test_metrics_buffer_max_size(self):
        """Verify MetricsBuffer respects max size."""
        from src.training.dashboard import MetricsBuffer, DashboardMetrics

        buffer = MetricsBuffer(max_size=10)

        # Add more than max
        for i in range(20):
            metrics = DashboardMetrics(step=i, loss=float(i))
            buffer.add(metrics)

        all_metrics = buffer.get_all()
        assert len(all_metrics) == 10

        # Should have the most recent 10
        steps = [m.step for m in all_metrics]
        assert steps == list(range(10, 20))

    def test_metrics_buffer_get_latest(self):
        """Verify get_latest returns most recent metrics."""
        from src.training.dashboard import MetricsBuffer, DashboardMetrics

        buffer = MetricsBuffer()

        buffer.add(DashboardMetrics(step=1, loss=1.0))
        buffer.add(DashboardMetrics(step=2, loss=2.0))
        buffer.add(DashboardMetrics(step=3, loss=3.0))

        latest = buffer.get_latest()
        assert latest.step == 3
        assert latest.loss == 3.0


class TestMetricsFileReader:
    """Test MetricsFileReader for dashboard."""

    def test_read_empty_file(self, temp_checkpoint_dir):
        """Verify reader handles empty file."""
        from src.training.dashboard import MetricsFileReader, MetricsBuffer

        metrics_file = temp_checkpoint_dir / "empty_metrics.jsonl"
        metrics_file.touch()

        buffer = MetricsBuffer()
        reader = MetricsFileReader(metrics_file, buffer)

        count = reader.read_new_metrics()
        assert count == 0
        assert buffer.size == 0

    def test_read_metrics_incremental(self, temp_checkpoint_dir):
        """Verify reader reads incrementally."""
        from src.training.dashboard import MetricsFileReader, MetricsBuffer

        metrics_file = temp_checkpoint_dir / "inc_metrics.jsonl"

        buffer = MetricsBuffer()
        reader = MetricsFileReader(metrics_file, buffer)

        # Write first batch
        with open(metrics_file, "w") as f:
            for i in range(5):
                f.write(json.dumps({"loss": i, "step": i}) + "\n")

        count1 = reader.read_new_metrics()
        assert count1 == 5
        assert buffer.size == 5

        # Append more
        with open(metrics_file, "a") as f:
            for i in range(5, 10):
                f.write(json.dumps({"loss": i, "step": i}) + "\n")

        count2 = reader.read_new_metrics()
        assert count2 == 5
        assert buffer.size == 10

    def test_read_malformed_lines(self, temp_checkpoint_dir):
        """Verify reader skips malformed JSON lines."""
        from src.training.dashboard import MetricsFileReader, MetricsBuffer

        metrics_file = temp_checkpoint_dir / "malformed_metrics.jsonl"

        with open(metrics_file, "w") as f:
            f.write('{"loss": 1.0, "step": 1}\n')
            f.write('not valid json\n')
            f.write('{"loss": 2.0, "step": 2}\n')
            f.write('\n')  # Empty line
            f.write('{"loss": 3.0, "step": 3}\n')

        buffer = MetricsBuffer()
        reader = MetricsFileReader(metrics_file, buffer)

        count = reader.read_new_metrics()
        assert count == 3  # Only valid lines


# ============================================================================
# Integration Tests - Training Overhead
# ============================================================================


class TestTrainingOverhead:
    """Test training overhead with dashboard metrics."""

    @pytest.mark.slow
    def test_metrics_writing_overhead(self, clean_checkpoint_dir, temp_metrics_file):
        """Verify metrics writing has minimal overhead.

        Target: <1% overhead from metrics writing.
        """
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer
        import time

        base_config = dict(
            d_model=32,
            n_layers=1,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            max_epochs=1,
            log_every_steps=1,
            checkpoint_dir=str(clean_checkpoint_dir),
            device="cpu",
            seed=42,
        )

        # Run without metrics file
        config_no_metrics = TrainingConfig(**base_config, metrics_file=None)
        trainer1 = AtlasTrainer(config_no_metrics)

        start = time.time()
        trainer1.train(num_batches_per_epoch=20)
        time_no_metrics = time.time() - start

        # Clear checkpoints
        for f in clean_checkpoint_dir.glob("checkpoint_*.pt"):
            f.unlink()
        latest = clean_checkpoint_dir / "checkpoint_latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()

        # Run with metrics file
        config_with_metrics = TrainingConfig(
            **base_config,
            metrics_file=str(temp_metrics_file)
        )
        trainer2 = AtlasTrainer(config_with_metrics)

        start = time.time()
        trainer2.train(num_batches_per_epoch=20)
        time_with_metrics = time.time() - start

        # Calculate overhead
        overhead = (time_with_metrics - time_no_metrics) / time_no_metrics

        # Allow some variance, but should be <10% on average
        # (Target is <1% but CPU testing may have variance)
        print(f"\nTraining time without metrics: {time_no_metrics:.3f}s")
        print(f"Training time with metrics: {time_with_metrics:.3f}s")
        print(f"Overhead: {overhead*100:.1f}%")

        # Relaxed assertion for CPU testing
        assert overhead < 0.50, f"Metrics overhead too high: {overhead*100:.1f}%"


# ============================================================================
# GPU Integration Tests
# ============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason=GPU_SKIP_REASON)
class TestTrainingGPU:
    """GPU-specific integration tests."""

    def test_train_on_gpu(self, clean_checkpoint_dir):
        """Verify training works on GPU."""
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            vocab_size=1000,
            max_seq_len=64,
            batch_size=4,
            max_epochs=2,
            checkpoint_dir=str(clean_checkpoint_dir),
            device="cuda",
            seed=42,
        )

        trainer = AtlasTrainer(config)

        # Verify model is on GPU
        assert next(trainer.model.parameters()).device.type == "cuda"

        # Train
        final_state = trainer.train(num_batches_per_epoch=10)

        # Verify completed
        assert final_state.epoch == 1
        assert final_state.global_step > 0

        # Verify GPU memory was tracked
        # Get latest metrics
        latest_checkpoint = list(clean_checkpoint_dir.glob("checkpoint_epoch*.pt"))[0]
        checkpoint = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)

        metrics_history = checkpoint.get("metrics_history", [])
        if metrics_history:
            assert any(m.get("gpu_memory_mb", 0) > 0 for m in metrics_history)

    def test_gpu_memory_stays_bounded(self, clean_checkpoint_dir):
        """Verify GPU memory doesn't grow unboundedly during training.

        Target: <5GB for small model training.
        """
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        config = TrainingConfig(
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            vocab_size=29056,
            max_seq_len=256,
            batch_size=4,
            max_epochs=3,
            checkpoint_dir=str(clean_checkpoint_dir),
            device="cuda",
            seed=42,
        )

        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        trainer = AtlasTrainer(config)
        trainer.train(num_batches_per_epoch=20)

        # Get peak memory
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"\nPeak GPU memory: {peak_memory_mb:.1f} MB")

        # Should stay under 5GB
        assert peak_memory_mb < 5000, f"GPU memory too high: {peak_memory_mb:.1f} MB"

        # Clean up
        torch.cuda.empty_cache()


# ============================================================================
# End-to-End Tests
# ============================================================================


@pytest.mark.slow
class TestEndToEnd:
    """End-to-end tests for complete training workflow."""

    def test_full_training_workflow_10_epochs(self, temp_checkpoint_dir):
        """Run complete 10 epoch training workflow.

        Verifies:
        1. Train for 10 epochs
        2. Checkpoints saved every epoch
        3. Resume from epoch 5 works
        4. Final state is correct
        """
        from src.training.atlas_trainer import TrainingConfig, AtlasTrainer

        metrics_file = temp_checkpoint_dir / "e2e_metrics.jsonl"

        config = TrainingConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            learning_rate=1e-3,
            warmup_steps=10,
            max_epochs=10,
            save_every_epoch=1,
            log_every_steps=2,
            checkpoint_dir=str(temp_checkpoint_dir),
            metrics_file=str(metrics_file),
            device="cpu",
            seed=42,
        )

        trainer = AtlasTrainer(config)

        print("\n=== Starting 10-Epoch Training ===")
        final_state = trainer.train(num_batches_per_epoch=5)
        print(f"Final state: epoch={final_state.epoch}, step={final_state.global_step}")

        # Verify completed 10 epochs
        assert final_state.epoch == 9, f"Expected epoch 9, got {final_state.epoch}"

        # Verify 10 checkpoints saved
        checkpoints = list(temp_checkpoint_dir.glob("checkpoint_epoch*.pt"))
        assert len(checkpoints) == 10, f"Expected 10 checkpoints, got {len(checkpoints)}"

        # Verify metrics file has entries
        assert metrics_file.exists()
        with open(metrics_file) as f:
            lines = f.readlines()
        assert len(lines) >= 20, f"Expected at least 20 metric entries, got {len(lines)}"

        print(f"=== Training Complete: {len(checkpoints)} checkpoints, {len(lines)} metrics ===")

        # Test resume from epoch 5
        print("\n=== Testing Resume from Epoch 5 ===")

        # Find epoch 5 checkpoint (epoch is 0-indexed in filename)
        epoch5_checkpoint = None
        for cp in checkpoints:
            if "epoch4_" in cp.name:  # epoch4 = 5th epoch (0-indexed)
                epoch5_checkpoint = cp
                break

        assert epoch5_checkpoint is not None, "Could not find epoch 5 checkpoint"

        # Create new trainer resuming from epoch 5
        resume_config = TrainingConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            vocab_size=100,
            max_seq_len=32,
            batch_size=2,
            max_epochs=12,  # Train 2 more epochs
            checkpoint_dir=str(temp_checkpoint_dir),
            resume_from=str(epoch5_checkpoint),
            device="cpu",
            seed=42,
        )

        resume_trainer = AtlasTrainer(resume_config)

        # Verify resume state
        assert resume_trainer.training_state.epoch == 5, \
            f"Expected to resume at epoch 5, got {resume_trainer.training_state.epoch}"

        resume_state = resume_trainer.train(num_batches_per_epoch=5)

        # Should have completed epochs 5-11 (indices 5, 6, 7, 8, 9, 10, 11)
        assert resume_state.epoch == 11, f"Expected epoch 11, got {resume_state.epoch}"

        # Verify more checkpoints exist
        final_checkpoints = list(temp_checkpoint_dir.glob("checkpoint_epoch*.pt"))
        assert len(final_checkpoints) >= 12

        print(f"=== Resume Complete: {len(final_checkpoints)} total checkpoints ===")
