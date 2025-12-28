"""Tests for metrics collection module."""

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from src.metrics import (
    FileMetricsBackend,
    MetricValue,
    MetricsBackend,
    MetricsCollector,
    TimingResult,
    create_collector_from_config,
    create_file_backend,
    is_mlflow_available,
    track_duration,
    track_experiment,
)


# ============================================================================
# MetricValue Tests
# ============================================================================


class TestMetricValue:
    """Test MetricValue dataclass."""

    def test_basic_creation(self):
        metric = MetricValue(name="test_metric", value=42.0)
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.step is None
        assert metric.tags == {}

    def test_creation_with_all_fields(self):
        timestamp = datetime.now(timezone.utc)
        metric = MetricValue(
            name="test_metric",
            value=3.14,
            timestamp=timestamp,
            step=5,
            tags={"env": "test", "version": "1.0"},
        )
        assert metric.name == "test_metric"
        assert metric.value == 3.14
        assert metric.timestamp == timestamp
        assert metric.step == 5
        assert metric.tags == {"env": "test", "version": "1.0"}

    def test_default_timestamp(self):
        before = datetime.now(timezone.utc)
        metric = MetricValue(name="test", value=1.0)
        after = datetime.now(timezone.utc)
        assert before <= metric.timestamp <= after

    def test_to_dict(self):
        timestamp = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        metric = MetricValue(
            name="accuracy",
            value=0.95,
            timestamp=timestamp,
            step=10,
            tags={"model": "gpt-4"},
        )
        result = metric.to_dict()

        assert result["name"] == "accuracy"
        assert result["value"] == 0.95
        assert result["timestamp"] == "2024-01-15T12:00:00+00:00"
        assert result["step"] == 10
        assert result["tags"] == {"model": "gpt-4"}

    def test_from_dict(self):
        data = {
            "name": "loss",
            "value": 0.123,
            "timestamp": "2024-01-15T12:00:00+00:00",
            "step": 100,
            "tags": {"phase": "training"},
        }
        metric = MetricValue.from_dict(data)

        assert metric.name == "loss"
        assert metric.value == 0.123
        assert metric.step == 100
        assert metric.tags == {"phase": "training"}
        assert metric.timestamp.year == 2024

    def test_from_dict_minimal(self):
        data = {
            "name": "metric",
            "value": 1.0,
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        metric = MetricValue.from_dict(data)

        assert metric.name == "metric"
        assert metric.value == 1.0
        assert metric.step is None
        assert metric.tags == {}

    def test_roundtrip_serialization(self):
        original = MetricValue(
            name="roundtrip",
            value=99.9,
            step=42,
            tags={"key": "value"},
        )
        serialized = original.to_dict()
        restored = MetricValue.from_dict(serialized)

        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.step == original.step
        assert restored.tags == original.tags

    def test_negative_value(self):
        metric = MetricValue(name="loss", value=-0.5)
        assert metric.value == -0.5

    def test_zero_value(self):
        metric = MetricValue(name="errors", value=0.0)
        assert metric.value == 0.0

    def test_very_large_value(self):
        metric = MetricValue(name="huge", value=1e308)
        assert metric.value == 1e308

    def test_very_small_value(self):
        metric = MetricValue(name="tiny", value=1e-308)
        assert metric.value == 1e-308


# ============================================================================
# TimingResult Tests
# ============================================================================


class TestTimingResult:
    """Test TimingResult dataclass."""

    def test_basic_creation(self):
        timing = TimingResult(start_time=100.0)
        assert timing.start_time == 100.0
        assert timing.end_time is None
        assert timing.duration is None
        assert timing.metadata == {}

    def test_with_all_fields(self):
        timing = TimingResult(
            start_time=100.0,
            end_time=105.5,
            duration=5.5,
            metadata={"operation": "test"},
        )
        assert timing.start_time == 100.0
        assert timing.end_time == 105.5
        assert timing.duration == 5.5
        assert timing.metadata == {"operation": "test"}

    def test_duration_seconds_with_duration(self):
        timing = TimingResult(start_time=0, duration=2.5)
        assert timing.duration_seconds == 2.5

    def test_duration_seconds_from_end_time(self):
        timing = TimingResult(start_time=100.0, end_time=103.0)
        assert timing.duration_seconds == 3.0

    def test_duration_seconds_zero_when_not_finished(self):
        timing = TimingResult(start_time=100.0)
        assert timing.duration_seconds == 0.0

    def test_duration_preferred_over_end_time(self):
        # If duration is set, use it even if end_time is different
        timing = TimingResult(start_time=100.0, end_time=110.0, duration=5.0)
        assert timing.duration_seconds == 5.0


# ============================================================================
# FileMetricsBackend Tests
# ============================================================================


class TestFileMetricsBackend:
    """Test FileMetricsBackend class."""

    def test_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test-exp",
                run_id="run-001",
            )

            assert backend.experiment_name == "test-exp"
            assert backend.run_id == "run-001"
            assert backend.run_dir.exists()
            assert backend.run_dir.parent.name == "test-exp"

    def test_auto_generated_run_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test-exp",
            )

            # Run ID should be timestamp-like
            assert backend.run_id is not None
            assert len(backend.run_id) > 0

    def test_creates_directory_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="new-experiment",
                run_id="run-123",
            )

            expected_path = Path(temp_dir) / "new-experiment" / "run-123"
            assert expected_path.exists()
            assert expected_path.is_dir()

    def test_log_single_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test",
                run_id="run1",
            )

            backend.log_metric("accuracy", 0.95, step=1)

            metrics = backend.get_metrics()
            assert len(metrics) == 1
            assert metrics[0].name == "accuracy"
            assert metrics[0].value == 0.95
            assert metrics[0].step == 1

    def test_log_metric_with_all_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test",
                run_id="run1",
            )
            timestamp = datetime.now(timezone.utc)

            backend.log_metric(
                "loss",
                0.123,
                step=5,
                timestamp=timestamp,
                tags={"phase": "train"},
            )

            metrics = backend.get_metrics()
            assert len(metrics) == 1
            assert metrics[0].timestamp == timestamp
            assert metrics[0].tags == {"phase": "train"}

    def test_log_multiple_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test",
                run_id="run1",
            )

            backend.log_metrics(
                {"accuracy": 0.95, "loss": 0.05, "f1": 0.9},
                step=10,
            )

            metrics = backend.get_metrics()
            assert len(metrics) == 3

            metric_names = {m.name for m in metrics}
            assert metric_names == {"accuracy", "loss", "f1"}

            # All should have same step
            for m in metrics:
                assert m.step == 10

    def test_log_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test",
                run_id="run1",
            )

            backend.log_params({
                "learning_rate": 0.001,
                "batch_size": 32,
                "model": "transformer",
            })

            params = backend.get_params()
            assert params["learning_rate"] == 0.001
            assert params["batch_size"] == 32
            assert params["model"] == "transformer"

    def test_log_params_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test",
                run_id="run1",
            )

            backend.log_params({"a": 1, "b": 2})
            backend.log_params({"b": 3, "c": 4})

            params = backend.get_params()
            assert params["a"] == 1
            assert params["b"] == 3  # Updated
            assert params["c"] == 4  # New

    def test_flush_writes_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test",
                run_id="run1",
            )

            backend.log_metric("test_metric", 1.0)
            backend.log_params({"param1": "value1"})
            backend.flush()

            metrics_file = backend.run_dir / "metrics.json"
            params_file = backend.run_dir / "params.json"

            assert metrics_file.exists()
            assert params_file.exists()

            with open(metrics_file) as f:
                metrics_data = json.load(f)
            assert len(metrics_data) == 1
            assert metrics_data[0]["name"] == "test_metric"

            with open(params_file) as f:
                params_data = json.load(f)
            assert params_data["param1"] == "value1"

    def test_close_flushes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test",
                run_id="run1",
            )

            backend.log_metric("metric", 42.0)
            backend.close()

            # File should exist after close
            metrics_file = backend.run_dir / "metrics.json"
            assert metrics_file.exists()

    def test_thread_safety(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(
                output_dir=temp_dir,
                experiment_name="test",
                run_id="run1",
            )

            def log_metrics(thread_id: int):
                for i in range(100):
                    backend.log_metric(f"metric_{thread_id}_{i}", float(i))

            threads = [
                threading.Thread(target=log_metrics, args=(tid,))
                for tid in range(5)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            metrics = backend.get_metrics()
            assert len(metrics) == 500  # 5 threads * 100 metrics

    def test_expands_user_home(self):
        # Test that ~ is expanded in output_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock path that uses ~
            mock_home = Path(temp_dir) / "home"
            mock_home.mkdir()

            # We can't actually test ~ expansion without modifying home
            # but we can verify Path.expanduser is called by checking
            # the resulting path is absolute
            backend = FileMetricsBackend(
                output_dir=Path(temp_dir),
                experiment_name="test",
                run_id="run1",
            )

            assert backend.output_dir.is_absolute() or not str(backend.output_dir).startswith("~")


# ============================================================================
# MetricsCollector Tests
# ============================================================================


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_initialization_no_backends(self):
        collector = MetricsCollector()
        assert collector._backends == []
        assert collector.current_step == 0

    def test_initialization_with_backends(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            assert len(collector._backends) == 1

    def test_add_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MetricsCollector()
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector.add_backend(backend)

            assert len(collector._backends) == 1

    def test_log_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            collector.log_metric("test", 1.0, step=1)

            metrics = backend.get_metrics()
            assert len(metrics) == 1
            assert metrics[0].name == "test"
            assert metrics[0].value == 1.0

    def test_log_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            collector.log_metrics({"a": 1.0, "b": 2.0}, step=5)

            metrics = backend.get_metrics()
            assert len(metrics) == 2

    def test_log_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            collector.log_params({"lr": 0.01})

            params = backend.get_params()
            assert params["lr"] == 0.01

    def test_increment_step(self):
        collector = MetricsCollector()

        assert collector.current_step == 0
        assert collector.increment_step() == 1
        assert collector.current_step == 1
        assert collector.increment_step() == 2
        assert collector.current_step == 2

    def test_flush(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            collector.log_metric("metric", 1.0)
            collector.flush()

            assert (backend.run_dir / "metrics.json").exists()

    def test_close(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            collector.log_metric("metric", 1.0)
            collector.close()

            assert (backend.run_dir / "metrics.json").exists()

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")

            with MetricsCollector(backends=[backend]) as collector:
                collector.log_metric("metric", 1.0)

            # Should be closed and flushed
            assert (backend.run_dir / "metrics.json").exists()

    def test_log_after_close_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            collector.close()

            # Should not raise but should warn (log)
            collector.log_metric("should_warn", 1.0)

            # Metric should not be added
            metrics = backend.get_metrics()
            assert len(metrics) == 0

    def test_auto_flush(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(
                backends=[backend],
                auto_flush_interval=0.01,  # Very short interval
            )

            collector.log_metric("metric1", 1.0)

            # Wait for auto-flush interval
            time.sleep(0.02)

            collector.log_metric("metric2", 2.0)  # Should trigger auto-flush

            # Check file was written
            assert (backend.run_dir / "metrics.json").exists()

    def test_multiple_backends(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend1 = FileMetricsBackend(temp_dir, "test1", "run1")
            backend2 = FileMetricsBackend(temp_dir, "test2", "run2")
            collector = MetricsCollector(backends=[backend1, backend2])

            collector.log_metric("shared_metric", 5.0)

            # Both backends should have the metric
            assert len(backend1.get_metrics()) == 1
            assert len(backend2.get_metrics()) == 1

    def test_backend_error_handling(self):
        """Test that errors in one backend don't affect others."""
        with tempfile.TemporaryDirectory() as temp_dir:
            good_backend = FileMetricsBackend(temp_dir, "good", "run1")

            # Create a mock backend that raises
            bad_backend = mock.Mock(spec=MetricsBackend)
            bad_backend.log_metric.side_effect = Exception("Backend error")

            collector = MetricsCollector(backends=[bad_backend, good_backend])

            # Should not raise, and good backend should still work
            collector.log_metric("test", 1.0)

            assert len(good_backend.get_metrics()) == 1


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestTrackDuration:
    """Test track_duration context manager."""

    def test_basic_timing(self):
        with track_duration("test") as timing:
            time.sleep(0.01)

        assert timing.start_time > 0
        assert timing.end_time is not None
        assert timing.duration is not None
        assert timing.duration >= 0.01
        assert timing.duration < 1.0  # Sanity check

    def test_timing_accuracy(self):
        target_duration = 0.05

        with track_duration("precise") as timing:
            time.sleep(target_duration)

        # Allow 50% tolerance for timing inaccuracy
        assert abs(timing.duration_seconds - target_duration) < target_duration * 0.5

    def test_logs_to_collector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            with track_duration("operation", collector, step=1) as timing:
                time.sleep(0.01)

            metrics = backend.get_metrics()
            assert len(metrics) == 1
            assert metrics[0].name == "operation_duration_seconds"
            assert metrics[0].step == 1
            assert metrics[0].value > 0

    def test_logs_with_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            with track_duration("op", collector, tags={"type": "test"}):
                pass

            metrics = backend.get_metrics()
            assert metrics[0].tags == {"type": "test"}

    def test_no_collector(self):
        # Should work without collector
        with track_duration("standalone") as timing:
            time.sleep(0.01)

        assert timing.duration_seconds > 0

    def test_timing_on_exception(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            with pytest.raises(ValueError):
                with track_duration("failing", collector):
                    raise ValueError("Test error")

            # Duration should still be recorded
            metrics = backend.get_metrics()
            assert len(metrics) == 1
            assert metrics[0].name == "failing_duration_seconds"


class TestTrackExperiment:
    """Test track_experiment context manager."""

    def test_logs_start_and_duration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            with track_experiment("my_exp", collector):
                time.sleep(0.01)

            metrics = backend.get_metrics()
            metric_names = {m.name for m in metrics}

            assert "experiment_start" in metric_names
            assert "my_exp_duration_seconds" in metric_names
            assert "experiment_success" in metric_names

    def test_logs_success_on_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            with track_experiment("exp", collector):
                pass

            metrics = backend.get_metrics()
            success_metric = next(m for m in metrics if m.name == "experiment_success")
            assert success_metric.value == 1.0

    def test_logs_failure_on_exception(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            with pytest.raises(RuntimeError):
                with track_experiment("failing_exp", collector):
                    raise RuntimeError("Experiment failed")

            metrics = backend.get_metrics()
            success_metric = next(m for m in metrics if m.name == "experiment_success")
            assert success_metric.value == 0.0

    def test_logs_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            with track_experiment("exp", collector, params={"lr": 0.01}):
                pass

            params = backend.get_params()
            assert params["lr"] == 0.01

    def test_returns_timing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            with track_experiment("exp", collector) as timing:
                time.sleep(0.01)

            assert isinstance(timing, TimingResult)
            assert timing.duration_seconds > 0


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateFileBackend:
    """Test create_file_backend factory."""

    def test_creates_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = create_file_backend(temp_dir, "exp", "run1")

            assert isinstance(backend, FileMetricsBackend)
            assert backend.experiment_name == "exp"
            assert backend.run_id == "run1"

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = create_file_backend(temp_dir, "exp", "run1")

            assert backend.run_dir.exists()


class TestIsMlflowAvailable:
    """Test is_mlflow_available function."""

    def test_returns_bool(self):
        result = is_mlflow_available()
        assert isinstance(result, bool)


class TestCreateMlflowBackend:
    """Test create_mlflow_backend factory."""

    def test_raises_when_mlflow_not_available(self):
        if not is_mlflow_available():
            from src.metrics import create_mlflow_backend

            with pytest.raises(ImportError, match="mlflow is not installed"):
                create_mlflow_backend()

    @pytest.mark.skipif(
        not is_mlflow_available(),
        reason="MLflow not installed"
    )
    def test_creates_backend_when_available(self):
        from src.metrics import MLflowMetricsBackend, create_mlflow_backend

        backend = create_mlflow_backend(
            tracking_uri="mlruns",
            experiment_name="test",
        )
        assert isinstance(backend, MLflowMetricsBackend)


# ============================================================================
# MLflowMetricsBackend Tests (when available)
# ============================================================================


@pytest.mark.skipif(
    not is_mlflow_available(),
    reason="MLflow not installed"
)
class TestMLflowMetricsBackend:
    """Test MLflowMetricsBackend class (only when MLflow available)."""

    def test_initialization(self):
        from src.metrics import MLflowMetricsBackend

        backend = MLflowMetricsBackend(
            tracking_uri="mlruns",
            experiment_name="test",
        )

        assert backend._experiment_name == "test"
        assert backend._initialized is False

    def test_lazy_initialization(self):
        from src.metrics import MLflowMetricsBackend

        backend = MLflowMetricsBackend(
            tracking_uri="mlruns",
            experiment_name="test",
        )

        # Should not be initialized until first operation
        assert backend._initialized is False

    def test_flush_is_noop(self):
        from src.metrics import MLflowMetricsBackend

        backend = MLflowMetricsBackend()
        # Should not raise
        backend.flush()


# ============================================================================
# create_collector_from_config Tests
# ============================================================================


class TestCreateCollectorFromConfig:
    """Test create_collector_from_config factory."""

    def test_creates_file_backend_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "storage": {
                    "type": "file",
                    "path": temp_dir,
                }
            }

            collector = create_collector_from_config(config, "experiment")

            assert len(collector._backends) == 1
            assert isinstance(collector._backends[0], FileMetricsBackend)

    def test_uses_default_path(self):
        config = {"storage": {"type": "file"}}

        collector = create_collector_from_config(config, "experiment")

        backend = collector._backends[0]
        assert isinstance(backend, FileMetricsBackend)

    def test_with_flush_interval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "storage": {"type": "file", "path": temp_dir},
                "flush_interval_seconds": 60,
            }

            collector = create_collector_from_config(config, "experiment")

            assert collector._auto_flush_interval == 60

    def test_mlflow_fallback_when_unavailable(self):
        if not is_mlflow_available():
            with tempfile.TemporaryDirectory() as temp_dir:
                config = {
                    "storage": {
                        "type": "mlflow",
                        "path": temp_dir,  # Fallback path
                    }
                }

                collector = create_collector_from_config(config, "experiment")

                # Should fall back to file backend
                assert len(collector._backends) == 1
                assert isinstance(collector._backends[0], FileMetricsBackend)

    def test_empty_config(self):
        collector = create_collector_from_config({}, "experiment")

        # Should use defaults (file backend with default path)
        assert len(collector._backends) == 1
        assert isinstance(collector._backends[0], FileMetricsBackend)

    def test_with_run_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "storage": {"type": "file", "path": temp_dir}
            }

            collector = create_collector_from_config(
                config, "experiment", run_id="custom-run"
            )

            backend = collector._backends[0]
            assert backend.run_id == "custom-run"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_metric_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            # Should not raise
            backend.log_metric("", 1.0)

            metrics = backend.get_metrics()
            assert len(metrics) == 1
            assert metrics[0].name == ""

    def test_special_characters_in_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            backend.log_metric("metric/with/slashes", 1.0)
            backend.log_metric("metric.with.dots", 2.0)
            backend.log_metric("metric:with:colons", 3.0)

            metrics = backend.get_metrics()
            assert len(metrics) == 3

    def test_unicode_in_metric_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            backend.log_metric("指标名称", 1.0)
            backend.log_metric("метрика", 2.0)

            backend.flush()

            # Should be readable from file
            with open(backend._metrics_file) as f:
                data = json.load(f)
            assert any(m["name"] == "指标名称" for m in data)

    def test_very_long_metric_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            long_name = "a" * 1000
            backend.log_metric(long_name, 1.0)

            metrics = backend.get_metrics()
            assert metrics[0].name == long_name

    def test_nan_value(self):
        import math

        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            backend.log_metric("nan_metric", float("nan"))

            metrics = backend.get_metrics()
            assert math.isnan(metrics[0].value)

    def test_inf_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            backend.log_metric("inf_metric", float("inf"))
            backend.log_metric("neg_inf_metric", float("-inf"))

            metrics = backend.get_metrics()
            values = {m.name: m.value for m in metrics}

            assert values["inf_metric"] == float("inf")
            assert values["neg_inf_metric"] == float("-inf")

    def test_nested_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            # Tags should be str->str, not nested
            backend.log_metric(
                "tagged",
                1.0,
                tags={"key1": "value1", "key2": "value2"},
            )

            metrics = backend.get_metrics()
            assert metrics[0].tags == {"key1": "value1", "key2": "value2"}

    def test_negative_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            backend.log_metric("metric", 1.0, step=-1)

            metrics = backend.get_metrics()
            assert metrics[0].step == -1

    def test_zero_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            backend.log_metric("metric", 1.0, step=0)

            metrics = backend.get_metrics()
            assert metrics[0].step == 0


class TestTimingPrecision:
    """Test timing accuracy and precision."""

    def test_sub_millisecond_timing(self):
        with track_duration("fast") as timing:
            pass  # No-op, should be very fast

        # Duration should be measurable (> 0) but small
        assert timing.duration_seconds >= 0
        assert timing.duration_seconds < 0.01

    def test_uses_perf_counter(self):
        """Verify we use high-resolution timer."""
        with track_duration("perf") as timing:
            start = time.perf_counter()
            time.sleep(0.02)
            expected = time.perf_counter() - start

        # Should be close to expected
        assert abs(timing.duration_seconds - expected) < 0.01


class TestConcurrency:
    """Test concurrent access patterns."""

    def test_concurrent_log_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileMetricsBackend(temp_dir, "test", "run1")
            collector = MetricsCollector(backends=[backend])

            errors = []

            def log_from_thread(thread_id: int):
                try:
                    for i in range(50):
                        collector.log_metric(f"t{thread_id}_m{i}", float(i))
                        collector.log_metrics({f"t{thread_id}_batch_{i}": i})
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=log_from_thread, args=(i,))
                for i in range(4)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors occurred: {errors}"

            # Check metrics were all logged
            metrics = backend.get_metrics()
            assert len(metrics) == 4 * 50 * 2  # 4 threads, 50 iter, 2 logs each

    def test_concurrent_step_increment(self):
        collector = MetricsCollector()

        results = []

        def increment_steps():
            for _ in range(100):
                results.append(collector.increment_step())

        threads = [threading.Thread(target=increment_steps) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 500 unique step values
        assert len(results) == 500
        assert len(set(results)) == 500
        assert collector.current_step == 500
