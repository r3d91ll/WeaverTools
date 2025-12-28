"""Tests for storage module - result persistence, querying, and export."""

import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
import yaml

from src.config import (
    AgentConfig,
    AgentType,
    ExperimentConfig,
    FullExperimentConfig,
    OutputFormat,
)
from src.storage import (
    ExperimentResult,
    ExperimentStatus,
    ResultStorage,
    ScenarioResult,
    aggregate_metrics,
    create_storage,
    create_storage_from_config,
    find_by_config_hash,
    find_by_tag,
    get_latest_result,
    query_experiments,
)


# ============================================================================
# ExperimentStatus Tests
# ============================================================================


class TestExperimentStatus:
    """Test ExperimentStatus enum."""

    def test_all_statuses_exist(self):
        assert ExperimentStatus.PENDING == "pending"
        assert ExperimentStatus.RUNNING == "running"
        assert ExperimentStatus.PAUSED == "paused"
        assert ExperimentStatus.COMPLETED == "completed"
        assert ExperimentStatus.FAILED == "failed"
        assert ExperimentStatus.CANCELLED == "cancelled"

    def test_status_is_string_enum(self):
        status = ExperimentStatus.COMPLETED
        assert isinstance(status, str)
        assert status == "completed"

    def test_status_from_string(self):
        assert ExperimentStatus("pending") == ExperimentStatus.PENDING
        assert ExperimentStatus("running") == ExperimentStatus.RUNNING
        assert ExperimentStatus("paused") == ExperimentStatus.PAUSED
        assert ExperimentStatus("completed") == ExperimentStatus.COMPLETED
        assert ExperimentStatus("failed") == ExperimentStatus.FAILED
        assert ExperimentStatus("cancelled") == ExperimentStatus.CANCELLED

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            ExperimentStatus("invalid")


# ============================================================================
# ScenarioResult Tests
# ============================================================================


class TestScenarioResult:
    """Test ScenarioResult dataclass."""

    def test_basic_creation(self):
        result = ScenarioResult(
            name="test-scenario",
            status=ExperimentStatus.COMPLETED,
        )
        assert result.name == "test-scenario"
        assert result.status == ExperimentStatus.COMPLETED
        assert result.steps_completed == 0
        assert result.steps_total == 0
        assert result.duration_seconds == 0.0
        assert result.outputs == {}
        assert result.errors == []
        assert result.started_at is None
        assert result.completed_at is None

    def test_creation_with_all_fields(self):
        started = datetime.now(timezone.utc)
        completed = started + timedelta(seconds=30)

        result = ScenarioResult(
            name="full-scenario",
            status=ExperimentStatus.COMPLETED,
            steps_completed=5,
            steps_total=5,
            duration_seconds=30.0,
            outputs={"step1": "output1", "step2": "output2"},
            errors=[],
            started_at=started,
            completed_at=completed,
        )

        assert result.steps_completed == 5
        assert result.steps_total == 5
        assert result.duration_seconds == 30.0
        assert len(result.outputs) == 2
        assert result.started_at == started
        assert result.completed_at == completed

    def test_to_dict(self):
        started = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        completed = datetime(2024, 1, 15, 12, 1, 0, tzinfo=timezone.utc)

        result = ScenarioResult(
            name="scenario1",
            status=ExperimentStatus.COMPLETED,
            steps_completed=3,
            steps_total=3,
            duration_seconds=60.0,
            outputs={"key": "value"},
            errors=["warning1"],
            started_at=started,
            completed_at=completed,
        )

        data = result.to_dict()

        assert data["name"] == "scenario1"
        assert data["status"] == "completed"
        assert data["steps_completed"] == 3
        assert data["steps_total"] == 3
        assert data["duration_seconds"] == 60.0
        assert data["outputs"] == {"key": "value"}
        assert data["errors"] == ["warning1"]
        assert data["started_at"] == "2024-01-15T12:00:00+00:00"
        assert data["completed_at"] == "2024-01-15T12:01:00+00:00"

    def test_to_dict_with_none_timestamps(self):
        result = ScenarioResult(
            name="scenario",
            status=ExperimentStatus.PENDING,
        )

        data = result.to_dict()
        assert data["started_at"] is None
        assert data["completed_at"] is None

    def test_from_dict(self):
        data = {
            "name": "test-scenario",
            "status": "completed",
            "steps_completed": 4,
            "steps_total": 5,
            "duration_seconds": 45.5,
            "outputs": {"step1": {"result": "ok"}},
            "errors": ["step5 failed"],
            "started_at": "2024-01-15T12:00:00+00:00",
            "completed_at": "2024-01-15T12:00:45+00:00",
        }

        result = ScenarioResult.from_dict(data)

        assert result.name == "test-scenario"
        assert result.status == ExperimentStatus.COMPLETED
        assert result.steps_completed == 4
        assert result.steps_total == 5
        assert result.duration_seconds == 45.5
        assert result.outputs == {"step1": {"result": "ok"}}
        assert result.errors == ["step5 failed"]
        assert result.started_at.year == 2024
        assert result.completed_at is not None

    def test_from_dict_minimal(self):
        data = {
            "name": "minimal",
            "status": "pending",
        }

        result = ScenarioResult.from_dict(data)

        assert result.name == "minimal"
        assert result.status == ExperimentStatus.PENDING
        assert result.steps_completed == 0
        assert result.outputs == {}
        assert result.started_at is None

    def test_roundtrip_serialization(self):
        original = ScenarioResult(
            name="roundtrip",
            status=ExperimentStatus.FAILED,
            steps_completed=2,
            steps_total=5,
            duration_seconds=15.0,
            outputs={"step1": "done", "step2": "partial"},
            errors=["step3 timed out"],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        data = original.to_dict()
        restored = ScenarioResult.from_dict(data)

        assert restored.name == original.name
        assert restored.status == original.status
        assert restored.steps_completed == original.steps_completed
        assert restored.steps_total == original.steps_total
        assert restored.duration_seconds == original.duration_seconds
        assert restored.outputs == original.outputs
        assert restored.errors == original.errors


# ============================================================================
# ExperimentResult Tests
# ============================================================================


class TestExperimentResult:
    """Test ExperimentResult dataclass."""

    def test_basic_creation(self):
        result = ExperimentResult(
            experiment_id="exp-001",
            experiment_name="test-experiment",
        )

        assert result.experiment_id == "exp-001"
        assert result.experiment_name == "test-experiment"
        assert result.status == ExperimentStatus.PENDING
        assert result.config_hash == ""
        assert result.seed is None
        assert result.scenarios == []
        assert result.metrics == {}
        assert result.parameters == {}
        assert result.tags == []
        assert result.duration_seconds == 0.0
        assert result.started_at is not None
        assert result.completed_at is None
        assert result.error_message is None
        assert result.checkpoint_path is None
        assert result.metadata == {}

    def test_creation_with_all_fields(self):
        started = datetime.now(timezone.utc)
        completed = started + timedelta(minutes=5)

        scenario = ScenarioResult(
            name="scenario1",
            status=ExperimentStatus.COMPLETED,
        )

        result = ExperimentResult(
            experiment_id="exp-full",
            experiment_name="full-experiment",
            status=ExperimentStatus.COMPLETED,
            config_hash="abc123",
            seed=42,
            scenarios=[scenario],
            metrics={"accuracy": 0.95, "loss": 0.05},
            parameters={"lr": 0.01, "batch_size": 32},
            tags=["test", "benchmark"],
            duration_seconds=300.0,
            started_at=started,
            completed_at=completed,
            error_message=None,
            checkpoint_path=None,
            metadata={"version": "1.0"},
        )

        assert result.config_hash == "abc123"
        assert result.seed == 42
        assert len(result.scenarios) == 1
        assert result.metrics["accuracy"] == 0.95
        assert result.parameters["batch_size"] == 32
        assert "benchmark" in result.tags
        assert result.metadata["version"] == "1.0"

    def test_to_dict(self):
        started = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        completed = datetime(2024, 1, 15, 12, 5, 0, tzinfo=timezone.utc)

        result = ExperimentResult(
            experiment_id="exp-123",
            experiment_name="dict-test",
            status=ExperimentStatus.COMPLETED,
            config_hash="hash123",
            seed=42,
            scenarios=[
                ScenarioResult(name="s1", status=ExperimentStatus.COMPLETED)
            ],
            metrics={"metric1": 1.0},
            parameters={"param1": "value1"},
            tags=["tag1"],
            duration_seconds=300.0,
            started_at=started,
            completed_at=completed,
            metadata={"key": "value"},
        )

        data = result.to_dict()

        assert data["experiment_id"] == "exp-123"
        assert data["experiment_name"] == "dict-test"
        assert data["status"] == "completed"
        assert data["config_hash"] == "hash123"
        assert data["seed"] == 42
        assert len(data["scenarios"]) == 1
        assert data["scenarios"][0]["name"] == "s1"
        assert data["metrics"] == {"metric1": 1.0}
        assert data["parameters"] == {"param1": "value1"}
        assert data["tags"] == ["tag1"]
        assert data["duration_seconds"] == 300.0
        assert data["started_at"] == "2024-01-15T12:00:00+00:00"
        assert data["completed_at"] == "2024-01-15T12:05:00+00:00"
        assert data["metadata"] == {"key": "value"}

    def test_from_dict(self):
        data = {
            "experiment_id": "exp-456",
            "experiment_name": "from-dict-test",
            "status": "failed",
            "config_hash": "xyz789",
            "seed": 123,
            "scenarios": [
                {"name": "s1", "status": "completed"},
                {"name": "s2", "status": "failed"},
            ],
            "metrics": {"accuracy": 0.9},
            "parameters": {"lr": 0.001},
            "tags": ["prod"],
            "duration_seconds": 120.5,
            "started_at": "2024-01-15T12:00:00+00:00",
            "completed_at": "2024-01-15T12:02:00+00:00",
            "error_message": "Test error",
            "checkpoint_path": "/path/to/checkpoint",
            "metadata": {"run": 1},
        }

        result = ExperimentResult.from_dict(data)

        assert result.experiment_id == "exp-456"
        assert result.experiment_name == "from-dict-test"
        assert result.status == ExperimentStatus.FAILED
        assert result.config_hash == "xyz789"
        assert result.seed == 123
        assert len(result.scenarios) == 2
        assert result.scenarios[0].name == "s1"
        assert result.scenarios[1].status == ExperimentStatus.FAILED
        assert result.metrics["accuracy"] == 0.9
        assert result.error_message == "Test error"
        assert result.checkpoint_path == "/path/to/checkpoint"

    def test_from_dict_minimal(self):
        data = {
            "experiment_id": "min-exp",
            "experiment_name": "minimal",
            "started_at": "2024-01-15T12:00:00+00:00",
        }

        result = ExperimentResult.from_dict(data)

        assert result.experiment_id == "min-exp"
        assert result.status == ExperimentStatus.PENDING
        assert result.scenarios == []
        assert result.metrics == {}

    def test_to_json(self):
        result = ExperimentResult(
            experiment_id="json-exp",
            experiment_name="json-test",
            metrics={"accuracy": 0.95},
        )

        json_str = result.to_json()
        data = json.loads(json_str)

        assert data["experiment_id"] == "json-exp"
        assert data["metrics"]["accuracy"] == 0.95

    def test_to_json_with_indent(self):
        result = ExperimentResult(
            experiment_id="indent-exp",
            experiment_name="indent-test",
        )

        json_str = result.to_json(indent=4)

        # Should have pretty-printed indentation
        assert "    " in json_str

    def test_from_json(self):
        json_str = json.dumps({
            "experiment_id": "from-json",
            "experiment_name": "json-test",
            "status": "completed",
            "started_at": "2024-01-15T12:00:00+00:00",
        })

        result = ExperimentResult.from_json(json_str)

        assert result.experiment_id == "from-json"
        assert result.status == ExperimentStatus.COMPLETED

    def test_to_yaml(self):
        result = ExperimentResult(
            experiment_id="yaml-exp",
            experiment_name="yaml-test",
            tags=["tag1", "tag2"],
        )

        yaml_str = result.to_yaml()
        data = yaml.safe_load(yaml_str)

        assert data["experiment_id"] == "yaml-exp"
        assert data["tags"] == ["tag1", "tag2"]

    def test_from_yaml(self):
        yaml_str = """
experiment_id: from-yaml
experiment_name: yaml-test
status: running
started_at: "2024-01-15T12:00:00+00:00"
"""

        result = ExperimentResult.from_yaml(yaml_str)

        assert result.experiment_id == "from-yaml"
        assert result.status == ExperimentStatus.RUNNING

    def test_roundtrip_json(self):
        original = ExperimentResult(
            experiment_id="json-roundtrip",
            experiment_name="json-test",
            status=ExperimentStatus.COMPLETED,
            seed=42,
            metrics={"accuracy": 0.95, "f1": 0.9},
            tags=["test", "benchmark"],
        )

        json_str = original.to_json()
        restored = ExperimentResult.from_json(json_str)

        assert restored.experiment_id == original.experiment_id
        assert restored.status == original.status
        assert restored.seed == original.seed
        assert restored.metrics == original.metrics
        assert restored.tags == original.tags

    def test_roundtrip_yaml(self):
        original = ExperimentResult(
            experiment_id="yaml-roundtrip",
            experiment_name="yaml-test",
            status=ExperimentStatus.PAUSED,
            parameters={"lr": 0.01},
        )

        yaml_str = original.to_yaml()
        restored = ExperimentResult.from_yaml(yaml_str)

        assert restored.experiment_id == original.experiment_id
        assert restored.status == original.status
        assert restored.parameters == original.parameters


class TestExperimentResultFromConfig:
    """Test ExperimentResult.from_config factory method."""

    def test_from_config_basic(self):
        config = FullExperimentConfig(
            experiment=ExperimentConfig(
                name="config-exp",
                description="Test experiment",
            ),
            agents=[
                AgentConfig(id="agent1", type=AgentType.CLAUDE),
            ],
        )

        result = ExperimentResult.from_config(
            config=config,
            experiment_id="exp-from-config",
            config_hash="hash123",
        )

        assert result.experiment_id == "exp-from-config"
        assert result.experiment_name == "config-exp"
        assert result.status == ExperimentStatus.PENDING
        assert result.config_hash == "hash123"

    def test_from_config_with_seed_and_params(self):
        config = FullExperimentConfig(
            experiment=ExperimentConfig(
                name="config-exp",
                seed=42,
                parameters={"lr": 0.01, "batch_size": 32},
                tags=["test", "v1"],
            ),
            agents=[
                AgentConfig(id="agent1", type=AgentType.LOCAL),
            ],
        )

        result = ExperimentResult.from_config(config, "exp-id")

        assert result.seed == 42
        assert result.parameters == {"lr": 0.01, "batch_size": 32}
        assert result.tags == ["test", "v1"]


class TestExperimentResultProperties:
    """Test ExperimentResult properties."""

    def test_is_finished_pending(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            status=ExperimentStatus.PENDING,
        )
        assert not result.is_finished

    def test_is_finished_running(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            status=ExperimentStatus.RUNNING,
        )
        assert not result.is_finished

    def test_is_finished_paused(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            status=ExperimentStatus.PAUSED,
        )
        assert not result.is_finished

    def test_is_finished_completed(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            status=ExperimentStatus.COMPLETED,
        )
        assert result.is_finished

    def test_is_finished_failed(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            status=ExperimentStatus.FAILED,
        )
        assert result.is_finished

    def test_is_finished_cancelled(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            status=ExperimentStatus.CANCELLED,
        )
        assert result.is_finished

    def test_is_successful_completed(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            status=ExperimentStatus.COMPLETED,
        )
        assert result.is_successful

    def test_is_successful_failed(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            status=ExperimentStatus.FAILED,
        )
        assert not result.is_successful


class TestExperimentResultLifecycle:
    """Test ExperimentResult lifecycle methods."""

    def test_mark_running(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
        )

        before = datetime.now(timezone.utc)
        result.mark_running()
        after = datetime.now(timezone.utc)

        assert result.status == ExperimentStatus.RUNNING
        assert before <= result.started_at <= after

    def test_mark_completed(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
        )
        result.mark_running()
        time.sleep(0.01)  # Small delay
        result.mark_completed()

        assert result.status == ExperimentStatus.COMPLETED
        assert result.completed_at is not None
        assert result.duration_seconds > 0

    def test_mark_failed(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
        )
        result.mark_running()
        time.sleep(0.01)
        result.mark_failed("Test error message")

        assert result.status == ExperimentStatus.FAILED
        assert result.error_message == "Test error message"
        assert result.completed_at is not None
        assert result.duration_seconds > 0

    def test_mark_paused(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
        )
        result.mark_running()
        result.mark_paused("/path/to/checkpoint")

        assert result.status == ExperimentStatus.PAUSED
        assert result.checkpoint_path == "/path/to/checkpoint"

    def test_mark_paused_no_checkpoint(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
        )
        result.mark_paused()

        assert result.status == ExperimentStatus.PAUSED
        assert result.checkpoint_path is None

    def test_mark_cancelled(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
        )
        result.mark_running()
        time.sleep(0.01)
        result.mark_cancelled()

        assert result.status == ExperimentStatus.CANCELLED
        assert result.completed_at is not None
        assert result.duration_seconds > 0

    def test_add_scenario_result(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
        )

        scenario1 = ScenarioResult(name="s1", status=ExperimentStatus.COMPLETED)
        scenario2 = ScenarioResult(name="s2", status=ExperimentStatus.FAILED)

        result.add_scenario_result(scenario1)
        result.add_scenario_result(scenario2)

        assert len(result.scenarios) == 2
        assert result.scenarios[0].name == "s1"
        assert result.scenarios[1].name == "s2"

    def test_update_metrics(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            metrics={"accuracy": 0.9},
        )

        result.update_metrics({"loss": 0.1, "f1": 0.85})

        assert result.metrics == {"accuracy": 0.9, "loss": 0.1, "f1": 0.85}

    def test_update_metrics_overwrites(self):
        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            metrics={"accuracy": 0.9},
        )

        result.update_metrics({"accuracy": 0.95})

        assert result.metrics["accuracy"] == 0.95


# ============================================================================
# ResultStorage Tests
# ============================================================================


class TestResultStorageInitialization:
    """Test ResultStorage initialization."""

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir) / "results"
            storage = ResultStorage(storage_dir)

            assert storage.base_dir.exists()
            assert storage.base_dir.is_dir()

    def test_creates_index_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            assert storage.index_file.exists()

            with open(storage.index_file) as f:
                data = json.load(f)
            assert data == {}

    def test_expands_user_home(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(Path(temp_dir))

            # Should be an absolute path
            assert storage.base_dir.is_absolute() or not str(storage.base_dir).startswith("~")

    def test_accepts_string_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            assert isinstance(storage.base_dir, Path)
            assert storage.base_dir.exists()

    def test_accepts_path_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(Path(temp_dir))

            assert isinstance(storage.base_dir, Path)


class TestResultStorageSaveLoad:
    """Test ResultStorage save and load operations."""

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="save-test",
                experiment_name="save-experiment",
            )

            path = storage.save(result)

            assert path.exists()
            assert path.name == "result.json"

    def test_save_returns_correct_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="exp-123",
                experiment_name="test-exp",
            )

            path = storage.save(result)

            assert "test-exp" in str(path) or "test_exp" in str(path)
            assert "exp-123" in str(path)

    def test_save_updates_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="indexed-exp",
                experiment_name="indexed-experiment",
                tags=["tag1"],
            )

            storage.save(result)

            with open(storage.index_file) as f:
                index = json.load(f)

            assert "indexed-exp" in index
            assert index["indexed-exp"]["experiment_name"] == "indexed-experiment"
            assert index["indexed-exp"]["tags"] == ["tag1"]

    def test_load_existing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            original = ExperimentResult(
                experiment_id="load-test",
                experiment_name="load-experiment",
                status=ExperimentStatus.COMPLETED,
                metrics={"accuracy": 0.95},
            )

            storage.save(original)
            loaded = storage.load("load-test")

            assert loaded is not None
            assert loaded.experiment_id == "load-test"
            assert loaded.experiment_name == "load-experiment"
            assert loaded.status == ExperimentStatus.COMPLETED
            assert loaded.metrics["accuracy"] == 0.95

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            result = storage.load("nonexistent-id")

            assert result is None

    def test_load_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="orphan-exp",
                experiment_name="orphan",
            )
            storage.save(result)

            # Remove the file but keep index entry
            result_dir = storage.base_dir / "orphan" / "orphan-exp"
            shutil.rmtree(result_dir)

            loaded = storage.load("orphan-exp")
            assert loaded is None

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            scenario = ScenarioResult(
                name="scenario1",
                status=ExperimentStatus.COMPLETED,
                steps_completed=3,
                steps_total=3,
            )

            original = ExperimentResult(
                experiment_id="roundtrip",
                experiment_name="roundtrip-test",
                status=ExperimentStatus.COMPLETED,
                seed=42,
                scenarios=[scenario],
                metrics={"accuracy": 0.95, "loss": 0.05},
                parameters={"lr": 0.01},
                tags=["test", "benchmark"],
            )

            storage.save(original)
            loaded = storage.load("roundtrip")

            assert loaded.experiment_id == original.experiment_id
            assert loaded.status == original.status
            assert loaded.seed == original.seed
            assert len(loaded.scenarios) == 1
            assert loaded.scenarios[0].name == "scenario1"
            assert loaded.metrics == original.metrics
            assert loaded.parameters == original.parameters
            assert loaded.tags == original.tags


class TestResultStorageExists:
    """Test ResultStorage exists method."""

    def test_exists_true(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="exists-test",
                experiment_name="exists-exp",
            )
            storage.save(result)

            assert storage.exists("exists-test") is True

    def test_exists_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            assert storage.exists("nonexistent") is False


class TestResultStorageDelete:
    """Test ResultStorage delete method."""

    def test_delete_existing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="delete-me",
                experiment_name="delete-exp",
            )
            storage.save(result)

            deleted = storage.delete("delete-me")

            assert deleted is True
            assert storage.exists("delete-me") is False
            assert storage.load("delete-me") is None

    def test_delete_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            deleted = storage.delete("nonexistent")

            assert deleted is False

    def test_delete_removes_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="cleanup",
                experiment_name="cleanup-exp",
            )
            path = storage.save(result)
            result_dir = path.parent

            storage.delete("cleanup")

            assert not result_dir.exists()


class TestResultStorageListResults:
    """Test ResultStorage list_results method."""

    def _create_test_results(self, storage: ResultStorage) -> list[ExperimentResult]:
        """Helper to create test results."""
        results = []
        for i in range(5):
            result = ExperimentResult(
                experiment_id=f"exp-{i}",
                experiment_name=f"experiment-{i % 2}",  # 0, 1, 0, 1, 0
                status=ExperimentStatus.COMPLETED if i % 2 == 0 else ExperimentStatus.FAILED,
                tags=[f"tag{i}", "common"],
            )
            # Set different start times
            result.started_at = datetime.now(timezone.utc) - timedelta(days=i)
            storage.save(result)
            results.append(result)
        return results

    def test_list_all_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            results = storage.list_results()

            assert len(results) == 5

    def test_list_filter_by_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            results = storage.list_results(experiment_name="experiment-0")

            assert len(results) == 3
            for r in results:
                assert r["experiment_name"] == "experiment-0"

    def test_list_filter_by_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            results = storage.list_results(status=ExperimentStatus.COMPLETED)

            assert len(results) == 3
            for r in results:
                assert r["status"] == "completed"

    def test_list_filter_by_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            results = storage.list_results(tags=["tag0"])

            assert len(results) == 1
            assert results[0]["experiment_id"] == "exp-0"

    def test_list_filter_by_tags_any_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            results = storage.list_results(tags=["common"])

            assert len(results) == 5

    def test_list_filter_by_since(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            since = datetime.now(timezone.utc) - timedelta(days=2, hours=1)
            results = storage.list_results(since=since)

            # Should only get exp-0, exp-1, exp-2 (within last 2 days)
            assert len(results) == 3

    def test_list_filter_by_until(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            until = datetime.now(timezone.utc) - timedelta(days=2, hours=1)
            results = storage.list_results(until=until)

            # Should only get exp-3, exp-4 (older than 2 days)
            assert len(results) == 2

    def test_list_with_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            results = storage.list_results(limit=3)

            assert len(results) == 3

    def test_list_sorted_by_date_descending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            results = storage.list_results()

            # Most recent should be first
            dates = [r["started_at"] for r in results]
            assert dates == sorted(dates, reverse=True)

    def test_list_combined_filters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._create_test_results(storage)

            results = storage.list_results(
                status=ExperimentStatus.COMPLETED,
                tags=["common"],
                limit=2,
            )

            assert len(results) == 2
            for r in results:
                assert r["status"] == "completed"

    def test_list_empty_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            results = storage.list_results()

            assert results == []


class TestResultStorageResultCount:
    """Test ResultStorage get_result_count method."""

    def test_count_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            assert storage.get_result_count() == 0

    def test_count_after_saves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            for i in range(5):
                result = ExperimentResult(
                    experiment_id=f"count-{i}",
                    experiment_name="count-test",
                )
                storage.save(result)

            assert storage.get_result_count() == 5

    def test_count_after_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            for i in range(3):
                result = ExperimentResult(
                    experiment_id=f"count-{i}",
                    experiment_name="count-test",
                )
                storage.save(result)

            storage.delete("count-1")

            assert storage.get_result_count() == 2


class TestResultStorageExport:
    """Test ResultStorage export methods."""

    def test_export_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="export-json",
                experiment_name="export-test",
                metrics={"accuracy": 0.95},
            )
            storage.save(result)

            content = storage.export("export-json", OutputFormat.JSON)
            data = json.loads(content)

            assert data["experiment_id"] == "export-json"
            assert data["metrics"]["accuracy"] == 0.95

    def test_export_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="export-yaml",
                experiment_name="export-test",
                tags=["tag1", "tag2"],
            )
            storage.save(result)

            content = storage.export("export-yaml", OutputFormat.YAML)
            data = yaml.safe_load(content)

            assert data["experiment_id"] == "export-yaml"
            assert data["tags"] == ["tag1", "tag2"]

    def test_export_to_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="export-file",
                experiment_name="export-test",
            )
            storage.save(result)

            output_path = Path(temp_dir) / "exports" / "result.json"
            storage.export("export-file", OutputFormat.JSON, output_path)

            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)
            assert data["experiment_id"] == "export-file"

    def test_export_excludes_raw_responses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            scenario = ScenarioResult(
                name="scenario1",
                status=ExperimentStatus.COMPLETED,
                outputs={"step1": "raw output data"},
            )
            result = ExperimentResult(
                experiment_id="export-filtered",
                experiment_name="export-test",
                scenarios=[scenario],
            )
            storage.save(result)

            content = storage.export(
                "export-filtered",
                OutputFormat.JSON,
                include_raw_responses=False,
            )
            data = json.loads(content)

            # Outputs should be removed
            assert "outputs" not in data["scenarios"][0]

    def test_export_includes_raw_responses_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            scenario = ScenarioResult(
                name="scenario1",
                status=ExperimentStatus.COMPLETED,
                outputs={"step1": "raw output data"},
            )
            result = ExperimentResult(
                experiment_id="export-full",
                experiment_name="export-test",
                scenarios=[scenario],
            )
            storage.save(result)

            content = storage.export(
                "export-full",
                OutputFormat.JSON,
                include_raw_responses=True,
            )
            data = json.loads(content)

            assert data["scenarios"][0]["outputs"] == {"step1": "raw output data"}

    def test_export_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            with pytest.raises(ValueError, match="Experiment not found"):
                storage.export("nonexistent")


class TestResultStorageExportAll:
    """Test ResultStorage export_all method."""

    def test_export_all_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            for i in range(3):
                result = ExperimentResult(
                    experiment_id=f"batch-{i}",
                    experiment_name="batch-test",
                )
                storage.save(result)

            export_dir = Path(temp_dir) / "exports"
            paths = storage.export_all(export_dir, OutputFormat.JSON)

            assert len(paths) == 3
            for path in paths:
                assert path.exists()
                assert path.suffix == ".json"

    def test_export_all_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            for i in range(2):
                result = ExperimentResult(
                    experiment_id=f"yaml-{i}",
                    experiment_name="yaml-test",
                )
                storage.save(result)

            export_dir = Path(temp_dir) / "exports"
            paths = storage.export_all(export_dir, OutputFormat.YAML)

            assert len(paths) == 2
            for path in paths:
                assert path.suffix == ".yaml"

    def test_export_all_with_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            result1 = ExperimentResult(
                experiment_id="completed",
                experiment_name="test",
                status=ExperimentStatus.COMPLETED,
            )
            result2 = ExperimentResult(
                experiment_id="failed",
                experiment_name="test",
                status=ExperimentStatus.FAILED,
            )
            storage.save(result1)
            storage.save(result2)

            export_dir = Path(temp_dir) / "exports"
            paths = storage.export_all(
                export_dir,
                filter_status=ExperimentStatus.COMPLETED,
            )

            assert len(paths) == 1
            assert "completed" in str(paths[0])


class TestResultStorageCleanup:
    """Test ResultStorage cleanup_old_results method."""

    def test_cleanup_old_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            # Create old results
            old_result = ExperimentResult(
                experiment_id="old-exp",
                experiment_name="old-test",
                status=ExperimentStatus.FAILED,
            )
            old_result.started_at = datetime.now(timezone.utc) - timedelta(days=60)
            storage.save(old_result)

            # Create new result
            new_result = ExperimentResult(
                experiment_id="new-exp",
                experiment_name="new-test",
            )
            storage.save(new_result)

            deleted = storage.cleanup_old_results(days=30)

            assert "old-exp" in deleted
            assert "new-exp" not in deleted
            assert not storage.exists("old-exp")
            assert storage.exists("new-exp")

    def test_cleanup_keeps_successful(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            # Create old successful result
            old_success = ExperimentResult(
                experiment_id="old-success",
                experiment_name="old-test",
                status=ExperimentStatus.COMPLETED,
            )
            old_success.started_at = datetime.now(timezone.utc) - timedelta(days=60)
            storage.save(old_success)

            deleted = storage.cleanup_old_results(days=30, keep_successful=True)

            assert "old-success" not in deleted
            assert storage.exists("old-success")

    def test_cleanup_dry_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            # Create old result
            old_result = ExperimentResult(
                experiment_id="dry-run-exp",
                experiment_name="dry-run",
                status=ExperimentStatus.FAILED,
            )
            old_result.started_at = datetime.now(timezone.utc) - timedelta(days=60)
            storage.save(old_result)

            deleted = storage.cleanup_old_results(days=30, dry_run=True)

            assert "dry-run-exp" in deleted
            # Should NOT be deleted in dry run
            assert storage.exists("dry-run-exp")


class TestResultStorageThreadSafety:
    """Test ResultStorage thread safety."""

    def test_concurrent_saves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            errors = []

            def save_result(thread_id: int):
                try:
                    for i in range(10):
                        result = ExperimentResult(
                            experiment_id=f"thread-{thread_id}-exp-{i}",
                            experiment_name=f"concurrent-{thread_id}",
                        )
                        storage.save(result)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=save_result, args=(tid,))
                for tid in range(5)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert storage.get_result_count() == 50

    def test_concurrent_reads_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            errors = []

            # Seed some data
            for i in range(5):
                result = ExperimentResult(
                    experiment_id=f"seed-{i}",
                    experiment_name="seed",
                )
                storage.save(result)

            def writer():
                try:
                    for i in range(10):
                        result = ExperimentResult(
                            experiment_id=f"write-{i}",
                            experiment_name="writer",
                        )
                        storage.save(result)
                except Exception as e:
                    errors.append(e)

            def reader():
                try:
                    for _ in range(20):
                        storage.list_results()
                        storage.load("seed-0")
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=writer),
                threading.Thread(target=reader),
                threading.Thread(target=reader),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0


class TestResultStorageEdgeCases:
    """Test edge cases for ResultStorage."""

    def test_special_characters_in_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="special-chars",
                experiment_name="test/with:special@chars!",
            )

            path = storage.save(result)

            # Should sanitize the name
            assert path.exists()
            loaded = storage.load("special-chars")
            assert loaded is not None
            assert loaded.experiment_name == "test/with:special@chars!"

    def test_unicode_in_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="unicode-exp",
                experiment_name="实验名称",
            )

            storage.save(result)
            loaded = storage.load("unicode-exp")

            assert loaded is not None
            assert loaded.experiment_name == "实验名称"

    def test_very_long_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            long_name = "a" * 200
            result = ExperimentResult(
                experiment_id="long-name-exp",
                experiment_name=long_name,
            )

            storage.save(result)
            loaded = storage.load("long-name-exp")

            assert loaded is not None
            assert loaded.experiment_name == long_name

    def test_empty_experiment_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result = ExperimentResult(
                experiment_id="empty-name-exp",
                experiment_name="",
            )

            storage.save(result)
            loaded = storage.load("empty-name-exp")

            assert loaded is not None
            assert loaded.experiment_name == ""


# ============================================================================
# Query Function Tests
# ============================================================================


class TestQueryExperiments:
    """Test query_experiments function."""

    def _setup_test_data(self, storage: ResultStorage) -> None:
        """Create test experiments."""
        for i in range(10):
            result = ExperimentResult(
                experiment_id=f"query-exp-{i}",
                experiment_name=f"query-test-{i % 3}",
                status=ExperimentStatus.COMPLETED if i % 2 == 0 else ExperimentStatus.FAILED,
                tags=[f"tag{i % 2}", "common"],
            )
            result.started_at = datetime.now(timezone.utc) - timedelta(days=i)
            storage.save(result)

    def test_query_all(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._setup_test_data(storage)

            results = query_experiments(storage)

            assert len(results) == 10

    def test_query_by_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._setup_test_data(storage)

            results = query_experiments(storage, name="query-test-0")

            assert len(results) == 4

    def test_query_by_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._setup_test_data(storage)

            results = query_experiments(storage, status=ExperimentStatus.COMPLETED)

            assert len(results) == 5

    def test_query_by_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._setup_test_data(storage)

            results = query_experiments(storage, tags=["tag0"])

            assert len(results) == 5

    def test_query_with_predicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._setup_test_data(storage)

            # Custom predicate: only experiments with ID containing "5" or "7"
            results = query_experiments(
                storage,
                predicate=lambda x: "5" in x["experiment_id"] or "7" in x["experiment_id"],
            )

            assert len(results) == 2

    def test_query_with_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._setup_test_data(storage)

            results = query_experiments(storage, limit=3)

            assert len(results) == 3

    def test_query_combined_with_predicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            self._setup_test_data(storage)

            results = query_experiments(
                storage,
                status=ExperimentStatus.COMPLETED,
                predicate=lambda x: int(x["experiment_id"].split("-")[-1]) > 5,
                limit=2,
            )

            assert len(results) == 2


class TestGetLatestResult:
    """Test get_latest_result function."""

    def test_get_latest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            for i in range(3):
                result = ExperimentResult(
                    experiment_id=f"latest-{i}",
                    experiment_name="test-experiment",
                )
                result.started_at = datetime.now(timezone.utc) - timedelta(days=i)
                storage.save(result)

            latest = get_latest_result(storage, "test-experiment")

            assert latest is not None
            assert latest.experiment_id == "latest-0"  # Most recent

    def test_get_latest_with_status_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            # Create latest as failed
            for i, status in enumerate([
                ExperimentStatus.FAILED,
                ExperimentStatus.COMPLETED,
                ExperimentStatus.COMPLETED,
            ]):
                result = ExperimentResult(
                    experiment_id=f"status-{i}",
                    experiment_name="test-experiment",
                    status=status,
                )
                result.started_at = datetime.now(timezone.utc) - timedelta(days=i)
                storage.save(result)

            latest = get_latest_result(
                storage,
                "test-experiment",
                status=ExperimentStatus.COMPLETED,
            )

            assert latest is not None
            assert latest.experiment_id == "status-1"

    def test_get_latest_no_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            latest = get_latest_result(storage, "nonexistent")

            assert latest is None


class TestFindByConfigHash:
    """Test find_by_config_hash function."""

    def test_find_by_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            # Create experiments with same config hash
            for i in range(3):
                result = ExperimentResult(
                    experiment_id=f"hash-{i}",
                    experiment_name="hash-test",
                    config_hash="abc123",
                )
                storage.save(result)

            # Create one with different hash
            result = ExperimentResult(
                experiment_id="different-hash",
                experiment_name="hash-test",
                config_hash="xyz789",
            )
            storage.save(result)

            results = find_by_config_hash(storage, "abc123")

            assert len(results) == 3
            for r in results:
                assert r.config_hash == "abc123"

    def test_find_by_hash_not_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            results = find_by_config_hash(storage, "nonexistent")

            assert results == []


class TestFindByTag:
    """Test find_by_tag function."""

    def test_find_by_tag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            for i in range(4):
                result = ExperimentResult(
                    experiment_id=f"tag-exp-{i}",
                    experiment_name="tag-test",
                    tags=["common", f"unique-{i}"],
                )
                storage.save(result)

            results = find_by_tag(storage, "common")

            assert len(results) == 4

    def test_find_by_unique_tag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            for i in range(3):
                result = ExperimentResult(
                    experiment_id=f"unique-{i}",
                    experiment_name="tag-test",
                    tags=[f"tag-{i}"],
                )
                storage.save(result)

            results = find_by_tag(storage, "tag-1")

            assert len(results) == 1
            assert results[0].experiment_id == "unique-1"

    def test_find_by_tag_not_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            results = find_by_tag(storage, "nonexistent")

            assert results == []


class TestAggregateMetrics:
    """Test aggregate_metrics function."""

    def test_aggregate_basic(self):
        results = [
            ExperimentResult(
                experiment_id=f"agg-{i}",
                experiment_name="agg-test",
                metrics={"accuracy": 0.9 + i * 0.01},
            )
            for i in range(5)
        ]

        agg = aggregate_metrics(results, "accuracy")

        assert agg["count"] == 5
        assert "mean" in agg
        assert "std" in agg
        assert "min" in agg
        assert "max" in agg
        assert "sum" in agg
        assert agg["min"] == 0.9
        assert agg["max"] == 0.94

    def test_aggregate_missing_metric(self):
        results = [
            ExperimentResult(
                experiment_id=f"missing-{i}",
                experiment_name="missing-test",
                metrics={"other": 1.0},
            )
            for i in range(3)
        ]

        agg = aggregate_metrics(results, "nonexistent")

        assert agg["count"] == 0
        assert "mean" not in agg

    def test_aggregate_partial_metrics(self):
        results = [
            ExperimentResult(
                experiment_id="has-metric",
                experiment_name="partial",
                metrics={"accuracy": 0.9},
            ),
            ExperimentResult(
                experiment_id="no-metric",
                experiment_name="partial",
                metrics={},
            ),
            ExperimentResult(
                experiment_id="has-metric-2",
                experiment_name="partial",
                metrics={"accuracy": 0.8},
            ),
        ]

        agg = aggregate_metrics(results, "accuracy")

        assert agg["count"] == 2
        assert agg["mean"] == 0.85

    def test_aggregate_empty_results(self):
        agg = aggregate_metrics([], "accuracy")

        assert agg["count"] == 0

    def test_aggregate_single_result(self):
        results = [
            ExperimentResult(
                experiment_id="single",
                experiment_name="single",
                metrics={"value": 42.0},
            )
        ]

        agg = aggregate_metrics(results, "value")

        assert agg["count"] == 1
        assert agg["mean"] == 42.0
        assert agg["min"] == 42.0
        assert agg["max"] == 42.0

    def test_aggregate_non_numeric_ignored(self):
        results = [
            ExperimentResult(
                experiment_id="numeric",
                experiment_name="test",
                metrics={"value": 10.0},
            ),
            ExperimentResult(
                experiment_id="string",
                experiment_name="test",
                metrics={"value": "not a number"},
            ),
        ]

        agg = aggregate_metrics(results, "value")

        assert agg["count"] == 1  # Only numeric counted


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateStorage:
    """Test create_storage factory function."""

    def test_creates_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = create_storage(temp_dir)

            assert isinstance(storage, ResultStorage)
            assert storage.base_dir == Path(temp_dir)

    def test_creates_with_path_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = create_storage(Path(temp_dir))

            assert isinstance(storage, ResultStorage)

    def test_default_path(self):
        storage = create_storage()

        assert isinstance(storage, ResultStorage)
        assert "experiments" in str(storage.base_dir)
        assert "results" in str(storage.base_dir)


class TestCreateStorageFromConfig:
    """Test create_storage_from_config factory function."""

    def test_from_config_with_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "storage": {
                    "path": temp_dir,
                }
            }

            storage = create_storage_from_config(config)

            assert isinstance(storage, ResultStorage)
            assert str(storage.base_dir) == temp_dir

    def test_from_config_default_path(self):
        config = {"storage": {}}

        storage = create_storage_from_config(config)

        assert isinstance(storage, ResultStorage)

    def test_from_empty_config(self):
        storage = create_storage_from_config({})

        assert isinstance(storage, ResultStorage)


# ============================================================================
# Disk Space Check Tests
# ============================================================================


class TestDiskSpaceCheck:
    """Test disk space checking functionality."""

    def test_check_disk_space_sufficient(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            # 1MB should be available on any test system
            assert storage._check_disk_space(1024 * 1024) is True

    def test_check_disk_space_handles_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)

            # Should handle errors gracefully
            with mock.patch("os.statvfs", side_effect=OSError("Test error")):
                result = storage._check_disk_space()
                assert result is True  # Falls back to True on error
