"""Tests for export functionality.

This module tests the ExperimentExporter class which provides multi-format
export capabilities for persisted experiments (JSON, CSV, Parquet).
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from src.persistence import ExperimentPersistence
from src.persistence.export import ExperimentExporter, ExportError


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()
    # Also clean up WAL and SHM files
    for ext in ["-wal", "-shm"]:
        wal_path = Path(str(path) + ext)
        if wal_path.exists():
            wal_path.unlink()


@pytest.fixture
def temp_storage_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for HDF5 storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_export_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for export output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def persistence(
    temp_db_path: Path, temp_storage_dir: Path
) -> Generator[ExperimentPersistence, None, None]:
    """Create an ExperimentPersistence instance for testing."""
    ep = ExperimentPersistence(db_path=temp_db_path, storage_dir=temp_storage_dir)
    yield ep
    ep.close()


@pytest.fixture
def exporter(
    temp_db_path: Path, temp_storage_dir: Path
) -> Generator[ExperimentExporter, None, None]:
    """Create an ExperimentExporter instance for testing."""
    exp = ExperimentExporter(db_path=temp_db_path, storage_dir=temp_storage_dir)
    yield exp
    exp.close()


@pytest.fixture
def populated_persistence(
    persistence: ExperimentPersistence,
) -> ExperimentPersistence:
    """Create persistence with pre-populated test data."""
    # Create test experiment with all data types
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]
    hidden_states = {
        -1: np.random.randn(1, 768).astype(np.float32),
        0: np.random.randn(1, 768).astype(np.float32),
    }
    metrics = {
        "latency_ms": 150.5,
        "tokens_per_second": 30.2,
        "perplexity": 12.5,
    }

    persistence.persist_experiment(
        model="test-model-7b",
        experiment_id="exp-001",
        messages=messages,
        hidden_states=hidden_states,
        metrics=metrics,
        config={"temperature": 0.7, "max_tokens": 100},
        notes="Test experiment for export",
    )

    return persistence


class TestExperimentExporterInit:
    """Tests for ExperimentExporter initialization."""

    def test_exporter_init(self, temp_db_path: Path, temp_storage_dir: Path):
        """Test that exporter initializes correctly."""
        exporter = ExperimentExporter(
            db_path=temp_db_path, storage_dir=temp_storage_dir
        )
        try:
            assert exporter is not None
        finally:
            exporter.close()

    def test_exporter_close(self, temp_db_path: Path, temp_storage_dir: Path):
        """Test that exporter closes without error."""
        exporter = ExperimentExporter(
            db_path=temp_db_path, storage_dir=temp_storage_dir
        )
        exporter.close()
        # Should not raise


class TestJSONExport:
    """Tests for JSON export functionality."""

    def test_export_json_basic(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test basic JSON export of an experiment."""
        output_path = temp_export_dir / "export.json"

        result_path = exporter.export_json("exp-001", output_path)

        assert result_path == output_path
        assert output_path.exists()

        # Verify JSON structure
        with open(output_path) as f:
            data = json.load(f)

        assert "export_timestamp" in data
        assert "experiment" in data
        assert "conversations" in data
        assert "metrics" in data
        assert "hidden_states" in data

    def test_export_json_experiment_data(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that JSON export contains correct experiment data."""
        output_path = temp_export_dir / "export.json"

        exporter.export_json("exp-001", output_path)

        with open(output_path) as f:
            data = json.load(f)

        exp = data["experiment"]
        assert exp["id"] == "exp-001"
        assert exp["model"] == "test-model-7b"
        assert exp["status"] == "completed"
        assert exp["notes"] == "Test experiment for export"
        assert exp["config"]["temperature"] == 0.7

    def test_export_json_conversations(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that JSON export contains correct conversation data."""
        output_path = temp_export_dir / "export.json"

        exporter.export_json("exp-001", output_path)

        with open(output_path) as f:
            data = json.load(f)

        convs = data["conversations"]
        assert len(convs) == 3

        assert convs[0]["role"] == "system"
        assert convs[1]["role"] == "user"
        assert convs[2]["role"] == "assistant"

        # Verify sequence order
        for i, conv in enumerate(convs):
            assert conv["sequence_num"] == i

    def test_export_json_metrics(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that JSON export contains correct metric data."""
        output_path = temp_export_dir / "export.json"

        exporter.export_json("exp-001", output_path)

        with open(output_path) as f:
            data = json.load(f)

        metrics = data["metrics"]
        assert len(metrics) == 3

        metric_names = {m["name"] for m in metrics}
        assert metric_names == {"latency_ms", "tokens_per_second", "perplexity"}

    def test_export_json_hidden_states_metadata(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that JSON export contains hidden state metadata."""
        output_path = temp_export_dir / "export.json"

        exporter.export_json("exp-001", output_path)

        with open(output_path) as f:
            data = json.load(f)

        hs = data["hidden_states"]
        assert len(hs) == 2

        layers = {h["layer"] for h in hs}
        assert layers == {-1, 0}

        # Check shape and dtype are included
        for h in hs:
            assert "shape" in h
            assert "dtype" in h
            assert "file_path" in h

    def test_export_json_with_hidden_state_data(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test JSON export with actual hidden state arrays included."""
        output_path = temp_export_dir / "export.json"

        exporter.export_json("exp-001", output_path, include_hidden_state_data=True)

        with open(output_path) as f:
            data = json.load(f)

        assert "hidden_state_data" in data
        assert "-1" in data["hidden_state_data"]
        assert "0" in data["hidden_state_data"]

        # Verify arrays are lists of floats
        assert isinstance(data["hidden_state_data"]["-1"], list)
        assert len(data["hidden_state_data"]["-1"]) > 0

    def test_export_json_custom_indent(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test JSON export with custom indentation."""
        output_path = temp_export_dir / "export.json"

        exporter.export_json("exp-001", output_path, indent=4)

        # Read raw content to check indentation
        content = output_path.read_text()
        # Should have multiple lines with 4-space indent
        lines = content.split("\n")
        assert any(line.startswith("    ") for line in lines)

    def test_export_json_not_found(
        self,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test JSON export raises error for non-existent experiment."""
        output_path = temp_export_dir / "export.json"

        with pytest.raises(ExportError) as exc_info:
            exporter.export_json("nonexistent", output_path)

        assert "not found" in str(exc_info.value).lower()

    def test_export_json_creates_parent_dirs(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that JSON export creates parent directories."""
        output_path = temp_export_dir / "subdir" / "nested" / "export.json"

        exporter.export_json("exp-001", output_path)

        assert output_path.exists()


class TestCSVExport:
    """Tests for CSV export functionality."""

    def test_export_csv_creates_files(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that CSV export creates multiple files."""
        base_path = temp_export_dir / "export"

        result = exporter.export_csv("exp-001", base_path)

        assert "experiment" in result
        assert result["experiment"].exists()
        assert result["experiment"].name == "export_experiment.csv"

    def test_export_csv_experiment_file(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that CSV experiment file has correct data."""
        base_path = temp_export_dir / "export"

        result = exporter.export_csv("exp-001", base_path)

        df = pd.read_csv(result["experiment"])

        assert len(df) == 1
        assert df.iloc[0]["id"] == "exp-001"
        assert df.iloc[0]["model"] == "test-model-7b"

    def test_export_csv_conversations_file(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that CSV conversations file has correct data."""
        base_path = temp_export_dir / "export"

        result = exporter.export_csv("exp-001", base_path)

        assert "conversations" in result
        df = pd.read_csv(result["conversations"])

        assert len(df) == 3
        assert list(df["role"]) == ["system", "user", "assistant"]

    def test_export_csv_metrics_file(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that CSV metrics file has correct data."""
        base_path = temp_export_dir / "export"

        result = exporter.export_csv("exp-001", base_path)

        assert "metrics" in result
        df = pd.read_csv(result["metrics"])

        assert len(df) == 3
        assert set(df["name"]) == {"latency_ms", "tokens_per_second", "perplexity"}

    def test_export_csv_hidden_states_file(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that CSV hidden states file has correct data."""
        base_path = temp_export_dir / "export"

        result = exporter.export_csv("exp-001", base_path)

        assert "hidden_states" in result
        df = pd.read_csv(result["hidden_states"])

        assert len(df) == 2
        assert set(df["layer"]) == {-1, 0}

    def test_export_csv_without_metadata(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test CSV export without metadata."""
        base_path = temp_export_dir / "export"

        result = exporter.export_csv("exp-001", base_path, include_metadata=False)

        df = pd.read_csv(result["experiment"])
        assert "config" not in df.columns

    def test_export_csv_strips_extension(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that .csv extension is stripped from base path."""
        base_path = temp_export_dir / "export.csv"

        result = exporter.export_csv("exp-001", base_path)

        # Should not double the .csv extension
        assert result["experiment"].name == "export_experiment.csv"

    def test_export_csv_not_found(
        self,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test CSV export raises error for non-existent experiment."""
        base_path = temp_export_dir / "export"

        with pytest.raises(ExportError) as exc_info:
            exporter.export_csv("nonexistent", base_path)

        assert "not found" in str(exc_info.value).lower()


class TestParquetExport:
    """Tests for Parquet export functionality."""

    def test_export_parquet_creates_files(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that Parquet export creates multiple files."""
        base_path = temp_export_dir / "export"

        result = exporter.export_parquet("exp-001", base_path)

        assert "experiment" in result
        assert result["experiment"].exists()
        assert result["experiment"].suffix == ".parquet"

    def test_export_parquet_experiment_file(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that Parquet experiment file has correct data."""
        base_path = temp_export_dir / "export"

        result = exporter.export_parquet("exp-001", base_path)

        df = pd.read_parquet(result["experiment"])

        assert len(df) == 1
        assert df.iloc[0]["id"] == "exp-001"
        assert df.iloc[0]["model"] == "test-model-7b"

    def test_export_parquet_conversations_file(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that Parquet conversations file has correct data."""
        base_path = temp_export_dir / "export"

        result = exporter.export_parquet("exp-001", base_path)

        assert "conversations" in result
        df = pd.read_parquet(result["conversations"])

        assert len(df) == 3
        assert list(df["role"]) == ["system", "user", "assistant"]

    def test_export_parquet_metrics_file(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that Parquet metrics file has correct data."""
        base_path = temp_export_dir / "export"

        result = exporter.export_parquet("exp-001", base_path)

        assert "metrics" in result
        df = pd.read_parquet(result["metrics"])

        assert len(df) == 3
        assert set(df["name"]) == {"latency_ms", "tokens_per_second", "perplexity"}

    def test_export_parquet_with_compression(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test Parquet export with different compression codecs."""
        for compression in ["snappy", "gzip"]:
            base_path = temp_export_dir / f"export_{compression}"

            result = exporter.export_parquet(
                "exp-001", base_path, compression=compression
            )

            assert result["experiment"].exists()
            # Verify file is readable
            df = pd.read_parquet(result["experiment"])
            assert len(df) == 1

    def test_export_parquet_strips_extension(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that .parquet extension is stripped from base path."""
        base_path = temp_export_dir / "export.parquet"

        result = exporter.export_parquet("exp-001", base_path)

        # Should not double the .parquet extension
        assert result["experiment"].name == "export_experiment.parquet"

    def test_export_parquet_not_found(
        self,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test Parquet export raises error for non-existent experiment."""
        base_path = temp_export_dir / "export"

        with pytest.raises(ExportError) as exc_info:
            exporter.export_parquet("nonexistent", base_path)

        assert "not found" in str(exc_info.value).lower()


class TestBatchExport:
    """Tests for batch export functionality."""

    @pytest.fixture
    def multi_experiment_persistence(
        self, persistence: ExperimentPersistence
    ) -> ExperimentPersistence:
        """Create persistence with multiple test experiments."""
        for i in range(3):
            messages = [
                {"role": "user", "content": f"Message {i}"},
                {"role": "assistant", "content": f"Response {i}"},
            ]
            persistence.persist_experiment(
                model=f"model-{i}",
                experiment_id=f"exp-{i:03d}",
                messages=messages,
                config={"index": i},
            )
        return persistence

    def test_export_experiments_json(
        self,
        multi_experiment_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test batch export to JSON format."""
        result = exporter.export_experiments(
            experiment_ids=["exp-000", "exp-001", "exp-002"],
            output_dir=temp_export_dir,
            format="json",
        )

        assert len(result) == 3
        for exp_id in ["exp-000", "exp-001", "exp-002"]:
            assert exp_id in result
            assert Path(result[exp_id]).exists()
            assert Path(result[exp_id]).suffix == ".json"

    def test_export_experiments_csv(
        self,
        multi_experiment_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test batch export to CSV format."""
        result = exporter.export_experiments(
            experiment_ids=["exp-000", "exp-001"],
            output_dir=temp_export_dir,
            format="csv",
        )

        assert len(result) == 2
        for exp_id in ["exp-000", "exp-001"]:
            assert exp_id in result
            assert isinstance(result[exp_id], dict)
            assert "experiment" in result[exp_id]

    def test_export_experiments_parquet(
        self,
        multi_experiment_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test batch export to Parquet format."""
        result = exporter.export_experiments(
            experiment_ids=["exp-000", "exp-001"],
            output_dir=temp_export_dir,
            format="parquet",
        )

        assert len(result) == 2
        for exp_id in ["exp-000", "exp-001"]:
            assert exp_id in result
            assert "experiment" in result[exp_id]

    def test_export_experiments_skips_missing(
        self,
        multi_experiment_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that batch export skips non-existent experiments."""
        result = exporter.export_experiments(
            experiment_ids=["exp-000", "nonexistent", "exp-001"],
            output_dir=temp_export_dir,
            format="json",
        )

        # Should skip the nonexistent one
        assert len(result) == 2
        assert "exp-000" in result
        assert "exp-001" in result
        assert "nonexistent" not in result

    def test_export_experiments_invalid_format(
        self,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that batch export raises error for invalid format."""
        with pytest.raises(ExportError) as exc_info:
            exporter.export_experiments(
                experiment_ids=["exp-000"],
                output_dir=temp_export_dir,
                format="invalid",
            )

        assert "invalid" in str(exc_info.value).lower()


class TestSummaryExport:
    """Tests for summary export functionality."""

    @pytest.fixture
    def multi_experiment_persistence(
        self, persistence: ExperimentPersistence
    ) -> ExperimentPersistence:
        """Create persistence with multiple test experiments."""
        for i in range(5):
            messages = [
                {"role": "user", "content": f"Message {i}"},
            ]
            hidden_states = {-1: np.random.randn(1, 128).astype(np.float32)}
            metrics = {"latency": float(100 + i)}

            persistence.persist_experiment(
                model=f"model-{i % 2}",
                experiment_id=f"exp-{i:03d}",
                messages=messages,
                hidden_states=hidden_states,
                metrics=metrics,
            )
        return persistence

    def test_export_summary_csv(
        self,
        multi_experiment_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test summary export to CSV format."""
        output_path = temp_export_dir / "summary.csv"

        result = exporter.export_summary(output_path, format="csv")

        assert result == output_path
        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 5
        assert "id" in df.columns
        assert "conversation_count" in df.columns
        assert "metric_count" in df.columns
        assert "hidden_state_count" in df.columns

    def test_export_summary_json(
        self,
        multi_experiment_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test summary export to JSON format."""
        output_path = temp_export_dir / "summary.json"

        result = exporter.export_summary(output_path, format="json")

        assert result == output_path
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert "export_timestamp" in data
        assert data["total_experiments"] == 5
        assert len(data["experiments"]) == 5

    def test_export_summary_parquet(
        self,
        multi_experiment_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test summary export to Parquet format."""
        output_path = temp_export_dir / "summary.parquet"

        result = exporter.export_summary(output_path, format="parquet")

        assert result == output_path
        assert output_path.exists()

        df = pd.read_parquet(output_path)
        assert len(df) == 5

    def test_export_summary_with_limit(
        self,
        multi_experiment_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test summary export with limit on number of experiments."""
        output_path = temp_export_dir / "summary.csv"

        exporter.export_summary(output_path, format="csv", limit=3)

        df = pd.read_csv(output_path)
        assert len(df) == 3

    def test_export_summary_counts_correct(
        self,
        multi_experiment_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that summary export has correct counts."""
        output_path = temp_export_dir / "summary.csv"

        exporter.export_summary(output_path, format="csv")

        df = pd.read_csv(output_path)

        # Each experiment has 1 conversation, 1 metric, 1 hidden state
        assert all(df["conversation_count"] == 1)
        assert all(df["metric_count"] == 1)
        assert all(df["hidden_state_count"] == 1)

    def test_export_summary_invalid_format(
        self,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that summary export raises error for invalid format."""
        output_path = temp_export_dir / "summary.txt"

        with pytest.raises(ExportError) as exc_info:
            exporter.export_summary(output_path, format="invalid")

        assert "invalid" in str(exc_info.value).lower()


class TestExportError:
    """Tests for ExportError exception."""

    def test_export_error_is_exception(self):
        """Test that ExportError is an exception."""
        error = ExportError("Test error message")
        assert isinstance(error, Exception)

    def test_export_error_message(self):
        """Test that ExportError preserves message."""
        error = ExportError("Specific error message")
        assert str(error) == "Specific error message"


class TestExportDataIntegrity:
    """Tests for data integrity in exports."""

    def test_json_roundtrip_preserves_data(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that JSON export preserves all data accurately."""
        output_path = temp_export_dir / "export.json"

        exporter.export_json("exp-001", output_path)

        with open(output_path) as f:
            data = json.load(f)

        # Verify experiment data matches original
        exp = data["experiment"]
        assert exp["config"]["temperature"] == 0.7
        assert exp["config"]["max_tokens"] == 100

        # Verify metric values match
        metrics = {m["name"]: m["value"] for m in data["metrics"]}
        assert abs(metrics["latency_ms"] - 150.5) < 0.01
        assert abs(metrics["tokens_per_second"] - 30.2) < 0.01
        assert abs(metrics["perplexity"] - 12.5) < 0.01

    def test_csv_preserves_conversation_order(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that CSV export preserves conversation order."""
        base_path = temp_export_dir / "export"

        result = exporter.export_csv("exp-001", base_path)

        df = pd.read_csv(result["conversations"])

        # Verify sequence numbers are in order
        assert list(df["sequence_num"]) == [0, 1, 2]
        assert list(df["role"]) == ["system", "user", "assistant"]

    def test_hidden_state_shape_preserved(
        self,
        populated_persistence: ExperimentPersistence,
        exporter: ExperimentExporter,
        temp_export_dir: Path,
    ):
        """Test that hidden state shapes are preserved in export."""
        output_path = temp_export_dir / "export.json"

        exporter.export_json("exp-001", output_path)

        with open(output_path) as f:
            data = json.load(f)

        for hs in data["hidden_states"]:
            assert hs["shape"] == [1, 768]
            assert hs["dtype"] == "float32"
