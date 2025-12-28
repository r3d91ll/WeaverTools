"""Tests for persistence layer database operations.

This module tests the persistence components including DatabaseManager,
HiddenStateStorage, ExperimentPersistence, and ExperimentQuery.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from src.persistence.database import (
    ConversationRecord,
    DatabaseManager,
    ExperimentRecord,
    HiddenStateRecord,
    MetricRecord,
    init_db,
)
from src.persistence.query import ExperimentQuery, ExperimentSummary
from src.persistence.storage import HiddenStateStorage
from src.persistence import ExperimentPersistence


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
def db_manager(temp_db_path: Path) -> Generator[DatabaseManager, None, None]:
    """Create a DatabaseManager instance for testing."""
    manager = DatabaseManager(temp_db_path)
    yield manager
    manager.close()


@pytest.fixture
def hidden_storage(temp_storage_dir: Path) -> HiddenStateStorage:
    """Create a HiddenStateStorage instance for testing."""
    return HiddenStateStorage(storage_dir=temp_storage_dir)


@pytest.fixture
def persistence(temp_db_path: Path, temp_storage_dir: Path) -> Generator[ExperimentPersistence, None, None]:
    """Create an ExperimentPersistence instance for testing."""
    ep = ExperimentPersistence(db_path=temp_db_path, storage_dir=temp_storage_dir)
    yield ep
    ep.close()


@pytest.fixture
def sample_experiment() -> ExperimentRecord:
    """Create a sample experiment record for testing."""
    return ExperimentRecord(
        id="exp-001",
        created_at=datetime.utcnow(),
        model="test-model-7b",
        config={"temperature": 0.7, "max_tokens": 100},
        status="completed",
        notes="Test experiment",
    )


@pytest.fixture
def sample_conversations() -> list[ConversationRecord]:
    """Create sample conversation records for testing."""
    now = datetime.utcnow()
    return [
        ConversationRecord(
            experiment_id="exp-001",
            sequence_num=0,
            timestamp=now,
            role="system",
            content="You are a helpful assistant.",
        ),
        ConversationRecord(
            experiment_id="exp-001",
            sequence_num=1,
            timestamp=now,
            role="user",
            content="Hello, how are you?",
        ),
        ConversationRecord(
            experiment_id="exp-001",
            sequence_num=2,
            timestamp=now,
            role="assistant",
            content="I'm doing well, thank you for asking!",
        ),
    ]


@pytest.fixture
def sample_metrics() -> list[MetricRecord]:
    """Create sample metric records for testing."""
    now = datetime.utcnow()
    return [
        MetricRecord(
            experiment_id="exp-001",
            name="latency_ms",
            value=150.5,
            unit="ms",
            timestamp=now,
            metadata={"source": "inference"},
        ),
        MetricRecord(
            experiment_id="exp-001",
            name="tokens_per_second",
            value=30.2,
            unit="tokens/s",
            timestamp=now,
            metadata={},
        ),
        MetricRecord(
            experiment_id="exp-001",
            name="perplexity",
            value=12.5,
            timestamp=now,
            metadata={},
        ),
    ]


class TestDatabaseInitialization:
    """Tests for database initialization and schema creation."""

    def test_init_db_creates_tables(self, temp_db_path: Path):
        """Test that init_db creates all required tables."""
        conn = init_db(temp_db_path)
        try:
            cursor = conn.cursor()

            # Check experiments table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'"
            )
            assert cursor.fetchone() is not None, "experiments table should exist"

            # Check conversations table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
            )
            assert cursor.fetchone() is not None, "conversations table should exist"

            # Check metrics table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
            )
            assert cursor.fetchone() is not None, "metrics table should exist"

            # Check hidden_states table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='hidden_states'"
            )
            assert cursor.fetchone() is not None, "hidden_states table should exist"
        finally:
            conn.close()

    def test_init_db_creates_indexes(self, temp_db_path: Path):
        """Test that init_db creates required indexes."""
        conn = init_db(temp_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = {row[0] for row in cursor.fetchall()}

            # Check for expected indexes
            expected_indexes = [
                "idx_experiments_created_at",
                "idx_experiments_model",
                "idx_conversations_experiment_id",
                "idx_metrics_experiment_id",
                "idx_metrics_name",
                "idx_hidden_states_experiment_id",
                "idx_hidden_states_layer",
            ]

            for idx_name in expected_indexes:
                assert idx_name in indexes, f"Index {idx_name} should exist"
        finally:
            conn.close()

    def test_init_db_enables_foreign_keys(self, temp_db_path: Path):
        """Test that foreign key constraints are enabled."""
        conn = init_db(temp_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            result = cursor.fetchone()
            assert result[0] == 1, "Foreign keys should be enabled"
        finally:
            conn.close()

    def test_init_db_enables_wal_mode(self, temp_db_path: Path):
        """Test that WAL mode is enabled for better concurrency."""
        conn = init_db(temp_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            result = cursor.fetchone()
            assert result[0].lower() == "wal", "WAL mode should be enabled"
        finally:
            conn.close()

    def test_init_db_creates_parent_directories(self):
        """Test that init_db creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "subdir" / "nested" / "test.db"
            conn = init_db(nested_path)
            try:
                assert nested_path.exists(), "Database file should be created"
                assert nested_path.parent.exists(), "Parent directories should be created"
            finally:
                conn.close()


class TestExperimentRecord:
    """Tests for ExperimentRecord dataclass operations."""

    def test_experiment_record_to_row(self, sample_experiment: ExperimentRecord):
        """Test conversion of ExperimentRecord to database row."""
        row = sample_experiment.to_row()

        assert len(row) == 6
        assert row[0] == "exp-001"
        assert isinstance(row[1], str)  # ISO format datetime
        assert row[2] == "test-model-7b"
        assert json.loads(row[3]) == {"temperature": 0.7, "max_tokens": 100}
        assert row[4] == "completed"
        assert row[5] == "Test experiment"

    def test_experiment_record_from_row(self):
        """Test creation of ExperimentRecord from database row."""
        now = datetime.utcnow()
        row = (
            "exp-002",
            now.isoformat(),
            "model-13b",
            '{"key": "value"}',
            "running",
            None,
        )

        record = ExperimentRecord.from_row(row)

        assert record.id == "exp-002"
        assert record.model == "model-13b"
        assert record.config == {"key": "value"}
        assert record.status == "running"
        assert record.notes is None

    def test_experiment_record_roundtrip(self, sample_experiment: ExperimentRecord):
        """Test that to_row/from_row maintains data integrity."""
        row = sample_experiment.to_row()
        restored = ExperimentRecord.from_row(row)

        assert restored.id == sample_experiment.id
        assert restored.model == sample_experiment.model
        assert restored.config == sample_experiment.config
        assert restored.status == sample_experiment.status
        assert restored.notes == sample_experiment.notes


class TestDatabaseManagerExperiments:
    """Tests for DatabaseManager experiment operations."""

    def test_insert_experiment(
        self, db_manager: DatabaseManager, sample_experiment: ExperimentRecord
    ):
        """Test inserting an experiment record."""
        db_manager.insert_experiment(sample_experiment)

        retrieved = db_manager.get_experiment("exp-001")
        assert retrieved is not None
        assert retrieved.id == "exp-001"
        assert retrieved.model == "test-model-7b"

    def test_get_experiment_not_found(self, db_manager: DatabaseManager):
        """Test retrieving a non-existent experiment returns None."""
        result = db_manager.get_experiment("nonexistent")
        assert result is None

    def test_list_experiments_empty(self, db_manager: DatabaseManager):
        """Test listing experiments on empty database."""
        experiments = db_manager.list_experiments()
        assert experiments == []

    def test_list_experiments_with_data(
        self, db_manager: DatabaseManager, sample_experiment: ExperimentRecord
    ):
        """Test listing experiments returns inserted records."""
        db_manager.insert_experiment(sample_experiment)

        experiments = db_manager.list_experiments()
        assert len(experiments) == 1
        assert experiments[0].id == "exp-001"

    def test_list_experiments_pagination(self, db_manager: DatabaseManager):
        """Test experiment listing with limit and offset."""
        # Insert multiple experiments
        for i in range(10):
            exp = ExperimentRecord(
                id=f"exp-{i:03d}",
                created_at=datetime.utcnow() - timedelta(hours=i),
                model="test-model",
                config={},
                status="completed",
            )
            db_manager.insert_experiment(exp)

        # Test limit
        experiments = db_manager.list_experiments(limit=5)
        assert len(experiments) == 5

        # Test offset
        experiments = db_manager.list_experiments(limit=5, offset=5)
        assert len(experiments) == 5

    def test_list_experiments_filter_by_model(self, db_manager: DatabaseManager):
        """Test filtering experiments by model."""
        for model in ["model-a", "model-b", "model-a"]:
            exp = ExperimentRecord(
                id=f"exp-{model}-{datetime.utcnow().timestamp()}",
                created_at=datetime.utcnow(),
                model=model,
                config={},
                status="completed",
            )
            db_manager.insert_experiment(exp)

        experiments = db_manager.list_experiments(model="model-a")
        assert len(experiments) == 2
        for exp in experiments:
            assert exp.model == "model-a"

    def test_list_experiments_filter_by_date_range(self, db_manager: DatabaseManager):
        """Test filtering experiments by date range."""
        now = datetime.utcnow()

        # Insert experiments at different times
        for i, delta in enumerate([timedelta(days=-5), timedelta(days=-1), timedelta(hours=-1)]):
            exp = ExperimentRecord(
                id=f"exp-{i}",
                created_at=now + delta,
                model="test-model",
                config={},
                status="completed",
            )
            db_manager.insert_experiment(exp)

        # Filter by date range
        date_from = now - timedelta(days=2)
        experiments = db_manager.list_experiments(date_from=date_from)
        assert len(experiments) == 2  # Last two experiments

    def test_delete_experiment(
        self, db_manager: DatabaseManager, sample_experiment: ExperimentRecord
    ):
        """Test deleting an experiment."""
        db_manager.insert_experiment(sample_experiment)

        deleted = db_manager.delete_experiment("exp-001")
        assert deleted is True

        # Verify deletion
        result = db_manager.get_experiment("exp-001")
        assert result is None

    def test_delete_experiment_not_found(self, db_manager: DatabaseManager):
        """Test deleting a non-existent experiment returns False."""
        deleted = db_manager.delete_experiment("nonexistent")
        assert deleted is False

    def test_delete_experiment_cascades_to_related_records(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
        sample_conversations: list[ConversationRecord],
        sample_metrics: list[MetricRecord],
    ):
        """Test that deleting an experiment cascades to related records."""
        db_manager.insert_experiment(sample_experiment)
        db_manager.insert_conversations(sample_conversations)
        db_manager.insert_metrics(sample_metrics)

        # Verify data exists
        assert len(db_manager.get_conversations("exp-001")) == 3
        assert len(db_manager.get_metrics("exp-001")) == 3

        # Delete experiment
        db_manager.delete_experiment("exp-001")

        # Verify cascade deletion
        assert len(db_manager.get_conversations("exp-001")) == 0
        assert len(db_manager.get_metrics("exp-001")) == 0

    def test_count_experiments(self, db_manager: DatabaseManager):
        """Test counting experiments."""
        assert db_manager.count_experiments() == 0

        for i in range(5):
            exp = ExperimentRecord(
                id=f"exp-{i}",
                created_at=datetime.utcnow(),
                model="test-model",
                config={},
                status="completed",
            )
            db_manager.insert_experiment(exp)

        assert db_manager.count_experiments() == 5


class TestDatabaseManagerConversations:
    """Tests for DatabaseManager conversation operations."""

    def test_insert_conversation(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
    ):
        """Test inserting a single conversation."""
        db_manager.insert_experiment(sample_experiment)

        conv = ConversationRecord(
            experiment_id="exp-001",
            sequence_num=0,
            timestamp=datetime.utcnow(),
            role="user",
            content="Hello!",
        )
        conv_id = db_manager.insert_conversation(conv)

        assert conv_id > 0
        conversations = db_manager.get_conversations("exp-001")
        assert len(conversations) == 1
        assert conversations[0].content == "Hello!"

    def test_insert_conversations_batch(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
        sample_conversations: list[ConversationRecord],
    ):
        """Test inserting multiple conversations in batch."""
        db_manager.insert_experiment(sample_experiment)
        db_manager.insert_conversations(sample_conversations)

        conversations = db_manager.get_conversations("exp-001")
        assert len(conversations) == 3

        # Verify sequence order
        for i, conv in enumerate(conversations):
            assert conv.sequence_num == i

    def test_get_conversations_preserves_order(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
    ):
        """Test that get_conversations returns messages in sequence order."""
        db_manager.insert_experiment(sample_experiment)

        # Insert in reverse order
        for i in range(4, -1, -1):
            conv = ConversationRecord(
                experiment_id="exp-001",
                sequence_num=i,
                timestamp=datetime.utcnow(),
                role="user",
                content=f"Message {i}",
            )
            db_manager.insert_conversation(conv)

        conversations = db_manager.get_conversations("exp-001")
        for i, conv in enumerate(conversations):
            assert conv.sequence_num == i
            assert conv.content == f"Message {i}"


class TestDatabaseManagerMetrics:
    """Tests for DatabaseManager metric operations."""

    def test_insert_metric(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
    ):
        """Test inserting a single metric."""
        db_manager.insert_experiment(sample_experiment)

        metric = MetricRecord(
            experiment_id="exp-001",
            name="accuracy",
            value=0.95,
            unit="ratio",
            timestamp=datetime.utcnow(),
            metadata={"validation_set": "test"},
        )
        metric_id = db_manager.insert_metric(metric)

        assert metric_id > 0
        metrics = db_manager.get_metrics("exp-001")
        assert len(metrics) == 1
        assert metrics[0].value == 0.95

    def test_insert_metrics_batch(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
        sample_metrics: list[MetricRecord],
    ):
        """Test inserting multiple metrics in batch."""
        db_manager.insert_experiment(sample_experiment)
        db_manager.insert_metrics(sample_metrics)

        metrics = db_manager.get_metrics("exp-001")
        assert len(metrics) == 3

    def test_get_metrics_filter_by_name(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
        sample_metrics: list[MetricRecord],
    ):
        """Test filtering metrics by name."""
        db_manager.insert_experiment(sample_experiment)
        db_manager.insert_metrics(sample_metrics)

        latency_metrics = db_manager.get_metrics("exp-001", metric_name="latency_ms")
        assert len(latency_metrics) == 1
        assert latency_metrics[0].name == "latency_ms"
        assert latency_metrics[0].value == 150.5

    def test_metric_metadata_roundtrip(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
    ):
        """Test that metric metadata is preserved through storage."""
        db_manager.insert_experiment(sample_experiment)

        metric = MetricRecord(
            experiment_id="exp-001",
            name="custom_metric",
            value=42.0,
            timestamp=datetime.utcnow(),
            metadata={"key1": "value1", "nested": {"a": 1, "b": 2}},
        )
        db_manager.insert_metric(metric)

        retrieved = db_manager.get_metrics("exp-001")[0]
        assert retrieved.metadata == {"key1": "value1", "nested": {"a": 1, "b": 2}}


class TestDatabaseManagerHiddenStates:
    """Tests for DatabaseManager hidden state operations."""

    def test_insert_hidden_state(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
    ):
        """Test inserting a hidden state record."""
        db_manager.insert_experiment(sample_experiment)

        hs = HiddenStateRecord(
            experiment_id="exp-001",
            layer=-1,
            file_path="/path/to/state.h5",
            shape=(1, 768),
            dtype="float32",
            timestamp=datetime.utcnow(),
        )
        hs_id = db_manager.insert_hidden_state(hs)

        assert hs_id > 0
        records = db_manager.get_hidden_states("exp-001")
        assert len(records) == 1
        assert records[0].layer == -1
        assert records[0].shape == (1, 768)

    def test_get_hidden_states_filter_by_layer(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
    ):
        """Test filtering hidden states by layer."""
        db_manager.insert_experiment(sample_experiment)

        for layer in [-1, 0, 5, 11]:
            hs = HiddenStateRecord(
                experiment_id="exp-001",
                layer=layer,
                file_path=f"/path/layer_{layer}.h5",
                shape=(1, 768),
                dtype="float32",
                timestamp=datetime.utcnow(),
            )
            db_manager.insert_hidden_state(hs)

        records = db_manager.get_hidden_states("exp-001", layer=-1)
        assert len(records) == 1
        assert records[0].layer == -1

    def test_hidden_state_shape_roundtrip(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
    ):
        """Test that hidden state shape is preserved as tuple."""
        db_manager.insert_experiment(sample_experiment)

        # Test various shapes
        shapes = [(1, 768), (10, 12, 768), (1,), (32, 512, 4096)]
        for i, shape in enumerate(shapes):
            hs = HiddenStateRecord(
                experiment_id="exp-001",
                layer=i,
                file_path=f"/path/layer_{i}.h5",
                shape=shape,
                dtype="float32",
                timestamp=datetime.utcnow(),
            )
            db_manager.insert_hidden_state(hs)

        records = db_manager.get_hidden_states("exp-001")
        for i, record in enumerate(records):
            assert record.shape == shapes[i]
            assert isinstance(record.shape, tuple)


class TestDatabaseManagerTransactions:
    """Tests for DatabaseManager transaction handling."""

    def test_transaction_commits_on_success(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
    ):
        """Test that transaction commits successfully on normal completion."""
        db_manager.insert_experiment(sample_experiment)

        with db_manager.transaction() as cursor:
            cursor.execute(
                "UPDATE experiments SET status = ? WHERE id = ?",
                ("updated", "exp-001"),
            )

        exp = db_manager.get_experiment("exp-001")
        assert exp.status == "updated"

    def test_transaction_rollback_on_error(
        self,
        db_manager: DatabaseManager,
        sample_experiment: ExperimentRecord,
    ):
        """Test that transaction rolls back on exception."""
        db_manager.insert_experiment(sample_experiment)

        try:
            with db_manager.transaction() as cursor:
                cursor.execute(
                    "UPDATE experiments SET status = ? WHERE id = ?",
                    ("updated", "exp-001"),
                )
                # Cause an error
                raise ValueError("Simulated error")
        except ValueError:
            pass

        exp = db_manager.get_experiment("exp-001")
        assert exp.status == "completed"  # Original value preserved


class TestHiddenStateStorage:
    """Tests for HiddenStateStorage HDF5 operations."""

    def test_save_and_load_array(self, hidden_storage: HiddenStateStorage):
        """Test saving and loading a numpy array."""
        array = np.random.randn(10, 768).astype(np.float32)

        file_path = hidden_storage.save("exp-001", array, layer=-1)

        assert Path(file_path).exists()

        loaded = hidden_storage.load(file_path)
        np.testing.assert_array_almost_equal(array, loaded)

    def test_save_and_load_preserves_dtype(self, hidden_storage: HiddenStateStorage):
        """Test that data type is preserved through save/load."""
        for dtype in [np.float32, np.float64, np.float16]:
            array = np.random.randn(5, 128).astype(dtype)
            file_path = hidden_storage.save(f"exp-{dtype}", array, layer=0)
            loaded = hidden_storage.load(file_path)

            assert loaded.dtype == dtype

    def test_save_and_load_preserves_shape(self, hidden_storage: HiddenStateStorage):
        """Test that shape is preserved through save/load."""
        shapes = [(1, 768), (10, 12, 768), (32, 512), (1,)]

        for i, shape in enumerate(shapes):
            array = np.random.randn(*shape).astype(np.float32)
            file_path = hidden_storage.save(f"exp-{i}", array, layer=0)
            loaded = hidden_storage.load(file_path)

            assert loaded.shape == shape

    def test_save_with_metadata(self, hidden_storage: HiddenStateStorage):
        """Test saving array with custom metadata."""
        array = np.random.randn(5, 128).astype(np.float32)
        metadata = {"model": "test-model", "inference_time": 100.5}

        file_path = hidden_storage.save(
            "exp-001", array, layer=-1, metadata=metadata
        )

        loaded_array, loaded_metadata = hidden_storage.load_with_metadata(file_path)

        np.testing.assert_array_almost_equal(array, loaded_array)
        assert loaded_metadata["model"] == "test-model"
        assert loaded_metadata["inference_time"] == 100.5

    def test_load_nonexistent_file_raises_error(self, hidden_storage: HiddenStateStorage):
        """Test that loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            hidden_storage.load("/nonexistent/path.h5")

    def test_delete_file(self, hidden_storage: HiddenStateStorage):
        """Test deleting an HDF5 file."""
        array = np.random.randn(5, 128).astype(np.float32)
        file_path = hidden_storage.save("exp-001", array, layer=0)

        assert Path(file_path).exists()

        deleted = hidden_storage.delete(file_path)
        assert deleted is True
        assert not Path(file_path).exists()

    def test_delete_nonexistent_file_returns_false(
        self, hidden_storage: HiddenStateStorage
    ):
        """Test that deleting non-existent file returns False."""
        deleted = hidden_storage.delete("/nonexistent/path.h5")
        assert deleted is False

    def test_list_files(self, hidden_storage: HiddenStateStorage):
        """Test listing HDF5 files in storage."""
        # Create multiple files
        for exp_id in ["exp-001", "exp-002", "exp-001"]:
            array = np.random.randn(5, 128).astype(np.float32)
            hidden_storage.save(exp_id, array, layer=0)

        all_files = hidden_storage.list_files()
        assert len(all_files) == 3

        exp_001_files = hidden_storage.list_files("exp-001")
        assert len(exp_001_files) == 2

    def test_get_file_info(self, hidden_storage: HiddenStateStorage):
        """Test getting file info without loading array."""
        array = np.random.randn(32, 768).astype(np.float32)
        file_path = hidden_storage.save(
            "exp-001", array, layer=-1, metadata={"model": "test"}
        )

        info = hidden_storage.get_file_info(file_path)

        assert info["shape"] == (32, 768)
        assert info["dtype"] == "float32"
        assert "compression" in info
        assert info["experiment_id"] == "exp-001"
        assert info["layer"] == -1

    def test_compression_reduces_file_size(self, hidden_storage: HiddenStateStorage):
        """Test that compression actually reduces file size."""
        # Create a compressible array (repeated patterns)
        array = np.tile(np.arange(768, dtype=np.float32), (100, 1))

        file_path = hidden_storage.save("exp-compress", array, layer=0)

        # Uncompressed size would be 100 * 768 * 4 bytes = ~307 KB
        actual_size = Path(file_path).stat().st_size
        uncompressed_size = array.nbytes

        # Compressed should be significantly smaller
        assert actual_size < uncompressed_size * 0.5


class TestExperimentPersistence:
    """Tests for the high-level ExperimentPersistence manager."""

    def test_create_experiment(self, persistence: ExperimentPersistence):
        """Test creating a new experiment."""
        experiment = persistence.create_experiment(
            model="test-model",
            config={"temperature": 0.7},
            experiment_id="exp-test",
            notes="Test notes",
        )

        assert experiment.id == "exp-test"
        assert experiment.model == "test-model"
        assert experiment.config == {"temperature": 0.7}
        assert experiment.notes == "Test notes"

    def test_create_experiment_auto_generates_id(self, persistence: ExperimentPersistence):
        """Test that experiment ID is auto-generated if not provided."""
        experiment = persistence.create_experiment(model="test-model")

        assert experiment.id is not None
        assert len(experiment.id) > 0

    def test_complete_experiment(self, persistence: ExperimentPersistence):
        """Test marking an experiment as completed."""
        experiment = persistence.create_experiment(model="test-model", status="running")

        assert experiment.status == "running"

        success = persistence.complete_experiment(experiment.id, status="completed")
        assert success is True

        retrieved = persistence.get_experiment(experiment.id)
        assert retrieved.status == "completed"

    def test_persist_conversation(self, persistence: ExperimentPersistence):
        """Test persisting a single conversation message."""
        experiment = persistence.create_experiment(model="test-model")

        conv = persistence.persist_conversation(
            experiment_id=experiment.id,
            role="user",
            content="Hello!",
        )

        assert conv.role == "user"
        assert conv.content == "Hello!"
        assert conv.sequence_num == 0

    def test_persist_conversations(self, persistence: ExperimentPersistence):
        """Test persisting multiple conversation messages."""
        experiment = persistence.create_experiment(model="test-model")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello!"},
        ]

        convs = persistence.persist_conversations(experiment.id, messages)

        assert len(convs) == 3
        for i, conv in enumerate(convs):
            assert conv.sequence_num == i

    def test_persist_hidden_states(self, persistence: ExperimentPersistence):
        """Test persisting hidden states from multiple layers."""
        experiment = persistence.create_experiment(model="test-model")

        hidden_states = {
            -1: np.random.randn(1, 768).astype(np.float32),
            0: np.random.randn(1, 768).astype(np.float32),
            11: np.random.randn(1, 768).astype(np.float32),
        }

        records = persistence.persist_hidden_states(experiment.id, hidden_states)

        assert len(records) == 3
        layers = {r.layer for r in records}
        assert layers == {-1, 0, 11}

    def test_persist_hidden_state_single(self, persistence: ExperimentPersistence):
        """Test persisting a single hidden state array."""
        experiment = persistence.create_experiment(model="test-model")
        array = np.random.randn(1, 768).astype(np.float32)

        record = persistence.persist_hidden_state(
            experiment_id=experiment.id,
            array=array,
            layer=-1,
        )

        assert record.layer == -1
        assert record.shape == (1, 768)

    def test_load_hidden_state(self, persistence: ExperimentPersistence):
        """Test loading a persisted hidden state array."""
        experiment = persistence.create_experiment(model="test-model")
        original = np.random.randn(10, 768).astype(np.float32)

        record = persistence.persist_hidden_state(
            experiment_id=experiment.id,
            array=original,
            layer=-1,
        )

        loaded = persistence.load_hidden_state(record)
        np.testing.assert_array_almost_equal(original, loaded)

    def test_persist_metric(self, persistence: ExperimentPersistence):
        """Test persisting a single metric."""
        experiment = persistence.create_experiment(model="test-model")

        metric = persistence.persist_metric(
            experiment_id=experiment.id,
            name="latency_ms",
            value=150.5,
            unit="ms",
        )

        assert metric.name == "latency_ms"
        assert metric.value == 150.5
        assert metric.unit == "ms"

    def test_persist_metrics_batch(self, persistence: ExperimentPersistence):
        """Test persisting multiple metrics at once."""
        experiment = persistence.create_experiment(model="test-model")

        metrics = {
            "latency_ms": 150.5,
            "tokens_per_second": 30.2,
            "perplexity": 12.5,
        }

        records = persistence.persist_metrics(experiment.id, metrics)

        assert len(records) == 3
        names = {r.name for r in records}
        assert names == {"latency_ms", "tokens_per_second", "perplexity"}

    def test_persist_experiment_complete(self, persistence: ExperimentPersistence):
        """Test persisting a complete experiment with all data."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        hidden_states = {-1: np.random.randn(1, 768).astype(np.float32)}
        metrics = {"latency_ms": 100.0, "tokens_per_second": 50.0}

        experiment = persistence.persist_experiment(
            model="test-model",
            messages=messages,
            hidden_states=hidden_states,
            metrics=metrics,
            notes="Complete experiment",
        )

        # Verify all data was persisted
        convs = persistence.get_conversations(experiment.id)
        assert len(convs) == 2

        hs_records = persistence.get_hidden_states(experiment.id)
        assert len(hs_records) == 1

        metric_records = persistence.get_metrics(experiment.id)
        assert len(metric_records) == 2

    def test_get_experiment_data(self, persistence: ExperimentPersistence):
        """Test retrieving complete experiment data."""
        messages = [{"role": "user", "content": "Test"}]
        hidden_states = {-1: np.random.randn(1, 768).astype(np.float32)}
        metrics = {"latency_ms": 100.0}

        experiment = persistence.persist_experiment(
            model="test-model",
            messages=messages,
            hidden_states=hidden_states,
            metrics=metrics,
        )

        data = persistence.get_experiment_data(experiment.id, include_hidden_states=True)

        assert data is not None
        assert data["experiment"].id == experiment.id
        assert len(data["conversations"]) == 1
        assert len(data["hidden_state_records"]) == 1
        assert len(data["metrics"]) == 1
        assert -1 in data["hidden_states"]

    def test_delete_experiment_removes_all_data(self, persistence: ExperimentPersistence):
        """Test that deleting experiment removes all associated data."""
        messages = [{"role": "user", "content": "Test"}]
        hidden_states = {-1: np.random.randn(1, 768).astype(np.float32)}

        experiment = persistence.persist_experiment(
            model="test-model",
            messages=messages,
            hidden_states=hidden_states,
        )

        # Get file path before deletion
        hs_records = persistence.get_hidden_states(experiment.id)
        file_path = hs_records[0].file_path

        # Delete experiment
        deleted = persistence.delete_experiment(experiment.id)
        assert deleted is True

        # Verify all data is removed
        assert persistence.get_experiment(experiment.id) is None
        assert len(persistence.get_conversations(experiment.id)) == 0
        assert len(persistence.get_hidden_states(experiment.id)) == 0
        assert not Path(file_path).exists()

    def test_session_context_manager(
        self, temp_db_path: Path, temp_storage_dir: Path
    ):
        """Test session context manager for cleanup."""
        with ExperimentPersistence(
            db_path=temp_db_path,
            storage_dir=temp_storage_dir,
        ).session() as ep:
            experiment = ep.create_experiment(model="test-model")
            assert experiment is not None


class TestExperimentQuery:
    """Tests for the ExperimentQuery read-only interface."""

    @pytest.fixture
    def populated_query(
        self, persistence: ExperimentPersistence
    ) -> Generator[ExperimentQuery, None, None]:
        """Create query interface with pre-populated test data."""
        # Create test experiments
        for i in range(5):
            messages = [
                {"role": "user", "content": f"Message {i}"},
                {"role": "assistant", "content": f"Response {i}"},
            ]
            hidden_states = {-1: np.random.randn(1, 768).astype(np.float32)}
            metrics = {"latency_ms": 100.0 + i * 10, "tokens": float(10 + i)}

            persistence.persist_experiment(
                model=f"model-{i % 2}",  # Alternate between model-0 and model-1
                experiment_id=f"exp-{i:03d}",
                messages=messages,
                hidden_states=hidden_states,
                metrics=metrics,
                config={"temperature": 0.5 + i * 0.1},
            )

        query = ExperimentQuery(persistence._db_path)
        yield query
        query.close()

    def test_list_experiments(self, populated_query: ExperimentQuery):
        """Test listing experiments."""
        experiments = populated_query.list_experiments()
        assert len(experiments) == 5

    def test_list_experiments_with_model_filter(self, populated_query: ExperimentQuery):
        """Test listing experiments filtered by model."""
        experiments = populated_query.list_experiments(model="model-0")
        assert len(experiments) == 3  # exp-000, exp-002, exp-004

    def test_get_experiment(self, populated_query: ExperimentQuery):
        """Test retrieving a specific experiment."""
        experiment = populated_query.get_experiment("exp-002")
        assert experiment is not None
        assert experiment.id == "exp-002"

    def test_get_experiment_summary(self, populated_query: ExperimentQuery):
        """Test getting experiment summary statistics."""
        summary = populated_query.get_experiment_summary("exp-001")

        assert summary is not None
        assert isinstance(summary, ExperimentSummary)
        assert summary.experiment_id == "exp-001"
        assert summary.conversation_count == 2
        assert summary.metric_count == 2
        assert summary.hidden_state_count == 1

    def test_count_experiments(self, populated_query: ExperimentQuery):
        """Test counting experiments."""
        total = populated_query.count_experiments()
        assert total == 5

        model_0_count = populated_query.count_experiments(model="model-0")
        assert model_0_count == 3

    def test_get_conversations_with_role_filter(self, populated_query: ExperimentQuery):
        """Test getting conversations filtered by role."""
        all_convs = populated_query.get_conversations("exp-001")
        assert len(all_convs) == 2

        user_convs = populated_query.get_conversations("exp-001", role="user")
        assert len(user_convs) == 1
        assert user_convs[0].role == "user"

    def test_search_conversations(self, populated_query: ExperimentQuery):
        """Test searching conversation content."""
        results = populated_query.search_conversations("Message")
        assert len(results) >= 5  # All user messages contain "Message"

    def test_get_metric_names(self, populated_query: ExperimentQuery):
        """Test getting distinct metric names."""
        names = populated_query.get_metric_names("exp-001")
        assert set(names) == {"latency_ms", "tokens"}

    def test_get_metric_range(self, populated_query: ExperimentQuery):
        """Test getting metric statistics across experiments."""
        stats = populated_query.get_metric_range("latency_ms")

        assert "min" in stats
        assert "max" in stats
        assert "avg" in stats
        assert stats["min"] == 100.0  # First experiment
        assert stats["max"] == 140.0  # Last experiment

    def test_get_hidden_state_layers(self, populated_query: ExperimentQuery):
        """Test getting list of hidden state layers."""
        layers = populated_query.get_hidden_state_layers("exp-001")
        assert layers == [-1]

    def test_get_experiments_with_metric(self, populated_query: ExperimentQuery):
        """Test finding experiments with specific metric criteria."""
        experiments = populated_query.get_experiments_with_metric(
            "latency_ms", min_value=120.0
        )

        # exp-002, exp-003, exp-004 have latency >= 120
        assert len(experiments) == 3

    def test_get_model_statistics(self, populated_query: ExperimentQuery):
        """Test getting experiment statistics by model."""
        stats = populated_query.get_model_statistics()

        assert len(stats) == 2
        models = {s["model"] for s in stats}
        assert models == {"model-0", "model-1"}


class TestConcurrentAccess:
    """Tests for concurrent database access."""

    def test_concurrent_writes(self, temp_db_path: Path, temp_storage_dir: Path):
        """Test that concurrent writes don't cause corruption."""
        errors: list[Exception] = []

        def create_experiments(start_id: int, count: int):
            try:
                ep = ExperimentPersistence(
                    db_path=temp_db_path,
                    storage_dir=temp_storage_dir,
                )
                for i in range(count):
                    ep.persist_experiment(
                        model="test-model",
                        experiment_id=f"exp-{start_id + i}",
                        messages=[{"role": "user", "content": f"Message {i}"}],
                    )
                ep.close()
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for t in range(4):
            thread = threading.Thread(target=create_experiments, args=(t * 25, 25))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check for errors
        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"

        # Verify all experiments were created
        ep = ExperimentPersistence(
            db_path=temp_db_path,
            storage_dir=temp_storage_dir,
        )
        experiments = ep.list_experiments(limit=200)
        ep.close()

        assert len(experiments) == 100

    def test_concurrent_read_write(self, temp_db_path: Path, temp_storage_dir: Path):
        """Test concurrent read and write operations."""
        # Pre-populate with some data
        ep = ExperimentPersistence(
            db_path=temp_db_path,
            storage_dir=temp_storage_dir,
        )
        for i in range(10):
            ep.persist_experiment(
                model="test-model",
                experiment_id=f"exp-{i}",
                messages=[{"role": "user", "content": f"Message {i}"}],
            )
        ep.close()

        errors: list[Exception] = []
        read_counts: list[int] = []

        def writer():
            try:
                ep = ExperimentPersistence(
                    db_path=temp_db_path,
                    storage_dir=temp_storage_dir,
                )
                for i in range(20):
                    ep.persist_experiment(
                        model="test-model",
                        experiment_id=f"exp-new-{i}",
                        messages=[{"role": "user", "content": f"New message {i}"}],
                    )
                ep.close()
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                query = ExperimentQuery(temp_db_path)
                for _ in range(50):
                    experiments = query.list_experiments(limit=100)
                    read_counts.append(len(experiments))
                query.close()
            except Exception as e:
                errors.append(e)

        # Run writer and reader concurrently
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert all(c >= 10 for c in read_counts), "Reader should see at least initial data"
