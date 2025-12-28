"""Database schema and models for experiment persistence.

This module provides SQLite-based storage for experiment metadata, conversations,
computed metrics, and hidden state file references.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


# SQL statements for schema creation
_SCHEMA_SQL = """
-- Experiments table: stores experiment metadata
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    config_json TEXT,
    status TEXT DEFAULT 'completed',
    notes TEXT
);

-- Index for timestamp-based queries
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);

-- Index for model filtering
CREATE INDEX IF NOT EXISTS idx_experiments_model ON experiments(model);

-- Conversations table: stores message history
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    sequence_num INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

-- Index for retrieving conversations by experiment
CREATE INDEX IF NOT EXISTS idx_conversations_experiment_id ON conversations(experiment_id);

-- Metrics table: stores computed numerical metrics
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    timestamp TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

-- Index for retrieving metrics by experiment
CREATE INDEX IF NOT EXISTS idx_metrics_experiment_id ON metrics(experiment_id);

-- Index for querying specific metric types
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);

-- Hidden states table: stores references to HDF5 files
CREATE TABLE IF NOT EXISTS hidden_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    layer INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    shape_json TEXT NOT NULL,
    dtype TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

-- Index for retrieving hidden states by experiment
CREATE INDEX IF NOT EXISTS idx_hidden_states_experiment_id ON hidden_states(experiment_id);

-- Index for layer-based queries
CREATE INDEX IF NOT EXISTS idx_hidden_states_layer ON hidden_states(layer);
"""


@dataclass
class ExperimentRecord:
    """Record representing an experiment in the database."""

    id: str
    created_at: datetime
    model: str
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    notes: str | None = None

    def to_row(self) -> tuple[str, str, str, str, str, str | None]:
        """Convert to a tuple suitable for database insertion."""
        return (
            self.id,
            self.created_at.isoformat(),
            self.model,
            json.dumps(self.config),
            self.status,
            self.notes,
        )

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> ExperimentRecord:
        """Create an ExperimentRecord from a database row."""
        return cls(
            id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            model=row[2],
            config=json.loads(row[3]) if row[3] else {},
            status=row[4] or "completed",
            notes=row[5],
        )


@dataclass
class ConversationRecord:
    """Record representing a conversation message in the database."""

    experiment_id: str
    sequence_num: int
    timestamp: datetime
    role: str
    content: str
    id: int | None = None

    def to_row(self) -> tuple[str, int, str, str, str]:
        """Convert to a tuple suitable for database insertion."""
        return (
            self.experiment_id,
            self.sequence_num,
            self.timestamp.isoformat(),
            self.role,
            self.content,
        )

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> ConversationRecord:
        """Create a ConversationRecord from a database row."""
        return cls(
            id=row[0],
            experiment_id=row[1],
            sequence_num=row[2],
            timestamp=datetime.fromisoformat(row[3]),
            role=row[4],
            content=row[5],
        )


@dataclass
class MetricRecord:
    """Record representing a computed metric in the database."""

    experiment_id: str
    name: str
    value: float
    timestamp: datetime
    unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None

    def to_row(self) -> tuple[str, str, float, str | None, str, str | None]:
        """Convert to a tuple suitable for database insertion."""
        return (
            self.experiment_id,
            self.name,
            self.value,
            self.unit,
            self.timestamp.isoformat(),
            json.dumps(self.metadata) if self.metadata else None,
        )

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> MetricRecord:
        """Create a MetricRecord from a database row."""
        return cls(
            id=row[0],
            experiment_id=row[1],
            name=row[2],
            value=row[3],
            unit=row[4],
            timestamp=datetime.fromisoformat(row[5]),
            metadata=json.loads(row[6]) if row[6] else {},
        )


@dataclass
class HiddenStateRecord:
    """Record representing a hidden state file reference in the database."""

    experiment_id: str
    layer: int
    file_path: str
    shape: tuple[int, ...]
    dtype: str
    timestamp: datetime
    id: int | None = None

    def to_row(self) -> tuple[str, int, str, str, str, str]:
        """Convert to a tuple suitable for database insertion."""
        return (
            self.experiment_id,
            self.layer,
            self.file_path,
            json.dumps(self.shape),
            self.dtype,
            self.timestamp.isoformat(),
        )

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> HiddenStateRecord:
        """Create a HiddenStateRecord from a database row."""
        return cls(
            id=row[0],
            experiment_id=row[1],
            layer=row[2],
            file_path=row[3],
            shape=tuple(json.loads(row[4])),
            dtype=row[5],
            timestamp=datetime.fromisoformat(row[6]),
        )


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """
    Initialize the database with the required schema.

    Creates all necessary tables and indexes if they don't exist.
    Enables foreign key support and WAL mode for better concurrency.

    Parameters:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection: An open connection to the initialized database.
    """
    # Ensure parent directory exists
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Connect with foreign key support
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    # Create schema
    conn.executescript(_SCHEMA_SQL)
    conn.commit()

    return conn


class DatabaseManager:
    """Manager for database operations on experiment data.

    Provides methods for inserting and querying experiments, conversations,
    metrics, and hidden state references with proper transaction handling.
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize the database manager.

        Parameters:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path).expanduser()
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Establish a database connection and initialize schema."""
        self._conn = init_db(self._db_path)

    def close(self) -> None:
        """Close the database connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Get the current connection, connecting if necessary."""
        if self._conn is None:
            self.connect()
        return self._conn  # type: ignore[return-value]

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """
        Context manager for database transactions.

        Yields a cursor and commits on success, rolls back on exception.

        Yields:
            sqlite3.Cursor: A cursor for executing SQL statements.
        """
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    # Experiment operations

    def insert_experiment(self, experiment: ExperimentRecord) -> None:
        """
        Insert an experiment record into the database.

        Parameters:
            experiment: The experiment record to insert.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO experiments (id, created_at, model, config_json, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                experiment.to_row(),
            )

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """
        Retrieve an experiment by its ID.

        Parameters:
            experiment_id: The unique identifier of the experiment.

        Returns:
            The experiment record if found, None otherwise.
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "SELECT id, created_at, model, config_json, status, notes FROM experiments WHERE id = ?",
                (experiment_id,),
            )
            row = cursor.fetchone()
            return ExperimentRecord.from_row(row) if row else None
        finally:
            cursor.close()

    def list_experiments(
        self,
        limit: int = 100,
        offset: int = 0,
        model: str | None = None,
        status: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[ExperimentRecord]:
        """
        List experiments with optional filtering.

        Parameters:
            limit: Maximum number of experiments to return.
            offset: Number of experiments to skip (for pagination).
            model: Filter by model name.
            status: Filter by experiment status.
            date_from: Filter experiments created after this datetime.
            date_to: Filter experiments created before this datetime.

        Returns:
            List of experiment records matching the filters.
        """
        query = "SELECT id, created_at, model, config_json, status, notes FROM experiments WHERE 1=1"
        params: list[Any] = []

        if model is not None:
            query += " AND model = ?"
            params.append(model)

        if status is not None:
            query += " AND status = ?"
            params.append(status)

        if date_from is not None:
            query += " AND created_at >= ?"
            params.append(date_from.isoformat())

        if date_to is not None:
            query += " AND created_at <= ?"
            params.append(date_to.isoformat())

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params)
            return [ExperimentRecord.from_row(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment and all related records.

        Parameters:
            experiment_id: The unique identifier of the experiment to delete.

        Returns:
            True if an experiment was deleted, False if not found.
        """
        with self.transaction() as cursor:
            cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            return cursor.rowcount > 0

    # Conversation operations

    def insert_conversation(self, conversation: ConversationRecord) -> int:
        """
        Insert a conversation record into the database.

        Parameters:
            conversation: The conversation record to insert.

        Returns:
            The auto-generated ID of the inserted record.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO conversations (experiment_id, sequence_num, timestamp, role, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                conversation.to_row(),
            )
            return cursor.lastrowid or 0

    def insert_conversations(self, conversations: list[ConversationRecord]) -> None:
        """
        Insert multiple conversation records in a single transaction.

        Parameters:
            conversations: List of conversation records to insert.
        """
        with self.transaction() as cursor:
            cursor.executemany(
                """
                INSERT INTO conversations (experiment_id, sequence_num, timestamp, role, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                [c.to_row() for c in conversations],
            )

    def get_conversations(self, experiment_id: str) -> list[ConversationRecord]:
        """
        Retrieve all conversations for an experiment, ordered by sequence.

        Parameters:
            experiment_id: The unique identifier of the experiment.

        Returns:
            List of conversation records in sequence order.
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT id, experiment_id, sequence_num, timestamp, role, content
                FROM conversations
                WHERE experiment_id = ?
                ORDER BY sequence_num
                """,
                (experiment_id,),
            )
            return [ConversationRecord.from_row(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    # Metric operations

    def insert_metric(self, metric: MetricRecord) -> int:
        """
        Insert a metric record into the database.

        Parameters:
            metric: The metric record to insert.

        Returns:
            The auto-generated ID of the inserted record.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO metrics (experiment_id, name, value, unit, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                metric.to_row(),
            )
            return cursor.lastrowid or 0

    def insert_metrics(self, metrics: list[MetricRecord]) -> None:
        """
        Insert multiple metric records in a single transaction.

        Parameters:
            metrics: List of metric records to insert.
        """
        with self.transaction() as cursor:
            cursor.executemany(
                """
                INSERT INTO metrics (experiment_id, name, value, unit, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [m.to_row() for m in metrics],
            )

    def get_metrics(
        self,
        experiment_id: str,
        metric_name: str | None = None,
    ) -> list[MetricRecord]:
        """
        Retrieve metrics for an experiment, optionally filtered by name.

        Parameters:
            experiment_id: The unique identifier of the experiment.
            metric_name: Optional filter for specific metric type.

        Returns:
            List of metric records matching the criteria.
        """
        cursor = self.connection.cursor()
        try:
            if metric_name is not None:
                cursor.execute(
                    """
                    SELECT id, experiment_id, name, value, unit, timestamp, metadata_json
                    FROM metrics
                    WHERE experiment_id = ? AND name = ?
                    ORDER BY timestamp
                    """,
                    (experiment_id, metric_name),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, experiment_id, name, value, unit, timestamp, metadata_json
                    FROM metrics
                    WHERE experiment_id = ?
                    ORDER BY timestamp
                    """,
                    (experiment_id,),
                )
            return [MetricRecord.from_row(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    # Hidden state operations

    def insert_hidden_state(self, hidden_state: HiddenStateRecord) -> int:
        """
        Insert a hidden state record into the database.

        Parameters:
            hidden_state: The hidden state record to insert.

        Returns:
            The auto-generated ID of the inserted record.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO hidden_states (experiment_id, layer, file_path, shape_json, dtype, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                hidden_state.to_row(),
            )
            return cursor.lastrowid or 0

    def insert_hidden_states(self, hidden_states: list[HiddenStateRecord]) -> None:
        """
        Insert multiple hidden state records in a single transaction.

        Parameters:
            hidden_states: List of hidden state records to insert.
        """
        with self.transaction() as cursor:
            cursor.executemany(
                """
                INSERT INTO hidden_states (experiment_id, layer, file_path, shape_json, dtype, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [hs.to_row() for hs in hidden_states],
            )

    def get_hidden_states(
        self,
        experiment_id: str,
        layer: int | None = None,
    ) -> list[HiddenStateRecord]:
        """
        Retrieve hidden state records for an experiment.

        Parameters:
            experiment_id: The unique identifier of the experiment.
            layer: Optional filter for specific layer index.

        Returns:
            List of hidden state records matching the criteria.
        """
        cursor = self.connection.cursor()
        try:
            if layer is not None:
                cursor.execute(
                    """
                    SELECT id, experiment_id, layer, file_path, shape_json, dtype, timestamp
                    FROM hidden_states
                    WHERE experiment_id = ? AND layer = ?
                    ORDER BY layer
                    """,
                    (experiment_id, layer),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, experiment_id, layer, file_path, shape_json, dtype, timestamp
                    FROM hidden_states
                    WHERE experiment_id = ?
                    ORDER BY layer
                    """,
                    (experiment_id,),
                )
            return [HiddenStateRecord.from_row(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    # Utility operations

    def count_experiments(self) -> int:
        """
        Count total number of experiments in the database.

        Returns:
            The total count of experiment records.
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM experiments")
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            cursor.close()

    def vacuum(self) -> None:
        """Reclaim unused database space."""
        self.connection.execute("VACUUM")
