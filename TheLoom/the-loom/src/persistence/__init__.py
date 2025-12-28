"""Experiment result persistence layer.

This module provides persistence for experiment results including database
operations, hidden state storage, and data export capabilities.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .database import (
    ConversationRecord,
    DatabaseManager,
    ExperimentRecord,
    HiddenStateRecord,
    MetricRecord,
    init_db,
)
from .export import ExperimentExporter, ExportError
from .query import ExperimentQuery, ExperimentSummary
from .storage import HiddenStateStorage

logger = logging.getLogger(__name__)


class ExperimentPersistence:
    """High-level manager coordinating database and hidden state storage.

    Provides unified API for persisting complete experiments including
    metadata, conversations, hidden states, and computed metrics.
    """

    def __init__(
        self,
        db_path: str | Path,
        storage_dir: str | Path,
        compression: str = HiddenStateStorage.DEFAULT_COMPRESSION,
        compression_level: int = HiddenStateStorage.DEFAULT_COMPRESSION_LEVEL,
        chunk_size: int = HiddenStateStorage.DEFAULT_CHUNK_SIZE,
    ) -> None:
        """
        Initialize the experiment persistence manager.

        Parameters:
            db_path: Path to the SQLite database file.
            storage_dir: Directory for HDF5 hidden state files.
            compression: Compression algorithm for HDF5 (default: gzip).
            compression_level: Compression level 0-9 (default: 4).
            chunk_size: Chunk size for HDF5 datasets (default: 1024).
        """
        self._db_path = Path(db_path).expanduser()
        self._storage_dir = Path(storage_dir).expanduser()

        # Initialize database manager
        self._db = DatabaseManager(self._db_path)

        # Initialize hidden state storage
        self._storage = HiddenStateStorage(
            storage_dir=self._storage_dir,
            compression=compression,
            compression_level=compression_level,
            chunk_size=chunk_size,
        )

        logger.debug(
            f"ExperimentPersistence initialized: db={self._db_path}, storage={self._storage_dir}"
        )

    @property
    def db(self) -> DatabaseManager:
        """Get the underlying database manager."""
        return self._db

    @property
    def storage(self) -> HiddenStateStorage:
        """Get the underlying hidden state storage manager."""
        return self._storage

    def close(self) -> None:
        """Close the database connection and clean up resources."""
        self._db.close()
        logger.debug("ExperimentPersistence closed")

    @contextmanager
    def session(self) -> Iterator[ExperimentPersistence]:
        """
        Context manager for a persistence session.

        Ensures proper cleanup of database connections on exit.

        Yields:
            This ExperimentPersistence instance.
        """
        try:
            yield self
        finally:
            self.close()

    # ========================================================================
    # Experiment Operations
    # ========================================================================

    def create_experiment(
        self,
        model: str,
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        status: str = "running",
        notes: str | None = None,
    ) -> ExperimentRecord:
        """
        Create and persist a new experiment.

        Parameters:
            model: The model identifier used in the experiment.
            config: Optional configuration dictionary.
            experiment_id: Optional custom ID (auto-generated if not provided).
            status: Initial status (default: "running").
            notes: Optional notes or description.

        Returns:
            The created ExperimentRecord.
        """
        exp_id = experiment_id or str(uuid.uuid4())
        record = ExperimentRecord(
            id=exp_id,
            created_at=datetime.utcnow(),
            model=model,
            config=config or {},
            status=status,
            notes=notes,
        )

        self._db.insert_experiment(record)
        logger.info(f"Created experiment: {exp_id}")

        return record

    def complete_experiment(
        self,
        experiment_id: str,
        status: str = "completed",
    ) -> bool:
        """
        Mark an experiment as completed.

        Parameters:
            experiment_id: The experiment ID to update.
            status: Final status (default: "completed").

        Returns:
            True if the experiment was found and updated.
        """
        # Note: SQLite doesn't have built-in UPDATE through our simple API,
        # so we implement a pattern that works with our current DatabaseManager
        experiment = self._db.get_experiment(experiment_id)
        if experiment is None:
            logger.warning(f"Experiment not found for completion: {experiment_id}")
            return False

        # Use raw SQL for update through transaction
        with self._db.transaction() as cursor:
            cursor.execute(
                "UPDATE experiments SET status = ? WHERE id = ?",
                (status, experiment_id),
            )

        logger.info(f"Completed experiment: {experiment_id} with status: {status}")
        return True

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """
        Retrieve an experiment by ID.

        Parameters:
            experiment_id: The experiment identifier.

        Returns:
            The ExperimentRecord if found, None otherwise.
        """
        return self._db.get_experiment(experiment_id)

    def list_experiments(
        self,
        limit: int = 100,
        offset: int = 0,
        model: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[ExperimentRecord]:
        """
        List experiments with optional filtering.

        Parameters:
            limit: Maximum number of experiments to return.
            offset: Number of experiments to skip.
            model: Optional model filter.
            date_from: Optional start date filter.
            date_to: Optional end date filter.

        Returns:
            List of matching ExperimentRecords.
        """
        return self._db.list_experiments(
            limit=limit,
            offset=offset,
            model=model,
            date_from=date_from,
            date_to=date_to,
        )

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment and all associated data.

        This removes the experiment record, all conversations, metrics,
        hidden state records, and the actual HDF5 files.

        Parameters:
            experiment_id: The experiment ID to delete.

        Returns:
            True if the experiment was found and deleted.
        """
        # First, get hidden state file paths to delete
        hidden_states = self._db.get_hidden_states(experiment_id)
        for hs in hidden_states:
            try:
                self._storage.delete(hs.file_path)
                logger.debug(f"Deleted hidden state file: {hs.file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete hidden state file {hs.file_path}: {e}")

        # Delete the experiment (cascades to conversations, metrics, hidden_states)
        deleted = self._db.delete_experiment(experiment_id)

        if deleted:
            logger.info(f"Deleted experiment: {experiment_id}")
        else:
            logger.warning(f"Experiment not found for deletion: {experiment_id}")

        return deleted

    # ========================================================================
    # Conversation Operations
    # ========================================================================

    def persist_conversation(
        self,
        experiment_id: str,
        role: str,
        content: str,
        sequence_num: int | None = None,
        timestamp: datetime | None = None,
    ) -> ConversationRecord:
        """
        Persist a single conversation message.

        Parameters:
            experiment_id: The experiment this message belongs to.
            role: Message role (e.g., "user", "assistant", "system").
            content: Message content text.
            sequence_num: Optional sequence number (auto-incremented if None).
            timestamp: Optional timestamp (current time if None).

        Returns:
            The created ConversationRecord.
        """
        # Get next sequence number if not provided
        if sequence_num is None:
            existing = self._db.get_conversations(experiment_id)
            sequence_num = len(existing)

        record = ConversationRecord(
            experiment_id=experiment_id,
            sequence_num=sequence_num,
            timestamp=timestamp or datetime.utcnow(),
            role=role,
            content=content,
        )

        record.id = self._db.insert_conversation(record)
        logger.debug(f"Persisted conversation {experiment_id}:{sequence_num} ({role})")

        return record

    def persist_conversations(
        self,
        experiment_id: str,
        messages: list[dict[str, str]],
        base_timestamp: datetime | None = None,
    ) -> list[ConversationRecord]:
        """
        Persist multiple conversation messages.

        Parameters:
            experiment_id: The experiment these messages belong to.
            messages: List of {"role": str, "content": str} dicts.
            base_timestamp: Optional base timestamp (current time if None).

        Returns:
            List of created ConversationRecords.
        """
        timestamp = base_timestamp or datetime.utcnow()

        records = [
            ConversationRecord(
                experiment_id=experiment_id,
                sequence_num=i,
                timestamp=timestamp,
                role=msg.get("role", "unknown"),
                content=msg.get("content", ""),
            )
            for i, msg in enumerate(messages)
        ]

        self._db.insert_conversations(records)
        logger.debug(f"Persisted {len(records)} conversation messages for {experiment_id}")

        return records

    def get_conversations(self, experiment_id: str) -> list[ConversationRecord]:
        """
        Retrieve all conversations for an experiment.

        Parameters:
            experiment_id: The experiment identifier.

        Returns:
            List of ConversationRecords in sequence order.
        """
        return self._db.get_conversations(experiment_id)

    # ========================================================================
    # Hidden State Operations
    # ========================================================================

    def persist_hidden_states(
        self,
        experiment_id: str,
        hidden_states: dict[int, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> list[HiddenStateRecord]:
        """
        Persist hidden states from multiple layers.

        Parameters:
            experiment_id: The experiment these states belong to.
            hidden_states: Dict mapping layer index to numpy array.
            metadata: Optional metadata to store with each array.

        Returns:
            List of created HiddenStateRecords.
        """
        records: list[HiddenStateRecord] = []
        timestamp = datetime.utcnow()

        for layer, array in hidden_states.items():
            # Save to HDF5
            file_path = self._storage.save(
                experiment_id=experiment_id,
                array=array,
                layer=layer,
                metadata=metadata,
            )

            # Create database record
            record = HiddenStateRecord(
                experiment_id=experiment_id,
                layer=layer,
                file_path=file_path,
                shape=tuple(array.shape),
                dtype=str(array.dtype),
                timestamp=timestamp,
            )
            record.id = self._db.insert_hidden_state(record)
            records.append(record)

            logger.debug(f"Persisted hidden state layer {layer} for {experiment_id}")

        logger.info(
            f"Persisted {len(records)} hidden state layers for {experiment_id}"
        )

        return records

    def persist_hidden_state(
        self,
        experiment_id: str,
        array: np.ndarray,
        layer: int,
        metadata: dict[str, Any] | None = None,
    ) -> HiddenStateRecord:
        """
        Persist a single hidden state array.

        Parameters:
            experiment_id: The experiment this state belongs to.
            array: The numpy array to persist.
            layer: The layer index for this hidden state.
            metadata: Optional metadata to store.

        Returns:
            The created HiddenStateRecord.
        """
        file_path = self._storage.save(
            experiment_id=experiment_id,
            array=array,
            layer=layer,
            metadata=metadata,
        )

        record = HiddenStateRecord(
            experiment_id=experiment_id,
            layer=layer,
            file_path=file_path,
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            timestamp=datetime.utcnow(),
        )
        record.id = self._db.insert_hidden_state(record)

        logger.debug(f"Persisted hidden state layer {layer} for {experiment_id}")

        return record

    def get_hidden_states(
        self,
        experiment_id: str,
        layer: int | None = None,
    ) -> list[HiddenStateRecord]:
        """
        Retrieve hidden state records for an experiment.

        Parameters:
            experiment_id: The experiment identifier.
            layer: Optional layer filter.

        Returns:
            List of HiddenStateRecords.
        """
        return self._db.get_hidden_states(experiment_id, layer)

    def load_hidden_state(self, record: HiddenStateRecord) -> np.ndarray:
        """
        Load the actual hidden state array for a record.

        Parameters:
            record: The HiddenStateRecord to load.

        Returns:
            The numpy array containing the hidden state data.
        """
        return self._storage.load(record.file_path)

    def load_hidden_state_with_metadata(
        self,
        record: HiddenStateRecord,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Load a hidden state array with its metadata.

        Parameters:
            record: The HiddenStateRecord to load.

        Returns:
            Tuple of (numpy array, metadata dict).
        """
        return self._storage.load_with_metadata(record.file_path)

    # ========================================================================
    # Metric Operations
    # ========================================================================

    def persist_metric(
        self,
        experiment_id: str,
        name: str,
        value: float,
        unit: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> MetricRecord:
        """
        Persist a single metric value.

        Parameters:
            experiment_id: The experiment this metric belongs to.
            name: Metric name/type (e.g., "perplexity", "loss").
            value: The numeric metric value.
            unit: Optional unit of measurement.
            metadata: Optional additional metadata.
            timestamp: Optional timestamp (current time if None).

        Returns:
            The created MetricRecord.
        """
        record = MetricRecord(
            experiment_id=experiment_id,
            name=name,
            value=value,
            unit=unit,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
        )

        record.id = self._db.insert_metric(record)
        logger.debug(f"Persisted metric {name}={value} for {experiment_id}")

        return record

    def persist_metrics(
        self,
        experiment_id: str,
        metrics: dict[str, float],
        unit: str | None = None,
        timestamp: datetime | None = None,
    ) -> list[MetricRecord]:
        """
        Persist multiple metrics at once.

        Parameters:
            experiment_id: The experiment these metrics belong to.
            metrics: Dict mapping metric name to value.
            unit: Optional unit (applied to all metrics).
            timestamp: Optional timestamp (current time if None).

        Returns:
            List of created MetricRecords.
        """
        ts = timestamp or datetime.utcnow()

        records = [
            MetricRecord(
                experiment_id=experiment_id,
                name=name,
                value=value,
                unit=unit,
                metadata={},
                timestamp=ts,
            )
            for name, value in metrics.items()
        ]

        self._db.insert_metrics(records)
        logger.debug(f"Persisted {len(records)} metrics for {experiment_id}")

        return records

    def get_metrics(
        self,
        experiment_id: str,
        metric_name: str | None = None,
    ) -> list[MetricRecord]:
        """
        Retrieve metrics for an experiment.

        Parameters:
            experiment_id: The experiment identifier.
            metric_name: Optional filter for specific metric type.

        Returns:
            List of MetricRecords.
        """
        return self._db.get_metrics(experiment_id, metric_name)

    # ========================================================================
    # Bulk Operations
    # ========================================================================

    def persist_experiment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        hidden_states: dict[int, np.ndarray] | None = None,
        metrics: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        notes: str | None = None,
    ) -> ExperimentRecord:
        """
        Persist a complete experiment with all associated data.

        This is the primary high-level method for persisting experiments.
        It creates the experiment record and optionally persists conversations,
        hidden states, and metrics in a single operation.

        Parameters:
            model: The model identifier used in the experiment.
            messages: Optional list of conversation messages.
            hidden_states: Optional dict of layer -> array hidden states.
            metrics: Optional dict of metric name -> value.
            config: Optional configuration dictionary.
            experiment_id: Optional custom ID (auto-generated if None).
            notes: Optional notes or description.

        Returns:
            The created ExperimentRecord.
        """
        # Create the experiment
        experiment = self.create_experiment(
            model=model,
            config=config,
            experiment_id=experiment_id,
            status="completed",
            notes=notes,
        )

        exp_id = experiment.id

        # Persist conversations if provided
        if messages:
            self.persist_conversations(exp_id, messages)

        # Persist hidden states if provided
        if hidden_states:
            self.persist_hidden_states(exp_id, hidden_states)

        # Persist metrics if provided
        if metrics:
            self.persist_metrics(exp_id, metrics)

        logger.info(f"Persisted complete experiment: {exp_id}")

        return experiment

    def get_experiment_data(
        self,
        experiment_id: str,
        include_hidden_states: bool = False,
    ) -> dict[str, Any] | None:
        """
        Retrieve complete experiment data.

        Parameters:
            experiment_id: The experiment identifier.
            include_hidden_states: Whether to load actual hidden state arrays.

        Returns:
            Dict with experiment, conversations, hidden_states, and metrics,
            or None if the experiment is not found.
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            return None

        conversations = self.get_conversations(experiment_id)
        hidden_state_records = self.get_hidden_states(experiment_id)
        metrics = self.get_metrics(experiment_id)

        result: dict[str, Any] = {
            "experiment": experiment,
            "conversations": conversations,
            "hidden_state_records": hidden_state_records,
            "metrics": metrics,
        }

        # Optionally load actual hidden state data
        if include_hidden_states:
            result["hidden_states"] = {
                record.layer: self.load_hidden_state(record)
                for record in hidden_state_records
            }

        return result


__all__ = [
    # Database models
    "DatabaseManager",
    "ExperimentRecord",
    "ConversationRecord",
    "MetricRecord",
    "HiddenStateRecord",
    "init_db",
    # Storage
    "HiddenStateStorage",
    # Query
    "ExperimentQuery",
    "ExperimentSummary",
    # Export
    "ExperimentExporter",
    "ExportError",
    # Manager
    "ExperimentPersistence",
]
