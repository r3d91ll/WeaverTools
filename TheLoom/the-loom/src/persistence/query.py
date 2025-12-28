"""Query interface for experiment persistence layer.

This module provides read-only querying and filtering of persisted experiments,
conversations, metrics, and hidden state references.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .database import (
    ConversationRecord,
    DatabaseManager,
    ExperimentRecord,
    HiddenStateRecord,
    MetricRecord,
)


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment."""

    experiment_id: str
    model: str
    created_at: datetime
    status: str
    conversation_count: int
    metric_count: int
    hidden_state_count: int


class ExperimentQuery:
    """Read-only query interface for experiment data.

    Provides methods for filtering and retrieving experiments, conversations,
    metrics, and hidden state references without modification capabilities.
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize the experiment query interface.

        Parameters:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path).expanduser()
        self._db = DatabaseManager(self._db_path)

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    # ========================================================================
    # Experiment Queries
    # ========================================================================

    def list_experiments(
        self,
        limit: int = 100,
        offset: int = 0,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        model: str | None = None,
        status: str | None = None,
    ) -> list[ExperimentRecord]:
        """
        List experiments with optional filtering.

        Parameters:
            limit: Maximum number of experiments to return.
            offset: Number of experiments to skip (for pagination).
            date_from: Filter experiments created on or after this datetime.
            date_to: Filter experiments created on or before this datetime.
            model: Filter by model name.
            status: Filter by experiment status.

        Returns:
            List of experiment records matching the filters.
        """
        # Use the base list_experiments for date/model filtering
        # Apply status filter at SQL level for correct pagination
        experiments = self._db.list_experiments(
            limit=limit,
            offset=offset,
            model=model,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )

        return experiments

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """
        Retrieve an experiment by its ID.

        Parameters:
            experiment_id: The unique identifier of the experiment.

        Returns:
            The experiment record if found, None otherwise.
        """
        return self._db.get_experiment(experiment_id)

    def get_experiment_summary(self, experiment_id: str) -> ExperimentSummary | None:
        """
        Get summary statistics for an experiment.

        Parameters:
            experiment_id: The unique identifier of the experiment.

        Returns:
            ExperimentSummary with counts, or None if experiment not found.
        """
        experiment = self._db.get_experiment(experiment_id)
        if experiment is None:
            return None

        conversations = self._db.get_conversations(experiment_id)
        metrics = self._db.get_metrics(experiment_id)
        hidden_states = self._db.get_hidden_states(experiment_id)

        return ExperimentSummary(
            experiment_id=experiment.id,
            model=experiment.model,
            created_at=experiment.created_at,
            status=experiment.status,
            conversation_count=len(conversations),
            metric_count=len(metrics),
            hidden_state_count=len(hidden_states),
        )

    def count_experiments(
        self,
        model: str | None = None,
        status: str | None = None,
    ) -> int:
        """
        Count experiments with optional filtering.

        Parameters:
            model: Optional model filter.
            status: Optional status filter.

        Returns:
            The count of matching experiments.
        """
        if model is None and status is None:
            return self._db.count_experiments()

        # For filtered counts, query and count
        cursor = self._db.connection.cursor()
        try:
            query = "SELECT COUNT(*) FROM experiments WHERE 1=1"
            params: list[Any] = []

            if model is not None:
                query += " AND model = ?"
                params.append(model)

            if status is not None:
                query += " AND status = ?"
                params.append(status)

            cursor.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            cursor.close()

    def find_experiments_by_config(
        self,
        config_key: str,
        config_value: Any,
        limit: int = 100,
    ) -> list[ExperimentRecord]:
        """
        Find experiments with a specific config value.

        Searches for experiments where the config JSON contains the specified
        key-value pair. Uses JSON extraction for SQLite.

        Parameters:
            config_key: The configuration key to search for.
            config_value: The expected value for the key.
            limit: Maximum number of results.

        Returns:
            List of matching experiment records.
        """
        cursor = self._db.connection.cursor()
        try:
            # SQLite JSON extraction
            cursor.execute(
                """
                SELECT id, created_at, model, config_json, status, notes
                FROM experiments
                WHERE json_extract(config_json, ?) = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"$.{config_key}", str(config_value), limit),
            )
            return [ExperimentRecord.from_row(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    # ========================================================================
    # Conversation Queries
    # ========================================================================

    def get_conversations(
        self,
        experiment_id: str,
        role: str | None = None,
    ) -> list[ConversationRecord]:
        """
        Retrieve conversations for an experiment.

        Parameters:
            experiment_id: The experiment identifier.
            role: Optional filter for specific role (e.g., "user", "assistant").

        Returns:
            List of conversation records in sequence order.
        """
        conversations = self._db.get_conversations(experiment_id)

        if role is not None:
            conversations = [c for c in conversations if c.role == role]

        return conversations

    def search_conversations(
        self,
        search_text: str,
        experiment_id: str | None = None,
        limit: int = 100,
    ) -> list[ConversationRecord]:
        """
        Search conversation content for text.

        Parameters:
            search_text: Text to search for in conversation content.
            experiment_id: Optional filter to specific experiment.
            limit: Maximum number of results.

        Returns:
            List of matching conversation records.
        """
        cursor = self._db.connection.cursor()
        try:
            if experiment_id is not None:
                cursor.execute(
                    """
                    SELECT id, experiment_id, sequence_num, timestamp, role, content
                    FROM conversations
                    WHERE experiment_id = ? AND content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (experiment_id, f"%{search_text}%", limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, experiment_id, sequence_num, timestamp, role, content
                    FROM conversations
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (f"%{search_text}%", limit),
                )
            return [ConversationRecord.from_row(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    # ========================================================================
    # Metric Queries
    # ========================================================================

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
            List of metric records.
        """
        return self._db.get_metrics(experiment_id, metric_name)

    def get_metric_names(self, experiment_id: str) -> list[str]:
        """
        Get distinct metric names for an experiment.

        Parameters:
            experiment_id: The experiment identifier.

        Returns:
            List of unique metric names.
        """
        cursor = self._db.connection.cursor()
        try:
            cursor.execute(
                "SELECT DISTINCT name FROM metrics WHERE experiment_id = ?",
                (experiment_id,),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_metric_range(
        self,
        metric_name: str,
        experiment_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Get min/max/avg for a metric across experiments.

        Parameters:
            metric_name: The metric name to analyze.
            experiment_ids: Optional list of experiment IDs to filter.

        Returns:
            Dict with 'min', 'max', 'avg' keys.
        """
        cursor = self._db.connection.cursor()
        try:
            if experiment_ids is not None and len(experiment_ids) > 0:
                placeholders = ",".join("?" for _ in experiment_ids)
                cursor.execute(
                    f"""
                    SELECT MIN(value), MAX(value), AVG(value)
                    FROM metrics
                    WHERE name = ? AND experiment_id IN ({placeholders})
                    """,
                    [metric_name] + experiment_ids,
                )
            else:
                cursor.execute(
                    """
                    SELECT MIN(value), MAX(value), AVG(value)
                    FROM metrics
                    WHERE name = ?
                    """,
                    (metric_name,),
                )

            row = cursor.fetchone()
            if row and row[0] is not None:
                return {
                    "min": row[0],
                    "max": row[1],
                    "avg": row[2],
                }
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        finally:
            cursor.close()

    # ========================================================================
    # Hidden State Queries
    # ========================================================================

    def get_hidden_states(
        self,
        experiment_id: str,
        layer: int | None = None,
    ) -> list[HiddenStateRecord]:
        """
        Retrieve hidden state records for an experiment.

        Parameters:
            experiment_id: The experiment identifier.
            layer: Optional filter for specific layer index.

        Returns:
            List of hidden state records.
        """
        return self._db.get_hidden_states(experiment_id, layer)

    def get_hidden_state_layers(self, experiment_id: str) -> list[int]:
        """
        Get list of layer indices with hidden states for an experiment.

        Parameters:
            experiment_id: The experiment identifier.

        Returns:
            List of layer indices, sorted ascending.
        """
        cursor = self._db.connection.cursor()
        try:
            cursor.execute(
                "SELECT DISTINCT layer FROM hidden_states WHERE experiment_id = ? ORDER BY layer",
                (experiment_id,),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()

    # ========================================================================
    # Cross-Entity Queries
    # ========================================================================

    def get_experiments_with_metric(
        self,
        metric_name: str,
        min_value: float | None = None,
        max_value: float | None = None,
        limit: int = 100,
    ) -> list[ExperimentRecord]:
        """
        Find experiments that have a specific metric, optionally filtered by value range.

        Parameters:
            metric_name: The metric name to search for.
            min_value: Optional minimum value threshold.
            max_value: Optional maximum value threshold.
            limit: Maximum number of results.

        Returns:
            List of experiment records.
        """
        cursor = self._db.connection.cursor()
        try:
            query = """
                SELECT DISTINCT e.id, e.created_at, e.model, e.config_json, e.status, e.notes
                FROM experiments e
                INNER JOIN metrics m ON e.id = m.experiment_id
                WHERE m.name = ?
            """
            params: list[Any] = [metric_name]

            if min_value is not None:
                query += " AND m.value >= ?"
                params.append(min_value)

            if max_value is not None:
                query += " AND m.value <= ?"
                params.append(max_value)

            query += " ORDER BY e.created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [ExperimentRecord.from_row(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_model_statistics(self) -> list[dict[str, Any]]:
        """
        Get experiment count and statistics grouped by model.

        Returns:
            List of dicts with 'model', 'count', 'latest', 'earliest' keys.
        """
        cursor = self._db.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT model, COUNT(*) as count,
                       MAX(created_at) as latest,
                       MIN(created_at) as earliest
                FROM experiments
                GROUP BY model
                ORDER BY count DESC
                """
            )
            return [
                {
                    "model": row[0],
                    "count": row[1],
                    "latest": datetime.fromisoformat(row[2]),
                    "earliest": datetime.fromisoformat(row[3]),
                }
                for row in cursor.fetchall()
            ]
        finally:
            cursor.close()


__all__ = [
    "ExperimentQuery",
    "ExperimentSummary",
]
