"""Export functionality for experiment data.

This module provides multi-format export capabilities for persisted experiments,
supporting CSV, JSON, and Parquet formats for research collaboration.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .database import (
    ConversationRecord,
    DatabaseManager,
    ExperimentRecord,
    HiddenStateRecord,
    MetricRecord,
)
from .storage import HiddenStateStorage


class ExportError(Exception):
    """Exception raised when export operations fail."""

    pass


class ExperimentExporter:
    """Multi-format exporter for experiment data.

    Provides methods for exporting experiments, conversations, metrics,
    and hidden state metadata to CSV, JSON, and Parquet formats.
    """

    def __init__(
        self,
        db_path: str | Path,
        storage_dir: str | Path,
    ) -> None:
        """
        Initialize the experiment exporter.

        Parameters:
            db_path: Path to the SQLite database file.
            storage_dir: Directory containing HDF5 hidden state files.
        """
        self._db_path = Path(db_path).expanduser()
        self._storage_dir = Path(storage_dir).expanduser()
        self._db = DatabaseManager(self._db_path)
        self._storage = HiddenStateStorage(self._storage_dir)

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    # ========================================================================
    # Data Collection Helpers
    # ========================================================================

    def _get_experiment_data(
        self,
        experiment_id: str,
    ) -> dict[str, Any] | None:
        """
        Collect all data for an experiment.

        Parameters:
            experiment_id: The experiment identifier.

        Returns:
            Dict with experiment, conversations, metrics, and hidden_states,
            or None if experiment not found.
        """
        experiment = self._db.get_experiment(experiment_id)
        if experiment is None:
            return None

        conversations = self._db.get_conversations(experiment_id)
        metrics = self._db.get_metrics(experiment_id)
        hidden_states = self._db.get_hidden_states(experiment_id)

        return {
            "experiment": experiment,
            "conversations": conversations,
            "metrics": metrics,
            "hidden_states": hidden_states,
        }

    def _experiment_to_dict(self, experiment: ExperimentRecord) -> dict[str, Any]:
        """
        Convert an ExperimentRecord to a serializable dictionary.

        Parameters:
            experiment: The experiment record to convert.

        Returns:
            A dictionary with serializable values.
        """
        return {
            "id": experiment.id,
            "created_at": experiment.created_at.isoformat(),
            "model": experiment.model,
            "config": experiment.config,
            "status": experiment.status,
            "notes": experiment.notes,
        }

    def _conversation_to_dict(self, conversation: ConversationRecord) -> dict[str, Any]:
        """
        Convert a ConversationRecord to a serializable dictionary.

        Parameters:
            conversation: The conversation record to convert.

        Returns:
            A dictionary with serializable values.
        """
        return {
            "id": conversation.id,
            "experiment_id": conversation.experiment_id,
            "sequence_num": conversation.sequence_num,
            "timestamp": conversation.timestamp.isoformat(),
            "role": conversation.role,
            "content": conversation.content,
        }

    def _metric_to_dict(self, metric: MetricRecord) -> dict[str, Any]:
        """
        Convert a MetricRecord to a serializable dictionary.

        Parameters:
            metric: The metric record to convert.

        Returns:
            A dictionary with serializable values.
        """
        return {
            "id": metric.id,
            "experiment_id": metric.experiment_id,
            "name": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "timestamp": metric.timestamp.isoformat(),
            "metadata": metric.metadata,
        }

    def _hidden_state_to_dict(self, hidden_state: HiddenStateRecord) -> dict[str, Any]:
        """
        Convert a HiddenStateRecord to a serializable dictionary.

        Parameters:
            hidden_state: The hidden state record to convert.

        Returns:
            A dictionary with serializable values.
        """
        return {
            "id": hidden_state.id,
            "experiment_id": hidden_state.experiment_id,
            "layer": hidden_state.layer,
            "file_path": hidden_state.file_path,
            "shape": list(hidden_state.shape),
            "dtype": hidden_state.dtype,
            "timestamp": hidden_state.timestamp.isoformat(),
        }

    def _ensure_parent_dir(self, path: Path) -> None:
        """
        Ensure the parent directory of a path exists.

        Parameters:
            path: The file path whose parent should be created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # JSON Export
    # ========================================================================

    def export_json(
        self,
        experiment_id: str,
        output_path: str | Path,
        include_hidden_state_data: bool = False,
        indent: int = 2,
    ) -> Path:
        """
        Export an experiment to JSON format.

        Parameters:
            experiment_id: The experiment identifier to export.
            output_path: Path for the output JSON file.
            include_hidden_state_data: Whether to include actual hidden state arrays.
                                       Warning: This can produce very large files.
            indent: JSON indentation level (default: 2).

        Returns:
            The path to the created JSON file.

        Raises:
            ExportError: If the experiment is not found or export fails.
        """
        data = self._get_experiment_data(experiment_id)
        if data is None:
            raise ExportError(f"Experiment not found: {experiment_id}")

        output = Path(output_path).expanduser()
        self._ensure_parent_dir(output)

        # Build the JSON structure
        export_data: dict[str, Any] = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "experiment": self._experiment_to_dict(data["experiment"]),
            "conversations": [
                self._conversation_to_dict(c) for c in data["conversations"]
            ],
            "metrics": [self._metric_to_dict(m) for m in data["metrics"]],
            "hidden_states": [
                self._hidden_state_to_dict(hs) for hs in data["hidden_states"]
            ],
        }

        # Optionally include actual hidden state data
        if include_hidden_state_data:
            hidden_state_arrays: dict[str, list[float]] = {}
            for hs in data["hidden_states"]:
                try:
                    array = self._storage.load(hs.file_path)
                    hidden_state_arrays[str(hs.layer)] = array.tolist()
                except FileNotFoundError:
                    hidden_state_arrays[str(hs.layer)] = []
            export_data["hidden_state_data"] = hidden_state_arrays

        try:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=indent, ensure_ascii=False)
        except OSError as e:
            raise ExportError(f"Failed to write JSON file: {e}") from e

        return output

    # ========================================================================
    # CSV Export
    # ========================================================================

    def export_csv(
        self,
        experiment_id: str,
        output_path: str | Path,
        include_metadata: bool = True,
    ) -> dict[str, Path]:
        """
        Export an experiment to CSV format.

        Creates multiple CSV files for different data types:
        - {output_path}_experiment.csv: Experiment metadata
        - {output_path}_conversations.csv: Conversation history
        - {output_path}_metrics.csv: Computed metrics
        - {output_path}_hidden_states.csv: Hidden state references

        Parameters:
            experiment_id: The experiment identifier to export.
            output_path: Base path for the output CSV files (without extension).
            include_metadata: Whether to include full metadata in CSV.

        Returns:
            Dict mapping data type to the created file path.

        Raises:
            ExportError: If the experiment is not found or export fails.
        """
        data = self._get_experiment_data(experiment_id)
        if data is None:
            raise ExportError(f"Experiment not found: {experiment_id}")

        base_path = Path(output_path).expanduser()
        # Remove .csv extension if provided
        if base_path.suffix.lower() == ".csv":
            base_path = base_path.with_suffix("")

        self._ensure_parent_dir(base_path)

        output_files: dict[str, Path] = {}

        try:
            # Export experiment metadata
            experiment_path = Path(f"{base_path}_experiment.csv")
            exp_dict = self._experiment_to_dict(data["experiment"])
            if not include_metadata:
                exp_dict.pop("config", None)
            else:
                # Flatten config to string for CSV
                exp_dict["config"] = json.dumps(exp_dict.get("config", {}))

            exp_df = pd.DataFrame([exp_dict])
            exp_df.to_csv(experiment_path, index=False)
            output_files["experiment"] = experiment_path

            # Export conversations
            if data["conversations"]:
                conv_path = Path(f"{base_path}_conversations.csv")
                conv_dicts = [
                    self._conversation_to_dict(c) for c in data["conversations"]
                ]
                conv_df = pd.DataFrame(conv_dicts)
                conv_df.to_csv(conv_path, index=False)
                output_files["conversations"] = conv_path

            # Export metrics
            if data["metrics"]:
                metrics_path = Path(f"{base_path}_metrics.csv")
                metric_dicts = [self._metric_to_dict(m) for m in data["metrics"]]
                # Flatten metadata to string for CSV
                for md in metric_dicts:
                    md["metadata"] = json.dumps(md.get("metadata", {}))
                metrics_df = pd.DataFrame(metric_dicts)
                metrics_df.to_csv(metrics_path, index=False)
                output_files["metrics"] = metrics_path

            # Export hidden state references
            if data["hidden_states"]:
                hs_path = Path(f"{base_path}_hidden_states.csv")
                hs_dicts = [
                    self._hidden_state_to_dict(hs) for hs in data["hidden_states"]
                ]
                # Flatten shape to string for CSV
                for hd in hs_dicts:
                    hd["shape"] = json.dumps(hd.get("shape", []))
                hs_df = pd.DataFrame(hs_dicts)
                hs_df.to_csv(hs_path, index=False)
                output_files["hidden_states"] = hs_path

        except OSError as e:
            raise ExportError(f"Failed to write CSV files: {e}") from e

        return output_files

    # ========================================================================
    # Parquet Export
    # ========================================================================

    def export_parquet(
        self,
        experiment_id: str,
        output_path: str | Path,
        compression: str = "snappy",
    ) -> dict[str, Path]:
        """
        Export an experiment to Parquet format.

        Creates multiple Parquet files for different data types:
        - {output_path}_experiment.parquet: Experiment metadata
        - {output_path}_conversations.parquet: Conversation history
        - {output_path}_metrics.parquet: Computed metrics
        - {output_path}_hidden_states.parquet: Hidden state references

        Parameters:
            experiment_id: The experiment identifier to export.
            output_path: Base path for the output Parquet files (without extension).
            compression: Compression codec ('snappy', 'gzip', 'brotli', 'lz4', None).

        Returns:
            Dict mapping data type to the created file path.

        Raises:
            ExportError: If the experiment is not found or export fails.
        """
        data = self._get_experiment_data(experiment_id)
        if data is None:
            raise ExportError(f"Experiment not found: {experiment_id}")

        base_path = Path(output_path).expanduser()
        # Remove .parquet extension if provided
        if base_path.suffix.lower() == ".parquet":
            base_path = base_path.with_suffix("")

        self._ensure_parent_dir(base_path)

        output_files: dict[str, Path] = {}

        try:
            # Export experiment metadata
            experiment_path = Path(f"{base_path}_experiment.parquet")
            exp_dict = self._experiment_to_dict(data["experiment"])
            # Flatten config to string for Parquet
            exp_dict["config"] = json.dumps(exp_dict.get("config", {}))
            exp_df = pd.DataFrame([exp_dict])
            exp_df.to_parquet(experiment_path, compression=compression, index=False)
            output_files["experiment"] = experiment_path

            # Export conversations
            if data["conversations"]:
                conv_path = Path(f"{base_path}_conversations.parquet")
                conv_dicts = [
                    self._conversation_to_dict(c) for c in data["conversations"]
                ]
                conv_df = pd.DataFrame(conv_dicts)
                conv_df.to_parquet(conv_path, compression=compression, index=False)
                output_files["conversations"] = conv_path

            # Export metrics
            if data["metrics"]:
                metrics_path = Path(f"{base_path}_metrics.parquet")
                metric_dicts = [self._metric_to_dict(m) for m in data["metrics"]]
                # Flatten metadata to string
                for md in metric_dicts:
                    md["metadata"] = json.dumps(md.get("metadata", {}))
                metrics_df = pd.DataFrame(metric_dicts)
                metrics_df.to_parquet(metrics_path, compression=compression, index=False)
                output_files["metrics"] = metrics_path

            # Export hidden state references
            if data["hidden_states"]:
                hs_path = Path(f"{base_path}_hidden_states.parquet")
                hs_dicts = [
                    self._hidden_state_to_dict(hs) for hs in data["hidden_states"]
                ]
                # Flatten shape to string
                for hd in hs_dicts:
                    hd["shape"] = json.dumps(hd.get("shape", []))
                hs_df = pd.DataFrame(hs_dicts)
                hs_df.to_parquet(hs_path, compression=compression, index=False)
                output_files["hidden_states"] = hs_path

        except OSError as e:
            raise ExportError(f"Failed to write Parquet files: {e}") from e

        return output_files

    # ========================================================================
    # Batch Export
    # ========================================================================

    def export_experiments(
        self,
        experiment_ids: list[str],
        output_dir: str | Path,
        format: str = "json",
    ) -> dict[str, Path | dict[str, Path]]:
        """
        Export multiple experiments to the specified format.

        Parameters:
            experiment_ids: List of experiment identifiers to export.
            output_dir: Directory to place exported files.
            format: Export format ('json', 'csv', or 'parquet').

        Returns:
            Dict mapping experiment_id to output path(s).

        Raises:
            ExportError: If export format is invalid or export fails.
        """
        if format not in ("json", "csv", "parquet"):
            raise ExportError(f"Invalid export format: {format}")

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        results: dict[str, Path | dict[str, Path]] = {}

        for exp_id in experiment_ids:
            try:
                if format == "json":
                    file_path = output_path / f"{exp_id}.json"
                    results[exp_id] = self.export_json(exp_id, file_path)
                elif format == "csv":
                    base_path = output_path / exp_id
                    results[exp_id] = self.export_csv(exp_id, base_path)
                else:  # parquet
                    base_path = output_path / exp_id
                    results[exp_id] = self.export_parquet(exp_id, base_path)
            except ExportError:
                # Skip experiments that don't exist
                continue

        return results

    # ========================================================================
    # Summary Export
    # ========================================================================

    def export_summary(
        self,
        output_path: str | Path,
        format: str = "csv",
        limit: int = 1000,
    ) -> Path:
        """
        Export a summary of all experiments.

        Creates a summary file containing basic information about all
        experiments without detailed conversations or hidden states.

        Parameters:
            output_path: Path for the output file.
            format: Export format ('csv', 'json', or 'parquet').
            limit: Maximum number of experiments to include.

        Returns:
            The path to the created summary file.

        Raises:
            ExportError: If export format is invalid or export fails.
        """
        if format not in ("csv", "json", "parquet"):
            raise ExportError(f"Invalid export format: {format}")

        experiments = self._db.list_experiments(limit=limit)
        output = Path(output_path).expanduser()
        self._ensure_parent_dir(output)

        summary_data = []
        for exp in experiments:
            conversations = self._db.get_conversations(exp.id)
            metrics = self._db.get_metrics(exp.id)
            hidden_states = self._db.get_hidden_states(exp.id)

            summary_data.append({
                "id": exp.id,
                "created_at": exp.created_at.isoformat(),
                "model": exp.model,
                "status": exp.status,
                "notes": exp.notes,
                "conversation_count": len(conversations),
                "metric_count": len(metrics),
                "hidden_state_count": len(hidden_states),
            })

        try:
            if format == "json":
                with open(output, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "export_timestamp": datetime.utcnow().isoformat(),
                            "total_experiments": len(summary_data),
                            "experiments": summary_data,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            else:
                df = pd.DataFrame(summary_data)
                if format == "csv":
                    df.to_csv(output, index=False)
                else:  # parquet
                    df.to_parquet(output, compression="snappy", index=False)

        except OSError as e:
            raise ExportError(f"Failed to write summary file: {e}") from e

        return output


__all__ = [
    "ExperimentExporter",
    "ExportError",
]
