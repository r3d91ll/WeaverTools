"""HDF5-based storage for hidden state arrays.

This module provides efficient storage and retrieval of large hidden state
tensors using HDF5 format with compression.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class HiddenStateStorage:
    """Manager for HDF5-based hidden state storage.

    Provides methods for saving and loading large numerical arrays with
    compression and chunking for efficient storage and retrieval.
    """

    # Default compression settings
    DEFAULT_COMPRESSION = "gzip"
    DEFAULT_COMPRESSION_LEVEL = 4
    DEFAULT_CHUNK_SIZE = 1024

    def __init__(
        self,
        storage_dir: str | Path,
        compression: str = DEFAULT_COMPRESSION,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """
        Initialize the hidden state storage manager.

        Parameters:
            storage_dir: Directory where HDF5 files will be stored.
            compression: Compression algorithm to use (default: gzip).
            compression_level: Compression level 0-9 (default: 4).
            chunk_size: Chunk size for HDF5 datasets (default: 1024).
        """
        self._storage_dir = Path(storage_dir).expanduser()
        self._compression = compression
        self._compression_level = compression_level
        self._chunk_size = chunk_size

        # Ensure storage directory exists
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def storage_dir(self) -> Path:
        """Get the storage directory path."""
        return self._storage_dir

    def _generate_filename(self, experiment_id: str, layer: int | None = None) -> str:
        """
        Generate a unique filename for a hidden state file.

        Parameters:
            experiment_id: The experiment identifier.
            layer: Optional layer index to include in filename.

        Returns:
            A unique filename string.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        if layer is not None:
            return f"{experiment_id}_layer{layer}_{timestamp}.h5"
        return f"{experiment_id}_{timestamp}.h5"

    def _compute_chunks(self, shape: tuple[int, ...]) -> tuple[int, ...] | bool:
        """
        Compute appropriate chunk sizes for a given array shape.

        Parameters:
            shape: The shape of the array to be stored.

        Returns:
            Tuple of chunk sizes, or True for automatic chunking.
        """
        if len(shape) == 0:
            return True

        # For 1D arrays, use chunk_size or array length if smaller
        if len(shape) == 1:
            return (min(self._chunk_size, shape[0]),)

        # For 2D+ arrays, chunk along the first dimension
        chunks = list(shape)
        chunks[0] = min(self._chunk_size, shape[0])
        return tuple(chunks)

    def save(
        self,
        experiment_id: str,
        array: np.ndarray,
        layer: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a hidden state array to an HDF5 file.

        Parameters:
            experiment_id: The experiment identifier.
            array: The numpy array to save.
            layer: Optional layer index for the hidden state.
            metadata: Optional metadata dictionary to store with the array.

        Returns:
            The full path to the saved HDF5 file.
        """
        filename = self._generate_filename(experiment_id, layer)
        file_path = self._storage_dir / filename

        # Compute appropriate chunk sizes
        chunks = self._compute_chunks(array.shape)

        with h5py.File(file_path, "w") as f:
            # Create dataset with compression and shuffle filter
            dset = f.create_dataset(
                "hidden_state",
                data=array,
                dtype=array.dtype,
                compression=self._compression,
                compression_opts=self._compression_level,
                chunks=chunks,
                shuffle=True,  # Improves compression for floating point data
            )

            # Store array metadata as attributes
            dset.attrs["shape"] = array.shape
            dset.attrs["dtype"] = str(array.dtype)
            dset.attrs["experiment_id"] = experiment_id
            dset.attrs["created_at"] = datetime.utcnow().isoformat()

            if layer is not None:
                dset.attrs["layer"] = layer

            # Store additional metadata if provided
            if metadata:
                for key, value in metadata.items():
                    # Convert non-compatible types to strings
                    if isinstance(value, (str, int, float, bool)):
                        dset.attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        dset.attrs[key] = list(value)
                    else:
                        dset.attrs[key] = str(value)

        return str(file_path)

    def load(self, file_path: str | Path) -> np.ndarray:
        """
        Load a hidden state array from an HDF5 file.

        Parameters:
            file_path: Path to the HDF5 file.

        Returns:
            The loaded numpy array.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If the file does not contain a 'hidden_state' dataset.
        """
        path = Path(file_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        with h5py.File(path, "r") as f:
            if "hidden_state" not in f:
                raise KeyError(f"No 'hidden_state' dataset found in {path}")

            # Load the entire array into memory
            array: np.ndarray = f["hidden_state"][:]

        return array

    def load_with_metadata(
        self,
        file_path: str | Path,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Load a hidden state array and its metadata from an HDF5 file.

        Parameters:
            file_path: Path to the HDF5 file.

        Returns:
            Tuple of (numpy array, metadata dictionary).

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If the file does not contain a 'hidden_state' dataset.
        """
        path = Path(file_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        with h5py.File(path, "r") as f:
            if "hidden_state" not in f:
                raise KeyError(f"No 'hidden_state' dataset found in {path}")

            dset = f["hidden_state"]

            # Load array
            array: np.ndarray = dset[:]

            # Extract metadata from attributes
            metadata: dict[str, Any] = {}
            for key in dset.attrs.keys():
                value = dset.attrs[key]
                # Convert numpy types to Python types
                if isinstance(value, np.ndarray):
                    metadata[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    metadata[key] = value.item()
                else:
                    metadata[key] = value

        return array, metadata

    def delete(self, file_path: str | Path) -> bool:
        """
        Delete an HDF5 file.

        Parameters:
            file_path: Path to the HDF5 file to delete.

        Returns:
            True if the file was deleted, False if it did not exist.
        """
        path = Path(file_path).expanduser()

        if path.exists():
            path.unlink()
            return True
        return False

    def list_files(self, experiment_id: str | None = None) -> list[Path]:
        """
        List HDF5 files in the storage directory.

        Parameters:
            experiment_id: Optional filter for specific experiment.

        Returns:
            List of paths to HDF5 files.
        """
        pattern = f"{experiment_id}_*.h5" if experiment_id else "*.h5"
        return sorted(self._storage_dir.glob(pattern))

    def get_file_info(self, file_path: str | Path) -> dict[str, Any]:
        """
        Get information about an HDF5 file without loading the full array.

        Parameters:
            file_path: Path to the HDF5 file.

        Returns:
            Dictionary with file information including shape, dtype, and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        with h5py.File(path, "r") as f:
            if "hidden_state" not in f:
                raise KeyError(f"No 'hidden_state' dataset found in {path}")

            dset = f["hidden_state"]

            info: dict[str, Any] = {
                "shape": dset.shape,
                "dtype": str(dset.dtype),
                "size_bytes": dset.nbytes,
                "compression": dset.compression,
                "chunks": dset.chunks,
            }

            # Add stored attributes
            for key in dset.attrs.keys():
                value = dset.attrs[key]
                if isinstance(value, np.ndarray):
                    info[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    info[key] = value.item()
                else:
                    info[key] = value

        return info
