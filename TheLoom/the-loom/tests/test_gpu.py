"""Tests for GPU device management utilities."""

import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.gpu import GPUInfo, GPUManager


class TestGPUManagerInitialization:
    """Tests for GPUManager initialization."""

    def test_init_without_cuda(self):
        """Test initialization when CUDA is not available."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            assert manager.available_devices == []
            assert manager.allowed_devices == []
            assert manager.memory_fraction == 0.9

    def test_init_with_cuda(self):
        """Test initialization when CUDA is available."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager()

            assert manager.available_devices == [0, 1]
            assert manager.allowed_devices == [0, 1]
            assert manager.memory_fraction == 0.9

    def test_init_with_allowed_devices_filter(self):
        """Test initialization with specific allowed devices."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=4),
        ):
            manager = GPUManager(allowed_devices=[0, 2])

            assert manager.available_devices == [0, 1, 2, 3]
            assert manager.allowed_devices == [0, 2]

    def test_init_with_invalid_allowed_devices(self):
        """Test that invalid devices are filtered out during initialization."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            # Request device 5 which doesn't exist
            manager = GPUManager(allowed_devices=[0, 5])

            assert manager.available_devices == [0, 1]
            assert manager.allowed_devices == [0]  # Only valid device kept

    def test_init_custom_memory_fraction(self):
        """Test initialization with custom memory fraction."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager(memory_fraction=0.75)

            assert manager.memory_fraction == 0.75


class TestGPUManagerProperties:
    """Tests for GPUManager properties."""

    def test_has_gpu_true(self):
        """Test has_gpu returns True when GPUs are available."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
        ):
            manager = GPUManager()

            assert manager.has_gpu is True

    def test_has_gpu_false(self):
        """Test has_gpu returns False when no GPUs are available."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            assert manager.has_gpu is False

    def test_default_device_cpu_fallback(self):
        """Test default_device returns cpu when no GPUs are available."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            assert manager.default_device == "cpu"

    def test_default_device_first_gpu(self):
        """Test default_device returns first allowed GPU."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=3),
        ):
            manager = GPUManager(allowed_devices=[1, 2])

            assert manager.default_device == "cuda:1"

    def test_default_device_with_explicit_override(self):
        """Test default_device returns explicitly set device."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=3),
        ):
            manager = GPUManager(allowed_devices=[0, 1, 2])
            manager.set_default_device(2)

            assert manager.default_device == "cuda:2"


class TestSetDefaultDevice:
    """Tests for set_default_device method."""

    def test_set_default_device_valid(self):
        """Test setting a valid default device."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=3),
        ):
            manager = GPUManager(allowed_devices=[0, 1, 2])
            manager.set_default_device(1)

            assert manager.default_device == "cuda:1"

    def test_set_default_device_invalid(self):
        """Test setting an invalid default device raises ValueError."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager(allowed_devices=[0, 1])

            with pytest.raises(ValueError) as exc_info:
                manager.set_default_device(5)

            assert "Device 5 not in allowed_devices" in str(exc_info.value)

    def test_set_default_device_not_in_allowed(self):
        """Test setting a device not in allowed_devices raises ValueError."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=4),
        ):
            manager = GPUManager(allowed_devices=[0, 2])

            with pytest.raises(ValueError) as exc_info:
                manager.set_default_device(1)  # Device 1 exists but not allowed

            assert "Device 1 not in allowed_devices" in str(exc_info.value)

    def test_set_default_device_thread_safety(self):
        """Test that set_default_device is thread-safe."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=4),
        ):
            manager = GPUManager(allowed_devices=[0, 1, 2, 3])
            results = []
            errors = []

            def set_device(device_idx: int) -> None:
                try:
                    manager.set_default_device(device_idx)
                    results.append(device_idx)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=set_device, args=(i,))
                for i in [0, 1, 2, 3] * 5  # 20 concurrent operations
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All operations should succeed
            assert len(errors) == 0
            assert len(results) == 20
            # Final device should be one of the valid options
            assert manager._explicit_default_device in [0, 1, 2, 3]


class TestSetAllowedDevices:
    """Tests for set_allowed_devices method."""

    def test_set_allowed_devices_valid(self):
        """Test setting valid allowed devices."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=4),
        ):
            manager = GPUManager()
            manager.set_allowed_devices([0, 2])

            assert manager.allowed_devices == [0, 2]

    def test_set_allowed_devices_empty_raises(self):
        """Test setting empty allowed devices raises ValueError."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager()

            with pytest.raises(ValueError) as exc_info:
                manager.set_allowed_devices([])

            assert "Cannot set empty allowed_devices" in str(exc_info.value)

    def test_set_allowed_devices_invalid_device(self):
        """Test setting invalid device raises ValueError."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager()

            with pytest.raises(ValueError) as exc_info:
                manager.set_allowed_devices([0, 5])  # Device 5 doesn't exist

            assert "Invalid device indices: [5]" in str(exc_info.value)

    def test_set_allowed_devices_updates_default(self):
        """Test that setting allowed devices affects default device selection."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=4),
        ):
            manager = GPUManager(allowed_devices=[0, 1, 2, 3])
            manager.set_default_device(0)
            assert manager.default_device == "cuda:0"

            # Now restrict to only devices 2 and 3
            manager.set_allowed_devices([2, 3])

            # Explicit default (0) is no longer valid, should use first allowed
            assert manager.default_device == "cuda:2"

    def test_set_allowed_devices_thread_safety(self):
        """Test that set_allowed_devices is thread-safe."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=4),
        ):
            manager = GPUManager()
            errors = []

            def set_devices(devices: list[int]) -> None:
                try:
                    manager.set_allowed_devices(devices)
                except Exception as e:
                    errors.append(e)

            device_options = [[0, 1], [1, 2], [2, 3], [0, 1, 2, 3]]
            threads = [
                threading.Thread(target=set_devices, args=(devices,))
                for devices in device_options * 5  # 20 concurrent operations
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All operations should succeed
            assert len(errors) == 0
            # Final state should be one of the valid options
            assert manager.allowed_devices in device_options


class TestGetDevice:
    """Tests for get_device method."""

    def test_get_device_none_returns_default(self):
        """Test get_device with None returns default device."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager(allowed_devices=[0, 1])

            device = manager.get_device(None)

            assert device == torch.device("cuda:0")

    def test_get_device_int_valid(self):
        """Test get_device with valid integer."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager(allowed_devices=[0, 1])

            device = manager.get_device(1)

            assert device == torch.device("cuda:1")

    def test_get_device_int_invalid(self):
        """Test get_device with invalid integer raises ValueError."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager(allowed_devices=[0, 1])

            with pytest.raises(ValueError) as exc_info:
                manager.get_device(5)

            assert "Device 5 not in allowed devices" in str(exc_info.value)

    def test_get_device_string_valid(self):
        """Test get_device with valid string."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager(allowed_devices=[0, 1])

            device = manager.get_device("cuda:1")

            assert device == torch.device("cuda:1")

    def test_get_device_string_cpu(self):
        """Test get_device with cpu string."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager(allowed_devices=[0, 1])

            device = manager.get_device("cpu")

            assert device == torch.device("cpu")

    def test_get_device_string_invalid(self):
        """Test get_device with invalid CUDA string raises ValueError."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager(allowed_devices=[0])  # Only device 0 allowed

            with pytest.raises(ValueError) as exc_info:
                manager.get_device("cuda:1")  # Device 1 not allowed

            assert "Device cuda:1 not in allowed devices" in str(exc_info.value)


class TestToDict:
    """Tests for to_dict method."""

    def test_to_dict_no_gpu(self):
        """Test to_dict when no GPU is available."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            result = manager.to_dict()

            assert result["has_gpu"] is False
            assert result["default_device"] == "cpu"
            assert result["allowed_devices"] == []
            assert result["memory_fraction"] == 0.9
            assert result["gpus"] == []

    def test_to_dict_with_gpu(self):
        """Test to_dict when GPU is available."""
        mock_props = MagicMock()
        mock_props.name = "NVIDIA GeForce RTX 3080"
        mock_props.total_memory = 10 * (1024**3)  # 10 GB
        mock_props.major = 8
        mock_props.minor = 6

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
            patch.object(torch.cuda, "get_device_properties", return_value=mock_props),
            patch.object(torch.cuda, "set_device"),
            patch.object(
                torch.cuda,
                "mem_get_info",
                return_value=(8 * (1024**3), 10 * (1024**3)),  # 8GB free, 10GB total
            ),
        ):
            manager = GPUManager()

            result = manager.to_dict()

            assert result["has_gpu"] is True
            assert result["default_device"] == "cuda:0"
            assert result["allowed_devices"] == [0]
            assert result["memory_fraction"] == 0.9
            assert len(result["gpus"]) == 1
            assert result["gpus"][0]["index"] == 0
            assert result["gpus"][0]["name"] == "NVIDIA GeForce RTX 3080"
            assert result["gpus"][0]["compute_capability"] == "8.6"

    def test_to_dict_thread_safety(self):
        """Test that to_dict is thread-safe during configuration changes."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=4),
        ):
            manager = GPUManager()
            results = []
            errors = []

            def call_to_dict() -> None:
                try:
                    result = manager.to_dict()
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            def modify_config() -> None:
                try:
                    manager.set_allowed_devices([0, 1, 2])
                    manager.set_default_device(1)
                except Exception as e:
                    errors.append(e)

            # Run concurrent reads and writes
            threads = []
            for i in range(10):
                threads.append(threading.Thread(target=call_to_dict))
                threads.append(threading.Thread(target=modify_config))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All operations should succeed without errors
            assert len(errors) == 0
            assert len(results) == 10


class TestEstimateModelMemory:
    """Tests for estimate_model_memory method."""

    def test_estimate_float16(self):
        """Test memory estimation for float16 model."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            # 1 billion parameters in float16
            memory_gb = manager.estimate_model_memory(1_000_000_000, dtype="float16")

            # Expected: 1B * 2 bytes / 1024^3 * 1.2 = ~2.23 GB
            assert 2.2 < memory_gb < 2.4

    def test_estimate_float32(self):
        """Test memory estimation for float32 model."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            # 1 billion parameters in float32
            memory_gb = manager.estimate_model_memory(1_000_000_000, dtype="float32")

            # Expected: 1B * 4 bytes / 1024^3 * 1.2 = ~4.47 GB
            assert 4.4 < memory_gb < 4.6

    def test_estimate_int8(self):
        """Test memory estimation for int8 model."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            # 1 billion parameters in int8
            memory_gb = manager.estimate_model_memory(1_000_000_000, dtype="int8")

            # Expected: 1B * 1 byte / 1024^3 * 1.2 = ~1.12 GB
            assert 1.0 < memory_gb < 1.2

    def test_estimate_unknown_dtype(self):
        """Test memory estimation defaults to 2 bytes for unknown dtype."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            # Unknown dtype should default to 2 bytes per param
            memory_gb = manager.estimate_model_memory(1_000_000_000, dtype="unknown")

            # Expected: same as float16
            assert 2.2 < memory_gb < 2.4


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_gpu_info_creation(self):
        """Test GPUInfo dataclass creation."""
        info = GPUInfo(
            index=0,
            name="NVIDIA RTX 3080",
            total_memory_gb=10.0,
            free_memory_gb=8.0,
            used_memory_gb=2.0,
            utilization_percent=25.0,
            compute_capability=(8, 6),
        )

        assert info.index == 0
        assert info.name == "NVIDIA RTX 3080"
        assert info.total_memory_gb == 10.0
        assert info.free_memory_gb == 8.0
        assert info.used_memory_gb == 2.0
        assert info.utilization_percent == 25.0
        assert info.compute_capability == (8, 6)

    def test_gpu_info_none_utilization(self):
        """Test GPUInfo with None utilization."""
        info = GPUInfo(
            index=0,
            name="Test GPU",
            total_memory_gb=10.0,
            free_memory_gb=8.0,
            used_memory_gb=2.0,
            utilization_percent=None,
            compute_capability=(7, 5),
        )

        assert info.utilization_percent is None


class TestClearCache:
    """Tests for clear_cache method."""

    def test_clear_cache_no_gpu(self):
        """Test clear_cache does nothing when no GPU is available."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            # Should not raise any errors
            manager.clear_cache()
            manager.clear_cache(device=0)

    def test_clear_cache_all_devices(self):
        """Test clear_cache clears all devices when device is None."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "empty_cache") as mock_clear,
            patch.object(torch.cuda, "device") as mock_device_ctx,
        ):
            manager = GPUManager()
            manager.clear_cache(device=None)

            # Should have been called for each device
            assert mock_device_ctx.call_count == 2

    def test_clear_cache_specific_device_int(self):
        """Test clear_cache with specific device as integer."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "empty_cache") as mock_clear,
            patch.object(torch.cuda, "device") as mock_device_ctx,
        ):
            manager = GPUManager()
            manager.clear_cache(device=1)

            mock_device_ctx.assert_called_once_with(1)

    def test_clear_cache_specific_device_string(self):
        """Test clear_cache with specific device as string."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "empty_cache") as mock_clear,
            patch.object(torch.cuda, "device") as mock_device_ctx,
        ):
            manager = GPUManager()
            manager.clear_cache(device="cuda:1")

            mock_device_ctx.assert_called_once_with(1)

    def test_clear_cache_invalid_device_string(self):
        """Test clear_cache with invalid device string raises ValueError."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager = GPUManager()

            with pytest.raises(ValueError) as exc_info:
                manager.clear_cache(device="invalid")

            assert "Invalid device string format" in str(exc_info.value)


class TestGetBestDevice:
    """Tests for get_best_device method."""

    def test_get_best_device_no_gpu(self):
        """Test get_best_device returns cpu when no GPU available."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            manager = GPUManager()

            assert manager.get_best_device() == "cpu"

    def test_get_best_device_sufficient_memory(self):
        """Test get_best_device returns device with most free memory."""
        mock_props = MagicMock()
        mock_props.name = "Test GPU"
        mock_props.total_memory = 10 * (1024**3)
        mock_props.major = 8
        mock_props.minor = 0

        # Device 0: 4GB free, Device 1: 8GB free
        memory_info = {
            0: (4 * (1024**3), 10 * (1024**3)),
            1: (8 * (1024**3), 10 * (1024**3)),
        }

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "get_device_properties", return_value=mock_props),
            patch.object(torch.cuda, "set_device"),
            patch.object(
                torch.cuda,
                "mem_get_info",
                side_effect=lambda idx: memory_info[idx],
            ),
        ):
            manager = GPUManager()

            best = manager.get_best_device(required_memory_gb=2.0)

            # Should return device 1 as it has more free memory
            assert best == "cuda:1"

    def test_get_best_device_insufficient_memory_fallback(self):
        """Test get_best_device falls back to first device when memory insufficient."""
        mock_props = MagicMock()
        mock_props.name = "Test GPU"
        mock_props.total_memory = 10 * (1024**3)
        mock_props.major = 8
        mock_props.minor = 0

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "get_device_properties", return_value=mock_props),
            patch.object(torch.cuda, "set_device"),
            patch.object(
                torch.cuda,
                "mem_get_info",
                return_value=(2 * (1024**3), 10 * (1024**3)),  # 2GB free
            ),
        ):
            manager = GPUManager()

            # Require 100GB - more than any device has
            best = manager.get_best_device(required_memory_gb=100.0)

            # Should fall back to first allowed device
            assert best == "cuda:0"
