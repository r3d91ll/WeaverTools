"""Tests for GPU device management utilities.

This module tests the GPU memory monitoring capabilities including:
- Peak memory tracking during operations
- Memory warning system with configurable thresholds
- Pre-allocation checks and memory status reporting
"""

import threading
from unittest.mock import MagicMock, PropertyMock, patch

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

# =============================================================================
# Additional monitoring tests from GPU memory optimization work
# =============================================================================

# =============================================================================
# Module-level test functions required by QA acceptance criteria
# These are wrappers around the class-based tests for pytest discovery
# =============================================================================


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.get_device_properties")
@patch("torch.cuda.set_device")
@patch("torch.cuda.mem_get_info")
@patch("torch.cuda.max_memory_allocated")
def test_gpu_manager_peak_tracking(
    mock_max_mem,
    mock_mem_info,
    mock_set_device,
    mock_props,
    mock_count,
    mock_available,
):
    """Verify GPUManager tracks peak memory correctly during operations.

    QA Acceptance Criteria: Verify GPUManager tracks peak memory correctly during operations.
    """
    # Setup mock GPU properties
    mock_device_props = MagicMock()
    mock_device_props.name = "NVIDIA RTX 4090"
    mock_device_props.total_memory = 24 * (1024**3)  # 24 GB
    mock_device_props.major = 8
    mock_device_props.minor = 9
    mock_props.return_value = mock_device_props

    # Setup memory info: (free_bytes, total_bytes)
    mock_mem_info.return_value = (20 * (1024**3), 24 * (1024**3))

    # Setup peak memory: 6 GB allocated at peak
    mock_max_mem.return_value = 6 * (1024**3)

    gpu = GPUManager()
    info = gpu.get_gpu_info(0)

    # Verify peak memory is tracked correctly
    assert info.peak_memory_gb == pytest.approx(6.0, abs=0.1)

    # Verify peak is included in memory status
    status = gpu.get_memory_status(0)
    assert status["devices"][0]["peak_gb"] == pytest.approx(6.0, abs=0.1)

    # Verify peak is included in to_dict() output
    gpu_dict = gpu.to_dict()
    assert "peak_memory_gb" in gpu_dict["gpus"][0]
    assert gpu_dict["gpus"][0]["peak_memory_gb"] == pytest.approx(6.0, abs=0.1)


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.get_device_properties")
@patch("torch.cuda.set_device")
@patch("torch.cuda.mem_get_info")
@patch("torch.cuda.max_memory_allocated", return_value=0)
def test_memory_warnings(
    mock_max_mem,
    mock_mem_info,
    mock_set_device,
    mock_props,
    mock_count,
    mock_available,
):
    """Verify warnings triggered at configured threshold (default 85%).

    QA Acceptance Criteria: Verify warnings triggered at configured threshold (default 85%).
    """
    mock_device_props = MagicMock()
    mock_device_props.name = "NVIDIA RTX 4090"
    mock_device_props.total_memory = 24 * (1024**3)  # 24 GB
    mock_device_props.major = 8
    mock_device_props.minor = 9
    mock_props.return_value = mock_device_props

    # Setup memory: 21 GB used out of 24 GB = 87.5% usage (above 85% threshold)
    free_bytes = 3 * (1024**3)  # 3 GB free
    total_bytes = 24 * (1024**3)  # 24 GB total
    mock_mem_info.return_value = (free_bytes, total_bytes)

    gpu = GPUManager()
    result = gpu.check_memory_threshold(threshold=0.85)

    # Should trigger warning since 87.5% > 85%
    assert result["devices_over_threshold"] == 1
    assert result["threshold"] == 0.85
    assert len(result["warnings"]) == 1

    warning = result["warnings"][0]
    assert warning["device"] == 0
    assert warning["usage_percent"] > 85.0
    assert warning["total_gb"] == pytest.approx(24.0, abs=0.1)

    # Also test that below threshold doesn't warn
    mock_mem_info.return_value = (12 * (1024**3), 24 * (1024**3))  # 50% usage
    result_low = gpu.check_memory_threshold(threshold=0.85)
    assert result_low["devices_over_threshold"] == 0
    assert len(result_low["warnings"]) == 0


# =============================================================================
# Class-based tests for more comprehensive coverage
# =============================================================================


class TestGPUManagerPeakTracking:
    """Tests for GPU manager peak memory tracking."""

    def test_gpu_info_has_peak_memory_field(self):
        """Verify GPUInfo dataclass includes peak_memory_gb field."""
        info = GPUInfo(
            index=0,
            name="Test GPU",
            total_memory_gb=24.0,
            free_memory_gb=20.0,
            used_memory_gb=4.0,
            peak_memory_gb=5.0,
            utilization_percent=None,
            compute_capability=(8, 6),
        )

        assert hasattr(info, "peak_memory_gb")
        assert info.peak_memory_gb == 5.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated")
    def test_gpu_manager_peak_tracking(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify GPUManager tracks peak memory correctly during operations."""
        # Setup mock GPU properties
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)  # 24 GB
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        # Setup memory info: (free_bytes, total_bytes)
        mock_mem_info.return_value = (20 * (1024**3), 24 * (1024**3))

        # Setup peak memory: 6 GB allocated at peak
        mock_max_mem.return_value = 6 * (1024**3)

        gpu = GPUManager()
        info = gpu.get_gpu_info(0)

        # Verify peak memory is tracked
        assert info.peak_memory_gb == pytest.approx(6.0, abs=0.1)

        # Verify peak is included in memory status
        status = gpu.get_memory_status(0)
        assert status["devices"][0]["peak_gb"] == pytest.approx(6.0, abs=0.1)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated")
    def test_peak_memory_included_in_to_dict(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify peak_memory_gb is included in to_dict() output."""
        # Setup mock GPU properties
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        mock_mem_info.return_value = (20 * (1024**3), 24 * (1024**3))
        mock_max_mem.return_value = 8 * (1024**3)  # 8 GB peak

        gpu = GPUManager()
        gpu_dict = gpu.to_dict()

        assert "gpus" in gpu_dict
        assert len(gpu_dict["gpus"]) == 1
        assert "peak_memory_gb" in gpu_dict["gpus"][0]
        assert gpu_dict["gpus"][0]["peak_memory_gb"] == pytest.approx(8.0, abs=0.1)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated")
    def test_peak_tracking_multiple_devices(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify peak memory is tracked independently per device."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        mock_mem_info.return_value = (20 * (1024**3), 24 * (1024**3))

        # Different peak memory for different devices
        def get_peak_memory(device_idx):
            return (3 + device_idx * 2) * (1024**3)  # Device 0: 3GB, Device 1: 5GB

        mock_max_mem.side_effect = get_peak_memory

        gpu = GPUManager()
        gpu_infos = gpu.get_gpu_info()

        assert len(gpu_infos) == 2
        assert gpu_infos[0].peak_memory_gb == pytest.approx(3.0, abs=0.1)
        assert gpu_infos[1].peak_memory_gb == pytest.approx(5.0, abs=0.1)


class TestMemoryWarnings:
    """Tests for memory warning system."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated", return_value=0)
    def test_memory_warnings(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify warnings triggered at configured threshold (default 85%)."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)  # 24 GB
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        # Setup memory: 21 GB used out of 24 GB = 87.5% usage (above 85% threshold)
        free_bytes = 3 * (1024**3)  # 3 GB free
        total_bytes = 24 * (1024**3)  # 24 GB total
        mock_mem_info.return_value = (free_bytes, total_bytes)

        gpu = GPUManager()
        result = gpu.check_memory_threshold(threshold=0.85)

        # Should trigger warning since 87.5% > 85%
        assert result["devices_over_threshold"] == 1
        assert result["threshold"] == 0.85
        assert len(result["warnings"]) == 1

        warning = result["warnings"][0]
        assert warning["device"] == 0
        assert warning["usage_percent"] > 85.0
        assert warning["total_gb"] == pytest.approx(24.0, abs=0.1)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated", return_value=0)
    def test_no_warning_below_threshold(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify no warning when memory usage is below threshold."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        # Setup memory: 12 GB used out of 24 GB = 50% usage (below 85% threshold)
        free_bytes = 12 * (1024**3)  # 12 GB free
        total_bytes = 24 * (1024**3)  # 24 GB total
        mock_mem_info.return_value = (free_bytes, total_bytes)

        gpu = GPUManager()
        result = gpu.check_memory_threshold(threshold=0.85)

        # Should not trigger warning since 50% < 85%
        assert result["devices_over_threshold"] == 0
        assert len(result["warnings"]) == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated", return_value=0)
    def test_custom_threshold(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify custom threshold values work correctly."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        # Setup memory: 18 GB used out of 24 GB = 75% usage
        free_bytes = 6 * (1024**3)
        total_bytes = 24 * (1024**3)
        mock_mem_info.return_value = (free_bytes, total_bytes)

        gpu = GPUManager()

        # At 85% threshold, should not warn (75% < 85%)
        result_85 = gpu.check_memory_threshold(threshold=0.85)
        assert result_85["devices_over_threshold"] == 0

        # At 70% threshold, should warn (75% > 70%)
        result_70 = gpu.check_memory_threshold(threshold=0.70)
        assert result_70["devices_over_threshold"] == 1
        assert result_70["threshold"] == 0.70

    @patch("torch.cuda.is_available", return_value=False)
    def test_no_warning_no_gpu(self, mock_available):
        """Verify check_memory_threshold handles no GPU gracefully."""
        gpu = GPUManager()
        result = gpu.check_memory_threshold(threshold=0.85)

        assert result["devices_checked"] == 0
        assert result["devices_over_threshold"] == 0
        assert len(result["warnings"]) == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated", return_value=0)
    def test_warning_specific_device(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify warning check can target a specific device."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        # High memory usage
        mock_mem_info.return_value = (2 * (1024**3), 24 * (1024**3))  # 91.7% usage

        gpu = GPUManager()

        # Check specific device
        result = gpu.check_memory_threshold(threshold=0.85, device=0)
        assert result["devices_checked"] == 1
        assert result["devices_over_threshold"] == 1
        assert result["warnings"][0]["device"] == 0


class TestGPUManagerMemoryStatus:
    """Tests for memory status reporting."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated")
    def test_get_memory_status(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify get_memory_status returns comprehensive status."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        mock_mem_info.return_value = (12 * (1024**3), 24 * (1024**3))  # 50% usage
        mock_max_mem.return_value = 10 * (1024**3)  # 10 GB peak

        gpu = GPUManager()
        status = gpu.get_memory_status()

        assert status["has_gpu"] is True
        assert len(status["devices"]) == 1

        device = status["devices"][0]
        assert device["index"] == 0
        assert device["name"] == "NVIDIA RTX 4090"
        assert device["total_gb"] == pytest.approx(24.0, abs=0.1)
        assert device["free_gb"] == pytest.approx(12.0, abs=0.1)
        assert device["used_gb"] == pytest.approx(12.0, abs=0.1)
        assert device["usage_percent"] == pytest.approx(50.0, abs=1.0)
        assert device["peak_gb"] == pytest.approx(10.0, abs=0.1)

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_memory_status_no_gpu(self, mock_available):
        """Verify get_memory_status handles no GPU gracefully."""
        gpu = GPUManager()
        status = gpu.get_memory_status()

        assert status["has_gpu"] is False
        assert status["devices"] == []


class TestGPUManagerCanAllocate:
    """Tests for memory pre-allocation checks."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated", return_value=0)
    def test_can_allocate_true(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify can_allocate returns True when memory is available."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        # 20 GB free out of 24 GB
        mock_mem_info.return_value = (20 * (1024**3), 24 * (1024**3))

        gpu = GPUManager(memory_fraction=0.9)

        # Should be able to allocate 10 GB
        assert gpu.can_allocate(10.0) is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.max_memory_allocated", return_value=0)
    def test_can_allocate_false(
        self,
        mock_max_mem,
        mock_mem_info,
        mock_set_device,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Verify can_allocate returns False when insufficient memory."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA RTX 4090"
        mock_device_props.total_memory = 24 * (1024**3)
        mock_device_props.major = 8
        mock_device_props.minor = 9
        mock_props.return_value = mock_device_props

        # Only 2 GB free out of 24 GB (8.3% free)
        mock_mem_info.return_value = (2 * (1024**3), 24 * (1024**3))

        gpu = GPUManager(memory_fraction=0.9)

        # Should not be able to allocate 10 GB
        assert gpu.can_allocate(10.0) is False

    @patch("torch.cuda.is_available", return_value=False)
    def test_can_allocate_no_gpu(self, mock_available):
        """Verify can_allocate returns False when no GPU available."""
        gpu = GPUManager()

        assert gpu.can_allocate(1.0) is False


# GPU-specific tests that require actual CUDA hardware
@pytest.mark.gpu
class TestGPUManagerHardware:
    """GPU-specific tests that require CUDA hardware.

    These tests require CUDA hardware and are marked with @pytest.mark.gpu.
    Run with: pytest -m gpu
    """

    @pytest.fixture
    def cuda_available(self):
        """Check if CUDA is available for GPU tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return True

    def test_real_gpu_peak_tracking(self, cuda_available):
        """Verify peak tracking works with actual GPU operations."""
        torch.cuda.reset_peak_memory_stats()

        gpu = GPUManager()

        # Get initial peak
        initial_info = gpu.get_gpu_info(0)
        initial_peak = initial_info.peak_memory_gb

        # Allocate some memory
        tensor = torch.randn(1000, 1000, 1000, device="cuda")

        # Get new peak
        after_alloc_info = gpu.get_gpu_info(0)
        after_alloc_peak = after_alloc_info.peak_memory_gb

        # Peak should have increased
        assert after_alloc_peak > initial_peak

        # Clean up
        del tensor
        torch.cuda.empty_cache()

    def test_real_memory_warning_threshold(self, cuda_available):
        """Verify memory warnings work on actual GPU."""
        gpu = GPUManager()

        # Get current status
        result = gpu.check_memory_threshold(threshold=0.85)

        # Should return valid result
        assert "threshold" in result
        assert "warnings" in result
        assert "devices_checked" in result
        assert result["devices_checked"] >= 1

    def test_real_memory_status(self, cuda_available):
        """Verify memory status reporting on actual GPU."""
        gpu = GPUManager()
        status = gpu.get_memory_status()

        assert status["has_gpu"] is True
        assert len(status["devices"]) >= 1

        device = status["devices"][0]
        assert device["total_gb"] > 0
        assert device["used_gb"] >= 0
        assert device["free_gb"] >= 0
        assert 0 <= device["usage_percent"] <= 100
        assert device["peak_gb"] >= 0
