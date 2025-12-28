"""Tests for activation patching utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.patching.cache import (
    ActivationCache,
    CachedActivation,
    CacheManager,
    CacheMetadata,
    compute_cache_size_estimate,
    get_cache_device_recommendation,
)
from src.patching.experiments import (
    ExecutionPath,
    ExperimentConfig,
    ExperimentRecord,
    MultiLayerPatchingStudy,
    PatchingExperiment,
    PatchingResult,
    PathOutput,
    PathRecorder,
    PathRecording,
    RecordingStore,
    compute_causal_effect,
)
from src.patching.hooks import (
    HookComponent,
    HookManager,
    HookPoint,
    HookRegistration,
    PatchingHookResult,
    build_hook_list,
    create_ablation_hook,
    create_mean_ablation_hook,
    create_noise_hook,
    create_patching_hook,
    get_hook_names_for_layer,
    validate_hook_shapes,
)


class TestHookComponent:
    """Tests for HookComponent enum."""

    def test_from_string_valid(self):
        assert HookComponent.from_string("resid_pre") == HookComponent.RESID_PRE
        assert HookComponent.from_string("resid_post") == HookComponent.RESID_POST
        assert HookComponent.from_string("attn") == HookComponent.ATTN
        assert HookComponent.from_string("mlp_post") == HookComponent.MLP_POST

    def test_from_string_case_insensitive(self):
        assert HookComponent.from_string("RESID_PRE") == HookComponent.RESID_PRE
        assert HookComponent.from_string("Resid_Pre") == HookComponent.RESID_PRE

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown hook component"):
            HookComponent.from_string("invalid_component")

    def test_all_components_have_values(self):
        for component in HookComponent:
            assert component.value is not None
            assert isinstance(component.value, str)


class TestHookPoint:
    """Tests for HookPoint dataclass."""

    def test_creation(self):
        hook_point = HookPoint(layer=5, component=HookComponent.RESID_PRE)
        assert hook_point.layer == 5
        assert hook_point.component == HookComponent.RESID_PRE
        assert hook_point.position is None
        assert hook_point.head is None

    def test_creation_with_position_and_head(self):
        hook_point = HookPoint(
            layer=3,
            component=HookComponent.ATTN_Q,
            position=10,
            head=2,
        )
        assert hook_point.position == 10
        assert hook_point.head == 2

    def test_to_hook_name_resid_pre(self):
        hook_point = HookPoint(layer=5, component=HookComponent.RESID_PRE)
        assert hook_point.to_hook_name(num_layers=12) == "blocks.5.hook_resid_pre"

    def test_to_hook_name_attn(self):
        hook_point = HookPoint(layer=3, component=HookComponent.ATTN)
        assert hook_point.to_hook_name(num_layers=12) == "blocks.3.attn.hook_result"

    def test_to_hook_name_mlp_post(self):
        hook_point = HookPoint(layer=0, component=HookComponent.MLP_POST)
        assert hook_point.to_hook_name(num_layers=12) == "blocks.0.mlp.hook_post"

    def test_to_hook_name_negative_layer(self):
        hook_point = HookPoint(layer=-1, component=HookComponent.RESID_POST)
        assert hook_point.to_hook_name(num_layers=12) == "blocks.11.hook_resid_post"

    def test_to_hook_name_attn_pattern(self):
        hook_point = HookPoint(layer=6, component=HookComponent.ATTN_PATTERN)
        assert hook_point.to_hook_name(num_layers=12) == "blocks.6.attn.hook_pattern"


class TestPatchingHookResult:
    """Tests for PatchingHookResult dataclass."""

    def test_creation(self):
        hook_point = HookPoint(layer=5, component=HookComponent.RESID_PRE)
        result = PatchingHookResult(
            hook_point=hook_point,
            original_shape=(1, 10, 768),
            patched_shape=(1, 10, 768),
            shape_matched=True,
        )
        assert result.shape_matched is True
        assert result.original_shape == (1, 10, 768)

    def test_creation_with_mismatch(self):
        hook_point = HookPoint(layer=5, component=HookComponent.RESID_PRE)
        result = PatchingHookResult(
            hook_point=hook_point,
            original_shape=(1, 10, 768),
            patched_shape=(1, 10, 512),
            shape_matched=False,
        )
        assert result.shape_matched is False


class TestHookRegistration:
    """Tests for HookRegistration dataclass."""

    def test_creation(self):
        hook_point = HookPoint(layer=5, component=HookComponent.RESID_PRE)
        hook_fn = lambda x, h: x
        registration = HookRegistration(
            hook_point=hook_point,
            hook_fn=hook_fn,
            is_active=True,
        )
        assert registration.is_active is True
        assert registration.handle is None

    def test_remove(self):
        hook_point = HookPoint(layer=5, component=HookComponent.RESID_PRE)
        registration = HookRegistration(
            hook_point=hook_point,
            hook_fn=lambda x, h: x,
            is_active=True,
        )
        registration.remove()
        assert registration.is_active is False


class TestValidateHookShapes:
    """Tests for validate_hook_shapes function."""

    def test_matching_shapes(self):
        original = torch.randn(1, 10, 768)
        patched = torch.randn(1, 10, 768)
        assert validate_hook_shapes(original, patched) is True

    def test_mismatched_shapes_strict(self):
        original = torch.randn(1, 10, 768)
        patched = torch.randn(1, 10, 512)
        with pytest.raises(ValueError, match="Hook shape mismatch"):
            validate_hook_shapes(original, patched, strict=True)

    def test_mismatched_shapes_non_strict(self):
        original = torch.randn(1, 10, 768)
        patched = torch.randn(1, 10, 512)
        assert validate_hook_shapes(original, patched, strict=False) is False

    def test_different_batch_size(self):
        original = torch.randn(2, 10, 768)
        patched = torch.randn(1, 10, 768)
        with pytest.raises(ValueError, match="Hook shape mismatch"):
            validate_hook_shapes(original, patched, strict=True)


class TestCreatePatchingHook:
    """Tests for create_patching_hook function."""

    def test_full_replacement(self):
        source = torch.tensor([[[1.0, 2.0, 3.0]]])
        target = torch.tensor([[[4.0, 5.0, 6.0]]])

        hook = create_patching_hook(source)
        result = hook(target.clone())

        torch.testing.assert_close(result, source)

    def test_position_specific_patching(self):
        source = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        target = torch.zeros(1, 3, 2)

        hook = create_patching_hook(source, position=1)
        result = hook(target.clone())

        # Only position 1 should be patched
        assert result[0, 0, 0] == 0.0  # Not patched
        assert result[0, 1, 0] == 3.0  # Patched
        assert result[0, 2, 0] == 0.0  # Not patched

    def test_head_specific_patching(self):
        source = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]])
        target = torch.zeros(1, 3, 1, 2)

        hook = create_patching_hook(source, head=1)
        result = hook(target.clone())

        # Only head 1 should be patched
        torch.testing.assert_close(result[:, 0, :, :], torch.zeros(1, 1, 2))
        torch.testing.assert_close(result[:, 1, :, :], source[:, 1, :, :])

    def test_position_and_head_patching(self):
        source = torch.randn(1, 4, 8, 64)  # [batch, heads, seq, head_dim]
        target = torch.zeros(1, 4, 8, 64)

        hook = create_patching_hook(source, position=3, head=2)
        result = hook(target.clone())

        # Only head 2, position 3 should be patched
        torch.testing.assert_close(
            result[:, 2, 3, :], source[:, 2, 3, :]
        )
        # Other positions should remain zero
        assert result[:, 0, 0, 0] == 0.0

    def test_shape_validation_enabled(self):
        source = torch.randn(1, 10, 768)
        target = torch.randn(1, 10, 512)

        hook = create_patching_hook(source, validate_shapes=True)
        with pytest.raises(ValueError, match="Hook shape mismatch"):
            hook(target)

    def test_shape_validation_disabled(self):
        source = torch.randn(1, 10, 768)
        target = torch.randn(1, 10, 768)

        hook = create_patching_hook(source, validate_shapes=False)
        result = hook(target.clone())
        # Should work without validation
        torch.testing.assert_close(result, source)


class TestCreateAblationHook:
    """Tests for create_ablation_hook function."""

    def test_full_ablation_zero(self):
        activation = torch.randn(1, 10, 768)

        hook = create_ablation_hook(value=0.0)
        result = hook(activation.clone())

        assert torch.all(result == 0.0)

    def test_full_ablation_nonzero(self):
        activation = torch.randn(1, 10, 768)

        hook = create_ablation_hook(value=5.0)
        result = hook(activation.clone())

        assert torch.all(result == 5.0)

    def test_position_specific_ablation(self):
        activation = torch.ones(1, 5, 10)

        hook = create_ablation_hook(value=0.0, position=2)
        result = hook(activation.clone())

        assert result[0, 0, 0] == 1.0  # Not ablated
        assert result[0, 2, 0] == 0.0  # Ablated
        assert result[0, 4, 0] == 1.0  # Not ablated

    def test_head_specific_ablation(self):
        activation = torch.ones(1, 4, 10, 64)  # [batch, heads, seq, head_dim]

        hook = create_ablation_hook(value=0.0, head=1)
        result = hook(activation.clone())

        assert result[0, 0, 0, 0] == 1.0  # Not ablated
        assert result[0, 1, 0, 0] == 0.0  # Ablated
        assert result[0, 2, 0, 0] == 1.0  # Not ablated


class TestCreateMeanAblationHook:
    """Tests for create_mean_ablation_hook function."""

    def test_mean_ablation_batch_dim(self):
        reference = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])

        hook = create_mean_ablation_hook(reference, dim=0)
        activation = torch.zeros(2, 2, 2)
        result = hook(activation)

        # Mean across batch dimension
        expected_mean = reference.mean(dim=0, keepdim=True)
        assert result.shape == activation.shape
        torch.testing.assert_close(result[0], expected_mean.squeeze(0))

    def test_mean_ablation_preserves_shape(self):
        reference = torch.randn(4, 10, 768)

        hook = create_mean_ablation_hook(reference, dim=0)
        activation = torch.randn(4, 10, 768)
        result = hook(activation)

        assert result.shape == activation.shape


class TestCreateNoiseHook:
    """Tests for create_noise_hook function."""

    def test_adds_noise(self):
        activation = torch.zeros(1, 10, 768)

        hook = create_noise_hook(scale=1.0)
        result = hook(activation.clone())

        # Result should not be all zeros
        assert not torch.all(result == 0.0)

    def test_noise_scale(self):
        torch.manual_seed(42)
        activation = torch.zeros(1, 1000, 768)

        hook = create_noise_hook(scale=0.1, seed=42)
        result = hook(activation.clone())

        # Standard deviation should be approximately the scale
        assert result.std().item() == pytest.approx(0.1, rel=0.1)

    def test_reproducible_with_seed(self):
        activation = torch.zeros(1, 10, 768)

        hook1 = create_noise_hook(scale=1.0, seed=123)
        hook2 = create_noise_hook(scale=1.0, seed=123)

        result1 = hook1(activation.clone())
        result2 = hook2(activation.clone())

        torch.testing.assert_close(result1, result2)

    def test_position_specific_noise(self):
        activation = torch.zeros(1, 5, 10)

        hook = create_noise_hook(scale=1.0, position=2, seed=42)
        result = hook(activation.clone())

        # Only position 2 should have noise
        assert torch.all(result[:, 0, :] == 0.0)
        assert not torch.all(result[:, 2, :] == 0.0)
        assert torch.all(result[:, 4, :] == 0.0)


class TestHookManager:
    """Tests for HookManager class."""

    def test_initialization(self):
        manager = HookManager()
        assert len(manager.active_hooks) == 0

    def test_register_patch_hook(self):
        manager = HookManager()
        hook_point = HookPoint(layer=5, component=HookComponent.RESID_PRE)
        source = torch.randn(1, 10, 768)

        registration = manager.register_patch_hook(hook_point, source)

        assert registration.is_active
        assert len(manager.active_hooks) == 1

    def test_register_ablation_hook(self):
        manager = HookManager()
        hook_point = HookPoint(layer=3, component=HookComponent.ATTN)

        registration = manager.register_ablation_hook(hook_point, value=0.0)

        assert registration.is_active
        assert len(manager.active_hooks) == 1

    def test_get_hooks_for_layer(self):
        manager = HookManager()
        source = torch.randn(1, 10, 768)

        manager.register_patch_hook(
            HookPoint(layer=5, component=HookComponent.RESID_PRE), source
        )
        manager.register_patch_hook(
            HookPoint(layer=5, component=HookComponent.RESID_POST), source
        )
        manager.register_patch_hook(
            HookPoint(layer=3, component=HookComponent.RESID_PRE), source
        )

        layer_5_hooks = manager.get_hooks_for_layer(5)
        assert len(layer_5_hooks) == 2

        layer_3_hooks = manager.get_hooks_for_layer(3)
        assert len(layer_3_hooks) == 1

    def test_get_hooks_for_component(self):
        manager = HookManager()
        source = torch.randn(1, 10, 768)

        manager.register_patch_hook(
            HookPoint(layer=5, component=HookComponent.RESID_PRE), source
        )
        manager.register_patch_hook(
            HookPoint(layer=3, component=HookComponent.RESID_PRE), source
        )
        manager.register_patch_hook(
            HookPoint(layer=5, component=HookComponent.ATTN), source
        )

        resid_hooks = manager.get_hooks_for_component(HookComponent.RESID_PRE)
        assert len(resid_hooks) == 2

    def test_remove_all_hooks(self):
        manager = HookManager()
        source = torch.randn(1, 10, 768)

        manager.register_patch_hook(
            HookPoint(layer=5, component=HookComponent.RESID_PRE), source
        )
        manager.register_patch_hook(
            HookPoint(layer=3, component=HookComponent.ATTN), source
        )

        count = manager.remove_all_hooks()
        assert count == 2
        assert len(manager.active_hooks) == 0

    def test_clear(self):
        manager = HookManager()
        source = torch.randn(1, 10, 768)

        manager.register_patch_hook(
            HookPoint(layer=5, component=HookComponent.RESID_PRE), source
        )

        manager.clear()
        assert len(manager.active_hooks) == 0


class TestBuildHookList:
    """Tests for build_hook_list function."""

    def test_builds_hook_list(self):
        manager = HookManager()
        source = torch.randn(1, 10, 768)

        reg1 = manager.register_patch_hook(
            HookPoint(layer=5, component=HookComponent.RESID_PRE), source
        )
        reg2 = manager.register_patch_hook(
            HookPoint(layer=3, component=HookComponent.ATTN), source
        )

        hooks = build_hook_list([reg1, reg2], num_layers=12)

        assert len(hooks) == 2
        assert hooks[0][0] == "blocks.5.hook_resid_pre"
        assert hooks[1][0] == "blocks.3.attn.hook_result"

    def test_excludes_inactive_hooks(self):
        manager = HookManager()
        source = torch.randn(1, 10, 768)

        reg1 = manager.register_patch_hook(
            HookPoint(layer=5, component=HookComponent.RESID_PRE), source
        )
        reg2 = manager.register_patch_hook(
            HookPoint(layer=3, component=HookComponent.ATTN), source
        )

        reg2.remove()

        hooks = build_hook_list([reg1, reg2], num_layers=12)
        assert len(hooks) == 1


class TestGetHookNamesForLayer:
    """Tests for get_hook_names_for_layer function."""

    def test_default_components(self):
        names = get_hook_names_for_layer(layer=5, num_layers=12)

        assert "blocks.5.hook_resid_pre" in names
        assert "blocks.5.hook_resid_post" in names
        assert "blocks.5.attn.hook_result" in names
        assert "blocks.5.mlp.hook_post" in names

    def test_specific_components(self):
        names = get_hook_names_for_layer(
            layer=3,
            components=[HookComponent.RESID_PRE, HookComponent.ATTN_Q],
            num_layers=12,
        )

        assert len(names) == 2
        assert "blocks.3.hook_resid_pre" in names
        assert "blocks.3.attn.hook_q" in names

    def test_negative_layer(self):
        names = get_hook_names_for_layer(
            layer=-1,
            components=[HookComponent.RESID_POST],
            num_layers=12,
        )

        assert names[0] == "blocks.11.hook_resid_post"


class TestCacheMetadata:
    """Tests for CacheMetadata dataclass."""

    def test_creation(self):
        metadata = CacheMetadata(
            layer=5,
            component="resid_pre",
            shape=(1, 10, 768),
            dtype="float32",
        )
        assert metadata.layer == 5
        assert metadata.component == "resid_pre"
        assert metadata.shape == (1, 10, 768)

    def test_to_dict(self):
        metadata = CacheMetadata(
            layer=5,
            component="resid_pre",
            shape=(1, 10, 768),
            dtype="float32",
            run_id="test_run",
        )
        d = metadata.to_dict()

        assert d["layer"] == 5
        assert d["component"] == "resid_pre"
        assert d["shape"] == [1, 10, 768]
        assert d["run_id"] == "test_run"


class TestCachedActivation:
    """Tests for CachedActivation dataclass."""

    def test_creation(self):
        tensor = torch.randn(1, 10, 768)
        metadata = CacheMetadata(
            layer=5,
            component="resid_pre",
            shape=(1, 10, 768),
            dtype="float32",
        )
        cached = CachedActivation(activation=tensor, metadata=metadata)

        assert cached.layer == 5
        assert cached.component == "resid_pre"
        assert cached.shape == (1, 10, 768)

    def test_size_bytes(self):
        tensor = torch.randn(1, 10, 768)  # float32 = 4 bytes per element
        metadata = CacheMetadata(
            layer=5,
            component="resid_pre",
            shape=(1, 10, 768),
            dtype="float32",
        )
        cached = CachedActivation(activation=tensor, metadata=metadata)

        expected_size = 1 * 10 * 768 * 4  # 4 bytes per float32
        assert cached.size_bytes == expected_size

    def test_to_numpy(self):
        tensor = torch.tensor([[[1.0, 2.0, 3.0]]])
        metadata = CacheMetadata(
            layer=5,
            component="resid_pre",
            shape=(1, 1, 3),
            dtype="float32",
        )
        cached = CachedActivation(activation=tensor, metadata=metadata)

        array = cached.to_numpy()
        np.testing.assert_array_almost_equal(array, [[[1.0, 2.0, 3.0]]])

    def test_clone(self):
        tensor = torch.randn(1, 10, 768)
        metadata = CacheMetadata(
            layer=5,
            component="resid_pre",
            shape=(1, 10, 768),
            dtype="float32",
        )
        cached = CachedActivation(activation=tensor, metadata=metadata)

        cloned = cached.clone()

        assert cloned.layer == cached.layer
        assert cloned.component == cached.component
        # Verify it's a true clone (different memory)
        cloned.activation[0, 0, 0] = 999.0
        assert tensor[0, 0, 0] != 999.0


class TestActivationCache:
    """Tests for ActivationCache class."""

    def test_initialization(self):
        cache = ActivationCache(run_id="test_run")
        assert cache.run_id == "test_run"
        assert cache.num_entries == 0
        assert cache.size_bytes == 0

    def test_store_and_get(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 10, 768)

        cached = cache.store(layer=5, component="resid_pre", activation=tensor)

        assert cache.num_entries == 1
        retrieved = cache.get(layer=5, component="resid_pre")
        assert retrieved is not None
        torch.testing.assert_close(retrieved.activation, tensor)

    def test_get_nonexistent(self):
        cache = ActivationCache()
        assert cache.get(layer=99, component="nonexistent") is None

    def test_get_tensor(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 10, 768)
        cache.store(layer=5, component="resid_pre", activation=tensor)

        retrieved = cache.get_tensor(layer=5, component="resid_pre")
        assert retrieved is not None
        torch.testing.assert_close(retrieved, tensor)

    def test_has(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 10, 768)
        cache.store(layer=5, component="resid_pre", activation=tensor)

        assert cache.has(layer=5, component="resid_pre")
        assert not cache.has(layer=5, component="resid_post")

    def test_remove(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 10, 768)
        cache.store(layer=5, component="resid_pre", activation=tensor)

        assert cache.remove(layer=5, component="resid_pre")
        assert cache.num_entries == 0
        assert not cache.remove(layer=5, component="resid_pre")  # Already removed

    def test_get_layers(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 10, 768)

        cache.store(layer=5, component="resid_pre", activation=tensor)
        cache.store(layer=3, component="resid_pre", activation=tensor)
        cache.store(layer=7, component="resid_pre", activation=tensor)

        layers = cache.get_layers()
        assert layers == [3, 5, 7]  # Sorted

    def test_get_components(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 10, 768)

        cache.store(layer=5, component="resid_pre", activation=tensor)
        cache.store(layer=5, component="resid_post", activation=tensor)
        cache.store(layer=5, component="attn", activation=tensor)

        components = cache.get_components(layer=5)
        assert set(components) == {"resid_pre", "resid_post", "attn"}

    def test_get_by_layer(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 10, 768)

        cache.store(layer=5, component="resid_pre", activation=tensor)
        cache.store(layer=5, component="resid_post", activation=tensor)
        cache.store(layer=3, component="resid_pre", activation=tensor)

        by_layer = cache.get_by_layer(5)
        assert len(by_layer) == 2
        assert "resid_pre" in by_layer
        assert "resid_post" in by_layer

    def test_clear(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 10, 768)

        cache.store(layer=5, component="resid_pre", activation=tensor)
        cache.store(layer=3, component="resid_pre", activation=tensor)

        count = cache.clear()
        assert count == 2
        assert cache.num_entries == 0

    def test_max_size_bytes_limit(self):
        cache = ActivationCache(max_size_bytes=1000)
        # Create a tensor larger than the limit
        large_tensor = torch.randn(100, 100, 100)  # Way larger than 1000 bytes

        with pytest.raises(MemoryError, match="Cache size limit exceeded"):
            cache.store(layer=0, component="resid_pre", activation=large_tensor)

    def test_clone(self):
        cache = ActivationCache(run_id="original")
        tensor = torch.randn(1, 10, 768)
        cache.store(layer=5, component="resid_pre", activation=tensor)

        cloned = cache.clone()

        assert cloned.run_id == cache.run_id
        assert cloned.num_entries == cache.num_entries
        # Verify independence
        cloned.clear()
        assert cache.num_entries == 1


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_initialization(self):
        manager = CacheManager()
        assert manager.num_caches == 0
        assert manager.total_size_bytes == 0

    def test_create_cache(self):
        manager = CacheManager()
        cache = manager.create_cache("test_cache")

        assert cache.run_id == "test_cache"
        assert manager.num_caches == 1
        assert "test_cache" in manager.cache_ids

    def test_create_duplicate_cache_raises(self):
        manager = CacheManager()
        manager.create_cache("test_cache")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_cache("test_cache")

    def test_get_cache(self):
        manager = CacheManager()
        cache = manager.create_cache("test_cache")

        retrieved = manager.get_cache("test_cache")
        assert retrieved is cache

        assert manager.get_cache("nonexistent") is None

    def test_get_or_create_cache(self):
        manager = CacheManager()

        # Creates new cache
        cache1 = manager.get_or_create_cache("test_cache")
        assert manager.num_caches == 1

        # Returns existing cache
        cache2 = manager.get_or_create_cache("test_cache")
        assert cache1 is cache2
        assert manager.num_caches == 1

    def test_remove_cache(self):
        manager = CacheManager()
        manager.create_cache("test_cache")

        assert manager.remove_cache("test_cache")
        assert manager.num_caches == 0
        assert not manager.remove_cache("test_cache")  # Already removed

    def test_clear_memory(self):
        manager = CacheManager()
        cache1 = manager.create_cache("cache1")
        cache2 = manager.create_cache("cache2")

        tensor = torch.randn(1, 10, 768)
        cache1.store(layer=0, component="resid_pre", activation=tensor)
        cache2.store(layer=0, component="resid_pre", activation=tensor)

        total_cleared = manager.clear_memory()
        assert total_cleared == 2
        assert manager.num_caches == 0

    def test_get_memory_stats(self):
        manager = CacheManager(max_size_mb=100)
        cache = manager.create_cache("test_cache")
        tensor = torch.randn(1, 10, 768)
        cache.store(layer=0, component="resid_pre", activation=tensor)

        stats = manager.get_memory_stats()

        assert stats["num_caches"] == 1
        assert stats["total_size_bytes"] > 0
        assert "test_cache" in stats["cache_sizes"]

    def test_check_memory_available(self):
        manager = CacheManager(max_size_mb=1)  # 1 MB limit
        assert manager.check_memory_available(1000)  # 1 KB should fit
        assert not manager.check_memory_available(10 * 1024 * 1024)  # 10 MB won't fit

    def test_persist_and_load_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir)
            cache = manager.create_cache("test_cache")
            tensor = torch.randn(1, 10, 768)
            cache.store(layer=5, component="resid_pre", activation=tensor)

            # Persist to disk
            path = manager.persist_cache("test_cache")
            assert path is not None
            assert path.exists()

            # Clear memory and reload
            manager.clear_memory()
            loaded = manager.load_cache("test_cache")

            assert loaded is not None
            assert loaded.num_entries == 1
            retrieved = loaded.get_tensor(layer=5, component="resid_pre")
            torch.testing.assert_close(retrieved, tensor)


class TestComputeCacheSizeEstimate:
    """Tests for compute_cache_size_estimate function."""

    def test_basic_estimate(self):
        estimate = compute_cache_size_estimate(
            num_layers=12,
            hidden_dim=768,
            seq_length=512,
            batch_size=1,
            components_per_layer=2,
            dtype_bytes=4,
        )

        # 12 layers * 2 components * 1 batch * 512 seq * 768 hidden * 4 bytes
        expected = 12 * 2 * 1 * 512 * 768 * 4
        assert estimate == expected


class TestGetCacheDeviceRecommendation:
    """Tests for get_cache_device_recommendation function."""

    def test_cpu_model_stays_on_cpu(self):
        recommendation = get_cache_device_recommendation(
            model_device="cpu",
            cache_size_estimate=1024 * 1024,
        )
        assert recommendation == "cpu"

    def test_gpu_with_enough_memory(self):
        recommendation = get_cache_device_recommendation(
            model_device="cuda:0",
            cache_size_estimate=1024 * 1024,  # 1 MB
            gpu_memory_available=100 * 1024 * 1024,  # 100 MB
        )
        assert recommendation == "cuda:0"

    def test_gpu_without_enough_memory(self):
        recommendation = get_cache_device_recommendation(
            model_device="cuda:0",
            cache_size_estimate=90 * 1024 * 1024,  # 90 MB
            gpu_memory_available=100 * 1024 * 1024,  # 100 MB (20% headroom means only 80 MB usable)
        )
        assert recommendation == "cpu"


class TestExecutionPath:
    """Tests for ExecutionPath enum."""

    def test_values(self):
        assert ExecutionPath.CLEAN.value == "clean"
        assert ExecutionPath.CORRUPTED.value == "corrupted"
        assert ExecutionPath.PATCHED.value == "patched"


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_creation(self):
        config = ExperimentConfig(
            name="test_experiment",
            layers=[0, 5, 10],
            components=["resid_pre", "attn"],
        )
        assert config.name == "test_experiment"
        assert config.layers == [0, 5, 10]
        assert config.batch_size == 1  # Default

    def test_get_hook_components(self):
        config = ExperimentConfig(
            name="test",
            components=["resid_pre", "attn", "mlp_post"],
        )
        components = config.get_hook_components()

        assert len(components) == 3
        assert HookComponent.RESID_PRE in components
        assert HookComponent.ATTN in components
        assert HookComponent.MLP_POST in components


class TestPathOutput:
    """Tests for PathOutput dataclass."""

    def test_creation(self):
        output = PathOutput(
            path_type=ExecutionPath.CLEAN,
            output=torch.randn(1, 10, 50257),
            cache=None,
            input_text="Hello world",
        )
        assert output.path_type == ExecutionPath.CLEAN
        assert output.input_text == "Hello world"
        assert not output.has_cache

    def test_has_cache(self):
        cache = ActivationCache()
        cache.store(layer=0, component="resid_pre", activation=torch.randn(1, 10, 768))

        output = PathOutput(
            path_type=ExecutionPath.CLEAN,
            output=None,
            cache=cache,
        )
        assert output.has_cache

    def test_to_dict(self):
        output = PathOutput(
            path_type=ExecutionPath.CORRUPTED,
            output=torch.randn(1, 10, 50257),
            cache=None,
            input_text="Test input",
            generation_time_ms=100.5,
        )
        d = output.to_dict()

        assert d["path_type"] == "corrupted"
        assert d["has_output"] is True
        assert d["input_text"] == "Test input"
        assert d["generation_time_ms"] == 100.5


class TestPatchingResult:
    """Tests for PatchingResult dataclass."""

    def test_creation(self):
        result = PatchingResult(
            layer=5,
            component="resid_pre",
            hook_name="blocks.5.hook_resid_pre",
            original_output=None,
            patched_output=torch.randn(1, 10, 50257),
            source_run_id="clean_run",
            target_run_id="patched_run",
        )
        assert result.layer == 5
        assert result.shape_matched is True  # Default

    def test_to_dict(self):
        result = PatchingResult(
            layer=5,
            component="resid_pre",
            hook_name="blocks.5.hook_resid_pre",
            original_output=None,
            patched_output=None,
            source_run_id="clean",
            target_run_id="patched",
            execution_time_ms=50.0,
        )
        d = result.to_dict()

        assert d["layer"] == 5
        assert d["hook_name"] == "blocks.5.hook_resid_pre"
        assert d["execution_time_ms"] == 50.0


class TestPathRecorder:
    """Tests for PathRecorder class."""

    def test_initialization(self):
        recorder = PathRecorder(experiment_id="test_exp")
        assert recorder.experiment_id == "test_exp"
        assert recorder.clean_path is None
        assert recorder.corrupted_path is None
        assert len(recorder.patched_paths) == 0

    def test_record_clean_path(self):
        recorder = PathRecorder()
        output = torch.randn(1, 10, 50257)

        path = recorder.record_clean_path(output=output, input_text="Hello")

        assert path.path_type == ExecutionPath.CLEAN
        assert recorder.clean_path is path
        assert recorder.has_path(ExecutionPath.CLEAN)

    def test_record_corrupted_path(self):
        recorder = PathRecorder()
        output = torch.randn(1, 10, 50257)

        path = recorder.record_corrupted_path(output=output, input_text="Corrupted")

        assert path.path_type == ExecutionPath.CORRUPTED
        assert recorder.corrupted_path is path

    def test_record_patched_path(self):
        recorder = PathRecorder()
        output = torch.randn(1, 10, 50257)

        path = recorder.record_patched_path(
            output=output,
            patch_info={"layer": 5, "component": "resid_pre"},
        )

        assert path.path_type == ExecutionPath.PATCHED
        assert len(recorder.patched_paths) == 1
        assert path.metadata["patch_info"]["layer"] == 5

    def test_has_all_paths(self):
        recorder = PathRecorder()
        output = torch.randn(1, 10, 50257)

        assert not recorder.has_all_paths()

        recorder.record_clean_path(output=output)
        assert not recorder.has_all_paths()

        recorder.record_corrupted_path(output=output)
        assert not recorder.has_all_paths()

        recorder.record_patched_path(output=output)
        assert recorder.has_all_paths()

    def test_iter_paths(self):
        recorder = PathRecorder()
        output = torch.randn(1, 10, 50257)

        recorder.record_clean_path(output=output)
        recorder.record_corrupted_path(output=output)
        recorder.record_patched_path(output=output)
        recorder.record_patched_path(output=output)

        paths = list(recorder.iter_paths())
        assert len(paths) == 4

    def test_clear(self):
        recorder = PathRecorder()
        output = torch.randn(1, 10, 50257)

        recorder.record_clean_path(output=output)
        recorder.record_patched_path(output=output)

        recorder.clear()
        assert recorder.clean_path is None
        assert len(recorder.patched_paths) == 0


class TestPathRecording:
    """Tests for PathRecording dataclass."""

    def test_from_recorder(self):
        recorder = PathRecorder(experiment_id="exp_123")
        output = torch.randn(1, 10, 50257)

        recorder.record_clean_path(output=output)
        recorder.record_corrupted_path(output=output)
        recorder.record_patched_path(output=output)

        recording = PathRecording.from_recorder(recorder, experiment_name="test")

        assert recording.recording_id == "exp_123"
        assert recording.experiment_name == "test"
        assert recording.has_all_paths

    def test_get_patched_by_layer(self):
        recording = PathRecording(
            recording_id="test",
            experiment_name="test",
        )

        # Add patched outputs with different layers
        recording.patched_outputs.append(
            PathOutput(
                path_type=ExecutionPath.PATCHED,
                output=None,
                cache=None,
                metadata={"patch_info": {"layer": 5, "component": "resid_pre"}},
            )
        )
        recording.patched_outputs.append(
            PathOutput(
                path_type=ExecutionPath.PATCHED,
                output=None,
                cache=None,
                metadata={"patch_info": {"layer": 3, "component": "resid_pre"}},
            )
        )
        recording.patched_outputs.append(
            PathOutput(
                path_type=ExecutionPath.PATCHED,
                output=None,
                cache=None,
                metadata={"patch_info": {"layer": 5, "component": "attn"}},
            )
        )

        layer_5_patches = recording.get_patched_by_layer(5)
        assert len(layer_5_patches) == 2

    def test_get_patched_by_component(self):
        recording = PathRecording(
            recording_id="test",
            experiment_name="test",
        )

        recording.patched_outputs.append(
            PathOutput(
                path_type=ExecutionPath.PATCHED,
                output=None,
                cache=None,
                metadata={"patch_info": {"layer": 5, "component": "resid_pre"}},
            )
        )
        recording.patched_outputs.append(
            PathOutput(
                path_type=ExecutionPath.PATCHED,
                output=None,
                cache=None,
                metadata={"patch_info": {"layer": 3, "component": "attn"}},
            )
        )

        resid_patches = recording.get_patched_by_component("resid_pre")
        assert len(resid_patches) == 1


class TestRecordingStore:
    """Tests for RecordingStore class."""

    def test_initialization(self):
        store = RecordingStore()
        assert store.num_recordings == 0

    def test_record_path_output(self):
        store = RecordingStore()
        output = torch.randn(1, 10, 50257)

        path_output = store.record_clean_path(
            recording_id="rec_1",
            output=output,
            input_text="Hello",
            experiment_name="test_exp",
        )

        assert store.num_recordings == 1
        recording = store.get_recording("rec_1")
        assert recording is not None
        assert recording.clean_output is not None

    def test_record_all_paths(self):
        store = RecordingStore()
        output = torch.randn(1, 10, 50257)

        store.record_clean_path(recording_id="rec_1", output=output)
        store.record_corrupted_path(recording_id="rec_1", output=output)
        store.record_patched_path(recording_id="rec_1", output=output, layer=5, component="resid_pre")

        recording = store.get_recording("rec_1")
        assert recording.has_all_paths

    def test_get_complete_recordings(self):
        store = RecordingStore()
        output = torch.randn(1, 10, 50257)

        # Complete recording
        store.record_clean_path(recording_id="complete", output=output)
        store.record_corrupted_path(recording_id="complete", output=output)
        store.record_patched_path(recording_id="complete", output=output, layer=0, component="resid_pre")

        # Incomplete recording
        store.record_clean_path(recording_id="incomplete", output=output)

        complete = store.get_complete_recordings()
        assert len(complete) == 1
        assert complete[0].recording_id == "complete"

    def test_save_and_load_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecordingStore(storage_dir=tmpdir)
            output = torch.randn(1, 10, 100)

            store.record_clean_path(recording_id="test", output=output)
            store.record_corrupted_path(recording_id="test", output=output)

            # Save
            assert store.save_recording("test")

            # Clear and reload
            store.clear()
            assert store.num_recordings == 0

            loaded = store.load_recording("test")
            assert loaded is not None
            assert loaded.clean_output is not None

    def test_remove_recording(self):
        store = RecordingStore()
        output = torch.randn(1, 10, 50257)

        store.record_clean_path(recording_id="rec_1", output=output)
        assert store.num_recordings == 1

        assert store.remove_recording("rec_1")
        assert store.num_recordings == 0
        assert not store.remove_recording("rec_1")  # Already removed


class TestExperimentRecord:
    """Tests for ExperimentRecord dataclass."""

    def test_creation(self):
        config = ExperimentConfig(name="test")
        record = ExperimentRecord(
            experiment_id="exp_123",
            config=config,
        )
        assert record.experiment_id == "exp_123"
        assert record.status == "pending"
        assert not record.has_all_paths

    def test_duration(self):
        config = ExperimentConfig(name="test")
        record = ExperimentRecord(
            experiment_id="exp_123",
            config=config,
            started_at=1000.0,
            completed_at=1500.0,
        )
        assert record.duration_ms == 500000.0  # 500 seconds in ms


class TestPatchingExperiment:
    """Tests for PatchingExperiment class."""

    def test_initialization(self):
        config = ExperimentConfig(name="test")
        experiment = PatchingExperiment(config=config)

        assert experiment.config is config
        assert experiment.experiment_id is not None
        assert experiment.record is None

    def test_create_record(self):
        config = ExperimentConfig(name="test")
        experiment = PatchingExperiment(config=config)

        record = experiment.create_record()

        assert record.status == "running"
        assert experiment.record is record

    def test_finalize_success(self):
        config = ExperimentConfig(name="test", cleanup_on_completion=False)
        experiment = PatchingExperiment(config=config)

        record = experiment.finalize(success=True)

        assert record.status == "completed"
        assert record.error is None

    def test_finalize_failure(self):
        config = ExperimentConfig(name="test", cleanup_on_completion=False)
        experiment = PatchingExperiment(config=config)

        record = experiment.finalize(success=False, error="Test error")

        assert record.status == "failed"
        assert record.error == "Test error"


class TestMultiLayerPatchingStudy:
    """Tests for MultiLayerPatchingStudy class."""

    def test_initialization(self):
        study = MultiLayerPatchingStudy(
            name="layer_sweep",
            layers=[0, 5, 10],
            components=["resid_pre", "attn"],
        )
        assert study.name == "layer_sweep"
        assert len(study.experiments) == 0

    def test_create_experiment(self):
        study = MultiLayerPatchingStudy(
            name="layer_sweep",
            layers=[0, 5],
        )

        experiment = study.create_experiment()

        assert len(study.experiments) == 1
        assert experiment.config.layers == [0, 5]

    def test_aggregate_results(self):
        study = MultiLayerPatchingStudy(name="test")

        aggregated = study.aggregate_results()

        assert aggregated["study_name"] == "test"
        assert aggregated["num_experiments"] == 0


class TestComputeCausalEffect:
    """Tests for compute_causal_effect function."""

    def test_positive_recovery(self):
        result = compute_causal_effect(
            clean_metric=1.0,
            corrupted_metric=0.5,
            patched_metric=0.8,
        )

        assert result["causal_effect"] == 0.3  # 0.8 - 0.5
        assert result["corruption_delta"] == -0.5  # 0.5 - 1.0
        assert result["recovery_rate"] == pytest.approx(0.6)  # 0.3 / 0.5

    def test_full_recovery(self):
        result = compute_causal_effect(
            clean_metric=1.0,
            corrupted_metric=0.0,
            patched_metric=1.0,
        )

        assert result["causal_effect"] == 1.0
        assert result["recovery_rate"] == pytest.approx(1.0)

    def test_no_corruption(self):
        result = compute_causal_effect(
            clean_metric=1.0,
            corrupted_metric=1.0,
            patched_metric=1.0,
        )

        assert result["causal_effect"] == 0.0
        # When corruption_delta is near zero, recovery_rate should be 0 or inf
        assert result["recovery_rate"] == 0.0

    def test_negative_effect(self):
        result = compute_causal_effect(
            clean_metric=1.0,
            corrupted_metric=0.5,
            patched_metric=0.3,  # Got worse
        )

        assert result["causal_effect"] == -0.2  # 0.3 - 0.5
        assert result["recovery_rate"] < 0

    def test_all_values_returned(self):
        result = compute_causal_effect(
            clean_metric=1.0,
            corrupted_metric=0.5,
            patched_metric=0.75,
        )

        assert "causal_effect" in result
        assert "recovery_rate" in result
        assert "clean_baseline" in result
        assert "corrupted_metric" in result
        assert "patched_metric" in result
        assert "corruption_delta" in result
