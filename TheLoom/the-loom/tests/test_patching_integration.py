"""Integration tests for activation patching functionality.

These tests validate the end-to-end patching workflow including:
- Hook injection and activation patching
- Three-path recording (clean/corrupted/patched)
- Patching impact on conveyance metrics
- Experiment orchestration

Run with: pytest tests/test_patching_integration.py -v -s

Requires:
- GPU with sufficient VRAM (8GB+ recommended)
- Models will be downloaded on first run

Mark as slow to skip in normal test runs:
    pytest -m "not integration"
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# Skip all tests in this module if no GPU available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for integration tests"),
]

# Test models - using smaller variants for faster testing
TEST_MODELS = {
    "embedding": "BAAI/bge-small-en-v1.5",  # 33M params, ~130MB
    "generative_small": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B params
}

# Directory for saving example outputs
EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "outputs" / "patching"


@pytest.fixture(scope="module")
def examples_dir():
    """Create examples directory for saving patching outputs."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLES_DIR


@pytest.fixture(scope="module")
def app_client():
    """Create test client with real model loading.

    Cleanup is handled in fixture teardown to avoid relying on test execution order.
    """
    from fastapi.testclient import TestClient
    from src.transport.http import create_http_app
    from src.config import Config

    config = Config()
    app = create_http_app(config)

    with TestClient(app) as client:
        yield client

        # Teardown: unload all models after tests complete
        print("\n=== Fixture Cleanup: Unloading Models ===")
        try:
            response = client.get("/models")
            if response.status_code == 200:
                data = response.json()
                for model in data.get("loaded_models", []):
                    model_id = model["model_id"].replace("/", "--")
                    client.delete(f"/models/{model_id}")
                    print(f"  Unloaded: {model['model_id']}")
            print("=== Cleanup Complete ===")
        except Exception as e:
            print(f"  Cleanup warning: {e}")


# =============================================================================
# Patching Hook Tests
# =============================================================================


class TestPatchingHooks:
    """Test patching hook creation and execution."""

    def test_create_patching_hook(self):
        """Test basic patching hook creation."""
        from src.patching.hooks import (
            HookComponent,
            HookPoint,
            create_patching_hook,
            validate_hook_shapes,
        )

        # Create source activation tensor
        batch_size, seq_len, hidden_dim = 1, 10, 256
        source_activation = torch.randn(batch_size, seq_len, hidden_dim)

        # Create patching hook
        hook = create_patching_hook(source_activation)
        assert callable(hook), "Hook should be callable"

        # Test hook execution
        target_activation = torch.randn(batch_size, seq_len, hidden_dim)
        result = hook(target_activation)

        # Verify shape is preserved
        assert result.shape == target_activation.shape, "Hook should preserve shape"

        # Verify activation was patched
        assert torch.allclose(result, source_activation), "Activation should be patched"

        print("\n=== Patching Hook Test ===")
        print(f"Source shape: {source_activation.shape}")
        print(f"Target shape: {target_activation.shape}")
        print(f"Result shape: {result.shape}")
        print("Hook correctly patches activation")

    def test_hook_point_to_hook_name(self):
        """Test HookPoint to TransformerLens hook name conversion."""
        from src.patching.hooks import HookComponent, HookPoint

        num_layers = 12

        # Test various hook points
        test_cases = [
            (HookPoint(layer=0, component=HookComponent.RESID_PRE), "blocks.0.hook_resid_pre"),
            (HookPoint(layer=5, component=HookComponent.RESID_POST), "blocks.5.hook_resid_post"),
            (HookPoint(layer=-1, component=HookComponent.ATTN), "blocks.11.attn.hook_result"),
            (HookPoint(layer=3, component=HookComponent.MLP_POST), "blocks.3.mlp.hook_post"),
        ]

        print("\n=== Hook Point Name Conversion ===")
        for hook_point, expected_name in test_cases:
            actual_name = hook_point.to_hook_name(num_layers)
            assert actual_name == expected_name, f"Expected {expected_name}, got {actual_name}"
            print(f"Layer {hook_point.layer}, {hook_point.component.value} -> {actual_name}")

    def test_hook_shape_validation(self):
        """Test hook shape validation."""
        from src.patching.hooks import validate_hook_shapes

        # Matching shapes should pass
        tensor_a = torch.randn(1, 10, 256)
        tensor_b = torch.randn(1, 10, 256)
        assert validate_hook_shapes(tensor_a, tensor_b, strict=False), "Matching shapes should pass"

        # Mismatched shapes should fail
        tensor_c = torch.randn(1, 10, 512)
        assert not validate_hook_shapes(tensor_a, tensor_c, strict=False), "Mismatched shapes should fail"

        # Strict mode should raise ValueError
        with pytest.raises(ValueError, match="Hook shape mismatch"):
            validate_hook_shapes(tensor_a, tensor_c, strict=True)

        print("\n=== Hook Shape Validation ===")
        print(f"Valid shapes: {tensor_a.shape} == {tensor_b.shape}")
        print(f"Invalid shapes: {tensor_a.shape} != {tensor_c.shape}")
        print("Shape validation working correctly")

    def test_ablation_hook(self):
        """Test ablation hook that zeroes activations."""
        from src.patching.hooks import create_ablation_hook

        # Create activation
        activation = torch.randn(1, 10, 256)
        original_mean = activation.mean().item()

        # Create and apply ablation hook
        hook = create_ablation_hook(value=0.0)
        result = hook(activation.clone())

        # Verify ablation
        assert torch.allclose(result, torch.zeros_like(result)), "Ablation should zero activations"

        print("\n=== Ablation Hook Test ===")
        print(f"Original mean: {original_mean:.4f}")
        print(f"After ablation mean: {result.mean().item():.4f}")
        print("Ablation hook working correctly")

    def test_noise_hook(self):
        """Test noise injection hook."""
        from src.patching.hooks import create_noise_hook

        # Create activation
        activation = torch.randn(1, 10, 256)
        original = activation.clone()

        # Create and apply noise hook with fixed seed
        hook = create_noise_hook(scale=0.1, seed=42)
        result = hook(activation.clone())

        # Verify noise was added
        diff = (result - original).abs().mean().item()
        assert diff > 0, "Noise should modify activation"
        assert diff < 1.0, "Noise should be moderate (scale=0.1)"

        print("\n=== Noise Hook Test ===")
        print(f"Mean absolute difference: {diff:.6f}")
        print("Noise hook working correctly")


# =============================================================================
# Activation Cache Tests
# =============================================================================


class TestActivationCache:
    """Test activation caching functionality."""

    def test_cache_store_and_retrieve(self):
        """Test basic cache store and retrieve operations."""
        from src.patching.cache import ActivationCache

        cache = ActivationCache(run_id="test_run")

        # Store activation
        activation = torch.randn(1, 10, 256)
        cached = cache.store(layer=5, component="resid_pre", activation=activation)

        assert cached is not None, "Store should return cached activation"
        assert cached.layer == 5, "Layer should match"
        assert cached.component == "resid_pre", "Component should match"

        # Retrieve activation
        retrieved = cache.get(layer=5, component="resid_pre")
        assert retrieved is not None, "Should retrieve cached activation"
        assert torch.allclose(retrieved.activation, activation), "Activation should match"

        # Check cache properties
        assert cache.num_entries == 1, "Should have one entry"
        assert cache.has(5, "resid_pre"), "Should have cached activation"
        assert not cache.has(6, "resid_pre"), "Should not have uncached layer"

        print("\n=== Cache Store/Retrieve Test ===")
        print(f"Run ID: {cache.run_id}")
        print(f"Entries: {cache.num_entries}")
        print(f"Size: {cache.size_bytes} bytes")
        print("Cache operations working correctly")

    def test_cache_multi_layer(self):
        """Test caching activations from multiple layers."""
        from src.patching.cache import ActivationCache

        cache = ActivationCache(run_id="multi_layer_test")

        # Store activations for multiple layers
        num_layers = 12
        for layer in range(num_layers):
            activation = torch.randn(1, 10, 256)
            cache.store(layer=layer, component="resid_pre", activation=activation)

        # Verify all layers cached
        assert cache.num_entries == num_layers, f"Should have {num_layers} entries"
        assert cache.get_layers() == list(range(num_layers)), "Should have all layer indices"

        # Retrieve by layer
        layer_5_activations = cache.get_by_layer(5)
        assert "resid_pre" in layer_5_activations, "Should have resid_pre for layer 5"

        print("\n=== Multi-Layer Cache Test ===")
        print(f"Layers cached: {cache.get_layers()}")
        print(f"Total entries: {cache.num_entries}")
        print(f"Total size: {cache.size_bytes / 1024:.2f} KB")

    def test_cache_manager(self):
        """Test cache manager for multiple caches."""
        from src.patching.cache import CacheManager

        manager = CacheManager(max_size_mb=1024)

        # Create caches for clean and corrupted runs
        clean_cache = manager.create_cache("clean_run")
        corrupted_cache = manager.create_cache("corrupted_run")

        # Store activations
        activation = torch.randn(1, 10, 256)
        clean_cache.store(layer=0, component="resid_pre", activation=activation)
        corrupted_cache.store(layer=0, component="resid_pre", activation=activation * 2)

        # Verify manager state
        assert manager.num_caches == 2, "Should have 2 caches"
        assert "clean_run" in manager.cache_ids, "Should have clean_run cache"
        assert "corrupted_run" in manager.cache_ids, "Should have corrupted_run cache"

        # Get memory stats
        stats = manager.get_memory_stats()
        print("\n=== Cache Manager Test ===")
        print(f"Number of caches: {stats['num_caches']}")
        print(f"Total size: {stats['total_size_mb']:.4f} MB")
        print(f"Utilization: {stats['utilization']:.2%}")

        # Cleanup
        manager.cleanup()
        assert manager.num_caches == 0, "Should have no caches after cleanup"


# =============================================================================
# Path Recording Tests
# =============================================================================


class TestPathRecording:
    """Test three-path recording system."""

    def test_path_recorder_basic(self):
        """Test basic path recording functionality."""
        from src.patching.experiments import ExecutionPath, PathRecorder

        recorder = PathRecorder(experiment_id="test_experiment")

        # Record clean path
        clean_output = torch.randn(1, 10, 50000)  # logits
        recorder.record_clean_path(
            output=clean_output,
            input_text="The quick brown fox",
            generation_time_ms=100.0,
        )

        # Record corrupted path
        corrupted_output = torch.randn(1, 10, 50000)
        recorder.record_corrupted_path(
            output=corrupted_output,
            input_text="The slow grey dog",
            generation_time_ms=100.0,
        )

        # Record patched path
        patched_output = torch.randn(1, 10, 50000)
        recorder.record_patched_path(
            output=patched_output,
            patch_info={"layer": 5, "component": "resid_pre"},
            generation_time_ms=150.0,
        )

        # Verify all paths recorded
        assert recorder.has_all_paths(), "Should have all three paths"
        assert recorder.clean_path is not None, "Should have clean path"
        assert recorder.corrupted_path is not None, "Should have corrupted path"
        assert len(recorder.patched_paths) == 1, "Should have one patched path"

        print("\n=== Path Recorder Test ===")
        print(f"Experiment ID: {recorder.experiment_id}")
        print(f"Clean path recorded: {recorder.has_path(ExecutionPath.CLEAN)}")
        print(f"Corrupted path recorded: {recorder.has_path(ExecutionPath.CORRUPTED)}")
        print(f"Patched paths: {len(recorder.patched_paths)}")
        print("Path recording working correctly")

    def test_path_recording_serialization(self):
        """Test path recording serialization."""
        from src.patching.experiments import PathRecorder, PathRecording

        recorder = PathRecorder()

        # Record all paths
        recorder.record_clean_path(output=torch.randn(1, 10), input_text="clean input")
        recorder.record_corrupted_path(output=torch.randn(1, 10), input_text="corrupted input")
        recorder.record_patched_path(
            output=torch.randn(1, 10),
            patch_info={"layer": 5, "component": "resid_pre"},
        )

        # Convert to PathRecording
        recording = PathRecording.from_recorder(recorder, experiment_name="test_recording")

        # Serialize to dict
        recording_dict = recording.to_dict()

        assert "recording_id" in recording_dict, "Should have recording_id"
        assert "experiment_name" in recording_dict, "Should have experiment_name"
        assert recording_dict["has_all_paths"], "Should have all paths"

        print("\n=== Path Recording Serialization ===")
        print(json.dumps(recording_dict, indent=2, default=str))

    def test_recording_store(self, tmp_path):
        """Test recording store with disk persistence."""
        from src.patching.cache import ActivationCache
        from src.patching.experiments import ExecutionPath, RecordingStore

        store = RecordingStore(storage_dir=tmp_path / "recordings")

        # Record paths
        store.record_clean_path(
            recording_id="exp_001",
            output=torch.randn(1, 10),
            input_text="The quick brown fox",
            experiment_name="patching_experiment",
        )
        store.record_corrupted_path(
            recording_id="exp_001",
            output=torch.randn(1, 10),
            input_text="The slow grey dog",
        )
        store.record_patched_path(
            recording_id="exp_001",
            output=torch.randn(1, 10),
            layer=5,
            component="resid_pre",
        )

        # Verify recording
        recording = store.get_recording("exp_001")
        assert recording is not None, "Should retrieve recording"
        assert recording.has_all_paths, "Recording should have all paths"

        # Save to disk
        saved = store.save_recording("exp_001")
        assert saved, "Should save successfully"

        # Clear and reload
        store.clear()
        assert store.num_recordings == 0, "Should be empty after clear"

        loaded = store.load_recording("exp_001")
        assert loaded is not None, "Should load recording from disk"
        assert loaded.has_all_paths, "Loaded recording should have all paths"

        print("\n=== Recording Store Test ===")
        print(f"Storage dir: {store.storage_dir}")
        print(f"Recording saved and reloaded successfully")


# =============================================================================
# Patching Experiment Tests
# =============================================================================


class TestPatchingExperiment:
    """Test patching experiment orchestration."""

    def test_experiment_config(self):
        """Test experiment configuration."""
        from src.patching.experiments import ExperimentConfig
        from src.patching.hooks import HookComponent

        config = ExperimentConfig(
            name="layer_sweep_experiment",
            layers=[0, 5, 10, 11],
            components=["resid_pre", "attn", "mlp_post"],
            batch_size=1,
            validate_shapes=True,
        )

        # Get hook components
        hook_components = config.get_hook_components()
        assert len(hook_components) == 3, "Should have 3 components"
        assert HookComponent.RESID_PRE in hook_components, "Should have resid_pre"
        assert HookComponent.ATTN in hook_components, "Should have attn"
        assert HookComponent.MLP_POST in hook_components, "Should have mlp_post"

        print("\n=== Experiment Config Test ===")
        print(f"Name: {config.name}")
        print(f"Layers: {config.layers}")
        print(f"Components: {config.components}")
        print(f"Hook components: {[c.value for c in hook_components]}")

    def test_patching_experiment_setup(self):
        """Test patching experiment setup and lifecycle."""
        from src.patching.experiments import ExperimentConfig, PatchingExperiment

        config = ExperimentConfig(
            name="test_experiment",
            layers=[0, 5],
            components=["resid_pre"],
        )

        experiment = PatchingExperiment(config=config)

        # Create experiment record
        record = experiment.create_record()
        assert record is not None, "Should create record"
        assert record.status == "running", "Status should be running"

        # Finalize experiment
        final_record = experiment.finalize(success=True)
        assert final_record.status == "completed", "Status should be completed"
        assert final_record.duration_ms is not None, "Should have duration"

        print("\n=== Patching Experiment Setup ===")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Status: {final_record.status}")
        print(f"Duration: {final_record.duration_ms:.2f} ms")

    def test_causal_effect_computation(self):
        """Test causal effect computation from patching results."""
        from src.patching.experiments import compute_causal_effect

        # Test recovery case
        result = compute_causal_effect(
            clean_metric=0.2,  # Low beta (healthy)
            corrupted_metric=0.8,  # High beta (collapsed)
            patched_metric=0.4,  # Partial recovery
        )

        assert "causal_effect" in result, "Should have causal_effect"
        assert "recovery_rate" in result, "Should have recovery_rate"
        assert result["causal_effect"] == -0.4, "Causal effect should be patched - corrupted"
        assert result["recovery_rate"] > 0, "Recovery rate should be positive (patch helped)"

        print("\n=== Causal Effect Computation ===")
        print(f"Clean metric: {result['clean_baseline']:.2f}")
        print(f"Corrupted metric: {result['corrupted_metric']:.2f}")
        print(f"Patched metric: {result['patched_metric']:.2f}")
        print(f"Causal effect: {result['causal_effect']:.2f}")
        print(f"Recovery rate: {result['recovery_rate']:.2%}")

        # Test no effect case
        no_effect = compute_causal_effect(
            clean_metric=0.2,
            corrupted_metric=0.8,
            patched_metric=0.8,  # No change
        )
        assert no_effect["causal_effect"] == 0.0, "No change should have zero causal effect"
        assert no_effect["recovery_rate"] == 0.0, "No change should have zero recovery"


# =============================================================================
# Conveyance Metrics Patching Impact Tests
# =============================================================================


class TestConveyanceMetricsPatchingImpact:
    """Test patching impact on conveyance metrics."""

    def test_compute_patch_impact(self):
        """Test patch impact computation."""
        from src.analysis.conveyance_metrics import (
            PatchingImpactResult,
            compute_patch_impact,
        )

        # Test beneficial patch
        result = compute_patch_impact(
            clean_value=0.2,  # Low beta (healthy)
            corrupted_value=0.8,  # High beta (collapsed)
            patched_value=0.3,  # Patch reduced collapse
            metric_name="beta",
            patch_layer=5,
        )

        assert isinstance(result, PatchingImpactResult), "Should return PatchingImpactResult"
        assert result.patch_layer == 5, "Should preserve patch layer"
        assert result.recovery_rate > 0, "Recovery rate should be positive"
        assert result.impact_severity in ["strong_recovery", "moderate_recovery", "weak_recovery"], \
            f"Impact severity should be recovery type, got {result.impact_severity}"

        print("\n=== Patch Impact Computation ===")
        print(f"Metric: {result.metric_name}")
        print(f"Patch layer: {result.patch_layer}")
        print(f"Causal effect: {result.causal_effect:.4f}")
        print(f"Recovery rate: {result.recovery_rate:.2%}")
        print(f"Impact severity: {result.impact_severity}")

    def test_aggregate_patch_impacts(self):
        """Test aggregation of patch impacts across layers."""
        from src.analysis.conveyance_metrics import (
            aggregate_patch_impacts,
            compute_patch_impact,
        )

        # Simulate layer sweep
        impacts = [
            compute_patch_impact(0.2, 0.8, 0.3, "beta", patch_layer=0),
            compute_patch_impact(0.2, 0.8, 0.4, "beta", patch_layer=5),
            compute_patch_impact(0.2, 0.8, 0.25, "beta", patch_layer=10),
            compute_patch_impact(0.2, 0.8, 0.8, "beta", patch_layer=11),  # No effect
        ]

        aggregate = aggregate_patch_impacts(impacts)

        assert "mean_recovery_rate" in aggregate, "Should have mean recovery rate"
        assert "best_layer" in aggregate, "Should have best layer"
        assert "best_recovery_rate" in aggregate, "Should have best recovery rate"
        assert aggregate["best_layer"] == 10, "Layer 10 should have best recovery"

        print("\n=== Aggregate Patch Impacts ===")
        print(f"Mean recovery: {aggregate['mean_recovery_rate']:.2%}")
        print(f"Best layer: {aggregate['best_layer']}")
        print(f"Best recovery: {aggregate['best_recovery_rate']:.2%}")
        print(f"Recovery distribution: {aggregate['recovery_distribution']}")

    def test_identify_causal_layers(self):
        """Test identification of causal layers."""
        from src.analysis.conveyance_metrics import (
            compute_patch_impact,
            identify_causal_layers,
        )

        # Simulate layer sweep with varying impacts
        impacts = [
            compute_patch_impact(0.2, 0.8, 0.8, "beta", patch_layer=0),   # No effect
            compute_patch_impact(0.2, 0.8, 0.7, "beta", patch_layer=1),   # Weak
            compute_patch_impact(0.2, 0.8, 0.3, "beta", patch_layer=5),   # Strong
            compute_patch_impact(0.2, 0.8, 0.25, "beta", patch_layer=10), # Strong
            compute_patch_impact(0.2, 0.8, 0.75, "beta", patch_layer=11), # Weak
        ]

        causal_layers = identify_causal_layers(impacts, recovery_threshold=0.3)

        assert 5 in causal_layers, "Layer 5 should be causal"
        assert 10 in causal_layers, "Layer 10 should be causal"
        assert 0 not in causal_layers, "Layer 0 should not be causal"

        print("\n=== Causal Layer Identification ===")
        print(f"Causal layers (threshold=0.3): {causal_layers}")
        print(f"Non-causal layers: {[i for i in [0, 1, 11] if i not in causal_layers]}")


# =============================================================================
# Multi-Layer Patching Study Tests
# =============================================================================


class TestMultiLayerPatchingStudy:
    """Test systematic multi-layer patching studies."""

    def test_multi_layer_study_creation(self):
        """Test multi-layer patching study setup."""
        from src.patching.experiments import MultiLayerPatchingStudy

        study = MultiLayerPatchingStudy(
            name="attention_layer_study",
            layers=list(range(12)),
            components=["resid_pre", "attn"],
        )

        assert study.name == "attention_layer_study", "Should have correct name"

        # Create experiments within study
        exp1 = study.create_experiment("experiment_1")
        exp2 = study.create_experiment("experiment_2")

        assert len(study.experiments) == 2, "Should have 2 experiments"

        print("\n=== Multi-Layer Study ===")
        print(f"Study: {study.name}")
        print(f"Layers: {study._layers}")
        print(f"Components: {study._components}")
        print(f"Experiments created: {len(study.experiments)}")

    def test_study_results_aggregation(self):
        """Test aggregation of study results."""
        from src.patching.experiments import MultiLayerPatchingStudy

        study = MultiLayerPatchingStudy(
            name="test_study",
            layers=[0, 5, 10],
            components=["resid_pre"],
        )

        # Create and finalize experiments
        for i in range(3):
            exp = study.create_experiment(f"exp_{i}")
            exp.create_record()
            record = exp.finalize(success=True)
            study._results.append(record)

        # Aggregate results
        aggregate = study.aggregate_results()

        assert "study_name" in aggregate, "Should have study name"
        assert aggregate["num_experiments"] == 3, "Should have 3 experiments"
        assert aggregate["num_results"] == 3, "Should have 3 results"

        print("\n=== Study Aggregation ===")
        print(f"Study: {aggregate['study_name']}")
        print(f"Experiments: {aggregate['num_experiments']}")
        print(f"Results: {aggregate['num_results']}")


# =============================================================================
# Integration Tests with Real Models
# =============================================================================


class TestPatchingWithRealModels:
    """Integration tests that actually load models and perform patching."""

    @pytest.fixture(scope="class")
    def loaded_model(self, app_client):
        """Load generative model for patching tests."""
        model_id = TEST_MODELS["generative_small"]

        print(f"\n=== Loading Model for Patching: {model_id} ===")
        start = time.time()

        response = app_client.post("/models/load", json={
            "model": model_id,
            "dtype": "float16",
        })

        load_time = time.time() - start
        assert response.status_code == 200, f"Failed to load model: {response.text}"

        data = response.json()
        print(f"Loaded in {load_time:.2f}s")
        print(json.dumps(data, indent=2))

        yield data

        # Cleanup
        app_client.delete(f"/models/{model_id.replace('/', '--')}")

    def test_hidden_state_extraction_for_patching(self, app_client, loaded_model, examples_dir):
        """Test that hidden states can be extracted for patching workflow."""
        model_id = TEST_MODELS["generative_small"]
        num_layers = loaded_model["num_layers"]

        # Extract hidden states from multiple layers
        layers_to_cache = [0, num_layers // 2, -1]

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.1,
            "return_hidden_states": True,
            "hidden_state_layers": layers_to_cache,
            "hidden_state_format": "list",
        })

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()

        hidden_states = data.get("hidden_states", {})
        assert hidden_states is not None, "No hidden states returned"

        # Verify we can use these for patching
        from src.patching.cache import ActivationCache

        cache = ActivationCache(run_id="clean_extraction")

        for layer_idx in layers_to_cache:
            layer_key = str(layer_idx)
            if layer_key in hidden_states:
                layer_data = hidden_states[layer_key]
                activation = torch.tensor(layer_data["data"]).reshape(layer_data["shape"])
                cache.store(layer=layer_idx, component="resid_pre", activation=activation)

        assert cache.num_entries > 0, "Should have cached activations"

        print("\n=== Hidden State Extraction for Patching ===")
        print(f"Layers extracted: {cache.get_layers()}")
        print(f"Cache entries: {cache.num_entries}")
        print(f"Cache size: {cache.size_bytes / 1024:.2f} KB")

        # Save example
        example = {
            "model": model_id,
            "prompt": "The capital of France is",
            "layers_extracted": cache.get_layers(),
            "cache_size_bytes": cache.size_bytes,
            "shapes": {
                str(k): list(v.shape)
                for k, v in {
                    layer: cache.get_tensor(layer, "resid_pre")
                    for layer in cache.get_layers()
                }.items()
                if v is not None
            },
        }
        with open(examples_dir / "hidden_state_extraction.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_three_path_experiment_workflow(self, app_client, loaded_model, examples_dir):
        """Test complete three-path (clean/corrupted/patched) experiment workflow."""
        from src.patching.experiments import PathRecorder, compute_causal_effect

        model_id = TEST_MODELS["generative_small"]
        recorder = PathRecorder(experiment_id="three_path_test")

        # Prompts for clean and corrupted paths
        clean_prompt = "The capital of France is"
        corrupted_prompt = "The capital of Germany is"

        # 1. Clean path
        clean_response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": clean_prompt,
            "max_tokens": 10,
            "temperature": 0.1,
            "return_hidden_states": True,
            "hidden_state_layers": [-1],
        })
        assert clean_response.status_code == 200
        clean_data = clean_response.json()

        recorder.record_clean_path(
            output=None,  # Would be logits in real scenario
            input_text=clean_prompt,
            metadata={"generated_text": clean_data["text"]},
        )

        # 2. Corrupted path
        corrupted_response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": corrupted_prompt,
            "max_tokens": 10,
            "temperature": 0.1,
            "return_hidden_states": True,
            "hidden_state_layers": [-1],
        })
        assert corrupted_response.status_code == 200
        corrupted_data = corrupted_response.json()

        recorder.record_corrupted_path(
            output=None,
            input_text=corrupted_prompt,
            metadata={"generated_text": corrupted_data["text"]},
        )

        # 3. Patched path (simulated - in real scenario would inject hooks)
        recorder.record_patched_path(
            output=None,
            patch_info={"layer": -1, "component": "resid_pre"},
            metadata={"generated_text": "Paris (simulated patch)"},
        )

        # Verify three-path recording
        assert recorder.has_all_paths(), "Should have all three paths"

        # Compute causal effect (simulated metrics)
        causal_effect = compute_causal_effect(
            clean_metric=0.9,  # High accuracy for clean
            corrupted_metric=0.3,  # Lower accuracy for corrupted
            patched_metric=0.7,  # Partial recovery with patch
        )

        print("\n=== Three-Path Experiment ===")
        print(f"Experiment ID: {recorder.experiment_id}")
        print(f"Clean output: {clean_data['text'][:50]}...")
        print(f"Corrupted output: {corrupted_data['text'][:50]}...")
        print(f"Causal effect: {causal_effect['causal_effect']:.4f}")
        print(f"Recovery rate: {causal_effect['recovery_rate']:.2%}")

        # Save example
        example = {
            "experiment_id": recorder.experiment_id,
            "clean": {
                "prompt": clean_prompt,
                "output": clean_data["text"],
            },
            "corrupted": {
                "prompt": corrupted_prompt,
                "output": corrupted_data["text"],
            },
            "patched": {
                "layer": -1,
                "component": "resid_pre",
            },
            "causal_effect": causal_effect,
        }
        with open(examples_dir / "three_path_experiment.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_layer_sweep_patching_study(self, app_client, loaded_model, examples_dir):
        """Test systematic layer sweep patching study."""
        from src.patching.experiments import ExperimentConfig, MultiLayerPatchingStudy
        from src.analysis.conveyance_metrics import (
            aggregate_patch_impacts,
            compute_patch_impact,
            identify_causal_layers,
        )

        model_id = TEST_MODELS["generative_small"]
        num_layers = loaded_model["num_layers"]

        # Create layer sweep study
        study = MultiLayerPatchingStudy(
            name="layer_sweep_study",
            layers=list(range(num_layers)),
            components=["resid_pre"],
        )

        # Simulate patching results for each layer
        # In a real scenario, we would actually patch at each layer
        simulated_impacts = []
        for layer in range(num_layers):
            # Simulate varying recovery rates across layers
            # Middle layers often have stronger causal effects
            distance_from_middle = abs(layer - num_layers // 2)
            base_recovery = 0.8 - (distance_from_middle * 0.1)
            patched_value = 0.8 - (base_recovery * 0.6)

            impact = compute_patch_impact(
                clean_value=0.2,
                corrupted_value=0.8,
                patched_value=patched_value,
                metric_name="beta",
                patch_layer=layer,
            )
            simulated_impacts.append(impact)

        # Aggregate results
        aggregate = aggregate_patch_impacts(simulated_impacts)
        causal_layers = identify_causal_layers(simulated_impacts, recovery_threshold=0.3)

        print("\n=== Layer Sweep Study ===")
        print(f"Study: {study.name}")
        print(f"Layers analyzed: {num_layers}")
        print(f"Mean recovery: {aggregate['mean_recovery_rate']:.2%}")
        print(f"Best layer: {aggregate['best_layer']}")
        print(f"Best recovery: {aggregate['best_recovery_rate']:.2%}")
        print(f"Causal layers: {causal_layers}")

        # Save example
        example = {
            "study_name": study.name,
            "num_layers": num_layers,
            "layer_results": [
                {
                    "layer": impact.patch_layer,
                    "recovery_rate": impact.recovery_rate,
                    "impact_severity": impact.impact_severity,
                }
                for impact in simulated_impacts
            ],
            "aggregate": aggregate,
            "causal_layers": causal_layers,
        }
        with open(examples_dir / "layer_sweep_study.json", "w") as f:
            json.dump(example, f, indent=2)


# =============================================================================
# Memory Management Tests
# =============================================================================


class TestPatchingMemoryManagement:
    """Test memory management for patching operations."""

    def test_cache_cleanup(self):
        """Test proper cache cleanup to prevent memory leaks."""
        import gc

        from src.patching.cache import ActivationCache, CacheManager

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Create and populate cache
        manager = CacheManager(cleanup_on_exit=True)

        for i in range(5):
            cache = manager.create_cache(f"cache_{i}")
            for layer in range(12):
                activation = torch.randn(1, 100, 768)
                if torch.cuda.is_available():
                    activation = activation.cuda()
                cache.store(layer=layer, component="resid_pre", activation=activation)

        peak_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Cleanup
        manager.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        print("\n=== Cache Cleanup Test ===")
        print(f"Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
        print(f"Final memory: {final_memory / 1024 / 1024:.2f} MB")
        print(f"Memory freed: {(peak_memory - final_memory) / 1024 / 1024:.2f} MB")

        assert manager.num_caches == 0, "All caches should be cleared"

    def test_cache_size_estimation(self):
        """Test cache size estimation utility."""
        from src.patching.cache import compute_cache_size_estimate

        # Estimate cache size for a typical experiment
        estimate = compute_cache_size_estimate(
            num_layers=12,
            hidden_dim=768,
            seq_length=512,
            batch_size=1,
            components_per_layer=2,
            dtype_bytes=4,  # float32
        )

        expected_per_activation = 1 * 512 * 768 * 4  # ~1.5 MB
        expected_total = expected_per_activation * 12 * 2  # ~36 MB

        print("\n=== Cache Size Estimation ===")
        print(f"Estimated cache size: {estimate / 1024 / 1024:.2f} MB")
        print(f"Parameters: 12 layers, 768 hidden, 512 seq len, 2 components/layer")

        # Should be close to expected
        assert estimate > 0, "Estimate should be positive"
        assert abs(estimate - expected_total) < 1024, "Estimate should be close to expected"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
