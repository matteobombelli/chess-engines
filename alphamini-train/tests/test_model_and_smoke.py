from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from alphamini_train.config import load_config

from conftest import PILOT_CONFIG
from test_schema import write_tensor_cache

torch_available = importlib.util.find_spec("torch") is not None
numpy_available = importlib.util.find_spec("numpy") is not None
onnx_available = importlib.util.find_spec("onnx") is not None
ort_available = importlib.util.find_spec("onnxruntime") is not None
cargo_available = shutil.which("cargo") is not None


@pytest.mark.skipif(not torch_available, reason="PyTorch training extra is not installed")
def test_policy_flatten_is_plane_major() -> None:
    import torch
    from torch import nn

    from alphamini_train.model import AlphaMiniNet

    model = AlphaMiniNet(input_planes=22, channels=22, residual_blocks=1, se_hidden=4)

    class FakePolicy(nn.Module):
        def forward(self, value):
            numbered = torch.arange(73 * 64, dtype=value.dtype, device=value.device)
            return numbered.reshape(1, 73, 8, 8).expand(value.shape[0], -1, -1, -1)

    class FakeValue(nn.Module):
        def forward(self, value):
            return torch.zeros((value.shape[0], 8, 8, 8), dtype=value.dtype, device=value.device)

    model.stem = nn.Identity()
    model.body = nn.Identity()
    model.policy = FakePolicy()
    model.value_conv = FakeValue()
    model.value_bn = nn.Identity()
    policy, _ = model(torch.zeros((1, 22, 8, 8)))
    for plane, origin in ((0, 0), (1, 0), (17, 42), (72, 63)):
        index = plane * 64 + origin
        assert policy[0, index].item() == index


@pytest.mark.skipif(
    not (torch_available and numpy_available and onnx_available and ort_available),
    reason="full train/ONNX extras are not installed",
)
def test_tiny_cpu_train_and_export(tmp_path: Path) -> None:
    from alphamini_train.data import ReplayDataset
    from alphamini_train.export import export_onnx
    from alphamini_train.schema import TensorCache
    from alphamini_train.trainer import Trainer

    config_text = PILOT_CONFIG.read_text().replace('device = "cuda"', 'device = "cpu"')
    config_path = tmp_path / "pilot.toml"
    config_path.write_text(config_text)
    config = load_config(config_path)
    cache = TensorCache(write_tensor_cache(tmp_path / "cache", records=8))
    dataset = ReplayDataset([cache], seed=1, validation_fraction=0.0)
    trainer = Trainer(config, dataset, cycle_id=1)
    digest, checkpoint, metrics = trainer.train(1, checkpoint_dir=tmp_path / "checkpoints")
    assert checkpoint.is_file() and metrics["total_loss"] > 0
    assert metrics["training_session_seconds"] > 0
    assert metrics["training_session_attempts"] == 1
    assert metrics["training_session_successful_updates"] == 1
    assert metrics["training_session_amp_overflows"] == 0
    assert metrics["training_session_samples"] == 8
    assert metrics["training_session_updates_per_second"] > 0
    assert metrics["training_session_samples_per_second"] > 0
    model, manifest_path, manifest = export_onnx(
        trainer.model,
        config,
        tmp_path / "model",
        cycle_id=1,
        global_step=1,
        parent_checkpoint_sha256=digest,
        seed=1,
    )
    assert model.is_file() and manifest_path.is_file()
    assert manifest["schema"] == "model-manifest-v1"
    assert manifest["input_name"] == "input"
    assert manifest["model_sha256"]


@pytest.mark.skipif(
    not (
        torch_available and numpy_available and onnx_available and ort_available and cargo_available
    ),
    reason="PyTorch/ONNX/Rust parity dependencies are not installed",
)
def test_fixed_input_pytorch_python_ort_rust_ort_parity(tmp_path: Path) -> None:
    import torch

    from alphamini_train.export import export_onnx
    from alphamini_train.model import build_model
    from alphamini_train.parity import GOLDEN_PARITY_INPUT_SHA256, verify_cross_runtime_parity

    config_path = tmp_path / "pilot.toml"
    config_path.write_text(PILOT_CONFIG.read_text().replace('device = "cuda"', 'device = "cpu"'))
    config = load_config(config_path)
    torch.manual_seed(747537)
    model = build_model(config).cpu()
    model_path, manifest_path, manifest = export_onnx(
        model,
        config,
        tmp_path / "model",
        cycle_id=0,
        global_step=0,
        parent_checkpoint_sha256="0" * 64,
        seed=747537,
    )

    evidence = verify_cross_runtime_parity(
        model,
        model_path,
        manifest_path,
        config,
        worktree=Path(__file__).resolve().parents[2],
        device="cpu",
        release=False,
    )

    assert evidence["status"] == "passed"
    assert evidence["device"] == "cpu"
    assert evidence["model_sha256"] == manifest["model_sha256"]
    assert evidence["golden_input_sha256"] == GOLDEN_PARITY_INPUT_SHA256
    assert len(evidence["golden_fixture_sha256"]) == 64
    assert len(evidence["rust_stdout_sha256"]) == 64
    assert all(len(digest) == 64 for digest in evidence["output_sha256"].values())
    assert all(comparison["passed"] for comparison in evidence["comparisons"].values())


@pytest.mark.skipif(
    not (torch_available and numpy_available),
    reason="training dependencies are not installed",
)
def test_interrupted_cpu_training_resumes_exactly(tmp_path: Path) -> None:
    import torch

    from alphamini_train.data import ReplayDataset
    from alphamini_train.schema import TensorCache
    from alphamini_train.trainer import Trainer, seed_everything

    config_path = tmp_path / "pilot.toml"
    config_path.write_text(PILOT_CONFIG.read_text().replace('device = "cuda"', 'device = "cpu"'))
    config = load_config(config_path)
    cache = TensorCache(write_tensor_cache(tmp_path / "cache", records=8))
    dataset = ReplayDataset([cache], seed=1, validation_fraction=0.0)

    seed_everything(123)
    uninterrupted = Trainer(config, dataset, cycle_id=1)
    _, _, uninterrupted_metrics = uninterrupted.train(2, checkpoint_dir=tmp_path / "uninterrupted")

    seed_everything(123)
    first_process = Trainer(config, dataset, cycle_id=1)
    _, recovery, _ = first_process.train(1, checkpoint_dir=tmp_path / "first")
    resumed = Trainer(config, dataset, cycle_id=1)
    resumed.resume(recovery)
    _, _, resumed_metrics = resumed.train(2, checkpoint_dir=tmp_path / "resumed")

    assert resumed.state.global_step == uninterrupted.state.global_step == 2
    assert resumed.state.sampler.to_dict() == uninterrupted.state.sampler.to_dict()
    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name]), name
    for name in ("policy_loss", "wdl_loss", "total_loss"):
        assert resumed_metrics[name] == uninterrupted_metrics[name]


@pytest.mark.skipif(
    not torch_available,
    reason="PyTorch training extra is not installed",
)
def test_restore_rng_accepts_checkpoint_tensors_mapped_to_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from alphamini_train.trainer import _restore_rng, _rng_state

    saved = _rng_state()
    saved["torch_cpu"] = saved["torch_cpu"].to("cuda")
    saved["torch_cuda"] = [state.to("cuda") for state in saved["torch_cuda"]]

    _restore_rng(saved)

    assert torch.get_rng_state().device.type == "cpu"
    assert all(state.device.type == "cpu" for state in torch.cuda.get_rng_state_all())


@pytest.mark.skipif(not torch_available, reason="PyTorch training extra is not installed")
def test_determinism_rejects_conflicting_cublas_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alphamini_train.errors import IntegrityError
    from alphamini_train.trainer import configure_determinism

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(IntegrityError, match="conflicts with frozen"):
        configure_determinism(True)
