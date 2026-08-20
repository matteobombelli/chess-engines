"""FP32 ONNX publication and Python-side numerical parity."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json, fsync_directory, sha256_file
from .config import ResolvedConfig
from .errors import DependencyUnavailable, IntegrityError
from .model import require_torch


def export_onnx(
    model: Any,
    config: ResolvedConfig,
    output_dir: Path,
    *,
    cycle_id: int,
    global_step: int,
    parent_checkpoint_sha256: str,
    seed: int,
) -> tuple[Path, Path, dict[str, Any]]:
    torch = require_torch()
    try:
        import onnx  # noqa: F401
    except ImportError as error:
        raise DependencyUnavailable("ONNX export requires the train extra") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").float().eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn((4, 22, 8, 8), generator=generator, dtype=torch.float32)
    with torch.no_grad():
        expected_policy, expected_wdl = model(sample)
    fd, temporary = tempfile.mkstemp(prefix=".model.", suffix=".onnx.partial", dir=output_dir)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.onnx.export(
            model,
            sample[:1],
            temporary_path,
            export_params=True,
            opset_version=int(config.values["export"]["opset"]),
            do_constant_folding=True,
            input_names=["input"],
            output_names=["policy_logits", "wdl_logits"],
            dynamic_axes={
                "input": {0: "batch"},
                "policy_logits": {0: "batch"},
                "wdl_logits": {0: "batch"},
            },
        )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = sha256_file(temporary_path)
        model_path = output_dir / f"model-{digest[:16]}.onnx"
        if model_path.exists():
            if sha256_file(model_path) != digest:
                raise IntegrityError(f"ONNX artifact name collision: {model_path}")
            temporary_path.unlink()
        else:
            os.replace(temporary_path, model_path)
            fsync_directory(output_dir)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()

    parity: dict[str, Any]
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        actual_policy, actual_wdl = session.run(None, {"input": sample.numpy()})
        expected_policy_np = expected_policy.numpy()
        expected_wdl_np = expected_wdl.numpy()
        max_policy = float(np.max(np.abs(expected_policy_np - actual_policy)))
        max_wdl = float(np.max(np.abs(expected_wdl_np - actual_wdl)))
        atol = float(config.values["export"]["parity_atol"])
        rtol = float(config.values["export"]["parity_rtol"])
        passed = bool(
            np.allclose(expected_policy_np, actual_policy, atol=atol, rtol=rtol)
            and np.allclose(expected_wdl_np, actual_wdl, atol=atol, rtol=rtol)
        )
        parity = {
            "status": "passed" if passed else "failed",
            "provider": "CPUExecutionProvider",
            "samples": 4,
            "atol": atol,
            "rtol": rtol,
            "max_abs_policy": max_policy,
            "max_abs_wdl": max_wdl,
        }
        if not passed:
            raise IntegrityError(f"PyTorch/ONNX parity failed: {parity}")
    except ImportError as error:
        raise DependencyUnavailable(
            "ONNX Runtime is required: an unverified model cannot be published"
        ) from error

    # Keep this manifest byte-for-byte compatible with Rust ModelManifestV1,
    # which intentionally denies unknown fields. Training provenance is separate.
    manifest = {
        "schema": "model-manifest-v1",
        "encoder_schema": "encoder-v1",
        "action_schema": "policy-v1",
        "onnx_opset": int(config.values["export"]["opset"]),
        "input_name": "input",
        "policy_output_name": "policy_logits",
        "wdl_output_name": "wdl_logits",
        "input_planes": 22,
        "policy_size": 4672,
        "wdl_size": 3,
        "residual_channels": int(config.values["model"]["channels"]),
        "residual_blocks": int(config.values["model"]["residual_blocks"]),
        "cycle": cycle_id,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "model_sha256": digest,
    }
    manifest_path = output_dir / f"model-{digest[:16]}.json"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        output_dir / f"model-{digest[:16]}.training.json",
        {
            "schema": "alphamini.training-model-provenance.v1",
            "semantic_hash": config.semantic_hash,
            "cycle_id": cycle_id,
            "global_step": global_step,
            "model_sha256": digest,
            "checkpoint_sha256": parent_checkpoint_sha256,
            "architecture": config.values["model"],
            "parity": parity,
        },
    )
    return model_path, manifest_path, manifest
