"""Three-runtime numerical parity over Rust's fixed encoded-position fixture."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .atomic import SHA256_RE, canonical_json_bytes, read_json, sha256_bytes
from .config import ResolvedConfig
from .errors import DependencyUnavailable, IntegrityError
from .model import require_torch

RUST_PARITY_SCHEMA = "alphamini-inference-parity-v1"
GOLDEN_PARITY_FEN = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 5 3"
GOLDEN_PARITY_INPUT_SHA256 = "a3c8eb105e9af08a4bb13315141f289af83f1ebfc9059ca6c19070a6f6976d7a"
RUST_PARITY_FIELDS = {
    "schema",
    "device",
    "cuda_device",
    "model_sha256",
    "encoder_schema",
    "action_schema",
    "fen",
    "input_shape",
    "input_sha256",
    "input_values",
    "policy_shape",
    "policy_logits",
    "wdl_shape",
    "wdl_logits",
}


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:
        raise DependencyUnavailable(
            "cross-runtime parity requires NumPy and ONNX Runtime"
        ) from error
    return np, ort


def _f32_sha256(np: Any, *arrays: Any) -> str:
    payload = b"".join(np.asarray(array, dtype="<f4").tobytes(order="C") for array in arrays)
    return sha256_bytes(payload)


def _finite_floats(value: Any, count: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise IntegrityError(f"Rust parity {field} must contain {count} values")
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise IntegrityError(f"Rust parity {field} contains a non-number")
        number = float(item)
        if not math.isfinite(number):
            raise IntegrityError(f"Rust parity {field} contains a non-finite value")
        result.append(number)
    return result


def _rust_command(
    model_path: Path,
    manifest_path: Path,
    *,
    device: str,
    cuda_device: int,
    release: bool,
) -> list[str]:
    if device not in {"cpu", "cuda"}:
        raise IntegrityError("parity device must be cpu or cuda")
    command = ["cargo", "run", "--quiet", "--locked"]
    if release:
        command.append("--release")
    command.extend(
        [
            "-p",
            "alphamini",
            "--bin",
            "alphamini-inference",
            "--features",
            "cuda" if device == "cuda" else "onnx",
            "--",
            "--model",
            str(model_path),
            "--manifest",
            str(manifest_path),
            "--device",
            device,
        ]
    )
    if device == "cuda":
        command.extend(["--cuda-device", str(cuda_device)])
    return command


def _load_rust_output(
    stdout: str,
    *,
    model_sha256: str,
    device: str,
    cuda_device: int,
) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise IntegrityError(f"Rust parity emitted invalid JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != RUST_PARITY_FIELDS:
        raise IntegrityError("Rust parity output fields disagree with its v1 schema")
    if value.get("schema") != RUST_PARITY_SCHEMA:
        raise IntegrityError("unsupported Rust inference parity schema")
    if value.get("device") != device:
        raise IntegrityError("Rust parity used the wrong inference device")
    expected_cuda = cuda_device if device == "cuda" else None
    if value.get("cuda_device") != expected_cuda:
        raise IntegrityError("Rust parity used the wrong CUDA device")
    if value.get("model_sha256") != model_sha256:
        raise IntegrityError("Rust parity loaded a different ONNX model")
    if value.get("encoder_schema") != "encoder-v1" or value.get("action_schema") != "policy-v1":
        raise IntegrityError("Rust parity used an incompatible encoder/action schema")
    if value.get("fen") != GOLDEN_PARITY_FEN:
        raise IntegrityError("Rust parity fixture FEN drifted from the frozen golden position")
    if value.get("input_shape") != [1, 22, 8, 8]:
        raise IntegrityError("Rust parity input shape is not [1,22,8,8]")
    if value.get("policy_shape") != [1, 73, 8, 8]:
        raise IntegrityError("Rust parity policy shape is not [1,73,8,8]")
    if value.get("wdl_shape") != [1, 3]:
        raise IntegrityError("Rust parity WDL shape is not [1,3]")
    if not isinstance(value.get("input_sha256"), str) or not SHA256_RE.fullmatch(
        value["input_sha256"]
    ):
        raise IntegrityError("Rust parity input_sha256 is invalid")
    if value["input_sha256"] != GOLDEN_PARITY_INPUT_SHA256:
        raise IntegrityError("Rust parity encoded input drifted from the frozen golden digest")
    value["input_values"] = _finite_floats(value.get("input_values"), 22 * 8 * 8, "input")
    value["policy_logits"] = _finite_floats(value.get("policy_logits"), 73 * 8 * 8, "policy_logits")
    value["wdl_logits"] = _finite_floats(value.get("wdl_logits"), 3, "wdl_logits")
    return value


def verify_cross_runtime_parity(
    model: Any,
    model_path: Path,
    manifest_path: Path,
    config: ResolvedConfig,
    *,
    worktree: Path,
    device: str,
    cuda_device: int = 0,
    release: bool = True,
    timeout_seconds: int = 900,
    rust_environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compare PyTorch, Python ORT CPU, and Rust ORT on one Rust-encoded input."""

    torch = require_torch()
    np, ort = _dependencies()
    model_manifest = read_json(manifest_path)
    model_sha256 = model_manifest.get("model_sha256")
    if not isinstance(model_sha256, str) or not SHA256_RE.fullmatch(model_sha256):
        raise IntegrityError("model manifest lacks a valid model_sha256")
    command = _rust_command(
        model_path,
        manifest_path,
        device=device,
        cuda_device=cuda_device,
        release=release,
    )
    if device == "cuda" and rust_environment is None:
        # Verify and scope CUDA 13 only to Rust. PyTorch and Python ORT retain
        # their locked cu126/CPU environment in this process.
        from .cuda_runtime import runtime_environment

        rust_environment, _ = runtime_environment(worktree)
    elif rust_environment is not None:
        rust_environment = dict(rust_environment)
    if rust_environment is not None and not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in rust_environment.items()
    ):
        raise IntegrityError("Rust parity environment must contain only strings")
    try:
        completed = subprocess.run(
            command,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=rust_environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise IntegrityError(f"could not execute Rust parity fixture: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise IntegrityError(f"Rust parity fixture exited {completed.returncode}: {detail}")
    rust = _load_rust_output(
        completed.stdout,
        model_sha256=model_sha256,
        device=device,
        cuda_device=cuda_device,
    )
    inputs = np.asarray(rust["input_values"], dtype=np.float32).reshape(1, 22, 8, 8)
    if not bool(np.isfinite(inputs).all()) or bool((inputs < 0).any()) or bool((inputs > 1).any()):
        raise IntegrityError("Rust parity input is outside the frozen [0,1] tensor contract")
    input_sha256 = _f32_sha256(np, inputs)
    if input_sha256 != rust["input_sha256"]:
        raise IntegrityError("Rust parity input values disagree with input_sha256")

    model = model.to("cpu").float().eval()
    with torch.no_grad():
        torch_policy, torch_wdl = model(torch.from_numpy(inputs.copy()))
    torch_policy = torch_policy.detach().cpu().numpy().astype(np.float32, copy=False)
    torch_wdl = torch_wdl.detach().cpu().numpy().astype(np.float32, copy=False)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_policy, ort_wdl = session.run(None, {"input": inputs})
    ort_policy = np.asarray(ort_policy, dtype=np.float32)
    ort_wdl = np.asarray(ort_wdl, dtype=np.float32)
    rust_policy = np.asarray(rust["policy_logits"], dtype=np.float32).reshape(1, 4672)
    rust_wdl = np.asarray(rust["wdl_logits"], dtype=np.float32).reshape(1, 3)
    expected_shapes = ((1, 4672), (1, 3))
    for name, policy, wdl in (
        ("PyTorch", torch_policy, torch_wdl),
        ("Python ORT", ort_policy, ort_wdl),
        ("Rust ORT", rust_policy, rust_wdl),
    ):
        if policy.shape != expected_shapes[0] or wdl.shape != expected_shapes[1]:
            raise IntegrityError(f"{name} returned wrong parity output shapes")
        if not bool(np.isfinite(policy).all()) or not bool(np.isfinite(wdl).all()):
            raise IntegrityError(f"{name} returned non-finite parity logits")

    atol = float(config.values["export"]["parity_atol"])
    rtol = float(config.values["export"]["parity_rtol"])
    comparisons: dict[str, dict[str, float | bool]] = {}
    for name, left_policy, left_wdl, right_policy, right_wdl in (
        ("pytorch_vs_python_ort", torch_policy, torch_wdl, ort_policy, ort_wdl),
        ("pytorch_vs_rust_ort", torch_policy, torch_wdl, rust_policy, rust_wdl),
        ("python_ort_vs_rust_ort", ort_policy, ort_wdl, rust_policy, rust_wdl),
    ):
        policy_close = bool(np.allclose(left_policy, right_policy, atol=atol, rtol=rtol))
        wdl_close = bool(np.allclose(left_wdl, right_wdl, atol=atol, rtol=rtol))
        comparisons[name] = {
            "passed": policy_close and wdl_close,
            "max_abs_policy": float(np.max(np.abs(left_policy - right_policy))),
            "max_abs_wdl": float(np.max(np.abs(left_wdl - right_wdl))),
        }
    if not all(bool(value["passed"]) for value in comparisons.values()):
        raise IntegrityError(f"cross-runtime inference parity failed: {comparisons}")

    fixture_identity = {
        "schema": RUST_PARITY_SCHEMA,
        "fen": rust["fen"],
        "input_shape": rust["input_shape"],
        "input_sha256": input_sha256,
        "encoder_schema": rust["encoder_schema"],
        "action_schema": rust["action_schema"],
    }
    return {
        "schema": "alphamini.cross-runtime-parity-evidence.v1",
        "status": "passed",
        "device": device,
        "cuda_device": cuda_device if device == "cuda" else None,
        "python_ort_provider": "CPUExecutionProvider",
        "model_sha256": model_sha256,
        "atol": atol,
        "rtol": rtol,
        "golden_fen": rust["fen"],
        "golden_input_sha256": input_sha256,
        "golden_fixture_sha256": sha256_bytes(canonical_json_bytes(fixture_identity)),
        "rust_stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "output_sha256": {
            "pytorch": _f32_sha256(np, torch_policy, torch_wdl),
            "python_ort": _f32_sha256(np, ort_policy, ort_wdl),
            "rust_ort": _f32_sha256(np, rust_policy, rust_wdl),
        },
        "comparisons": comparisons,
    }
