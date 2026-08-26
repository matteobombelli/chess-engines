"""FP32 ONNX publication, ONNX Runtime parity, and the served `current` artifact."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

from .atomic import atomic_write_bytes, atomic_write_json, fsync_directory, sha256_file
from .config import BOS_TOKEN, PAD_TOKEN, POLICY_SIZE, TOKENIZER, ResolvedConfig
from .errors import DependencyUnavailable, IntegrityError
from .model import require_torch

MANIFEST_SCHEMA = "minigpt.manifest.v1"
PROVENANCE_SCHEMA = "minigpt.training-model-provenance.v1"
INPUT_NAME = "tokens"
OUTPUT_NAME = "logits"
MANIFEST_FIELDS = {
    "schema",
    "tokenizer",
    "onnx_opset",
    "input_name",
    "output_name",
    "vocab_size",
    "context",
    "bos_token",
    "pad_token",
    "policy_size",
    "d_model",
    "n_layers",
    "n_heads",
    "d_ff",
    "decode_temperature",
    "model_sha256",
}


def parity_lengths(context: int) -> list[int]:
    return sorted({1, min(4, context), min(64, context), context})


def sample_tokens(length: int, *, seed: int) -> Any:
    """A deterministic, in-vocabulary token sequence: BOS then move tokens."""

    try:
        import numpy as np
    except ImportError as error:
        raise DependencyUnavailable("export requires NumPy") from error
    generator = np.random.default_rng(np.random.SeedSequence([seed, length, 0x70A1]))
    tokens = np.empty((1, length), dtype=np.int64)
    tokens[0, 0] = BOS_TOKEN
    if length > 1:
        tokens[0, 1:] = generator.integers(0, POLICY_SIZE, size=length - 1, dtype=np.int64)
    return tokens


def export_onnx(
    model: Any,
    config: ResolvedConfig,
    output_dir: Path,
    *,
    global_step: int,
    parent_checkpoint_sha256: str,
    seed: int,
) -> tuple[Path, Path, dict[str, Any]]:
    torch = require_torch()
    try:
        import numpy as np
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError as error:
        raise DependencyUnavailable("ONNX export requires the train extra") from error
    architecture = config.values["model"]
    context = int(architecture["ctx"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").float().eval()
    trace_input = torch.from_numpy(sample_tokens(context, seed=seed))
    fd, temporary = tempfile.mkstemp(prefix=".model.", suffix=".onnx.partial", dir=output_dir)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.onnx.export(
            model,
            (trace_input,),
            temporary_path,
            export_params=True,
            opset_version=int(config.values["export"]["opset"]),
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            # Batch stays 1: the engine evaluates one game at a time; only the
            # sequence axis varies, and the causal mask follows it.
            dynamic_axes={INPUT_NAME: {1: "sequence"}, OUTPUT_NAME: {1: "sequence"}},
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

    atol = float(config.values["export"]["parity_atol"])
    rtol = float(config.values["export"]["parity_rtol"])
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    comparisons: list[dict[str, Any]] = []
    for length in parity_lengths(context):
        tokens = sample_tokens(length, seed=seed)
        with torch.no_grad():
            expected = model(torch.from_numpy(tokens)).numpy()
        actual = session.run(None, {INPUT_NAME: tokens})[0]
        if actual.shape != (1, length, int(architecture["vocab"])):
            raise IntegrityError(f"exported logits shape {actual.shape} is wrong at T={length}")
        comparisons.append(
            {
                "sequence_length": length,
                "max_abs": float(np.max(np.abs(expected - actual))),
                "passed": bool(np.allclose(expected, actual, atol=atol, rtol=rtol)),
            }
        )
    parity = {
        "status": "passed" if all(item["passed"] for item in comparisons) else "failed",
        "provider": "CPUExecutionProvider",
        "atol": atol,
        "rtol": rtol,
        "comparisons": comparisons,
    }
    if parity["status"] != "passed":
        raise IntegrityError(f"PyTorch/ONNX parity failed: {parity}")

    # Keep this manifest byte-for-byte compatible with the Rust MiniGPT manifest,
    # which intentionally denies unknown fields. Training provenance is separate.
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tokenizer": TOKENIZER,
        "onnx_opset": int(config.values["export"]["opset"]),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "vocab_size": int(architecture["vocab"]),
        "context": context,
        "bos_token": BOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "policy_size": POLICY_SIZE,
        "d_model": int(architecture["d_model"]),
        "n_layers": int(architecture["n_layers"]),
        "n_heads": int(architecture["n_heads"]),
        "d_ff": int(architecture["d_ff"]),
        "decode_temperature": float(config.values["export"]["decode_temperature"]),
        "model_sha256": digest,
    }
    if set(manifest) != MANIFEST_FIELDS:
        raise IntegrityError("model manifest fields drifted from the v1 contract")
    manifest_path = output_dir / f"model-{digest[:16]}.json"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        output_dir / f"model-{digest[:16]}.training.json",
        {
            "schema": PROVENANCE_SCHEMA,
            "semantic_hash": config.semantic_hash,
            "global_step": global_step,
            "model_sha256": digest,
            "checkpoint_sha256": parent_checkpoint_sha256,
            "architecture": architecture,
            "parity": parity,
        },
    )
    return model_path, manifest_path, manifest


def publish_current(model_path: Path, manifest_path: Path, destination: Path) -> dict[str, Path]:
    """Republish the verified pair under the stable `current` names, atomically."""

    published_model = destination / "model.onnx"
    published_manifest = destination / "manifest.json"
    atomic_write_bytes(published_model, model_path.read_bytes())
    atomic_write_bytes(published_manifest, manifest_path.read_bytes())
    if sha256_file(published_model) != sha256_file(model_path):
        raise IntegrityError("published current model differs from its verified source")
    return {"model": published_model, "manifest": published_manifest}
