"""Frozen token/logit fixtures the Rust engine's inference parity check is written against.

The fixture directory holds one JSON index plus one raw logit file per case:

* `parity.json` - schema `minigpt.parity-fixture.v1`, listing every case.
* `logits-tNNNN.f32` - little-endian `f32`, C order, exactly `1 * T * vocab_size`
  values, so the file is `T * vocab_size * 4` bytes.

`tokens_sha256` digests the ONNX input tensor exactly as the engine must build it:
little-endian `i64`, C order, shape `[1, T]`. `logits_sha256` digests the raw logit
file. Expected values are PyTorch FP32 CPU outputs; ONNX Runtime agreement within
`atol` is verified when the fixture is written and recorded per case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .atomic import atomic_write_bytes, atomic_write_json, read_json, sha256_bytes
from .config import TOKENIZER, ResolvedConfig
from .errors import DependencyUnavailable, IntegrityError
from .export import INPUT_NAME, parity_lengths, sample_tokens
from .model import require_torch
from .run import utc_now

PARITY_FIXTURE_SCHEMA = "minigpt.parity-fixture.v1"
FIXTURE_INDEX_FILE = "parity.json"


def write_parity_fixture(
    model: Any,
    model_path: Path,
    config: ResolvedConfig,
    output_dir: Path,
    *,
    model_sha256: str,
    seed: int,
) -> dict[str, Any]:
    torch = require_torch()
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:
        raise DependencyUnavailable("parity fixtures require NumPy and ONNX Runtime") from error
    architecture = config.values["model"]
    context = int(architecture["ctx"])
    vocab = int(architecture["vocab"])
    atol = float(config.values["export"]["parity_atol"])
    rtol = float(config.values["export"]["parity_rtol"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").float().eval()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    cases: list[dict[str, Any]] = []
    for length in parity_lengths(context):
        tokens = sample_tokens(length, seed=seed)
        with torch.no_grad():
            logits = model(torch.from_numpy(tokens)).numpy().astype(np.float32, copy=False)
        if logits.shape != (1, length, vocab):
            raise IntegrityError(f"fixture logits shape {logits.shape} is wrong at T={length}")
        if not bool(np.isfinite(logits).all()):
            raise IntegrityError(f"fixture logits are non-finite at T={length}")
        payload = np.ascontiguousarray(logits, dtype="<f4").tobytes(order="C")
        name = f"t{length:04d}"
        logits_path = output_dir / f"logits-{name}.f32"
        atomic_write_bytes(logits_path, payload, mode=0o444)
        if logits_path.read_bytes() != payload:
            raise IntegrityError(f"fixture logit file did not round-trip: {logits_path}")
        runtime = np.asarray(session.run(None, {INPUT_NAME: tokens})[0], dtype=np.float32)
        if not bool(np.allclose(logits, runtime, atol=atol, rtol=rtol)):
            raise IntegrityError(f"fixture disagrees with ONNX Runtime at T={length}")
        cases.append(
            {
                "name": name,
                "sequence_length": length,
                "tokens": [int(token) for token in tokens[0]],
                "tokens_sha256": sha256_bytes(np.ascontiguousarray(tokens, dtype="<i8").tobytes()),
                "logits_path": logits_path.name,
                "logits_shape": [1, length, vocab],
                "logits_sha256": sha256_bytes(payload),
                "python_ort_max_abs": float(np.max(np.abs(logits - runtime))),
            }
        )

    index = {
        "schema": PARITY_FIXTURE_SCHEMA,
        "generated_at": utc_now(),
        "tokenizer": TOKENIZER,
        "vocab_size": vocab,
        "context": context,
        "model_sha256": model_sha256,
        "input_name": INPUT_NAME,
        "input_dtype": "int64",
        "logits_dtype": "float32-le",
        "atol": atol,
        "rtol": rtol,
        "cases": cases,
    }
    atomic_write_json(output_dir / FIXTURE_INDEX_FILE, index)
    return index


def verify_parity_fixture(fixture_dir: Path) -> dict[str, Any]:
    """Re-read a fixture and check every recorded size and digest identity."""

    index = read_json(fixture_dir / FIXTURE_INDEX_FILE)
    if not isinstance(index, dict) or index.get("schema") != PARITY_FIXTURE_SCHEMA:
        raise IntegrityError(f"unsupported parity fixture: {fixture_dir}")
    if index.get("tokenizer") != TOKENIZER:
        raise IntegrityError("parity fixture uses another tokenizer")
    for case in index.get("cases", []):
        path = fixture_dir / case["logits_path"]
        payload = path.read_bytes()
        expected_bytes = case["sequence_length"] * index["vocab_size"] * 4
        if len(payload) != expected_bytes:
            raise IntegrityError(f"parity fixture {path.name} is {len(payload)} bytes")
        if sha256_bytes(payload) != case["logits_sha256"]:
            raise IntegrityError(f"parity fixture checksum mismatch: {path.name}")
        if len(case["tokens"]) != case["sequence_length"]:
            raise IntegrityError(f"parity fixture {case['name']} token count is wrong")
    return index
