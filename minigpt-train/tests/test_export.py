from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from minigpt_train.config import BOS_TOKEN, PAD_TOKEN, POLICY_SIZE, load_config

from conftest import write_config

extras_available = all(
    importlib.util.find_spec(module) is not None
    for module in ("torch", "onnx", "onnxruntime", "numpy")
)
requires_extras = pytest.mark.skipif(
    not extras_available, reason="full train/ONNX extras are not installed"
)

EXPORT_MODEL = {"d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 128, "ctx": 256}


def _exported(tmp_path: Path):
    from minigpt_train.export import export_onnx
    from minigpt_train.model import build_model

    import torch

    config = load_config(
        write_config(tmp_path / "config.toml", tmp_path / "shards", model=EXPORT_MODEL)
    )
    torch.manual_seed(int(config.values["run"]["seed"]))
    model = build_model(config)
    model_path, manifest_path, manifest = export_onnx(
        model,
        config,
        tmp_path / "models",
        global_step=12,
        parent_checkpoint_sha256="0" * 64,
        seed=3,
    )
    return config, model, model_path, manifest_path, manifest


@requires_extras
def test_export_matches_torch_at_every_sequence_length(tmp_path: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch

    from minigpt_train.export import INPUT_NAME, OUTPUT_NAME, sample_tokens

    config, model, model_path, _, manifest = _exported(tmp_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    atol = float(config.values["export"]["parity_atol"])
    model = model.eval()
    for length in (4, 64, 256):
        tokens = sample_tokens(length, seed=11)
        assert tokens[0, 0] == BOS_TOKEN
        assert int(tokens[0, 1:].max(initial=0)) < POLICY_SIZE
        with torch.no_grad():
            expected = model(torch.from_numpy(tokens)).numpy()
        actual = session.run(None, {INPUT_NAME: tokens})[0]
        assert actual.shape == (1, length, manifest["vocab_size"])
        assert np.max(np.abs(expected - actual)) < atol
    assert session.get_inputs()[0].name == INPUT_NAME
    assert session.get_outputs()[0].name == OUTPUT_NAME


@requires_extras
def test_exported_graph_keeps_the_sequence_axis_dynamic(tmp_path: Path) -> None:
    import onnx

    _, _, model_path, _, _ = _exported(tmp_path)
    graph = onnx.load(str(model_path)).graph
    tokens = graph.input[0]
    dimensions = tokens.type.tensor_type.shape.dim
    assert tokens.type.tensor_type.elem_type == onnx.TensorProto.INT64
    assert dimensions[0].dim_value == 1
    assert dimensions[1].dim_param == "sequence"
    logits = graph.output[0]
    assert logits.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert logits.type.tensor_type.shape.dim[1].dim_param == "sequence"
    assert logits.type.tensor_type.shape.dim[2].dim_value == 4736


@requires_extras
def test_manifest_and_provenance_describe_the_published_model(tmp_path: Path) -> None:
    from minigpt_train.atomic import sha256_file
    from minigpt_train.export import MANIFEST_FIELDS, publish_current

    _, _, model_path, manifest_path, manifest = _exported(tmp_path)
    assert set(manifest) == MANIFEST_FIELDS
    assert manifest["schema"] == "minigpt.manifest.v1"
    assert manifest["tokenizer"] == "policy-v1"
    assert manifest["onnx_opset"] == 17
    assert manifest["bos_token"] == BOS_TOKEN and manifest["pad_token"] == PAD_TOKEN
    assert manifest["policy_size"] == POLICY_SIZE
    assert manifest["decode_temperature"] == 0.5
    assert manifest["d_model"] == EXPORT_MODEL["d_model"]
    assert manifest["n_layers"] == EXPORT_MODEL["n_layers"]
    assert manifest["n_heads"] == EXPORT_MODEL["n_heads"]
    assert manifest["context"] == EXPORT_MODEL["ctx"]
    assert manifest["model_sha256"] == sha256_file(model_path)
    assert json.loads(manifest_path.read_text()) == manifest

    provenance = json.loads(
        manifest_path.with_name(manifest_path.stem + ".training.json").read_text()
    )
    assert provenance["parity"]["status"] == "passed"
    assert [case["sequence_length"] for case in provenance["parity"]["comparisons"]] == [
        1,
        4,
        64,
        256,
    ]
    assert provenance["global_step"] == 12

    published = publish_current(model_path, manifest_path, tmp_path / "current")
    assert published["model"].name == "model.onnx"
    assert sha256_file(published["model"]) == manifest["model_sha256"]
    assert json.loads(published["manifest"].read_text()) == manifest


@requires_extras
def test_parity_fixture_is_self_consistent(tmp_path: Path) -> None:
    from minigpt_train.atomic import sha256_bytes
    from minigpt_train.parity import verify_parity_fixture, write_parity_fixture

    import numpy as np

    config, model, model_path, _, manifest = _exported(tmp_path)
    fixture_dir = tmp_path / "fixtures"
    index = write_parity_fixture(
        model,
        model_path,
        config,
        fixture_dir,
        model_sha256=manifest["model_sha256"],
        seed=3,
    )
    assert index["schema"] == "minigpt.parity-fixture.v1"
    assert index["input_dtype"] == "int64" and index["logits_dtype"] == "float32-le"
    assert [case["name"] for case in index["cases"]] == ["t0001", "t0004", "t0064", "t0256"]
    for case in index["cases"]:
        payload = (fixture_dir / case["logits_path"]).read_bytes()
        assert len(payload) == case["sequence_length"] * 4736 * 4
        assert sha256_bytes(payload) == case["logits_sha256"]
        assert case["tokens"][0] == BOS_TOKEN
        assert len(case["tokens"]) == case["sequence_length"]
        tokens = np.asarray([case["tokens"]], dtype="<i8")
        assert sha256_bytes(tokens.tobytes()) == case["tokens_sha256"]
        assert case["python_ort_max_abs"] < index["atol"]
        logits = np.frombuffer(payload, dtype="<f4").reshape(case["logits_shape"])
        assert np.isfinite(logits).all()
    assert verify_parity_fixture(fixture_dir) == index


@requires_extras
def test_parity_fixture_verification_catches_a_flipped_byte(tmp_path: Path) -> None:
    from minigpt_train.errors import IntegrityError
    from minigpt_train.parity import verify_parity_fixture, write_parity_fixture

    config, model, model_path, _, manifest = _exported(tmp_path)
    fixture_dir = tmp_path / "fixtures"
    write_parity_fixture(
        model, model_path, config, fixture_dir, model_sha256=manifest["model_sha256"], seed=3
    )
    target = fixture_dir / "logits-t0004.f32"
    payload = bytearray(target.read_bytes())
    payload[0] ^= 0x01
    target.chmod(0o644)
    target.write_bytes(bytes(payload))
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        verify_parity_fixture(fixture_dir)
