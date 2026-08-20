from __future__ import annotations

import copy
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from alphamini_train.errors import DependencyUnavailable, IntegrityError
from alphamini_train.schema import (
    TensorCache,
    derive_game_seed,
    iter_shard,
    validate_collection_manifest,
)

from conftest import REPOSITORY


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor(path: Path, dtype: str, shape: list[int], values: bytes) -> dict[str, object]:
    path.write_bytes(values)
    return {
        "path": path.name,
        "dtype": dtype,
        "shape": shape,
        "bytes": len(values),
        "sha256": _sha(path),
    }


def write_tensor_cache(root: Path, records: int = 2) -> Path:
    root.mkdir()
    inputs = struct.pack("<" + "f" * (records * 22 * 8 * 8), *([0.0] * (records * 22 * 8 * 8)))
    offsets = struct.pack("<" + "Q" * (records + 1), *range(records + 1))
    indices = struct.pack("<" + "H" * records, *range(records))
    values = struct.pack("<" + "f" * records, *([1.0] * records))
    wdl_values = sum(([0.0, 1.0, 0.0] for _ in range(records)), [])
    wdl = struct.pack("<" + "f" * (records * 3), *wdl_values)
    game_ids = struct.pack("<" + "Q" * records, *range(records))
    manifest = {
        "schema": "tensor-cache-manifest-v1",
        "encoder_schema": "encoder-v1",
        "action_schema": "policy-v1",
        "source_collection_sha256": "a" * 64,
        "record_count": records,
        "policy_size": 4672,
        "input_shape": [22, 8, 8],
        "inputs": _tensor(root / "inputs.f32.bin", "f32-le", [records, 22, 8, 8], inputs),
        "policy_offsets": _tensor(root / "offsets.u64.bin", "u64-le", [records + 1], offsets),
        "policy_indices": _tensor(root / "indices.u16.bin", "u16-le", [records], indices),
        "policy_values": _tensor(root / "values.f32.bin", "f32-le", [records], values),
        "wdl": _tensor(root / "wdl.f32.bin", "f32-le", [records, 3], wdl),
        "game_ids": _tensor(root / "games.u64.bin", "u64-le", [records], game_ids),
    }
    path = root / "tensors.json"
    path.write_text(json.dumps(manifest))
    return path


def _valid_shard() -> dict[str, object]:
    bitboards = [0] * 12
    bitboards[5] = 1 << 4
    bitboards[11] = 1 << 60
    position = {
        "schema": "position-record-v1",
        "game_id": 7,
        "ply": 0,
        "piece_bitboards": bitboards,
        "side_to_move": "white",
        "castling_rights": 0,
        "en_passant_square": None,
        "halfmove_clock": 0,
        "fullmove_number": 1,
        "prior_occurrences": 0,
        "previous_move_uci": None,
        "selected_move_uci": "e1e2",
        "policy": [{"from": 4, "to": 12, "promotion": None, "visits": 1}],
        "outcome": "draw",
        "termination": "ply_limit",
    }
    return {
        "schema": "self-play-shard-v1",
        "encoder_schema": "encoder-v1",
        "action_schema": "policy-v1",
        "seed": 5,
        "simulations": 1,
        "max_plies": 1,
        "games": [
            {
                "schema": "game-record-v1",
                "game_id": 7,
                "seed": derive_game_seed(5, 7),
                "model_sha256": "a" * 64,
                "outcome": "draw",
                "termination": "ply_limit",
                "plies": 1,
                "positions": [position],
            }
        ],
    }


def _write_shard(path: Path, shard: dict[str, object]) -> None:
    try:
        import msgpack
        import zstandard
    except ImportError:
        pytest.skip("raw-shard dependencies are not installed")
    path.write_bytes(zstandard.ZstdCompressor().compress(msgpack.packb(shard, use_bin_type=True)))


def test_rust_tensor_manifest_shape_and_checksums(tmp_path: Path) -> None:
    path = write_tensor_cache(tmp_path / "cache")
    cache = TensorCache(path)
    assert cache.record_count == 2
    try:
        arrays = cache.arrays()
    except DependencyUnavailable:
        pytest.skip("NumPy is not installed")
    assert arrays["inputs"].shape == (2, 22, 8, 8)
    assert list(arrays["policy_indices"]) == [0, 1]


def test_tensor_corruption_is_rejected(tmp_path: Path) -> None:
    path = write_tensor_cache(tmp_path / "cache")
    values = path.parent / "values.f32.bin"
    values.write_bytes(b"bad!")
    with pytest.raises(IntegrityError, match="checksum|truncated"):
        TensorCache(path)


def test_collection_rejects_corrupt_shard(tmp_path: Path) -> None:
    shard = tmp_path / "shard.msgpack.zst"
    shard.write_bytes(b"sealed raw bytes")
    manifest = {
        "schema": "collection-manifest-v1",
        "encoder_schema": "encoder-v1",
        "action_schema": "policy-v1",
        "run_id": "test-run",
        "cycle_id": 0,
        "game_id_start": 0,
        "model_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "seed": 1,
        "simulations": 16,
        "max_plies": 128,
        "game_count": 1,
        "position_count": 3,
        "shards": [
            {
                "path": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": _sha(shard),
                "first_game_id": 0,
                "last_game_id": 0,
                "game_count": 1,
                "position_count": 3,
            }
        ],
    }
    path = tmp_path / "collection.json"
    path.write_text(json.dumps(manifest))
    validate_collection_manifest(path)
    shard.write_bytes(b"tampered raw bytes")
    with pytest.raises(IntegrityError, match="checksum|truncated"):
        validate_collection_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("zero_visit", "visits"),
        ("duplicate_target", "duplicate sparse target"),
        ("wrong_ply", "metadata disagree"),
        ("too_many_positions", "max_plies contract"),
        ("outcome_termination", "inconsistent with termination"),
        ("prior_occurrences", "prior_occurrences"),
        ("selected_move", "selected_move_uci"),
        ("game_seed", "deterministic SplitMix64"),
        ("visit_sum", "visit sum"),
        ("ply_limit_length", "ply-limit game length"),
        ("max_plies_above_v1_cap", "max_plies"),
    ],
)
def test_raw_shard_rejects_rust_schema_violations(
    tmp_path: Path, mutation: str, message: str
) -> None:
    shard = _valid_shard()
    game = shard["games"][0]
    position = game["positions"][0]
    if mutation == "zero_visit":
        position["policy"][0]["visits"] = 0
    elif mutation == "duplicate_target":
        position["policy"].append(copy.deepcopy(position["policy"][0]))
    elif mutation == "wrong_ply":
        position["ply"] = 1
    elif mutation == "too_many_positions":
        game["positions"] = [copy.deepcopy(position) for _ in range(513)]
        game["plies"] = 513
    elif mutation == "outcome_termination":
        game["outcome"] = "white_win"
        position["outcome"] = "white_win"
    elif mutation == "prior_occurrences":
        position["prior_occurrences"] = 3
    elif mutation == "selected_move":
        position["selected_move_uci"] = ""
    elif mutation == "game_seed":
        game["seed"] ^= 1
    elif mutation == "visit_sum":
        position["policy"][0]["visits"] = 2
    elif mutation == "ply_limit_length":
        shard["max_plies"] = 2
    elif mutation == "max_plies_above_v1_cap":
        shard["max_plies"] = 513
    path = tmp_path / f"{mutation}.msgpack.zst"
    _write_shard(path, shard)

    with pytest.raises(IntegrityError, match=message):
        list(iter_shard(path))


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is unavailable")
def test_python_consumes_rust_produced_fixture(tmp_path: Path) -> None:
    output = tmp_path / "rust-fixture"
    # Always let Cargo freshness-check the Rust producer. Calling an existing target/
    # binary directly can silently test yesterday's wire schema after source changes.
    command = [
        "cargo",
        "run",
        "--quiet",
        "--locked",
        "-p",
        "alphamini",
        "--bin",
        "alphamini-fixtures",
        "--",
        str(output),
    ]
    subprocess.run(
        command,
        cwd=REPOSITORY,
        check=True,
    )
    collection_path = output / "collection.json"
    collection = validate_collection_manifest(collection_path, decode_shards=True)
    assert collection["game_id_start"] == 100
    assert collection["seed"] == 7
    assert collection["simulations"] == 2
    assert collection["max_plies"] == 2
    records = list(iter_shard(output / collection["shards"][0]["path"]))
    assert records and all(record.game_id == 100 for record in records)
    cache = TensorCache(output / "tensors.json")
    arrays = cache.arrays()
    assert cache.record_count == len(records)
    assert arrays["inputs"].shape == (len(records), 22, 8, 8)
    assert arrays["policy_indices"].max() < 4672
    assert set(arrays["game_ids"].tolist()) == {100}
