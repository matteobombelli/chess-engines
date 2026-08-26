from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Sequence

REPOSITORY = Path(__file__).resolve().parents[2]
PILOT_CONFIG = REPOSITORY / "configs" / "minigpt" / "pilot.toml"
V1_CONFIG = REPOSITORY / "configs" / "minigpt" / "v1.toml"

BOS = 4672
PAD = 4673


def make_game(index: int, plies: int) -> list[int]:
    """A deterministic in-vocabulary game: BOS then `plies` move tokens."""

    return [BOS] + [(index * 2_654_435_761 + ply * 40_503) % 4672 for ply in range(plies)]


def make_games(count: int, *, first: int = 0, minimum: int = 4, span: int = 20) -> list[list[int]]:
    return [make_game(first + index, minimum + (index % span)) for index in range(count)]


def write_shard(
    directory: Path, prefix: str, number: int, games: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Write one `.bin`/`.idx` pair exactly as the Rust ShardWriter does."""

    directory.mkdir(parents=True, exist_ok=True)
    tokens = b"".join(struct.pack("<H", token) for game in games for token in game)
    offsets = [0]
    for game in games:
        offsets.append(offsets[-1] + len(game))
    index = struct.pack("<Q", len(games)) + b"".join(struct.pack("<Q", value) for value in offsets)
    stem = f"{prefix}-{number:04d}"
    (directory / f"{stem}.bin").write_bytes(tokens)
    (directory / f"{stem}.idx").write_bytes(index)
    return {
        "tokens_path": f"{stem}.bin",
        "index_path": f"{stem}.idx",
        "tokens_sha256": hashlib.sha256(tokens).hexdigest(),
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "token_count": len(tokens) // 2,
        "game_count": len(games),
    }


def write_shards(
    directory: Path,
    train: Sequence[Sequence[Sequence[int]]],
    validation: Sequence[Sequence[Sequence[int]]] = (),
) -> Path:
    """Write shard files plus the `shards.json` manifest; each argument is a list of shards."""

    train_files = [
        write_shard(directory, "shard", number, games) for number, games in enumerate(train)
    ]
    val_files = [
        write_shard(directory, "val", number, games) for number, games in enumerate(validation)
    ]
    manifest = {
        "schema": "minigpt.shards.v1",
        "tokenizer": "policy-v1",
        "vocab_size": 4736,
        "bos_token": BOS,
        "pad_token": PAD,
        "filters": {
            "min_elo": 1800,
            "min_plies": 4,
            "max_plies": 300,
            "token_target": 1000,
            "val_fraction_ppm": 2000,
            "shard_tokens": 4096,
        },
        "sources": [
            {
                "path": "fixture.pgn.zst",
                "sha256": "0" * 64,
                "compressed_bytes": 0,
                "games_seen": sum(len(games) for games in [*train, *validation]),
            }
        ],
        "counts": {
            "games_seen": sum(len(games) for games in [*train, *validation]),
            "games_accepted": sum(len(games) for games in [*train, *validation]),
            "games_train": sum(entry["game_count"] for entry in train_files),
            "games_val": sum(entry["game_count"] for entry in val_files),
            "tokens_train": sum(entry["token_count"] for entry in train_files),
            "tokens_val": sum(entry["token_count"] for entry in val_files),
            "rejected": {
                "non_standard_start": 0,
                "event": 0,
                "elo": 0,
                "termination": 0,
                "variation": 0,
                "ply_bounds": 0,
                "san_error": 0,
            },
        },
        "train_shards": train_files,
        "val_shards": val_files,
        "san_error_samples": [],
        "started_unix_seconds": 1_700_000_000,
        "completed_unix_seconds": 1_700_000_100,
    }
    path = directory / "shards.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


TINY_CONFIG: dict[str, Any] = {
    "run": {
        "name": "minigpt-test",
        "description": "unit-test fixture run",
        "seed": 7,
        "active_budget_hours": 1.0,
        "disposable": True,
    },
    "model": {
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 64,
        "ctx": 32,
        "vocab": 4736,
        "dropout": 0.1,
    },
    "data": {
        "shards_dir": "",
        "tokenizer": "policy-v1",
        "bos_token": BOS,
        "pad_token": PAD,
        "length_buckets": 4,
    },
    "training": {
        "micro_batch": 4,
        "grad_accum": 2,
        "total_steps": 8,
        "segment_steps": 4,
        "learning_rate": 0.001,
        "minimum_learning_rate": 0.0001,
        "warmup_fraction": 0.02,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "eval_interval_steps": 1,
        "eval_batches": 2,
        "checkpoint_interval_steps": 2,
        "early_stop_patience_evals": 100,
        "checkpoint_keep_last": 2,
        "checkpoint_milestone_every_steps": 4,
        "disk_floor_bytes": 0,
        "device": "cpu",
        "amp": False,
        "deterministic": True,
    },
    "export": {
        "opset": 17,
        "dtype": "float32",
        "parity_atol": 0.001,
        "parity_rtol": 0.0,
        "decode_temperature": 0.5,
    },
    "operations": {"heartbeat_seconds": 1},
}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(f"unsupported TOML value: {value!r}")


def write_config(path: Path, shards_dir: Path, **overrides: dict[str, Any]) -> Path:
    values: dict[str, Any] = {section: dict(table) for section, table in TINY_CONFIG.items()}
    values["data"]["shards_dir"] = str(shards_dir)
    for section, table in overrides.items():
        values[section].update(table)
    lines = ['schema = "minigpt.config.v1"', ""]
    for section, table in values.items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {_toml_value(item)}" for key, item in table.items())
        lines.append("")
    path.write_text("\n".join(lines))
    return path
