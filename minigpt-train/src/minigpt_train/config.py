"""Strict TOML configuration loading and semantic identity."""

from __future__ import annotations

import copy
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic import canonical_json_bytes, sha256_bytes
from .errors import ConfigError

CONFIG_SCHEMA = "minigpt.config.v1"
TOKENIZER = "policy-v1"
POLICY_SIZE = 4672
BOS_TOKEN = 4672
PAD_TOKEN = 4673
VOCAB_SIZE = 4736
# Excluded keys describe how a run is operated, never what it computes. Two configs
# that differ only here produce the same model and share a semantic hash.
SEMANTIC_EXCLUSIONS = {
    "run": {"name", "description", "active_budget_hours", "output_dir"},
    "training": {
        "checkpoint_keep_last",
        "checkpoint_milestone_every_steps",
        "disk_floor_bytes",
    },
    "operations": {"heartbeat_seconds"},
}


@dataclass(frozen=True)
class ResolvedConfig:
    source: Path
    values: dict[str, Any]
    config_hash: str
    semantic_values: dict[str, Any]
    semantic_hash: str

    def get(self, dotted: str, default: Any = None) -> Any:
        current: Any = self.values
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


def _semantic_projection(values: dict[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(values)
    for section, keys in SEMANTIC_EXCLUSIONS.items():
        table = projected.get(section)
        if isinstance(table, dict):
            for key in keys:
                table.pop(key, None)
    return projected


def _require_table(values: dict[str, Any], name: str) -> dict[str, Any]:
    value = values.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] table")
    return value


def _integer(
    table: dict[str, Any],
    key: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    value = table.get(key)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = f" in [{minimum}, {maximum}]" if maximum is not None else f" >= {minimum}"
        raise ConfigError(f"{key} must be an integer{bound}")
    return value


def _number(table: dict[str, Any], key: str, *, minimum: float = 0.0) -> float:
    value = table.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ConfigError(f"{key} must be finite and >= {minimum}")
    return result


def _boolean(table: dict[str, Any], key: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def validate_config(values: dict[str, Any]) -> None:
    if values.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"schema must be {CONFIG_SCHEMA!r}")

    run = _require_table(values, "run")
    if not isinstance(run.get("name"), str) or not run["name"].strip():
        raise ConfigError("run.name must be a non-empty string")
    if not isinstance(run.get("description"), str):
        raise ConfigError("run.description must be a string")
    _integer(run, "seed", maximum=2**64 - 1)
    _number(run, "active_budget_hours", minimum=0.001)
    _boolean(run, "disposable")

    model = _require_table(values, "model")
    if _integer(model, "vocab", minimum=1) != VOCAB_SIZE:
        raise ConfigError(f"v1 model.vocab must be {VOCAB_SIZE}")
    ctx = _integer(model, "ctx", minimum=2, maximum=2**16 - 1)
    d_model = _integer(model, "d_model", minimum=1)
    n_heads = _integer(model, "n_heads", minimum=1)
    if d_model % n_heads != 0:
        raise ConfigError("model.d_model must be divisible by model.n_heads")
    _integer(model, "n_layers", minimum=1)
    _integer(model, "d_ff", minimum=1)
    dropout = _number(model, "dropout", minimum=0.0)
    if dropout >= 1:
        raise ConfigError("model.dropout must be < 1")

    data = _require_table(values, "data")
    if not isinstance(data.get("shards_dir"), str) or not data["shards_dir"].strip():
        raise ConfigError("data.shards_dir must be a non-empty string")
    if data.get("tokenizer") != TOKENIZER:
        raise ConfigError(f"data.tokenizer must be {TOKENIZER!r}")
    if _integer(data, "bos_token", maximum=VOCAB_SIZE - 1) != BOS_TOKEN:
        raise ConfigError(f"v1 data.bos_token must be {BOS_TOKEN}")
    if _integer(data, "pad_token", maximum=VOCAB_SIZE - 1) != PAD_TOKEN:
        raise ConfigError(f"v1 data.pad_token must be {PAD_TOKEN}")
    _integer(data, "length_buckets", minimum=1, maximum=ctx)

    training = _require_table(values, "training")
    _integer(training, "micro_batch", minimum=1)
    _integer(training, "grad_accum", minimum=1)
    total_steps = _integer(training, "total_steps", minimum=1)
    segment_steps = _integer(training, "segment_steps", minimum=1)
    if segment_steps > total_steps:
        raise ConfigError("training.segment_steps must not exceed training.total_steps")
    peak = _number(training, "learning_rate", minimum=0.0)
    floor = _number(training, "minimum_learning_rate", minimum=0.0)
    if floor > peak:
        raise ConfigError("training.minimum_learning_rate must not exceed learning_rate")
    warmup = _number(training, "warmup_fraction", minimum=0.0)
    if warmup >= 1:
        raise ConfigError("training.warmup_fraction must be < 1")
    _number(training, "weight_decay", minimum=0.0)
    _number(training, "gradient_clip", minimum=0.0)
    _integer(training, "eval_interval_steps", minimum=1)
    _integer(training, "eval_batches", minimum=1)
    _integer(training, "checkpoint_interval_steps", minimum=1)
    _integer(training, "early_stop_patience_evals", minimum=1)
    _integer(training, "checkpoint_keep_last", minimum=1)
    _integer(training, "checkpoint_milestone_every_steps", minimum=1)
    _integer(training, "disk_floor_bytes", minimum=0)
    if training.get("device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("training.device must be auto, cpu, or cuda")
    _boolean(training, "amp")
    _boolean(training, "deterministic")

    export = _require_table(values, "export")
    if export.get("dtype") != "float32":
        raise ConfigError("v1 export.dtype must be float32")
    if _integer(export, "opset", minimum=17) != 17:
        raise ConfigError("v1 export.opset must be exactly 17")
    _number(export, "parity_atol", minimum=0.0)
    _number(export, "parity_rtol", minimum=0.0)
    temperature = _number(export, "decode_temperature", minimum=0.0)
    if temperature <= 0:
        raise ConfigError("export.decode_temperature must be positive")

    operations = _require_table(values, "operations")
    _integer(operations, "heartbeat_seconds", minimum=1)


def load_config(path: Path | str) -> ResolvedConfig:
    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
        values = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot load {source}: {error}") from error
    validate_config(values)
    config_hash = sha256_bytes(raw)
    semantic = _semantic_projection(values)
    semantic_hash = sha256_bytes(canonical_json_bytes(semantic))
    # Round-trip through JSON to ensure snapshots contain no TOML-only objects.
    try:
        json.dumps(values)
    except TypeError as error:
        raise ConfigError(f"configuration contains a non-JSON value: {error}") from error
    return ResolvedConfig(source, values, config_hash, semantic, semantic_hash)
