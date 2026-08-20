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

CONFIG_SCHEMA = "alphamini.config.v1"
SEMANTIC_EXCLUSIONS = {
    "run": {"name", "description", "active_budget_hours", "output_dir"},
    "operations": {"heartbeat_seconds", "command_timeout_seconds"},
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


def validate_config(values: dict[str, Any]) -> None:
    if values.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"schema must be {CONFIG_SCHEMA!r}")

    run = _require_table(values, "run")
    if not isinstance(run.get("name"), str) or not run["name"].strip():
        raise ConfigError("run.name must be a non-empty string")
    _integer(run, "seed", maximum=2**64 - 1)
    _number(run, "active_budget_hours", minimum=0.001)
    if not isinstance(run.get("disposable"), bool):
        raise ConfigError("run.disposable must be a boolean")

    schemas = _require_table(values, "schemas")
    expected_schemas = {
        "position_record": "position-record-v1",
        "tensor_cache": "tensor-cache-manifest-v1",
        "encoder": "encoder-v1",
        "action": "policy-v1",
        "model_manifest": "model-manifest-v1",
    }
    for key, expected in expected_schemas.items():
        if schemas.get(key) != expected:
            raise ConfigError(f"schemas.{key} must be {expected!r}")

    model = _require_table(values, "model")
    if _integer(model, "input_planes", minimum=1) != 22:
        raise ConfigError("v1 model.input_planes must be 22")
    if _integer(model, "action_size", minimum=1) != 4672:
        raise ConfigError("v1 model.action_size must be 4672")
    _integer(model, "channels", minimum=1)
    _integer(model, "residual_blocks", minimum=1)
    _integer(model, "se_hidden", minimum=1)

    self_play = _require_table(values, "self_play")
    _integer(self_play, "games_per_cycle", minimum=1)
    _integer(self_play, "simulations", minimum=1, maximum=2**32 - 1)
    _integer(self_play, "batch_size", minimum=1)
    _integer(self_play, "max_plies", minimum=1, maximum=512)
    alpha = _number(self_play, "dirichlet_alpha", minimum=0.0)
    if alpha <= 0:
        raise ConfigError("self_play.dirichlet_alpha must be positive")
    epsilon = _number(self_play, "dirichlet_epsilon", minimum=0.0)
    if epsilon > 1:
        raise ConfigError("self_play.dirichlet_epsilon must be <= 1")
    _integer(self_play, "sample_until_ply", minimum=0, maximum=2**16 - 1)
    cpuct = _number(self_play, "cpuct", minimum=0.0)
    if cpuct <= 0:
        raise ConfigError("self_play.cpuct must be positive")
    _number(self_play, "fpu_reduction", minimum=0.0)

    training = _require_table(values, "training")
    _integer(training, "batch_size", minimum=1)
    _number(training, "sample_ratio", minimum=0.001)
    _integer(training, "replay_positions", minimum=1)
    _number(training, "validation_fraction", minimum=0.0)
    if float(training["validation_fraction"]) >= 1:
        raise ConfigError("training.validation_fraction must be < 1")
    _integer(training, "validation_batches", minimum=1)
    _number(training, "learning_rate", minimum=0.0)
    _number(training, "minimum_learning_rate", minimum=0.0)
    _number(training, "weight_decay", minimum=0.0)
    _number(training, "gradient_clip", minimum=0.0)
    _integer(training, "checkpoint_every_steps", minimum=1)
    _integer(training, "frozen_horizon_steps", minimum=1)
    if not isinstance(training.get("horizon_confirmed"), bool):
        raise ConfigError("training.horizon_confirmed must be a boolean")
    warmup = _number(training, "warmup_fraction", minimum=0.0)
    if warmup >= 1:
        raise ConfigError("training.warmup_fraction must be < 1")
    if training.get("device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("training.device must be auto, cpu, or cuda")
    if not isinstance(training.get("amp"), bool) or not isinstance(
        training.get("deterministic"), bool
    ):
        raise ConfigError("training.amp and training.deterministic must be booleans")

    export = _require_table(values, "export")
    if export.get("dtype") != "float32":
        raise ConfigError("v1 export.dtype must be float32")
    if _integer(export, "opset", minimum=17) != 17:
        raise ConfigError("v1 export.opset must be exactly 17")
    _number(export, "parity_atol", minimum=0.0)
    _number(export, "parity_rtol", minimum=0.0)

    operations = _require_table(values, "operations")
    if operations.get("self_play_device") not in {"cpu", "cuda"}:
        raise ConfigError("operations.self_play_device must be cpu or cuda")
    for key in ("collect_command", "materialize_command"):
        command = operations.get(key, [])
        if not isinstance(command, list) or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ConfigError(f"operations.{key} must be an array of non-empty strings")


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
