"""Verified arena-log ingestion and deterministic Bradley-Terry fitting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .atomic import SHA256_RE, atomic_write_json, read_json, sha256_bytes, sha256_file
from .errors import DependencyUnavailable, IntegrityError
from .run import utc_now

LADDER_INPUT_SCHEMA = "alphamini.ladder-input.v1"
ARENA_LADDER_INPUT_SCHEMA = "alphamini.arena-ladder-input.v1"
LADDER_OUTPUT_SCHEMA = "alphamini.bradley-terry-ladder.v1"
ARENA_HEADER_SCHEMA = "alphamini-paired-evaluation-v1"
ARENA_PAIR_SCHEMA = "alphamini-paired-opening-result-v1"
ELO_SCALE = 400.0 / math.log(10.0)

_ARENA_HEADER_FIELDS = {
    "schema",
    "engine_a",
    "engine_b",
    "model_sha256",
    "opponent_model_sha256",
    "opening_suite_sha256",
    "opening_ids",
    "depth",
    "seed",
    "max_plies",
    "simulations",
    "time_ms",
    "batch_size",
    "cpuct_ppm",
    "fpu_reduction_ppm",
    "bootstrap_samples",
    "required_lower_score_ppm",
    "minimax_v1_move_digest",
    "evaluation_binary_sha256",
    "target",
    "inference_device",
    "exploratory",
}
_ARENA_IDENTITY_FIELDS = _ARENA_HEADER_FIELDS - {
    "schema",
    "engine_a",
    "engine_b",
    "model_sha256",
    "opponent_model_sha256",
}
_ARENA_PAIR_FIELDS = {
    "schema",
    "opening_id",
    "engine_a_as_white",
    "engine_a_as_black",
    "metrics",
}
_ARENA_GAME_FIELDS = {"winner", "termination", "plies"}
_ARENA_METRIC_FIELDS = {
    "moves",
    "completed_simulations",
    "neural_evaluations",
    "inference_batches",
    "largest_batch",
    "elapsed_micros",
    "deadlines_reached",
}
_TERMINATIONS = {
    "checkmate",
    "stalemate",
    "insufficient_material",
    "threefold_repetition",
    "fifty_move_rule",
    "ply_limit",
}


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise DependencyUnavailable("Bradley-Terry fitting requires NumPy") from error
    return np


def _count(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise IntegrityError(f"{field} must be a non-negative integer")
    return value


def _prior(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise IntegrityError("prior_sigma_elo must be numeric")
    prior = float(value)
    if not math.isfinite(prior) or prior <= 0:
        raise IntegrityError("prior_sigma_elo must be finite and positive")
    return prior


def _aggregate_input(value: Any) -> tuple[list[str], list[dict[str, Any]], float]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "matches",
        "prior_sigma_elo",
    }:
        raise IntegrityError("ladder input must contain schema, matches, and prior_sigma_elo")
    if value.get("schema") != LADDER_INPUT_SCHEMA:
        raise IntegrityError(f"ladder schema must be {LADDER_INPUT_SCHEMA}")
    prior = _prior(value.get("prior_sigma_elo"))
    raw_matches = value.get("matches")
    if not isinstance(raw_matches, list) or not raw_matches:
        raise IntegrityError("ladder matches must be a non-empty array")
    matches: list[dict[str, Any]] = []
    players: set[str] = set()
    expected = {"player_a", "player_b", "wins_a", "draws", "wins_b"}
    for number, raw in enumerate(raw_matches):
        if not isinstance(raw, dict) or set(raw) != expected:
            raise IntegrityError(f"ladder match {number} has wrong fields")
        player_a = raw.get("player_a")
        player_b = raw.get("player_b")
        if (
            not isinstance(player_a, str)
            or not player_a.strip()
            or not isinstance(player_b, str)
            or not player_b.strip()
            or player_a == player_b
        ):
            raise IntegrityError(f"ladder match {number} has invalid players")
        wins_a = _count(raw.get("wins_a"), "wins_a")
        draws = _count(raw.get("draws"), "draws")
        wins_b = _count(raw.get("wins_b"), "wins_b")
        if wins_a + draws + wins_b == 0:
            raise IntegrityError(f"ladder match {number} contains no games")
        players.update((player_a, player_b))
        matches.append(
            {
                "player_a": player_a,
                "player_b": player_b,
                "wins_a": wins_a,
                "draws": draws,
                "wins_b": wins_b,
            }
        )
    ordered = sorted(players)
    _require_connected(ordered, matches)
    return ordered, matches, prior


def _arena_game(value: Any, field: str) -> str | None:
    if not isinstance(value, dict) or set(value) != _ARENA_GAME_FIELDS:
        raise IntegrityError(f"{field} has wrong fields")
    winner = value.get("winner")
    if winner not in {None, "white", "black"}:
        raise IntegrityError(f"{field}.winner is invalid")
    termination = value.get("termination")
    if termination not in _TERMINATIONS:
        raise IntegrityError(f"{field}.termination is invalid")
    _count(value.get("plies"), f"{field}.plies")
    if (winner is not None) != (termination == "checkmate"):
        raise IntegrityError(f"{field} must have a winner exactly for checkmate")
    return winner


def _arena_pair_counts(pair: Any, opening_id: str, line: int) -> tuple[int, int, int]:
    prefix = f"arena pair line {line}"
    if not isinstance(pair, dict) or set(pair) != _ARENA_PAIR_FIELDS:
        raise IntegrityError(f"{prefix} has wrong fields")
    if pair.get("schema") != ARENA_PAIR_SCHEMA:
        raise IntegrityError(f"{prefix} has unsupported schema")
    if pair.get("opening_id") != opening_id:
        raise IntegrityError(f"{prefix} is not the exact opening-suite prefix")
    metrics = pair.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != _ARENA_METRIC_FIELDS:
        raise IntegrityError(f"{prefix}.metrics has wrong fields")
    for field, value in metrics.items():
        _count(value, f"{prefix}.metrics.{field}")

    wins_a = 0
    draws = 0
    wins_b = 0
    for field, color_a in (("engine_a_as_white", "white"), ("engine_a_as_black", "black")):
        winner = _arena_game(pair.get(field), f"{prefix}.{field}")
        if winner is None:
            draws += 1
        elif winner == color_a:
            wins_a += 1
        else:
            wins_b += 1
    return wins_a, draws, wins_b


def _arena_log(
    path: Path,
    *,
    expected_sha256: str,
    model_a_sha256: str,
    model_b_sha256: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise IntegrityError(f"cannot read arena pair log {path}: {error}") from error
    if sha256_bytes(payload) != expected_sha256:
        raise IntegrityError(f"arena pair-log checksum mismatch: {path}")
    if not payload or not payload.endswith(b"\n"):
        raise IntegrityError(f"arena pair log is empty or has a torn final record: {path}")
    try:
        lines = payload.decode("utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"invalid arena pair log {path}: {error}") from error
    if any(not line.strip() for line in lines):
        raise IntegrityError(f"arena pair log contains a blank record: {path}")
    header = values[0]
    if not isinstance(header, dict) or set(header) != _ARENA_HEADER_FIELDS:
        raise IntegrityError(f"arena header has wrong fields: {path}")
    if header.get("schema") != ARENA_HEADER_SCHEMA:
        raise IntegrityError(f"unsupported arena header schema: {path}")
    if header.get("model_sha256") != model_a_sha256:
        raise IntegrityError(f"arena model A hash does not match ladder descriptor: {path}")
    if header.get("opponent_model_sha256") != model_b_sha256:
        raise IntegrityError(f"arena model B hash does not match ladder descriptor: {path}")
    if header.get("exploratory") is not True:
        raise IntegrityError(f"checkpoint ladder requires an exploratory dual-model log: {path}")
    for field in ("engine_a", "engine_b", "target", "inference_device"):
        if not isinstance(header.get(field), str) or not header[field]:
            raise IntegrityError(f"arena header {field} is invalid: {path}")
    for field in ("opening_suite_sha256", "evaluation_binary_sha256"):
        if not isinstance(header.get(field), str) or not SHA256_RE.fullmatch(header[field]):
            raise IntegrityError(f"arena header {field} is not SHA-256: {path}")
    for field in (
        "depth",
        "seed",
        "max_plies",
        "simulations",
        "time_ms",
        "batch_size",
        "cpuct_ppm",
        "fpu_reduction_ppm",
        "bootstrap_samples",
        "required_lower_score_ppm",
        "minimax_v1_move_digest",
    ):
        _count(header.get(field), f"arena header {field}")
    for field in (
        "max_plies",
        "simulations",
        "time_ms",
        "batch_size",
        "cpuct_ppm",
        "bootstrap_samples",
    ):
        if header[field] == 0:
            raise IntegrityError(f"arena header {field} must be positive: {path}")
    opening_ids = header.get("opening_ids")
    if (
        not isinstance(opening_ids, list)
        or not opening_ids
        or any(not isinstance(item, str) or not item for item in opening_ids)
        or len(set(opening_ids)) != len(opening_ids)
    ):
        raise IntegrityError(f"arena header opening_ids are invalid: {path}")
    if len(values) - 1 != len(opening_ids):
        raise IntegrityError(
            f"arena pair log is incomplete: {path} has {len(values) - 1}/{len(opening_ids)} pairs"
        )
    totals = {"wins_a": 0, "draws": 0, "wins_b": 0}
    for line, (pair, opening_id) in enumerate(zip(values[1:], opening_ids, strict=True), start=2):
        wins_a, draws, wins_b = _arena_pair_counts(pair, opening_id, line)
        totals["wins_a"] += wins_a
        totals["draws"] += draws
        totals["wins_b"] += wins_b
    return header, totals


def _arena_input(
    value: Any, source: Path
) -> tuple[list[str], list[dict[str, Any]], float, list[dict[str, Any]]]:
    expected = {"schema", "pair_logs", "prior_sigma_elo"}
    if not isinstance(value, dict) or set(value) != expected:
        raise IntegrityError(
            "arena ladder input must contain schema, pair_logs, and prior_sigma_elo"
        )
    if value.get("schema") != ARENA_LADDER_INPUT_SCHEMA:
        raise IntegrityError(f"ladder schema must be {ARENA_LADDER_INPUT_SCHEMA}")
    prior = _prior(value.get("prior_sigma_elo"))
    descriptors = value.get("pair_logs")
    if not isinstance(descriptors, list) or not descriptors:
        raise IntegrityError("arena ladder pair_logs must be a non-empty array")

    descriptor_fields = {
        "player_a",
        "player_b",
        "model_a_sha256",
        "model_b_sha256",
        "path",
        "sha256",
    }
    matches: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    players: set[str] = set()
    player_models: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    common_identity: dict[str, Any] | None = None
    for number, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict) or set(descriptor) != descriptor_fields:
            raise IntegrityError(f"arena pair-log descriptor {number} has wrong fields")
        player_a = descriptor.get("player_a")
        player_b = descriptor.get("player_b")
        if (
            not isinstance(player_a, str)
            or not player_a.strip()
            or not isinstance(player_b, str)
            or not player_b.strip()
            or player_a == player_b
        ):
            raise IntegrityError(f"arena pair-log descriptor {number} has invalid players")
        model_a = descriptor.get("model_a_sha256")
        model_b = descriptor.get("model_b_sha256")
        log_sha = descriptor.get("sha256")
        if any(
            not isinstance(item, str) or not SHA256_RE.fullmatch(item)
            for item in (model_a, model_b, log_sha)
        ):
            raise IntegrityError(f"arena pair-log descriptor {number} has invalid SHA-256")
        for player, model in ((player_a, model_a), (player_b, model_b)):
            previous = player_models.setdefault(player, model)
            if previous != model:
                raise IntegrityError(f"player {player!r} is bound to multiple model hashes")
        pair_key = tuple(sorted((player_a, player_b)))
        if pair_key in seen_pairs:
            raise IntegrityError(f"duplicate arena pairing: {pair_key[0]} / {pair_key[1]}")
        seen_pairs.add(pair_key)
        raw_path = descriptor.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise IntegrityError(f"arena pair-log descriptor {number} has invalid path")
        log_path = Path(raw_path)
        if not log_path.is_absolute():
            log_path = source.parent / log_path
        log_path = log_path.resolve()
        header, totals = _arena_log(
            log_path,
            expected_sha256=log_sha,
            model_a_sha256=model_a,
            model_b_sha256=model_b,
        )
        identity = {field: header[field] for field in sorted(_ARENA_IDENTITY_FIELDS)}
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise IntegrityError("arena pair logs do not share one evaluation configuration")
        players.update((player_a, player_b))
        matches.append({"player_a": player_a, "player_b": player_b, **totals})
        sources.append(
            {
                "path": str(log_path),
                "sha256": log_sha,
                "player_a": player_a,
                "player_b": player_b,
                "model_a_sha256": model_a,
                "model_b_sha256": model_b,
                "opening_pairs": len(header["opening_ids"]),
                "games": sum(totals.values()),
            }
        )
    ordered = sorted(players)
    _require_connected(ordered, matches)
    return ordered, matches, prior, sources


def _load_ladder_input(
    path: Path | str,
) -> tuple[list[str], list[dict[str, Any]], float, list[dict[str, Any]]]:
    source = Path(path).resolve()
    value = read_json(source)
    if isinstance(value, dict) and value.get("schema") == ARENA_LADDER_INPUT_SCHEMA:
        return _arena_input(value, source)
    players, matches, prior = _aggregate_input(value)
    return players, matches, prior, []


def load_ladder_input(path: Path | str) -> tuple[list[str], list[dict[str, Any]], float]:
    players, matches, prior, _ = _load_ladder_input(path)
    return players, matches, prior


def _require_connected(players: list[str], matches: list[dict[str, Any]]) -> None:
    adjacent = {player: set() for player in players}
    for match in matches:
        adjacent[match["player_a"]].add(match["player_b"])
        adjacent[match["player_b"]].add(match["player_a"])
    reached = {players[0]}
    pending = [players[0]]
    while pending:
        for neighbor in adjacent[pending.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    if reached != set(players):
        raise IntegrityError("ladder match graph is disconnected")


def _probability(difference: float) -> float:
    if difference >= 0:
        exp_negative = math.exp(-difference)
        return 1.0 / (1.0 + exp_negative)
    exp_positive = math.exp(difference)
    return exp_positive / (1.0 + exp_positive)


def fit_bradley_terry(
    players: list[str],
    matches: list[dict[str, Any]],
    prior_sigma_elo: float,
    *,
    maximum_iterations: int = 200,
    tolerance: float = 1e-11,
) -> dict[str, Any]:
    """Fit draw-as-half-win BT ratings with a frozen zero-mean Gaussian prior."""

    np = _numpy()
    count = len(players)
    index = {player: number for number, player in enumerate(players)}
    ratings = np.zeros(count, dtype=np.float64)
    prior_sigma = prior_sigma_elo / ELO_SCALE
    prior_precision = 1.0 / (prior_sigma * prior_sigma)
    converged = False
    maximum_delta = math.inf
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        gradient = -ratings * prior_precision
        information = np.eye(count, dtype=np.float64) * prior_precision
        for match in matches:
            left = index[match["player_a"]]
            right = index[match["player_b"]]
            games = match["wins_a"] + match["draws"] + match["wins_b"]
            observed = match["wins_a"] + 0.5 * match["draws"]
            probability = _probability(float(ratings[left] - ratings[right]))
            residual = observed - games * probability
            gradient[left] += residual
            gradient[right] -= residual
            weight = games * probability * (1.0 - probability)
            information[left, left] += weight
            information[right, right] += weight
            information[left, right] -= weight
            information[right, left] -= weight
        update = np.linalg.solve(information, gradient)
        ratings += update
        ratings -= ratings.mean()
        maximum_delta = float(np.max(np.abs(update)))
        if maximum_delta < tolerance:
            converged = True
            break
    if not converged:
        raise IntegrityError(
            f"Bradley-Terry optimizer did not converge after {maximum_iterations} iterations"
        )

    # Rebuild observed information at the optimum for approximate standard errors.
    information = np.eye(count, dtype=np.float64) * prior_precision
    log_likelihood = -0.5 * float(np.dot(ratings, ratings)) * prior_precision
    games_total = 0
    for match in matches:
        left = index[match["player_a"]]
        right = index[match["player_b"]]
        games = match["wins_a"] + match["draws"] + match["wins_b"]
        observed = match["wins_a"] + 0.5 * match["draws"]
        probability = _probability(float(ratings[left] - ratings[right]))
        probability = min(max(probability, 1e-15), 1.0 - 1e-15)
        log_likelihood += observed * math.log(probability)
        log_likelihood += (games - observed) * math.log(1.0 - probability)
        weight = games * probability * (1.0 - probability)
        information[left, left] += weight
        information[right, right] += weight
        information[left, right] -= weight
        information[right, left] -= weight
        games_total += games
    covariance = np.linalg.inv(information)
    centering = np.eye(count) - np.ones((count, count)) / count
    centered_covariance = centering @ covariance @ centering
    elo = ratings * ELO_SCALE
    standard_errors = np.sqrt(np.maximum(np.diag(centered_covariance), 0.0)) * ELO_SCALE
    ranking = sorted(range(count), key=lambda item: (-float(elo[item]), players[item]))
    return {
        "method": "Bradley-Terry; draws count as half a win; Gaussian prior",
        "prior_sigma_elo": prior_sigma_elo,
        "games": games_total,
        "aggregate_matches": len(matches),
        "iterations": iterations,
        "converged": converged,
        "maximum_update": maximum_delta,
        "penalized_log_likelihood": log_likelihood,
        "players": [
            {
                "id": players[item],
                "rating_elo": float(elo[item]),
                "standard_error_elo": float(standard_errors[item]),
                "rank": rank + 1,
            }
            for rank, item in enumerate(ranking)
        ],
    }


def fit_ladder_file(source: Path | str, destination: Path | str) -> dict[str, Any]:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    players, matches, prior, pair_logs = _load_ladder_input(source)
    result = fit_bradley_terry(players, matches, prior)
    output = {
        "schema": LADDER_OUTPUT_SCHEMA,
        "created_at": utc_now(),
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "verified_arena_pair_logs": pair_logs,
        **result,
    }
    atomic_write_json(destination, output)
    return output
