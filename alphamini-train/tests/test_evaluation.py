from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphamini_train.atomic import sha256_file
from alphamini_train.errors import IntegrityError
from alphamini_train.evaluation import fit_ladder_file, load_ladder_input


def ladder(matches: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "alphamini.ladder-input.v1",
        "prior_sigma_elo": 800.0,
        "matches": matches,
    }


def test_bradley_terry_ladder_is_ordered_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "matches.json"
    source.write_text(
        json.dumps(
            ladder(
                [
                    {"player_a": "m2", "player_b": "m1", "wins_a": 60, "draws": 20, "wins_b": 20},
                    {"player_a": "m1", "player_b": "m0", "wins_a": 55, "draws": 30, "wins_b": 15},
                    {"player_a": "m2", "player_b": "m0", "wins_a": 70, "draws": 20, "wins_b": 10},
                ]
            )
        )
    )
    first = fit_ladder_file(source, tmp_path / "first.json")
    second = fit_ladder_file(source, tmp_path / "second.json")
    ratings = {player["id"]: player["rating_elo"] for player in first["players"]}
    assert ratings["m2"] > ratings["m1"] > ratings["m0"]
    assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-9)
    for field in ("players", "iterations", "penalized_log_likelihood"):
        assert first[field] == second[field]


def test_ladder_rejects_disconnected_match_graph(tmp_path: Path) -> None:
    source = tmp_path / "matches.json"
    source.write_text(
        json.dumps(
            ladder(
                [
                    {"player_a": "a", "player_b": "b", "wins_a": 1, "draws": 0, "wins_b": 1},
                    {"player_a": "c", "player_b": "d", "wins_a": 1, "draws": 0, "wins_b": 1},
                ]
            )
        )
    )
    with pytest.raises(IntegrityError, match="disconnected"):
        load_ladder_input(source)


def _arena_header(model_a: str, model_b: str) -> dict[str, object]:
    return {
        "schema": "alphamini-paired-evaluation-v1",
        "engine_a": f"AlphaMini-{model_a[:12]}",
        "engine_b": f"AlphaMini-{model_b[:12]}",
        "model_sha256": model_a,
        "opponent_model_sha256": model_b,
        "opening_suite_sha256": "1" * 64,
        "opening_ids": ["opening-0001"],
        "depth": 3,
        "seed": 1,
        "max_plies": 1000,
        "simulations": 128,
        "time_ms": 60000,
        "batch_size": 8,
        "cpuct_ppm": 1500000,
        "fpu_reduction_ppm": 250000,
        "bootstrap_samples": 20000,
        "required_lower_score_ppm": 500000,
        "minimax_v1_move_digest": 123,
        "evaluation_binary_sha256": "2" * 64,
        "target": "x86_64-linux",
        "inference_device": "onnx-cpu",
        "exploratory": True,
    }


def _won_pair() -> dict[str, object]:
    metrics = {
        "moves": 80,
        "completed_simulations": 10240,
        "neural_evaluations": 9000,
        "inference_batches": 2000,
        "largest_batch": 8,
        "elapsed_micros": 1000000,
        "deadlines_reached": 0,
    }
    return {
        "schema": "alphamini-paired-opening-result-v1",
        "opening_id": "opening-0001",
        "engine_a_as_white": {"winner": "white", "termination": "checkmate", "plies": 41},
        "engine_a_as_black": {"winner": "black", "termination": "checkmate", "plies": 42},
        "metrics": metrics,
    }


def _pair_log(path: Path, model_a: str, model_b: str) -> None:
    path.write_text(
        "\n".join(
            json.dumps(value, sort_keys=True)
            for value in (_arena_header(model_a, model_b), _won_pair())
        )
        + "\n"
    )


def test_ladder_consumes_hash_bound_dual_model_arena_logs(tmp_path: Path) -> None:
    model_0 = "a" * 64
    model_1 = "b" * 64
    model_2 = "c" * 64
    first = tmp_path / "m2-v-m1.jsonl"
    second = tmp_path / "m1-v-m0.jsonl"
    _pair_log(first, model_2, model_1)
    _pair_log(second, model_1, model_0)
    source = tmp_path / "arena-ladder.json"
    source.write_text(
        json.dumps(
            {
                "schema": "alphamini.arena-ladder-input.v1",
                "prior_sigma_elo": 800.0,
                "pair_logs": [
                    {
                        "player_a": "cycle-20",
                        "player_b": "cycle-10",
                        "model_a_sha256": model_2,
                        "model_b_sha256": model_1,
                        "path": first.name,
                        "sha256": sha256_file(first),
                    },
                    {
                        "player_a": "cycle-10",
                        "player_b": "cycle-01",
                        "model_a_sha256": model_1,
                        "model_b_sha256": model_0,
                        "path": second.name,
                        "sha256": sha256_file(second),
                    },
                ],
            }
        )
    )

    result = fit_ladder_file(source, tmp_path / "ladder.json")

    ratings = {player["id"]: player["rating_elo"] for player in result["players"]}
    assert ratings["cycle-20"] > ratings["cycle-10"] > ratings["cycle-01"]
    assert result["games"] == 4
    assert len(result["verified_arena_pair_logs"]) == 2


def test_arena_ladder_rejects_log_hash_or_identity_drift(tmp_path: Path) -> None:
    model_a = "a" * 64
    model_b = "b" * 64
    pair_log = tmp_path / "pair.jsonl"
    _pair_log(pair_log, model_a, model_b)
    source = tmp_path / "arena-ladder.json"
    descriptor = {
        "player_a": "a",
        "player_b": "b",
        "model_a_sha256": model_a,
        "model_b_sha256": model_b,
        "path": pair_log.name,
        "sha256": "0" * 64,
    }
    source.write_text(
        json.dumps(
            {
                "schema": "alphamini.arena-ladder-input.v1",
                "prior_sigma_elo": 800.0,
                "pair_logs": [descriptor],
            }
        )
    )
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        load_ladder_input(source)

    descriptor["sha256"] = sha256_file(pair_log)
    descriptor["model_b_sha256"] = "c" * 64
    source.write_text(
        json.dumps(
            {
                "schema": "alphamini.arena-ladder-input.v1",
                "prior_sigma_elo": 800.0,
                "pair_logs": [descriptor],
            }
        )
    )
    with pytest.raises(IntegrityError, match="model B hash"):
        load_ladder_input(source)
