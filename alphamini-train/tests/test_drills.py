from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphamini_train.atomic import sha256_file
from alphamini_train.cli import build_parser
from alphamini_train.config import load_config
from alphamini_train.data import game_is_validation
from alphamini_train.drills import (
    _has_periodic_training_recovery,
    _summarize_pair_log,
    run_cpu_serving_benchmark,
    run_recovery_drill,
)
from alphamini_train.errors import ConfigError, IntegrityError

from conftest import REPOSITORY

RECOVERY_CONFIG = REPOSITORY / "configs" / "alphamini" / "recovery-drill.toml"


def _pair_log(path: Path, *, deadline: bool = False) -> None:
    values = [
        {
            "schema": "alphamini-paired-evaluation-v1",
            "model_sha256": "a" * 64,
            "simulations": 10,
            "time_ms": 9000,
            "batch_size": 4,
            "inference_device": "onnx-cpu",
            "exploratory": True,
        }
    ]
    for index in range(2):
        values.append(
            {
                "schema": "alphamini-paired-opening-result-v1",
                "opening_id": f"opening-{index}",
                "metrics": {
                    "moves": 2,
                    "completed_simulations": 19 if deadline and index == 0 else 20,
                    "neural_evaluations": 16,
                    "inference_batches": 4,
                    "largest_batch": 4,
                    "elapsed_micros": 100_000,
                    "deadlines_reached": 1 if deadline and index == 0 else 0,
                },
            }
        )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_pair_log_summary_is_machine_readable_and_deadline_sensitive(tmp_path: Path) -> None:
    passed_path = tmp_path / "passed.jsonl"
    _pair_log(passed_path)
    passed = _summarize_pair_log(
        passed_path,
        model_sha256="a" * 64,
        simulations=10,
        time_ms=9000,
        batch_size=4,
        opening_pairs=2,
    )
    assert passed["passed"] is True
    assert passed["moves"] == 4
    assert passed["completed_simulations"] == 40
    assert passed["mean_search_latency_ms"] == 50.0
    assert passed["mean_batch_fill"] == 1.0

    failed_path = tmp_path / "deadline.jsonl"
    _pair_log(failed_path, deadline=True)
    failed = _summarize_pair_log(
        failed_path,
        model_sha256="a" * 64,
        simulations=10,
        time_ms=9000,
        batch_size=4,
        opening_pairs=2,
    )
    assert failed["passed"] is False
    assert failed["deadlines_reached"] == 1


def test_pair_log_summary_rejects_torn_evidence(tmp_path: Path) -> None:
    path = tmp_path / "torn.jsonl"
    _pair_log(path)
    path.write_text(path.read_text().rsplit("\n", 2)[0] + "\n")
    with pytest.raises(IntegrityError, match="incomplete"):
        _summarize_pair_log(
            path,
            model_sha256="a" * 64,
            simulations=10,
            time_ms=9000,
            batch_size=4,
            opening_pairs=2,
        )


def test_cpu_benchmark_gates_on_frozen_batch_eight_and_keeps_diagnostics(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps({"model_sha256": sha256_file(model)}))
    openings = tmp_path / "openings.json"
    openings.write_text("{}")
    fake_arena = tmp_path / "arena"
    fake_arena.write_text(
        """#!/usr/bin/env python3
import json, sys
def value(name):
    return sys.argv[sys.argv.index(name) + 1]
manifest = json.load(open(value('--alphamini-manifest')))
batch = int(value('--alphamini-batch-size'))
simulations = int(value('--alphamini-simulations'))
time_ms = int(value('--alphamini-time-ms'))
games = int(value('--games'))
path = value('--results')
header = {
    'schema': 'alphamini-paired-evaluation-v1',
    'model_sha256': manifest['model_sha256'],
    'simulations': simulations,
    'time_ms': time_ms,
    'batch_size': batch,
    'inference_device': 'onnx-cpu',
    'exploratory': True,
}
with open(path, 'x') as output:
    output.write(json.dumps(header) + '\\n')
    for index in range(games):
        output.write(json.dumps({
            'schema': 'alphamini-paired-opening-result-v1',
            'opening_id': f'opening-{index}',
            'metrics': {
                'moves': 2,
                'completed_simulations': simulations * 2 if batch != 1 else simulations,
                'neural_evaluations': batch * 2,
                'inference_batches': 2,
                'largest_batch': batch,
                'elapsed_micros': 1000,
                'deadlines_reached': 0 if batch != 1 else 2,
            },
        }) + '\\n')
"""
    )
    fake_arena.chmod(0o755)

    report = run_cpu_serving_benchmark(
        arena=fake_arena,
        model=model,
        manifest=manifest,
        openings=openings,
        output_dir=tmp_path / "benchmark",
        worktree=REPOSITORY,
        simulations=10,
        time_ms=9000,
        opening_pairs=1,
    )
    assert report["passed"] is True
    assert report["production_batch_size"] == 8
    assert report["diagnostic_all_batches_passed"] is False
    assert report["runs"][0]["passed"] is False
    assert report["runs"][2]["passed"] is True
    assert [run["batch_size"] for run in report["runs"]] == [1, 4, 8]
    assert json.loads((tmp_path / "benchmark" / "summary.json").read_text())["status"] == "passed"


def test_recovery_drill_config_is_bounded_and_has_a_checkpoint_window() -> None:
    config = load_config(RECOVERY_CONFIG)
    assert config.values["run"]["disposable"] is True
    assert config.values["run"]["active_budget_hours"] <= 0.1
    assert config.values["self_play"]["games_per_cycle"] == 24
    assert config.values["training"]["checkpoint_every_steps"] == 25
    assert config.values["training"]["sample_ratio"] == 1.0
    assert any(
        game_is_validation(game_id, 1, 0.05)
        for game_id in range(config.values["self_play"]["games_per_cycle"])
    )


def test_training_interruption_requires_a_periodic_not_final_checkpoint() -> None:
    periodic = {
        "phase": "training",
        "cycle_step": 25,
        "target_cycle_steps": 100,
        "recovery_checkpoint": {"path": "recovery.pt"},
    }
    assert _has_periodic_training_recovery(periodic) is True
    assert _has_periodic_training_recovery({**periodic, "cycle_step": 100}) is False
    assert _has_periodic_training_recovery({**periodic, "recovery_checkpoint": None}) is False


def test_training_drill_requires_an_uninterrupted_control(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="control-run-dir"):
        run_recovery_drill(
            config_path=RECOVERY_CONFIG,
            run_dir=tmp_path / "run",
            evidence_path=tmp_path / "evidence.json",
            phase="training",
            worktree=REPOSITORY,
            control_run_dir=None,
            timeout_seconds=1,
        )


def test_cli_exposes_bounded_drill_and_cpu_benchmark() -> None:
    parser = build_parser()
    recovery = parser.parse_args(
        [
            "drill-recovery",
            "--config",
            str(RECOVERY_CONFIG),
            "--run-dir",
            "run",
            "--evidence",
            "evidence.json",
            "--phase",
            "collection",
        ]
    )
    assert recovery.phase == "collection"
    benchmark = parser.parse_args(
        [
            "benchmark-cpu",
            "--arena",
            "arena",
            "--model",
            "model.onnx",
            "--manifest",
            "model.json",
            "--openings",
            "openings.json",
            "--output-dir",
            "output",
        ]
    )
    assert benchmark.simulations == 10_000
    assert benchmark.time_ms == 9_000
