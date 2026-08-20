from __future__ import annotations

import json
from pathlib import Path

from alphamini_train.atomic import atomic_write_json, read_json
from alphamini_train.config import load_config
from alphamini_train.operations import production_benchmark_report
from alphamini_train.run import RunRepository

from conftest import REPOSITORY


BENCHMARK_CONFIG = REPOSITORY / "configs" / "alphamini" / "production-benchmark.toml"


def _write_cycle_files(run_dir: Path, cycle_id: int, *, search_seconds: float) -> dict:
    source_cycle = cycle_id - 1
    collection_dir = run_dir / "cycles" / f"cycle-{source_cycle:06d}" / "collection"
    collection_dir.mkdir(parents=True)
    event = {
        "event": "self_play_shard_complete",
        "batch_capacity": 256,
        "worker_count": 512,
        "games": 1024,
        "positions": 12800,
        "completed_simulations": 12800 * 128,
        "neural_evaluations": 1638400,
        "inference_batches": 8000,
        "elapsed_seconds": search_seconds,
        "inference_seconds": search_seconds * 0.8,
        "maximum_batch": 256,
    }
    (collection_dir / "collect.log").write_text(
        "cargo output\n" + json.dumps(event, sort_keys=True) + "\n"
    )
    atomic_write_json(
        collection_dir / "collect-command.json",
        {
            "schema": "alphamini.external-invocation.v1",
            "status": "completed",
            "return_code": 0,
            "elapsed_seconds": search_seconds + 10.0,
            "log_path": "collect.log",
        },
    )

    cache_dir = run_dir / "cache" / f"cycle-{source_cycle:06d}"
    cache_dir.mkdir(parents=True)
    atomic_write_json(
        cache_dir / "materialize-command.json",
        {
            "schema": "alphamini.external-invocation.v1",
            "status": "completed",
            "return_code": 0,
            "elapsed_seconds": 0.5,
            "log_path": "materialize.log",
        },
    )

    model_dir = run_dir / "artifacts" / "models" / f"cycle-{cycle_id:06d}"
    model_dir.mkdir(parents=True)
    atomic_write_json(
        model_dir / "model.training.json",
        {
            "schema": "alphamini.training-model-provenance.v1",
            "parity": {"status": "passed", "max_abs_policy": 1e-6, "max_abs_wdl": 1e-7},
        },
    )
    return {
        "cycle_id": cycle_id,
        "successful_updates": 50,
        "collection": {
            "path": f"cycles/cycle-{source_cycle:06d}/collection/collection.json",
            "sha256": "1" * 64,
            "game_count": 1024,
            "position_count": 12800,
            "invocation_path": (
                f"cycles/cycle-{source_cycle:06d}/collection/collect-command.json"
            ),
        },
        "tensor_cache": {
            "path": f"cache/cycle-{source_cycle:06d}/tensors.json",
            "sha256": "2" * 64,
            "invocation_path": f"cache/cycle-{source_cycle:06d}/materialize-command.json",
        },
        "model": {
            "provenance_path": (
                f"artifacts/models/cycle-{cycle_id:06d}/model.training.json"
            )
        },
        "metrics": {
            "policy_loss": 8.0,
            "wdl_loss": 1.0,
            "total_loss": 9.0,
            "training_session_seconds": 100.0,
            "training_session_attempts": 51,
            "training_session_successful_updates": 50,
            "training_session_amp_overflows": 1,
            "training_session_samples": 51 * 512,
            "training_session_updates_per_second": 0.5,
            "training_session_samples_per_second": 261.12,
            "validation_batches": 4.0,
            "validation_total_loss": 3.0,
            "export_session_seconds": 5.0,
            "train_export_session_seconds": 120.0,
        },
    }


def test_production_benchmark_report_scores_two_stable_cycles(
    tmp_path: Path, monkeypatch
) -> None:
    repository = RunRepository.create(
        tmp_path / "run", load_config(BENCHMARK_CONFIG), worktree=REPOSITORY
    )
    manifest = read_json(repository.root / "RUN.json")
    manifest["runtime"]["cuda_available"] = True
    manifest["runtime"]["gpu"] = "NVIDIA GeForce RTX 3070"
    atomic_write_json(repository.root / "RUN.json", manifest)

    cycles = [
        _write_cycle_files(repository.root, 1, search_seconds=50.0),
        _write_cycle_files(repository.root, 2, search_seconds=52.0),
    ]
    _, state = repository.head()
    state["phase"] = "ready_collect"
    state["cycle_id"] = 2
    state["global_step"] = 100
    state["completed_cycles"] = cycles
    repository.commit_head(state)
    monkeypatch.setattr(
        "alphamini_train.operations.verify_run",
        lambda _repository, *, deep: {"deep": deep, "state_objects": 2},
    )

    report = production_benchmark_report(repository)

    assert report["schema"] == "alphamini.production-benchmark-report.v1"
    assert report["automated_status"] == "passed"
    assert report["failures"] == 0
    assert len(report["cycles"]) == 2
    assert report["cycles"][0]["collection"]["mean_batch_fill"] == 0.8
    assert report["cycles"][0]["collection"]["worker_count"] == 512
    assert report["aggregate"]["naive_72h_successful_update_projection"] > 0
    assert report["horizon_freeze_ready"] is False


def test_production_benchmark_matches_confirmed_v1_shape() -> None:
    benchmark = load_config(BENCHMARK_CONFIG)
    v1 = load_config(REPOSITORY / "configs" / "alphamini" / "v1.toml")

    assert benchmark.values["training"]["horizon_confirmed"] is True
    assert v1.values["training"]["horizon_confirmed"] is True
    assert v1.values["training"]["frozen_horizon_steps"] == 180000
    for key in ("channels", "residual_blocks", "se_hidden"):
        assert benchmark.values["model"][key] == v1.values["model"][key]
    assert benchmark.values["self_play"]["simulations"] == 128
    assert benchmark.values["self_play"]["batch_size"] == 256
    assert benchmark.values["self_play"]["games_per_cycle"] == 1024
    assert benchmark.values["training"]["batch_size"] == 512
