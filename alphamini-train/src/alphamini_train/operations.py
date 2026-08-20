"""Read-only verification/reporting plus deliberately conservative garbage collection."""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .atomic import atomic_write_bytes, read_json, sha256_file
from .config import ResolvedConfig
from .cuda_runtime import runtime_environment
from .errors import IntegrityError
from .export import export_onnx
from .model import build_model
from .parity import verify_cross_runtime_parity
from .run import (
    RunRepository,
    git_identity,
    read_initial_budget_milestone,
    runtime_identity,
    utc_now,
    verify_budget_ledger,
)
from .schema import TensorCache, validate_collection_manifest


def doctor(config: ResolvedConfig, *, worktree: Path, production: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, status: str, detail: Any) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    version = sys.version_info
    if version[:2] == (3, 12):
        record("python", "pass", sys.version.split()[0])
    else:
        record(
            "python", "fail" if production else "warn", f"need 3.12, found {sys.version.split()[0]}"
        )

    required = ["numpy", "msgpack", "zstandard", "torch", "onnx", "onnxruntime"]
    for module in required:
        try:
            imported = __import__(module)
            record(module, "pass", str(getattr(imported, "__version__", "installed")))
        except ImportError:
            record(module, "fail" if production else "warn", "not installed")
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        detail = (
            f"{torch.cuda.get_device_name(0)}; torch CUDA {torch.version.cuda}"
            if cuda_available
            else "PyTorch cannot access CUDA"
        )
        record(
            "training_cuda",
            "pass" if cuda_available else ("fail" if production else "warn"),
            detail,
        )
    except ImportError:
        record("training_cuda", "fail" if production else "warn", "PyTorch is unavailable")

    for key in ("collect_command", "materialize_command"):
        command = config.values["operations"].get(key, [])
        if not command:
            record(key, "fail" if production else "warn", "not configured")
        elif shutil.which(command[0]) is None:
            record(key, "fail", f"executable not found: {command[0]}")
        else:
            record(key, "pass", " ".join(command))
    collect_command = config.values["operations"].get("collect_command", [])
    expected_device = config.values["operations"].get("self_play_device")
    has_cuda_feature = any(
        part == "cuda" or part.endswith("=cuda") or "cuda," in part or ",cuda" in part
        for part in collect_command
    )
    record(
        "self_play_provider",
        "pass"
        if (expected_device != "cuda" or has_cuda_feature)
        else ("fail" if production else "warn"),
        (
            f"{expected_device} requested; CUDA feature selected"
            if has_cuda_feature
            else f"{expected_device} requested; collect command does not select CUDA"
        ),
    )
    cuda_runtime_environment: dict[str, str] | None = None
    if expected_device == "cuda":
        try:
            cuda_runtime_environment, cuda_runtime_identity = runtime_environment(worktree)
            record("rust_cuda_runtime", "pass", cuda_runtime_identity)
        except Exception as error:
            record(
                "rust_cuda_runtime",
                "fail" if production else "warn",
                str(error),
            )

    identity = git_identity(worktree)
    if identity["commit"] is None:
        record("git", "fail", "worktree commit cannot be resolved")
    elif identity["tracked_dirty"] is not False:
        detail = (
            f"worktree dirty at {identity['commit']}"
            if identity["tracked_dirty"] is True
            else f"could not determine worktree cleanliness at {identity['commit']}"
        )
        record("git", "fail" if production else "warn", detail)
    else:
        record("git", "pass", identity["commit"])

    try:
        with tempfile.TemporaryDirectory(prefix="alphamini-doctor-") as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            atomic_write_bytes(source, b"atomic-rename-probe\n")
            os.replace(source, destination)
            if destination.read_bytes() != b"atomic-rename-probe\n":
                raise OSError("probe content changed")
        record("atomic_rename", "pass", "same-filesystem rename and fsync succeeded")
    except OSError as error:
        record("atomic_rename", "fail", str(error))

    try:
        usage = shutil.disk_usage(worktree)
        free_gib = usage.free / 1024**3
        status = "pass" if free_gib >= (100 if production else 5) else "fail"
        record("disk", status, f"{free_gib:.1f} GiB free")
    except OSError as error:
        record("disk", "fail", str(error))
    horizon_confirmed = bool(config.values["training"]["horizon_confirmed"])
    record(
        "optimizer_horizon",
        "pass" if horizon_confirmed else ("fail" if production else "warn"),
        (
            f"frozen at {config.values['training']['frozen_horizon_steps']} successful steps"
            if horizon_confirmed
            else "placeholder: measure pilot, update v1.toml, and commit before Run 1"
        ),
    )

    if (
        production
        and horizon_confirmed
        and config.values["operations"].get("self_play_device") == "cuda"
    ):
        try:
            import torch

            with tempfile.TemporaryDirectory(prefix="alphamini-cuda-doctor-") as temporary:
                root = Path(temporary)
                torch.manual_seed(int(config.values["run"]["seed"]))
                model = build_model(config).cpu()
                model_path, manifest_path, _ = export_onnx(
                    model,
                    config,
                    root / "model",
                    cycle_id=0,
                    global_step=0,
                    parent_checkpoint_sha256="0" * 64,
                    seed=int(config.values["run"]["seed"]),
                )
                try:
                    evidence = verify_cross_runtime_parity(
                        model,
                        model_path,
                        manifest_path,
                        config,
                        worktree=worktree,
                        device="cuda",
                        cuda_device=0,
                        release=True,
                        rust_environment=cuda_runtime_environment,
                    )
                    record("cross_runtime_cuda_parity", "pass", evidence)
                except Exception as error:
                    record("cross_runtime_cuda_parity", "fail", str(error))

                try:
                    collection_dir = root / "collection"
                    collection_dir.mkdir()
                    collection_manifest = collection_dir / "collection.json"
                    command = [
                        *config.values["operations"]["collect_command"],
                        "--model",
                        str(model_path),
                        "--manifest",
                        str(manifest_path),
                        "--device",
                        "cuda",
                        "--run-dir",
                        str(root),
                        "--run-id",
                        "doctor-cuda-smoke",
                        "--cycle-id",
                        "0",
                        "--game-id-start",
                        "0",
                        "--output-dir",
                        str(collection_dir),
                        "--collection-manifest",
                        str(collection_manifest),
                        "--config-sha256",
                        config.config_hash,
                        "--games",
                        "1",
                        "--shard-games",
                        "1",
                        "--simulations",
                        "1",
                        "--batch-size",
                        "1",
                        "--seed",
                        "1",
                        "--max-plies",
                        "1",
                        "--dirichlet-alpha",
                        str(config.values["self_play"]["dirichlet_alpha"]),
                        "--dirichlet-epsilon",
                        str(config.values["self_play"]["dirichlet_epsilon"]),
                        "--sample-until-ply",
                        "0",
                        "--cpuct",
                        str(config.values["self_play"]["cpuct"]),
                        "--fpu-reduction",
                        str(config.values["self_play"]["fpu_reduction"]),
                    ]
                    result = subprocess.run(
                        command,
                        cwd=worktree,
                        capture_output=True,
                        text=True,
                        timeout=900,
                    )
                    if result.returncode != 0:
                        detail = (result.stderr or result.stdout).strip()[-2000:]
                        raise IntegrityError(
                            f"CUDA collector smoke exited {result.returncode}: {detail}"
                        )
                    validate_collection_manifest(collection_manifest, decode_shards=True)
                    record(
                        "rust_cuda_selfplay_smoke",
                        "pass",
                        "real Rust ORT CUDA collection completed",
                    )
                except Exception as error:
                    record("rust_cuda_selfplay_smoke", "fail", str(error))
        except Exception as error:
            # Export/model setup failed before either target-runtime check could start.
            record("cross_runtime_cuda_parity", "fail", str(error))
            record("rust_cuda_selfplay_smoke", "fail", str(error))

    failures = sum(check["status"] == "fail" for check in checks)
    warnings = sum(check["status"] == "warn" for check in checks)
    return {
        "schema": "alphamini.doctor-report.v1",
        "checked_at": utc_now(),
        "config_sha256": config.config_hash,
        "semantic_hash": config.semantic_hash,
        "production": production,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
    }


def _verify_artifact(repository: RunRepository, descriptor: dict[str, Any], name: str) -> None:
    path_value = descriptor.get("path")
    digest = descriptor.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise IntegrityError(f"invalid {name} descriptor")
    path = (repository.root / path_value).resolve()
    try:
        path.relative_to(repository.root)
    except ValueError as error:
        raise IntegrityError(f"{name} path escapes the run") from error
    if not path.is_file() or sha256_file(path) != digest:
        raise IntegrityError(f"{name} is missing or corrupt: {path}")


def verify_run(repository: RunRepository, *, deep: bool) -> dict[str, Any]:
    summary = repository.verify_state_chain(deep=deep)
    _, state = repository.effective()
    checked = verify_budget_ledger(repository, state)
    if state.get("current_checkpoint"):
        _verify_artifact(repository, state["current_checkpoint"], "current checkpoint")
        checked += 1
    if state.get("current_model"):
        _verify_artifact(repository, state["current_model"], "current ONNX model")
        manifest_path = repository.root / state["current_model"]["manifest_path"]
        if sha256_file(manifest_path) != state["current_model"].get("manifest_sha256"):
            raise IntegrityError("current model manifest checksum mismatch")
        manifest = read_json(manifest_path)
        exact_fields = {
            "schema",
            "encoder_schema",
            "action_schema",
            "onnx_opset",
            "input_name",
            "policy_output_name",
            "wdl_output_name",
            "input_planes",
            "policy_size",
            "wdl_size",
            "residual_channels",
            "residual_blocks",
            "cycle",
            "parent_checkpoint_sha256",
            "model_sha256",
        }
        if set(manifest) != exact_fields or manifest.get("schema") != "model-manifest-v1":
            raise IntegrityError("invalid current model manifest")
        if (
            manifest.get("encoder_schema") != "encoder-v1"
            or manifest.get("action_schema") != "policy-v1"
            or manifest.get("input_name") != "input"
            or manifest.get("policy_output_name") != "policy_logits"
            or manifest.get("wdl_output_name") != "wdl_logits"
            or manifest.get("model_sha256") != state["current_model"]["sha256"]
        ):
            raise IntegrityError("current model manifest contract mismatch")
        provenance_path = repository.root / state["current_model"]["provenance_path"]
        if sha256_file(provenance_path) != state["current_model"].get("provenance_sha256"):
            raise IntegrityError("model training provenance checksum mismatch")
        provenance = read_json(provenance_path)
        if provenance.get("model_sha256") != state["current_model"]["sha256"]:
            raise IntegrityError("training provenance references another model")
        if provenance.get("parity", {}).get("status") != "passed":
            raise IntegrityError("current model lacks passing ONNX parity")
        checked += 3
    collections: list[dict[str, Any]] = []
    if state.get("pending_collection"):
        collections.append(state["pending_collection"])
    collections.extend(
        cycle["collection"]
        for cycle in state.get("completed_cycles", [])
        if cycle.get("collection")
    )
    caches: list[dict[str, Any]] = list(state.get("replay_caches", []))
    if state.get("pending_tensor_cache"):
        caches.append(state["pending_tensor_cache"])
    if deep:
        for collection in collections:
            manifest_path = repository.root / collection["path"]
            if sha256_file(manifest_path) != collection["sha256"]:
                raise IntegrityError(f"collection manifest checksum mismatch: {manifest_path}")
            validate_collection_manifest(manifest_path, decode_shards=True)
            checked += 1
        for cache in caches:
            manifest_path = repository.root / cache["path"]
            if sha256_file(manifest_path) != cache["sha256"]:
                raise IntegrityError(f"tensor manifest checksum mismatch: {manifest_path}")
            TensorCache(manifest_path, verify_hashes=True).arrays()
            checked += 1
    summary["artifacts_checked"] = checked
    summary["deep"] = deep
    summary["verified_at"] = utc_now()
    return summary


BENCHMARK_REPORT_SCHEMA = "alphamini.production-benchmark-report.v1"
_BENCHMARK_CONTRACT = {
    "model.input_planes": 22,
    "model.action_size": 4672,
    "model.channels": 64,
    "model.residual_blocks": 6,
    "model.se_hidden": 8,
    "self_play.games_per_cycle": 1024,
    "self_play.simulations": 128,
    "self_play.batch_size": 256,
    "self_play.max_plies": 512,
    "training.batch_size": 512,
    "training.sample_ratio": 2.0,
    "training.device": "cuda",
    "operations.self_play_device": "cuda",
}


def _nested_value(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _run_relative_path(repository: RunRepository, relative: Any, name: str) -> Path:
    if not isinstance(relative, str):
        raise IntegrityError(f"benchmark {name} path is invalid")
    path = (repository.root / relative).resolve()
    try:
        path.relative_to(repository.root)
    except ValueError as error:
        raise IntegrityError(f"benchmark {name} path escapes the run") from error
    return path


def _invocation_record(
    repository: RunRepository, descriptor: dict[str, Any], name: str
) -> tuple[dict[str, Any], Path]:
    path = _run_relative_path(repository, descriptor.get("invocation_path"), name)
    invocation = read_json(path)
    if invocation.get("schema") != "alphamini.external-invocation.v1":
        raise IntegrityError(f"benchmark {name} invocation has the wrong schema")
    return invocation, path


def _collection_performance(
    repository: RunRepository, descriptor: dict[str, Any]
) -> dict[str, Any]:
    invocation, invocation_path = _invocation_record(repository, descriptor, "collection")
    log_path = _run_relative_path(
        repository,
        str(invocation_path.parent.relative_to(repository.root) / invocation.get("log_path", "")),
        "collection log",
    )
    events: list[dict[str, Any]] = []
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            try:
                event = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("event") == "self_play_shard_complete":
                events.append(event)
    except OSError as error:
        raise IntegrityError(f"cannot read benchmark collection log: {error}") from error
    if not events:
        raise IntegrityError("collection log has no self_play_shard_complete telemetry")

    capacities = {int(event["batch_capacity"]) for event in events}
    if len(capacities) != 1:
        raise IntegrityError("collection shards disagree on batch capacity")
    capacity = capacities.pop()
    games = sum(int(event["games"]) for event in events)
    positions = sum(int(event["positions"]) for event in events)
    simulations = sum(int(event["completed_simulations"]) for event in events)
    evaluations = sum(int(event["neural_evaluations"]) for event in events)
    batches = sum(int(event["inference_batches"]) for event in events)
    search_seconds = sum(float(event["elapsed_seconds"]) for event in events)
    inference_seconds = sum(float(event["inference_seconds"]) for event in events)
    if (
        min(games, positions, simulations, evaluations, batches) <= 0
        or search_seconds <= 0
        or inference_seconds <= 0
    ):
        raise IntegrityError("collection telemetry contains a non-positive counter")
    invocation_seconds = float(invocation.get("elapsed_seconds", 0.0))
    return {
        "invocation_status": invocation.get("status"),
        "invocation_return_code": invocation.get("return_code"),
        "invocation_seconds": invocation_seconds,
        "search_seconds": search_seconds,
        "inference_seconds": inference_seconds,
        "games": games,
        "positions": positions,
        "completed_simulations": simulations,
        "neural_evaluations": evaluations,
        "inference_batches": batches,
        "batch_capacity": capacity,
        "worker_count": int(events[0].get("worker_count", 0)),
        "maximum_batch": max(int(event["maximum_batch"]) for event in events),
        "mean_batch_fill": evaluations / (batches * capacity),
        "games_per_hour": games * 3600.0 / search_seconds,
        "positions_per_second": positions / search_seconds,
        "simulations_per_second": simulations / search_seconds,
        "evaluations_per_wall_second": evaluations / search_seconds,
        "evaluations_per_inference_second": evaluations / inference_seconds,
    }


def production_benchmark_report(repository: RunRepository) -> dict[str, Any]:
    """Verify and score a production-shaped disposable throughput run."""

    verification = verify_run(repository, deep=True)
    run_manifest = read_json(repository.root / "RUN.json")
    config = read_json(repository.root / "config.resolved.json")
    _, state = repository.effective()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, required: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "pass" if passed else "fail",
                "actual": actual,
                "required": required,
            }
        )

    check(
        "run.disposable",
        run_manifest.get("disposable") is True,
        run_manifest.get("disposable"),
        True,
    )
    check(
        "runtime.cuda_available",
        run_manifest.get("runtime", {}).get("cuda_available") is True,
        run_manifest.get("runtime", {}).get("cuda_available"),
        True,
    )
    for dotted, expected in _BENCHMARK_CONTRACT.items():
        actual = _nested_value(config, dotted)
        check(dotted, actual == expected, actual, expected)
    games_per_cycle = int(_nested_value(config, "self_play.games_per_cycle"))
    interruptions = len(state.get("interruptions", []))
    check("uninterrupted throughput run", interruptions == 0, interruptions, 0)
    check(
        "completed measured cycles",
        len(state.get("completed_cycles", [])) >= 2,
        len(state.get("completed_cycles", [])),
        ">= 2",
    )

    measured_cycles: list[dict[str, Any]] = []
    simulation_rates: list[float] = []
    total_cycle_seconds = 0.0
    total_successful_updates = 0
    for cycle in state.get("completed_cycles", []):
        cycle_id = int(cycle["cycle_id"])
        prefix = f"cycle_{cycle_id:06d}"
        collection = _collection_performance(repository, cycle["collection"])
        materialize, _ = _invocation_record(repository, cycle["tensor_cache"], "materialization")
        materialize_seconds = float(materialize.get("elapsed_seconds", 0.0))
        metrics = cycle.get("metrics", {})
        train_export_seconds = float(metrics.get("train_export_session_seconds", 0.0))
        successful_updates = int(cycle.get("successful_updates", 0))
        attempts = int(metrics.get("training_session_attempts", -1))
        session_successes = int(metrics.get("training_session_successful_updates", -1))
        overflows = int(metrics.get("training_session_amp_overflows", -1))
        overflow_fraction = overflows / attempts if attempts > 0 else None
        recorded_positions = int(cycle["collection"]["position_count"])
        expected_simulations = recorded_positions * 128
        expected_updates = max(
            1,
            math.ceil(
                recorded_positions
                * float(_nested_value(config, "training.sample_ratio"))
                / int(_nested_value(config, "training.batch_size"))
            ),
        )
        stage_seconds_valid = (
            collection["invocation_seconds"] > 0
            and materialize_seconds > 0
            and train_export_seconds > 0
        )
        cycle_seconds = (
            collection["invocation_seconds"] + materialize_seconds + train_export_seconds
            if stage_seconds_valid
            else None
        )
        parity_path = _run_relative_path(
            repository, cycle["model"].get("provenance_path"), "model provenance"
        )
        parity = read_json(parity_path).get("parity", {})

        check(
            f"{prefix}.collection invocation",
            collection["invocation_status"] == "completed"
            and collection["invocation_return_code"] == 0,
            {
                "status": collection["invocation_status"],
                "return_code": collection["invocation_return_code"],
            },
            {"status": "completed", "return_code": 0},
        )
        check(
            f"{prefix}.materialization invocation",
            materialize.get("status") == "completed" and materialize.get("return_code") == 0,
            {"status": materialize.get("status"), "return_code": materialize.get("return_code")},
            {"status": "completed", "return_code": 0},
        )
        check(
            f"{prefix}.exact simulation budget",
            collection["completed_simulations"] == expected_simulations,
            collection["completed_simulations"],
            expected_simulations,
        )
        check(
            f"{prefix}.collection counters",
            collection["positions"] == recorded_positions
            and collection["games"] == games_per_cycle
            and int(cycle["collection"]["game_count"]) == games_per_cycle
            and collection["batch_capacity"] == 256
            and collection["worker_count"] == 512,
            {
                "positions": collection["positions"],
                "games": collection["games"],
                "batch_capacity": collection["batch_capacity"],
                "worker_count": collection["worker_count"],
            },
            {
                "positions": recorded_positions,
                "games": games_per_cycle,
                "batch_capacity": 256,
                "worker_count": 512,
            },
        )
        check(
            f"{prefix}.batch capacity reached",
            collection["maximum_batch"] == 256,
            collection["maximum_batch"],
            256,
        )
        check(
            f"{prefix}.mean batch fill",
            collection["mean_batch_fill"] >= 0.65,
            collection["mean_batch_fill"],
            ">= 0.65 (one bounded 512-game terminal drain)",
        )
        check(
            f"{prefix}.simulation throughput",
            collection["simulations_per_second"] >= 30_000.0,
            collection["simulations_per_second"],
            ">= 30000 simulations/second",
        )
        check(
            f"{prefix}.training segment covers cycle",
            successful_updates == expected_updates
            and session_successes == successful_updates
            and attempts >= session_successes > 0,
            {"attempts": attempts, "successful": session_successes},
            {"successful": expected_updates, "attempts": ">= successful"},
        )
        check(
            f"{prefix}.AMP overflow fraction",
            overflow_fraction is not None
            and 0.0 <= overflow_fraction <= 0.05
            and attempts - session_successes == overflows,
            overflow_fraction,
            "<= 0.05 and exactly accounts for unsuccessful attempts",
        )
        validation_batches = metrics.get("validation_batches")
        check(
            f"{prefix}.validation populated",
            isinstance(validation_batches, (int, float)) and validation_batches >= 1,
            validation_batches,
            ">= 1 batch",
        )
        loss_names = ("policy_loss", "wdl_loss", "total_loss", "validation_total_loss")
        losses = {name: metrics.get(name) for name in loss_names}
        check(
            f"{prefix}.finite losses",
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in losses.values()
            ),
            losses,
            "all finite",
        )
        check(
            f"{prefix}.ONNX parity",
            parity.get("status") == "passed",
            parity.get("status"),
            "passed",
        )
        check(
            f"{prefix}.stage telemetry",
            stage_seconds_valid,
            {
                "collect_seconds": collection["invocation_seconds"],
                "materialize_seconds": materialize_seconds,
                "train_export_seconds": train_export_seconds,
            },
            "all positive",
        )

        simulation_rates.append(collection["simulations_per_second"])
        if cycle_seconds is not None:
            total_cycle_seconds += cycle_seconds
            total_successful_updates += successful_updates
        measured_cycles.append(
            {
                "cycle_id": cycle_id,
                "collection": collection,
                "materialize_seconds": materialize_seconds,
                "training": {
                    key: metrics.get(key)
                    for key in (
                        "training_session_seconds",
                        "training_session_attempts",
                        "training_session_successful_updates",
                        "training_session_amp_overflows",
                        "training_session_samples",
                        "training_session_updates_per_second",
                        "training_session_samples_per_second",
                        "validation_batches",
                        "validation_total_loss",
                    )
                },
                "export_seconds": metrics.get("export_session_seconds"),
                "train_export_seconds": train_export_seconds,
                "measured_cycle_seconds": cycle_seconds,
                "successful_updates_per_hour": (
                    successful_updates * 3600.0 / cycle_seconds
                    if cycle_seconds is not None
                    else None
                ),
                "parity": parity,
            }
        )

    if len(simulation_rates) >= 2:
        mean_rate = sum(simulation_rates) / len(simulation_rates)
        rate_spread = (max(simulation_rates) - min(simulation_rates)) / mean_rate
    else:
        rate_spread = None
    check(
        "steady simulation throughput",
        rate_spread is not None and rate_spread <= 0.10,
        rate_spread,
        "<= 0.10 spread",
    )

    failures = sum(item["status"] == "fail" for item in checks)
    projected_updates = (
        math.floor(total_successful_updates * 72.0 * 3600.0 / total_cycle_seconds)
        if failures == 0 and total_cycle_seconds > 0
        else None
    )
    return {
        "schema": BENCHMARK_REPORT_SCHEMA,
        "generated_at": utc_now(),
        "run_id": run_manifest["run_id"],
        "head": repository.head()[0],
        "automated_status": "passed" if failures == 0 else "failed",
        "failures": failures,
        "verification": verification,
        "hardware": {
            key: run_manifest.get("runtime", {}).get(key)
            for key in (
                "gpu",
                "gpu_memory_bytes",
                "nvidia_driver",
                "cuda_runtime",
                "torch",
                "onnxruntime",
            )
        },
        "checks": checks,
        "cycles": measured_cycles,
        "aggregate": {
            "measured_cycle_seconds": total_cycle_seconds,
            "successful_updates": total_successful_updates,
            "successful_updates_per_hour": (
                total_successful_updates * 3600.0 / total_cycle_seconds
                if total_cycle_seconds > 0
                else None
            ),
            "simulation_rate_spread": rate_spread,
            "naive_72h_successful_update_projection": projected_updates,
        },
        "horizon_freeze_ready": False,
        "manual_requirements_before_horizon_freeze": [
            (
                "Review an external nvidia-smi trace: investigate and explain median GPU "
                "utilization below 80%, require peak memory <= 90% of VRAM, and require "
                "no sustained thermal throttling."
            ),
            (
                "Complete and verify separate collection and training "
                "interruption/recovery rehearsals."
            ),
            "Run CPU serving batches 1/4/8 under the frozen nine-second serving budget.",
            (
                "Choose and document a conservative horizon from the measured whole-cycle "
                "update rate; this command never edits v1.toml."
            ),
        ],
    }


def reproduction_record(repository: RunRepository) -> dict[str, Any]:
    manifest = read_json(repository.root / "RUN.json")
    _, state = repository.effective()
    milestone = read_initial_budget_milestone(repository, state)
    run_dir = shlex.quote(str(repository.root))
    return {
        "schema": "alphamini.reproduction.v1",
        "run_id": manifest["run_id"],
        "config_sha256": manifest["config_sha256"],
        "semantic_hash": manifest["semantic_hash"],
        "disposable": manifest.get("disposable", False),
        "source_git": manifest["git"],
        "locks": manifest["locks"],
        "original_runtime": manifest["runtime"],
        "current_runtime": runtime_identity(),
        "head": repository.head()[0],
        "recovery": repository.recovery()[0],
        "cycle_id": state["cycle_id"],
        "global_step": state["global_step"],
        "active_used_seconds": state["active_used_seconds"],
        "active_budget_seconds": state["active_budget_seconds"],
        "initial_active_budget_seconds": state["initial_active_budget_seconds"],
        "initial_budget_milestone_sha256": state.get("initial_budget_milestone_sha256"),
        "initial_budget_milestone": milestone,
        "budget_extensions": list(state.get("budget_extensions", [])),
        "commands": {
            "verify": (
                "uv run --project alphamini-train alphamini-train verify "
                f"--run-dir {run_dir} --deep"
            ),
            "continue": (
                f"uv run --project alphamini-train alphamini-train resume --run-dir {run_dir}"
            ),
        },
    }


def render_report(repository: RunRepository) -> str:
    manifest = read_json(repository.root / "RUN.json")
    _, state = repository.effective()
    used = float(state["active_used_seconds"]) / 3600
    budget = float(state["active_budget_seconds"]) / 3600
    milestone_sha256 = state.get("initial_budget_milestone_sha256")
    milestone = read_initial_budget_milestone(repository, state)
    if milestone is None:
        milestone_status = "not reached"
    else:
        milestone_status = (
            f"`{milestone_sha256}` at cycle {milestone['cycle_id']}, "
            f"step {milestone['global_step']}, "
            f"{float(milestone['accounted_active_seconds']) / 3600:.3f} active hours"
        )
    lines = [
        f"# {manifest['name']}",
        "",
        "> This file is generated from the immutable run ledger. Unknown results are left unknown.",
        "",
        "## Lineage",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Source commit: `{manifest['git'].get('commit')}`",
        f"- Disposable/non-publishable: `{manifest.get('disposable', False)}`",
        f"- Worktree content: `{manifest['git'].get('worktree_sha256')}`",
        f"- Semantic configuration: `{manifest['semantic_hash']}`",
        f"- Parent: `{json.dumps(manifest.get('parent'), sort_keys=True)}`",
        "",
        "## Training status",
        "",
        f"- Phase: `{state['phase']}`",
        f"- Committed model cycle: {state['cycle_id']}",
        f"- Successful optimizer updates: {state['global_step']}",
        f"- Counted active time: {used:.3f} / {budget:.3f} hours",
        f"- Initial-budget milestone: {milestone_status}",
        f"- Budget extensions: {len(state.get('budget_extensions', []))}",
        f"- Completed self-play cycles: {len(state.get('completed_cycles', []))}",
        f"- Interrupted sessions recovered: {len(state.get('interruptions', []))}",
        "",
        "## Evaluation",
        "",
        "No arena or Elo result has been imported into this run ledger yet.",
        "",
    ]
    return "\n".join(lines)


def write_report(repository: RunRepository, destination: Path | None) -> Path:
    destination = destination or repository.root / "report.md"
    atomic_write_bytes(destination, render_report(repository).encode("utf-8"))
    return destination


def gc_candidates(repository: RunRepository) -> list[Path]:
    _, state = repository.effective()
    referenced_cache_dirs: set[Path] = set()
    caches = list(state.get("replay_caches", []))
    if state.get("pending_tensor_cache"):
        caches.append(state["pending_tensor_cache"])
    for cache in caches:
        referenced_cache_dirs.add((repository.root / cache["path"]).resolve().parent)
    candidates: set[Path] = set(repository.root.rglob("*.partial"))
    cache_root = repository.root / "cache"
    if cache_root.exists():
        for child in cache_root.iterdir():
            if child.is_dir() and child.resolve() not in referenced_cache_dirs:
                candidates.add(child)
    return sorted(candidates)


def apply_gc(repository: RunRepository, candidates: list[Path], *, backup_marker: Path) -> None:
    if repository.active_session_path.exists():
        raise IntegrityError("ACTIVE_SESSION exists; recover or finish the run before GC apply")
    marker = read_json(backup_marker)
    if marker.get("schema") != "alphamini.backup-verification.v1":
        raise IntegrityError("backup marker has the wrong schema")
    if marker.get("run_id") != read_json(repository.root / "RUN.json").get("run_id"):
        raise IntegrityError("backup marker belongs to another run")
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(repository.root)
        except ValueError as error:
            raise IntegrityError(f"GC candidate escapes the run: {resolved}") from error
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()
