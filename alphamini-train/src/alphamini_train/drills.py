"""Bounded, disposable operational drills and CPU search-budget evidence."""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .atomic import atomic_write_bytes, atomic_write_json, read_json, sha256_file
from .config import load_config
from .errors import ConfigError, DependencyUnavailable, IntegrityError
from .operations import verify_run
from .run import (
    RunRepository,
    git_identity,
    runtime_fingerprint,
    runtime_identity,
    utc_now,
)

RECOVERY_DRILL_SCHEMA = "alphamini.recovery-drill.v1"
CPU_BENCHMARK_SCHEMA = "alphamini.cpu-serving-benchmark.v1"
CPU_BATCH_SIZES = (1, 4, 8)
PRODUCTION_CPU_BATCH_SIZE = 8
DETERMINISTIC_METRIC_FIELDS = (
    "policy_loss",
    "wdl_loss",
    "total_loss",
    "validation_policy_loss",
    "validation_wdl_loss",
    "validation_total_loss",
    "validation_batches",
    "learning_rate",
)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_ignored_or_external(worktree: Path, path: Path, label: str) -> None:
    """Prevent drill outputs from invalidating their own frozen worktree identity."""

    resolved = path.resolve()
    if not _within(resolved, worktree.resolve()):
        return
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "--", str(resolved)],
        cwd=worktree,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ConfigError(
            f"{label} must be outside the worktree or ignored by Git so drill writes do not "
            "change the frozen source digest"
        )


def _require_new_path(path: Path, label: str) -> None:
    if path.exists():
        if path.is_dir() and not any(path.iterdir()):
            return
        raise ConfigError(f"{label} already exists and is not an empty directory: {path}")


def _child_cli(worktree: Path, *arguments: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "alphamini_train",
        "--worktree",
        str(worktree),
        *arguments,
    ]


def _write_process_output(
    directory: Path,
    label: str,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, Any]:
    stdout_path = directory / f"{label}.stdout.log"
    stderr_path = directory / f"{label}.stderr.log"
    atomic_write_bytes(stdout_path, stdout)
    atomic_write_bytes(stderr_path, stderr)
    return {
        "stdout": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
    }


def _run_cli(
    worktree: Path,
    evidence_directory: Path,
    label: str,
    arguments: Sequence[str],
    *,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any]]:
    started = time.monotonic()
    completed = subprocess.run(
        _child_cli(worktree, *arguments),
        cwd=worktree,
        check=False,
        capture_output=True,
        timeout=timeout_seconds,
    )
    record = {
        "command": _child_cli(worktree, *arguments),
        "return_code": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        **_write_process_output(
            evidence_directory, label, completed.stdout, completed.stderr
        ),
    }
    return completed, record


def _has_periodic_training_recovery(state: Mapping[str, Any]) -> bool:
    recovery = state.get("recovery_checkpoint")
    cycle_step = state.get("cycle_step")
    target_steps = state.get("target_cycle_steps")
    return (
        state.get("phase") == "training"
        and isinstance(recovery, dict)
        and isinstance(cycle_step, int)
        and not isinstance(cycle_step, bool)
        and isinstance(target_steps, int)
        and not isinstance(target_steps, bool)
        and 0 < cycle_step < target_steps
    )


def _wait_for_interruption_point(
    process: subprocess.Popen[bytes],
    run_dir: Path,
    phase: str,
    *,
    timeout_seconds: float,
    collection_settle_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise IntegrityError(
                f"drill process exited {return_code} before reaching the {phase} interruption point"
            )
        if phase == "collection":
            invocation = run_dir / "cycles" / "cycle-000000" / "collection" / "collect-command.json"
            if invocation.is_file():
                value = read_json(invocation)
                if value.get("status") == "running":
                    settled_at = time.monotonic() + collection_settle_seconds
                    while time.monotonic() < settled_at:
                        if process.poll() is not None:
                            raise IntegrityError(
                                "collector completed before the controlled signal; "
                                "use the recovery-drill config"
                            )
                        time.sleep(0.02)
                    value = read_json(invocation)
                    if value.get("status") != "running":
                        raise IntegrityError(
                            "collector completed before the controlled signal; "
                            "use the recovery-drill config"
                        )
                    return {
                        "phase": phase,
                        "invocation_path": str(invocation),
                        "invocation_started_at": value.get("started_at"),
                    }
        else:
            if (run_dir / "RUN.json").is_file():
                repository = RunRepository(run_dir)
                try:
                    recovery_sha256, state = repository.recovery()
                except IntegrityError:
                    # Atomic pointers are never torn, but the surrounding directory may
                    # still be appearing when the poll first observes RUN.json.
                    pass
                else:
                    if _has_periodic_training_recovery(state):
                        recovery = state["recovery_checkpoint"]
                        return {
                            "phase": phase,
                            "recovery_state_sha256": recovery_sha256,
                            "cycle_step": state["cycle_step"],
                            "target_cycle_steps": state["target_cycle_steps"],
                            "global_step": state.get("global_step"),
                            "recovery_checkpoint": recovery,
                        }
        time.sleep(0.02)
    raise IntegrityError(f"timed out waiting for the {phase} interruption point")


def _completed_cycle(repository: RunRepository) -> tuple[dict[str, Any], dict[str, Any]]:
    _, state = repository.head()
    completed = state.get("completed_cycles", [])
    if state.get("phase") != "ready_collect" or len(completed) != 1:
        raise IntegrityError("recovery drill did not finish exactly one promoted cycle")
    return state, completed[0]


def _collection_payload_identity(
    repository: RunRepository, cycle: dict[str, Any]
) -> list[dict[str, Any]]:
    manifest = read_json(repository.root / cycle["collection"]["path"])
    return [
        {
            "sha256": descriptor["sha256"],
            "first_game_id": descriptor["first_game_id"],
            "last_game_id": descriptor["last_game_id"],
            "game_count": descriptor["game_count"],
            "position_count": descriptor["position_count"],
        }
        for descriptor in manifest["shards"]
    ]


def _tensor_payload_identity(repository: RunRepository, cycle: dict[str, Any]) -> dict[str, str]:
    manifest = read_json(repository.root / cycle["tensor_cache"]["path"])
    return {
        name: manifest[name]["sha256"]
        for name in (
            "inputs",
            "policy_offsets",
            "policy_indices",
            "policy_values",
            "wdl",
            "game_ids",
        )
    }


def _values_equal(left: Any, right: Any) -> bool:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise DependencyUnavailable("checkpoint comparison requires NumPy and PyTorch") from error

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_values_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if (
        isinstance(left, float)
        and isinstance(right, float)
        and math.isnan(left)
        and math.isnan(right)
    ):
        return True
    return left == right


def _checkpoint_training_state_equal(left_path: Path, right_path: Path) -> dict[str, bool]:
    try:
        import torch
    except ImportError as error:
        raise DependencyUnavailable("checkpoint comparison requires PyTorch") from error

    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    return {
        field: _values_equal(left.get(field), right.get(field))
        for field in ("model", "optimizer", "scaler", "rng", "training_state")
    }


def compare_recovery_run(control_dir: Path, drill_dir: Path) -> dict[str, Any]:
    control, _ = RunRepository.open(control_dir)
    drill, _ = RunRepository.open(drill_dir)
    verify_run(control, deep=True)
    verify_run(drill, deep=True)
    control_manifest = read_json(control.root / "RUN.json")
    drill_manifest = read_json(drill.root / "RUN.json")
    if control_manifest["config_sha256"] != drill_manifest["config_sha256"]:
        raise IntegrityError("control and recovery drill use different frozen configurations")
    if control_manifest["git"] != drill_manifest["git"]:
        raise IntegrityError("control and recovery drill use different frozen source identities")
    if control_manifest["locks"] != drill_manifest["locks"]:
        raise IntegrityError("control and recovery drill use different dependency locks")
    if runtime_fingerprint(control_manifest["runtime"]) != runtime_fingerprint(
        drill_manifest["runtime"]
    ):
        raise IntegrityError("control and recovery drill use different runtime fingerprints")
    _, control_cycle = _completed_cycle(control)
    _, drill_cycle = _completed_cycle(drill)
    checkpoint_fields = _checkpoint_training_state_equal(
        control.root / control_cycle["checkpoint"]["path"],
        drill.root / drill_cycle["checkpoint"]["path"],
    )
    deterministic_metrics_equal = all(
        _values_equal(
            control_cycle["metrics"].get(field), drill_cycle["metrics"].get(field)
        )
        for field in DETERMINISTIC_METRIC_FIELDS
    )
    result = {
        "raw_shards_equal": _collection_payload_identity(control, control_cycle)
        == _collection_payload_identity(drill, drill_cycle),
        "tensor_payloads_equal": _tensor_payload_identity(control, control_cycle)
        == _tensor_payload_identity(drill, drill_cycle),
        "checkpoint_fields_equal": checkpoint_fields,
        # Wall-time, throughput, process-segment, and overflow-attempt diagnostics
        # intentionally differ after a restart. The cumulative losses and frozen
        # schedule result must not.
        "deterministic_metrics_equal": deterministic_metrics_equal,
        "control_session_metrics": {
            key: value
            for key, value in control_cycle["metrics"].items()
            if key not in DETERMINISTIC_METRIC_FIELDS
        },
        "drill_session_metrics": {
            key: value
            for key, value in drill_cycle["metrics"].items()
            if key not in DETERMINISTIC_METRIC_FIELDS
        },
        "onnx_model_sha256_equal": control_cycle["model_sha256"]
        == drill_cycle["model_sha256"],
        "successful_updates_equal": control_cycle["successful_updates"]
        == drill_cycle["successful_updates"],
    }
    result["exact_training_continuation"] = all(
        (
            result["raw_shards_equal"],
            result["tensor_payloads_equal"],
            *checkpoint_fields.values(),
            result["deterministic_metrics_equal"],
            result["onnx_model_sha256_equal"],
            result["successful_updates_equal"],
        )
    )
    return result


def _preflight_control(
    control_run_dir: Path, config_sha256: str, worktree: Path
) -> None:
    control, _ = RunRepository.open(control_run_dir)
    if control.active_session_path.exists():
        raise IntegrityError("uninterrupted control has an unfinished active session")
    verify_run(control, deep=True)
    _completed_cycle(control)
    manifest = read_json(control.root / "RUN.json")
    if manifest.get("config_sha256") != config_sha256:
        raise IntegrityError("uninterrupted control uses a different frozen configuration")
    if manifest.get("git") != git_identity(worktree):
        raise IntegrityError("uninterrupted control does not match the current source identity")
    if runtime_fingerprint(manifest.get("runtime", {})) != runtime_fingerprint(
        runtime_identity()
    ):
        raise IntegrityError("uninterrupted control does not match the current runtime")


def run_recovery_drill(
    *,
    config_path: Path,
    run_dir: Path,
    evidence_path: Path,
    phase: str,
    worktree: Path,
    control_run_dir: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    if phase not in {"collection", "training"}:
        raise ConfigError("recovery drill phase must be collection or training")
    config = load_config(config_path)
    if not config.values["run"]["disposable"]:
        raise ConfigError("recovery drills require a disposable configuration")
    if (
        config.values["training"]["device"] != "cuda"
        or config.values["operations"]["self_play_device"] != "cuda"
    ):
        raise ConfigError("target recovery drills require CUDA training and self-play")
    if phase == "training" and control_run_dir is None:
        raise ConfigError("the training recovery drill requires --control-run-dir")

    run_dir = run_dir.resolve()
    evidence_path = evidence_path.resolve()
    worktree = worktree.resolve()
    evidence_directory = evidence_path.with_suffix(evidence_path.suffix + ".files")
    _require_ignored_or_external(worktree, run_dir, "run directory")
    _require_ignored_or_external(worktree, evidence_path, "evidence path")
    _require_new_path(run_dir, "run directory")
    if evidence_path.exists() or evidence_directory.exists():
        raise ConfigError("recovery drill evidence path already exists")
    if control_run_dir is not None:
        _preflight_control(control_run_dir.resolve(), config.config_hash, worktree)
    evidence_directory.mkdir(parents=True)

    report: dict[str, Any] = {
        "schema": RECOVERY_DRILL_SCHEMA,
        "created_at": utc_now(),
        "phase": phase,
        "config": str(config.source),
        "config_sha256": config.config_hash,
        "semantic_hash": config.semantic_hash,
        "run_dir": str(run_dir),
        "control_run_dir": str(control_run_dir.resolve()) if control_run_dir else None,
        "commands": {},
        "status": "running",
    }
    process: subprocess.Popen[bytes] | None = None
    start_stdout = evidence_directory / "start.stdout.log"
    start_stderr = evidence_directory / "start.stderr.log"
    try:
        with start_stdout.open("wb") as stdout, start_stderr.open("wb") as stderr:
            process = subprocess.Popen(
                _child_cli(
                    worktree,
                    "start",
                    "--config",
                    str(config.source),
                    "--run-dir",
                    str(run_dir),
                    "--one-cycle",
                ),
                cwd=worktree,
                stdout=stdout,
                stderr=stderr,
            )
        trigger = _wait_for_interruption_point(
            process,
            run_dir,
            phase,
            timeout_seconds=timeout_seconds,
            collection_settle_seconds=0.25,
        )
        process.send_signal(signal.SIGTERM)
        try:
            interrupted_return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait(timeout=10)
            raise IntegrityError("drill child ignored controlled SIGTERM") from error
        start_record = {
            "command": _child_cli(
                worktree,
                "start",
                "--config",
                str(config.source),
                "--run-dir",
                str(run_dir),
                "--one-cycle",
            ),
            "return_code": interrupted_return_code,
            "stdout": str(start_stdout),
            "stdout_sha256": sha256_file(start_stdout),
            "stderr": str(start_stderr),
            "stderr_sha256": sha256_file(start_stderr),
        }
        report["commands"]["interrupted_start"] = start_record
        report["trigger"] = trigger
        if interrupted_return_code != 130:
            raise IntegrityError(
                f"controlled SIGTERM returned {interrupted_return_code}, expected 130"
            )
        if not (run_dir / "ACTIVE_SESSION.json").is_file():
            raise IntegrityError("controlled interruption did not retain ACTIVE_SESSION.json")

        recovered, recovery_record = _run_cli(
            worktree,
            evidence_directory,
            "recover",
            ("recover", "--run-dir", str(run_dir)),
            timeout_seconds=30,
        )
        report["commands"]["recover"] = recovery_record
        if recovered.returncode != 0:
            raise IntegrityError("recovery command failed; inspect its durable stderr evidence")
        if (run_dir / "ACTIVE_SESSION.json").exists():
            raise IntegrityError("recovery left ACTIVE_SESSION.json behind")

        verified, verify_record = _run_cli(
            worktree,
            evidence_directory,
            "verify-after-recover",
            ("verify", "--run-dir", str(run_dir), "--deep"),
            timeout_seconds=60,
        )
        report["commands"]["verify_after_recover"] = verify_record
        if verified.returncode != 0:
            raise IntegrityError("deep verification after recovery failed")

        resumed, resume_record = _run_cli(
            worktree,
            evidence_directory,
            "resume",
            ("resume", "--run-dir", str(run_dir), "--one-cycle"),
            timeout_seconds=max(timeout_seconds, 180),
        )
        report["commands"]["resume"] = resume_record
        if resumed.returncode != 0:
            raise IntegrityError("resumed drill failed; inspect its durable stderr evidence")

        verified, verify_record = _run_cli(
            worktree,
            evidence_directory,
            "verify-final",
            ("verify", "--run-dir", str(run_dir), "--deep"),
            timeout_seconds=60,
        )
        report["commands"]["verify_final"] = verify_record
        if verified.returncode != 0:
            raise IntegrityError("final deep verification failed")

        repository, _ = RunRepository.open(run_dir)
        final_state, completed = _completed_cycle(repository)
        interruptions = final_state.get("interruptions", [])
        if len(interruptions) != 1:
            raise IntegrityError("recovery drill did not record exactly one interrupted session")
        report["final"] = {
            "head": repository.head()[0],
            "recovery": repository.recovery()[0],
            "cycle_id": final_state["cycle_id"],
            "global_step": final_state["global_step"],
            "successful_updates": completed["successful_updates"],
            "model_sha256": completed["model_sha256"],
            "checkpoint_sha256": completed["checkpoint_sha256"],
            "interruption": interruptions[0],
            "collection_payloads": _collection_payload_identity(repository, completed),
            "tensor_payloads": _tensor_payload_identity(repository, completed),
        }
        if phase == "collection" and not interruptions[0].get("quarantined"):
            raise IntegrityError(
                "collection interruption did not quarantine its unsealed directory"
            )
        if control_run_dir is not None:
            comparison = compare_recovery_run(control_run_dir.resolve(), run_dir)
            report["comparison"] = comparison
            if phase == "training" and not comparison["exact_training_continuation"]:
                raise IntegrityError("resumed training differs from the uninterrupted control")
        report["status"] = "passed"
        report["completed_at"] = utc_now()
        atomic_write_json(evidence_path, report)
        return report
    except BaseException as error:
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        report["status"] = "failed"
        report["failed_at"] = utc_now()
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_write_json(evidence_path, report)
        raise


def _summarize_pair_log(
    path: Path,
    *,
    model_sha256: str,
    simulations: int,
    time_ms: int,
    batch_size: int,
    opening_pairs: int,
) -> dict[str, Any]:
    try:
        values = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"cannot read CPU benchmark pair log {path}: {error}") from error
    if len(values) != opening_pairs + 1:
        raise IntegrityError("CPU benchmark pair log is incomplete")
    header, pairs = values[0], values[1:]
    expected_header = {
        "schema": "alphamini-paired-evaluation-v1",
        "model_sha256": model_sha256,
        "simulations": simulations,
        "time_ms": time_ms,
        "batch_size": batch_size,
        "inference_device": "onnx-cpu",
        "exploratory": True,
    }
    for field, expected in expected_header.items():
        if header.get(field) != expected:
            raise IntegrityError(f"CPU benchmark pair-log {field} differs from its request")
    totals = {
        "moves": 0,
        "completed_simulations": 0,
        "neural_evaluations": 0,
        "inference_batches": 0,
        "largest_batch": 0,
        "elapsed_micros": 0,
        "deadlines_reached": 0,
    }
    for pair in pairs:
        if pair.get("schema") != "alphamini-paired-opening-result-v1":
            raise IntegrityError("CPU benchmark contains an invalid pair record")
        metrics = pair.get("metrics")
        if not isinstance(metrics, dict):
            raise IntegrityError("CPU benchmark pair has no metrics")
        for field in totals:
            value = metrics.get(field)
            if not isinstance(value, int) or value < 0:
                raise IntegrityError(f"CPU benchmark metric {field} is invalid")
            if field == "largest_batch":
                totals[field] = max(totals[field], value)
            else:
                totals[field] += value
    expected_moves = opening_pairs * 2
    if totals["moves"] != expected_moves:
        raise IntegrityError(
            f"two-ply paired benchmark expected {expected_moves} AlphaMini moves, "
            f"found {totals['moves']}"
        )
    seconds = totals["elapsed_micros"] / 1_000_000
    mean_latency_ms = totals["elapsed_micros"] / totals["moves"] / 1000
    mean_batch_fill = (
        totals["neural_evaluations"] / (totals["inference_batches"] * batch_size)
        if totals["inference_batches"]
        else 0.0
    )
    passed = (
        totals["deadlines_reached"] == 0
        and totals["completed_simulations"] == totals["moves"] * simulations
        and totals["largest_batch"] <= batch_size
    )
    return {
        "batch_size": batch_size,
        **totals,
        "mean_search_latency_ms": mean_latency_ms,
        "completed_simulations_per_second": (
            totals["completed_simulations"] / seconds if seconds else 0.0
        ),
        "mean_batch_fill": mean_batch_fill,
        "pair_log": str(path),
        "pair_log_sha256": sha256_file(path),
        "passed": passed,
    }


def run_cpu_serving_benchmark(
    *,
    arena: Path,
    model: Path,
    manifest: Path,
    openings: Path,
    output_dir: Path,
    worktree: Path,
    simulations: int,
    time_ms: int,
    opening_pairs: int,
) -> dict[str, Any]:
    if simulations <= 0 or time_ms <= 0 or opening_pairs <= 0:
        raise ConfigError("CPU benchmark limits and opening pairs must be positive")
    arena = arena.resolve()
    model = model.resolve()
    manifest = manifest.resolve()
    openings = openings.resolve()
    output_dir = output_dir.resolve()
    for path, label in (
        (arena, "arena binary"),
        (model, "model"),
        (manifest, "manifest"),
        (openings, "opening suite"),
    ):
        if not path.is_file():
            raise ConfigError(f"CPU benchmark {label} is unavailable: {path}")
    if not os.access(arena, os.X_OK):
        raise ConfigError("CPU benchmark arena binary is not executable")
    _require_ignored_or_external(worktree.resolve(), output_dir, "benchmark output directory")
    _require_new_path(output_dir, "benchmark output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_manifest = read_json(manifest)
    model_sha256 = model_manifest.get("model_sha256")
    if not isinstance(model_sha256, str) or sha256_file(model) != model_sha256:
        raise IntegrityError("CPU benchmark model differs from its manifest")

    report: dict[str, Any] = {
        "schema": CPU_BENCHMARK_SCHEMA,
        "created_at": utc_now(),
        "git": git_identity(worktree),
        "runtime": runtime_identity(),
        "arena": str(arena),
        "arena_sha256": sha256_file(arena),
        "model": str(model),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "model_sha256": model_sha256,
        "opening_suite": str(openings),
        "opening_suite_sha256": sha256_file(openings),
        "opening_pairs": opening_pairs,
        "plies_after_opening": 2,
        "simulations_per_move": simulations,
        "deadline_ms": time_ms,
        "runs": [],
        "status": "running",
    }
    try:
        for batch_size in CPU_BATCH_SIZES:
            label = f"batch-{batch_size}"
            pair_log = output_dir / f"{label}.jsonl"
            command = [
                str(arena),
                "--alphamini-model",
                str(model),
                "--alphamini-manifest",
                str(manifest),
                "--opponent",
                "random",
                "--openings",
                str(openings),
                "--games",
                str(opening_pairs),
                "--seed",
                "1",
                "--max-plies",
                "2",
                "--bootstrap",
                "1",
                "--alphamini-simulations",
                str(simulations),
                "--alphamini-time-ms",
                str(time_ms),
                "--alphamini-batch-size",
                str(batch_size),
                "--exploratory",
                "true",
                "--results",
                str(pair_log),
            ]
            started = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=worktree,
                check=False,
                capture_output=True,
                timeout=max(60, opening_pairs * 4 * time_ms / 1000),
            )
            logs = _write_process_output(
                output_dir, label, completed.stdout, completed.stderr
            )
            if completed.returncode != 0:
                raise IntegrityError(
                    f"CPU benchmark batch {batch_size} failed; inspect {logs['stderr']}"
                )
            summary = _summarize_pair_log(
                pair_log,
                model_sha256=model_sha256,
                simulations=simulations,
                time_ms=time_ms,
                batch_size=batch_size,
                opening_pairs=opening_pairs,
            )
            summary.update(
                {
                    "command": command,
                    "process_elapsed_seconds": time.monotonic() - started,
                    **logs,
                }
            )
            report["runs"].append(summary)
        production_run = next(
            run for run in report["runs"] if run["batch_size"] == PRODUCTION_CPU_BATCH_SIZE
        )
        report["production_batch_size"] = PRODUCTION_CPU_BATCH_SIZE
        report["diagnostic_all_batches_passed"] = all(
            run["passed"] for run in report["runs"]
        )
        report["passed"] = production_run["passed"]
        report["status"] = "passed" if report["passed"] else "failed"
        report["completed_at"] = utc_now()
        atomic_write_json(output_dir / "summary.json", report)
        return report
    except BaseException as error:
        report["status"] = "failed"
        report["failed_at"] = utc_now()
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_write_json(output_dir / "summary.json", report)
        raise
