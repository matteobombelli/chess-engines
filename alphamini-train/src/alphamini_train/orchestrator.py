"""Transactional collect -> materialize -> train -> export cycle orchestration."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from .atomic import atomic_write_json, read_json, sha256_file
from .config import ResolvedConfig
from .data import ReplayDataset, updates_for_new_positions
from .errors import ConfigError, IntegrityError
from .export import export_onnx
from .run import (
    ActiveSession,
    RunRepository,
    attach_initial_budget_milestone,
    budget_exhausted,
    git_identity,
    runtime_fingerprint,
    runtime_identity,
    safe_budget_boundary,
    utc_now,
)
from .schema import U64_MASK, TensorCache, validate_collection_manifest
from .trainer import Trainer, create_initial_checkpoint


def _relative(repository: RunRepository, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.root))
    except ValueError as error:
        raise IntegrityError(f"artifact is outside run directory: {path}") from error


def _required_command(config: ResolvedConfig, key: str) -> list[str]:
    command = config.values["operations"].get(key, [])
    if not command:
        raise ConfigError(f"operations.{key} is required to advance this phase")
    executable = shutil.which(command[0])
    if executable is None:
        raise ConfigError(f"configured executable is unavailable: {command[0]}")
    return list(command)


def _cycle_collection_seed(run_seed: int, cycle_id: int) -> int:
    """Derive the frozen u64 collection seed from run identity and cycle."""

    return ((run_seed << 32) ^ cycle_id) & U64_MASK


def _verify_collection_request(manifest: dict[str, Any], *, expected: dict[str, Any]) -> None:
    """Reject a valid-but-unrequested collection before cache admission."""

    for field, requested in expected.items():
        if manifest.get(field) != requested:
            raise IntegrityError(f"collection {field} does not match the request")


def _checkpoint_lineage(
    repository: RunRepository, config: ResolvedConfig, worktree: Path
) -> dict[str, Any]:
    run_manifest = read_json(repository.root / "RUN.json")
    current_git = git_identity(worktree)
    disposable = bool(config.values["run"]["disposable"])
    if run_manifest.get("disposable") is not disposable:
        raise IntegrityError("run disposable identity differs from the frozen configuration")
    if current_git["commit"] != run_manifest["git"]["commit"]:
        raise IntegrityError("current Git commit differs from the run's frozen source commit")
    if current_git.get("tracked_dirty") is None or current_git.get("worktree_sha256") is None:
        raise IntegrityError("worktree cleanliness or content identity is unknown; resume refused")
    if not disposable and current_git["tracked_dirty"] is not False:
        raise IntegrityError("worktree cleanliness is dirty or unknown; exact resume is refused")
    if current_git["worktree_sha256"] != run_manifest["git"].get("worktree_sha256"):
        raise IntegrityError("current worktree content differs from the run's frozen source tree")
    uv_lock = worktree / "alphamini-train" / "uv.lock"
    cargo_lock = worktree / "Cargo.lock"
    if not uv_lock.is_file() or not cargo_lock.is_file():
        raise IntegrityError("both uv.lock and Cargo.lock are required for checkpoint lineage")
    lock_hashes = {
        "uv_lock_sha256": sha256_file(uv_lock),
        "cargo_lock_sha256": sha256_file(cargo_lock),
    }
    if run_manifest.get("locks") != {
        "cargo_lock_sha256": lock_hashes["cargo_lock_sha256"],
        "uv_lock_sha256": lock_hashes["uv_lock_sha256"],
    }:
        raise IntegrityError("current Python/Rust lockfiles differ from the frozen run")
    if runtime_fingerprint(runtime_identity()) != runtime_fingerprint(run_manifest["runtime"]):
        raise IntegrityError("current determinism-relevant runtime differs; create a fork")
    return {
        "source_commit": current_git["commit"],
        "source_disposable": disposable,
        "source_worktree_sha256": current_git["worktree_sha256"],
        "config_sha256": config.config_hash,
        "semantic_hash": config.semantic_hash,
        **lock_hashes,
        "encoder_schema": config.values["schemas"]["encoder"],
        "action_schema": config.values["schemas"]["action"],
    }


def _run_external(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    cwd: Path,
    session: ActiveSession,
    timeout_seconds: int,
    record_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    record = {
        "schema": "alphamini.external-invocation.v1",
        "command": list(command),
        "environment": environment,
        "cwd": str(cwd),
        "started_at": started_at,
        "status": "running",
        "log_path": log_path.name,
    }
    atomic_write_json(record_path, record)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env={**os.environ, **environment},
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        while True:
            try:
                return_code = process.wait(timeout=session.heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                session.heartbeat(force=True)
                if timeout_seconds and time.monotonic() - started > timeout_seconds:
                    _terminate_process(process)
                    raise IntegrityError(f"external command exceeded {timeout_seconds}s: {command}")
        if return_code != 0:
            raise IntegrityError(f"external command exited {return_code}: {command}")
        record.update(
            {
                "status": "completed",
                "return_code": return_code,
                "finished_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        atomic_write_json(record_path, record)
        return record
    except BaseException:
        _terminate_process(process)
        record.update(
            {
                "status": "failed",
                "return_code": process.poll(),
                "finished_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        atomic_write_json(record_path, record)
        raise
    finally:
        log.close()


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    """Stop a child deterministically, escalating once after a bounded grace period."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def bootstrap(
    repository: RunRepository,
    config: ResolvedConfig,
    state: dict[str, Any],
    worktree: Path,
) -> dict[str, Any]:
    if state["phase"] not in {"initialized", "warm_start_ready"}:
        return state
    artifacts = repository.root / "artifacts"
    warm_start = state.get("current_checkpoint") if state["phase"] == "warm_start_ready" else None
    model, checkpoint_hash, checkpoint_path = create_initial_checkpoint(
        config,
        artifacts / "checkpoints" / "cycle-000000",
        warm_start_checkpoint=(repository.root / warm_start["path"]) if warm_start else None,
        warm_start_sha256=warm_start.get("sha256") if warm_start else None,
        lineage=_checkpoint_lineage(repository, config, worktree),
    )
    model_path, manifest_path, manifest = export_onnx(
        model,
        config,
        artifacts / "models" / "cycle-000000",
        cycle_id=0,
        global_step=0,
        parent_checkpoint_sha256=checkpoint_hash,
        seed=int(config.values["run"]["seed"]),
    )
    state = copy.deepcopy(state)
    state["phase"] = "ready_collect"
    state["current_checkpoint"] = {
        "path": _relative(repository, checkpoint_path),
        "sha256": checkpoint_hash,
        "cycle_id": 0,
        "global_step": 0,
    }
    state["current_model"] = {
        "path": _relative(repository, model_path),
        "manifest_path": _relative(repository, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_path": _relative(repository, manifest_path.with_suffix(".training.json")),
        "provenance_sha256": sha256_file(manifest_path.with_suffix(".training.json")),
        "sha256": manifest["model_sha256"],
        "cycle_id": 0,
        "global_step": 0,
    }
    repository.commit_head(state)
    return repository.head()[1]


def collect(
    repository: RunRepository,
    config: ResolvedConfig,
    state: dict[str, Any],
    session: ActiveSession,
    worktree: Path,
) -> dict[str, Any]:
    if state["phase"] != "ready_collect":
        return state
    command = _required_command(config, "collect_command")
    cycle_id = int(state["cycle_id"])
    cycle_dir = repository.root / "cycles" / f"cycle-{cycle_id:06d}"
    collection_dir = cycle_dir / "collection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    collection_manifest = collection_dir / "collection.json"
    if collection_manifest.exists():
        raise IntegrityError(
            f"uncommitted collection manifest already exists: {collection_manifest}; run recover"
        )
    model = state["current_model"]
    games = int(config.values["self_play"]["games_per_cycle"])
    game_id_start = int(state["game_id_next"])
    run_id = read_json(repository.root / "RUN.json")["run_id"]
    # Cycle and game identifiers derive only from the frozen run seed/counters.
    cycle_seed = _cycle_collection_seed(int(config.values["run"]["seed"]), cycle_id)
    arguments = [
        *command,
        "--model",
        str(repository.root / model["path"]),
        "--device",
        str(config.values["operations"]["self_play_device"]),
        "--manifest",
        str(repository.root / model["manifest_path"]),
        "--output-dir",
        str(collection_dir),
        "--collection-manifest",
        str(collection_manifest),
        "--run-id",
        run_id,
        "--games",
        str(games),
        "--simulations",
        str(config.values["self_play"]["simulations"]),
        "--batch-size",
        str(config.values["self_play"]["batch_size"]),
        "--seed",
        str(cycle_seed),
        "--max-plies",
        str(config.values["self_play"]["max_plies"]),
        "--dirichlet-alpha",
        str(config.values["self_play"]["dirichlet_alpha"]),
        "--dirichlet-epsilon",
        str(config.values["self_play"]["dirichlet_epsilon"]),
        "--sample-until-ply",
        str(config.values["self_play"]["sample_until_ply"]),
        "--cpuct",
        str(config.values["self_play"]["cpuct"]),
        "--fpu-reduction",
        str(config.values["self_play"]["fpu_reduction"]),
    ]
    environment = {
        "ALPHAMINI_RUN_DIR": str(repository.root),
        "ALPHAMINI_RUN_ID": run_id,
        "ALPHAMINI_CYCLE_ID": str(cycle_id),
        "ALPHAMINI_GAME_ID_START": str(game_id_start),
        "ALPHAMINI_MODEL_PATH": str(repository.root / model["path"]),
        "ALPHAMINI_MODEL_MANIFEST": str(repository.root / model["manifest_path"]),
        "ALPHAMINI_COLLECTION_DIR": str(collection_dir),
        "ALPHAMINI_COLLECTION_MANIFEST": str(collection_manifest),
        "ALPHAMINI_CONFIG_JSON": str(repository.root / "config.resolved.json"),
        "ALPHAMINI_CONFIG_SHA256": config.config_hash,
        "ALPHAMINI_SEMANTIC_HASH": config.semantic_hash,
    }
    collection_invocation = collection_dir / "collect-command.json"
    _run_external(
        arguments,
        environment=environment,
        cwd=worktree,
        session=session,
        timeout_seconds=int(config.values["operations"].get("command_timeout_seconds", 0)),
        record_path=collection_invocation,
        log_path=collection_dir / "collect.log",
    )
    manifest = validate_collection_manifest(collection_manifest)
    expected_fields = {
        "run_id": run_id,
        "cycle_id": cycle_id,
        "game_id_start": game_id_start,
        "game_count": games,
        "model_sha256": model["sha256"],
        "config_sha256": config.config_hash,
        "seed": cycle_seed,
        "simulations": int(config.values["self_play"]["simulations"]),
        "max_plies": int(config.values["self_play"]["max_plies"]),
    }
    _verify_collection_request(manifest, expected=expected_fields)
    state = copy.deepcopy(state)
    state["phase"] = "ready_materialize"
    state["pending_collection"] = {
        "path": _relative(repository, collection_manifest),
        "sha256": sha256_file(collection_manifest),
        "cycle_id": cycle_id,
        "game_id_start": game_id_start,
        "game_count": games,
        "position_count": manifest["position_count"],
        "invocation_path": _relative(repository, collection_invocation),
    }
    state["game_id_next"] = game_id_start + games
    repository.commit_head(state)
    return repository.head()[1]


def materialize(
    repository: RunRepository,
    config: ResolvedConfig,
    state: dict[str, Any],
    session: ActiveSession,
    worktree: Path,
) -> dict[str, Any]:
    if state["phase"] != "ready_materialize":
        return state
    command = _required_command(config, "materialize_command")
    collection = state["pending_collection"]
    cycle_id = int(state["cycle_id"])
    cache_dir = repository.root / "cache" / f"cycle-{cycle_id:06d}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensor_manifest = cache_dir / "tensors.json"
    arguments = [
        *command,
        "--collection-manifest",
        str(repository.root / collection["path"]),
        "--output-dir",
        str(cache_dir),
        "--tensor-manifest",
        str(tensor_manifest),
    ]
    environment = {
        "ALPHAMINI_RUN_DIR": str(repository.root),
        "ALPHAMINI_CYCLE_ID": str(cycle_id),
        "ALPHAMINI_COLLECTION_MANIFEST": str(repository.root / collection["path"]),
        "ALPHAMINI_TENSOR_MANIFEST": str(tensor_manifest),
        "ALPHAMINI_CONFIG_JSON": str(repository.root / "config.resolved.json"),
    }
    materialize_invocation = cache_dir / "materialize-command.json"
    _run_external(
        arguments,
        environment=environment,
        cwd=worktree,
        session=session,
        timeout_seconds=int(config.values["operations"].get("command_timeout_seconds", 0)),
        record_path=materialize_invocation,
        log_path=cache_dir / "materialize.log",
    )
    cache = TensorCache(tensor_manifest, verify_hashes=True)
    if cache.record_count != int(collection["position_count"]):
        raise IntegrityError("tensor cache record count differs from collection")
    if cache.manifest["encoder_schema"] != config.values["schemas"]["encoder"]:
        raise IntegrityError("tensor cache uses the wrong encoder schema")
    if cache.manifest["action_schema"] != config.values["schemas"]["action"]:
        raise IntegrityError("tensor cache uses the wrong action schema")
    if cache.manifest.get("source_collection_sha256") != collection["sha256"]:
        raise IntegrityError("tensor cache does not reference the committed collection")
    state = copy.deepcopy(state)
    state["phase"] = "ready_train"
    state["pending_tensor_cache"] = {
        "path": _relative(repository, tensor_manifest),
        "sha256": sha256_file(tensor_manifest),
        "record_count": cache.record_count,
        "cycle_id": cycle_id,
        "invocation_path": _relative(repository, materialize_invocation),
    }
    repository.commit_head(state)
    return repository.head()[1]


def _selected_replay(state: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    candidates = [*state.get("replay_caches", []), state["pending_tensor_cache"]]
    selected: list[dict[str, Any]] = []
    positions = 0
    for cache in reversed(candidates):
        selected.append(cache)
        positions += int(cache["record_count"])
        if positions >= limit:
            break
    return list(reversed(selected))


def train_and_export(
    repository: RunRepository,
    config: ResolvedConfig,
    state: dict[str, Any],
    session: ActiveSession,
    worktree: Path,
) -> dict[str, Any]:
    if state["phase"] not in {"ready_train", "training"}:
        return state
    train_export_started = time.monotonic()
    cycle_out = int(state["cycle_id"]) + 1
    replay = _selected_replay(state, int(config.values["training"]["replay_positions"]))
    caches = [TensorCache(repository.root / item["path"], verify_hashes=True) for item in replay]
    dataset = ReplayDataset(
        caches,
        seed=int(config.values["run"]["seed"]),
        validation_fraction=float(config.values["training"]["validation_fraction"]),
    )
    trainer = Trainer(
        config,
        dataset,
        cycle_id=cycle_out,
        lineage=_checkpoint_lineage(repository, config, worktree),
    )
    target_steps = updates_for_new_positions(
        int(state["pending_tensor_cache"]["record_count"]),
        float(config.values["training"]["sample_ratio"]),
        int(config.values["training"]["batch_size"]),
    )
    if state["phase"] == "training":
        recovery = state.get("recovery_checkpoint")
        if not recovery:
            raise IntegrityError("training phase has no recovery checkpoint")
        trainer.resume(repository.root / recovery["path"])
        if int(state.get("target_cycle_steps", -1)) != target_steps:
            raise IntegrityError("recomputed update count differs from recovery state")
    else:
        trainer.load_parent(repository.root / state["current_checkpoint"]["path"])

    base_state = copy.deepcopy(state)

    def checkpoint_callback(digest: str, path: Path, training_state: Any) -> None:
        recovery_state = copy.deepcopy(base_state)
        recovery_state["phase"] = "training"
        recovery_state["target_cycle_steps"] = target_steps
        recovery_state["cycle_step"] = training_state.cycle_step
        recovery_state["global_step"] = training_state.global_step
        recovery_state["recovery_checkpoint"] = {
            "path": _relative(repository, path),
            "sha256": digest,
            "cycle_id": cycle_out,
            "global_step": training_state.global_step,
        }
        repository.commit_recovery(recovery_state)
        session.heartbeat(force=True)

    digest, checkpoint_path, metrics = trainer.train(
        target_steps,
        checkpoint_dir=repository.root / "artifacts" / "checkpoints" / f"cycle-{cycle_out:06d}",
        checkpoint_callback=checkpoint_callback,
        heartbeat=session.heartbeat,
    )
    export_started = time.monotonic()
    model_path, manifest_path, model_manifest = export_onnx(
        trainer.model,
        config,
        repository.root / "artifacts" / "models" / f"cycle-{cycle_out:06d}",
        cycle_id=cycle_out,
        global_step=trainer.state.global_step,
        parent_checkpoint_sha256=digest,
        seed=int(config.values["run"]["seed"]) ^ cycle_out,
    )
    metrics["export_session_seconds"] = time.monotonic() - export_started
    metrics["train_export_session_seconds"] = time.monotonic() - train_export_started
    session.heartbeat(force=True)
    checkpoint_descriptor = {
        "path": _relative(repository, checkpoint_path),
        "sha256": digest,
        "cycle_id": cycle_out,
        "global_step": trainer.state.global_step,
    }
    model_descriptor = {
        "path": _relative(repository, model_path),
        "manifest_path": _relative(repository, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_path": _relative(repository, manifest_path.with_suffix(".training.json")),
        "provenance_sha256": sha256_file(manifest_path.with_suffix(".training.json")),
        "sha256": model_manifest["model_sha256"],
        "cycle_id": cycle_out,
        "global_step": trainer.state.global_step,
    }
    # Promote the fully verified cycle in one HEAD transaction.
    _, head_state = repository.head()
    completed = {
        "cycle_id": cycle_out,
        "completed_at": utc_now(),
        "collection": state["pending_collection"],
        "tensor_cache": state["pending_tensor_cache"],
        "checkpoint_sha256": digest,
        "model_sha256": model_manifest["model_sha256"],
        "checkpoint": checkpoint_descriptor,
        "model": model_descriptor,
        "successful_updates": target_steps,
        "metrics": metrics,
    }
    final_state = copy.deepcopy(head_state)
    final_state["phase"] = "ready_collect"
    final_state["cycle_id"] = cycle_out
    final_state["global_step"] = trainer.state.global_step
    final_state["current_checkpoint"] = checkpoint_descriptor
    final_state["current_model"] = model_descriptor
    final_state["replay_caches"] = replay
    final_state["completed_cycles"] = [*state.get("completed_cycles", []), completed]
    final_state["pending_collection"] = None
    final_state["pending_tensor_cache"] = None
    final_state.pop("recovery_checkpoint", None)
    final_state.pop("target_cycle_steps", None)
    final_state.pop("cycle_step", None)
    repository.commit_head(final_state)
    return repository.head()[1]


def _finalize_session(repository: RunRepository, session: ActiveSession) -> dict[str, Any]:
    """Account one orchestrator session and expose its durable boundary once."""

    elapsed = session.seal()
    _, durable = repository.effective()
    durable = copy.deepcopy(durable)
    durable["active_used_seconds"] = float(durable["active_used_seconds"]) + elapsed
    durable = attach_initial_budget_milestone(repository, durable)
    repository.commit_head(durable)
    # Clear only after the new HEAD is durable. A crash before here remains an
    # explicit, recoverable session instead of silently losing active time.
    session.clear()
    return repository.head()[1]


def run_training(
    repository: RunRepository,
    config: ResolvedConfig,
    *,
    worktree: Path,
    one_cycle: bool,
    initialize_only: bool = False,
) -> dict[str, Any]:
    if not bool(config.values["training"]["horizon_confirmed"]):
        raise ConfigError(
            "training horizon is not confirmed; freeze it from pilot evidence before training"
        )
    # Refuse a no-op resume before creating ACTIVE_SESSION. A recovered mid-cycle
    # state is allowed to reach the next fully promoted model boundary first.
    _, preflight = repository.effective()
    if budget_exhausted(preflight) and safe_budget_boundary(preflight):
        marked = attach_initial_budget_milestone(repository, preflight)
        if marked is not preflight:
            repository.commit_head(marked)
        raise ConfigError("active-time budget is exhausted; run extend before resuming")
    # Verify code, locks, schemas, and determinism-relevant runtime before collection can run.
    _checkpoint_lineage(repository, config, worktree)
    heartbeat = int(config.values["operations"].get("heartbeat_seconds", 30))
    session = repository.begin_session("resume", heartbeat)
    initial_cycle: int | None = None
    try:
        _, state = repository.effective()
        initial_cycle = int(state["cycle_id"])
        state = bootstrap(repository, config, state, worktree)
        if initialize_only:
            return _finalize_session(repository, session)
        while not (
            safe_budget_boundary(state)
            and budget_exhausted(state, additional_seconds=session.elapsed)
        ):
            state = collect(repository, config, state, session, worktree)
            state = materialize(repository, config, state, session, worktree)
            state = train_and_export(repository, config, state, session, worktree)
            if one_cycle and int(state["cycle_id"]) > initial_cycle:
                break
        return _finalize_session(repository, session)
    except BaseException:
        # Leave ACTIVE_SESSION and the latest RECOVERY pointer intact for explicit recovery.
        session.abandon()
        raise
