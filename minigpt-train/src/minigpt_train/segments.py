"""Transactional training-segment lifecycle over the immutable run ledger."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .atomic import read_json, sha256_file
from .config import ResolvedConfig
from .data import (
    SHARDS_MANIFEST_FILE,
    BucketedSampler,
    ShardSplit,
    load_shards_manifest,
    shards_identity,
)
from .errors import ConfigError, IntegrityError
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
from .trainer import Trainer, create_initial_checkpoint

CHECKPOINT_DIR = Path("artifacts/checkpoints")


def _relative(repository: RunRepository, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.root))
    except ValueError as error:
        raise IntegrityError(f"artifact is outside run directory: {path}") from error


def shards_directory(config: ResolvedConfig, worktree: Path) -> Path:
    configured = Path(config.values["data"]["shards_dir"])
    return configured if configured.is_absolute() else (worktree / configured)


def open_corpus(config: ResolvedConfig, worktree: Path, *, verify_hashes: bool) -> dict[str, Any]:
    directory = shards_directory(config, worktree)
    manifest_path = directory / SHARDS_MANIFEST_FILE
    if not manifest_path.is_file():
        raise ConfigError(f"shard manifest is missing: {manifest_path}")
    manifest = load_shards_manifest(manifest_path, verify_hashes=verify_hashes)
    context = int(config.values["model"]["ctx"])
    train = ShardSplit(directory, manifest["train_shards"], context=context)
    validation = (
        ShardSplit(directory, manifest["val_shards"], context=context)
        if manifest["val_shards"]
        else None
    )
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "train": train,
        "validation": validation,
        "identity": shards_identity(manifest_path, manifest),
    }


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
    uv_lock = worktree / "minigpt-train" / "uv.lock"
    if not uv_lock.is_file():
        raise IntegrityError("minigpt-train/uv.lock is required for checkpoint lineage")
    cargo_lock = worktree / "Cargo.lock"
    lock_hashes = {
        "uv_lock_sha256": sha256_file(uv_lock),
        "cargo_lock_sha256": sha256_file(cargo_lock) if cargo_lock.is_file() else None,
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
        "tokenizer": config.values["data"]["tokenizer"],
    }


def bootstrap(
    repository: RunRepository,
    config: ResolvedConfig,
    state: dict[str, Any],
    worktree: Path,
) -> dict[str, Any]:
    if state["phase"] not in {"initialized", "warm_start_ready"}:
        return state
    # The corpus is hashed in full exactly once, at the step-0 boundary; later
    # sessions trust that recorded identity plus the cheap size identities.
    corpus = open_corpus(config, worktree, verify_hashes=True)
    warm_start = state.get("current_checkpoint") if state["phase"] == "warm_start_ready" else None
    _, digest, checkpoint_path = create_initial_checkpoint(
        config,
        repository.root / CHECKPOINT_DIR,
        run_root=repository.root,
        shards_identity=corpus["identity"],
        warm_start_checkpoint=(repository.root / warm_start["path"]) if warm_start else None,
        lineage=_checkpoint_lineage(repository, config, worktree),
    )
    state = copy.deepcopy(state)
    state["phase"] = "ready_train"
    state["shards"] = corpus["identity"]
    state["current_checkpoint"] = {
        "path": _relative(repository, checkpoint_path),
        "sha256": digest,
        "global_step": 0,
        "warm_start_from": warm_start["sha256"] if warm_start else None,
    }
    repository.commit_head(state)
    return repository.head()[1]


def train_segment(
    repository: RunRepository,
    config: ResolvedConfig,
    state: dict[str, Any],
    session: ActiveSession,
    worktree: Path,
) -> dict[str, Any]:
    if state["phase"] not in {"ready_train", "training"}:
        return state
    values = config.values["training"]
    total_steps = int(values["total_steps"])
    corpus = open_corpus(config, worktree, verify_hashes=False)
    if corpus["identity"] != state["shards"]:
        raise IntegrityError("shard corpus differs from the one this run was started on")
    sampler = BucketedSampler(
        corpus["train"],
        seed=int(config.values["run"]["seed"]),
        micro_batch=int(values["micro_batch"]),
        buckets=int(config.values["data"]["length_buckets"]),
    )
    checkpoint_dir = repository.root / CHECKPOINT_DIR
    protected = [
        repository.root / descriptor["path"]
        for key in ("current_checkpoint", "best_checkpoint")
        if (descriptor := state.get(key)) is not None
    ]
    trainer = Trainer(
        config,
        sampler,
        validation=corpus["validation"],
        run_root=repository.root,
        shards_identity=state["shards"],
        lineage=_checkpoint_lineage(repository, config, worktree),
        protected_checkpoints=protected,
    )
    if state["phase"] == "training":
        recovery = state.get("recovery_checkpoint")
        if not recovery:
            raise IntegrityError("training phase has no recovery checkpoint")
        trainer.resume(repository.root / recovery["path"])
        target_step = int(state["segment_target_step"])
    else:
        trainer.resume(repository.root / state["current_checkpoint"]["path"])
        target_step = min(trainer.state.global_step + int(values["segment_steps"]), total_steps)
    trainer.best_checkpoint = copy.deepcopy(state.get("best_checkpoint"))
    if trainer.state.global_step > target_step:
        raise IntegrityError("resumed step is beyond this segment's frozen target")
    segment_index = int(state["segment_index"])
    trainer.state.segment_index = segment_index
    base_state = copy.deepcopy(state)

    def checkpoint_callback(digest: str, path: Path, training_state: Any) -> None:
        recovery_state = copy.deepcopy(base_state)
        recovery_state["phase"] = "training"
        recovery_state["segment_target_step"] = target_step
        recovery_state["global_step"] = training_state.global_step
        recovery_state["recovery_checkpoint"] = {
            "path": _relative(repository, path),
            "sha256": digest,
            "global_step": training_state.global_step,
        }
        recovery_state["best_checkpoint"] = copy.deepcopy(trainer.best_checkpoint)
        recovery_state["best_validation_loss"] = training_state.best_validation_loss
        recovery_state["best_validation_step"] = training_state.best_validation_step
        recovery_state["evaluations_without_improvement"] = (
            training_state.evaluations_without_improvement
        )
        repository.commit_recovery(recovery_state)
        session.heartbeat(force=True)

    summary = trainer.train_segment(
        target_step,
        checkpoint_dir=checkpoint_dir,
        metrics_path=repository.metrics_path,
        checkpoint_callback=checkpoint_callback,
        heartbeat=session.heartbeat,
    )
    digest, checkpoint_path = trainer.ensure_checkpoint(checkpoint_dir)
    session.heartbeat(force=True)
    checkpoint_descriptor = {
        "path": _relative(repository, checkpoint_path),
        "sha256": digest,
        "global_step": trainer.state.global_step,
    }
    completed = {
        "segment_index": segment_index,
        "completed_at": utc_now(),
        "first_step": int(state["global_step"]),
        "last_step": trainer.state.global_step,
        "target_step": target_step,
        "attempts": summary.attempts,
        "amp_overflows": summary.amp_overflows,
        "seconds": summary.seconds,
        "target_tokens": summary.target_tokens,
        "tokens_per_second": summary.target_tokens / max(summary.seconds, 1e-9),
        "train_loss": summary.last_train_loss,
        "evaluation": summary.last_evaluation,
        "checkpoint": checkpoint_descriptor,
    }
    finished = trainer.state.global_step >= total_steps or trainer.state.early_stopped
    _, head_state = repository.head()
    final_state = copy.deepcopy(head_state)
    final_state["phase"] = "complete" if finished else "ready_train"
    final_state["segment_index"] = segment_index + 1
    final_state["global_step"] = trainer.state.global_step
    final_state["current_checkpoint"] = checkpoint_descriptor
    final_state["best_checkpoint"] = copy.deepcopy(trainer.best_checkpoint)
    final_state["best_validation_loss"] = trainer.state.best_validation_loss
    final_state["best_validation_step"] = trainer.state.best_validation_step
    final_state["evaluations_without_improvement"] = trainer.state.evaluations_without_improvement
    final_state["early_stopped"] = trainer.state.early_stopped
    final_state["recovery_checkpoint"] = None
    final_state["completed_segments"] = [*state.get("completed_segments", []), completed]
    final_state.pop("segment_target_step", None)
    repository.commit_head(final_state)
    # Prune once more against the promoted HEAD so a superseded segment tail is not kept.
    trainer.protected = {
        (repository.root / descriptor["path"]).resolve()
        for key in ("current_checkpoint", "best_checkpoint")
        if (descriptor := final_state.get(key)) is not None
    }
    trainer.prune(checkpoint_dir)
    return repository.head()[1]


def _finalize_session(repository: RunRepository, session: ActiveSession) -> dict[str, Any]:
    """Account one training session and expose its durable boundary once."""

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
    one_segment: bool,
    initialize_only: bool = False,
) -> dict[str, Any]:
    # Refuse a no-op resume before creating ACTIVE_SESSION. A recovered mid-segment
    # state is allowed to reach its next segment boundary first.
    _, preflight = repository.effective()
    if preflight["phase"] == "complete":
        raise ConfigError("run is complete; fork it to continue training")
    if budget_exhausted(preflight) and safe_budget_boundary(preflight):
        marked = attach_initial_budget_milestone(repository, preflight)
        if marked is not preflight:
            repository.commit_head(marked)
        raise ConfigError("active-time budget is exhausted; run extend before resuming")
    # Verify code, locks, and determinism-relevant runtime before any weight changes.
    _checkpoint_lineage(repository, config, worktree)
    heartbeat = int(config.values["operations"]["heartbeat_seconds"])
    session = repository.begin_session("train", heartbeat)
    try:
        _, state = repository.effective()
        initial_segment = int(state["segment_index"])
        state = bootstrap(repository, config, state, worktree)
        if initialize_only:
            return _finalize_session(repository, session)
        while state["phase"] != "complete" and not (
            safe_budget_boundary(state)
            and budget_exhausted(state, additional_seconds=session.elapsed)
        ):
            state = train_segment(repository, config, state, session, worktree)
            if one_segment and int(state["segment_index"]) > initial_segment:
                break
        return _finalize_session(repository, session)
    except BaseException:
        # Leave ACTIVE_SESSION and the latest RECOVERY pointer intact for explicit recovery.
        session.abandon()
        raise
