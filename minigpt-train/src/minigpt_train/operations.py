"""Read-only verification/reporting, export publication, and conservative garbage collection."""

from __future__ import annotations

import copy
import json
import math
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

from .atomic import atomic_write_bytes, free_bytes, read_json, sha256_file
from .config import ResolvedConfig
from .errors import IntegrityError
from .export import MANIFEST_FIELDS, MANIFEST_SCHEMA, export_onnx, publish_current
from .model import build_model, parameter_count
from .parity import verify_parity_fixture, write_parity_fixture
from .run import (
    RunRepository,
    git_identity,
    read_initial_budget_milestone,
    runtime_identity,
    utc_now,
    verify_budget_ledger,
)
from .segments import CHECKPOINT_DIR, open_corpus, shards_directory
from .trainer import load_model_weights, prunable_checkpoints, remove_checkpoint

DOCTOR_SCHEMA = "minigpt.doctor-report.v1"
REPRODUCTION_SCHEMA = "minigpt.reproduction.v1"
PUBLISHED_RELATIVE = Path("artifacts/minigpt/current")


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

    for module in ("numpy", "torch", "onnx", "onnxruntime"):
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
        record("parameter_count", "pass", parameter_count(build_model(config)))
    except ImportError:
        record("training_cuda", "fail" if production else "warn", "PyTorch is unavailable")

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
        with tempfile.TemporaryDirectory(prefix="minigpt-doctor-") as temporary:
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

    floor = int(config.values["training"]["disk_floor_bytes"])
    try:
        available = free_bytes(worktree)
        record(
            "disk",
            "pass" if available >= floor else "fail",
            f"{available / 1024**3:.1f} GiB free; floor {floor / 1024**3:.1f} GiB",
        )
    except OSError as error:
        record("disk", "fail", str(error))

    try:
        # The full per-shard hash check lives here so that training start can
        # trust the identity a prior doctor/verify recorded.
        corpus = open_corpus(config, worktree, verify_hashes=True)
        counts = corpus["manifest"]["counts"]
        record(
            "shards",
            "pass",
            {
                "directory": str(shards_directory(config, worktree)),
                "games_train": counts["games_train"],
                "games_val": counts["games_val"],
                "tokens_train": counts["tokens_train"],
                "tokens_val": counts["tokens_val"],
            },
        )
    except Exception as error:
        record("shards", "fail" if production else "warn", str(error))

    failures = sum(check["status"] == "fail" for check in checks)
    warnings = sum(check["status"] == "warn" for check in checks)
    return {
        "schema": DOCTOR_SCHEMA,
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


def verify_run(
    repository: RunRepository, config: ResolvedConfig, *, deep: bool, worktree: Path
) -> dict[str, Any]:
    summary = repository.verify_state_chain(deep=deep)
    _, state = repository.effective()
    checked = verify_budget_ledger(repository, state)
    for name in ("current_checkpoint", "best_checkpoint", "recovery_checkpoint"):
        if state.get(name):
            _verify_artifact(repository, state[name], name)
            checked += 1
    for export in state.get("exports", []):
        _verify_artifact(repository, export, "export")
        manifest_path = repository.root / export["manifest_path"]
        if sha256_file(manifest_path) != export.get("manifest_sha256"):
            raise IntegrityError("export manifest checksum mismatch")
        manifest = read_json(manifest_path)
        if set(manifest) != MANIFEST_FIELDS or manifest.get("schema") != MANIFEST_SCHEMA:
            raise IntegrityError("invalid export manifest")
        if manifest.get("model_sha256") != export["sha256"]:
            raise IntegrityError("export manifest references another model")
        provenance_path = repository.root / export["provenance_path"]
        if read_json(provenance_path).get("parity", {}).get("status") != "passed":
            raise IntegrityError("published model lacks passing ONNX parity")
        checked += 3
        if deep:
            fixture = verify_parity_fixture(repository.root / export["fixture_path"])
            if fixture["model_sha256"] != export["sha256"]:
                raise IntegrityError("parity fixture belongs to another model")
            checked += 1
    if deep and state.get("shards"):
        corpus = open_corpus(config, worktree, verify_hashes=True)
        if corpus["identity"] != state["shards"]:
            raise IntegrityError("shard corpus differs from the one this run was started on")
        checked += 1
    summary["artifacts_checked"] = checked
    summary["deep"] = deep
    summary["verified_at"] = utc_now()
    return summary


def export_best(
    repository: RunRepository,
    config: ResolvedConfig,
    *,
    publish_root: Path,
) -> dict[str, Any]:
    """Export the best-validation checkpoint, verify it, and publish `current`."""

    _, state = repository.effective()
    descriptor = state.get("best_checkpoint") or state.get("current_checkpoint")
    if descriptor is None:
        raise IntegrityError("run has no checkpoint to export")
    _verify_artifact(repository, descriptor, "export source checkpoint")
    checkpoint = repository.root / descriptor["path"]
    model = load_model_weights(config, checkpoint)
    global_step = int(descriptor["global_step"])
    output_dir = repository.root / "artifacts" / "models" / f"step-{global_step:09d}"
    seed = int(config.values["run"]["seed"]) ^ global_step
    model_path, manifest_path, manifest = export_onnx(
        model,
        config,
        output_dir,
        global_step=global_step,
        parent_checkpoint_sha256=descriptor["sha256"],
        seed=seed,
    )
    fixture = write_parity_fixture(
        model,
        model_path,
        config,
        output_dir / "fixtures",
        model_sha256=manifest["model_sha256"],
        seed=seed,
    )
    published = publish_current(model_path, manifest_path, publish_root)
    provenance_path = manifest_path.with_name(manifest_path.stem + ".training.json")
    export_record = {
        "path": str(model_path.relative_to(repository.root)),
        "manifest_path": str(manifest_path.relative_to(repository.root)),
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_path": str(provenance_path.relative_to(repository.root)),
        "provenance_sha256": sha256_file(provenance_path),
        "fixture_path": str((output_dir / "fixtures").relative_to(repository.root)),
        "sha256": manifest["model_sha256"],
        "global_step": global_step,
        "checkpoint_sha256": descriptor["sha256"],
        "exported_at": utc_now(),
    }
    _, head_state = repository.head()
    final_state = copy.deepcopy(head_state)
    final_state["exports"] = [*head_state.get("exports", []), export_record]
    repository.commit_head(final_state)
    return {
        "export": export_record,
        "published": {key: str(value) for key, value in published.items()},
        "parity_cases": [case["name"] for case in fixture["cases"]],
    }


def read_metrics(repository: RunRepository) -> list[dict[str, Any]]:
    if not repository.metrics_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in repository.metrics_path.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise IntegrityError(f"metrics.jsonl contains invalid JSON: {error}") from error
        records.append(value)
    return records


def progress_summary(repository: RunRepository) -> dict[str, Any]:
    config = read_json(repository.root / "config.resolved.json")
    _, state = repository.effective()
    total_steps = int(config["training"]["total_steps"])
    segments = state.get("completed_segments", [])
    trained_steps = sum(
        int(segment["last_step"]) - int(segment["first_step"]) for segment in segments
    )
    seconds = sum(float(segment["seconds"]) for segment in segments)
    steps_per_second = trained_steps / seconds if seconds > 0 and trained_steps > 0 else None
    remaining = max(0, total_steps - int(state["global_step"]))
    metrics = read_metrics(repository)
    return {
        "phase": state["phase"],
        "global_step": int(state["global_step"]),
        "total_steps": total_steps,
        "fraction_complete": int(state["global_step"]) / total_steps,
        "completed_segments": len(segments),
        "measured_steps_per_second": steps_per_second,
        "estimated_remaining_seconds": (remaining / steps_per_second if steps_per_second else None),
        "best_validation_loss": state.get("best_validation_loss"),
        "best_validation_step": state.get("best_validation_step"),
        "best_validation_perplexity": (
            math.exp(min(float(state["best_validation_loss"]), 60.0))
            if state.get("best_validation_loss") is not None
            else None
        ),
        "early_stopped": bool(state.get("early_stopped")),
        "evaluations_recorded": len(metrics),
        "last_metrics": metrics[-1] if metrics else None,
        "active_used_hours": float(state["active_used_seconds"]) / 3600,
        "active_budget_hours": float(state["active_budget_seconds"]) / 3600,
    }


def reproduction_record(repository: RunRepository) -> dict[str, Any]:
    manifest = read_json(repository.root / "RUN.json")
    _, state = repository.effective()
    milestone = read_initial_budget_milestone(repository, state)
    run_dir = shlex.quote(str(repository.root))
    return {
        "schema": REPRODUCTION_SCHEMA,
        "run_id": manifest["run_id"],
        "config_sha256": manifest["config_sha256"],
        "semantic_hash": manifest["semantic_hash"],
        "disposable": manifest.get("disposable", False),
        "source_git": manifest["git"],
        "locks": manifest["locks"],
        "shards": state.get("shards"),
        "original_runtime": manifest["runtime"],
        "current_runtime": runtime_identity(),
        "head": repository.head()[0],
        "recovery": repository.recovery()[0],
        "segment_index": state["segment_index"],
        "global_step": state["global_step"],
        "active_used_seconds": state["active_used_seconds"],
        "active_budget_seconds": state["active_budget_seconds"],
        "initial_active_budget_seconds": state["initial_active_budget_seconds"],
        "initial_budget_milestone_sha256": state.get("initial_budget_milestone_sha256"),
        "initial_budget_milestone": milestone,
        "budget_extensions": list(state.get("budget_extensions", [])),
        "commands": {
            "verify": (
                f"uv run --project minigpt-train minigpt-train verify --run-dir {run_dir} --deep"
            ),
            "continue": (
                f"uv run --project minigpt-train minigpt-train resume --run-dir {run_dir}"
            ),
        },
    }


def render_report(repository: RunRepository) -> str:
    manifest = read_json(repository.root / "RUN.json")
    _, state = repository.effective()
    progress = progress_summary(repository)
    milestone_sha256 = state.get("initial_budget_milestone_sha256")
    milestone = read_initial_budget_milestone(repository, state)
    if milestone is None:
        milestone_status = "not reached"
    else:
        milestone_status = (
            f"`{milestone_sha256}` at segment {milestone['segment_index']}, "
            f"step {milestone['global_step']}, "
            f"{float(milestone['accounted_active_seconds']) / 3600:.3f} active hours"
        )
    remaining = progress["estimated_remaining_seconds"]
    estimate = "unknown" if remaining is None else f"{remaining / 3600:.2f} hours"
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
        f"- Shard manifest: `{(state.get('shards') or {}).get('manifest_sha256')}`",
        "",
        "## Training status",
        "",
        f"- Phase: `{state['phase']}`",
        f"- Optimizer steps: {progress['global_step']} / {progress['total_steps']} "
        f"({progress['fraction_complete'] * 100:.2f}%)",
        f"- Completed segments: {progress['completed_segments']}",
        f"- Measured steps/second: {progress['measured_steps_per_second']}",
        f"- Estimated remaining: {estimate}",
        f"- Counted active time: {progress['active_used_hours']:.3f} / "
        f"{progress['active_budget_hours']:.3f} hours",
        f"- Initial-budget milestone: {milestone_status}",
        f"- Budget extensions: {len(state.get('budget_extensions', []))}",
        f"- Interrupted sessions recovered: {len(state.get('interruptions', []))}",
        "",
        "## Validation",
        "",
        f"- Best loss: {progress['best_validation_loss']} at step "
        f"{progress['best_validation_step']}",
        f"- Best perplexity: {progress['best_validation_perplexity']}",
        f"- Evaluations recorded: {progress['evaluations_recorded']}",
        f"- Early stopped: {progress['early_stopped']}",
        "",
        "## Published models",
        "",
    ]
    exports = state.get("exports", [])
    if not exports:
        lines.append("No ONNX model has been exported from this run yet.")
    else:
        lines.extend(
            f"- step {export['global_step']}: `{export['sha256']}` (`{export['path']}`)"
            for export in exports
        )
    lines.append("")
    return "\n".join(lines)


def write_report(repository: RunRepository, destination: Path | None) -> Path:
    destination = destination or repository.root / "report.md"
    atomic_write_bytes(destination, render_report(repository).encode("utf-8"))
    return destination


def gc_candidates(repository: RunRepository) -> list[Path]:
    """Superseded non-milestone checkpoints plus unsealed partial files."""

    config = read_json(repository.root / "config.resolved.json")
    _, state = repository.effective()
    protected = [
        repository.root / descriptor["path"]
        for key in ("current_checkpoint", "best_checkpoint", "recovery_checkpoint")
        if (descriptor := state.get(key)) is not None
    ]
    candidates = prunable_checkpoints(
        repository.root / CHECKPOINT_DIR,
        keep_last=int(config["training"]["checkpoint_keep_last"]),
        milestone_every=int(config["training"]["checkpoint_milestone_every_steps"]),
        protected=protected,
    )
    return sorted({*candidates, *repository.root.rglob("*.partial")})


def apply_gc(repository: RunRepository, candidates: list[Path]) -> list[str]:
    if repository.active_session_path.exists():
        raise IntegrityError("ACTIVE_SESSION exists; recover or finish the run before GC apply")
    removed: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(repository.root)
        except ValueError as error:
            raise IntegrityError(f"GC candidate escapes the run: {resolved}") from error
        if resolved.suffix == ".pt":
            remove_checkpoint(resolved)
        elif resolved.is_file():
            resolved.unlink()
        removed.append(str(relative))
    return removed
