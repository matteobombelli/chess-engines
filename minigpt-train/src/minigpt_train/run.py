"""Immutable run ledger, HEAD/RECOVERY pointers, and strict lineage rules."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import math
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .atomic import (
    AdvisoryLock,
    ObjectStore,
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    read_pointer,
    sha256_file,
    write_pointer,
)
from .config import ResolvedConfig, load_config
from .errors import ConfigError, IntegrityError

STATE_SCHEMA = "minigpt.run-state.v1"
RUN_SCHEMA = "minigpt.run-manifest.v1"
EXTENSION_SCHEMA = "minigpt.budget-extension.v1"
BUDGET_MILESTONE_SCHEMA = "minigpt.budget-milestone.v1"
# A segment boundary is the only point at which no optimizer state is in flight.
BOUNDARY_PHASES = {"ready_train", "complete"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise IntegrityError(f"invalid UTC timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        raise IntegrityError("timestamp lacks timezone")
    return parsed


def git_identity(worktree: Path) -> dict[str, Any]:
    def invoke(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=worktree, check=True, capture_output=True, text=True
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = invoke("rev-parse", "HEAD")
    # Ignored run/build artifacts stay excluded, but untracked source/config must block a run.
    status = invoke("status", "--porcelain", "--untracked-files=all")
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=worktree,
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256(b"minigpt-worktree-content-v1\0")
        for relative_bytes in sorted(filter(None, listed.split(b"\0"))):
            path = worktree / os.fsdecode(relative_bytes)
            if path.is_symlink():
                kind = b"symlink"
                mode = path.lstat().st_mode & 0o111
                payload = os.fsencode(os.readlink(path))
            elif path.is_file():
                kind = b"file"
                mode = path.stat().st_mode & 0o111
                payload = path.read_bytes()
            else:
                # A tracked path deleted from the worktree remains in ls-files.
                kind = b"missing"
                mode = 0
                payload = b""
            for field in (relative_bytes, kind, mode.to_bytes(2, "little"), payload):
                digest.update(len(field).to_bytes(8, "little"))
                digest.update(field)
        worktree_sha256: str | None = digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        worktree_sha256 = None
    return {
        "commit": commit,
        "tracked_dirty": bool(status) if status is not None else None,
        "worktree_sha256": worktree_sha256,
    }


def runtime_identity() -> dict[str, Any]:
    result = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    try:
        result["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        result["ram_bytes"] = None
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_runtime"] = torch.version.cuda
        result["cudnn"] = torch.backends.cudnn.version()
        result["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        result["torch"] = None
    try:
        result["onnxruntime"] = __import__("onnxruntime").__version__
    except ImportError:
        result["onnxruntime"] = None
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            # WSL exposes the host utility here without necessarily adding it
            # to PATH. Driver identity is resume-relevant, so do not silently
            # lose it on an otherwise GPU-visible host.
            wsl_nvidia_smi = Path("/usr/lib/wsl/lib/nvidia-smi")
            if wsl_nvidia_smi.is_file() and os.access(wsl_nvidia_smi, os.X_OK):
                nvidia_smi = str(wsl_nvidia_smi)
        if nvidia_smi is None:
            raise FileNotFoundError("nvidia-smi is unavailable")
        driver = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()
        result["nvidia_driver"] = driver[0].strip() if driver else None
    except (OSError, subprocess.SubprocessError):
        result["nvidia_driver"] = None
    return result


RUNTIME_COMPATIBILITY_FIELDS = (
    "python",
    "implementation",
    "platform",
    "machine",
    "torch",
    "cuda_runtime",
    "cudnn",
    "cuda_available",
    "gpu",
    "gpu_memory_bytes",
    "onnxruntime",
    "nvidia_driver",
)


def runtime_fingerprint(identity: dict[str, Any]) -> dict[str, Any]:
    """Determinism-relevant runtime subset; host, PID, CPU count, and RAM are provenance only."""

    return {field: identity.get(field) for field in RUNTIME_COMPATIBILITY_FIELDS}


class RunRepository:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.store = ObjectStore(self.root / "objects")
        self.pointers = self.root / "pointers"
        self.head_path = self.pointers / "HEAD"
        self.recovery_path = self.pointers / "RECOVERY"
        self.lock_path = self.root / "run.lock"
        self.active_session_path = self.root / "ACTIVE_SESSION.json"
        self.metrics_path = self.root / "metrics.jsonl"

    @classmethod
    def create(
        cls,
        root: Path | str,
        config: ResolvedConfig,
        *,
        worktree: Path,
        parent: dict[str, Any] | None = None,
    ) -> "RunRepository":
        repository = cls(root)
        if repository.root.exists() and any(repository.root.iterdir()):
            raise IntegrityError(f"new run directory is not empty: {repository.root}")
        repository.root.mkdir(parents=True, exist_ok=True)
        for child in (
            "objects",
            "pointers",
            "artifacts/checkpoints",
            "artifacts/models",
            "quarantine",
        ):
            (repository.root / child).mkdir(parents=True, exist_ok=True)
        frozen_config = repository.root / "config.toml"
        atomic_write_bytes(frozen_config, config.source.read_bytes(), mode=0o444)
        atomic_write_json(repository.root / "config.resolved.json", config.values)
        manifest = {
            "schema": RUN_SCHEMA,
            "run_id": str(uuid.uuid4()),
            "name": config.values["run"]["name"],
            "created_at": utc_now(),
            "config_sha256": sha256_file(frozen_config),
            "semantic_hash": config.semantic_hash,
            "disposable": bool(config.values["run"]["disposable"]),
            "git": git_identity(worktree),
            "runtime": runtime_identity(),
            "locks": {
                "cargo_lock_sha256": (
                    sha256_file(worktree / "Cargo.lock")
                    if (worktree / "Cargo.lock").is_file()
                    else None
                ),
                "uv_lock_sha256": (
                    sha256_file(worktree / "minigpt-train" / "uv.lock")
                    if (worktree / "minigpt-train" / "uv.lock").is_file()
                    else None
                ),
            },
            "parent": parent,
        }
        atomic_write_json(repository.root / "RUN.json", manifest)
        initial_budget_seconds = float(config.values["run"]["active_budget_hours"]) * 3600
        initial = {
            "schema": STATE_SCHEMA,
            "run_id": manifest["run_id"],
            "sequence": 0,
            "previous_state_hash": None,
            "head_base_hash": None,
            "updated_at": utc_now(),
            "phase": "initialized",
            "segment_index": 0,
            "global_step": 0,
            "active_used_seconds": 0.0,
            "active_budget_seconds": initial_budget_seconds,
            "initial_active_budget_seconds": initial_budget_seconds,
            "initial_budget_milestone_sha256": None,
            "budget_extensions": [],
            "current_checkpoint": None,
            "best_checkpoint": None,
            "recovery_checkpoint": None,
            "shards": None,
            "best_validation_loss": None,
            "best_validation_step": None,
            "evaluations_without_improvement": 0,
            "early_stopped": False,
            "completed_segments": [],
            "exports": [],
            "interruptions": [],
        }
        digest = repository.store.put(initial)
        write_pointer(repository.head_path, digest)
        write_pointer(repository.recovery_path, digest)
        return repository

    @classmethod
    def open(cls, root: Path | str) -> tuple["RunRepository", ResolvedConfig]:
        repository = cls(root)
        if not (repository.root / "RUN.json").is_file():
            raise IntegrityError(f"not a MiniGPT run: {repository.root}")
        manifest = read_json(repository.root / "RUN.json")
        if manifest.get("schema") != RUN_SCHEMA:
            raise IntegrityError("unsupported run manifest")
        config = load_config(repository.root / "config.toml")
        if sha256_file(repository.root / "config.toml") != manifest.get("config_sha256"):
            raise IntegrityError("frozen config checksum mismatch")
        if config.semantic_hash != manifest.get("semantic_hash"):
            raise IntegrityError("frozen semantic config hash mismatch")
        repository.head()
        repository.recovery()
        return repository, config

    def lock(self) -> AdvisoryLock:
        return AdvisoryLock(self.lock_path)

    def _state_at(self, pointer: Path) -> tuple[str, dict[str, Any]]:
        digest = read_pointer(pointer)
        state = self.store.get(digest)
        if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
            raise IntegrityError(f"pointer {pointer.name} does not reference a run state")
        run_id = read_json(self.root / "RUN.json").get("run_id")
        if state.get("run_id") != run_id:
            raise IntegrityError(f"pointer {pointer.name} references another run")
        return digest, state

    def head(self) -> tuple[str, dict[str, Any]]:
        return self._state_at(self.head_path)

    def recovery(self) -> tuple[str, dict[str, Any]]:
        return self._state_at(self.recovery_path)

    def effective(self) -> tuple[str, dict[str, Any]]:
        head_hash, head = self.head()
        recovery_hash, recovery = self.recovery()
        if recovery_hash == head_hash:
            return head_hash, head
        if (
            head.get("head_base_hash") is None
            and head.get("previous_state_hash") == recovery_hash
            and int(head.get("sequence", -1)) > int(recovery.get("sequence", -1))
        ):
            # commit_head publishes HEAD before repairing RECOVERY. A crash in between
            # leaves this unambiguous one-step relation; HEAD is already authoritative.
            return head_hash, head
        if recovery.get("head_base_hash") != head_hash:
            raise IntegrityError("RECOVERY was not derived from the current HEAD")
        if int(recovery.get("sequence", -1)) <= int(head.get("sequence", -1)):
            raise IntegrityError("RECOVERY is not newer than HEAD")
        return recovery_hash, recovery

    def commit_head(self, value: dict[str, Any], *, base_hash: str | None = None) -> str:
        prior_hash, prior = self.head()
        if base_hash is not None and prior_hash != base_hash:
            raise IntegrityError("HEAD changed during transaction")
        effective_hash, effective = self.effective()
        state = copy.deepcopy(value)
        state.update(
            {
                "schema": STATE_SCHEMA,
                "run_id": prior["run_id"],
                "sequence": max(int(prior["sequence"]), int(effective["sequence"])) + 1,
                "previous_state_hash": effective_hash,
                "head_base_hash": None,
                "updated_at": utc_now(),
            }
        )
        digest = self.store.put(state)
        write_pointer(self.head_path, digest)
        write_pointer(self.recovery_path, digest)
        return digest

    def commit_recovery(self, value: dict[str, Any]) -> str:
        head_hash, head = self.head()
        _, prior = self.recovery()
        state = copy.deepcopy(value)
        state.update(
            {
                "schema": STATE_SCHEMA,
                "run_id": head["run_id"],
                "sequence": max(int(head["sequence"]), int(prior["sequence"])) + 1,
                "previous_state_hash": read_pointer(self.recovery_path),
                "head_base_hash": head_hash,
                "updated_at": utc_now(),
            }
        )
        digest = self.store.put(state)
        write_pointer(self.recovery_path, digest)
        return digest

    def verify_state_chain(self, *, deep: bool) -> dict[str, int]:
        visited: set[str] = set()
        pending = [read_pointer(self.head_path), read_pointer(self.recovery_path)]
        while pending:
            digest = pending.pop()
            if digest in visited:
                continue
            value = self.store.get(digest)
            if value.get("run_id") != read_json(self.root / "RUN.json").get("run_id"):
                raise IntegrityError("state chain crosses run identity")
            visited.add(digest)
            if deep and value.get("previous_state_hash") is not None:
                pending.append(value["previous_state_hash"])
        return {"state_objects": len(visited)}

    def begin_session(self, command: str, heartbeat_seconds: int) -> "ActiveSession":
        if self.active_session_path.exists():
            value = read_json(self.active_session_path)
            raise IntegrityError(
                f"unfinished active session {value.get('session_id', 'unknown')}; run recover first"
            )
        return ActiveSession(self, command, heartbeat_seconds)


def budget_exhausted(state: dict[str, Any], *, additional_seconds: float = 0.0) -> bool:
    try:
        additional = float(additional_seconds)
        used = float(state["active_used_seconds"]) + max(0.0, additional)
        budget = float(state["active_budget_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("run contains invalid active-time accounting") from error
    if (
        not math.isfinite(additional)
        or not math.isfinite(used)
        or not math.isfinite(budget)
        or used < 0
        or budget <= 0
    ):
        raise IntegrityError("run contains invalid active-time accounting")
    return used >= budget


def safe_budget_boundary(state: dict[str, Any]) -> bool:
    """A completed segment boundary at which training may stop."""

    return state.get("phase") in BOUNDARY_PHASES


def attach_initial_budget_milestone(
    repository: RunRepository, state: dict[str, Any]
) -> dict[str, Any]:
    """Attach the first-budget snapshot without creating an object/state hash cycle."""

    existing = state.get("initial_budget_milestone_sha256")
    if existing is not None:
        # Loading here makes every attempted reuse fail closed on a corrupt reference.
        read_initial_budget_milestone(repository, state)
        return state
    try:
        initial_budget = float(state.get("initial_active_budget_seconds", -1))
        used = float(state.get("active_used_seconds", -1))
        active_budget = float(state["active_budget_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("run lacks valid initial active-budget accounting") from error
    if (
        not math.isfinite(initial_budget)
        or not math.isfinite(used)
        or initial_budget <= 0
        or used < 0
    ):
        raise IntegrityError("run lacks valid initial active-budget accounting")
    if used < initial_budget or not safe_budget_boundary(state):
        return state
    if state.get("budget_extensions") or active_budget != initial_budget:
        raise IntegrityError("initial budget was changed before its milestone was sealed")

    parent_state_sha256, _ = repository.effective()
    milestone = {
        "schema": BUDGET_MILESTONE_SCHEMA,
        "run_id": state["run_id"],
        "created_at": utc_now(),
        # This is the already-existing parent, never the state that will reference this object.
        "parent_state_sha256": parent_state_sha256,
        "initial_budget_seconds": initial_budget,
        "accounted_active_seconds": used,
        "overshoot_seconds": used - initial_budget,
        "phase": state["phase"],
        "segment_index": int(state["segment_index"]),
        "global_step": int(state["global_step"]),
        "current_checkpoint": copy.deepcopy(state.get("current_checkpoint")),
        "best_checkpoint": copy.deepcopy(state.get("best_checkpoint")),
        "completed_segment_count": len(state.get("completed_segments", [])),
    }
    milestone_sha256 = repository.store.put(milestone)
    marked = copy.deepcopy(state)
    marked["initial_budget_milestone_sha256"] = milestone_sha256
    return marked


def read_initial_budget_milestone(
    repository: RunRepository, state: dict[str, Any]
) -> dict[str, Any] | None:
    reference = state.get("initial_budget_milestone_sha256")
    if reference is None:
        return None
    if not isinstance(reference, str):
        raise IntegrityError("initial budget milestone reference is invalid")
    milestone = repository.store.get(reference)
    expected_fields = {
        "schema",
        "run_id",
        "created_at",
        "parent_state_sha256",
        "initial_budget_seconds",
        "accounted_active_seconds",
        "overshoot_seconds",
        "phase",
        "segment_index",
        "global_step",
        "current_checkpoint",
        "best_checkpoint",
        "completed_segment_count",
    }
    if not isinstance(milestone, dict) or set(milestone) != expected_fields:
        raise IntegrityError("initial budget milestone has an invalid schema")
    if (
        milestone.get("schema") != BUDGET_MILESTONE_SCHEMA
        or milestone.get("run_id") != state.get("run_id")
        or milestone.get("phase") not in BOUNDARY_PHASES
        or milestone.get("initial_budget_seconds") != state.get("initial_active_budget_seconds")
    ):
        raise IntegrityError("initial budget milestone disagrees with the run")
    if not isinstance(milestone.get("created_at"), str):
        raise IntegrityError("initial budget milestone timestamp is invalid")
    parse_utc(milestone["created_at"])
    parent_hash = milestone.get("parent_state_sha256")
    if not isinstance(parent_hash, str):
        raise IntegrityError("initial budget milestone parent reference is invalid")
    if parent_hash == reference:
        raise IntegrityError("initial budget milestone is self-referential")
    parent = repository.store.get(parent_hash)
    if parent.get("schema") != STATE_SCHEMA or parent.get("run_id") != state.get("run_id"):
        raise IntegrityError("initial budget milestone parent is not a state from this run")
    try:
        initial_budget = float(milestone["initial_budget_seconds"])
        accounted = float(milestone["accounted_active_seconds"])
        overshoot = float(milestone["overshoot_seconds"])
    except (TypeError, ValueError) as error:
        raise IntegrityError("initial budget milestone accounting is invalid") from error
    if (
        not math.isfinite(initial_budget)
        or not math.isfinite(accounted)
        or not math.isfinite(overshoot)
        or initial_budget <= 0
        or accounted < initial_budget
        or overshoot != accounted - initial_budget
    ):
        raise IntegrityError("initial budget milestone accounting is invalid")
    return milestone


def verify_budget_ledger(repository: RunRepository, state: dict[str, Any]) -> int:
    """Verify the immutable initial marker and ordered extension arithmetic."""

    try:
        config = read_json(repository.root / "config.resolved.json")
        expected_initial = float(config["run"]["active_budget_hours"]) * 3600
        recorded_initial = float(state["initial_active_budget_seconds"])
        recorded_budget = float(state["active_budget_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("run budget ledger is malformed") from error
    if (
        not math.isfinite(expected_initial)
        or not math.isfinite(recorded_initial)
        or not math.isfinite(recorded_budget)
        or expected_initial <= 0
        or recorded_initial != expected_initial
    ):
        raise IntegrityError("initial active budget differs from the frozen configuration")

    milestone = read_initial_budget_milestone(repository, state)
    extensions = state.get("budget_extensions")
    if not isinstance(extensions, list) or not all(isinstance(item, str) for item in extensions):
        raise IntegrityError("budget extension references are invalid")
    if extensions and milestone is None:
        raise IntegrityError("budget was extended without sealing the initial milestone")
    running_budget = recorded_initial
    checked = 1 if milestone is not None else 0
    expected_extension_fields = {
        "schema",
        "created_at",
        "additional_seconds",
        "reason",
        "prior_budget_seconds",
        "initial_budget_milestone_sha256",
    }
    for reference in extensions:
        extension = repository.store.get(reference)
        if not isinstance(extension, dict) or set(extension) != expected_extension_fields:
            raise IntegrityError("budget extension has an invalid schema")
        try:
            additional = float(extension["additional_seconds"])
            prior = float(extension["prior_budget_seconds"])
        except (TypeError, ValueError) as error:
            raise IntegrityError("budget extension accounting is invalid") from error
        if (
            extension.get("schema") != EXTENSION_SCHEMA
            or not isinstance(extension.get("created_at"), str)
            or not isinstance(extension.get("reason"), str)
            or not extension["reason"].strip()
            or extension.get("initial_budget_milestone_sha256")
            != state.get("initial_budget_milestone_sha256")
            or not math.isfinite(additional)
            or additional <= 0
            or prior != running_budget
        ):
            raise IntegrityError("budget extension disagrees with the run ledger")
        parse_utc(extension["created_at"])
        running_budget += additional
        checked += 1
    if recorded_budget != running_budget:
        raise IntegrityError("active budget does not equal its initial value plus extensions")
    try:
        active_used = float(state["active_used_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("run active-time accounting is invalid") from error
    if not math.isfinite(active_used) or active_used < 0:
        raise IntegrityError("run active-time accounting is invalid")
    if safe_budget_boundary(state) and active_used >= recorded_initial and milestone is None:
        raise IntegrityError("initial budget was reached without an immutable milestone")
    return checked


class ActiveSession:
    """Persisted active-time heartbeat; crash undercount is bounded by one heartbeat."""

    def __init__(self, repository: RunRepository, command: str, heartbeat_seconds: int):
        self.repository = repository
        self.command = command
        self.heartbeat_seconds = max(1, heartbeat_seconds)
        self.session_id = str(uuid.uuid4())
        self.started_monotonic = time.monotonic()
        self.started_at = utc_now()
        self.base_active_seconds = float(repository.effective()[1]["active_used_seconds"])
        self.last_write = self.started_monotonic
        self.closed = False
        self._mutex = threading.Lock()
        self._stop = threading.Event()
        self._write(0.0)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"minigpt-heartbeat-{self.session_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.heartbeat(force=True)

    def _write(self, elapsed: float) -> None:
        atomic_write_json(
            self.repository.active_session_path,
            {
                "schema": "minigpt.active-session.v1",
                "session_id": self.session_id,
                "command": self.command,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": self.started_at,
                "base_active_seconds": self.base_active_seconds,
                "last_heartbeat_at": utc_now(),
                "recorded_elapsed_seconds": elapsed,
            },
        )

    def heartbeat(self, *, force: bool = False) -> None:
        with self._mutex:
            if self.closed:
                return
            now = time.monotonic()
            if force or now - self.last_write >= self.heartbeat_seconds:
                self._write(now - self.started_monotonic)
                self.last_write = now

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started_monotonic)

    def seal(self) -> float:
        """Stop heartbeats but preserve the session until its elapsed time is committed."""
        if self.closed:
            return self.elapsed
        self._stop.set()
        self._thread.join(timeout=self.heartbeat_seconds + 1)
        self.heartbeat(force=True)
        elapsed = self.elapsed
        with self._mutex:
            self.closed = True
        return elapsed

    def clear(self) -> None:
        """Remove a sealed session only after the accounting state is durable."""
        if not self.closed:
            raise IntegrityError("cannot clear a live active session")
        self.repository.active_session_path.unlink()

    def abandon(self) -> None:
        """Seal the last heartbeat but retain the session as an explicit recovery gate."""
        if self.closed:
            return
        self._stop.set()
        self._thread.join(timeout=self.heartbeat_seconds + 1)
        self.heartbeat(force=True)
        with self._mutex:
            self.closed = True


def recover_interrupted(repository: RunRepository, *, force: bool = False) -> dict[str, Any]:
    if not repository.active_session_path.exists():
        _, state = repository.effective()
        return state
    session = read_json(repository.active_session_path)
    same_host = session.get("host") == socket.gethostname()
    pid = session.get("pid")
    alive = False
    if same_host and isinstance(pid, int):
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            # EPERM proves that the process exists even though this user cannot signal it.
            alive = True
    if alive and not force:
        raise IntegrityError(
            f"active session still belongs to live pid {pid}; use --force only after inspection"
        )
    _, state = repository.effective()
    session_id = session.get("session_id")
    if any(
        interruption.get("session_id") == session_id
        for interruption in state.get("interruptions", [])
    ):
        # The state commit won a prior recovery attempt but deleting ACTIVE_SESSION did not.
        # Its recorded accounting/quarantine transaction is already authoritative.
        repository.active_session_path.unlink()
        return state
    state = copy.deepcopy(state)
    quarantine = repository.root / "quarantine" / f"recovery-{uuid.uuid4()}"
    quarantined: list[str] = []

    def quarantine_path(path: Path) -> None:
        if not path.exists():
            return
        quarantine.mkdir(parents=True, exist_ok=True)
        destination = quarantine / f"{len(quarantined):04d}-{path.name}"
        os.replace(path, destination)
        quarantined.append(str(destination.relative_to(repository.root)))

    # A final filename is never selected by mtime. Unsealed partials are preserved
    # under quarantine and the recovery checkpoint regenerates their work.
    for partial in list(repository.root.rglob("*.partial")):
        if quarantine in partial.parents:
            continue
        quarantine_path(partial)
    recorded = max(0.0, float(session.get("recorded_elapsed_seconds", 0.0)))
    base_active = max(0.0, float(session.get("base_active_seconds", 0.0)))
    already_counted = float(state["active_used_seconds"])
    recovered_delta = max(0.0, base_active + recorded - already_counted)
    state["active_used_seconds"] = already_counted + recovered_delta
    state.setdefault("interruptions", []).append(
        {
            "session_id": session_id,
            "command": session.get("command"),
            "started_at": session.get("started_at"),
            "last_heartbeat_at": session.get("last_heartbeat_at"),
            "counted_active_seconds": recovered_delta,
            "quarantined": quarantined,
            "recovered_at": utc_now(),
        }
    )
    state = attach_initial_budget_milestone(repository, state)
    repository.commit_head(state)
    # Commit first. If this unlink fails, the same-session branch above makes retry idempotent.
    repository.active_session_path.unlink()
    return repository.head()[1]


def extend_budget(repository: RunRepository, seconds: float, reason: str) -> dict[str, Any]:
    if repository.active_session_path.exists():
        raise IntegrityError("unfinished active session exists; recover it before extending budget")
    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as error:
        raise ConfigError("additional budget must be a finite positive number") from error
    if not math.isfinite(seconds) or not seconds > 0:
        raise ConfigError("additional budget must be positive")
    if not isinstance(reason, str) or not reason.strip():
        raise ConfigError("budget extension requires a reason")
    _, state = repository.effective()
    if not budget_exhausted(state):
        raise ConfigError("active-time budget is not exhausted; extension is premature")
    if not safe_budget_boundary(state):
        raise ConfigError("finish the in-flight segment at its safe boundary before extending")
    state = attach_initial_budget_milestone(repository, state)
    milestone_sha256 = state.get("initial_budget_milestone_sha256")
    if not isinstance(milestone_sha256, str):
        raise IntegrityError("initial budget milestone was not sealed before extension")
    extension = {
        "schema": EXTENSION_SCHEMA,
        "created_at": utc_now(),
        "additional_seconds": seconds,
        "reason": reason.strip(),
        "prior_budget_seconds": float(state["active_budget_seconds"]),
        "initial_budget_milestone_sha256": milestone_sha256,
    }
    extension_hash = repository.store.put(extension)
    state = copy.deepcopy(state)
    state["active_budget_seconds"] = float(state["active_budget_seconds"]) + seconds
    state.setdefault("budget_extensions", []).append(extension_hash)
    repository.commit_head(state)
    return extension


def fork_run(
    source: RunRepository,
    destination: Path,
    config: ResolvedConfig,
    *,
    worktree: Path,
    reason: str,
) -> RunRepository:
    if not reason.strip():
        raise ConfigError("fork requires a reason")
    source_manifest = read_json(source.root / "RUN.json")
    _, source_state = source.effective()
    # A fork warm-starts from the best model the parent produced, not its last step.
    checkpoint = source_state.get("best_checkpoint") or source_state.get("current_checkpoint")
    parent = {
        "run_id": source_manifest["run_id"],
        "source_head": read_pointer(source.head_path),
        "semantic_hash": source_manifest["semantic_hash"],
        "relationship": "weights-only-warm-start",
        "reason": reason.strip(),
    }
    child = RunRepository.create(destination, config, worktree=worktree, parent=parent)
    if checkpoint is None:
        return child
    source_path = source.root / checkpoint["path"]
    if not source_path.is_file() or sha256_file(source_path) != checkpoint["sha256"]:
        raise IntegrityError("source checkpoint is missing or corrupt")
    target = child.root / "artifacts" / "checkpoints" / source_path.name
    shutil.copy2(source_path, target)
    if sha256_file(target) != checkpoint["sha256"]:
        raise IntegrityError("forked checkpoint copy failed checksum validation")
    _, state = child.head()
    state["current_checkpoint"] = {
        **checkpoint,
        "path": str(target.relative_to(child.root)),
        "warm_start_from_semantic_hash": source_manifest["semantic_hash"],
    }
    state["phase"] = "warm_start_ready"
    child.commit_head(state)
    return child
