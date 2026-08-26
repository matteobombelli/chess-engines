"""One deliberately narrow operations CLI for the complete run lifecycle."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from .config import load_config
from .errors import MiniGptError
from .operations import (
    PUBLISHED_RELATIVE,
    apply_gc,
    doctor,
    export_best,
    gc_candidates,
    progress_summary,
    reproduction_record,
    verify_run,
    write_report,
)
from .run import RunRepository, extend_budget, fork_run, git_identity, recover_interrupted
from .segments import run_training


def duration_seconds(value: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    value = value.strip().lower()
    if len(value) < 2 or value[-1] not in units:
        raise argparse.ArgumentTypeError("duration must end in s, m, h, or d (for example 24h)")
    try:
        amount = float(value[:-1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid duration: {value}") from error
    if amount <= 0:
        raise argparse.ArgumentTypeError("duration must be positive")
    return amount * units[value[-1]]


def _worktree(value: str | None) -> Path:
    return Path(value or Path(__file__).resolve().parents[3]).resolve()


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minigpt-train")
    parser.add_argument("--worktree", help="repository root (normally auto-detected)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="validate environment and shards without starting a run"
    )
    doctor_parser.add_argument("--config", type=Path, required=True)
    doctor_parser.add_argument(
        "--production", action="store_true", help="treat optional pilot warnings as failures"
    )

    start_parser = subparsers.add_parser("start", help="create a new run and begin training")
    start_parser.add_argument("--config", type=Path, required=True)
    start_parser.add_argument("--run-dir", type=Path, required=True)
    start_parser.add_argument(
        "--initialize-only", action="store_true", help="publish the seeded step 0, then stop"
    )
    start_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="test-only: initialize ledger without requiring PyTorch/ONNX",
    )
    start_parser.add_argument("--one-segment", action="store_true")

    resume_parser = subparsers.add_parser("resume", help="continue with the frozen configuration")
    resume_parser.add_argument("--run-dir", type=Path, required=True)
    resume_parser.add_argument("--one-segment", action="store_true")

    extend_parser = subparsers.add_parser(
        "extend", help="append active-time budget without changing identity"
    )
    extend_parser.add_argument("--run-dir", type=Path, required=True)
    extend_parser.add_argument("--additional-active-budget", type=duration_seconds, required=True)
    extend_parser.add_argument("--reason", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="validate pointers, lineage, and artifacts"
    )
    verify_parser.add_argument("--run-dir", type=Path, required=True)
    verify_parser.add_argument("--deep", action="store_true")

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="emit exact run identity and replay commands"
    )
    reproduce_parser.add_argument("--run-dir", type=Path, required=True)

    fork_parser = subparsers.add_parser(
        "fork", help="start a parent-linked weights-only experiment"
    )
    fork_parser.add_argument("--source-run", type=Path, required=True)
    fork_parser.add_argument("--config", type=Path, required=True)
    fork_parser.add_argument("--run-dir", type=Path, required=True)
    fork_parser.add_argument("--reason", required=True)

    recover_parser = subparsers.add_parser(
        "recover", help="close a crashed session at its durable heartbeat"
    )
    recover_parser.add_argument("--run-dir", type=Path, required=True)
    recover_parser.add_argument("--force", action="store_true")

    export_parser = subparsers.add_parser(
        "export", help="publish the best-validation checkpoint as a verified ONNX model"
    )
    export_parser.add_argument("--run-dir", type=Path, required=True)
    export_parser.add_argument(
        "--publish-dir", type=Path, help="defaults to <worktree>/artifacts/minigpt/current"
    )

    gc_parser = subparsers.add_parser(
        "gc", help="list superseded non-milestone checkpoints and partial files"
    )
    gc_parser.add_argument("--run-dir", type=Path, required=True)
    gc_parser.add_argument("--apply", action="store_true")

    report_parser = subparsers.add_parser("report", help="render a factual report from the ledger")
    report_parser.add_argument("--run-dir", type=Path, required=True)
    report_parser.add_argument("--output", type=Path)
    return parser


def dispatch(arguments: argparse.Namespace) -> int:
    worktree = _worktree(arguments.worktree)
    if arguments.command == "doctor":
        report = doctor(
            load_config(arguments.config), worktree=worktree, production=arguments.production
        )
        _json(report)
        return 1 if report["failures"] else 0

    if arguments.command == "start":
        config = load_config(arguments.config)
        if not arguments.metadata_only:
            identity = git_identity(worktree)
            if (
                identity["commit"] is None
                or identity["tracked_dirty"] is None
                or identity["worktree_sha256"] is None
            ):
                raise MiniGptError("start could not establish an exact Git/worktree identity")
            if identity["tracked_dirty"] and not config.values["run"]["disposable"]:
                raise MiniGptError("a non-disposable run requires a committed clean worktree")
            if not (worktree / "minigpt-train" / "uv.lock").is_file():
                raise MiniGptError("start requires minigpt-train/uv.lock")
        repository = RunRepository.create(arguments.run_dir, config, worktree=worktree)
        if arguments.metadata_only:
            _json(
                {
                    "run_dir": str(repository.root),
                    "head": repository.head()[0],
                    "phase": "initialized",
                }
            )
            return 0
        with repository.lock():
            state = run_training(
                repository,
                config,
                worktree=worktree,
                one_segment=arguments.one_segment,
                initialize_only=arguments.initialize_only,
            )
        _json(state)
        return 0

    if arguments.command == "fork":
        source, _ = RunRepository.open(arguments.source_run)
        child_config = load_config(arguments.config)
        with source.lock():
            child = fork_run(
                source,
                arguments.run_dir,
                child_config,
                worktree=worktree,
                reason=arguments.reason,
            )
        _json({"run_dir": str(child.root), "head": child.head()[0]})
        return 0

    repository, config = RunRepository.open(arguments.run_dir)
    if arguments.command == "resume":
        with repository.lock():
            state = run_training(
                repository, config, worktree=worktree, one_segment=arguments.one_segment
            )
        _json(state)
    elif arguments.command == "extend":
        with repository.lock():
            _json(extend_budget(repository, arguments.additional_active_budget, arguments.reason))
    elif arguments.command == "verify":
        with repository.lock():
            _json(verify_run(repository, config, deep=arguments.deep, worktree=worktree))
    elif arguments.command == "reproduce":
        with repository.lock():
            verify_run(repository, config, deep=True, worktree=worktree)
            _json(reproduction_record(repository))
    elif arguments.command == "recover":
        with repository.lock():
            _json(recover_interrupted(repository, force=arguments.force))
    elif arguments.command == "export":
        publish_root = arguments.publish_dir or (worktree / PUBLISHED_RELATIVE)
        with repository.lock():
            _json(export_best(repository, config, publish_root=publish_root))
    elif arguments.command == "gc":
        with repository.lock():
            candidates = gc_candidates(repository)
            removed = apply_gc(repository, candidates) if arguments.apply else []
            _json(
                {
                    "applied": arguments.apply,
                    "active_session_present": repository.active_session_path.exists(),
                    "candidates": [str(path) for path in candidates],
                    "removed": removed,
                }
            )
    elif arguments.command == "report":
        with repository.lock():
            destination = write_report(repository, arguments.output)
            summary = progress_summary(repository)
        _json({"report": str(destination), "progress": summary})
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    # Convert supervisor SIGTERM into the same controlled unwind as Ctrl-C, so the
    # durable pointers and heartbeat survive and recovery stays explicit.
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        raise SystemExit(dispatch(parser.parse_args(argv)))
    except MiniGptError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except KeyboardInterrupt as error:
        print(
            "interrupted safely; run `minigpt-train recover --run-dir ...`, then verify and resume",
            file=sys.stderr,
        )
        raise SystemExit(130) from error
