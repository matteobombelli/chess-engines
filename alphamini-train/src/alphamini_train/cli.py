"""One deliberately narrow operations CLI for the complete run lifecycle."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from .config import load_config
from .drills import run_cpu_serving_benchmark, run_recovery_drill
from .errors import AlphaMiniError
from .evaluation import fit_ladder_file
from .operations import (
    apply_gc,
    doctor,
    gc_candidates,
    production_benchmark_report,
    reproduction_record,
    verify_run,
    write_report,
)
from .orchestrator import run_training
from .run import RunRepository, extend_budget, fork_run, recover_interrupted
from .run import git_identity


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
    parser = argparse.ArgumentParser(prog="alphamini-train")
    parser.add_argument("--worktree", help="repository root (normally auto-detected)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="validate environment without starting a run"
    )
    doctor_parser.add_argument("--config", type=Path, required=True)
    doctor_parser.add_argument(
        "--production", action="store_true", help="treat optional pilot warnings as failures"
    )

    start_parser = subparsers.add_parser("start", help="create a new run and begin training")
    start_parser.add_argument("--config", type=Path, required=True)
    start_parser.add_argument("--run-dir", type=Path, required=True)
    start_parser.add_argument(
        "--initialize-only", action="store_true", help="publish seeded M0, then stop"
    )
    start_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="test-only: initialize ledger without requiring PyTorch/ONNX",
    )
    start_parser.add_argument("--one-cycle", action="store_true")

    resume_parser = subparsers.add_parser("resume", help="continue with the frozen configuration")
    resume_parser.add_argument("--run-dir", type=Path, required=True)
    resume_parser.add_argument("--one-cycle", action="store_true")

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

    gc_parser = subparsers.add_parser("gc", help="list disposable partial files and tensor caches")
    gc_parser.add_argument("--run-dir", type=Path, required=True)
    gc_parser.add_argument("--apply", action="store_true")
    gc_parser.add_argument("--backup-marker", type=Path)

    report_parser = subparsers.add_parser("report", help="render a factual report from the ledger")
    report_parser.add_argument("--run-dir", type=Path, required=True)
    report_parser.add_argument("--output", type=Path)

    benchmark_parser = subparsers.add_parser(
        "benchmark-report",
        help="deep-verify and score a production-shaped disposable benchmark",
    )
    benchmark_parser.add_argument("--run-dir", type=Path, required=True)

    ladder_parser = subparsers.add_parser(
        "ladder",
        help="verify arena pair logs (or aggregate matches) and fit Bradley-Terry ratings",
    )
    ladder_parser.add_argument("--input", type=Path, required=True)
    ladder_parser.add_argument("--output", type=Path, required=True)

    recovery_drill_parser = subparsers.add_parser(
        "drill-recovery",
        help="signal, recover, verify, and resume one disposable target-GPU cycle",
    )
    recovery_drill_parser.add_argument("--config", type=Path, required=True)
    recovery_drill_parser.add_argument("--run-dir", type=Path, required=True)
    recovery_drill_parser.add_argument("--evidence", type=Path, required=True)
    recovery_drill_parser.add_argument(
        "--phase", choices=("collection", "training"), required=True
    )
    recovery_drill_parser.add_argument("--control-run-dir", type=Path)
    recovery_drill_parser.add_argument("--timeout-seconds", type=float, default=180.0)

    cpu_benchmark_parser = subparsers.add_parser(
        "benchmark-cpu",
        help="measure the fixed CPU search budget at inference batches 1, 4, and 8",
    )
    cpu_benchmark_parser.add_argument("--arena", type=Path, required=True)
    cpu_benchmark_parser.add_argument("--model", type=Path, required=True)
    cpu_benchmark_parser.add_argument("--manifest", type=Path, required=True)
    cpu_benchmark_parser.add_argument("--openings", type=Path, required=True)
    cpu_benchmark_parser.add_argument("--output-dir", type=Path, required=True)
    cpu_benchmark_parser.add_argument("--simulations", type=int, default=10_000)
    cpu_benchmark_parser.add_argument("--time-ms", type=int, default=9_000)
    cpu_benchmark_parser.add_argument("--opening-pairs", type=int, default=2)
    return parser


def dispatch(arguments: argparse.Namespace) -> int:
    worktree = _worktree(arguments.worktree)
    if arguments.command == "doctor":
        report = doctor(
            load_config(arguments.config), worktree=worktree, production=arguments.production
        )
        _json(report)
        return 1 if report["failures"] else 0

    if arguments.command == "ladder":
        _json(fit_ladder_file(arguments.input, arguments.output))
        return 0

    if arguments.command == "drill-recovery":
        if arguments.timeout_seconds <= 0:
            raise AlphaMiniError("--timeout-seconds must be positive")
        _json(
            run_recovery_drill(
                config_path=arguments.config,
                run_dir=arguments.run_dir,
                evidence_path=arguments.evidence,
                phase=arguments.phase,
                worktree=worktree,
                control_run_dir=arguments.control_run_dir,
                timeout_seconds=arguments.timeout_seconds,
            )
        )
        return 0

    if arguments.command == "benchmark-cpu":
        report = run_cpu_serving_benchmark(
            arena=arguments.arena,
            model=arguments.model,
            manifest=arguments.manifest,
            openings=arguments.openings,
            output_dir=arguments.output_dir,
            worktree=worktree,
            simulations=arguments.simulations,
            time_ms=arguments.time_ms,
            opening_pairs=arguments.opening_pairs,
        )
        _json(report)
        return 0 if report["passed"] else 1

    if arguments.command == "start":
        config = load_config(arguments.config)
        if not arguments.metadata_only:
            if not config.values["training"]["horizon_confirmed"]:
                raise AlphaMiniError(
                    "training horizon is not confirmed; freeze it from pilot evidence before start"
                )
            identity = git_identity(worktree)
            if (
                identity["commit"] is None
                or identity["tracked_dirty"] is None
                or identity["worktree_sha256"] is None
            ):
                raise AlphaMiniError("start could not establish an exact Git/worktree identity")
            if identity["tracked_dirty"] and not config.values["run"]["disposable"]:
                raise AlphaMiniError("a non-disposable run requires a committed clean worktree")
            if (
                not (worktree / "Cargo.lock").is_file()
                or not (worktree / "alphamini-train" / "uv.lock").is_file()
            ):
                raise AlphaMiniError("start requires Cargo.lock and alphamini-train/uv.lock")
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
                one_cycle=arguments.one_cycle,
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
                repository, config, worktree=worktree, one_cycle=arguments.one_cycle
            )
        _json(state)
    elif arguments.command == "extend":
        with repository.lock():
            _json(extend_budget(repository, arguments.additional_active_budget, arguments.reason))
    elif arguments.command == "verify":
        with repository.lock():
            _json(verify_run(repository, deep=arguments.deep))
    elif arguments.command == "reproduce":
        with repository.lock():
            verify_run(repository, deep=True)
            _json(reproduction_record(repository))
    elif arguments.command == "recover":
        with repository.lock():
            _json(recover_interrupted(repository, force=arguments.force))
    elif arguments.command == "gc":
        with repository.lock():
            candidates = gc_candidates(repository)
            if arguments.apply:
                if arguments.backup_marker is None:
                    raise AlphaMiniError("--apply requires --backup-marker")
                apply_gc(repository, candidates, backup_marker=arguments.backup_marker)
            _json(
                {
                    "applied": arguments.apply,
                    "active_session_present": repository.active_session_path.exists(),
                    "candidates": [str(path) for path in candidates],
                }
            )
    elif arguments.command == "report":
        with repository.lock():
            destination = write_report(repository, arguments.output)
        _json({"report": str(destination)})
    elif arguments.command == "benchmark-report":
        with repository.lock():
            report = production_benchmark_report(repository)
        _json(report)
        return 1 if report["failures"] else 0
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    # Convert supervisor SIGTERM into the same controlled unwind as Ctrl-C.
    # The orchestrator then terminates its active Rust child, preserves the
    # durable pointers/heartbeat, and requires the explicit recovery path.
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        raise SystemExit(dispatch(parser.parse_args(argv)))
    except AlphaMiniError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except KeyboardInterrupt as error:
        print(
            "interrupted safely; run `alphamini-train recover --run-dir ...`, then verify and resume",
            file=sys.stderr,
        )
        raise SystemExit(130) from error
