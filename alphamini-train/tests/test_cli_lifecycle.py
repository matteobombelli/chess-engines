from __future__ import annotations

import json
from pathlib import Path

from alphamini_train.cli import build_parser, dispatch
from alphamini_train.run import RunRepository

from conftest import PILOT_CONFIG, REPOSITORY


def invoke(*arguments: str) -> int:
    parser = build_parser()
    return dispatch(parser.parse_args(["--worktree", str(REPOSITORY), *arguments]))


def test_metadata_lifecycle_commands(tmp_path: Path, capsys) -> None:
    run = tmp_path / "run"
    child = tmp_path / "child"
    report = tmp_path / "report.md"
    assert (
        invoke(
            "start",
            "--config",
            str(PILOT_CONFIG),
            "--run-dir",
            str(run),
            "--metadata-only",
        )
        == 0
    )
    # Extensions are deliberately unavailable until the original budget has a
    # safe-boundary milestone. Simulate that boundary in this metadata-only fixture.
    repository = RunRepository(run)
    _, exhausted = repository.head()
    exhausted["phase"] = "ready_collect"
    exhausted["active_used_seconds"] = exhausted["active_budget_seconds"]
    repository.commit_head(exhausted)
    assert (
        invoke(
            "extend",
            "--run-dir",
            str(run),
            "--additional-active-budget",
            "1h",
            "--reason",
            "CLI lifecycle fixture",
        )
        == 0
    )
    assert invoke("verify", "--run-dir", str(run), "--deep") == 0
    assert invoke("reproduce", "--run-dir", str(run)) == 0
    assert invoke("report", "--run-dir", str(run), "--output", str(report)) == 0
    assert "No arena or Elo result" in report.read_text()
    assert (
        invoke(
            "fork",
            "--source-run",
            str(run),
            "--config",
            str(PILOT_CONFIG),
            "--run-dir",
            str(child),
            "--reason",
            "CLI fork fixture",
        )
        == 0
    )
    parent = json.loads((child / "RUN.json").read_text())["parent"]
    assert parent["relationship"] == "weights-only-warm-start"
    assert invoke("gc", "--run-dir", str(run)) == 0
    # Commands emit machine-readable JSON rather than prose-only status.
    assert '"run_dir"' in capsys.readouterr().out
