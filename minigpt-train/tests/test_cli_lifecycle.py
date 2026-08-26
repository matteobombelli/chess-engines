from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from minigpt_train.cli import build_parser, dispatch
from minigpt_train.errors import ConfigError
from minigpt_train.run import RunRepository

from conftest import make_games, write_config, write_shards

extras_available = all(
    importlib.util.find_spec(module) is not None
    for module in ("torch", "onnx", "onnxruntime", "numpy")
)
REPOSITORY = Path(__file__).resolve().parents[2]


def invoke(*arguments: str) -> int:
    parser = build_parser()
    return dispatch(parser.parse_args(["--worktree", str(REPOSITORY), *arguments]))


def test_metadata_lifecycle_commands(tmp_path: Path, capsys) -> None:
    run = tmp_path / "run"
    child = tmp_path / "child"
    report = tmp_path / "report.md"
    config = write_config(tmp_path / "config.toml", tmp_path / "shards")
    assert invoke("start", "--config", str(config), "--run-dir", str(run), "--metadata-only") == 0

    # Extensions are unavailable until the original budget reaches a segment boundary.
    repository = RunRepository(run)
    _, exhausted = repository.head()
    exhausted["phase"] = "ready_train"
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
    assert invoke("verify", "--run-dir", str(run)) == 0
    assert invoke("reproduce", "--run-dir", str(run)) == 0
    assert invoke("report", "--run-dir", str(run), "--output", str(report)) == 0
    assert "No ONNX model has been exported" in report.read_text()
    assert (
        invoke(
            "fork",
            "--source-run",
            str(run),
            "--config",
            str(config),
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


@pytest.mark.skipif(not extras_available, reason="full train/ONNX extras are not installed")
def test_train_resume_export_report_through_the_ledger(tmp_path: Path, capsys) -> None:
    shards = tmp_path / "shards"
    write_shards(shards, [make_games(32)], [make_games(8, first=700)])
    config = write_config(
        tmp_path / "config.toml",
        shards,
        model={"d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "ctx": 32},
        training={"total_steps": 4, "segment_steps": 2, "eval_interval_steps": 2},
    )
    run = tmp_path / "run"

    assert invoke("start", "--config", str(config), "--run-dir", str(run), "--one-segment") == 0
    repository, _ = RunRepository.open(run)
    _, state = repository.effective()
    assert state["phase"] == "ready_train"
    assert state["global_step"] == 2
    assert state["segment_index"] == 1
    assert state["shards"]["manifest_sha256"]
    assert len(state["completed_segments"]) == 1

    assert invoke("resume", "--run-dir", str(run)) == 0
    _, state = repository.effective()
    assert state["phase"] == "complete"
    assert state["global_step"] == 4
    assert state["best_checkpoint"] is not None
    assert (run / state["current_checkpoint"]["path"]).is_file()
    assert (run / "metrics.jsonl").is_file()

    # A complete run is refused rather than silently trained past its frozen horizon.
    with pytest.raises(ConfigError, match="complete"):
        invoke("resume", "--run-dir", str(run))
    assert invoke("verify", "--run-dir", str(run), "--deep") == 0

    assert invoke("export", "--run-dir", str(run), "--publish-dir", str(tmp_path / "current")) == 0
    published = json.loads((tmp_path / "current" / "manifest.json").read_text())
    assert published["schema"] == "minigpt.manifest.v1"
    assert (tmp_path / "current" / "model.onnx").is_file()
    _, state = repository.effective()
    assert len(state["exports"]) == 1
    fixture = run / state["exports"][0]["fixture_path"] / "parity.json"
    assert json.loads(fixture.read_text())["schema"] == "minigpt.parity-fixture.v1"

    assert invoke("report", "--run-dir", str(run)) == 0
    report = (run / "report.md").read_text()
    assert published["model_sha256"] in report
    assert invoke("verify", "--run-dir", str(run), "--deep") == 0
    assert invoke("gc", "--run-dir", str(run), "--apply") == 0
    assert invoke("verify", "--run-dir", str(run), "--deep") == 0
    capsys.readouterr()
