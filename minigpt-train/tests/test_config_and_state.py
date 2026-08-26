from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from minigpt_train.atomic import AdvisoryLock, sha256_bytes
from minigpt_train.config import load_config, validate_config
from minigpt_train.errors import ConfigError, IntegrityError, RunLockedError
from minigpt_train.operations import gc_candidates, render_report, reproduction_record, verify_run
from minigpt_train.run import RunRepository, extend_budget, fork_run

from conftest import PILOT_CONFIG, REPOSITORY, V1_CONFIG, write_config, write_shards

torch_available = importlib.util.find_spec("torch") is not None


def test_shipped_configurations_validate() -> None:
    for path in (PILOT_CONFIG, V1_CONFIG):
        config = load_config(path)
        assert config.values["schema"] == "minigpt.config.v1"
        assert config.config_hash == sha256_bytes(path.read_bytes())


@pytest.mark.skipif(not torch_available, reason="PyTorch training extra is not installed")
def test_v1_parameter_count_is_within_the_40m_target() -> None:
    from minigpt_train.model import build_model, parameter_count

    count = parameter_count(build_model(load_config(V1_CONFIG)))
    assert 38_000_000 <= count <= 44_000_000


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('schema = "minigpt.config.v1"', 'schema = "minigpt.config.v2"', "schema"),
        ("vocab = 4736", "vocab = 4096", "vocab"),
        ("n_heads = 8", "n_heads = 7", "divisible"),
        ('tokenizer = "policy-v1"', 'tokenizer = "policy-v2"', "tokenizer"),
        ("opset = 17", "opset = 18", "opset"),
        ("total_steps = 800", "total_steps = 50", "segment_steps"),
        ("warmup_fraction = 0.02", "warmup_fraction = 1.5", "warmup_fraction"),
        ('device = "cuda"', 'device = "tpu"', "device"),
        ("minimum_learning_rate = 0.00003", "minimum_learning_rate = 0.5", "minimum_learning_rate"),
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, old: str, new: str, message: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(PILOT_CONFIG.read_text().replace(old, new))
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize("section", ["run", "model", "data", "training", "export", "operations"])
def test_config_rejects_a_missing_table(section: str) -> None:
    values = json.loads(json.dumps(load_config(PILOT_CONFIG).values))
    values.pop(section)
    with pytest.raises(ConfigError, match=section):
        validate_config(values)


def test_operational_knobs_are_not_semantic_but_model_and_data_are(tmp_path: Path) -> None:
    original = PILOT_CONFIG.read_text()
    base = load_config(PILOT_CONFIG).semantic_hash

    def variant(name: str, old: str, new: str) -> str:
        path = tmp_path / f"{name}.toml"
        path.write_text(original.replace(old, new))
        return load_config(path).semantic_hash

    assert variant("budget", "active_budget_hours = 1.0", "active_budget_hours = 12.0") == base
    assert variant("heartbeat", "heartbeat_seconds = 30", "heartbeat_seconds = 5") == base
    assert variant("keep", "checkpoint_keep_last = 2", "checkpoint_keep_last = 5") == base
    assert variant("floor", "disk_floor_bytes = 53687091200", "disk_floor_bytes = 1") == base
    assert variant("depth", "n_layers = 8", "n_layers = 9") != base
    assert variant("steps", "total_steps = 800", "total_steps = 900") != base
    assert variant("shards", 'shards_dir = "data/minigpt/shards"', 'shards_dir = "other"') != base


def _run(tmp_path: Path, name: str = "run") -> RunRepository:
    config_path = write_config(tmp_path / f"{name}.toml", tmp_path / "shards")
    return RunRepository.create(tmp_path / name, load_config(config_path), worktree=REPOSITORY)


def test_created_run_freezes_its_configuration_and_identity(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    manifest = json.loads((repository.root / "RUN.json").read_text())
    assert manifest["schema"] == "minigpt.run-manifest.v1"
    assert manifest["locks"]["uv_lock_sha256"] is not None
    assert (repository.root / "config.toml").stat().st_mode & 0o222 == 0
    assert repository.head()[1]["phase"] == "initialized"
    reopened, config = RunRepository.open(repository.root)
    assert config.semantic_hash == manifest["semantic_hash"]
    assert reopened.head()[0] == repository.head()[0]


def test_open_refuses_a_config_whose_hash_changed(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    frozen = repository.root / "config.toml"
    frozen.chmod(0o644)
    frozen.write_text(frozen.read_text().replace("total_steps = 8", "total_steps = 9"))
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        RunRepository.open(repository.root)


def test_run_directory_must_be_empty(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    config = load_config(repository.root / "config.toml")
    with pytest.raises(IntegrityError, match="not empty"):
        RunRepository.create(repository.root, config, worktree=REPOSITORY)


def test_advisory_lock_is_exclusive(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    with repository.lock():
        with pytest.raises(RunLockedError, match="locked by"):
            with AdvisoryLock(repository.lock_path):
                pass


def test_head_recovery_atomic_state_and_extension(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    write_shards(tmp_path / "shards", [[[4672, 1, 2, 3]]])
    config = load_config(repository.root / "config.toml")
    head_before, state = repository.head()
    assert repository.recovery()[0] == head_before
    state["phase"] = "ready_train"
    state["active_used_seconds"] = state["active_budget_seconds"]
    repository.commit_head(state)
    exhausted_head = repository.head()[0]

    extension = extend_budget(repository, 3600, "test continuation")
    head_after, after = repository.head()
    assert head_after != head_before
    assert repository.recovery()[0] == head_after
    assert after["previous_state_hash"] == exhausted_head
    assert after["active_budget_seconds"] == state["active_budget_seconds"] + 3600
    assert repository.store.get(after["budget_extensions"][0]) == extension
    milestone_sha256 = after["initial_budget_milestone_sha256"]
    assert extension["initial_budget_milestone_sha256"] == milestone_sha256
    assert repository.store.get(milestone_sha256)["parent_state_sha256"] != milestone_sha256

    summary = verify_run(repository, config, deep=False, worktree=REPOSITORY)
    assert summary["artifacts_checked"] == 2
    assert milestone_sha256 in render_report(repository)
    assert reproduction_record(repository)["initial_budget_milestone"]["global_step"] == 0
    with pytest.raises(ConfigError, match="premature"):
        extend_budget(repository, 60, "must wait for the extended boundary")


def test_budget_extension_is_rejected_before_the_original_milestone(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    head_before = repository.head()[0]
    with pytest.raises(ConfigError, match="not exhausted"):
        extend_budget(repository, 60, "premature extension")
    assert repository.head()[0] == head_before
    assert repository.head()[1]["initial_budget_milestone_sha256"] is None


def test_extension_is_rejected_mid_segment(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    _, state = repository.head()
    state["phase"] = "training"
    state["active_used_seconds"] = state["active_budget_seconds"]
    repository.commit_head(state)
    with pytest.raises(ConfigError, match="safe boundary"):
        extend_budget(repository, 60, "mid-segment extension")


def test_effective_prefers_a_newer_recovery_and_refuses_an_unrelated_one(tmp_path: Path) -> None:
    from minigpt_train.atomic import write_pointer

    repository = _run(tmp_path)
    original = repository.head()[0]
    _, state = repository.head()
    state["phase"] = "training"
    state["global_step"] = 3
    mid_segment = repository.commit_recovery(state)
    # A crashed segment leaves RECOVERY ahead of HEAD; that state is authoritative.
    assert repository.effective() == (mid_segment, repository.recovery()[1])

    _, promoted = repository.head()
    promoted["phase"] = "ready_train"
    repository.commit_head(promoted)
    _, moved = repository.head()
    moved["global_step"] = 4
    repository.commit_head(moved)
    write_pointer(repository.recovery_path, original)
    with pytest.raises(IntegrityError, match="RECOVERY was not derived"):
        repository.effective()


def test_fork_warm_starts_from_the_best_checkpoint_only(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    checkpoint = (
        repository.root / "artifacts" / "checkpoints" / "step-000000004-abcdef0123456789.pt"
    )
    checkpoint.write_bytes(b"weights")
    _, state = repository.head()
    state["phase"] = "ready_train"
    state["global_step"] = 4
    state["best_checkpoint"] = {
        "path": str(checkpoint.relative_to(repository.root)),
        "sha256": sha256_bytes(b"weights"),
        "global_step": 4,
        "validation_loss": 1.5,
    }
    repository.commit_head(state)

    child_config = write_config(tmp_path / "child.toml", tmp_path / "shards")
    child = fork_run(
        repository,
        tmp_path / "child",
        load_config(child_config),
        worktree=REPOSITORY,
        reason="branch the schedule",
    )
    manifest = json.loads((child.root / "RUN.json").read_text())
    assert manifest["parent"]["relationship"] == "weights-only-warm-start"
    assert manifest["parent"]["run_id"] != manifest["run_id"]
    _, child_state = child.head()
    assert child_state["phase"] == "warm_start_ready"
    assert child_state["global_step"] == 0
    assert child_state["best_checkpoint"] is None
    assert (child.root / child_state["current_checkpoint"]["path"]).read_bytes() == b"weights"
    assert (
        child_state["current_checkpoint"]["warm_start_from_semantic_hash"]
        == (manifest["parent"]["semantic_hash"])
    )


def test_fork_refuses_a_corrupt_source_checkpoint(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    checkpoint = (
        repository.root / "artifacts" / "checkpoints" / "step-000000004-abcdef0123456789.pt"
    )
    checkpoint.write_bytes(b"weights")
    _, state = repository.head()
    state["best_checkpoint"] = {
        "path": str(checkpoint.relative_to(repository.root)),
        "sha256": "0" * 64,
        "global_step": 4,
        "validation_loss": 1.5,
    }
    repository.commit_head(state)
    with pytest.raises(IntegrityError, match="missing or corrupt"):
        fork_run(
            repository,
            tmp_path / "child",
            load_config(repository.root / "config.toml"),
            worktree=REPOSITORY,
            reason="branch",
        )


def test_gc_lists_partial_files_on_a_fresh_run(tmp_path: Path) -> None:
    repository = _run(tmp_path)
    partial = repository.root / "artifacts" / "checkpoints" / ".step.partial"
    partial.write_bytes(b"")
    assert gc_candidates(repository) == [partial]
