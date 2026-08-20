from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from alphamini_train.atomic import AdvisoryLock, read_pointer
from alphamini_train.config import load_config
from alphamini_train.errors import ConfigError, IntegrityError, RunLockedError
from alphamini_train.operations import (
    apply_gc,
    gc_candidates,
    render_report,
    reproduction_record,
    verify_run,
)
from alphamini_train.orchestrator import (
    _cycle_collection_seed,
    _verify_collection_request,
    run_training,
)
from alphamini_train.run import (
    RunRepository,
    extend_budget,
    fork_run,
    git_identity,
    recover_interrupted,
)

from conftest import PILOT_CONFIG, REPOSITORY


def _clean_git_worktree(path: Path) -> Path:
    path.mkdir()
    (path / "alphamini-train").mkdir()
    (path / ".gitignore").write_text("/runs\n")
    (path / "source.txt").write_text("committed\n")
    (path / "Cargo.lock").write_text("cargo-lock-fixture\n")
    (path / "alphamini-train" / "uv.lock").write_text("uv-lock-fixture\n")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "AlphaMini test"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return path


def test_active_budget_is_not_semantic_but_model_is(tmp_path: Path) -> None:
    original = PILOT_CONFIG.read_text()
    budget = tmp_path / "budget.toml"
    budget.write_text(original.replace("active_budget_hours = 0.25", "active_budget_hours = 12.0"))
    model = tmp_path / "model.toml"
    model.write_text(original.replace("channels = 16", "channels = 24"))
    base = load_config(PILOT_CONFIG)
    assert load_config(budget).semantic_hash == base.semantic_hash
    assert load_config(model).semantic_hash != base.semantic_hash


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("max_plies = 128", "max_plies = 513", "max_plies"),
        ("simulations = 16", f"simulations = {2**32}", "simulations"),
        ("seed = 1", f"seed = {2**64}", "seed"),
    ],
)
def test_config_rejects_values_outside_rust_wire_bounds(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(PILOT_CONFIG.read_text().replace(old, new))
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_collection_request_binds_seed_search_budget_and_cap() -> None:
    expected = {
        "run_id": "run-uuid",
        "cycle_id": 9,
        "seed": _cycle_collection_seed(747537, 9),
        "simulations": 128,
        "max_plies": 512,
    }
    _verify_collection_request(dict(expected), expected=expected)
    assert expected["seed"] == ((747537 << 32) ^ 9) & (2**64 - 1)

    for field in ("seed", "simulations", "max_plies"):
        altered = dict(expected)
        altered[field] += 1
        with pytest.raises(IntegrityError, match=field):
            _verify_collection_request(altered, expected=expected)


def test_reproduction_commands_run_from_repository_root(tmp_path: Path) -> None:
    repository = RunRepository.create(
        tmp_path / "run", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    commands = reproduction_record(repository)["commands"]
    expected = "uv run --project alphamini-train alphamini-train"
    assert commands["verify"].startswith(expected)
    assert commands["continue"].startswith(expected)


def test_head_recovery_atomic_state_and_extension(tmp_path: Path) -> None:
    config = load_config(PILOT_CONFIG)
    repository = RunRepository.create(tmp_path / "run", config, worktree=REPOSITORY)
    head_before, state_before = repository.head()
    assert repository.recovery()[0] == head_before
    state_before["phase"] = "ready_collect"
    state_before["active_used_seconds"] = state_before["active_budget_seconds"]
    repository.commit_head(state_before)
    exhausted_head = repository.head()[0]
    extension = extend_budget(repository, 3600, "test continuation")
    head_after, state_after = repository.head()
    assert head_after != head_before
    assert repository.recovery()[0] == head_after
    assert state_after["previous_state_hash"] == exhausted_head
    assert state_after["active_budget_seconds"] == state_before["active_budget_seconds"] + 3600
    assert repository.store.get(state_after["budget_extensions"][0]) == extension
    milestone_sha256 = state_after["initial_budget_milestone_sha256"]
    assert extension["initial_budget_milestone_sha256"] == milestone_sha256
    assert repository.store.get(milestone_sha256)["parent_state_sha256"] != milestone_sha256
    assert verify_run(repository, deep=False)["artifacts_checked"] == 2
    assert repository.verify_state_chain(deep=True)["state_objects"] >= 2
    with pytest.raises(ConfigError, match="premature"):
        extend_budget(repository, 60, "must wait for the extended boundary")


def test_budget_extension_is_rejected_before_original_milestone(tmp_path: Path) -> None:
    repository = RunRepository.create(
        tmp_path / "run", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    head_before = repository.head()[0]
    with pytest.raises(ConfigError, match="not exhausted"):
        extend_budget(repository, 60, "premature extension")
    assert repository.head()[0] == head_before
    assert repository.head()[1]["initial_budget_milestone_sha256"] is None


def test_exhausted_resume_seals_milestone_before_session_and_requires_extend(
    tmp_path: Path,
) -> None:
    config = load_config(PILOT_CONFIG)
    repository = RunRepository.create(tmp_path / "run", config, worktree=REPOSITORY)
    _, exhausted = repository.head()
    exhausted["phase"] = "ready_collect"
    exhausted["active_used_seconds"] = exhausted["active_budget_seconds"] + 2.5
    repository.commit_head(exhausted)

    with pytest.raises(ConfigError, match="run extend"):
        run_training(repository, config, worktree=REPOSITORY, one_cycle=True)
    assert not repository.active_session_path.exists()
    _, marked = repository.head()
    milestone_sha256 = marked["initial_budget_milestone_sha256"]
    milestone = repository.store.get(milestone_sha256)
    assert milestone["cycle_id"] == marked["cycle_id"]
    assert milestone["overshoot_seconds"] == pytest.approx(2.5)

    reproduction = reproduction_record(repository)
    assert reproduction["initial_budget_milestone_sha256"] == milestone_sha256
    assert reproduction["initial_budget_milestone"] == milestone
    assert milestone_sha256 in render_report(repository)

    extend_budget(repository, 60, "continue after frozen marker")
    _, extended = repository.head()
    assert extended["initial_budget_milestone_sha256"] == milestone_sha256


def test_head_is_authoritative_if_recovery_pointer_repair_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import alphamini_train.run as run_module

    repository = RunRepository.create(
        tmp_path / "run", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    old_head, state = repository.head()
    original_write_pointer = run_module.write_pointer

    def fail_recovery_pointer(path: Path, digest: str) -> None:
        if path == repository.recovery_path:
            raise OSError("injected recovery-pointer crash")
        original_write_pointer(path, digest)

    monkeypatch.setattr(run_module, "write_pointer", fail_recovery_pointer)
    with pytest.raises(OSError, match="injected"):
        repository.commit_head(state)
    new_head, effective = repository.effective()
    assert new_head != old_head
    assert effective["previous_state_hash"] == old_head
    assert repository.recovery()[0] == old_head

    monkeypatch.setattr(run_module, "write_pointer", original_write_pointer)
    repository.commit_head(effective)
    assert repository.head()[0] == repository.recovery()[0]


def test_git_identity_rejects_untracked_source_but_ignores_ignored_artifacts(
    tmp_path: Path,
) -> None:
    worktree = _clean_git_worktree(tmp_path / "source")
    clean = git_identity(worktree)
    assert clean["tracked_dirty"] is False
    assert clean["worktree_sha256"] is not None

    (worktree / "untracked.py").write_text("print('drift')\n")
    dirty = git_identity(worktree)
    assert dirty["tracked_dirty"] is True
    assert dirty["worktree_sha256"] != clean["worktree_sha256"]
    (worktree / "untracked.py").unlink()
    (worktree / "runs").mkdir()
    (worktree / "runs" / "artifact.bin").write_bytes(b"ignored")
    ignored = git_identity(worktree)
    assert ignored["tracked_dirty"] is False
    assert ignored["worktree_sha256"] == clean["worktree_sha256"]


def test_disposable_run_binds_dirty_untracked_content_but_v1_stays_clean_only(
    tmp_path: Path,
) -> None:
    import alphamini_train.orchestrator as orchestrator_module

    worktree = _clean_git_worktree(tmp_path / "source")
    untracked = worktree / "new-engine.rs"
    untracked.write_text("first pilot source\n")
    disposable = load_config(PILOT_CONFIG)
    pilot = RunRepository.create(tmp_path / "pilot", disposable, worktree=worktree)
    lineage = orchestrator_module._checkpoint_lineage(pilot, disposable, worktree)
    assert lineage["source_disposable"] is True
    assert lineage["source_worktree_sha256"] == git_identity(worktree)["worktree_sha256"]

    untracked.write_text("changed after pilot creation\n")
    with pytest.raises(IntegrityError, match="worktree content differs"):
        orchestrator_module._checkpoint_lineage(pilot, disposable, worktree)

    non_disposable_path = tmp_path / "non-disposable.toml"
    non_disposable_path.write_text(
        PILOT_CONFIG.read_text().replace("disposable = true", "disposable = false")
    )
    non_disposable = load_config(non_disposable_path)
    published = RunRepository.create(tmp_path / "published", non_disposable, worktree=worktree)
    with pytest.raises(IntegrityError, match="cleanliness is dirty"):
        orchestrator_module._checkpoint_lineage(published, non_disposable, worktree)


def test_run_training_rejects_lock_drift_before_active_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import alphamini_train.orchestrator as orchestrator_module

    worktree = _clean_git_worktree(tmp_path / "source")
    config = load_config(PILOT_CONFIG)
    repository = RunRepository.create(tmp_path / "run", config, worktree=worktree)
    frozen_git = json.loads((repository.root / "RUN.json").read_text())["git"]
    (worktree / "Cargo.lock").write_text("changed-without-a-new-run\n")
    # Isolate the lock identity check from the independent dirty-tree check.
    monkeypatch.setattr(orchestrator_module, "git_identity", lambda _: frozen_git)

    with pytest.raises(IntegrityError, match="lockfiles differ"):
        run_training(repository, config, worktree=worktree, one_cycle=True)
    assert not repository.active_session_path.exists()
    assert not any((repository.root / "cycles").iterdir())


def test_run_training_rejects_unknown_git_status_before_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import alphamini_train.orchestrator as orchestrator_module

    worktree = _clean_git_worktree(tmp_path / "source")
    config = load_config(PILOT_CONFIG)
    repository = RunRepository.create(tmp_path / "run", config, worktree=worktree)
    commit = json.loads((repository.root / "RUN.json").read_text())["git"]["commit"]
    monkeypatch.setattr(
        orchestrator_module,
        "git_identity",
        lambda _: {"commit": commit, "tracked_dirty": None},
    )

    with pytest.raises(IntegrityError, match="cleanliness or content identity is unknown"):
        run_training(repository, config, worktree=worktree, one_cycle=True)
    assert not repository.active_session_path.exists()


def test_run_training_rejects_unconfirmed_horizon_before_session(tmp_path: Path) -> None:
    config_path = tmp_path / "unconfirmed.toml"
    config_path.write_text(
        PILOT_CONFIG.read_text().replace("horizon_confirmed = true", "horizon_confirmed = false")
    )
    config = load_config(config_path)
    repository = RunRepository.create(tmp_path / "run", config, worktree=REPOSITORY)

    with pytest.raises(ConfigError, match="horizon is not confirmed"):
        run_training(repository, config, worktree=REPOSITORY, one_cycle=True)
    assert not repository.active_session_path.exists()


def test_run_training_rejects_runtime_drift_before_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import alphamini_train.orchestrator as orchestrator_module

    worktree = _clean_git_worktree(tmp_path / "source")
    config = load_config(PILOT_CONFIG)
    repository = RunRepository.create(tmp_path / "run", config, worktree=worktree)
    recorded = json.loads((repository.root / "RUN.json").read_text())["runtime"]
    changed = {**recorded, "nvidia_driver": f"different-{recorded.get('nvidia_driver')}"}
    monkeypatch.setattr(orchestrator_module, "runtime_identity", lambda: changed)

    with pytest.raises(IntegrityError, match="runtime differs"):
        run_training(repository, config, worktree=worktree, one_cycle=True)
    assert not repository.active_session_path.exists()


def test_corrupt_pointer_and_object_are_rejected(tmp_path: Path) -> None:
    repository = RunRepository.create(
        tmp_path / "run", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    repository.head_path.write_text("not-a-hash\n")
    with pytest.raises(IntegrityError, match="malformed pointer"):
        repository.head()

    repository = RunRepository.create(
        tmp_path / "second", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    digest = read_pointer(repository.head_path)
    path = repository.store.path_for(digest)
    os.chmod(path, 0o644)
    path.write_bytes(b"{}\n")
    with pytest.raises(IntegrityError, match="checksum"):
        repository.head()


def test_advisory_lock_rejects_second_owner(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    with AdvisoryLock(path):
        with pytest.raises(RunLockedError):
            with AdvisoryLock(path):
                pass


def test_fork_records_lineage_and_does_not_claim_resume(tmp_path: Path) -> None:
    config = load_config(PILOT_CONFIG)
    source = RunRepository.create(tmp_path / "source", config, worktree=REPOSITORY)
    child = fork_run(
        source,
        tmp_path / "child",
        config,
        worktree=REPOSITORY,
        reason="test child experiment",
    )
    manifest = json.loads((child.root / "RUN.json").read_text())
    assert manifest["parent"]["relationship"] == "weights-only-warm-start"
    assert manifest["parent"]["source_head"] == source.head()[0]


def test_recover_quarantines_uncommitted_materialization(tmp_path: Path) -> None:
    repository = RunRepository.create(
        tmp_path / "run", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    _, state = repository.head()
    state["phase"] = "ready_materialize"
    repository.commit_head(state)
    cache = repository.root / "cache" / "cycle-000000"
    cache.mkdir()
    (cache / "inputs.f32.bin").write_bytes(b"final-name-but-uncommitted")
    session = repository.begin_session("resume", 60)
    session.abandon()
    recovered = recover_interrupted(repository, force=True)
    assert not cache.exists()
    quarantined = recovered["interruptions"][-1]["quarantined"]
    assert quarantined and (repository.root / quarantined[0]).exists()


def test_recover_does_not_double_count_a_committed_sealed_session(tmp_path: Path) -> None:
    repository = RunRepository.create(
        tmp_path / "run", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    session = repository.begin_session("resume", 60)
    elapsed = session.seal()
    _, state = repository.head()
    state["active_used_seconds"] += elapsed
    repository.commit_head(state)
    before = repository.head()[1]["active_used_seconds"]
    recovered = recover_interrupted(repository, force=True)
    assert recovered["active_used_seconds"] == pytest.approx(before)
    assert recovered["interruptions"][-1]["counted_active_seconds"] == pytest.approx(0.0)


def test_recover_commit_precedes_active_session_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = RunRepository.create(
        tmp_path / "run", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    session = repository.begin_session("resume", 60)
    session.abandon()
    original_unlink = Path.unlink
    failed = False

    def fail_active_unlink(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal failed
        if path == repository.active_session_path and not failed:
            failed = True
            raise OSError("injected unlink crash window")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_active_unlink)
    with pytest.raises(OSError, match="injected"):
        recover_interrupted(repository, force=True)
    committed = repository.head()[1]
    assert repository.active_session_path.exists()
    assert len(committed["interruptions"]) == 1
    counted = committed["active_used_seconds"]

    recovered = recover_interrupted(repository, force=True)
    assert not repository.active_session_path.exists()
    assert recovered["active_used_seconds"] == pytest.approx(counted)
    assert len(recovered["interruptions"]) == 1


def test_recovery_seals_budget_milestone_at_safe_boundary(tmp_path: Path) -> None:
    repository = RunRepository.create(
        tmp_path / "run", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    _, state = repository.head()
    state["phase"] = "ready_collect"
    repository.commit_head(state)
    session = repository.begin_session("resume", 60)
    session.abandon()
    active = json.loads(repository.active_session_path.read_text())
    active["recorded_elapsed_seconds"] = state["active_budget_seconds"] + 1.0
    repository.active_session_path.write_text(json.dumps(active))

    recovered = recover_interrupted(repository, force=True)
    milestone = repository.store.get(recovered["initial_budget_milestone_sha256"])
    assert milestone["accounted_active_seconds"] == pytest.approx(
        state["active_budget_seconds"] + 1.0
    )
    assert verify_run(repository, deep=False)["artifacts_checked"] == 1


def test_gc_apply_rejects_unfinished_active_session(tmp_path: Path) -> None:
    repository = RunRepository.create(
        tmp_path / "run", load_config(PILOT_CONFIG), worktree=REPOSITORY
    )
    partial = repository.root / "cache" / "needed.partial"
    partial.write_bytes(b"needed for recovery")
    session = repository.begin_session("resume", 60)
    session.abandon()

    candidates = gc_candidates(repository)
    assert partial in candidates
    with pytest.raises(IntegrityError, match="ACTIVE_SESSION"):
        apply_gc(repository, candidates, backup_marker=tmp_path / "unused-marker.json")
    assert partial.exists()
