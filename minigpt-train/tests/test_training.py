from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

from minigpt_train.config import PAD_TOKEN, VOCAB_SIZE, load_config
from minigpt_train.errors import DiskSpaceError
from minigpt_train.trainer import ensure_free_space, learning_rate_at, prunable_checkpoints

from conftest import make_games, write_config, write_shards

torch_available = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(
    not torch_available, reason="PyTorch training extra is not installed"
)


def _fake_checkpoints(directory: Path, steps: list[int]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for step in steps:
        path = directory / f"step-{step:09d}-{step:016x}.pt"
        path.write_bytes(b"checkpoint")
        path.with_suffix(".pt.json").write_text("{}")
        paths.append(path)
    return paths


def test_retention_keeps_last_two_and_every_milestone(tmp_path: Path) -> None:
    steps = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    paths = _fake_checkpoints(tmp_path, steps)
    prunable = prunable_checkpoints(tmp_path, keep_last=2, milestone_every=4, protected=[])
    kept = {path.name for path in paths} - {path.name for path in prunable}
    assert kept == {
        "step-000000000-0000000000000000.pt",  # milestone
        "step-000000004-0000000000000004.pt",  # milestone
        "step-000000008-0000000000000008.pt",  # milestone and last
        "step-000000007-0000000000000007.pt",  # last two
    }
    assert [path.name for path in prunable] == [
        "step-000000001-0000000000000001.pt",
        "step-000000002-0000000000000002.pt",
        "step-000000003-0000000000000003.pt",
        "step-000000005-0000000000000005.pt",
        "step-000000006-0000000000000006.pt",
    ]


def test_retention_never_removes_a_protected_checkpoint(tmp_path: Path) -> None:
    paths = _fake_checkpoints(tmp_path, [0, 1, 2, 3, 4, 5])
    protected = paths[1]
    prunable = prunable_checkpoints(
        tmp_path, keep_last=1, milestone_every=100, protected=[protected]
    )
    assert protected not in prunable
    assert paths[2] in prunable


def test_disk_floor_pauses_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = os.statvfs

    class Status:
        f_bavail = 10
        f_frsize = 4096

    monkeypatch.setattr(os, "statvfs", lambda path: Status())
    assert ensure_free_space(tmp_path, 40_000) == 40_960
    with pytest.raises(DiskSpaceError, match="below the configured floor"):
        ensure_free_space(tmp_path, 50_000)
    monkeypatch.setattr(os, "statvfs", real)


def test_learning_rate_warms_up_then_cosines_to_the_floor() -> None:
    config = load_config(Path(__file__).resolve().parents[2] / "configs" / "minigpt" / "v1.toml")
    peak = float(config.values["training"]["learning_rate"])
    floor = float(config.values["training"]["minimum_learning_rate"])
    horizon = int(config.values["training"]["total_steps"])
    warmup = round(horizon * float(config.values["training"]["warmup_fraction"]))

    assert learning_rate_at(config, 0) == pytest.approx(peak / warmup)
    assert learning_rate_at(config, warmup - 1) == pytest.approx(peak)
    assert learning_rate_at(config, horizon - 1) == pytest.approx(floor, abs=1e-9)
    assert learning_rate_at(config, horizon + 1000) == floor
    middle = learning_rate_at(config, (warmup + horizon) // 2)
    assert floor < middle < peak


@requires_torch
def test_pad_targets_contribute_exactly_zero_loss() -> None:
    import torch

    from minigpt_train.trainer import next_token_loss

    logits = torch.zeros((1, 3, VOCAB_SIZE))
    # A PAD target with wildly different logits must not move the loss at all.
    logits[0, 2, :] = 25.0
    logits[0, 2, 17] = -100.0
    targets = torch.tensor([[3, 9, PAD_TOKEN]])
    assert float(next_token_loss(logits, targets)) == pytest.approx(math.log(VOCAB_SIZE))

    logits[0, 0, 3] = 5.0
    expected = 0.5 * (math.log(math.exp(5.0) + VOCAB_SIZE - 1) - 5.0 + math.log(VOCAB_SIZE))
    assert float(next_token_loss(logits, targets)) == pytest.approx(expected, rel=1e-6)


def _prepare(tmp_path: Path, **overrides: dict) -> tuple:
    from minigpt_train.data import BucketedSampler
    from minigpt_train.segments import open_corpus

    shards = tmp_path / "shards"
    write_shards(shards, [make_games(24)], [make_games(8, first=800)])
    config = load_config(write_config(tmp_path / "config.toml", shards, **overrides))
    corpus = open_corpus(config, tmp_path, verify_hashes=True)
    sampler = BucketedSampler(
        corpus["train"],
        seed=int(config.values["run"]["seed"]),
        micro_batch=int(config.values["training"]["micro_batch"]),
        buckets=int(config.values["data"]["length_buckets"]),
    )
    return config, corpus, sampler


def _train(tmp_path: Path, config, corpus, sampler, *, name: str, plan: list[int]) -> list[float]:
    """Train to each step in `plan`, restarting the trainer from disk between them."""

    from minigpt_train.trainer import Trainer, create_initial_checkpoint

    run_root = tmp_path / name
    checkpoints = run_root / "artifacts" / "checkpoints"
    metrics = run_root / "metrics.jsonl"
    _, _, checkpoint = create_initial_checkpoint(
        config, checkpoints, run_root=run_root, shards_identity=corpus["identity"]
    )
    for target in plan:
        trainer = Trainer(
            config,
            sampler,
            validation=corpus["validation"],
            run_root=run_root,
            shards_identity=corpus["identity"],
        )
        trainer.resume(checkpoint)
        trainer.train_segment(target, checkpoint_dir=checkpoints, metrics_path=metrics)
        _, checkpoint = trainer.save_checkpoint(checkpoints)
    return [
        json.loads(line)["train_loss"] for line in metrics.read_text().splitlines() if line.strip()
    ]


@requires_torch
def test_resumed_training_reproduces_an_uninterrupted_run(tmp_path: Path) -> None:
    config, corpus, sampler = _prepare(
        tmp_path, training={"checkpoint_keep_last": 8, "eval_interval_steps": 1}
    )
    uninterrupted = _train(tmp_path, config, corpus, sampler, name="whole", plan=[6])
    interrupted = _train(tmp_path, config, corpus, sampler, name="split", plan=[3, 6])
    assert len(uninterrupted) == 6
    assert interrupted == uninterrupted


@requires_torch
def test_metrics_record_every_reported_field(tmp_path: Path) -> None:
    config, corpus, sampler = _prepare(tmp_path, training={"eval_interval_steps": 2})
    _train(tmp_path, config, corpus, sampler, name="metrics", plan=[4])
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [record["step"] for record in records] == [2, 4]
    for record in records:
        assert record["schema"] == "minigpt.metrics.v1"
        assert record["train_loss"] > 0
        assert record["validation_loss"] > 0
        assert 0.0 <= record["validation_top1"] <= 1.0
        assert record["validation_perplexity"] == pytest.approx(math.exp(record["validation_loss"]))
        assert record["tokens_per_second"] > 0
        assert record["learning_rate"] > 0
        assert record["free_disk_bytes"] > 0
        assert record["vram_bytes"] is None


@requires_torch
def test_best_checkpoint_tracks_the_best_validation_loss(tmp_path: Path) -> None:
    from minigpt_train.trainer import Trainer, create_initial_checkpoint

    config, corpus, sampler = _prepare(tmp_path, training={"eval_interval_steps": 1})
    run_root = tmp_path / "best"
    checkpoints = run_root / "artifacts" / "checkpoints"
    _, _, checkpoint = create_initial_checkpoint(
        config, checkpoints, run_root=run_root, shards_identity=corpus["identity"]
    )
    trainer = Trainer(
        config,
        sampler,
        validation=corpus["validation"],
        run_root=run_root,
        shards_identity=corpus["identity"],
    )
    trainer.resume(checkpoint)
    trainer.train_segment(4, checkpoint_dir=checkpoints, metrics_path=run_root / "metrics.jsonl")
    assert trainer.best_checkpoint is not None
    assert trainer.best_checkpoint["global_step"] == trainer.state.best_validation_step
    assert trainer.state.best_validation_loss == trainer.best_checkpoint["validation_loss"]
    assert (run_root / trainer.best_checkpoint["path"]).is_file()


@requires_torch
def test_checkpoint_refuses_another_corpus(tmp_path: Path) -> None:
    from minigpt_train.errors import IntegrityError
    from minigpt_train.trainer import Trainer, create_initial_checkpoint

    config, corpus, sampler = _prepare(tmp_path)
    run_root = tmp_path / "corpus"
    checkpoints = run_root / "artifacts" / "checkpoints"
    _, _, checkpoint = create_initial_checkpoint(
        config, checkpoints, run_root=run_root, shards_identity=corpus["identity"]
    )
    trainer = Trainer(
        config,
        sampler,
        validation=None,
        run_root=run_root,
        shards_identity={**corpus["identity"], "tokens_train": 999},
    )
    with pytest.raises(IntegrityError, match="another shard corpus"):
        trainer.resume(checkpoint)


@requires_torch
def test_training_prunes_superseded_checkpoints_as_it_writes(tmp_path: Path) -> None:
    from minigpt_train.trainer import Trainer, create_initial_checkpoint

    config, corpus, sampler = _prepare(
        tmp_path,
        training={
            "eval_interval_steps": 100,
            "checkpoint_interval_steps": 1,
            "checkpoint_keep_last": 2,
            "checkpoint_milestone_every_steps": 3,
        },
    )
    run_root = tmp_path / "retention"
    checkpoints = run_root / "artifacts" / "checkpoints"
    _, _, checkpoint = create_initial_checkpoint(
        config, checkpoints, run_root=run_root, shards_identity=corpus["identity"]
    )
    trainer = Trainer(
        config,
        sampler,
        validation=None,
        run_root=run_root,
        shards_identity=corpus["identity"],
        protected_checkpoints=[checkpoint],
    )
    trainer.resume(checkpoint)
    trainer.train_segment(7, checkpoint_dir=checkpoints)
    trainer.ensure_checkpoint(checkpoints)
    steps = sorted(int(path.name.split("-")[1]) for path in checkpoints.glob("step-*.pt"))
    # {milestones 0, 3, 6} + {last two: 6, 7} + the protected step 0 parent.
    assert steps == [0, 3, 6, 7]
    for step in steps:
        path = next(checkpoints.glob(f"step-{step:09d}-*.pt"))
        assert path.with_suffix(".pt.json").is_file()
    assert not list(checkpoints.glob("*.partial"))


@requires_torch
def test_gc_removes_only_superseded_non_milestone_checkpoints(tmp_path: Path) -> None:
    from minigpt_train.config import load_config as load
    from minigpt_train.operations import apply_gc, gc_candidates
    from minigpt_train.run import RunRepository

    from conftest import REPOSITORY

    config_path = write_config(
        tmp_path / "gc.toml",
        tmp_path / "shards",
        training={"checkpoint_keep_last": 1, "checkpoint_milestone_every_steps": 4},
    )
    repository = RunRepository.create(tmp_path / "run", load(config_path), worktree=REPOSITORY)
    checkpoints = repository.root / "artifacts" / "checkpoints"
    paths = _fake_checkpoints(checkpoints, [0, 1, 2, 3, 4, 5])
    _, state = repository.head()
    state["current_checkpoint"] = {
        "path": str(paths[5].relative_to(repository.root)),
        "sha256": "0" * 64,
        "global_step": 5,
    }
    state["best_checkpoint"] = {
        "path": str(paths[2].relative_to(repository.root)),
        "sha256": "0" * 64,
        "global_step": 2,
        "validation_loss": 1.0,
    }
    repository.commit_head(state)

    candidates = gc_candidates(repository)
    assert [path.name for path in candidates] == [paths[1].name, paths[3].name]
    removed = apply_gc(repository, candidates)
    assert len(removed) == 2
    assert not paths[1].exists() and not paths[1].with_suffix(".pt.json").exists()
    assert paths[0].exists() and paths[2].exists() and paths[4].exists() and paths[5].exists()
