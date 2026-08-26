from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from minigpt_train.data import (
    BucketedSampler,
    SamplerState,
    ShardSplit,
    bucket_of,
    evaluation_batches,
    load_shards_manifest,
    shards_identity,
    split_inputs_and_targets,
)
from minigpt_train.errors import IntegrityError

from conftest import BOS, PAD, make_games, write_shards


def _corpus(tmp_path: Path, context: int = 32) -> tuple[Path, ShardSplit]:
    directory = tmp_path / "shards"
    manifest_path = write_shards(
        directory,
        [make_games(9), make_games(7, first=100)],
        [make_games(5, first=900)],
    )
    manifest = load_shards_manifest(manifest_path, verify_hashes=True)
    return manifest_path, ShardSplit(directory, manifest["train_shards"], context=context)


def test_shard_bytes_round_trip_through_the_reader(tmp_path: Path) -> None:
    directory = tmp_path / "shards"
    first, second = make_games(9), make_games(7, first=100)
    manifest_path = write_shards(directory, [first, second])
    manifest = load_shards_manifest(manifest_path, verify_hashes=True)

    tokens = (directory / "shard-0000.bin").read_bytes()
    index = (directory / "shard-0000.idx").read_bytes()
    assert len(tokens) == manifest["train_shards"][0]["token_count"] * 2
    assert len(index) == (manifest["train_shards"][0]["game_count"] + 2) * 8
    assert struct.unpack_from("<Q", index, 0)[0] == len(first)
    assert struct.unpack_from("<Q", index, 8)[0] == 0
    assert struct.unpack_from("<Q", index, len(index) - 8)[0] == len(tokens) // 2

    split = ShardSplit(directory, manifest["train_shards"], context=64)
    assert len(split) == len(first) + len(second)
    for position, game in enumerate([*first, *second]):
        assert split.game(position).tolist() == list(game)
    assert split.lengths.tolist() == [len(game) for game in [*first, *second]]


def test_identity_records_every_shard_digest(tmp_path: Path) -> None:
    manifest_path, _ = _corpus(tmp_path)
    manifest = load_shards_manifest(manifest_path, verify_hashes=False)
    identity = shards_identity(manifest_path, manifest)
    assert identity["train_shard_sha256"] == [
        entry["tokens_sha256"] for entry in manifest["train_shards"]
    ]
    assert identity["tokens_train"] == manifest["counts"]["tokens_train"]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema", "minigpt.shards.v2", "unsupported"),
        ("tokenizer", "policy-v2", "tokenizer"),
        ("vocab_size", 4096, "vocab_size"),
        ("bos_token", 1, "BOS/PAD"),
    ],
)
def test_manifest_rejects_wrong_frozen_constants(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    manifest_path, _ = _corpus(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest[key] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(IntegrityError, match=message):
        load_shards_manifest(manifest_path, verify_hashes=False)


def test_manifest_rejects_truncated_and_flipped_shards(tmp_path: Path) -> None:
    manifest_path, _ = _corpus(tmp_path)
    tokens_path = manifest_path.parent / "shard-0000.bin"
    original = tokens_path.read_bytes()

    tokens_path.write_bytes(original[:-2])
    with pytest.raises(IntegrityError, match="token shard size"):
        load_shards_manifest(manifest_path, verify_hashes=False)

    flipped = bytearray(original)
    flipped[0] ^= 0x01
    tokens_path.write_bytes(bytes(flipped))
    # A single flipped byte keeps every size identity intact; only the hash catches it.
    load_shards_manifest(manifest_path, verify_hashes=False)
    with pytest.raises(IntegrityError, match="token shard checksum"):
        load_shards_manifest(manifest_path, verify_hashes=True)

    tokens_path.write_bytes(original)
    index_path = manifest_path.parent / "shard-0000.idx"
    index_path.write_bytes(index_path.read_bytes() + b"\x00" * 8)
    with pytest.raises(IntegrityError, match="index size"):
        load_shards_manifest(manifest_path, verify_hashes=False)


def test_manifest_rejects_counts_that_disagree(tmp_path: Path) -> None:
    manifest_path, _ = _corpus(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["counts"]["tokens_train"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(IntegrityError, match="counts.tokens_train"):
        load_shards_manifest(manifest_path, verify_hashes=False)


def test_long_game_keeps_bos_and_the_last_context_moves(tmp_path: Path) -> None:
    directory = tmp_path / "shards"
    context = 16
    long_game = [BOS] + list(range(1, 40))
    manifest_path = write_shards(directory, [[long_game, [BOS, 1, 2, 3]]])
    manifest = load_shards_manifest(manifest_path, verify_hashes=True)
    split = ShardSplit(directory, manifest["train_shards"], context=context)

    truncated = split.game(0).tolist()
    assert len(truncated) == context
    assert truncated[0] == BOS
    assert truncated[1:] == long_game[-(context - 1) :]
    assert split.raw_lengths.tolist() == [40, 4]
    assert split.lengths.tolist() == [context, 4]


def test_batches_pad_with_pad_and_only_move_tokens_are_supervised(tmp_path: Path) -> None:
    directory = tmp_path / "shards"
    manifest_path = write_shards(directory, [[[BOS, 1, 2, 3, 4], [BOS, 9, 8]]])
    manifest = load_shards_manifest(manifest_path, verify_hashes=True)
    split = ShardSplit(directory, manifest["train_shards"], context=32)

    batch = split.batch([0, 1])
    assert batch["tokens"].tolist() == [[BOS, 1, 2, 3, 4], [BOS, 9, 8, PAD, PAD]]
    assert batch["target_tokens"] == 4 + 2
    inputs, targets = split_inputs_and_targets(batch["tokens"])
    assert inputs.tolist() == [[BOS, 1, 2, 3], [BOS, 9, 8, PAD]]
    assert targets.tolist() == [[1, 2, 3, 4], [9, 8, PAD, PAD]]
    assert int((targets != PAD).sum()) == batch["target_tokens"]


def test_bucket_assignment_is_equal_width_over_the_context() -> None:
    assert bucket_of([1, 8, 9, 16, 17, 24, 25, 32], context=32, buckets=4).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert bucket_of([64], context=32, buckets=4).tolist() == [3]


def test_sampler_schedule_is_a_function_of_seed_and_epoch(tmp_path: Path) -> None:
    _, split = _corpus(tmp_path)
    first = BucketedSampler(split, seed=5, micro_batch=3, buckets=4)
    second = BucketedSampler(split, seed=5, micro_batch=3, buckets=4)
    other = BucketedSampler(split, seed=6, micro_batch=3, buckets=4)

    schedule = [batch.tolist() for batch in first.epoch_batches(0)]
    assert schedule == [batch.tolist() for batch in second.epoch_batches(0)]
    assert schedule != [batch.tolist() for batch in other.epoch_batches(0)]
    assert schedule != [batch.tolist() for batch in first.epoch_batches(1)]
    assert sorted(index for batch in schedule for index in batch) == list(range(len(split)))
    # Every batch is drawn from one length bucket, which is what bounds padding.
    for batch in schedule:
        assert (
            len({int(bucket) for bucket in bucket_of(split.lengths[batch], context=32, buckets=4)})
            == 1
        )


def test_sampler_resumes_the_exact_batch_sequence(tmp_path: Path) -> None:
    _, split = _corpus(tmp_path)
    sampler = BucketedSampler(split, seed=5, micro_batch=3, buckets=4)

    uninterrupted_state = SamplerState()
    stream = sampler.batches(uninterrupted_state)
    uninterrupted = [next(stream)["tokens"].tolist() for _ in range(9)]

    partial_state = SamplerState()
    stream = sampler.batches(partial_state)
    prefix = [next(stream)["tokens"].tolist() for _ in range(4)]
    resumed_state = SamplerState.from_dict(partial_state.to_dict())
    stream = sampler.batches(resumed_state)
    suffix = [next(stream)["tokens"].tolist() for _ in range(5)]

    assert prefix + suffix == uninterrupted
    assert resumed_state.to_dict() != partial_state.to_dict() or resumed_state.epoch > 0


def test_sampler_rejects_a_cursor_beyond_the_epoch(tmp_path: Path) -> None:
    _, split = _corpus(tmp_path)
    sampler = BucketedSampler(split, seed=5, micro_batch=3, buckets=4)
    with pytest.raises(IntegrityError, match="cursor"):
        next(sampler.batches(SamplerState(epoch=0, cursor=10_000)))


def test_evaluation_batches_are_stable_and_bounded(tmp_path: Path) -> None:
    directory = tmp_path / "shards"
    manifest_path = write_shards(directory, [make_games(4)], [make_games(6, first=500)])
    manifest = load_shards_manifest(manifest_path, verify_hashes=True)
    validation = ShardSplit(directory, manifest["val_shards"], context=32)

    first = [batch["tokens"].tolist() for batch in evaluation_batches(validation, 2, 2)]
    second = [batch["tokens"].tolist() for batch in evaluation_batches(validation, 2, 2)]
    assert first == second
    assert len(first) == 2
