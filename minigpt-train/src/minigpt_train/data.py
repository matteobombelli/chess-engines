"""Byte-exact reader and deterministic bucketed sampler for `minigpt-ingest` shards.

`shard-NNNN.bin` is a little-endian `u16` token stream: games concatenated, each game
being BOS followed by one action token per ply, with no header and no padding.
`shard-NNNN.idx` is little-endian `u64` throughout: a game count `G`, then `G + 1`
offsets measured in *tokens*, so byte offset is twice the token offset. Game `i`
occupies `offsets[i]..offsets[i + 1]`, `offsets[0]` is zero, `offsets[G]` is the token
count, and the index file is exactly `(G + 2) * 8` bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .atomic import read_json, sha256_file
from .config import BOS_TOKEN, PAD_TOKEN, POLICY_SIZE, TOKENIZER, VOCAB_SIZE
from .errors import DependencyUnavailable, IntegrityError

SHARDS_MANIFEST_SCHEMA = "minigpt.shards.v1"
SHARDS_MANIFEST_FILE = "shards.json"
SHARD_FILE_FIELDS = {
    "tokens_path",
    "index_path",
    "tokens_sha256",
    "index_sha256",
    "token_count",
    "game_count",
}


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise DependencyUnavailable("training requires NumPy") from error
    return np


def _positive_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise IntegrityError(f"shard manifest {field} is not a non-negative integer")
    return value


def load_shards_manifest(path: Path, *, verify_hashes: bool) -> dict[str, Any]:
    """Validate the frozen constants, then every per-file size and length identity."""

    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema") != SHARDS_MANIFEST_SCHEMA:
        raise IntegrityError(f"unsupported shards manifest: {path}")
    if manifest.get("tokenizer") != TOKENIZER:
        raise IntegrityError(f"shards manifest tokenizer must be {TOKENIZER!r}")
    if manifest.get("vocab_size") != VOCAB_SIZE:
        raise IntegrityError(f"shards manifest vocab_size must be {VOCAB_SIZE}")
    if manifest.get("bos_token") != BOS_TOKEN or manifest.get("pad_token") != PAD_TOKEN:
        raise IntegrityError("shards manifest BOS/PAD tokens differ from the v1 tokenizer")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise IntegrityError("shards manifest lacks counts")
    directory = path.parent
    for split, tokens_key, games_key in (
        ("train_shards", "tokens_train", "games_train"),
        ("val_shards", "tokens_val", "games_val"),
    ):
        files = manifest.get(split)
        if not isinstance(files, list):
            raise IntegrityError(f"shards manifest {split} must be a list")
        tokens = 0
        games = 0
        for entry in files:
            if not isinstance(entry, dict) or set(entry) != SHARD_FILE_FIELDS:
                raise IntegrityError(f"shards manifest {split} entry has an invalid schema")
            token_count = _positive_integer(entry["token_count"], "token_count")
            game_count = _positive_integer(entry["game_count"], "game_count")
            tokens_path = _shard_path(directory, entry["tokens_path"])
            index_path = _shard_path(directory, entry["index_path"])
            if tokens_path.stat().st_size != token_count * 2:
                raise IntegrityError(f"token shard size disagrees with token_count: {tokens_path}")
            if index_path.stat().st_size != (game_count + 2) * 8:
                raise IntegrityError(f"index size disagrees with game_count: {index_path}")
            if verify_hashes:
                if sha256_file(tokens_path) != entry["tokens_sha256"]:
                    raise IntegrityError(f"token shard checksum mismatch: {tokens_path}")
                if sha256_file(index_path) != entry["index_sha256"]:
                    raise IntegrityError(f"index checksum mismatch: {index_path}")
            tokens += token_count
            games += game_count
        if _positive_integer(counts.get(tokens_key), tokens_key) != tokens:
            raise IntegrityError(f"shards manifest counts.{tokens_key} disagrees with {split}")
        if _positive_integer(counts.get(games_key), games_key) != games:
            raise IntegrityError(f"shards manifest counts.{games_key} disagrees with {split}")
    if not manifest["train_shards"]:
        raise IntegrityError("shards manifest contains no training shards")
    return manifest


def _shard_path(directory: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise IntegrityError(f"shard file name is not a plain file name: {name!r}")
    path = directory / name
    if not path.is_file():
        raise IntegrityError(f"shard file is missing: {path}")
    return path


def shards_identity(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """The exact corpus a checkpoint was trained from, recorded in the run ledger."""

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "tokenizer": manifest["tokenizer"],
        "vocab_size": manifest["vocab_size"],
        "tokens_train": manifest["counts"]["tokens_train"],
        "tokens_val": manifest["counts"]["tokens_val"],
        "games_train": manifest["counts"]["games_train"],
        "games_val": manifest["counts"]["games_val"],
        "train_shard_sha256": [entry["tokens_sha256"] for entry in manifest["train_shards"]],
        "val_shard_sha256": [entry["tokens_sha256"] for entry in manifest["val_shards"]],
    }


class ShardSplit:
    """One split's shards, memory-mapped, with a flat game index across files."""

    def __init__(self, directory: Path, files: Sequence[dict[str, Any]], *, context: int):
        np = _require_numpy()
        if context < 2:
            raise IntegrityError("model.ctx must be at least 2")
        self.context = context
        self.tokens: list[Any] = []
        self.offsets: list[Any] = []
        starts = [0]
        lengths: list[Any] = []
        for entry in files:
            tokens_path = _shard_path(directory, entry["tokens_path"])
            index_path = _shard_path(directory, entry["index_path"])
            index = np.memmap(index_path, dtype="<u8", mode="r")
            if int(index[0]) != int(entry["game_count"]):
                raise IntegrityError(f"index game count disagrees with the manifest: {index_path}")
            offsets = np.asarray(index[1:], dtype=np.int64)
            if offsets[0] != 0 or int(offsets[-1]) != int(entry["token_count"]):
                raise IntegrityError(f"index offsets do not span the token shard: {index_path}")
            shard_lengths = np.diff(offsets)
            if shard_lengths.size and int(shard_lengths.min()) < 2:
                raise IntegrityError(
                    f"index contains a game shorter than BOS + one move: {index_path}"
                )
            self.tokens.append(np.memmap(tokens_path, dtype="<u2", mode="r"))
            self.offsets.append(offsets)
            lengths.append(shard_lengths)
            starts.append(starts[-1] + int(entry["game_count"]))
        self.shard_starts = np.asarray(starts, dtype=np.int64)
        self.raw_lengths = (
            np.concatenate(lengths).astype(np.int64, copy=False)
            if lengths
            else np.zeros(0, dtype=np.int64)
        )
        # A game longer than the context is truncated to BOS plus its last ctx-1 moves.
        self.lengths = np.minimum(self.raw_lengths, context)

    def __len__(self) -> int:
        return int(self.lengths.size)

    def game(self, index: int) -> Any:
        np = _require_numpy()
        if index < 0 or index >= len(self):
            raise IntegrityError(f"game index out of range: {index}")
        shard = int(np.searchsorted(self.shard_starts, index, side="right")) - 1
        local = index - int(self.shard_starts[shard])
        offsets = self.offsets[shard]
        tokens = np.asarray(
            self.tokens[shard][int(offsets[local]) : int(offsets[local + 1])], dtype=np.int64
        )
        if tokens.size > self.context:
            # Keep BOS, then the most recent moves: the tail is what a resumed game sees.
            tokens = np.concatenate((tokens[:1], tokens[tokens.size - self.context + 1 :]))
        return tokens

    def batch(self, indices: Any) -> dict[str, Any]:
        np = _require_numpy()
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            raise IntegrityError("cannot assemble an empty batch")
        width = int(self.lengths[indices].max())
        tokens = np.full((indices.size, width), PAD_TOKEN, dtype=np.int64)
        for row, index in enumerate(indices):
            game = self.game(int(index))
            tokens[row, : game.size] = game
        # Each game contributes one supervised target per move token.
        return {"tokens": tokens, "target_tokens": int(self.lengths[indices].sum() - indices.size)}


def bucket_of(lengths: Any, *, context: int, buckets: int) -> Any:
    """Map effective game lengths in 1..context onto equal-width buckets."""

    np = _require_numpy()
    index = (np.asarray(lengths, dtype=np.int64) - 1) * buckets // context
    return np.clip(index, 0, buckets - 1)


@dataclass
class SamplerState:
    epoch: int = 0
    cursor: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SamplerState":
        epoch = value.get("epoch")
        cursor = value.get("cursor")
        if not isinstance(epoch, int) or epoch < 0 or not isinstance(cursor, int) or cursor < 0:
            raise IntegrityError("invalid sampler state")
        return cls(epoch, cursor)


class BucketedSampler:
    """Length-bucketed batches whose whole schedule is a function of (seed, epoch)."""

    def __init__(self, split: ShardSplit, *, seed: int, micro_batch: int, buckets: int):
        if micro_batch < 1 or buckets < 1:
            raise IntegrityError("micro_batch and length_buckets must be positive")
        if len(split) == 0:
            raise IntegrityError("split contains no games")
        self.split = split
        self.seed = seed
        self.micro_batch = micro_batch
        self.buckets = buckets

    def epoch_batches(self, epoch: int) -> list[Any]:
        np = _require_numpy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 0x6D67]))
        order = rng.permutation(len(self.split))
        assignment = bucket_of(self.split.lengths, context=self.split.context, buckets=self.buckets)
        batches: list[Any] = []
        for bucket in range(self.buckets):
            members = order[assignment[order] == bucket]
            for start in range(0, members.size, self.micro_batch):
                batches.append(members[start : start + self.micro_batch])
        # Bucket order would otherwise feed the model every short game first.
        return [batches[index] for index in rng.permutation(len(batches))]

    def batches(self, state: SamplerState) -> Iterator[dict[str, Any]]:
        while True:
            schedule = self.epoch_batches(state.epoch)
            if state.cursor > len(schedule):
                raise IntegrityError("sampler cursor is beyond the epoch")
            while state.cursor < len(schedule):
                selected = schedule[state.cursor]
                state.cursor += 1
                yield self.split.batch(selected)
            state.epoch += 1
            state.cursor = 0


def evaluation_batches(split: ShardSplit, micro_batch: int, limit: int) -> Iterator[dict[str, Any]]:
    """A fixed prefix of the validation split in index order; never sampler-dependent."""

    np = _require_numpy()
    produced = 0
    for start in range(0, len(split), micro_batch):
        if produced >= limit:
            return
        yield split.batch(np.arange(start, min(start + micro_batch, len(split)), dtype=np.int64))
        produced += 1


def split_inputs_and_targets(tokens: Any) -> tuple[Any, Any]:
    """Next-token pairs; targets are always move tokens or PAD, never BOS."""

    np = _require_numpy()
    inputs = tokens[:, :-1]
    targets = np.array(tokens[:, 1:], copy=True)
    if int(np.max(targets, initial=0)) >= VOCAB_SIZE:
        raise IntegrityError("shard contains a token outside the vocabulary")
    supervised = targets != PAD_TOKEN
    if bool(np.any(targets[supervised] >= POLICY_SIZE)):
        raise IntegrityError("supervised target is not a move token")
    return inputs, targets
