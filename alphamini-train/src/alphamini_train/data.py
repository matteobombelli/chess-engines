"""Deterministic replay sampling over one or more validated tensor caches."""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from .errors import DependencyUnavailable, IntegrityError
from .atomic import sha256_file
from .schema import TensorCache


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise DependencyUnavailable("training requires NumPy") from error
    return np


def game_is_validation(game_id: int, seed: int, fraction: float) -> bool:
    if fraction <= 0:
        return False
    payload = seed.to_bytes(8, "little", signed=False) + game_id.to_bytes(8, "little", signed=False)
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return bucket < int(fraction * 2**64)


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


class ReplayDataset:
    """Logical concatenation with a stable game-grouped train/validation split."""

    def __init__(self, caches: Sequence[TensorCache], *, seed: int, validation_fraction: float):
        if not caches:
            raise IntegrityError("replay manifest contains no tensor caches")
        self.caches = list(caches)
        self.arrays = [cache.arrays() for cache in caches]
        self.seed = seed
        self.validation_fraction = validation_fraction
        self.boundaries: list[int] = []
        total = 0
        for cache in caches:
            total += cache.record_count
            self.boundaries.append(total)
        np = _require_numpy()
        train: list[int] = []
        validation: list[int] = []
        base = 0
        for cache, arrays in zip(caches, self.arrays, strict=True):
            for local, game_id in enumerate(arrays["game_ids"]):
                destination = (
                    validation
                    if game_is_validation(int(game_id), seed, validation_fraction)
                    else train
                )
                destination.append(base + local)
            base += cache.record_count
        self.train_indices = np.asarray(train, dtype=np.int64)
        self.validation_indices = np.asarray(validation, dtype=np.int64)
        if len(self.train_indices) == 0:
            raise IntegrityError("game-grouped split produced no training records")

    def identity(self) -> list[dict[str, Any]]:
        return [
            {
                "tensor_manifest_sha256": sha256_file(cache.manifest_path),
                "source_collection_sha256": cache.manifest["source_collection_sha256"],
                "record_count": cache.record_count,
                "encoder_schema": cache.manifest["encoder_schema"],
                "action_schema": cache.manifest["action_schema"],
            }
            for cache in self.caches
        ]

    def _location(self, global_index: int) -> tuple[int, int]:
        cache_index = bisect.bisect_right(self.boundaries, global_index)
        prior = 0 if cache_index == 0 else self.boundaries[cache_index - 1]
        return cache_index, global_index - prior

    def permutation(self, state: SamplerState) -> Any:
        np = _require_numpy()
        # Epochs are independently reproducible; the full permutation need not live in checkpoints.
        sequence = np.random.SeedSequence([self.seed, state.epoch, 0xA17A])
        result = np.random.default_rng(sequence).permutation(self.train_indices)
        if state.cursor > len(result):
            raise IntegrityError("sampler cursor is beyond the epoch")
        return result

    def batches(self, state: SamplerState, batch_size: int) -> Iterator[dict[str, Any]]:
        np = _require_numpy()
        while True:
            permutation = self.permutation(state)
            while state.cursor < len(permutation):
                selected = permutation[state.cursor : state.cursor + batch_size]
                state.cursor += len(selected)
                grouped: dict[int, list[tuple[int, int]]] = {}
                for row, global_index in enumerate(selected):
                    cache_index, local = self._location(int(global_index))
                    grouped.setdefault(cache_index, []).append((row, local))
                inputs = np.empty((len(selected), 22, 8, 8), dtype=np.float32)
                wdl = np.empty((len(selected), 3), dtype=np.float32)
                policy_row_chunks: list[Any] = []
                policy_index_chunks: list[Any] = []
                policy_value_chunks: list[Any] = []
                for cache_index, rows in grouped.items():
                    arrays = self.arrays[cache_index]
                    locals_ = np.asarray([local for _, local in rows], dtype=np.int64)
                    destinations = np.asarray([row for row, _ in rows], dtype=np.int64)
                    inputs[destinations] = arrays["inputs"][locals_]
                    wdl[destinations] = arrays["wdl"][locals_]
                    starts = np.asarray(arrays["policy_offsets"][locals_], dtype=np.int64)
                    ends = np.asarray(arrays["policy_offsets"][locals_ + 1], dtype=np.int64)
                    lengths = ends - starts
                    if np.any(lengths <= 0):
                        raise IntegrityError("training record has an empty policy target")
                    policy_row_chunks.append(np.repeat(destinations, lengths))
                    # Keep sparse targets as NumPy views/chunks. Converting each
                    # uint16/float32 scalar through Python and then packing it
                    # back into NumPy dominated batch assembly for large batches.
                    policy_index_chunks.extend(
                        arrays["policy_indices"][start:end] for start, end in zip(starts, ends)
                    )
                    policy_value_chunks.extend(
                        arrays["policy_values"][start:end] for start, end in zip(starts, ends)
                    )
                policy_rows = np.concatenate(policy_row_chunks).astype(np.int64, copy=False)
                policy_indices = np.concatenate(policy_index_chunks).astype(np.int64, copy=False)
                policy_values = np.concatenate(policy_value_chunks).astype(np.float32, copy=False)
                yield {
                    "inputs": inputs,
                    "wdl": wdl,
                    "policy_rows": policy_rows,
                    "policy_indices": policy_indices,
                    "policy_values": policy_values,
                }
            state.epoch += 1
            state.cursor = 0

    def validation_batches(self, batch_size: int) -> Iterator[dict[str, Any]]:
        # Validation order is deliberately stable and never updates sampler state.
        if len(self.validation_indices) == 0:
            return
        state = SamplerState()
        original = self.train_indices
        self.train_indices = self.validation_indices
        try:
            iterator = self.batches(state, batch_size)
            remaining = len(self.validation_indices)
            while remaining > 0:
                batch = next(iterator)
                remaining -= len(batch["inputs"])
                yield batch
        finally:
            self.train_indices = original


def updates_for_new_positions(new_positions: int, sample_ratio: float, batch_size: int) -> int:
    if new_positions < 1:
        raise IntegrityError("new collection contains no training positions")
    return max(1, math.ceil(new_positions * sample_ratio / batch_size))
