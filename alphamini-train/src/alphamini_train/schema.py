"""Exact readers for the wire types defined in Rust ``alphamini/src/record.rs``."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .atomic import read_json, sha256_file
from .errors import DependencyUnavailable, IntegrityError

POSITION_SCHEMA = "position-record-v1"
GAME_SCHEMA = "game-record-v1"
SHARD_SCHEMA = "self-play-shard-v1"
COLLECTION_SCHEMA = "collection-manifest-v1"
TENSOR_CACHE_SCHEMA = "tensor-cache-manifest-v1"
ENCODER_SCHEMA = "encoder-v1"
ACTION_SCHEMA = "policy-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DTYPE_BYTES = {"f32-le": 4, "u16-le": 2, "u64-le": 8}
U64_MASK = 2**64 - 1
OUTCOMES = frozenset({"white_win", "draw", "black_win"})
TERMINATIONS = frozenset(
    {
        "checkmate",
        "stalemate",
        "insufficient_material",
        "threefold_repetition",
        "fifty_move_rule",
        "ply_limit",
    }
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int(value: Any, field: str) -> int:
    if not _is_int(value):
        raise IntegrityError(f"{field} must be an integer")
    return value


def _bounded(mapping: dict[str, Any], field: str, minimum: int, maximum: int) -> int:
    result = _int(mapping.get(field), field)
    if not minimum <= result <= maximum:
        raise IntegrityError(f"{field} must be in [{minimum}, {maximum}]")
    return result


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise IntegrityError(
            f"{context} fields disagree with Rust wire schema; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise IntegrityError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_result(outcome: Any, termination: Any) -> tuple[str, str]:
    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        raise IntegrityError("invalid absolute game outcome")
    if not isinstance(termination, str) or termination not in TERMINATIONS:
        raise IntegrityError("invalid termination")
    if (termination == "checkmate") != (outcome != "draw"):
        raise IntegrityError("game outcome is inconsistent with termination")
    return outcome, termination


def _safe_child(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise IntegrityError("manifest path must be a non-empty string")
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as error:
        raise IntegrityError(f"manifest path escapes its directory: {relative}") from error
    return candidate


def derive_game_seed(collection_seed: int, game_id: int) -> int:
    """Exact wrapping SplitMix64 seed derivation exported by Rust record v1."""

    value = (collection_seed ^ game_id) & U64_MASK
    value = (value + 0x9E3779B97F4A7C15) & U64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & U64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & U64_MASK
    return (value ^ (value >> 31)) & U64_MASK


@dataclass(frozen=True)
class PolicyTarget:
    from_square: int
    to_square: int
    promotion: str | None
    visits: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PolicyTarget":
        if not isinstance(value, dict):
            raise IntegrityError("policy visit is not an object")
        _exact_keys(value, {"from", "to", "promotion", "visits"}, "PolicyVisitV1")
        result = cls(
            _bounded(value, "from", 0, 63),
            _bounded(value, "to", 0, 63),
            value.get("promotion"),
            _bounded(value, "visits", 1, 2**32 - 1),
        )
        if result.promotion not in {None, "queen", "rook", "bishop", "knight"}:
            raise IntegrityError(f"invalid promotion: {result.promotion!r}")
        return result


@dataclass(frozen=True)
class PositionRecordV1:
    game_id: int
    ply: int
    piece_bitboards: tuple[int, ...]
    side_to_move: str
    castling_rights: int
    en_passant_square: int | None
    halfmove_clock: int
    fullmove_number: int
    prior_occurrences: int
    previous_move_uci: str | None
    selected_move_uci: str
    policy: tuple[PolicyTarget, ...]
    outcome: str
    termination: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PositionRecordV1":
        if not isinstance(value, dict):
            raise IntegrityError("position record is not an object")
        expected = {
            "schema",
            "game_id",
            "ply",
            "piece_bitboards",
            "side_to_move",
            "castling_rights",
            "en_passant_square",
            "halfmove_clock",
            "fullmove_number",
            "prior_occurrences",
            "previous_move_uci",
            "selected_move_uci",
            "policy",
            "outcome",
            "termination",
        }
        _exact_keys(value, expected, "PositionRecordV1")
        if value.get("schema") != POSITION_SCHEMA:
            raise IntegrityError("unsupported position record schema")
        bitboards = value.get("piece_bitboards")
        if not isinstance(bitboards, list) or len(bitboards) != 12:
            raise IntegrityError("piece_bitboards must contain 12 integers")
        parsed = tuple(_int(number, "piece_bitboards") for number in bitboards)
        if any(number < 0 or number >= 2**64 for number in parsed):
            raise IntegrityError("piece bitboard is outside u64")
        occupied = 0
        for bitboard in parsed:
            if occupied & bitboard:
                raise IntegrityError("piece bitboards overlap")
            occupied |= bitboard
        if parsed[5].bit_count() != 1 or parsed[11].bit_count() != 1:
            raise IntegrityError("position must contain one king of each color")
        side = value.get("side_to_move")
        if side not in {"white", "black"}:
            raise IntegrityError("side_to_move must be white or black")
        en_passant = value.get("en_passant_square")
        if en_passant is not None and (not _is_int(en_passant) or not 0 <= en_passant < 64):
            raise IntegrityError("en_passant_square must be null or a square index")
        previous = value.get("previous_move_uci")
        if previous is not None and not isinstance(previous, str):
            raise IntegrityError("previous_move_uci must be null or a string")
        selected = value.get("selected_move_uci")
        if not isinstance(selected, str) or not selected:
            raise IntegrityError("selected_move_uci must be a non-empty string")
        raw_policy = value.get("policy")
        if not isinstance(raw_policy, list) or not raw_policy:
            raise IntegrityError("policy must be a non-empty array")
        policy = tuple(PolicyTarget.from_dict(item) for item in raw_policy)
        targets = {(target.from_square, target.to_square, target.promotion) for target in policy}
        if len(targets) != len(policy):
            raise IntegrityError("policy contains a duplicate sparse target")
        outcome, termination = _validate_result(value.get("outcome"), value.get("termination"))
        return cls(
            _bounded(value, "game_id", 0, 2**64 - 1),
            _bounded(value, "ply", 0, 2**16 - 1),
            parsed,
            side,
            _bounded(value, "castling_rights", 0, 15),
            en_passant,
            _bounded(value, "halfmove_clock", 0, 2**32 - 1),
            _bounded(value, "fullmove_number", 0, 2**32 - 1),
            _bounded(value, "prior_occurrences", 0, 2),
            previous,
            selected,
            policy,
            outcome,
            termination,
        )


def _decode_shard(path: Path) -> dict[str, Any]:
    try:
        import msgpack
        import zstandard
    except ImportError as error:
        raise DependencyUnavailable("reading raw shards requires msgpack and zstandard") from error
    try:
        with path.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                decoded = reader.read()
        value = msgpack.unpackb(decoded, raw=False, strict_map_key=False)
    except (OSError, ValueError, zstandard.ZstdError) as error:
        raise IntegrityError(f"cannot decode Rust SelfPlayShardV1 {path}: {error}") from error
    if not isinstance(value, dict):
        raise IntegrityError("SelfPlayShardV1 is not an object")
    return value


def iter_shard(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_model_sha256: str | None = None,
    expected_seed: int | None = None,
    expected_simulations: int | None = None,
    expected_max_plies: int | None = None,
) -> Iterator[PositionRecordV1]:
    """Decode Rust's nested zstd MessagePack ``SelfPlayShardV1``."""

    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise IntegrityError(f"raw shard checksum mismatch: {path}")
    shard = _decode_shard(path)
    _exact_keys(
        shard,
        {
            "schema",
            "encoder_schema",
            "action_schema",
            "seed",
            "simulations",
            "max_plies",
            "games",
        },
        "SelfPlayShardV1",
    )
    if (
        shard.get("schema") != SHARD_SCHEMA
        or shard.get("encoder_schema") != ENCODER_SCHEMA
        or shard.get("action_schema") != ACTION_SCHEMA
    ):
        raise IntegrityError("unsupported SelfPlayShardV1 header")
    seed = _bounded(shard, "seed", 0, U64_MASK)
    simulations = _bounded(shard, "simulations", 1, 2**32 - 1)
    max_plies = _bounded(shard, "max_plies", 1, 512)
    if expected_seed is not None and seed != expected_seed:
        raise IntegrityError("shard seed differs from its collection")
    if expected_simulations is not None and simulations != expected_simulations:
        raise IntegrityError("shard simulations differs from its collection")
    if expected_max_plies is not None and max_plies != expected_max_plies:
        raise IntegrityError("shard max_plies differs from its collection")
    games = shard.get("games")
    if not isinstance(games, list) or not games:
        raise IntegrityError("SelfPlayShardV1 games must be non-empty")
    prior_game: int | None = None
    for game in games:
        if not isinstance(game, dict):
            raise IntegrityError("GameRecordV1 is not an object")
        _exact_keys(
            game,
            {
                "schema",
                "game_id",
                "seed",
                "model_sha256",
                "outcome",
                "termination",
                "plies",
                "positions",
            },
            "GameRecordV1",
        )
        if game.get("schema") != GAME_SCHEMA:
            raise IntegrityError("unsupported GameRecordV1 schema")
        game_id = _bounded(game, "game_id", 0, 2**64 - 1)
        game_seed = _bounded(game, "seed", 0, U64_MASK)
        if game_seed != derive_game_seed(seed, game_id):
            raise IntegrityError("game seed differs from deterministic SplitMix64 derivation")
        game_model = _hash(game.get("model_sha256"), "game.model_sha256")
        if expected_model_sha256 is not None and game_model != expected_model_sha256:
            raise IntegrityError("shard contains a game generated by another model")
        positions = game.get("positions")
        if not isinstance(positions, list) or not 1 <= len(positions) <= min(512, max_plies):
            raise IntegrityError("game positions exceed the shard max_plies contract")
        if _bounded(game, "plies", 0, 2**16 - 1) != len(positions):
            raise IntegrityError("game plies differs from position count")
        outcome, termination = _validate_result(game.get("outcome"), game.get("termination"))
        if termination == "ply_limit" and len(positions) != max_plies:
            raise IntegrityError("ply-limit game length differs from shard max_plies")
        if prior_game is not None and game_id <= prior_game:
            raise IntegrityError("game IDs must be strictly increasing")
        prior_game = game_id
        for ply, raw in enumerate(positions):
            record = PositionRecordV1.from_dict(raw)
            if (
                record.game_id != game_id
                or record.ply != ply
                or record.outcome != outcome
                or record.termination != termination
            ):
                raise IntegrityError("position and game metadata disagree")
            if sum(target.visits for target in record.policy) != simulations:
                raise IntegrityError("position visit sum differs from shard simulations")
            yield record


def validate_collection_manifest(
    path: Path | str, *, decode_shards: bool = False
) -> dict[str, Any]:
    path = Path(path).resolve()
    value = read_json(path)
    if not isinstance(value, dict):
        raise IntegrityError("CollectionManifestV1 is not an object")
    _exact_keys(
        value,
        {
            "schema",
            "encoder_schema",
            "action_schema",
            "run_id",
            "cycle_id",
            "game_id_start",
            "model_sha256",
            "config_sha256",
            "seed",
            "simulations",
            "max_plies",
            "game_count",
            "position_count",
            "shards",
        },
        "CollectionManifestV1",
    )
    if (
        value.get("schema") != COLLECTION_SCHEMA
        or value.get("encoder_schema") != ENCODER_SCHEMA
        or value.get("action_schema") != ACTION_SCHEMA
    ):
        raise IntegrityError("unsupported CollectionManifestV1 header")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise IntegrityError("collection run_id must be non-empty")
    for field in ("cycle_id", "game_id_start", "seed", "game_count", "position_count"):
        _bounded(value, field, 0, 2**64 - 1)
    _bounded(value, "simulations", 1, 2**32 - 1)
    _bounded(value, "max_plies", 1, 512)
    _hash(value.get("model_sha256"), "collection.model_sha256")
    _hash(value.get("config_sha256"), "collection.config_sha256")
    shards = value.get("shards")
    if not isinstance(shards, list) or not shards:
        raise IntegrityError("collection shards must be non-empty")
    total_games = 0
    total_positions = 0
    prior_last: int | None = None
    seen: set[Path] = set()
    expected_fields = {
        "path",
        "bytes",
        "sha256",
        "first_game_id",
        "last_game_id",
        "game_count",
        "position_count",
    }
    for descriptor in shards:
        if not isinstance(descriptor, dict):
            raise IntegrityError("ShardDescriptorV1 is not an object")
        _exact_keys(descriptor, expected_fields, "ShardDescriptorV1")
        shard_path = _safe_child(path.parent, descriptor.get("path"))
        if shard_path in seen:
            raise IntegrityError("duplicate shard path")
        seen.add(shard_path)
        size = _bounded(descriptor, "bytes", 0, 2**64 - 1)
        digest = _hash(descriptor.get("sha256"), "shard.sha256")
        if not shard_path.is_file() or shard_path.stat().st_size != size:
            raise IntegrityError(f"missing or truncated shard: {shard_path}")
        if sha256_file(shard_path) != digest:
            raise IntegrityError(f"shard checksum mismatch: {shard_path}")
        first = _bounded(descriptor, "first_game_id", 0, 2**64 - 1)
        last = _bounded(descriptor, "last_game_id", 0, 2**64 - 1)
        count = _bounded(descriptor, "game_count", 0, 2**64 - 1)
        positions = _bounded(descriptor, "position_count", 0, 2**64 - 1)
        if first > last or count != last - first + 1:
            raise IntegrityError("invalid shard game range")
        if prior_last is not None and first != prior_last + 1:
            raise IntegrityError("shard game ranges have a gap or overlap")
        prior_last = last
        total_games += count
        total_positions += positions
        if decode_shards:
            records = list(
                iter_shard(
                    shard_path,
                    expected_sha256=digest,
                    expected_model_sha256=value["model_sha256"],
                    expected_seed=value["seed"],
                    expected_simulations=value["simulations"],
                    expected_max_plies=value["max_plies"],
                )
            )
            game_ids = sorted({record.game_id for record in records})
            if (
                len(records) != positions
                or len(game_ids) != count
                or game_ids[0] != first
                or game_ids[-1] != last
                or any(right != left + 1 for left, right in zip(game_ids, game_ids[1:]))
            ):
                raise IntegrityError("decoded shard counts differ from descriptor")
    if total_games != value["game_count"] or total_positions != value["position_count"]:
        raise IntegrityError("collection totals differ from shard descriptors")
    if shards[0]["first_game_id"] != value["game_id_start"]:
        raise IntegrityError("collection game_id_start differs from first shard")
    return value


@dataclass(frozen=True)
class TensorDescriptor:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    sha256: str


class TensorCache:
    """Validated mmap view of Rust ``TensorCacheManifestV1``."""

    NAMES = ("inputs", "policy_offsets", "policy_indices", "policy_values", "wdl", "game_ids")

    def __init__(self, manifest_path: Path | str, *, verify_hashes: bool = True):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = read_json(self.manifest_path)
        self.tensors: dict[str, TensorDescriptor] = {}
        self._validate(verify_hashes)

    def _validate(self, verify_hashes: bool) -> None:
        value = self.manifest
        if not isinstance(value, dict):
            raise IntegrityError("TensorCacheManifestV1 is not an object")
        _exact_keys(
            value,
            {
                "schema",
                "encoder_schema",
                "action_schema",
                "source_collection_sha256",
                "record_count",
                "policy_size",
                "input_shape",
                *self.NAMES,
            },
            "TensorCacheManifestV1",
        )
        if (
            value.get("schema") != TENSOR_CACHE_SCHEMA
            or value.get("encoder_schema") != ENCODER_SCHEMA
            or value.get("action_schema") != ACTION_SCHEMA
        ):
            raise IntegrityError("unsupported TensorCacheManifestV1 header")
        _hash(value.get("source_collection_sha256"), "source_collection_sha256")
        count = _bounded(value, "record_count", 1, 2**64 - 1)
        if _bounded(value, "policy_size", 1, 2**64 - 1) != 4672:
            raise IntegrityError("tensor policy_size must be 4672")
        if value.get("input_shape") != [22, 8, 8]:
            raise IntegrityError("tensor input_shape must be [22,8,8]")
        descriptor_fields = {"path", "dtype", "shape", "bytes", "sha256"}
        for name in self.NAMES:
            raw = value.get(name)
            if not isinstance(raw, dict):
                raise IntegrityError(f"tensor descriptor {name} is not an object")
            _exact_keys(raw, descriptor_fields, "TensorDescriptorV1")
            dtype = raw.get("dtype")
            shape = raw.get("shape")
            if dtype not in DTYPE_BYTES:
                raise IntegrityError(f"unsupported Rust tensor dtype: {dtype!r}")
            if (
                not isinstance(shape, list)
                or not shape
                or any(not _is_int(x) or x < 0 for x in shape)
            ):
                raise IntegrityError(f"invalid shape for {name}")
            byte_length = _bounded(raw, "bytes", 0, 2**64 - 1)
            if math.prod(shape) * DTYPE_BYTES[dtype] != byte_length:
                raise IntegrityError(f"shape/byte mismatch for {name}")
            tensor_path = _safe_child(self.manifest_path.parent, raw.get("path"))
            if not tensor_path.is_file() or tensor_path.stat().st_size != byte_length:
                raise IntegrityError(f"missing or truncated tensor: {tensor_path}")
            digest = _hash(raw.get("sha256"), f"{name}.sha256")
            if verify_hashes and sha256_file(tensor_path) != digest:
                raise IntegrityError(f"checksum mismatch for {name}: {tensor_path}")
            self.tensors[name] = TensorDescriptor(
                tensor_path, dtype, tuple(shape), byte_length, digest
            )
        shapes = {name: descriptor.shape for name, descriptor in self.tensors.items()}
        if shapes["inputs"] != (count, 22, 8, 8):
            raise IntegrityError("inputs shape differs from record_count")
        if shapes["wdl"] != (count, 3) or shapes["game_ids"] != (count,):
            raise IntegrityError("WDL/game_ids shape differs from record_count")
        if shapes["policy_offsets"] != (count + 1,):
            raise IntegrityError("policy_offsets must have N+1 elements")
        nnz = shapes["policy_indices"][0]
        if len(shapes["policy_indices"]) != 1 or shapes["policy_values"] != (nnz,):
            raise IntegrityError("sparse policy arrays disagree")
        expected_dtypes = {
            "inputs": "f32-le",
            "policy_offsets": "u64-le",
            "policy_indices": "u16-le",
            "policy_values": "f32-le",
            "wdl": "f32-le",
            "game_ids": "u64-le",
        }
        for name, expected in expected_dtypes.items():
            if self.tensors[name].dtype != expected:
                raise IntegrityError(f"{name} dtype must be {expected}")

    def arrays(self) -> dict[str, Any]:
        try:
            import numpy as np
        except ImportError as error:
            raise DependencyUnavailable("tensor-cache mmap requires NumPy") from error
        mapping = {"f32-le": "<f4", "u16-le": "<u2", "u64-le": "<u8"}
        arrays = {
            name: np.memmap(
                descriptor.path, dtype=mapping[descriptor.dtype], mode="r", shape=descriptor.shape
            )
            for name, descriptor in self.tensors.items()
        }
        offsets = arrays["policy_offsets"]
        indices = arrays["policy_indices"]
        values = arrays["policy_values"]
        if (
            int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or bool((offsets[1:] <= offsets[:-1]).any())
        ):
            raise IntegrityError("policy offsets must be strictly increasing from zero to NNZ")
        if len(indices) and int(indices.max()) >= 4672:
            raise IntegrityError("policy index is out of range")
        if not bool(np.isfinite(values).all()) or bool((values < 0).any()):
            raise IntegrityError("policy values are invalid")
        sums = np.add.reduceat(values, offsets[:-1].astype(np.intp, copy=False))
        if not bool(np.allclose(sums, 1.0, atol=1e-5)):
            raise IntegrityError("each policy row must sum to one")
        inputs = arrays["inputs"]
        if (
            not bool(np.isfinite(inputs).all())
            or bool((inputs < 0).any())
            or bool((inputs > 1).any())
        ):
            raise IntegrityError("input tensor must contain finite values in [0,1]")
        wdl = arrays["wdl"]
        if not bool(np.isfinite(wdl).all()) or bool((wdl < 0).any()):
            raise IntegrityError("WDL tensor contains invalid values")
        if not bool(np.allclose(wdl.sum(axis=1), 1.0, atol=1e-5)):
            raise IntegrityError("each WDL row must sum to one")
        return arrays

    @property
    def record_count(self) -> int:
        return int(self.manifest["record_count"])
