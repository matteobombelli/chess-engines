# AlphaMini v1 design

## Purpose and ownership

AlphaMini is a compact AlphaZero-style chess engine intended to make the full
ML-systems lifecycle inspectable on one RTX 3070. Rust owns chess legality,
canonical encoding, MCTS, self-play, and raw-to-tensor materialization. Python
owns the CNN, optimization, checkpoint recovery, ONNX export, and immutable run
ledger. Python never reconstructs chess positions or legal moves.

The first published experiment is pure self-play. Its fixed strength ladder is
Random, `MinimaxDepth1V1`, `MinimaxDepth2V1`, then `MinimaxDepth3V1`. An 1800
Chess.com move-quality estimate is a stretch goal, not a model label.

## Frozen model interfaces

The FP32 input has shape `[batch,22,8,8]`. The mover is canonicalized to the
white side of the tensor by rank-flipping black positions while retaining file
order. Planes are 12 mover/opponent piece planes, two repetition planes, one
absolute-color plane, four castling planes, en-passant target, clipped
halfmove-clock, and constant one. No position history or last-move attention cue
is present.

The action space is 73 movement planes over 64 canonical origins. Its exact
linear index is:

```text
action_index = movement_plane * 64 + canonical_origin
```

The PyTorch policy head returns NCHW `[batch,73,8,8]` and flattens it directly;
it must never permute to origin-major order. The network contains a 64-channel
stem, six 64-channel squeeze-excitation residual blocks, a 73-plane spatial
policy head, and a three-logit W/D/L head. The WDL scalar used by search is
`P(win) - P(loss)` from the leaf side-to-move perspective.

ONNX uses input `input` and outputs `policy_logits` and `wdl_logits`, dynamic
batch, FP32, and opset 17. A sidecar binds architecture and encoder/action
schemas, parent checkpoint, cycle/step, semantic configuration, file checksum,
and measured PyTorch/ONNX Runtime parity. Missing or unverified manifests are not
production models.

The production doctor also runs the same frozen encoded tensor through PyTorch,
Python ORT CPU, and Rust ORT CUDA, comparing policy and WDL logits with the
configured absolute/relative tolerances. The fixture is independently bound to
FEN `rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 5 3` and input
SHA-256 `a3c8eb105e9af08a4bb13315141f289af83f1ebfc9059ca6c19070a6f6976d7a`.
Doctor evidence retains the fixture, Rust JSON, and three output digests; the
subsequent one-ply CUDA collection remains a separate end-to-end search check.

## Authoritative data and materialized cache

Raw self-play shards are immutable zstd-compressed MessagePack. Their
`PositionRecordV1` values carry absolute bitboards and rule state, sparse visit
targets, the required selected move plus prior-move provenance, absolute
`white_win`/`draw`/`black_win` outcome, termination, and stable game/ply
identity. Rust replays each game trajectory from the initial position and checks
recorded states, legal selected/policy moves, repetition, transitions, and final
result before materialization. Each shard and collection also bind the exact u64
collection seed, positive u32 simulation count, and v1 ply cap in `1..=512`.
Every game seed is the frozen wrapping SplitMix64 derivation of collection seed
and game ID, and every position's visit sum equals the shard simulation count.
Rust then converts the absolute outcome into a W/D/L row from each position's
side-to-move perspective. Plane tensors are deliberately not authoritative: an
encoder correction rematerializes raw data rather than invalidating self-play.

`CollectionManifestV1` (`collection-manifest-v1`) lists sorted, non-overlapping,
contiguous shard game ranges. Each descriptor binds relative path, SHA-256, byte
length, positions, games, and first/last game ID. Its totals must agree with its
descriptors before a cycle can advance. The orchestrator independently rejects a
collection whose cycle seed, simulations, or ply cap differ from the frozen
request before admitting it for materialization.

Rust emits `TensorCacheManifestV1` (`tensor-cache-manifest-v1`) containing:

| Tensor | Type | Shape |
|---|---|---|
| `inputs` | `f32-le` | `[N,22,8,8]` |
| `policy_offsets` | `u64-le` | `[N+1]` |
| `policy_indices` | `u16-le` | `[NNZ]` |
| `policy_values` | `f32-le` | `[NNZ]` |
| `wdl` | `f32-le` | `[N,3]` |
| `game_ids` | `u64-le` | `[N]` |

The six descriptors are top-level fields, each with relative `path`, `dtype`,
`shape`, `bytes`, and `sha256`. The top level binds encoder/action schema, source collection SHA-256,
`record_count`, `policy_size=4672`, and `input_shape=[22,8,8]`. Policy and WDL
rows are normalized distributions. Python checks all hashes and numeric
invariants once, then uses read-only mmap. Sparse policy avoids an approximately
18.7 GB dense policy cache at one million positions.

## Training and transactional cycles

Model `M_k` exclusively creates cycle `k`'s 1,024 games. Only after every raw
shard validates does Rust materialize tensors. Training draws the newest whole
caches covering at least one million positions and performs
`ceil(new_positions * 2 / 512)` successful updates. A stable game-ID hash assigns
the 5% validation split; no game's positions cross train and validation.

AdamW uses weight decay `1e-4`, global gradient clipping 1.0, policy CE plus WDL
CE, and AMP on CUDA. The learning-rate function is frozen before the run: 2%
warmup to `1e-3`, cosine to `1e-4` at the frozen horizon, then `1e-4` forever.
Only successful non-overflow updates advance its counter.

Every 250 successful steps and phase boundary serializes model/buffers,
optimizer, AMP scaler, counters, deterministic sampler epoch/cursor, Python,
NumPy, Torch CPU, every CUDA RNG state, and cumulative cycle loss sums/count.
Consequently a recovered cycle reports the same whole-cycle training metrics,
not merely its post-recovery tail. Deterministic CUDA runs require and pin
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; a conflicting inherited value fails before
training. The file is written to a sibling `.partial`, fsynced, reloaded, hashed,
atomically renamed, and only then exposed through `RECOVERY`.

`HEAD` names the latest fully committed phase/cycle state. `RECOVERY` can name a
newer mid-training state derived from that exact HEAD. Both are atomic pointers
to immutable content-addressed JSON. A cycle promotes the checkpoint and ONNX in
one HEAD transaction; self-play then uses the newest verified model without an
arena gate. Deployment selection remains separate.

Every fifth-cycle progress match and the final 12-checkpoint round-robin use the
dual-model arena with identical fixed-simulation search settings for both sides.
Its durable paired-opening JSONL header binds both full model hashes, the arena
binary, openings, and search limits. Python verifies those records and derives
W/D/L before fitting a regularized, mean-zero Bradley–Terry ladder. These
exploratory relative ratings never gate self-play or deployment. The separate
frozen Random/Depth-1/Depth-2/Depth-3 rungs use 100/100/100/200 opening pairs;
a Depth-3 verdict may accompany a production model but is not required by it.

## Frozen search and self-play

Every node value is from the player-to-move perspective at that node; backup
negates it once per edge. Checkmate is -1 for the mated side to move, and
stalemate, insufficient material, threefold repetition, the 50-move rule, and
the 512-ply self-play cap are draws. The cap therefore writes a draw W/D/L
target, not an adjudicated estimate.

PUCT is deliberately fixed at `cpuct=1.5`, rather than using AlphaZero's
log-schedule. An unvisited root edge uses the root network value, so Dirichlet
noise is not suppressed by pessimistic root FPU. At non-root nodes FPU is
`clamp(node_network_value - 0.25, -1, 1)`. Pending paths add one virtual visit
and a virtual value of -1; collision cancellation removes both, while a real
backup first removes them and then records the evaluated result. Tests require
an exact simulation count even when a batch mixes terminal and neural leaves.

Self-play uses 128 simulations, root priors mixed as `0.75P + 0.25D` with
`Dirichlet(0.3)`, samples proportional to visits through ply 30, then chooses
maximum visits. Serving and evaluation have no noise or temperature and use a
stable action-index tie-break. No resignation is enabled in v1; predicted
resignation savings and false-positive diagnostics are not implemented and are
explicitly deferred to a later experiment. The single 3070 is fed by one
centralized evaluator batching leaves across up to 128 concurrent games.
Training and collection alternate rather than compete for the GPU.

## Frozen opponents and the optional arena gate

`MinimaxDepth1V1`, `MinimaxDepth2V1`, and `MinimaxDepth3V1` share the repository's
integer evaluation, deterministic total move order, alpha-beta search, and
quiescence over captures, promotions, and check evasions; only fixed depth
changes. The frozen chosen-move digest is `16258623573026552286`. Depth 3's
current historical-position calibration is 1642 Chess.com 30+0 move-quality
Elo, with a whole-player 95% interval from at or below 1400 to 1780. That
calibration is contextual evidence, not the release criterion.

The committed 200-opening suite contains seeded random eight-ply prefixes with
at least eight legal continuations and absolute Depth-3 score at most 100 cp.
Every opening is played twice with colors reversed, and uncertainty resamples
whole pairs. Random and Depth 1/2 are 100-pair progress rungs. A complete Depth-3
gate runs all 200 pairs at 10,000 simulations, 9 seconds, batch 8, no noise, and
a 20,000-sample seed-1 pair bootstrap whose lower score bound must be strictly
above 50%. Both the resumable pair header and immutable gate verdict bind
`cpuct=1.5` and non-root FPU reduction `0.25` as exact integer millionths, so a
search-default change cannot reuse an old result. A miss is published, not
relabeled.

The gate is a model-comparison instrument, not a deployment precondition.
Deployment validates the model against its manifest; a verdict is optional, and
when one is supplied the server checks it against the model and frozen baseline
identities and serves at the frozen budget it certifies. Cross-model matches are
run as their own work rather than as a release step.

## Experiment identity

The semantic hash includes schemas, architecture, training, self-play/MCTS,
export, and command configuration. Human name, notes, output path, active-time
budget, and heartbeat cadence do not affect learning and are excluded. `resume`
accepts no overrides and rechecks the source-content identity, both lockfiles,
and the recorded Python/PyTorch/ONNX/CUDA/GPU/driver fingerprint before it can
collect self-play. Published v1 runs require a committed clean worktree;
explicitly disposable pilots may instead freeze a dirty commit plus the digest
of every tracked and untracked nonignored file, and refuse resume after any
content drift. `extend` appends budget without changing identity. Any
learning-affecting or determinism-runtime change—including a relevant bug fix—
requires a parent-linked `fork`; a weights-only fork is never represented as an
exact resume.

The 72-hour marker is orchestrator-active monotonic time. Heartbeats bound lost
time accounting after a crash to one heartbeat interval. Collection, training,
materialization, and export count; an intentionally stopped process does not.
At the first fully promoted-model boundary at or beyond the original budget, the
ledger seals a content-addressed `alphamini.budget-milestone.v1` snapshot. It
records the original threshold, accounted time and overshoot, cycle/step/game
counters, and checkpoint/model descriptors. The snapshot references only its
already-existing parent state; the newer state references its hash, avoiding a
self-referential hash cycle. A no-op `resume` at an exhausted safe boundary is
rejected before `ACTIVE_SESSION` is created. `extend` is accepted only after the
current budget is exhausted and preserves the first milestone hash permanently.
The v1 every-fifth-cycle arena cadence is a manual post-cycle operation, so its
separately logged wall time is reported as additional evaluation compute rather
than silently included in or attributed to the active-training budget.

## Explicit v1 deferrals

Gumbel search, teacher/minimax warm-start, resignation, history planes, FP16
serving, graph search, tablebases, UCI, and tactical hybrid search are separate
experiments. This deferral includes resignation savings/false-positive
instrumentation; v1 emits no such diagnostic. Exact-input inference caching and
cross-move tree reuse do not ship in v1. An instrumented cache prototype is
worth a later fork only if an A/B benchmark shows at least 5% more completed
simulations per second; halfmove and repetition near misses must be reported
rather than hidden.
