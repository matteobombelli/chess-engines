# MiniGPT v1 design

## Purpose and ownership

MiniGPT is a compact decoder-only transformer that plays chess by predicting the
next move a strong human would play. It exists to contrast a supervised
sequence model against AlphaMini's search-and-self-play pipeline on the same
board, the same legal-move boundary, and the same single RTX 3070.

Rust owns chess legality, SAN replay, tokenization, corpus ingest, shard
publication, ONNX serving, and decoding. Python owns the transformer, the
optimizer, crash-safe checkpoints, ONNX export, parity fixtures, and the
immutable run ledger. Python never reconstructs a chess position and never
decides which moves are legal.

There is no search. The model produces one distribution over the action space
per position, and a legality mask decides what that distribution is allowed to
mean.

## Frozen token vocabulary

The move vocabulary is exactly AlphaMini's `policy-v1` action space, so a token
id below `BOS` is decodable by `alphamini::policy` with no second mapping to
keep in sync:

| Id range | Meaning |
|---|---|
| `0..=4671` | `policy-v1` actions, `movement_plane * 64 + canonical_origin` |
| `4672` | `BOS`, the start-of-game token |
| `4673` | `PAD`, used only to fill a length bucket |
| `4674..=4735` | Never emitted; padding to a GPU-friendly width |

`vocab_size` is therefore 4,736 rather than the 4,674 ids in use: the embedding
and output matmuls land on a multiple of 64. The unused ids are trained on
nothing and are unreachable at decode time, because the legality mask is
computed over `0..4672` only.

A game is `BOS` followed by one action token per ply, in order, from the
standard start position. There is no result token, no clock token, no Elo
token, and no side-to-move token: side to move is implied by ply parity, and
the position is implied by replaying the whole prefix.

## Frozen shard format `minigpt.shards.v1`

Ingest publishes a pair of files per shard, `<prefix>-NNNN.bin` and
`<prefix>-NNNN.idx`, and one `shards.json` manifest describing them.

`.bin` is the raw token stream: little-endian `u16`, games concatenated in the
order they were read, each game being its `BOS` token followed by one action
token per ply. No padding is stored on disk; padding is a training-time
concern.

`.idx` locates those games, little-endian `u64` throughout: a game count `G`,
then `G + 1` token offsets. Offsets are in **tokens**, not bytes, so a byte
offset into the `.bin` is the offset times two. Game `i` occupies
`offsets[i]..offsets[i + 1]`, `offsets[0]` is always zero, and `offsets[G]` is
the shard's total token count. The index file is therefore exactly
`(G + 2) * 8` bytes and the token file exactly `token_count * 2` bytes. Both
sizes are recomputable from the manifest, so a truncated shard is detectable
without reading it.

`shards.json` (`minigpt.shards.v1`, `deny_unknown_fields`) binds the tokenizer
name, `vocab_size`, `bos_token`, `pad_token`, the exact filter settings, one
`SourceV1` per dump with its whole-file SHA-256 and compressed byte count, the
accept/reject counts, and the train and validation shard lists. Each shard
descriptor carries both paths, both SHA-256 digests, `token_count`, and
`game_count`. Every game read is either accepted or attributed to exactly one
reject reason, so `games_seen == games_accepted + rejected.total()` is an
invariant the manifest asserts rather than a summary it prints.

Shards are published atomically and never overwritten. The validation split is
per game and derived from a stable hash of the game's identity, so no game
contributes to both splits and re-ingesting the same dumps reproduces the same
split.

## Frozen configuration schema `minigpt.config.v1`

Configuration is strict TOML: unknown shapes, missing tables, non-finite
numbers, and out-of-range integers fail before a run is created. The v1 schema
pins several values outright — `model.vocab` must be 4736, `data.tokenizer`
must be `policy-v1`, `data.bos_token` must be 4672, `data.pad_token` must be
4673, `export.dtype` must be `float32`, and `export.opset` must be exactly 17.

Two hashes are derived from every config. `config_hash` is the SHA-256 of the
raw file bytes. `semantic_hash` is the SHA-256 of a canonical JSON projection
with the operational keys removed:

| Table | Excluded keys |
|---|---|
| `run` | `name`, `description`, `active_budget_hours`, `output_dir` |
| `training` | `checkpoint_keep_last`, `checkpoint_milestone_every_steps`, `disk_floor_bytes` |
| `operations` | `heartbeat_seconds` |

Those keys describe how a run is operated, never what it computes. Two configs
differing only there produce the same model and share a semantic hash;
everything else — architecture, horizon, learning-rate schedule, decode
temperature, determinism flags — changes identity and requires a `fork` rather
than a `resume`.

The production `v1.toml` freezes the M variant: `d_model` 512, 12 layers, 8
heads, `d_ff` 2048, `ctx` 256, dropout 0.1, 40.3M parameters with a tied
embedding. `micro_batch` 64 times `grad_accum` 4 is 256 whole games per
optimizer step, and the horizon is 77,000 steps, about 1.76 epochs of the
corpus. AdamW uses weight decay 0.1 and gradient clipping 1.0; the learning
rate warms up over the first 2% of the horizon to `3e-4` and cosines to `3e-5`
at exactly step 77,000. The schedule is defined over the frozen horizon, so
truncating or extending the run changes the schedule and therefore the model.

## Frozen model manifest `minigpt.manifest.v1`

Export writes an FP32 ONNX graph with input `tokens` and output `logits`,
opset 17. Batch is fixed at 1 — the engine evaluates one game at a time — and
only the sequence axis is dynamic, with the causal mask following it.

The manifest is a closed field set; export fails if the set drifts:

| Field | Meaning |
|---|---|
| `schema` | `minigpt.manifest.v1` |
| `tokenizer` | `policy-v1` |
| `onnx_opset` | 17 |
| `input_name` / `output_name` | `tokens` / `logits` |
| `vocab_size`, `context` | Graph dimensions |
| `bos_token`, `pad_token`, `policy_size` | 4672, 4673, 4672 |
| `d_model`, `n_layers`, `n_heads`, `d_ff` | Architecture |
| `decode_temperature` | The serving temperature this model was published with |
| `model_sha256` | SHA-256 of the `.onnx` file |

The manifest is intentionally byte-compatible with the Rust reader, which
denies unknown fields. Training provenance — semantic hash, global step, parent
checkpoint digest, full architecture table, and the measured parity record —
lives in a separate `model-<digest>.training.json`
(`minigpt.training-model-provenance.v1`) so the served manifest stays minimal.

Export verifies PyTorch against ONNX Runtime CPU before publishing, at sequence
lengths `{1, 4, 64, context}`, using the configured `parity_atol` and
`parity_rtol`. A failed comparison raises rather than publishing. Artifacts are
content-addressed as `model-<first 16 hex>.onnx`; the served pair is
republished atomically as `model.onnx` and `manifest.json`.

## Frozen parity fixtures `minigpt.parity-fixture.v1`

The Rust engine's inference parity check is written against a fixture
directory, not against a live PyTorch process. It holds one `parity.json` index
plus one `logits-tNNNN.f32` per case: little-endian `f32`, C order, exactly
`1 * T * vocab_size` values, so each file is `T * vocab_size * 4` bytes.

Each case records the token list, `tokens_sha256` over the ONNX input tensor
exactly as the engine must build it (little-endian `i64`, C order, shape
`[1, T]`), `logits_sha256` over the raw logit file, the logits shape, and the
measured `python_ort_max_abs`. Expected values are PyTorch FP32 CPU outputs;
ONNX Runtime agreement within `atol` is verified at the moment the fixture is
written and recorded per case. Verification re-reads the fixture and checks
every recorded size and digest, so a silently rewritten fixture cannot pass.

## Frozen decode rules

Legality is owned by `chess_core` through `alphamini::policy`. The model only
ranks actions that `legal_action_mask` already proved legal, so no logit —
including the ones on `BOS`, `PAD`, and the unused padding ids — can produce an
illegal move. Illegal actions are dropped from the candidate set rather than
set to `-inf`, which is the same distribution without the `exp(-inf)` edge
cases.

Sampling is a softmax over the legal logits divided by the temperature, shifted
by the maximum before exponentiating so a large logit cannot overflow. The
frozen serving default is temperature 0.5, recorded in the manifest so a served
model carries the temperature it was published with. Temperature zero is
greedy, taking the highest legal logit. A position with no legal moves is a
decode error, not a fallback move.

Context is truncated to the graph's exact `ctx` of 256, keeping `BOS` plus the
most recent 255 move tokens. **The truncation is lossy in a way the model
cannot see:** past ply 255 the retained prefix no longer replays to the current
position from the start, so the model conditions on a suffix whose implied
history is wrong. Castling rights, repetition, and the halfmove clock are not
represented in the token stream at all and are invisible to the model at every
length; the mask still guarantees a legal move, but the model's ranking of it
is uninformed. Games long enough to truncate are a small tail of the corpus
(the ply filter caps training games at 300 plies), and no positional
compensation ships in v1.

## Corpus filters

Ingest streams compressed Lichess PGN dumps and applies tag-only filters before
any replay work:

| Filter | Rule |
|---|---|
| Start position | Standard only; a `FEN`/`SetUp` game is rejected |
| Event | Must name Blitz, Rapid, or Classical, and must not name Bullet |
| Elo | **Both** `WhiteElo` and `BlackElo` present and at least 2000 |
| Termination | `Normal` or `Time forfeit` only |
| Plies | 10 to 300 inclusive, counted after sanitizing |
| Movetext | Variations rejected; comments and NAGs stripped |
| SAN | Every token must replay legally at its ply, or the whole game is rejected |

Both-Elo is deliberate: a 2400 player beating a 1200 player produces moves that
are not a strong-play target on both sides. Time forfeit is kept because the
moves played before the flag are still real moves; abandoned and adjudicated
games are not. The ply floor drops trivial disconnects and the ceiling drops
the long tail that the 256-token context could not represent anyway.

A SAN failure rejects the whole game rather than truncating it, and the first
few failing `Site` tags are retained in the manifest so a run that starts
rejecting everything is diagnosable after the fact.

## Run-ledger identity

The run ledger mirrors AlphaMini's: immutable content-addressed JSON objects,
atomic pointers, and an append-only metrics stream. `minigpt.run-manifest.v1`
binds the run UUID, the resolved config hash and semantic hash, the shard
manifest digest, the git identity, the runtime fingerprint (Python, PyTorch,
ONNX Runtime, CUDA, GPU, driver), and both lockfile digests.

Training advances in segments of `segment_steps`. **At every segment boundary
the ledger re-derives the worktree identity and refuses to continue unless the
tree is byte-identical to the one the run was frozen at.** Concretely, a
non-disposable run requires `tracked_dirty` to be exactly `False` and requires
the recomputed `worktree_sha256` — a domain-separated digest over every tracked
path's mode, name, and content — to equal the value in the run manifest. An
unknown cleanliness state is also refused rather than assumed clean.

This is stricter than it first looks and it is the intended behavior: editing
*any* tracked file in the training worktree, including a documentation file
that cannot affect the model, stalls the run at the next segment boundary. The
run does not lose work — it has a durable checkpoint — but it will not take
another segment until the tree matches. Disposable pilots may instead freeze a
dirty tree's digest, and still refuse to continue after any content drift.

`resume` accepts no overrides and rechecks source identity, both lockfiles, and
the runtime fingerprint. `extend` appends active-time budget without changing
identity, and is accepted only once the current budget is exhausted. Any
learning-affecting change requires a parent-linked `fork`; a weights-only fork
is never represented as an exact resume. Active time is orchestrator-monotonic
and bounded after a crash by one heartbeat interval.

## Explicit v1 deferrals

- **No UCI.** MiniGPT serves the same HTTP move endpoint as the other engines
  and is not addressable by a chess GUI.
- **No KV cache.** Every move re-runs the full prefix through the graph. At
  `ctx` 256 on CPU this measured about 9 ms per move, which is far inside the
  serving budget, so the cache is not worth the correctness surface in v1.
- **No Elo conditioning.** The corpus is filtered to strong play rather than
  conditioned on a rating token, so the model has one playing strength and no
  strength dial.
- **`policy.rs` stays in AlphaMini.** MiniGPT depends on
  `alphamini::policy` for the action space, the legality mask, and
  action-to-move. Extracting it into a shared crate is a refactor with no
  behavioral change, and doing it during a frozen run would alter the worktree
  digest; it is deliberately not part of v1.
- No search of any kind, no opening book, no tablebase, no endgame fallback,
  no resignation, no FP16 or quantized serving, and no multi-game batching.
