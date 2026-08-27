# MiniGPT implementation log

This file records engineering work, decisions, measured evidence, and
incidents. It is not a benchmark report. Add UTC dates, source commits,
artifact/config hashes, and exact commands whenever measurements begin.

Current operational status and continuation steps are in
[`status.md`](status.md). The frozen interfaces are in
[`design.md`](design.md).

## Development process

MiniGPT reuses AlphaMini's reproducibility discipline — frozen schemas,
content-addressed state, atomic pointers, identity-checked resume — and applies
it to a supervised sequence model rather than a self-play loop:

1. Freeze the token vocabulary against the existing `policy-v1` action space so
   no second move mapping exists to drift.
2. Freeze the shard, config, manifest, and parity-fixture schemas, and write
   the byte-layout invariants as assertions rather than prose.
3. Build ingest and training against those schemas independently, then require
   Rust-produced shards to load and train in Python.
4. Run the disposable pilot end to end — ingest, train, export, parity, serve —
   at a smaller architecture, and measure throughput before freezing the
   production horizon.
5. Start the published run only from a committed clean worktree, and keep an
   append-only incident record during it.

## 2026-08-26 — corpus ingest

Ingested the Lichess 2026-06 and 2026-07 standard rated dumps with the frozen
v1 filters: both Elos at least 2000, Blitz/Rapid/Classical only, `Normal` or
`Time forfeit` termination, standard start position, no variations, 10 to 300
plies.

| Measure | Value |
|---|---:|
| Games seen | 175,771,749 |
| Games accepted | 11,229,433 |
| Acceptance rate | 6.39% |
| Train tokens | 876,242,928 |
| Validation tokens | 4,460,910 |
| Mean tokens per game | ~78 |
| SAN replay errors | 0 |

Zero SAN errors across 11.2M accepted games is the result worth recording: the
whole corpus replays through `chess-core` with every move legal at the ply it
appears, so the tokenizer is not silently dropping or misreading games. The
`san_error_samples` field exists for the case where that stops being true.

The 6.39% acceptance rate is dominated by the Elo filter — most Lichess games
have at least one player below 2000 — followed by the Bullet exclusion. Both
are deliberate: the training target is what a strong player does with time to
think, and a lopsided game is not a strong-play target on both sides.

The validation split is 0.5% by game, assigned by a stable hash of game
identity, so no game contributes to both splits and re-ingesting the same dumps
reproduces the same split.

## 2026-08-26 — chess-core replay optimization

Ingest is replay-bound: every accepted game is replayed move by move to convert
SAN to actions, and at 175M games seen the replay path dominates wall time.
Profiling an 80-ply replay showed the cost concentrated in repeated full
recomputation rather than in move generation itself.

The optimization is **224× on an 80-ply replay** and **bit-identical**: the
frozen chosen-move digest over the fixed regression corpus is unchanged at
`0x9c19_e902_7dc8_fc14`. That digest is the acceptance criterion — a replay
speedup that changed any chosen move would have changed AlphaMini's frozen
opponents and invalidated the existing calibration, so bit-identity was
required, not merely hoped for.

## 2026-08-26 — pilot `minigpt-pilot`

Disposable end-to-end pilot on the S variant (8 layers, `d_model` 512, 8 heads,
`d_ff` 2048, `ctx` 256, 27.7M parameters), 2,500 steps at the same optimizer
settings as production.

| Measure | Value |
|---|---:|
| Parameters | 27.7M |
| Steps | 2,500 |
| Wall time | ~24 min |
| Validation loss | 2.414 |
| Validation top-1 | 35.4% |
| Validation perplexity | 11.2 |
| Throughput | 71.9k tokens/s |
| Peak VRAM | 4.01 GB |
| Export parity, max abs | 1.4e-5 |
| CPU serving | ~9 ms/move |

What the pilot established:

- **The pipeline is correct end to end.** Rust shards load in Python, training
  converges, export traces, parity passes, and the Rust engine serves legal
  moves from the exported graph.
- **VRAM has headroom.** 4.01 GB peak on an 8 GiB card at the S variant left
  room to go to 12 layers for production without changing `micro_batch` or
  `grad_accum`.
- **Export parity is comfortable inside tolerance.** The measured 1.4e-5
  maximum absolute difference is nearly two orders of magnitude below the
  frozen `parity_atol` of 1e-3. The 1e-3 tolerance was chosen after an earlier
  1e-4 proved too strict for FP32 ONNX Runtime on this graph.
- **Serving is not the bottleneck.** ~9 ms per move on CPU with no KV cache,
  re-running the full prefix every move, is far inside the serving budget. This
  is why the KV cache is deferred rather than built.

Pilot weights were not transferred into the production run.

## 2026-08-26 — production run `minigpt-v1` started

Started the published run at commit `aa16c42`, worktree content
`fd35422c20bdecbd45214a0aa5c709513374438884ca3880d250d3c90d864bc7`, semantic
configuration `80d1e8da...`, shard manifest `df0c2a5d...`.

Frozen production shape: the M variant at 12 layers, `d_model` 512, 8 heads,
`d_ff` 2048, `ctx` 256, dropout 0.1 — **40.3M parameters** with a tied
embedding. `micro_batch` 64 times `grad_accum` 4 is 256 whole games per
optimizer step; games are padded within a length bucket rather than packed into
fixed 256-token rows.

The horizon is **77,000 steps ≈ 1.76 epochs**. The arithmetic is recorded in
`configs/minigpt/v1.toml` and repeated here because the cosine schedule is
defined over exactly that horizon and cannot be changed without a fork:
11,172,722 train games at 256 games per step is about 43,644 steps per epoch at
roughly 20k real tokens per step, so 77,000 steps is about 1.55e9 trained
tokens. AdamW with weight decay 0.1 and gradient clipping 1.0; 2% warmup to
`3e-4`, cosine to `3e-5` at step 77,000.

### Incident: dirty worktree stalled the run at step 2,000

The run stalled at step 2,000 — the end of segment 2 — with an identity error
rather than a training error. The cause was an edit to a tracked file in the
training worktree while the run was in flight. The segment boundary re-derives
`worktree_sha256` over every tracked path and requires it to equal the value
frozen in the run manifest, and separately requires `tracked_dirty` to be
exactly `False`; the edit failed both.

This is the check working as designed, and it is deliberately strict enough to
trip on a change that cannot affect the model. The trade is intentional: a
run's source identity is either byte-exact or it is not evidence.

Recovery was `git stash`, then `recover` to close the session at its last
durable heartbeat, then `resume`:

```bash
git stash push --include-untracked
uv run --project minigpt-train minigpt-train recover --run-dir runs/minigpt-v1
uv run --project minigpt-train minigpt-train verify --run-dir runs/minigpt-v1 --deep
uv run --project minigpt-train minigpt-train resume  --run-dir runs/minigpt-v1
```

No training was lost — step 2,000 had a durable checkpoint — but roughly
**15 minutes of wall time** were. Operational consequence, now in the runbook:
do repository work in a separate `git worktree` while a run is in flight, never
in the training checkout.

Measured throughput at the time of the incident was 2.78 steps/s, which puts
the 77,000-step horizon at well under the 72-hour active budget.

## 2026-08-27 — run complete, calibrated

minigpt-v1 reached its 77,000-step horizon (no early stop): val loss 1.4234,
perplexity 4.15, top-1 53.4%. Export parity 7.6e-6; deep verify clean; CPU
serving ~13 ms/move. Calibrated **1928 Chess.com 30+0** (95% CI 1610–≥1999),
same method/corpus/seed as AlphaMini. Arena frozen rungs skipped by decision;
random smoke 97.0%. Incidents and the VRAM-spill note are recorded in
`results/run-001.md`.
