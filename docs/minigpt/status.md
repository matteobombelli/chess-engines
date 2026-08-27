# MiniGPT status and continuation handoff

Read this before starting or resuming work. The frozen interfaces are in
[`design.md`](design.md), the operational procedure is in
[`training-runbook.md`](training-runbook.md), and the dated engineering record
is in [`implementation-log.md`](implementation-log.md).

## Current state

| Item | State |
|---|---|
| Rust tokenizer, ingest, shards, decode, serving | Implemented and tested |
| Python training, export, parity fixtures, run ledger | Implemented and tested |
| Frozen schemas (`shards`/`config`/`manifest`/`parity-fixture` v1) | Frozen |
| Corpus ingest (2026-06 + 07 dumps) | Complete; 11,229,433 games, 880.7M tokens, 0 SAN errors |
| `chess-core` replay optimization | Complete; 224× on 80-ply, bit-identical |
| Disposable pilot (S, 27.7M params) | Passed 2026-08-26; val 2.414, top-1 35.4% |
| Production run `minigpt-v1` (M, 40.3M params) | **In progress**; 77,000-step horizon |
| Evaluation against the fixed ladder | Not started |
| Chess.com move-quality calibration | Not started |
| Deployment (service, proxy route, frontend) | Not started |
| Published run result document | Not written |

## The production run

| Identity | Value |
|---|---|
| Run ID | `e79c7d6f-93d4-48bf-85d5-3f7a9fa47a35` |
| Source commit | `aa16c42e34e199b87ebd98ad77b538fe030d3a73` |
| Worktree content | `fd35422c20bdecbd45214a0aa5c709513374438884ca3880d250d3c90d864bc7` |
| Semantic configuration | `80d1e8dae13e3886fe756006f62ed6f8980f937b980a516d4f0329bea9fd26e7` |
| Shard manifest | `df0c2a5dd4e571f492f87039befdaf3705bce8d146bdf21823df49c44fe8af0a` |
| Parent | none |
| Config | `configs/minigpt/v1.toml` |
| Run directory | `runs/minigpt-v1` |

Architecture: 12 layers, `d_model` 512, 8 heads, `d_ff` 2048, `ctx` 256,
dropout 0.1, vocab 4,736, tied embedding, 40.3M parameters. Horizon 77,000
steps at 256 games per optimizer step, about 1.76 epochs. Active-time budget
72 hours; measured throughput puts the horizon well inside it.

The run has recovered from one incident, a dirty worktree stalling it at the
step-2,000 segment boundary. See the implementation log.

## What is implemented

- **Tokenizer.** The move vocabulary is exactly `policy-v1`'s 4,672 actions
  plus `BOS` 4,672 and `PAD` 4,673, padded to 4,736. Compile-time assertions
  bind it to `alphamini::policy::POLICY_SIZE`.
- **Ingest.** Streaming zstd, tag-only filters applied before any replay,
  parallel SAN replay and tokenization, atomic shard publication, a manifest
  binding every source digest, filter value, count, and shard digest.
- **Training.** Length-bucketed batching over memory-mapped shards, AdamW with
  a frozen warmup/cosine schedule, AMP, deterministic mode, evaluation every
  500 steps, best-validation tracking, early stopping on patience.
- **Ledger.** Content-addressed immutable state, atomic pointers, advisory
  locking, heartbeats and active-time accounting, per-segment worktree identity
  enforcement, `recover`/`resume`/`extend`/`fork`/`verify`/`reproduce`/`gc`/
  `report`.
- **Export.** FP32 ONNX opset 17, PyTorch/ORT parity at four sequence lengths
  before publication, closed-field manifest, separate provenance record, atomic
  `current` republication.
- **Decode and serving.** Legality-masked temperature sampling over the legal
  action set only, `ctx`-256 truncation keeping `BOS` plus the newest 255
  tokens, and the same HTTP move endpoint the other engines expose.

## What is not implemented

- No UCI, no KV cache, no Elo conditioning, no search, no opening book, no
  tablebase. See the deferrals section of [`design.md`](design.md).
- `alphamini::policy` is not extracted into a shared crate. MiniGPT depends on
  it directly.
- No arena rungs, no calibration corpus, and no deployment wiring for MiniGPT
  yet.

## Continuation steps, in order

1. **Let the run reach its horizon.** Do not edit the training worktree; use a
   separate `git worktree`. Monitor with `report` and `metrics.jsonl`.
2. **Export and verify.** `export` publishes the best-validation checkpoint;
   follow with `verify --deep` and `reproduce`.
3. **Regenerate the training figures.** Re-run
   `scripts/minigpt-training-figures` against the completed
   `metrics.jsonl` and overwrite `frontend/src/minigpt_training.rs`. The
   current file was generated mid-run and shows a partial curve.
4. **Evaluate.** Play MiniGPT against the frozen Random and Minimax depth
   rungs through the arena, at the frozen decode temperature, with paired
   openings and colors reversed.
5. **Calibrate.** Run the Chess.com 30+0 move-quality estimate the way
   AlphaMini's was run, so the two engines' numbers are comparable.
6. **Deploy.** Add the service unit, the proxy route, and the frontend model
   entry; publish `model.onnx` and `manifest.json` under
   `artifacts/minigpt/current` outside Git.
7. **Write the run result.** `docs/minigpt/results/run-001.md`, with lineage,
   incidents, curves, and the measured evaluation.
