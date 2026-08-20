# AlphaMini production-shaped benchmark

This benchmark answers one narrow question: how many complete self-play and
training updates the RTX 3070 can sustain with v1's actual shapes. It is
disposable and is never a parent of the published run. It does not measure Elo.

The final benchmark config uses the 22-plane, 64-channel, six-block network,
128 MCTS simulations, inference batches of 256, 512-ply games, training batches
of 512, and a 2.0 sample ratio. Each measured cycle uses v1's full 1,024 games;
the collector services them through a bounded rolling pool of 512 workers, then
writes the unchanged 128-game shards. The disposable benchmark's internal
horizon did not set the production schedule. The measured decision is now
frozen in `v1.toml`: inference batch 256, 180,000 successful updates, and
`horizon_confirmed = true`.

## Run two measured cycles

Do not edit source, config, or lockfiles between initialization and the final
report. Use a new directory for every attempt:

```bash
export ALPHAMINI_BENCH_DIR="$PWD/runs/alphamini-production-benchmark-001"

uv run --project alphamini-train alphamini-train doctor \
  --config configs/alphamini/production-benchmark.toml \
  --production

uv run --project alphamini-train alphamini-train start \
  --config configs/alphamini/production-benchmark.toml \
  --run-dir "$ALPHAMINI_BENCH_DIR" \
  --initialize-only
```

Start an external hardware trace after the run directory exists:

```bash
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm \
  --format=csv -l 2 > "$ALPHAMINI_BENCH_DIR/gpu.csv" &
export ALPHAMINI_GPU_TRACE_PID=$!
```

Run exactly two cycles. Always retain `--one-cycle`; omitting it can consume the
entire six-hour guardrail.

```bash
uv run --project alphamini-train alphamini-train resume \
  --run-dir "$ALPHAMINI_BENCH_DIR" --one-cycle

uv run --project alphamini-train alphamini-train resume \
  --run-dir "$ALPHAMINI_BENCH_DIR" --one-cycle

kill "$ALPHAMINI_GPU_TRACE_PID"
```

Then produce the machine-readable acceptance report:

```bash
uv run --project alphamini-train alphamini-train benchmark-report \
  --run-dir "$ALPHAMINI_BENCH_DIR" \
  > "$ALPHAMINI_BENCH_DIR/benchmark-report.json"
```

The command deep-verifies the immutable ledger and exits nonzero if an automated
gate fails. It never edits `v1.toml`.

## Automated acceptance gates

The report requires:

- a disposable CUDA run with exactly the production model, search, and training
  shapes;
- at least two uninterrupted completed 1,024-game cycles;
- exactly 512 bounded rolling game workers (`2 * batch_capacity`) per cycle;
- successful collection and materialization invocations;
- `completed_simulations == positions * 128` in every cycle;
- a realized batch of 256, mean batch fill of at least 65%, and at least 30,000
  simulations/second per cycle;
- nonempty game-grouped validation and passing ONNX parity;
- complete training-stage telemetry, with unsuccessful attempts exactly
  accounted for by AMP overflow and an overflow fraction no greater than 5%;
- no more than 10% spread between the two cycle-level simulation rates.

The aggregate `naive_72h_successful_update_projection` uses measured collection,
materialization, training, checkpoint, validation, and export time. It is a
starting point, not an automatic schedule decision.

## Manual gates used for the horizon freeze

`horizon_freeze_ready` is deliberately always false. Review `gpu.csv` and
require peak memory no greater than 90% of physical VRAM and no sustained
thermal throttling. Median GPU utilization of at least 80% is an
optimization/investigation target, not an acceptance gate. Utilization is not
an outcome invariant, and WSL `nvidia-smi` is an aggregate host signal that can
include a nonzero background and changing memory owners. Record and explain a
miss; do not claim saturation. The hard throughput signals are at least 65%
mean batch fill with one bounded 512-game terminal drain, at least 30,000
simulations/second, exact simulation accounting, and repeat throughput within
10%. Fill is interpreted together with absolute throughput: a larger batch can
issue fewer, fuller evaluator calls and deliver more simulations/second even
when its percentage fill is lower.

Separately rehearse interruption during collection and after a durable training
checkpoint. Recovery must preserve the requested model/config/seed/game-ID
range, quarantine unsealed output, pass deep verification, and resume from the
exact optimizer/scaler/RNG/sampler checkpoint. Dynamic collection batching does
not promise byte-identical regenerated games. Also complete CPU serving tests at
batches 1, 4, and 8 under the frozen nine-second budget.

The production setting is batch 8; batches 1 and 4 are retained as diagnostic
comparisons. A batch-1 deadline miss does not override a passing frozen batch-8
serving result. The generated overall status follows the production batch-8
decision, while every diagnostic setting retains and publishes its own status.

The production gates passed, the measured whole-cycle update rate was converted
into the conservative 180,000-step horizon, and `configs/alphamini/v1.toml` now
freezes it with `horizon_confirmed = true`.

Do not claim sustained production throughput, training improvement, Elo, or
playing strength from the earlier 1x16 pilot or from this benchmark alone.

## Recorded result: batch-256 benchmark 003

Benchmark 003 is the final accepted readiness and throughput measurement. It
remains disposable and must not seed Run 1. Batch-128 benchmark 002 remains a
superseded tuning baseline: it first proved the rolling scheduler, but its
120,000-step recommendation is no longer the production decision.

| Identity | Value |
|---|---|
| Run ID | `a69d4465-789c-4286-adcd-a82b322b3027` |
| Frozen worktree SHA-256 | `570b54ad674351ff30fab2eb7fb6512bc5c60b7324d13011019e80b1117c332b` |
| Resolved config SHA-256 | `e4e4f826f9481ccfe19268b582f12ab443db5ba3ba105683038899bbbe0ea98d` |
| Semantic config SHA-256 | `c86e83e462f2d99fd916d2ba5c4089c109bd8c94849622ee99be147fb8efce91` |
| Final HEAD | `33b7140b2d33e13ceeab59ef35616479c0b5119a8b7ff459b27a54e7c5ecd26e` |
| `benchmark-report.json` SHA-256 | `919868f2b6e3d7eea949daa7fbd1d3dbc0d1e257f8efe4a06c3b81102eced332` |
| `gpu.csv` SHA-256 | `f9ebf08bbc775583f645df671892d488a4ed8af126c7925773704222064de179` |

The two cycles completed 1,024 games each through exactly 512 workers. They
produced 164,920 and 186,200 positions, 21,109,760 and 23,833,600 exact
simulations, mean batch fills of 69.444% and 75.892%, and 32,174.43 and
34,798.85 simulations/s. Their full measured cycle times were 771.216 and
797.541 seconds, with 645 and 728 successful updates. Simulation-rate spread
was 7.837%. Both cycles had nonempty validation, finite losses, passing parity,
and successful immutable publication; cycle 2's single AMP overflow was exactly
accounted for and only 0.137% of attempts. Deep verification checked 16 states
and eight referenced artifacts.

The aggregate was 1,373 successful updates in 1,568.757 measured seconds:
3,150.775 updates/hour and a naive 226,855-update projection over 72 active
hours. The slower cycle alone projects 216,780 updates; a 15% reserve gives
184,263. The final decision is therefore **180,000 successful steps**, rounded
down below that slower-cycle haircut. It is now frozen and confirmed in
`v1.toml`, together with inference batch 256.

The whole-run WSL trace had about 45% median aggregate GPU utilization, below
the 80% optimization target. Absolute maxima were 4,791/8,192 MiB, 60 C, and
73% utilization, so the memory and thermal safety gates passed. This trace is a
sampled, aggregate host signal rather than a process-attributable duty-cycle
measurement. More importantly, cycle-1 simulation throughput improved 65%
over batch-128 benchmark 002 (32,174 versus 19,446 simulations/s), while
aggregate updates/hour improved about 60%. The lower `nvidia-smi` percentage is
therefore not a release gate; it remains an explicit profiling target. CUDA
sessions had fail-closed CPU fallback enabled and the run proved they loaded,
collected, trained, exported, and passed parity without fallback.

At 180,000 steps, the 2.0 sample ratio implies about 46.08 million generated
positions and roughly 263 cycles at the observed update yield. Scaling the
upper observed raw/cache bytes per position and retained checkpoint/model
artifacts gives a conservative local high-water of about 18 GB with verified
off-volume backup and safe-boundary replay-cache GC, or about 278 GB if every
cache is retained. Reserve at least 300 GB for unattended no-GC operation, or
at least 30 GB only when the backup-and-GC procedure is operational.

Two preliminary batch-256 directories are intentionally not evidence. Runs
`2477a2e4-951d-4eff-8373-5dfe7d445c46` and
`5d0237bb-5240-4c40-bad4-1a7eab1cfb70` stopped at the initialized state and
published no cycle, model, or training artifact. The first also demonstrated
that a documentation edit after initialization invalidates the frozen dirty
worktree identity. Both were abandoned rather than resumed; fresh run 003 is
the only accepted batch-256 measurement.

Recovery rehearsal 002 passed exact state/model equivalence for both controlled
paths. Collection evidence SHA-256 is
`f4060c0a1f7f49b764311771b7211b6ab9ff1fe8ba97b586db2a7f1f0a382c3b`;
training evidence SHA-256 is
`97df4703d068574d73a28447de01e6b240b3c559e109e81bdf46c339fa56c201`.

CPU production benchmark 002 is bound to benchmark 003's final M2 model
`6bac11c0b2742884d11aaeb4ca4e5f983641541cd9f3fbcb2e6f32eb87f69f54`
and has summary SHA-256
`ad938d374803e15507937ffef2d38dda7bcb91a0fe3a2465d834614b03eba701`.
Frozen production batch 8 passed with exactly 40,000 simulations, zero
deadlines, 4.051 seconds/move, and 2,468.25 simulations/s. Diagnostic batch 4
also passed at 4.870 seconds/move. Diagnostic batch 1 failed at 9.001
seconds/move after 25,567 simulations. The overall summary correctly passed
because batch 8 is the production decision; batch 1 remains an explicit
diagnostic failure.
