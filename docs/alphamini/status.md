# AlphaMini status and continuation handoff

Last updated: 2026-08-25 UTC.

This is the short operational handoff for the current checkout. The detailed
contracts live in [design.md](design.md), and routine commands and recovery
rules live in [training-runbook.md](training-runbook.md).

## Current state

| Item | State |
|---|---|
| Rust chess/MCTS/self-play/materialization | Implemented and tested |
| Python training/export/run ledger | Implemented and tested |
| Arena rungs, paired statistics, and arena gate | Implemented; gate optional |
| Frozen MinimaxDepth3V1 calibration | 1642 move-quality Elo; documented separately |
| Disposable RTX 3070 end-to-end pilot | Passed on 2026-08-20 |
| Production-shaped RTX 3070 benchmark | Passed automated gates on 2026-08-20 |
| Controlled collection/training recovery | Recovery 002 passed exact equivalence |
| Frozen production CPU batch 8 | Passed 40,000 simulations; 4.051 s/move |
| Self-play inference batch | 256, frozen in `v1.toml` |
| Optimizer horizon | 180,000, frozen and confirmed in `v1.toml` |
| Published 72-hour v1 run | Complete; 72.09 h cumulative across run-001/002/003 |
| AlphaMini calibration | 1969 Chess.com 30+0 move-quality Elo, measured 2026-08-25 |
| Deployed model | `128875d3a28138e3...`, provisioned and served |
| Depth-3 arena gate | Started, stopped, and no longer a deploy requirement |

The passing pilot proved that real Rust CUDA self-play can feed the Rust data
validator/materializer, PyTorch can train and checkpoint on CUDA, and the new
model can be exported and loaded again by Rust ONNX Runtime CUDA. The pilot
weights are deliberately non-publishable and never seeded v1.

The published run is finished. Its full lineage, incidents, final model identity,
and calibration are in [results/run-003.md](results/run-003.md).

## What has been implemented

- A native `alphamini` Rust crate owns the 22-plane canonical encoder, the
  4,672-action policy mapping, exact terminal handling, PUCT MCTS, batched
  cross-game CUDA evaluation, self-play, trajectory validation, tensor-cache
  materialization, HTTP serving, and model-manifest validation.
- `chess-core::SearchPosition` provides incremental make/unmake, compact
  repetition identity, shared en-passant semantics, and search-oriented state
  access without cloning public game histories on every simulation.
- The Python `alphamini-train` package owns the CNN, deterministic AdamW/AMP
  training, sparse replay loading, full checkpoints, ONNX export/parity,
  content-addressed run state, recovery, verification, reporting, extension,
  forking, reproduction, and reference-driven GC.
- Raw shards carry enough information for Rust to replay every selected move
  from the initial position and verify state, legality, visit totals, seed,
  termination, and outcome before Python can consume the data.
- The arena has fixed Random and MinimaxDepth1/2/3 rungs, paired openings,
  cluster-bootstrap uncertainty, resumable pair logs, immutable gate verdicts,
  dual-model matches, and a Bradley-Terry checkpoint-ladder consumer. This
  machinery remains available for cross-model comparisons; it is no longer a
  deployment precondition.
- The frozen MinimaxDepth3V1 identity is
  `16258623573026552286`; its history-aware calibration is 1642 Chess.com
  30+0 move-quality Elo with the documented wide interval. This calibration is
  contextual evidence, not AlphaMini's release result.
- Deployment preserves the existing `{san} -> {san, fen}` API and validates the
  model against its manifest before serving. A gate verdict is optional: when one
  is supplied it is still checked against the model and frozen baseline
  identities, and supplying it pins serving to the frozen search budget.

## Pilot history

### `alphamini-pilot-001`: useful failed run

Run ID: `a5b07451-c559-4a89-893a-653871e3d1a2`.

The first attempt exposed two issues before any valuable run began:

1. Rust ONNX Runtime's CUDA provider libraries were not staged beside the
   release binary, and the host had CUDA 12.6 libraries while the pinned ORT
   binary requires CUDA 13 libraries. The stopped collection was recovered and
   quarantined through the normal ledger path.
2. After CUDA self-play collected eight games and Rust materialized 884 records,
   training failed before optimizer step 1. `torch.load(...,
   map_location="cuda")` had moved the saved CPU RNG ByteTensor to CUDA, while
   `torch.set_rng_state` requires a host ByteTensor. The run was recovered at
   `ready_train` and retained as incident evidence rather than modified after
   its source identity had been frozen.

The RNG restore path now validates saved state and converts CPU and CUDA
generator states to contiguous host ByteTensors before restoration. A real RTX
3070 regression reproduces the old mapping and passes with the fix.

### `alphamini-pilot-002`: passing end-to-end run

The authoritative generated report is outside Git at
`runs/alphamini-pilot-002/report.md`.

| Identity | Value |
|---|---|
| Run ID | `b1edc24c-5058-4720-acd3-49728e512d6a` |
| Source commit anchor | `ccdce3d18a3217df77bcd02e8074ae7a8b903345` |
| Frozen dirty-worktree digest | `64a8e5b7e77fdc32f02a0406a02cb4a8a69f9800a5c42e10397ba2adb2a98048` |
| Resolved config SHA-256 | `b3a0619381e5c46188f53756642a57d9b08e490103f71ab20ced83dea918add0` |
| Semantic configuration | `8d57074d64dd4523a661c6da8139320147f535ba9a4542cf3e3c7d999df6db20` |
| Hardware | NVIDIA GeForce RTX 3070, 8 GiB; 16 GiB host RAM |
| Driver/runtime | driver 591.86; PyTorch 2.7.1+cu126; cuDNN 9.5.1; Python ORT 1.22.0; ORT crate rc.13 |
| Pilot architecture | 16 channels, 1 residual block; 75,456 parameters |
| M1 model SHA-256 | `4eb0a925c2e9dacba706d90b5276d59b3f903e629478ee2209554234fdb7fd59` |
| M1 checkpoint SHA-256 | `fa8c85bf1820b667526ba3ed9fa7ad275e5fc5c331b579d1e318ba7a5338c38f` |

Measured self-play results:

| Metric | Result |
|---|---:|
| Games / positions | 8 / 884 |
| Completed simulations | 14,144 |
| Neural evaluations | 15,005 |
| Self-play wall time | 1.892 s |
| Complete simulations/s | 7,476 |
| Neural evaluations/s, whole self-play wall time | 7,931 |
| Neural evaluations/s, inference time only | 9,880 |
| Mean / maximum batch fill | 86.2% / 8 of 8 |
| Positions/s | 467 |
| Compressed shard bytes/position | 56.8 |
| Materialized tensor bytes/position | 5,724 |
| Successful AMP optimizer updates | 28 |
| Total orchestrator-active cycle time | 4.651 s |

The collector's 15,223 games/hour value is only a tiny-pilot extrapolation. It
must not be used to size v1 because v1 uses a 64-channel, six-block model, 128
simulations, batch 128, and up to 512 plies.

Validation completed after the cycle:

- Deep ledger verification checked nine immutable states and six referenced
  artifacts, including decoded raw games and tensor hashes.
- The full Python suite passed 49 tests; the only skipped test in the managed
  shell was the GPU-only RNG case, which passed separately on the RTX 3070.
- Strict fixed-input parity passed across PyTorch, Python ORT CPU, and Rust ORT
  CUDA. Maximum absolute differences were `6.71e-8` for policy and `5.59e-9`
  for WDL, well below the frozen tolerances.
- Rust CUDA inference of the newly trained M1 model passed with the frozen
  encoder-input digest
  `a3c8eb105e9af08a4bb13315141f289af83f1ebfc9059ca6c19070a6f6976d7a`.
- The strict production doctor passed every dependency, CUDA, parity,
  collection, storage, and horizon check. Its sole failure was the intentionally
  dirty worktree, as required before a published run. That terminal result was
  inspected but not archived as a release artifact; the clean v1 doctor record
  is still required.

The reported policy/WDL losses are pipeline evidence only. Eight games produced
no validation games under the frozen game-group split, and 28 updates say
nothing useful about playing strength.

The retry in `pilot-001` and the collection in `pilot-002` produced the same
50,244-byte raw-shard SHA-256
`2af3bf0ac9feb4a95376371e4bffe9f86edc90b53989920f60a62ea354a32c6b`
and byte-identical six tensor payloads. This is useful deterministic collection
evidence, although dynamic CUDA batching is not promised to reproduce bytes in
all future interruption timings.

## CUDA runtime packaging

The pilot required an operational workaround that is not the published-run
packaging contract:

- ORT rc.13 downloaded a CUDA 13 provider into its Cargo cache. Matching
  `libonnxruntime_providers_shared.so` and
  `libonnxruntime_providers_cuda.so` were linked into `target/release`.
- The successful pilot used an ignored `runs/.runtime/cuda13` directory with
  `cuda-toolkit 13.0.0`, `nvidia-cublas 13.0.0.19`,
  `nvidia-cuda-runtime 13.0.48`, and `nvidia-curand 10.4.0.35`. Its generic
  `libcudnn.so` resolved to the cuDNN 9.5.1 library bundled with PyTorch.
- The Rust provider itself identifies as a CUDA 13.2 build. The pilot proved
  this particular host combination works, but manual cache paths and mixed
  runtime provenance are not acceptable for v1.

The implemented v1 stage is an isolated CUDA 13.2 runtime containing these
exact hash-locked pins:

```text
nvidia-cuda-runtime==13.2.86
nvidia-cublas==13.2.2.2
nvidia-cuda-nvrtc==13.2.86
nvidia-cufft==12.2.0.57
nvidia-curand==10.4.2.66
nvidia-cudnn-cu13==9.23.2.1
nvidia-nvjitlink==13.2.86
```

`scripts/alphamini-cuda-runtime setup` installs them beneath ignored
`target/alphamini-cuda-runtime/13.2`, never over the PyTorch cu126 environment.
It creates a machine-readable manifest with wheel/package/runtime/provider/ORT
identities and copies regular provider files beside the release binary.
`verify` rehashes the closure, rejects old Cargo-cache symlinks, and fails on an
unresolved or externally satisfied CUDA provider dependency. Configured CUDA
collectors use `scripts/alphamini-cuda-runtime exec -- ...`, so the stage's
loader path is scoped to Rust inference. Run/doctor provenance now records the
manifest SHA, reviewed wheel hashes, runtime-library-set hash, and exact Rust
ORT provider hashes.

The automation and its corruption/symlink unit tests are complete. On
2026-08-20, `setup` installed the new stage from the reviewed hashes and
`verify` passed for all 21 staged runtime files plus both regular ORT provider
copies. The resulting manifest SHA-256 is
`c2684ff2a56be71d14d232baa4df4580c6892fab01e82bde39795fe9c3793544` and
its runtime-library-set SHA-256 is
`a1d60d34cfa7228d9dc19f23f191fdac59c48686891bb28dcefcc767f5bbe39a`.
Every staged CUDA/cuDNN dependency resolved inside the stage; only the driver
and standard system libraries resolved outside it. The staged runtime then
executed the trained pilot M1 through Rust CUDA successfully. A strict doctor
on the production-shaped 64x6 graph also passed PyTorch CUDA, fixed-input
PyTorch/Python-ORT/Rust-ORT CUDA parity, and a real Rust CUDA self-play smoke.
Its policy/WDL maximum absolute differences against PyTorch were `4.62e-5` and
`6.85e-7`. The doctor's only failure was the deliberately dirty worktree. A
zero-failure report from the eventual clean commit is still required and must
be archived before Run 1.

## Production-shaped benchmark 001: useful failed gate

Run ID: `36affa20-a10e-43c0-87c4-6ac0eaeb27ed`.

This disposable run used the real 22-plane 64x6 network, 128 simulations,
batch 128, training batch 512, and sample ratio 2.0. It completed two full
collect/materialize/CUDA-AMP-train/export cycles and deep verification checked
13 ledger states and eight referenced artifacts. The run is immutable evidence
and must not be resumed after the scheduler change.

| Metric | Cycle 1 | Cycle 2 |
|---|---:|---:|
| Games / positions | 128 / 21,134 | 128 / 18,097 |
| Exact completed simulations | 2,705,152 | 2,316,416 |
| Self-play wall time | 240.26 s | 247.84 s |
| Simulations/s | 11,259 | 9,346 |
| Neural evaluations/s during inference | 12,425 | 10,344 |
| Mean / maximum batch fill | 32.2% / 128 | 28.1% / 128 |
| Successful updates | 83 | 71 |
| Training updates/s | 19.91 | 18.12 |
| Validation batches | 2 | 5 |
| Raw bytes/position | 75.2 | 85.9 |
| Tensor-cache bytes/position | 5,809 | 5,825 |

Both cycles had zero AMP overflows, finite train/validation losses, passing
ONNX parity, exact `positions * 128` visit totals, and successful immutable
publication. The two-cycle directory occupies 253,149,334 bytes.

The external 576-sample GPU trace measured 55% median utilization, 59% p90,
4,500 MiB peak allocation (54.9% of 8,192 MiB), 58 C maximum, 83.5 W median,
and 102.5 W peak. There was no memory or thermal constraint.

The automated report correctly failed. With exactly one batch's worth of games
launched at once, fill was initially 86--99%, then collapsed while the final
long games drained. Those tails dominated wall time. The cycle simulation-rate
spread was also 18.6%, above the predeclared 10% limit. The aggregate 1,047
updates/hour is therefore diagnostic only and must not set the 72-hour
horizon. The next code change is a bounded rolling game pool that replenishes
completed workers; the acceptance metric will not be weakened.

## Production-shaped benchmark 002: superseded tuning baseline

Run ID: `cd60ce38-1317-4835-bb9e-643f6c426cab`.

Benchmark 002 reran the exact v1 model, search, 1,024-game cycle, and training
shapes with batch 128 and a bounded rolling pool of 256 workers. It completed
two uninterrupted cycles and first proved the rolling scheduler. It is now a
superseded tuning baseline, not the measurement used to freeze the horizon.

| Metric | Cycle 1 | Cycle 2 |
|---|---:|---:|
| Games / positions | 1,024 / 164,920 | 1,024 / 180,762 |
| Exact completed simulations | 21,109,760 | 23,137,536 |
| Self-play wall time | 1,085.54 s | 1,137.85 s |
| Simulations/s | 19,446 | 20,334 |
| Mean / maximum batch fill | 81.76% / 128 | 82.65% / 128 |
| Successful updates | 645 | 707 |
| Whole-cycle updates/hour | 1,923 | 2,022 |
| Validation batches | 15 | 36 |
| Raw bytes/position | 75.67 | 64.98 |
| Tensor-cache bytes/position | 5,811 | 5,750 |

The aggregate was 1,352 successful updates in 2,466.324 measured seconds, or
1,973 updates/hour. The two-cycle simulation-rate spread was 4.46%. Its former
120,000-step recommendation has been superseded by batch-256 benchmark 003.

Benchmark identities:

| Identity | SHA-256 / value |
|---|---|
| Frozen worktree | `ff5079305abb08740febe7fabb96c9ce067b00eeb1c737345613a9384c7ce987` |
| Resolved config | `b0ec014d5414617b7529f924c51ba2edb03b82a94a67a9d884a2689936d3a43d` |
| Semantic config | `fdae18872cd6c7b5b528f5d216f48d6831104e5267e7d4fd67519592ae63dcfe` |
| Final ledger HEAD | `c860bf8b57cb174969a98410b4431003b4c658ed6a51184af7f000a160cace1d` |
| Benchmark report | `8a019679d98d82d771b2e58ebd4acd65c1acfaf6a62191b05b69d9674dbd2106` |
| GPU trace | `bfe75d1a11edc8cf3455388767f780105bec6ab064fd21023036fa0c02a4654a` |

The 2,489-sample WSL `nvidia-smi` trace measured 58% median utilization in
both saturated self-play phases, with p90 values of 61--62%. Absolute maxima
were 4,812 of 8,192 MiB (58.7%), 62 C, and 112.6 W; self-play clocks remained
at 1,875 MHz, so memory pressure and thermal throttling were absent. The trace
also showed 36--37% utilization outside collection and changing multi-process
memory baselines, so it is aggregate host evidence rather than an attributable
AlphaMini process profile.

The trace established memory and thermal headroom and motivated testing a
larger inference batch. Its utilization percentage is retained as diagnostic
context, not a release threshold.

## Production-shaped batch-256 benchmark 003: final readiness evidence

Run ID: `a69d4465-789c-4286-adcd-a82b322b3027`.

Benchmark 003 used inference batch 256 and exactly 512 rolling workers with the
unchanged 64x6 model, 128-simulation search, 1,024-game cycles, training batch
512, and 2.0 sample ratio. It completed two uninterrupted cycles; the machine
report passed every automated gate and deep verification checked 16 immutable
states and eight referenced artifacts.

| Metric | Cycle 1 | Cycle 2 |
|---|---:|---:|
| Games / positions | 1,024 / 164,920 | 1,024 / 186,200 |
| Exact completed simulations | 21,109,760 | 23,833,600 |
| Simulations/s | 32,174.43 | 34,798.85 |
| Mean / maximum batch fill | 69.444% / 256 | 75.892% / 256 |
| Full measured cycle time | 771.216 s | 797.541 s |
| Successful updates | 645 | 728 |
| Validation batches | 15 | 35 |

The rate spread was 7.837%. The aggregate was 1,373 successful updates in
1,568.757 seconds, or 3,150.775 updates/hour, for a naive 226,855-update
72-active-hour projection. The slower cycle alone projects 216,780 updates;
its 15% haircut is 184,263. The final decision is therefore **180,000
successful updates**, rounded down below the slower-cycle haircut. It is now
frozen in `v1.toml` with `horizon_confirmed = true`; inference batch 256 is
frozen in the same configuration.

| Identity | SHA-256 / value |
|---|---|
| Frozen worktree | `570b54ad674351ff30fab2eb7fb6512bc5c60b7324d13011019e80b1117c332b` |
| Resolved config | `e4e4f826f9481ccfe19268b582f12ab443db5ba3ba105683038899bbbe0ea98d` |
| Semantic config | `c86e83e462f2d99fd916d2ba5c4089c109bd8c94849622ee99be147fb8efce91` |
| Final ledger HEAD | `33b7140b2d33e13ceeab59ef35616479c0b5119a8b7ff459b27a54e7c5ecd26e` |
| Benchmark report | `919868f2b6e3d7eea949daa7fbd1d3dbc0d1e257f8efe4a06c3b81102eced332` |
| GPU trace | `f9ebf08bbc775583f645df671892d488a4ed8af126c7925773704222064de179` |

The whole-run WSL `nvidia-smi` trace measured about 45% median aggregate GPU
utilization and maxima of 4,791/8,192 MiB, 60 C, and 73% utilization. It is a
sampled aggregate host signal, not a process-attributable duty-cycle measure.
Cycle-1 simulation throughput nevertheless improved 65% over benchmark 002
(32,174 versus 19,446 simulations/s), and aggregate updates/hour improved about
60%. The old 80% utilization rule is therefore a documented optimization and
investigation target, not a release gate. Fill, repeatability, exact accounting,
VRAM, thermal safety, and fail-closed no-CPU-fallback behavior all passed.

At 180,000 steps the 2.0 ratio implies about 46.08 million generated positions
and roughly 263 cycles at the observed yield. Conservative scaling from the
observed upper raw/cache bytes per position and artifact retention gives about
18 GB local high-water with verified backup and safe-boundary cache GC, or
about 278 GB without cache GC. Reserve at least 300 GB unattended without GC,
or at least 30 GB only with the documented backup-and-GC procedure operating.

Two earlier batch-256 directories are guard evidence only. Runs
`2477a2e4-951d-4eff-8373-5dfe7d445c46` and
`5d0237bb-5240-4c40-bad4-1a7eab1cfb70` stopped at initialized sequence 0 and
published no cycle/model/training artifact. The first proved that changing
documentation after initialization invalidates the frozen worktree identity;
both were abandoned instead of resumed. Only fresh run 003 is accepted.

## Recovery and CPU-serving evidence

Recovery rehearsal 002 passed both controlled cases with exact equivalence.
Collection recovery retained exit 130 and `ACTIVE_SESSION`, quarantined the
unsealed directory, followed the ordinary recovery path, and reproduced the
control state exactly. Training recovery resumed from its durable checkpoint
and reproduced the control optimizer/scaler/RNG/sampler state, metrics, model,
and ONNX bytes exactly. The evidence SHA-256 values are
`f4060c0a1f7f49b764311771b7211b6ab9ff1fe8ba97b586db2a7f1f0a382c3b`
for collection and
`97df4703d068574d73a28447de01e6b240b3c559e109e81bdf46c339fa56c201`
for training.

CPU production benchmark 002 used benchmark 003's final M2 model
`6bac11c0b2742884d11aaeb4ca4e5f983641541cd9f3fbcb2e6f32eb87f69f54`.
The frozen production batch 8 passed with exactly 40,000 simulations, zero
deadlines, 4.051 seconds/move, and 2,468.25 simulations/s. Diagnostic batch 4
also passed at 4.870 seconds/move. Diagnostic batch 1 honestly failed at 9.001
seconds/move after 25,567 simulations. Because batch 8 is the production
decision, the generated overall status is correctly **passed**; batch 1's own
diagnostic status remains failed and documented. The summary SHA-256 is
`ad938d374803e15507937ffef2d38dda7bcb91a0fe3a2465d834614b03eba701`.

## Published v1 run

The v1 run is complete. It ran as a three-directory weights-only lineage
(`run-001 -> run-002 -> run-003`) totalling 259,533 s of active compute, 72.09
hours against the precommitted 72-hour cumulative target, and an external watcher
stopped it at that target with a 0.13% overshoot. Run 002 is retained as a failed
fork: its configuration was seeded with an unedited copy of the parent, so the
`parity_atol` relaxation it was created for was never applied and its cycle-4
ONNX export correctly failed.

The final model is the cycle-338 export `model-128875d3a28138e3`, SHA-256
`128875d3a28138e35cc03a4d072e7337cbc3c9c0906bba8208f8e1d2d73632c5`, provisioned
outside Git to `artifacts/alphamini/current/{model.onnx,manifest.json}`.

AlphaMini calibrates to **1969 Chess.com 30+0 move-quality Elo** with a 95%
whole-player bootstrap interval from 1758 to at or above 1999, measured on
2026-08-25 by the same method as the published `MinimaxDepth3V1` result. Full
lineage, incidents, evaluation, and calibration evidence are in
[results/run-003.md](results/run-003.md).

All four run directories are Git-ignored. Copy `runs/alphamini-pilot-001`,
`runs/alphamini-pilot-002`, and the three `runs/alphamini-run-00*` directories to
independent storage before any cleanup that could remove ignored files, and
preserve their paths and hashes in the backup record. Recheck a preserved run
without changing it:

```bash
export ALPHAMINI_RUN_DIR="$PWD/runs/alphamini-run-003"
uv run --project alphamini-train alphamini-train verify \
  --run-dir "$ALPHAMINI_RUN_DIR" --deep
uv run --project alphamini-train alphamini-train report \
  --run-dir "$ALPHAMINI_RUN_DIR"
```

`configs/alphamini/v1.toml` remains settled at inference batch 256 and 180,000
successful updates with `horizon_confirmed = true`. Any further training is a new
fork with its own recorded reason, not a retune of this identity.

## Serving and deployment

The model is served by `chessengines-alphamini.service` at the frozen serving
budget: 9,000 ms, a 10,000-simulation cap, batch 8, one bounded search at a time
on CPU. `scripts/deploy.sh` runs the workspace suite, builds the release
binaries, revalidates the model and manifest with `alphamini --verify-only`,
publishes the frontend, restarts the services, and smoke-tests every route.

A passing Depth-3 gate verdict is **not** a deployment requirement. That is a
deliberate policy decision: cross-model matches will be run separately as their
own comparison work rather than as a release precondition. The arena gate
machinery is unchanged and still available — `--gate`/`ALPHAMINI_GATE_PATH` is
accepted, and when a verdict is supplied it is validated against the model and
frozen baseline identities exactly as before, which also pins serving to the
frozen search budget. When no verdict is supplied the server starts normally on
its configured search parameters.

The Depth-3 match was started before the requirement was dropped and stopped
partway, at 65W–0D–7L over 72 games across 36 of the 200 opening pairs. No
verdict is claimed from a partial match. The results file
`artifacts/alphamini/release-gate/pairs.jsonl` is durable and resumable, so that
comparison can continue from where it stopped.
