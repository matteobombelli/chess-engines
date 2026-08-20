# AlphaMini implementation log

This file records engineering work, decisions, measured evidence, and incidents.
It is not a benchmark report. Add UTC dates, source commits, artifact/config
hashes, and exact commands whenever measurements begin.

Current operational status and continuation steps are in
[`status.md`](status.md).

## Development process

The implementation follows a reproducibility-first sequence so performance work
cannot conceal correctness drift:

1. Inspect and freeze existing chess, arena, HTTP, and Minimax behavior. Record
   baseline tests and opponent digests before adding neural code.
2. Freeze encoder, policy, raw-record, tensor-cache, model-manifest, and MCTS sign
   contracts. Add golden fixtures before producing valuable self-play.
3. Implement Rust search/self-play and Python training independently against
   those contracts, then require a Rust-generated shard/materialization fixture
   to load and train in Python. Handwritten lookalike fixtures are supplemental,
   not cross-language evidence.
4. Make every lifecycle transition crash-testable: interrupt collection,
   materialization, training, export, and pointer promotion; recover only through
   durable hashes and deterministic IDs.
5. Run the disposable pilot, measure throughput and resource utilization, and
   freeze the v1 optimizer horizon. Do not transfer pilot weights into Run 1.
6. Start Run 1 only from a committed source/config/lock identity after production
   doctor completes real Rust CUDA inference. Keep an append-only incident and
   decision record during the run.
7. At 72 active hours, preserve the immutable milestone, run the predeclared
   evaluation ladder, and publish the measured result. Continue unchanged with
   `extend`, or use a parent-linked `fork` for any semantic change.

Reviews found and closed several boundary failures during implementation: action
flattening was corrected to plane-major; Python was aligned to Rust's nested raw
shard and absolute outcome; tensor paths became manifest-relative; collector run
UUID and semantic flags became explicit; and materialization crash recovery now
quarantines final-named but uncommitted tensors. A final cross-language audit
also aligned positive/unique policy targets, exact ply indices, the 512-ply cap,
outcome/termination validation, and a required selected move that lets Rust
replay and verify every full game trajectory. Shard and collection identities
now additionally bind the collection seed, simulation budget, and ply cap;
deterministic SplitMix64 game seeds and exact per-position visit sums are checked
across the Rust/Python boundary before cache admission. These are kept here
because the review/fix trail is part of the systems result.

## 2026-08-18 — v1 implementation baseline

- Chose pure self-play, Rust-owned chess/MCTS/materialization, PyTorch training,
  FP32 ONNX CPU serving, and an ungated latest-model self-play loop.
- Replaced an inherited eight-frame/119-plane input with a 22-plane current-state
  representation. This reduces inference cost and avoids making every
  transposition distinct by history. It deliberately loses the last-move cue.
- Kept halfmove/repetition in the authoritative input. Exact-input caching is
  deferred; a future instrumented prototype must quantify clock/repetition
  near misses and show at least 5% more complete simulations per second.
- Defined schema-stable raw `PositionRecordV1` shards and disposable Rust-built
  mmap tensor caches. Sparse policy targets prevent roughly 18.7 GB of dense
  policy storage per million positions.
- Froze policy indexing as `movement_plane * 64 + canonical_origin`; the model
  flattens NCHW directly.
- Added immutable content-addressed state, atomic `HEAD`/`RECOVERY`, advisory
  locking, complete RNG/optimizer/AMP checkpoints, deterministic sampling, active
  time heartbeats, cumulative cycle loss accounting across recovery, strict
  resume/extend/fork boundaries, deep verification, and conservative
  reference-driven GC. Deterministic CUDA startup pins the frozen cuBLAS
  workspace setting and rejects conflicting inherited state.
- Added an immutable first-budget safe-boundary milestone containing the exact
  checkpoint/model and counters at the original threshold. Exhausted resumes
  fail before opening an active session; extensions are ordered objects that
  retain the original milestone hash and cannot be applied prematurely.
- Added milestone rungs against frozen Depth 1/2/3 Minimax and a precommitted miss
  protocol so a 72-hour Depth-3 miss remains a useful measured result.
- Added equal-budget dual-AlphaMini arena matches and a deterministic
  Bradley–Terry consumer that validates both model hashes, pair-log checksum,
  complete opening prefix, binary/search identity, and every paired game before
  deriving counts. The checkpoint ladder never depends on hand-copied scores.

## 2026-08-19 — independent pre-run review hardening

- Made deployment execute the feature-enabled arena/AlphaMini identity test;
  the default workspace test alone does not compile that cross-crate check.
- Extracted one dependency-free artifact helper for shared SHA-256 and durable
  no-clobber publication, including racing-writer and idempotent-retry tests.
- Consolidated en-passant structure/effectiveness so application, generation,
  SAN, Board repetition identity, SearchPosition Zobrist identity, and undo use
  the same predicate. Valid and malformed white/black cases assert Board/Search
  parity, including pinned pawns and invalid targets.
- Moved encoded leaves and evaluation rows through MCTS/cross-game scheduling
  without clone handoffs. Root visit records now retain their legal `Move`, so
  self-play neither regenerates legal lists per action nor decodes its own root
  policy. The shard verifier/materializer also share one exact descriptor and
  collection validation path.
- Centralized the production simulations/time/batch/ply/bootstrap constants and
  persisted exact `cpuct`/FPU millionths in pair logs and gate verdicts. Serving,
  arena search, and deployment validation now consume the same contract.
- Reduced per-step metric transfers to one device synchronization and replaced
  Python scalar-by-scalar sparse-policy packing with NumPy chunk assembly. The
  crash-ordering refactors retained exact interrupted/resumed model tensors,
  sampler state, and reported metrics in the CPU recovery test.
- Added an explicit disposable-pilot lineage mode: an uncommitted pilot freezes
  the base commit and a digest of every tracked/untracked nonignored worktree
  file, refuses resume after any content drift, and is labeled non-publishable.
  The v1 configuration remains clean-commit-only.

## 2026-08-20 — RTX 3070 disposable pilot

- `alphamini-pilot-001` exposed missing CUDA 13 runtime/provider staging, then a
  checkpoint restore bug before optimizer step 1. The collection failure was
  quarantined through normal recovery; a second attempt completed eight games
  and materialized 884 records before training found that CUDA `map_location`
  had moved the saved CPU RNG state off host.
- Fixed RNG restoration by validating CPU/CUDA generator state and normalizing
  every ByteTensor to contiguous host memory. Added a target-GPU regression
  that recreates CUDA-mapped checkpoint tensors. The frozen failed run was not
  altered or mislabeled after the source fix.
- `alphamini-pilot-002` completed one full real cycle: eight games, 884
  positions, 14,144 simulations, 15,005 neural evaluations, 28 successful AMP
  updates, checkpoint/export promotion, and Rust CUDA loading of M1. Deep
  verification checked nine state objects and six artifacts.
- Tiny-pilot self-play measured 7,476 complete simulations/s, 86.2% batch fill
  at capacity 8, and 56.8 compressed shard bytes/position. These are smoke-run
  measurements for a 16x1 model, not projections for the 64x6 v1 model.
- Strict PyTorch/Python-ORT/Rust-ORT CUDA parity passed with maximum absolute
  errors `6.71e-8` policy and `5.59e-9` WDL. The production doctor passed every
  runtime check and failed only the deliberately dirty Git identity.
- The pilot used manually staged ORT CUDA provider libraries and an ignored
  CUDA 13 runtime directory. Reproducible, versioned CUDA packaging is now an
  explicit pre-v1 blocker; see `status.md` for the exact evidence and sequence.

Incident timeline, UTC:

1. At `06:07:46`, `pilot-001` created and exported M0. At `06:07:53`, its first
   collection exited because `libonnxruntime_providers_shared.so` was absent.
2. Matching provider libraries and a CUDA 13 runtime were staged without
   changing source. Recovery at `06:15:05` quarantined the failed collection and
   accounted 6.829 active seconds.
3. The retry sealed eight games and materialized 884 records, then the RNG
   restore exception stopped training before step 1. The exception was visible
   in operator output but was not persisted in a run log. Recovery at `06:16:44`
   accounted another 7.177 seconds and retained the run at `ready_train`.
4. The RNG source fix changed the frozen worktree content, so `pilot-001` was
   deliberately abandoned rather than resumed under different code.
5. `pilot-002` started at `06:18:35` and promoted M1 at `06:18:40` without an
   interruption. Its committed cycle remains immutable and non-publishable.

## Verification completed in the implementation checkout

- Rust encoder/action tests cover both orientations, castling, en passant, and
  every promotion; the production doctor records the published fixture digest.
- A Rust-produced raw shard and materialized tensor cache are consumed by the
  Python reader in tests.
- A real one-step CPU train/export test checks PyTorch against Python ONNX
  Runtime. A separate fixed-input integration test rebuilds the Rust inference
  binary and compares PyTorch, Python ORT, and Rust ORT logits while binding the
  golden encoded-input and runtime-output digests.
- Reversible search-position tests cover nested capture, castling, en passant,
  promotion, repetition, terminal precedence, and starting-position perft.
- MCTS tests cover exact mixed terminal/neural budgets, mate signs, FPU,
  virtual-loss cleanup, legality masks, and shared cross-game batching.

## Measurement status before Run 1

- Target-machine golden fixture digest and Rust/PyTorch/ORT CUDA parity are now
  measured and passing for the disposable pilot.
- The final 64x6 rehearsal is complete. Batch-256 benchmark 003 used two full
  1,024-game cycles with 512 bounded rolling workers and retained utilization,
  VRAM, power, fill, throughput, and disk evidence. Batch-128 benchmark 002 is
  retained as a superseded tuning baseline.
- Search-position throughput versus the prior clone-based baseline.
- Full-run disk high-water is projected conservatively from measured storage:
  approximately 18 GB with verified off-volume backup plus safe-boundary cache
  GC, or 278 GB without cache GC, at the recommended 180,000-step horizon.
- Exact-cache benefit and clock/repetition near-miss fraction require the
  explicitly deferred instrumented-cache prototype.
- CPU production benchmark 002 passed at the frozen production batch 8 with
  40,000 exact simulations, zero deadlines, and 4.051 seconds/move. Diagnostic
  batch 4 passed; batch 1 remains a documented diagnostic failure.
- The measured horizon decision is 180,000 successful updates. It is frozen in
  `v1.toml` with `horizon_confirmed = true`, alongside inference batch 256, and
  must never change after Run 1 begins.

## 2026-08-20 — CUDA 13.2 stage and production benchmark 001

- Replaced the pilot's manual mixed CUDA stage with seven exact, hash-locked
  CUDA 13.2/cuDNN wheels and regular copies of the reviewed ORT rc.13 CUDA and
  shared providers. The machine manifest is `c2684ff2...3544`; the runtime
  library-set digest is `a1d60d34...e39a`. Setup/verification checked 21
  runtime files and the complete loader closure.
- The RTX 3070 executed pilot M1 through that stage. The strict doctor then ran
  a random 64x6 production graph through PyTorch, Python ORT CPU, and Rust ORT
  CUDA and completed one-ply Rust CUDA self-play. All runtime gates passed; Git
  cleanliness was the sole expected doctor failure.
- Production benchmark 001 completed two uninterrupted cycles with exact
  2,705,152 and 2,316,416 simulation totals, nonempty grouped validation, zero
  AMP overflows, finite losses, passing ONNX parity, and deep ledger/artifact
  verification.
- The predeclared performance gate failed rather than being relaxed. Mean
  batch fill was 32.2% and 28.1%, and simulation-rate spread was 18.6%. Fill
  began near 100% but collapsed as a finite 128-game cohort drained; the final
  long games dominated both cycles. Measured rates were 11,259 and 9,346
  simulations/s.
- GPU telemetry ruled out memory/thermal pressure: 55% median utilization,
  4,500 MiB peak memory, and 58 C maximum temperature. This evidence motivated
  a bounded replenishing worker pool before the next benchmark. The measured
  1,047 updates/hour remains explicitly ineligible for the v1 horizon.

## 2026-08-20 — production benchmarks 002/003 and operational drills

- Benchmark 002 run `cd60ce38-1317-4835-bb9e-643f6c426cab` froze resolved
  config `b0ec014d...3a43d`, semantic config `fdae1887...dcfe`, and worktree
  `ff507930...e987`. Its final ledger HEAD is `c860bf8b...ace1d`.
- The 256-worker rolling collector completed 1,024 games per cycle. Cycles 1
  and 2 produced 164,920 and 180,762 positions, exact simulation totals of
  21,109,760 and 23,137,536, mean fills of 81.76% and 82.65%, and rates of
  19,446 and 20,334 simulations/s. The rate spread was 4.46%, both cycles had
  zero AMP overflow, and all validation, finite-loss, parity, and ledger gates
  passed.
- Whole-cycle throughput was 1,352 successful updates in 2,466.324 seconds,
  or 1,973 updates/hour. The naive 72-active-hour projection is 142,089. The
  then-provisional 120,000 recommendation was 15.5% below the aggregate
  projection and 13.3% below the slower cycle's projection; benchmark 003
  superseded it before `v1.toml` was changed.
- The report SHA-256 is `8a019679...d2106`; the 2,489-sample GPU trace SHA-256
  is `bfe75d1a...54a`. Saturated self-play measured 58% median utilization and
  61--62% p90. Absolute maxima were 4,812/8,192 MiB, 62 C, and 112.6 W, with
  stable 1,875 MHz self-play clocks. WSL reported 36--37% utilization outside
  collection and changing memory baselines, so the trace is aggregate rather
  than process-attributable.
- The earlier `>= 80%` utilization rule became an explicit optimization target,
  not a correctness or release gate. Benchmark 002's hard signals were bounded
  worker accounting, exact visits, repeatability, finite/parity-valid training,
  VRAM below 90%, and no fallback/OOM/throttling. It did not establish GPU
  saturation; the evaluator remained the next profiling target.
- Benchmark 002's historical 120,000-step storage projection was 14.44 GB with
  backup/cache GC or 185.45 GB without GC. Those figures are superseded by the
  final 180,000-step projection below.
- Recovery rehearsal 002 passed controlled collection and training recovery.
  Collection evidence SHA-256 is
  `f4060c0a1f7f49b764311771b7211b6ab9ff1fe8ba97b586db2a7f1f0a382c3b`;
  training evidence SHA-256 is
  `97df4703d068574d73a28447de01e6b240b3c559e109e81bdf46c339fa56c201`.
  Both ordinary recovery paths deep-verified and reproduced the control
  state/model exactly, including optimizer/scaler/RNG/sampler state, metrics,
  and ONNX bytes where applicable.
- CPU production benchmark 002 used benchmark 003 M2
  `6bac11c0b2742884d11aaeb4ca4e5f983641541cd9f3fbcb2e6f32eb87f69f54`.
  Frozen batch 8 passed with 40,000/40,000 simulations,
  zero deadlines, 4.051 seconds/move, and 2,468.25 simulations/s. Diagnostic
  batch 4 passed at 4.870 seconds/move. Diagnostic batch 1 failed at 9.001
  seconds/move after 25,567 simulations. The overall summary correctly passed
  because batch 8 is the production decision; batch 1's diagnostic status
  remains failed. Summary SHA-256 is
  `ad938d374803e15507937ffef2d38dda7bcb91a0fe3a2465d834614b03eba701`.
- Benchmark 002 is now explicitly a superseded batch-128 tuning baseline. Its
  rolling scheduler and safety evidence remain useful, but its former 120,000
  horizon recommendation is not the final production decision.
- Final batch-256 benchmark 003 is run
  `a69d4465-789c-4286-adcd-a82b322b3027`. It froze resolved config
  `e4e4f826...a98d`, semantic config `c86e83e4...ce91`, worktree
  `570b54ad...32b`, and final ledger HEAD `33b7140b...26e`.
- Its two 1,024-game cycles used exactly 512 rolling workers. They produced
  164,920 and 186,200 positions, exact totals of 21,109,760 and 23,833,600
  simulations, 69.444% and 75.892% mean batch fill, and 32,174.43 and
  34,798.85 simulations/s. Full cycle times were 771.216 and 797.541 seconds;
  successful updates were 645 and 728. The simulation-rate spread was 7.837%.
- Aggregate throughput was 1,373 successful updates in 1,568.757 seconds, or
  3,150.775 updates/hour, with a naive 226,855-update 72-hour projection. The
  slower cycle projects 216,780; its 15% haircut is 184,263. The final horizon
  recommendation is 180,000, rounded down below that conservative boundary.
  It is now frozen in `v1.toml` with `horizon_confirmed = true`, together with
  inference batch 256.
- The report SHA-256 is `919868f2...d332`; the GPU trace SHA-256 is
  `f9ebf08b...e179`. Whole-trace maxima were 4,791/8,192 MiB, 60 C, and 73%
  utilization; median aggregate utilization was about 45%. WSL sampling is not
  process attributable. Cycle-1 simulation throughput rose 65% over benchmark
  002 and aggregate updates/hour rose about 60%, so the lower utilization signal
  is not a release gate. The 80% value remains an optimization/profiling target.
- CUDA inference was configured to fail closed with CPU fallback disabled and
  benchmark 003 proved end-to-end collection, training/export, and parity under
  that contract. Exact accounting, repeatability, fill/throughput, VRAM, and
  thermal gates passed.
- At 180,000 steps, observed yield implies about 46.08 million positions and
  roughly 263 cycles. Conservative observed storage scaling gives about 18 GB
  local high-water with verified backup and cache GC, or about 278 GB without
  GC; reserve at least 300 GB for unattended no-GC operation.
- Preliminary batch-256 runs `2477a2e4-951d-4eff-8373-5dfe7d445c46` and
  `5d0237bb-5240-4c40-bad4-1a7eab1cfb70` stopped at initialized sequence 0 and
  produced no cycle/model/training evidence. The first proved a post-freeze doc
  edit changes the worktree identity. Both were abandoned; fresh run 003 is the
  sole final measurement.

## Incident template

```text
UTC time:
Run ID / cycle / step:
HEAD / RECOVERY:
Observed failure:
Last durable heartbeat:
Checks performed:
Recovery command:
Lost or retried active compute:
Determinism impact:
Follow-up change (same run or required fork):
```
