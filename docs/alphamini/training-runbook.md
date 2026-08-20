# AlphaMini training runbook

This is the operational source of truth. Run commands from the repository root
on Linux with Python 3.12. Do not edit a run directory by hand.

The current implementation/pilot state and ordered pre-v1 checklist are in
[`status.md`](status.md). Read that handoff before starting or resuming work.

## 1. Install and inspect

Install `uv`, select Python 3.12, and sync the committed environment:

```bash
uv python install 3.12
uv sync --project alphamini-train --extra train --extra test --locked
uv run --project alphamini-train alphamini-train doctor \
  --config configs/alphamini/pilot.toml
```

Before a published run, use the stricter check:

```bash
mkdir -p artifacts/alphamini/doctor
set -o pipefail
uv run --project alphamini-train alphamini-train doctor \
  --config configs/alphamini/v1.toml --production | \
  tee artifacts/alphamini/doctor/run-001-production.json
```

`doctor` checks Python and ML/runtime packages, Git cleanliness, configured Rust
commands, atomic rename/fsync behavior, and disk headroom. Fix every production
failure. Production mode also exports a temporary seeded model and compares one
fixed Rust-encoded input across PyTorch, Python ORT CPU, and Rust ORT CUDA under
the frozen tolerances. Its evidence records the golden FEN, encoded-input and
fixture SHA-256, each runtime's output digest, and pairwise maximum errors. It
then completes a one-ply Rust ORT CUDA self-play fixture, proving target-GPU
inference and collection rather than trusting feature names alone. A configured
command is optional only for manifest/unit tests; training cannot advance
without both collect and materialize commands.

Keep run data on a local filesystem that supports advisory locks and atomic
same-directory rename. Confirm roughly 100 GiB free for the pilot and estimate
the full-run high-water mark from measured compressed bytes/position before
starting v1.

### CUDA runtime packaging

The pinned Rust ORT rc.13 Linux artifact uses a CUDA 13 provider; PyTorch's
bundled CUDA 12.6 libraries do not satisfy its `libcublas.so.13` and
`libcudart.so.13` dependencies. A `--features cuda` build therefore needs both
matching ORT provider libraries beside the executable and a compatible CUDA
13/cuDNN runtime on `LD_LIBRARY_PATH`.

The 2026-08-20 disposable pilot proved CUDA execution with manually staged
CUDA 13.0 libraries and a cuDNN library borrowed from the Torch environment.
That remains pilot evidence only. The reproducible contract is the separately
installed, hash-locked CUDA 13.2 stage:

```bash
scripts/alphamini-cuda-runtime setup
scripts/alphamini-cuda-runtime verify
python3 -m json.tool target/alphamini-cuda-runtime/13.2/manifest.json
```

`setup` uses
[`cuda13-runtime-requirements.txt`](../../configs/alphamini/cuda13-runtime-requirements.txt)
with pip's `--require-hashes --no-deps --only-binary` checks, but installs only
beneath ignored `target/alphamini-cuda-runtime/13.2`; it cannot target or modify
`alphamini-train/.venv`. It forces a fresh rc.13 ORT build, verifies the
downloaded distribution/static archive/provider identities, copies regular
provider files beside `target/release/alphamini-selfplay`, and emits both
`manifest.json` and `with-alphamini-cuda`. The manifest records every wheel
hash, installed version, runtime-library hash, provider hash, and the Rust ORT
runtime/distribution identity.

`verify` rehashes that closure, rejects provider cache symlinks, checks that
runtime symlinks remain inside the stage, and fails on any unresolved `ldd`
dependency. All CUDA collector commands use
`scripts/alphamini-cuda-runtime exec -- ...`, which repeats verification and
scopes CUDA 13.2's `LD_LIBRARY_PATH` to the Rust child. The Torch cu126 process
is never overlaid. `doctor` includes the verified machine-readable identity in
its report, while `RUN.json` freezes the manifest, package, provider, and
runtime-library-set identities for resume compatibility.

After `cargo clean` or deletion of ignored `target/`, run `setup` again. An
existing valid stage is reused and repairs regular provider copies; an invalid
stage fails closed. Use `setup --replace` only after a reviewed requirements or
ORT identity change. A passing `setup`/`verify` establishes the loader closure;
the production doctor's real CUDA parity and self-play checks are still the
required target-GPU execution gate.

## 2. Run the disposable pilot

Use a new directory every time:

```bash
export ALPHAMINI_PILOT_DIR="$PWD/runs/alphamini-pilot-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$ALPHAMINI_PILOT_DIR"
uv run --project alphamini-train alphamini-train start \
  --config configs/alphamini/pilot.toml \
  --run-dir "$ALPHAMINI_PILOT_DIR" \
  --one-cycle
uv run --project alphamini-train alphamini-train verify \
  --run-dir "$ALPHAMINI_PILOT_DIR" --deep
uv run --project alphamini-train alphamini-train report \
  --run-dir "$ALPHAMINI_PILOT_DIR"
```

The pilot config is explicitly `run.disposable = true`, so it may start from a
dirty worktree. `RUN.json` freezes the commit plus a content digest of every
tracked and untracked nonignored file; changing any source during the pilot
blocks resume. The generated report labels the run non-publishable. The v1
config is `disposable = false` and still requires a clean committed checkout.

`start` creates the immutable run manifest, seeds Python/NumPy/Torch, exports
random model `M0`, verifies it with ONNX Runtime, collects self-play, asks Rust to
materialize tensors, trains, checkpoints, exports `M1`, and atomically commits
cycle 1. There is no dummy production evaluator. Rust's explicitly selected
uniform evaluator is permitted only in fixtures.

The pilot is a throwaway run and never becomes v1's parent. Before v1, also:

1. Interrupt collection and training separately; follow the recovery procedure
   below and confirm the same deterministic game IDs and successful-step stream.
2. Confirm golden encoder/action fixtures and PyTorch/ONNX/Rust inference parity.
3. Run a scale-representative 64x6 rehearsal for at least 15 minutes. Measure
   clone versus make/unmake, move generation, complete simulations/sec,
   positions/hour, GPU utilization/VRAM/power and batch fill, cycle phase time,
   disk bytes/position, and CPU batches 1/4/8 under the nine-second serving
   limit. The eight-game 16x1 pilot is only an end-to-end smoke test. Inference
   caching is an explicit post-v1 prototype, so the pilot does not claim a
   nonexistent cache A/B result.
4. Confirm that `v1.toml` still has the benchmark-003 decisions:
   `self_play.batch_size = 256`,
   `training.frozen_horizon_steps = 180000`, and
   `training.horizon_confirmed = true`. Commit those exact values with the
   reviewed implementation, then rerun the production doctor. `start` rejects
   any unconfirmed schedule. Once v1 begins, do not edit it.

## 3. Start the published run

Make sure the code and lockfiles are committed and the production doctor passes.
Initialize and inspect seeded `M0` without yet collecting:

```bash
export ALPHAMINI_RUN_DIR="$PWD/runs/alphamini-run-001"
uv run --project alphamini-train alphamini-train start \
  --config configs/alphamini/v1.toml \
  --run-dir "$ALPHAMINI_RUN_DIR" \
  --initialize-only
uv run --project alphamini-train alphamini-train verify \
  --run-dir "$ALPHAMINI_RUN_DIR" --deep
```

Then run under a process supervisor or a persistent terminal:

```bash
uv run --project alphamini-train alphamini-train resume \
  --run-dir "$ALPHAMINI_RUN_DIR"
```

There may be only one orchestrator for a run. The advisory `run.lock` makes a
second process fail immediately. The config is copied read-only into the run;
resume has no config flag and therefore cannot accept silent overrides.

For a controlled cycle-at-a-time rehearsal:

```bash
uv run --project alphamini-train alphamini-train resume \
  --run-dir "$ALPHAMINI_RUN_DIR" --one-cycle
```

Do not use `--one-cycle` for the unattended 72-hour run unless an external
supervisor deliberately repeats it.

## 4. Monitor without changing state

The active command holds the advisory lock, so monitoring reads files directly:

```bash
python3 -m json.tool "$ALPHAMINI_RUN_DIR/ACTIVE_SESSION.json"
python3 -m json.tool "$ALPHAMINI_RUN_DIR/RUN.json"
readlink -f "$ALPHAMINI_RUN_DIR/pointers/HEAD" 2>/dev/null || \
  sed -n '1p' "$ALPHAMINI_RUN_DIR/pointers/HEAD"
nvidia-smi
find "$ALPHAMINI_RUN_DIR/cycles" -name collect.log -print -exec tail -n 5 {} \;
```

`ACTIVE_SESSION.json` contains its last durable heartbeat and counted elapsed
seconds. Tensor/checkpoint `.partial` files are never inputs. `HEAD` and
`RECOVERY` contain SHA-256 object names, not mutable state.
Each external command also has an atomic `*-command.json` record with resolved
arguments/environment, start/end/status/elapsed time, plus a persistent log.
Rust's per-shard JSON in `collect.log` supplies batch fill, inference time,
games/hour, simulations, and termination histograms for the final report.
Every sealed collection and shard binds the requested collection seed,
simulation count, and `max_plies` (never above the v1 cap of 512); every game
seed is deterministically derived from that collection seed and game ID. Deep
verification decodes the raw shards and checks these identities plus exact
per-position visit sums before their tensor cache is trusted.

After the process stops normally, generate a factual status report:

```bash
uv run --project alphamini-train alphamini-train report \
  --run-dir "$ALPHAMINI_RUN_DIR"
```

The report leaves Elo unknown until arena/calibration results are explicitly
imported; it never infers success from training loss.

## 5. Stop, recover, and continue from the last durable state

For a planned stop, send Ctrl-C or SIGTERM once. The CLI converts either signal
into a controlled unwind, terminates its active Rust child, retains
`ACTIVE_SESSION.json`, and exits 130. Do not send SIGKILL for a planned pause,
and do not immediately invoke `resume`: use the recovery sequence below. Never
delete a lock, pointer, or partial file to make progress.

After power loss, kill, OOM, or a nonzero worker exit, the active-session file is
intentionally retained. First inspect it and verify the recorded PID is gone:

```bash
python3 -m json.tool "$ALPHAMINI_RUN_DIR/ACTIVE_SESSION.json"
ps -p "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' \
  "$ALPHAMINI_RUN_DIR/ACTIVE_SESSION.json")"
```

Then recover and verify:

```bash
uv run --project alphamini-train alphamini-train recover \
  --run-dir "$ALPHAMINI_RUN_DIR"
uv run --project alphamini-train alphamini-train verify \
  --run-dir "$ALPHAMINI_RUN_DIR" --deep
uv run --project alphamini-train alphamini-train resume \
  --run-dir "$ALPHAMINI_RUN_DIR"
```

Recovery counts active time only through the last durable heartbeat, adopts a
compatible `RECOVERY` checkpoint if present, and preserves every partial file
for inspection. During training this discards only successful updates after the
latest periodic checkpoint (at most the configured checkpoint interval). An
unsealed collection or materialization phase is quarantined and deterministically
reissued from its committed input identity. Dynamic CUDA batch scheduling means
a regenerated unsealed collection is not promised byte-identical, but it cannot
be mixed with the abandoned attempt. Recovery never guesses by newest filename.
`--force` is only for a confirmed-stale PID that appears alive because a PID was
reused; record that incident in the implementation log.

If Rust crashed during materialization, some tensor files may already have their
final immutable names even though `tensors.json` was never sealed. Because HEAD
is still `ready_materialize`, recovery moves that entire uncommitted cycle cache
directory into quarantine; the retry starts with an empty directory and the same
deterministic inputs. The analogous rule applies to an uncommitted collection.

If verification reports corruption, stop. Restore the exact hash from backup,
then re-run deep verification. If no compatible optimizer/RNG/replay checkpoint
survives, create a weights-only child with `fork`; do not call it resume.

## 6. Continue training beyond 72 hours

When the active budget is reached, the orchestrator finishes the in-flight cycle
so the boundary has a fully promoted model, accounts the session, and seals the
first-budget milestone object. `report` and `reproduce` show its SHA-256,
accounted time/overshoot, model cycle, and optimizer step. At that point a plain
`resume` fails before creating `ACTIVE_SESSION`; it cannot silently turn into a
no-op or continue past the declared experiment.

The original 72-hour milestone remains immutable. Run the frozen evaluation
against the model descriptor captured by that object. To add another 24 active
hours to the exact same experiment:

```bash
uv run --project alphamini-train alphamini-train extend \
  --run-dir "$ALPHAMINI_RUN_DIR" \
  --additional-active-budget 24h \
  --reason "Precommitted Run 1 continuation after Depth-3 gate miss"
uv run --project alphamini-train alphamini-train verify \
  --run-dir "$ALPHAMINI_RUN_DIR" --deep
uv run --project alphamini-train alphamini-train resume \
  --run-dir "$ALPHAMINI_RUN_DIR"
```

This preserves model, optimizer, AMP scaler, RNG streams, sampler cursor,
replay policy, seed/game counters, cycle numbering, MCTS behavior, and the frozen
learning-rate horizon. After the horizon, LR remains `1e-4`; extension never
stretches or restarts the cosine. Each extension is an immutable object containing
its prior budget, added seconds, timestamp, reason, and original milestone hash.
`extend` is fail-closed until the current active budget is exhausted at a safe
model boundary, so it cannot be pre-applied in a way that skips the original
marker. Later extensions retain that same first-budget milestone unchanged.

`resume` also recomputes the source commit plus `Cargo.lock` and `uv.lock` hashes
stored inside checkpoints. If the repository has moved on, use a clean worktree
checked out at the run's recorded commit for the continuation; do not weaken the
lineage check to make a newer checkout appear identical.

Use `extend` only when all learning semantics and relevant runtime behavior are
unchanged. A reboot with the identical pinned stack is an exact resume. A GPU,
driver, CUDA, Python, PyTorch, or ONNX Runtime change is rejected before any new
self-play is collected. Use `fork` and label the child warm-start; v1 has no
unimplemented "non-bitwise segment" escape hatch.

Create a new experiment for any change to architecture, encoder/action mapping,
loss, optimizer, schedule, batch/accumulation, replay, self-play/MCTS, inference
precision used for self-play, chess rules, external data, or a relevant bug fix:

```bash
uv run --project alphamini-train alphamini-train fork \
  --source-run "$ALPHAMINI_RUN_DIR" \
  --config configs/alphamini/experiment-example.toml \
  --run-dir "$PWD/runs/alphamini-experiment-002" \
  --reason "Test changed replay ratio; weights-only warm start"
```

A fork copies and verifies weights, records parent HEAD and semantic hash, resets
optimizer/schedule/RNG by explicit migration, and is labeled warm-start. If an
architecture/schema migration adapter has not been implemented and tested, the
fork remains staging rather than silently dropping incompatible parameters.

## 7. Backup, retention, and garbage collection

Back up at least `RUN.json`, frozen config, pointers, `objects/`, current plus two
prior recovery sets, all raw shards, evaluated models, and reports. Verify the
copy by hashes and create a backup marker with schema
`alphamini.backup-verification.v1` and the run ID.

GC is always a preview first:

```bash
uv run --project alphamini-train alphamini-train gc \
  --run-dir "$ALPHAMINI_RUN_DIR"
```

It proposes only `.partial` files and materialized cache directories unreferenced
by current replay/recovery state. It never proposes raw shards, checkpoints,
models, manifests, or the state chain. Preview reports whether an
`ACTIVE_SESSION` marker is present; apply is fail-closed until that session has
finished or been recovered because its partial files may still be recovery
inputs. Apply only after independent backup verification:

```bash
uv run --project alphamini-train alphamini-train gc \
  --run-dir "$ALPHAMINI_RUN_DIR" --apply \
  --backup-marker /independent/storage/run-001-backup-verified.json
```

## 8. Reproduce and hand off

Before evaluation or machine migration:

```bash
uv run --project alphamini-train alphamini-train reproduce \
  --run-dir "$ALPHAMINI_RUN_DIR"
```

This first performs deep verification, then emits run/config/semantic hashes,
source commit, original/current runtime, HEAD/RECOVERY, progress counters, and
exact verification/continuation commands. It also emits original/current budget,
extension hashes, and the complete first-budget milestone object/reference.
Preserve that JSON with the evaluation suite hashes and match records.

## 9. Run checkpoint matches and fit the learning curve

Build the arena once from the run's recorded commit. Reusing that exact binary
keeps its hash identical in every pair-log header:

```bash
cargo build --locked --release -p arena --features alphamini
mkdir -p "$ALPHAMINI_RUN_DIR/evaluation/checkpoint-pairs"
```

At every cycle divisible by five, set the challenger and current deployment
champion to their immutable ONNX and manifest paths, then run 50 opening pairs:

This cadence is a documented post-cycle operation; the v1 orchestrator does not
launch it automatically. Stop after the committed cycle with `--one-cycle`, run
the command below, record its elapsed wall time, then resume training. Because it
runs outside `ACTIVE_SESSION`, report it as additional evaluation compute rather
than claiming it inside the 72-hour active-training budget.

```bash
export CHALLENGER_MODEL=/absolute/path/to/challenger.onnx
export CHALLENGER_MANIFEST=/absolute/path/to/challenger.json
export CHAMPION_MODEL=/absolute/path/to/champion.onnx
export CHAMPION_MANIFEST=/absolute/path/to/champion.json
export PAIR_LOG="$ALPHAMINI_RUN_DIR/evaluation/checkpoint-pairs/cycle-000005-v-champion.jsonl"

target/release/arena \
  --alphamini-model "$CHALLENGER_MODEL" \
  --alphamini-manifest "$CHALLENGER_MANIFEST" \
  --opponent-model "$CHAMPION_MODEL" \
  --opponent-manifest "$CHAMPION_MANIFEST" \
  --openings arena/openings/alphamini-v1.json \
  --games 50 \
  --seed 1 \
  --max-plies 1000 \
  --bootstrap 20000 \
  --alphamini-simulations 128 \
  --alphamini-time-ms 60000 \
  --alphamini-batch-size 8 \
  --exploratory true \
  --results "$PAIR_LOG"
```

The arena gives both AlphaMini engines the same search configuration. The large
time cap is a safety limit; record any deadline hit rather than calling that a
fixed-128-simulation result. `--results` is append-durable and resumes only when
the full header—including both model hashes, binary hash, opening IDs, and search
limits—matches. Exploratory matches never write a deployment verdict and never
choose the next self-play model. Log their wall time separately so the 50-pair
cadence does not disappear from throughput accounting.

After Run 1, select 12 completed cycle models at evenly spaced cycle indices,
including the first and final completed cycles. Run the command above for all 66
unordered model pairs with the same binary and arguments. Do not reverse a pair
manually: each arena invocation already plays every opening with both colors.

Create a hash-bound input manifest. Paths are resolved relative to this JSON;
repeat the descriptor for every pair log:

```json
{
  "schema": "alphamini.arena-ladder-input.v1",
  "prior_sigma_elo": 800.0,
  "pair_logs": [
    {
      "player_a": "cycle-000005",
      "player_b": "cycle-000010",
      "model_a_sha256": "64-lowercase-hex-from-model-a-manifest",
      "model_b_sha256": "64-lowercase-hex-from-model-b-manifest",
      "path": "checkpoint-pairs/cycle-000005-v-cycle-000010.jsonl",
      "sha256": "64-lowercase-hex-from-sha256sum-of-pair-log"
    }
  ]
}
```

Extract hashes without copying them from terminal output by eye:

```bash
python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_sha256"])' \
  "$CHALLENGER_MANIFEST"
sha256sum "$PAIR_LOG"
```

Then validate every log and fit the frozen draw-as-half-win Bradley–Terry model:

```bash
uv run --project alphamini-train alphamini-train ladder \
  --input "$ALPHAMINI_RUN_DIR/evaluation/checkpoint-arena-input.json" \
  --output "$ALPHAMINI_RUN_DIR/evaluation/bradley-terry.json"
```

The consumer rejects a torn/incomplete log, checksum or model-identity mismatch,
duplicate pairing, disconnected graph, changed opening/search/binary identity,
or malformed game record. It derives W/D/L directly from the paired games and
records every verified source hash in its output. The fitter sorts player IDs,
uses a fixed zero-mean Gaussian prior with the input's 800-Elo sigma, requires
optimizer convergence, and reports mean-zero Elo, approximate standard errors,
likelihood, games, and iterations. This relative ladder is separate from
full-game arena Elo and Chess.com move-quality calibration.

The older `alphamini.ladder-input.v1` aggregate-count schema remains available
for imported historical results, but the native Run 1 path is the verified
dual-model JSONL schema above.

## 10. Run the frozen milestone rungs

Use one candidate model for all four rungs. The committed suite checksum, seed,
bootstrap, search/time limits, batch size, exact PUCT/FPU millionths, ply limit,
and Minimax digest are
validated by the arena. Each opening is played twice with colors reversed.
Interrupted `*.jsonl` files resume; verdict files are created only after all
pairs finish and are never overwritten.

```bash
export CANDIDATE_MODEL=/absolute/path/to/model-<hash>.onnx
export CANDIDATE_MANIFEST=/absolute/path/to/model-<hash>.json
export RUNG_DIR="$ALPHAMINI_RUN_DIR/evaluation/frozen-rungs"
mkdir -p "$RUNG_DIR"
cargo build --locked --release -p arena --features alphamini
```

Random, 100 opening pairs:

```bash
target/release/arena \
  --alphamini-model "$CANDIDATE_MODEL" \
  --alphamini-manifest "$CANDIDATE_MANIFEST" \
  --opponent random --openings arena/openings/alphamini-v1.json \
  --games 100 --seed 1 --max-plies 1000 --bootstrap 20000 \
  --alphamini-simulations 10000 --alphamini-time-ms 9000 \
  --alphamini-batch-size 8 --require-lower-score 0.5 \
  --results "$RUNG_DIR/random-pairs.jsonl" \
  --verdict "$RUNG_DIR/random-verdict.json"
```

`MinimaxDepth1V1`, 100 opening pairs:

```bash
target/release/arena \
  --alphamini-model "$CANDIDATE_MODEL" \
  --alphamini-manifest "$CANDIDATE_MANIFEST" \
  --opponent minimax --depth 1 --openings arena/openings/alphamini-v1.json \
  --games 100 --seed 1 --max-plies 1000 --bootstrap 20000 \
  --alphamini-simulations 10000 --alphamini-time-ms 9000 \
  --alphamini-batch-size 8 --require-lower-score 0.5 \
  --results "$RUNG_DIR/depth1-pairs.jsonl" \
  --verdict "$RUNG_DIR/depth1-verdict.json"
```

`MinimaxDepth2V1`, 100 opening pairs:

```bash
target/release/arena \
  --alphamini-model "$CANDIDATE_MODEL" \
  --alphamini-manifest "$CANDIDATE_MANIFEST" \
  --opponent minimax --depth 2 --openings arena/openings/alphamini-v1.json \
  --games 100 --seed 1 --max-plies 1000 --bootstrap 20000 \
  --alphamini-simulations 10000 --alphamini-time-ms 9000 \
  --alphamini-batch-size 8 --require-lower-score 0.5 \
  --results "$RUNG_DIR/depth2-pairs.jsonl" \
  --verdict "$RUNG_DIR/depth2-verdict.json"
```

`MinimaxDepth3V1`, 200 opening pairs and the only verdict the server accepts:

```bash
target/release/arena \
  --alphamini-model "$CANDIDATE_MODEL" \
  --alphamini-manifest "$CANDIDATE_MANIFEST" \
  --opponent minimax --depth 3 --openings arena/openings/alphamini-v1.json \
  --games 200 --seed 1 --max-plies 1000 --bootstrap 20000 \
  --alphamini-simulations 10000 --alphamini-time-ms 9000 \
  --alphamini-batch-size 8 --require-lower-score 0.5 \
  --results "$RUNG_DIR/depth3-pairs.jsonl" \
  --verdict "$RUNG_DIR/gate-verdict.json"
```

A missed lower bound is an expected nonzero command exit, not permission to
change the suite or limits. Preserve the verdict and pair log, publish all rung
results, and continue the same run with `extend` if desired. A passing Depth-3
verdict is the only verdict the server will accept, but deployment does not
require one; the rungs are model-comparison evidence rather than a release step.

## 11. Provision the deployment artifact

Stage a new immutable release directory, verify its hash/contract binding, then
switch the `current` symlink atomically. A gate verdict is optional here: include
one only if the Depth-3 command reported `PASSED` for this exact model, in which
case the server will validate it and serve at the frozen budget it certifies.

```bash
export MODEL_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_sha256"])' "$CANDIDATE_MANIFEST")"
export RELEASES_DIR="$PWD/artifacts/alphamini/releases"
export RELEASE_DIR="$RELEASES_DIR/$MODEL_SHA"
mkdir -p "$RELEASES_DIR"
export STAGE_DIR="$(mktemp -d "$RELEASES_DIR/.stage.XXXXXX")"

install -m 0444 "$CANDIDATE_MODEL" "$STAGE_DIR/model.onnx"
install -m 0444 "$CANDIDATE_MANIFEST" "$STAGE_DIR/manifest.json"
chmod 0755 "$STAGE_DIR"

cargo build --locked --release -p alphamini --features onnx
ALPHAMINI_MODEL_PATH="$STAGE_DIR/model.onnx" \
ALPHAMINI_MANIFEST_PATH="$STAGE_DIR/manifest.json" \
  target/release/alphamini --verify-only

mv -Tn "$STAGE_DIR" "$RELEASE_DIR"
if test -e "$STAGE_DIR"; then
  echo "release already exists; staging directory was not published" >&2
  exit 1
fi
ln -s "releases/$MODEL_SHA" "$PWD/artifacts/alphamini/.current-$MODEL_SHA"
mv -Tf "$PWD/artifacts/alphamini/.current-$MODEL_SHA" \
  "$PWD/artifacts/alphamini/current"
```

`mv -Tn` refuses to replace an existing release; the following existence check
turns GNU `mv`'s no-clobber no-op into a hard failure. Inspect the existing
immutable release instead of overwriting it.
The final directory has `model.onnx` and `manifest.json`, matching the service
and deployment script. To ship a gate verdict alongside them, install it as
`gate-verdict.json` in the same staging step and add
`ALPHAMINI_GATE_PATH="$STAGE_DIR/gate-verdict.json"` to the `--verify-only`
invocation above; the server then refuses a failed, wrong-model, or non-Depth-3
verdict, and requires the frozen search budget. On the target host, run
`scripts/deploy.sh`; it rebuilds the server and repeats `--verify-only` against
the model and manifest.

## 12. Execute controlled recovery and CPU serving drills

Run these after the implementation has stopped changing and before Run 1. All
paths below are Git-ignored; the drill command refuses a nonignored in-worktree
run or evidence path because writing it would invalidate the source digest that
the run is trying to verify. Use new paths for every attempt.

The recovery config is deliberately bounded: a 16-channel/one-block network,
24 games at 16 simulations, and one training cycle with checkpoints every 25
successful updates. It is operational recovery evidence, not v1 throughput or
strength evidence. First produce an uninterrupted control:

```bash
export ALPHAMINI_CLI="$PWD/alphamini-train/.venv/bin/alphamini-train"
export RECOVERY_CONFIG="$PWD/configs/alphamini/recovery-drill.toml"
export RECOVERY_CONTROL="$PWD/runs/alphamini-recovery-control-001"

"$ALPHAMINI_CLI" start \
  --config "$RECOVERY_CONFIG" \
  --run-dir "$RECOVERY_CONTROL" \
  --one-cycle
"$ALPHAMINI_CLI" verify --run-dir "$RECOVERY_CONTROL" --deep
```

Then exercise a live collection interruption. The driver starts only its own
child orchestrator, waits for a running collection invocation, sends that PID a
controlled `SIGTERM`, requires exit 130 and a retained active session, invokes
normal recovery without `--force`, deep-verifies, resumes one cycle, and
deep-verifies again:

```bash
"$ALPHAMINI_CLI" drill-recovery \
  --phase collection \
  --config "$RECOVERY_CONFIG" \
  --run-dir "$PWD/runs/alphamini-recovery-collection-001" \
  --control-run-dir "$RECOVERY_CONTROL" \
  --evidence "$PWD/artifacts/alphamini/drills/collection-001.json"
```

Collection recovery must record exactly one interruption and quarantine the
unsealed collection directory. The regenerated collection must bind the same
seed, game range, simulations, and ply cap and pass full replay validation.
Dynamic CUDA batch composition means its shard hash is recorded and compared
with the control but is not required to be byte-identical.

Next exercise recovery from a durable mid-training checkpoint:

```bash
"$ALPHAMINI_CLI" drill-recovery \
  --phase training \
  --config "$RECOVERY_CONFIG" \
  --run-dir "$PWD/runs/alphamini-recovery-training-001" \
  --control-run-dir "$RECOVERY_CONTROL" \
  --evidence "$PWD/artifacts/alphamini/drills/training-001.json"
```

The training drill waits until `RECOVERY` names a real periodic checkpoint
before signaling. It requires the control and drill to have the same config,
source, locks, runtime, raw shards, and tensor payloads, then compares final
model, optimizer, AMP scaler, RNG, sampler/training state, cumulative losses,
validation losses, schedule step, and ONNX bytes exactly. Process-segment wall
time and throughput diagnostics are retained but deliberately excluded from
the equality claim. A mismatch fails the command and remains recorded in the
immutable evidence JSON and sibling stdout/stderr files.

Finally, measure the CPU search budget with the production-shaped model chosen
for the benchmark. Build the feature-enabled arena once and use a fresh output
directory:

```bash
cargo build --locked --release -p arena --features alphamini
export CPU_MODEL=/absolute/path/to/production-shaped-model.onnx
export CPU_MANIFEST=/absolute/path/to/production-shaped-model.json

"$ALPHAMINI_CLI" benchmark-cpu \
  --arena "$PWD/target/release/arena" \
  --model "$CPU_MODEL" \
  --manifest "$CPU_MANIFEST" \
  --openings "$PWD/arena/openings/alphamini-v1.json" \
  --output-dir "$PWD/artifacts/alphamini/cpu-serving-001"
```

This runs the first two balanced opening pairs for two plies after each opening,
so AlphaMini makes four moves per batch setting. It uses exploratory arena mode
at the serving limits—10,000 simulations, 9,000 ms, batches 1/4/8—and emits the
hash-bound pair logs, process logs, and `summary.json`. Each setting records its
own pass/fail result. The command's production verdict is bound to batch 8,
which is also frozen in the server, service unit, and release gate; batches 1
and 4 are retained as scaling diagnostics. A setting passes only if all four
moves finish all simulations without a deadline and never exceed the requested
batch. This measures engine search time, not HTTP transport or Elo; retain a
separate `/move` API smoke test before deployment.
