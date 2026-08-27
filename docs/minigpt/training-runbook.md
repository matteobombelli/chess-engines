# MiniGPT training runbook

This is the operational source of truth. Run commands from the repository root
on Linux with Python 3.12. Do not edit a run directory by hand.

The current implementation state is in [`status.md`](status.md); the frozen
interfaces the steps below assume are in [`design.md`](design.md).

Two environment variables are used throughout:

```bash
export MINIGPT_RUN_DIR="$PWD/runs/minigpt-v1"
export MINIGPT_DUMPS="$PWD/data/minigpt/dumps"
```

## 1. Download the Lichess dumps

Fetch the monthly standard-rated dumps from
<https://database.lichess.org/> into `$MINIGPT_DUMPS`. v1 used
`lichess_db_standard_rated_2026-06.pgn.zst` and the 2026-07 dump.

```bash
mkdir -p "$MINIGPT_DUMPS"
curl -L -o "$MINIGPT_DUMPS/lichess_db_standard_rated_2026-06.pgn.zst" \
  https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst
```

These are tens of gigabytes compressed. Keep them outside Git — the repository
`.gitignore` already excludes `data/`. Verify each download against the
published checksum before ingest; ingest hashes the whole compressed file
anyway and records that digest in the shard manifest, but a truncated download
is cheaper to catch here.

## 2. Ingest the dumps into token shards

`minigpt-ingest` streams zstd, applies the tag-only filters, replays and
tokenizes the survivors on a worker pool, and writes shards plus `shards.json`.
Dumps are read in the order given.

```bash
cargo build -p minigpt --release --bin minigpt-ingest

./target/release/minigpt-ingest \
  --dump "$MINIGPT_DUMPS/lichess_db_standard_rated_2026-06.pgn.zst" \
  --dump "$MINIGPT_DUMPS/lichess_db_standard_rated_2026-07.pgn.zst" \
  --out data/minigpt/shards \
  --min-elo 2000 --min-plies 10 --max-plies 300 \
  --token-target 1000000000 --val-fraction 0.005 \
  --shard-tokens 50000000
```

The defaults are the v1 values and are spelled out above so the command is a
record of what was run. `--token-target` stops decoding once the accepted token
count is reached; the source digest still covers the whole file. Progress is
printed every ten seconds.

When it finishes, read the manifest before trusting it:

```bash
python3 -m json.tool data/minigpt/shards/shards.json | head -60
```

Check three things: `counts.games_seen` equals `games_accepted` plus the sum of
`rejected`, the reject reasons are distributed the way you expect (a spike in
`san_error` means the replay path is broken, not the corpus), and
`san_error_samples` is empty or short. Only then delete the dumps — see
step 8.

## 3. Doctor

```bash
uv python install 3.12
uv sync --project minigpt-train --extra train --extra test --locked

uv run --project minigpt-train minigpt-train doctor \
  --config configs/minigpt/pilot.toml
```

Before the published run, use the stricter form and keep its evidence:

```bash
mkdir -p artifacts/minigpt/doctor
set -o pipefail
uv run --project minigpt-train minigpt-train doctor \
  --config configs/minigpt/v1.toml --production | \
  tee artifacts/minigpt/doctor/v1-production.json
```

`doctor` validates the config against `minigpt.config.v1`, opens the shard
manifest and verifies every shard digest and file size, checks Python and the
ML runtime, checks Git cleanliness, exercises atomic rename and fsync, and
checks free disk against `training.disk_floor_bytes`. In `--production` mode
every optional warning becomes a failure. The command exits non-zero if any
check failed; fix all of them before starting.

Keep run data on a local filesystem supporting advisory locks and atomic
same-directory rename.

## 4. Start the run

A published run requires a committed, clean worktree at the commit you intend
to freeze. Commit first, then:

```bash
uv run --project minigpt-train minigpt-train start \
  --config configs/minigpt/v1.toml \
  --run-dir "$MINIGPT_RUN_DIR"
```

`start` creates the run directory, freezes the config (both hashes), the shard
manifest digest, the git identity including the worktree content digest, the
runtime fingerprint, and both lockfiles; publishes the seeded step-0
checkpoint; and then trains segment by segment until the active-time budget is
exhausted or the horizon is reached.

Useful variants:

- `--initialize-only` publishes the seeded step 0 and stops, so the ledger can
  be inspected before any GPU time is spent.
- `--one-segment` runs exactly one segment and exits, which is the safe way to
  smoke-test a fresh configuration.

**Do not edit the worktree once the run has started.** See step 6.

## 5. Monitor without changing state

The active command holds the advisory lock, so monitoring reads files directly:

```bash
uv run --project minigpt-train minigpt-train report \
  --run-dir "$MINIGPT_RUN_DIR"

python3 -m json.tool "$MINIGPT_RUN_DIR/ACTIVE_SESSION.json"
tail -n 3 "$MINIGPT_RUN_DIR/metrics.jsonl"
nvidia-smi
```

`report` renders a factual summary from the immutable ledger: lineage hashes,
phase, steps completed against the horizon, measured steps per second and the
estimate that follows from it, counted active time, best validation loss and
perplexity with the step they occurred at, and the published models. It leaves
unknowns unknown and never infers success from training loss.

`metrics.jsonl` is append-only, one `minigpt.metrics.v1` object per evaluation
(every `eval_interval_steps`, 500 in v1). The fields the training figures are
built from are `step`, `train_loss`, `validation_loss`, `validation_top1`, and
`learning_rate`; the same record also carries `validation_perplexity`,
`tokens_per_second`, `vram_bytes`, `free_disk_bytes`, `segment_index`, and the
running best.

`ACTIVE_SESSION.json` holds the PID and the last durable heartbeat. Pointer
files under `pointers/` contain SHA-256 object names, not mutable state.
`.partial` files are never inputs.

## 6. Recover and resume

For a planned stop, send Ctrl-C or SIGTERM **once**. The CLI converts either
signal into a controlled unwind and retains `ACTIVE_SESSION.json`. Do not send
SIGKILL for a planned pause, and do not immediately invoke `resume` — go
through recovery. Never delete a lock, pointer, or partial file to make
progress.

After any abnormal exit — power loss, kill, OOM, nonzero worker exit — the
active-session file is intentionally retained. Confirm the recorded process is
gone, then recover, verify, and resume:

```bash
python3 -m json.tool "$MINIGPT_RUN_DIR/ACTIVE_SESSION.json"
ps -p "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' \
  "$MINIGPT_RUN_DIR/ACTIVE_SESSION.json")"

uv run --project minigpt-train minigpt-train recover \
  --run-dir "$MINIGPT_RUN_DIR"
uv run --project minigpt-train minigpt-train verify \
  --run-dir "$MINIGPT_RUN_DIR" --deep
uv run --project minigpt-train minigpt-train resume \
  --run-dir "$MINIGPT_RUN_DIR"
```

`recover` closes the crashed session at its last durable heartbeat, so active
time is never credited past evidence. `verify --deep` re-reads and re-hashes
the artifacts the pointers name.

> **`resume` requires a clean worktree at the exact frozen commit.** The
> segment boundary re-derives `worktree_sha256` over every tracked path and
> compares it to the run manifest, and separately requires `tracked_dirty` to
> be exactly `False`. Any repository edit — including one to a documentation
> file, a config unrelated to this run, or a frontend source file — stalls the
> run at the next segment boundary with an identity error. The fix is to make
> the tree byte-identical again (`git stash`, or `git stash push --include-untracked`
> for new files), then `recover` and `resume`. No training is lost, but the
> stalled segment's time is.

Do work in a separate `git worktree` while a run is in flight, not in the
training checkout.

`extend` appends active-time budget without changing identity, and is accepted
only after the current budget is exhausted:

```bash
uv run --project minigpt-train minigpt-train extend \
  --run-dir "$MINIGPT_RUN_DIR" \
  --additional-active-budget 24h --reason "horizon not reached at 72 h"
```

Any learning-affecting change — architecture, horizon, schedule, decode
temperature, determinism flags — is a `fork`, not a `resume`.

## 7. Export and verify

Export publishes the **best-validation** checkpoint, not the last one:

```bash
uv run --project minigpt-train minigpt-train export \
  --run-dir "$MINIGPT_RUN_DIR"
```

It traces FP32 ONNX at opset 17 with input `tokens` and output `logits`, then
compares PyTorch against ONNX Runtime CPU at sequence lengths
`{1, 4, 64, 256}` under the configured `parity_atol` (1e-3) and `parity_rtol`
(0). A failed comparison raises and publishes nothing. On success it writes the
content-addressed `model-<digest>.onnx`, its `minigpt.manifest.v1` sidecar, and
a separate training-provenance record, then republishes the pair atomically as
`artifacts/minigpt/current/model.onnx` and `manifest.json`. Use `--publish-dir`
to publish elsewhere.

Then verify the whole ledger once more:

```bash
uv run --project minigpt-train minigpt-train verify \
  --run-dir "$MINIGPT_RUN_DIR" --deep
uv run --project minigpt-train minigpt-train reproduce \
  --run-dir "$MINIGPT_RUN_DIR"
```

`reproduce` emits the exact run identity and the commands needed to replay it.

## 8. Disk safety

The corpus and the run compete for the same disk, and running it out is the
failure mode that costs the most time.

**Checkpoint retention.** The run keeps the union of three sets: the last
`checkpoint_keep_last` checkpoints (2 in v1), every milestone checkpoint
(`checkpoint_milestone_every_steps`, 5,000 in v1), and the best-validation
checkpoint. Nothing else is retained, and nothing in that union is ever a GC
candidate. At 40.3M parameters a checkpoint carries model plus optimizer plus
RNG state, so the milestone set is the part that grows with the horizon —
budget for `total_steps / 5000` of them.

**Disk floor.** `training.disk_floor_bytes` is 50 GiB (53,687,091,200). Doctor
fails below it and the trainer refuses to write below it rather than filling
the volume. `free_disk_bytes` in every metrics record is the early warning;
watch it trend, not just its current value.

**Garbage collection.** Preview first, always:

```bash
uv run --project minigpt-train minigpt-train gc \
  --run-dir "$MINIGPT_RUN_DIR"
```

It proposes only superseded non-milestone checkpoints and unsealed `.partial`
files, and never the current, best, or recovery checkpoints, the milestone set,
the object chain, or the shards. Apply only after reading the list:

```bash
uv run --project minigpt-train minigpt-train gc \
  --run-dir "$MINIGPT_RUN_DIR" --apply
```

**Delete the dumps after the shards verify.** The compressed dumps are the
largest single item on disk and are re-downloadable; the shards are not
(cheaply). Delete them only once step 2's manifest checks pass and `doctor` has
verified every shard digest — after that the dumps contribute nothing but their
recorded SHA-256.

**Reclaim WSL disk.** Deleting files inside WSL does not shrink the backing
VHDX. From an elevated Windows PowerShell, with WSL shut down:

```powershell
wsl --shutdown
Optimize-VHD -Path "$env:LOCALAPPDATA\Packages\<distro>\LocalState\ext4.vhdx" `
  -Mode Full
```

Do this after deleting the dumps, not while a run is active.
