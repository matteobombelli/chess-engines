# AlphaMini Rust vertical slice

This workspace crate owns the versioned chess-to-network contracts, PUCT
search, raw self-play records, an HTTP move service, and command-line tools.

## Runtime contract

- Input: FP32 NCHW `[batch, 22, 8, 8]`, canonicalized for side to move.
- Policy: `[batch, 73, 8, 8]` / 4,672 flattened plane-major logits using
  `policy-v1`.
- Value: three W/D/L logits, ordered win, draw, loss for side to move.
- Rust generates and validates every legal move. A model can only rank that set.
- Production defaults to fixed `cpuct=1.5`, root-unreduced FPU, no noise, and
  max-visit selection. Self-play optionally adds 25% Dirichlet(0.3) noise.

The default build includes a deterministic uniform evaluator for tests and
fixtures. Build with `--features onnx` to enable the CPU ONNX backend:

```sh
cargo test --manifest-path alphamini/Cargo.toml
cargo test -p alphamini --features onnx
cargo run -p alphamini --features onnx \
  --bin alphamini -- --model model.onnx --manifest manifest.json
```

Self-play builds use `--features cuda` and select the CUDA execution provider
with failure-on-unavailable semantics; they never fall back to CPU. The server
uses the CPU session deliberately. CUDA requires an ONNX Runtime build and
CUDA/cuDNN versions compatible with the host driver.
The pinned Linux prebuilt is currently CUDA 13; `doctor` must create a session
and execute an inference on the target 3070 rather than trusting compilation.

The optional backend pins `ort` 2.0.0-rc.13. Feature `onnx` uses its downloaded
CPU runtime for serving; feature `cuda` additionally selects the CUDA-enabled
distribution and registers CUDA with error-on-failure. There is no silent CPU
or uniform fallback during CUDA self-play: model, checksum, provider, or session
errors fail startup. CUDA sessions also disable ONNX Runtime's per-operation CPU
fallback, so model loading fails unless the complete optimized graph can be
placed on the requested non-CPU execution provider. Output tensors are selected
by manifest names.

From the repository root, install and verify the isolated, hash-locked CUDA
13.2 runtime before any CUDA collector or inference probe:

```sh
scripts/alphamini-cuda-runtime setup
scripts/alphamini-cuda-runtime verify
```

The stage lives at `target/alphamini-cuda-runtime/13.2` and does not modify the
Torch cu126 virtual environment. Its `manifest.json` records the CUDA wheel and
runtime-library hashes plus the exact ORT rc.13 distribution/static archive and
provider hashes. The command rejects stale provider cache symlinks and any
unresolved `ldd` dependency. To run an ad-hoc Rust CUDA command with the same
verified environment used by the orchestrator:

```sh
scripts/alphamini-cuda-runtime exec -- \
  cargo run --locked --release -p alphamini --bin alphamini-inference \
  --features cuda -- --model model.onnx --manifest manifest.json \
  --device cuda --cuda-device 0
```

Before training or deployment, run the fixed full-tensor parity probe. It emits
one JSON object containing Rust's exact `[1,22,8,8]` golden input, its
little-endian f32 SHA-256, all 4,672 plane-major policy logits, and all three
W/D/L logits. Feed the emitted `input_values` to PyTorch and Python ONNX Runtime
and compare both outputs to Rust with the tolerances frozen by `doctor`:

```sh
cargo run -p alphamini --release --features onnx \
  --bin alphamini-inference -- \
  --model model.onnx --manifest manifest.json --device cpu > parity-cpu.json

cargo run -p alphamini --release --features cuda \
  --bin alphamini-inference -- \
  --model model.onnx --manifest manifest.json --device cuda --cuda-device 0 \
  > parity-cuda.json
```

The output schema is `alphamini-inference-parity-v1`. CUDA selection is
explicit and fail-closed; requesting it from an `onnx`-only build is an error.
The frozen little-endian input digest is
`a3c8eb105e9af08a4bb13315141f289af83f1ebfc9059ca6c19070a6f6976d7a`;
Rust refuses to emit a report if encoder drift changes it.

## Self-play integration

The collector accepts flags or the orchestration environment variables:

```sh
cargo run -p alphamini --features cuda \
  --bin alphamini-selfplay -- collect \
  --model artifacts/model.onnx --manifest artifacts/model.json \
  --run-dir runs/run-001 --run-id 00000000-0000-0000-0000-000000000001 \
  --cycle-id 1 --game-id-start 0 --config-sha256 <64-hex-sha256> \
  --output-dir runs/run-001/collection/cycle-000001 \
  --collection-manifest runs/run-001/collection/cycle-000001/manifest.json \
  --games 1024 --simulations 128 --batch-size 128 --seed 1
```

The collector recognizes `ALPHAMINI_RUN_DIR`, `ALPHAMINI_RUN_ID`,
`ALPHAMINI_CYCLE_ID`, `ALPHAMINI_GAME_ID_START`, `ALPHAMINI_CONFIG_SHA256`,
`ALPHAMINI_CONFIG_JSON`, `ALPHAMINI_MODEL_PATH`, `ALPHAMINI_MANIFEST_PATH`,
`ALPHAMINI_INFERENCE_DEVICE`, `ALPHAMINI_COLLECTION_DIR`, and
`ALPHAMINI_COLLECTION_MANIFEST`, plus env forms of the five search/noise flags.
Each sealed
`.msgpack.zst` shard is immutable and checksum-addressed by its collection
manifest. Every position stores the selected UCI move alongside its sparse
visit policy; sealing replays the complete trajectory and checks states, move
linkage, repetition counts, the exact simulation/visit budget, and the final
label. Shards and their collection manifest both bind the base seed,
simulation count, and `max_plies` (`1..=512`). Each game seed is recomputed as
frozen SplitMix64 of `collection_seed XOR game_id`, and an ongoing game is a
valid ply-limit draw only after exactly the bound cap. Collection writes use a
same-directory temporary file followed by atomic no-replace publication.
Existing targets are rejected rather than overwritten.

Search converts a request boundary once and uses reversible `SearchPosition`
make/unmake thereafter. Self-play keeps one search position per active game and
routes one-leaf requests through a centralized inference batcher, without
changing `encoder-v1`, `policy-v1`, or `PositionRecordV1`. Collection uses a
bounded rolling pool of `min(game_count, 2 * inference_batch_size)` long-lived
workers. Each worker atomically claims the next `(game_id, seed)` in input order
as soon as its current game ends, so finished games are replenished until only
the final pool remains; it never creates one operating-system thread per game.
The rolling schedule spans the whole requested collection. Sorted completed
records are partitioned into the configured immutable physical shards only
after search, so a shard boundary cannot drain and restart the GPU cohort.
Scheduler telemetry is consequently emitted once per collection (under the
legacy `self_play_shard_complete` event name with
`telemetry_scope="collection"`) and must not be multiplied by physical shard
count.

Game IDs and RNG streams are stable. Dynamic cross-game GPU batch composition
depends on thread scheduling, however, and floating-point kernels may vary with
batch shape; regenerated CUDA shards are therefore not promised to be
byte-for-byte identical. A failed unsealed collection is quarantined and
regenerated in full, and exact artifact hashes—not filenames—control admission.
