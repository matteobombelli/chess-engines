# Chess Engines

This repository is a learning project: build several chess engines yourself, understand why they work, train the learned ones from scratch, and put each engine behind the same playable web interface.

## Run the existing baseline

Install the Rust toolchain and [Trunk](https://trunkrs.dev/), then use two terminals:

```sh
# Terminal 1: API
cargo run -p random

# Terminal 2: web UI
cd frontend
trunk serve --open
```

The current API accepts the whole game as PGN movetext:

```http
POST http://127.0.0.1:3000/move
Content-Type: application/json

{"san":"1. e4 e5 2. Nf3"}
```

It returns the bot's legal SAN move and the resulting FEN. An absent or empty `san` means the game has not started and the bot moves first.

One field, one source of truth: the bot rebuilds the position by replaying the
movetext through `chess-core`, so there is no FEN that could contradict the
moves, and repetition state remains exact. The replay is strict: move numbers
and result tokens are dropped, and every other token must be a legal move at the
ply it appears, so a game is never silently replayed as a different one.

Before adding an ML model, run the repository tests:

```sh
cargo test --workspace
```

## Evaluate bot strength

The `arena` crate plays Minimax and Random directly in-process, alternates
colors, checks every move for legality, and reports their win/draw/loss record,
match score, and relative Elo with an approximate 95% interval:

```sh
cargo run -p arena --release --bin arena -- --games 200 --depth 3 --seed 1
```

The seed makes the Random moves reproducible. Keep the game count, Minimax
depth, seed, and maximum plies fixed when comparing code changes. Use more than
one seed and substantially more games for a result you intend to publish.

This is a **relative** rating: the report defines Random as 0 Elo and estimates
Minimax's difference from it. Two bots cannot establish an absolute human or
online-platform Elo. That requires a calibrated pool of reference engines.

Deterministic comparisons use the committed, balanced opening suite and play
every prefix twice with colors reversed. AlphaMini evaluation additionally
bootstraps whole opening pairs, writes a resumable JSONL record, and freezes the
model, opponent, search, suite, and binary identities.

The ratings shown on the site come from full games against Stockfish 17.1 held
at fixed limited-strength levels, fitted by maximum likelihood with a bootstrap
95% interval. The method, results, and their limits are in
[docs/ratings/full-game-elo.md](docs/ratings/full-game-elo.md). Time-budgeted
bots were measured through the production API so the numbers describe the
deployed hardware.

The `calibrate` crate is the older, superseded method: it compares bot move
quality with rated humans in public Chess.com games at exactly 30+0. See
[`calibrate/README.md`](calibrate/README.md) for what it measures and its
limits. Its estimates remain valid for single-move quality, and the gap between
the two methods for MiniGPT is discussed in the full-game document.

## Minimax

The `minimax` crate has a guided alpha-beta search scaffold in
`minimax/src/search.rs`. See `minimax/README.md` for the implementation order.

## AlphaMini

`alphamini` is a compact AlphaZero-style CNN/PUCT engine. Rust owns rules,
encoding, legal masks, MCTS, concurrent GPU-batched self-play, immutable raw
shards, and CPU ONNX serving. Python owns the residual CNN, replay sampling,
optimization, crash-safe checkpoints, ONNX export, and the run ledger.

Start with the [current status and continuation handoff](docs/alphamini/status.md),
then read the [design](docs/alphamini/design.md) and follow the
[training runbook](docs/alphamini/training-runbook.md). The v1 run is complete:
72.09 hours of cumulative active compute across a three-directory weights-only
lineage, at inference batch 256, with the resulting model served in production.
Its lineage, incidents, and evaluation are in the
[run 003 result](docs/alphamini/results/run-003.md).

In full games against Stockfish at fixed strength levels, Depth-3 rates 1698
(95% CI 1627 to 1766), the deployed 9-second search rates 2057 (95% CI 1961 to
2142), and AlphaMini rates 2289 (95% CI 2229 to 2353), the two time-budgeted
bots measured through the production API. The older move-quality calibration
put AlphaMini near 1970; full games rate it higher. All current numbers live in
[docs/ratings/full-game-elo.md](docs/ratings/full-game-elo.md).

## MiniGPT

`minigpt` is a 40.3M-parameter decoder-only GPT that plays chess by predicting
the next move, with no search. It is trained on 11.2 million strong Lichess
games (both players at least 2000, Blitz/Rapid/Classical, cleanly terminated),
tokenized as `BOS` plus one move token per ply over the same `policy-v1` action
space AlphaMini uses.

Rust owns rules, SAN replay, corpus ingest, token shards, CPU ONNX serving, and
decoding. Python owns the transformer, optimization, crash-safe checkpoints,
ONNX export, and the run ledger. The model produces one distribution per
position and a **legality mask** filters it, so like every other engine here it
can only ever play a move `chess-core` already generated.

Start with the [current status and continuation handoff](docs/minigpt/status.md),
then read the [design](docs/minigpt/design.md) and follow the
[training runbook](docs/minigpt/training-runbook.md). The dated engineering
record, including the corpus measurements and the pilot, is in the
[implementation log](docs/minigpt/implementation-log.md). The v1 run is
complete and the model is deployed; its training result is in the
[run 001 result](docs/minigpt/results/run-001.md) and its full-game rating of
1395 (95% CI 1322 to 1455) is in
[docs/ratings/full-game-elo.md](docs/ratings/full-game-elo.md).

## Deploy

From the production checkout, fetch, fast-forward, and deploy the exact commit
currently on `origin/main` with:

```sh
./scripts/pull-and-deploy.sh
```

This command refuses dirty checkouts, non-`main` branches, divergent history,
and local-only commits. It never creates a merge commit or rewrites production
history. For a build of the already-checked-out commit without fetching, use
`./scripts/deploy.sh` directly.

The active Caddy site must include the four namespaced API handlers from
`deploy/caddy/chessengines.caddy` before its Chess Engines static-file handler.
After changing Caddy, validate and reload it once:

```sh
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

The deploy script checks those routes before building or changing production,
so an outdated proxy configuration cannot leave a partially deployed release.
Set `CHESSENGINES_CADDY_CONFIG` if the active Caddyfile lives elsewhere.

The script runs the workspace tests, builds all four bot APIs and the frontend
in release mode, validates AlphaMini's ONNX model and manifest, publishes the
frontend to `/srv/chessengines`, restarts the four user services, and
smoke-tests all routes.

Production service and reverse-proxy templates live under `deploy/`. Random,
Minimax, AlphaMini, and MiniGPT listen on ports 3002, 3004, 3006, and 3008.
AlphaMini serves FP32 ONNX on CPU with one bounded search at a time and the
frozen 9-second, 10,000-simulation, batch-8 release budget. Its immutable
`model.onnx` and `manifest.json` live outside Git under
`artifacts/alphamini/current`.

## Working rule

Every new engine must preserve one invariant: the Rust boundary generates the legal moves and only applies a move from that set. A model supplies scores or preferences; it never gets authority to invent a move and alter the board directly.

Generated data and checkpoints can become enormous. Keep raw downloads, processed shards, experiment runs, and model artifacts outside Git. Commit small configs, manifests, plots, and evaluation summaries instead.
