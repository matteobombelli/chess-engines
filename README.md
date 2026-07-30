# ChessBots

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

One field, one source of truth: the bot rebuilds the position by replaying the movetext through `chess-core`, so there is no FEN that could contradict the moves. It also gives the learned engines the move history they predict from, which a position cannot supply — many move orders reach the same board, and a sequence model has to know which one was played. The replay is strict: move numbers and result tokens are dropped, and every other token must be a legal move at the ply it appears, so a game is never silently replayed as a different one.

Before adding an ML model, run the repository tests:

```sh
cargo test --workspace
```

## Deploy

Deploy the committed `main` working tree to production with:

```sh
./scripts/deploy.sh
```

The script runs the workspace tests, builds the random bot and frontend in release
mode, publishes the frontend to `/srv/chessengines`, restarts the
`chessengines-random` user service, and verifies the live page and API.

## Working rule

Every new engine must preserve one invariant: the Rust boundary generates the legal moves and only applies a move from that set. A model supplies scores or preferences; it never gets authority to invent a move and alter the board directly.

Generated data and checkpoints can become enormous. Keep raw downloads, processed shards, experiment runs, and model artifacts outside Git. Commit small configs, manifests, plots, and evaluation summaries instead.
