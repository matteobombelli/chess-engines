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

## Evaluate bot strength

The `arena` crate plays Minimax and Random directly in-process, alternates
colors, checks every move for legality, and reports their win/draw/loss record,
match score, and relative Elo with an approximate 95% interval:

```sh
cargo run -p arena --release -- --games 200 --depth 3 --seed 1
```

The seed makes the Random moves reproducible. Keep the game count, Minimax
depth, seed, and maximum plies fixed when comparing code changes. Use more than
one seed and substantially more games for a result you intend to publish.

This is a **relative** rating: the report defines Random as 0 Elo and estimates
Minimax's difference from it. Two bots cannot establish an absolute human or
online-platform Elo. That requires a calibrated pool of reference engines.

When the arena is later used for two deterministic bots, give both bots the
same suite of opening positions and play every opening twice with colors
reversed. Random already supplies game-to-game variation in this first matchup.

For a platform-calibrated estimate instead, the `calibrate` crate compares bot
move quality with rated humans in public Chess.com games at exactly 30+0. See
[`calibrate/README.md`](calibrate/README.md) for corpus collection, Stockfish
analysis, statistical fitting, and the limits of that estimate. Chess.com calls
30+0 Rapid, even though it is the slow/classical-style target used here.

## Minimax

The `minimax` crate has a guided alpha-beta search scaffold in
`minimax/src/search.rs`. See `minimax/README.md` for the implementation order.

## Deploy

Deploy the committed `main` working tree to production with:

```sh
./scripts/deploy.sh
```

The script runs the workspace tests, builds both bot APIs and the frontend in
release mode, publishes the frontend to `/srv/chessengines`, restarts the
`chessengines-random` and `chessengines-minimax` user services, and verifies
all three playable bot configurations.

Production service and reverse-proxy templates live under `deploy/`. Random
listens on port 3002 and Minimax listens on port 3004. The Minimax service
offers a fixed depth-3 route and a timed route. The timed version uses a
9-second move budget and a depth ceiling of 64. The time budget normally stops
the search first.

## Working rule

Every new engine must preserve one invariant: the Rust boundary generates the legal moves and only applies a move from that set. A model supplies scores or preferences; it never gets authority to invent a move and alter the board directly.

Generated data and checkpoints can become enormous. Keep raw downloads, processed shards, experiment runs, and model artifacts outside Git. Commit small configs, manifests, plots, and evaluation summaries instead.
