# Chess.com 30+0 calibration

This tool estimates a bot's **Chess.com rated 30+0 move-quality equivalent**.
Chess.com groups 30+0 in its Rapid rating pool; the collector nevertheless
accepts only the exact `TimeControl` value `1800`, so other Rapid controls never
enter the corpus.

The estimate is not produced by playing an engine in rated games. That would
violate Chess.com's Fair Play rules. Instead, the evaluator gives a human and a
bot the same public historical position and asks Stockfish how much expected
score each move lost. It then finds the rating at which human and bot loss are
equal.

## Requirements

Install Stockfish 17 or another recent build that supports `UCI_ShowWDL`. On
Debian/Ubuntu:

```sh
sudo apt-get install stockfish
```

The default executable is `/usr/games/stockfish`; use `--stockfish PATH` for a
different installation.

## 1. Collect a corpus

Raw data belongs in the ignored `data/` directory. Start from several people
who have played rated 30+0 games. The collector follows their 30+0 opponents,
queries archives serially, deduplicates games, and limits every player's total
corpus participation to 20 randomly selected games to reduce player-level
sampling bias.

```sh
cargo run -p calibrate --release -- collect \
  --output data/chesscom-30-0.jsonl \
  --seed-user Nkai20 \
  --max-users 500 \
  --max-games 5000 \
  --user-agent 'chess-engines-calibration/0.1 (contact: you@example.com)'
```

Use multiple `--seed-user` arguments from different rating ranges when
possible. The example username is only a known entry point into the public
30+0 opponent graph, not a representative sample by itself.

For extra coverage around a strong bot's likely crossing point, run a separate
collector with strong 30+0 seed users and `--min-participant-rating 1700`, then
analyze that corpus with matching `--min-rating` and `--max-rating` bounds.

## 2. Analyze each bot

The sampler chooses at most one eligible position per player-color per game,
between plies 12 and 60 by default. Forced moves and positions that Stockfish
already considers essentially decided are excluded. Every reference, human,
and bot move receives the same Stockfish node budget.

```sh
cargo run -p calibrate --release -- analyze \
  --corpus data/chesscom-30-0.jsonl \
  --output runs/random-30-0.json \
  --bot random \
  --nodes 100000 \
  --bot-seed 1 \
  --sample-seed 1

cargo run -p calibrate --release -- analyze \
  --corpus data/chesscom-30-0.jsonl \
  --output runs/minimax-d3-30-0.json \
  --bot minimax \
  --minimax-depth 3 \
  --nodes 100000 \
  --sample-seed 1
```

`--corpus` is repeatable; duplicate game URLs are removed before sampling. For
long timed-engine runs, independent processes can split players without
overlap. Give every process the same inputs and `--shard-count`, and vary only
the zero-based `--shard-index` and output path:

```sh
# Run this once per index 0, 1, 2, and 3 (concurrently if cores are available).
cargo run -p calibrate --release -- analyze \
  --corpus data/chesscom-30-0.jsonl \
  --corpus data/chesscom-30-0-high.jsonl \
  --output runs/minimax-9s-shard-0.json \
  --bot minimax \
  --minimax-time-ms 9000 \
  --minimax-max-depth 64 \
  --nodes 100000 \
  --positions-per-player 1 \
  --shard-count 4 \
  --shard-index 0
```

To measure the production 9-second-per-move configuration instead of a fixed
depth, use a 64-ply iterative-deepening ceiling (the time budget normally stops
it first):

```sh
cargo run -p calibrate --release -- analyze \
  --corpus data/chesscom-30-0.jsonl \
  --output runs/minimax-9s-30-0.json \
  --bot minimax \
  --minimax-time-ms 9000 \
  --minimax-max-depth 64 \
  --nodes 100000 \
  --positions-per-player 3 \
  --analyzed-positions-per-player 1 \
  --sample-seed 1
```

Keep the corpus, sample seed, sampling bounds, and reference node budget fixed
when comparing bot versions. Random's bot seed is separate from the position
sampling seed.

## 3. Fit and inspect the estimate

```sh
cargo run -p calibrate --release -- report \
  --analysis runs/minimax-9s-shard-0.json \
  --analysis runs/minimax-9s-shard-1.json \
  --analysis runs/minimax-9s-shard-2.json \
  --analysis runs/minimax-9s-shard-3.json \
  --bin-width 200 \
  --min-samples 25 \
  --bootstrap 1000
```

The report shows every rating band's human and bot mean expected-point loss,
the equal-quality rating, fit quality, and an interval from resampling whole
human players. This cluster bootstrap avoids pretending that repeated moves by
one person are independent. An estimate outside the populated bands is reported
only as above or below the calibrated range.

Treat low fit quality or a wide interval as a request for more diverse games,
not as a precise rating. This method measures move quality under the bot's
configured search limit; it does not measure human clock management, resigning,
or the distribution of positions the bot creates during full games. The JSON
analysis artifact is intentionally suitable for later plotting or exploratory
modeling in Python.

The first reproducible 1,000-game run is summarized in
[`INITIAL_RESULTS.md`](INITIAL_RESULTS.md). It places Random below the populated
400+ range and fixed-depth-3 Minimax at roughly 1750, with substantial
uncertainty above 2000 due to sparse exact-30+0 data there.

The refined fixed-depth-3 result is in
[`DEPTH_THREE_RESULTS.md`](DEPTH_THREE_RESULTS.md). It replaces the preliminary
Minimax figure with an estimate of about 1675 and a 95% player-bootstrap
confidence interval of 1572 to 1748.

The stronger-player calibration for the deployed 9-second move budget is in
[`NINE_SECOND_RESULTS.md`](NINE_SECOND_RESULTS.md). It estimates roughly 2050
Chess.com 30+0. Its 95% player-bootstrap confidence interval starts at 1889 and
is right-censored at or above 2199.
