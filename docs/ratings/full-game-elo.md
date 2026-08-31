# Full-game Elo ratings

This document is the source of the ratings shown on the site. Each bot played
complete games against Stockfish 17.1 set to fixed limited-strength levels, and
its rating was fitted from the game scores. This supersedes the single-move
calibration in `calibrate/` (see "Relation to the move-quality calibration"
below).

Status: final for all five bots (2026-08-28).

## Method

The tool is the `rate` binary in the `arena` crate.

- Opponent: Stockfish 17.1 (`/usr/games/stockfish`) with `UCI_LimitStrength
  true`, `UCI_Elo` at the rung under test, `Threads 1`, `Hash 16`, and
  `go movetime 100`. At these settings its strength comes from the Elo cap, not
  from time, so the short movetime does not weaken the opponent.
- Games: opening pairs from the committed suite
  `arena/openings/alphamini-v1.json` (eight-ply balanced prefixes), each opening
  played twice with colors swapped. Results append to a durable per-rung JSONL
  log, so any interrupted run resumes at the pair it stopped on.
- Ladder: `rate auto` starts at a seed rung and probes in 10-pair blocks. A
  block score above 85% moves the rung up 150 Elo, below 15% moves it down,
  within the legal `UCI_Elo` range of 1320 to 3190. Once the bot is bracketed by
  informative rungs, blocks accumulate until the 95% interval is narrow enough
  or the game budget runs out.
- Fit: maximum likelihood over all games under the logistic Elo model. The 95%
  interval is a seeded stratified bootstrap (20,000 resamples of opening pairs
  within each rung). A fit or interval endpoint outside the ladder range is
  reported as censored.
- Bots whose strength depends on the machine they run on (the 9-second Minimax
  and AlphaMini, both time-budgeted) were measured through the production API at
  `apps.matteob.dev`, one request at a time, so the ratings describe the exact
  deployed service. The fixed-budget bots (Random, Depth-3, MiniGPT) ran
  locally, where their moves are identical to production.

### What the numbers mean

The scale is anchored to Stockfish's `UCI_Elo`, which Stockfish calibrates
against online human play. It is close to, but not the same thing as, a
Chess.com rating: engines cannot play rated Chess.com games (Fair Play rules),
so no bot rating on this page is directly measurable on Chess.com. Treat the
numbers as approximate Chess.com-scale Elo, good to roughly the width of the
stated intervals, and exactly comparable to each other since every bot faced
the same opponent under the same conditions.

## Results

| Bot | Rating | 95% CI | Games | Basis |
|---|---|---|---|---|
| Random | displayed as below 400 | see note | 240 | shutouts plus cross-play |
| Depth-3 Minimax | 1698 | 1627 to 1766 | 80 | ladder |
| MiniGPT | 1395 | 1322 to 1455 | 160 | ladder |
| 9-second Minimax | 2057 | 1961 to 2142 | 60 | ladder, production API |
| AlphaMini | 2289 | 2229 to 2353 | 160 | ladder, production API |

### Depth-3 Minimax

Seeded at 1650. `runs/full-game-elo/depth3/summary.json`.

| UCI_Elo | W-D-L | Score |
|---|---|---|
| 1650 | 24-2-14 | 62.5% |
| 1800 | 11-2-27 | 30.0% |

Fitted 1698, 95% CI 1627 to 1766, over 80 games. The old move-quality estimate
was 1642; the two methods agree for this bot.

### MiniGPT

Seeded at 1900, the old move-quality label. The ladder walked down three rungs
before it found the bracket. `runs/full-game-elo/minigpt/summary.json`.

| UCI_Elo | W-D-L | Score |
|---|---|---|
| 1450 | 5-2-13 | 30.0% |
| 1600 | 13-4-43 | 25.0% |
| 1750 | 5-4-51 | 11.7% |
| 1900 | 2-1-17 | 12.5% |

Fitted 1395, 95% CI 1322 to 1455, over 160 games. This is roughly 500 points
below the old move-quality figure of 1928. The gap is the method, not noise:
sampled one position at a time, MiniGPT picks strong-looking moves, but over a
whole game a model with no search eventually walks into tactics it cannot see,
and one lost piece against an engine decides the game. Full games price that
in; single moves do not.

### Random

Random is below the ladder floor, so its rating comes from games against the
other bots. `runs/full-game-elo/random/summary.json`.

| Opponent | W-D-L | Score | Result |
|---|---|---|---|
| Stockfish at 1320 | 0-0-20 | 0% | censored, at or below 1320 |
| Depth-3 (1698) | 0-0-100 | 0% | at least 566 below Depth-3, so at or below about 1200 |
| MiniGPT (1395) | 0-7-93 | 3.5% | 819, 95% CI 596 to 977 |

The cross-play fit lands at 819 (95% CI 596 to 977, anchored to MiniGPT), but
that number rests entirely on 7 draws in 100 games, and all 7 were stalemates:
MiniGPT reached winning positions and accidentally stalemated, which says more
about MiniGPT's endgames than about Random's strength. The logistic model is not
credible at a mismatch this extreme, and a random mover loses to any competent
player at any rating. The site keeps the label "below 400", which every
measurement here is consistent with.

### 9-second Minimax

Measured through the production API because its nine-second budget buys more
depth on faster hardware; the deployed machine is what visitors play. Seeded at
2050. `runs/full-game-elo/minimax9s-http/summary.json`.

| UCI_Elo | W-D-L | Score |
|---|---|---|
| 2050 | 21-1-18 | 53.8% |
| 2200 | 5-0-15 | 25.0% |

Fitted 2057, 95% CI 1961 to 2142, over 60 games. The old move-quality estimate
was 2030, displayed as 2050; the two methods agree for this bot.

### AlphaMini

Also time-budgeted (nine seconds or 10,000 simulations, whichever ends first)
and also measured through the production API. Seeded at 1950.
`runs/full-game-elo/alphamini-http/summary.json`.

| UCI_Elo | W-D-L | Score |
|---|---|---|
| 1950 | 31-1-8 | 78.8% |
| 2100 | 28-1-11 | 71.2% |
| 2250 | 26-1-13 | 66.2% |
| 2400 | 14-1-25 | 36.2% |

Fitted 2289, 95% CI 2229 to 2353, over 160 games. The adaptive ladder stopped
early with both tested rungs below the estimate, so two manual rungs at 2250
and 2400 were added to bracket it; the 2400 rung dropped the score to 36% and
pinned the fit. The score declines unusually slowly from 1950 to 2250, which
reads as the logistic slope being shallow against limited-strength Stockfish in
that range; the bracketed fit absorbs it.

The old move-quality estimate was 1969. Full games rate AlphaMini about 300
points higher, the mirror image of MiniGPT: a searching engine converts small
advantages into wins more reliably than its single-move quality suggests.

## Relation to the move-quality calibration

The `calibrate/` crate estimates a different quantity: how much expected score
a bot's single moves lose compared to rated Chess.com players given the same
positions. Those estimates remain valid for what they measure, and its result
documents stay in the repo. The site no longer displays them, because a rating
that predicts full-game results is the number a visitor actually experiences
when they play. The two methods agree where search is involved (Depth-3) and
disagree sharply for the no-search model (MiniGPT), which is itself a finding:
move quality and playing strength are not the same thing.

`docs/alphamini/results/run-003.md` distinguishes three Elo notions
(engine-vs-engine relative, move-quality calibrated, and site-displayed). The
ratings on this page are a fourth: full-game, externally anchored. They are the
site-displayed numbers now.

## Reproducing

```sh
cargo run -p arena --release --features alphamini,minigpt --bin rate -- \
  auto --bot depth3
cargo run -p arena --release --features alphamini,minigpt --bin rate -- \
  crossplay --bot random --opponent minigpt --pairs 50
cargo run -p arena --release --features alphamini,minigpt --bin rate -- \
  fit --bot random
```

Logs and summaries are under `runs/full-game-elo/<bot>/`. Stockfish is
time-limited, so reruns will not reproduce game-for-game identical results;
the fitted ratings should land within the stated intervals.
