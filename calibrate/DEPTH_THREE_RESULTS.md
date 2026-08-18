# Minimax at fixed depth 3

Date: 2026-08-18

## Result

The fixed-depth engine is best labeled **about 1675 Chess.com 30+0 Elo**. Its
95% player-bootstrap confidence interval is **1572 to 1748**.

The full 100-rating-band regression estimated 1669. Leaving out the smaller
2200 to 2299 band estimated 1674. Using 200-rating bands estimated 1676 across
the full range and 1680 through 2199. The point estimate stayed within 11
rating points across these checks.

The interval uses 20,000 whole-player bootstrap repetitions. It resamples
players rather than individual positions so repeated observations from one
person are not treated as independent evidence.

## Evidence

- 5,793 deduplicated rated standard games with exact Chess.com
  `TimeControl = 1800`
- 3,290 sampled positions
- 1,814 analyzed positions from 1,696 games
- 1,814 distinct human players, with one useful position per player
- 20,000 whole-player bootstrap repetitions
- Stockfish 17.1 at 100,000 nodes for every independently scored move
- Nine populated 100-point rating groups from 1400 through 2299

| Chess.com rating | Players | Human loss | Bot loss | Human minus bot |
|---:|---:|---:|---:|---:|
| 1400 to 1499 | 473 | 0.1040 | 0.0820 | +0.0220 |
| 1500 to 1599 | 279 | 0.1014 | 0.0891 | +0.0123 |
| 1600 to 1699 | 154 | 0.0789 | 0.0799 | -0.0010 |
| 1700 to 1799 | 270 | 0.0635 | 0.0744 | -0.0110 |
| 1800 to 1899 | 267 | 0.0671 | 0.0887 | -0.0216 |
| 1900 to 1999 | 159 | 0.0425 | 0.0655 | -0.0230 |
| 2000 to 2099 | 113 | 0.0502 | 0.0750 | -0.0248 |
| 2100 to 2199 | 70 | 0.0471 | 0.0783 | -0.0312 |
| 2200 to 2299 | 29 | 0.0126 | 0.1024 | -0.0898 |

A positive difference means Minimax chose the better move on average. The
change from positive to negative around the 1600s agrees with the fitted
estimate.

## Engine and reference configuration

- Release build of the repository's Minimax engine
- Fixed search depth of 3
- Single-threaded search
- Stockfish 17.1 WDL reference at 100,000 nodes for the reference, human, and
  bot move
- Forced and essentially decided positions excluded
- Four player-disjoint workers

This is a move-quality estimate, not a rating earned through full games. It
does not measure clock management, resigning, or the positions the bot would
create over a complete game.
