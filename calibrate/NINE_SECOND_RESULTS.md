# Minimax at 9 seconds per move

## Result

The production Minimax configuration is best labeled **about 2050 Chess.com
30+0**, with a conservative calibrated band of **1900-2200**.

The primary 100-rating-band regression estimated 2030. Moving the lower fit
boundary from 1600 to 1700 estimated 2077, while unbinned regressions over
reasonable windows ranged from 2009 to 2120. Reporting 2050 avoids implying
more precision than the data supports.

The player-bootstrap lower endpoint was about 1900. Its upper endpoint crossed
the highest adequately populated band, so 2200 is a data-coverage boundary,
not proof of a hard 2200 ceiling. The 2200-2299 sample had only 19 observations
and was non-monotonic; it is useful evidence that uncertainty remains, but is
not enough to support extrapolating a precise higher rating.

## Evidence

- 5,793 deduplicated rated standard games with exact Chess.com
  `TimeControl = 1800`
- 1,020 analyzed positions from 967 games
- 1,020 distinct human players, one informative position per player
- 20,000 whole-player bootstrap repetitions
- Six adequately populated 100-point bands from 1600 through 2199
- Fit estimate 2030, slope -0.0065 expected points per 100 rating, R-squared
  0.801 on band means

| Chess.com rating | Players | Human loss | Bot loss | Human - bot |
|---:|---:|---:|---:|---:|
| 1600-1699 | 183 | 0.0899 | 0.0577 | +0.0322 |
| 1700-1799 | 261 | 0.0774 | 0.0651 | +0.0123 |
| 1800-1899 | 234 | 0.0800 | 0.0690 | +0.0110 |
| 1900-1999 | 151 | 0.0671 | 0.0630 | +0.0041 |
| 2000-2099 | 97 | 0.0547 | 0.0509 | +0.0038 |
| 2100-2199 | 65 | 0.0412 | 0.0473 | -0.0061 |

Positive `Human - bot` means Minimax chose the better move on average. The
sign changes between the 2000s and 2100s, consistent with the roughly 2050
summary label.

## Engine and reference configuration

- Release build of the repository's Minimax engine
- 9,000 ms wall-clock move budget
- Iterative deepening with a 64-ply ceiling
- Single-threaded search per move
- Stockfish 17.1 WDL reference at 100,000 nodes for each independently scored
  reference, human, and bot move
- Positions sampled between plies 12 and 60; forced and essentially decided
  positions excluded
- Measured on an Intel Core i5-14500 host, with four player-disjoint workers

This is a move-quality-equivalent rating, not an estimate from playing rated
engine games. It does not model clock management or the different position
distribution the bot would create over complete games. Because the limit is
wall-clock time, a different deployment CPU or concurrent load can change the
bot's effective strength and should be recalibrated.
