# MinimaxDepth3V1 calibration

Date: 2026-08-18

## Result

The frozen release-gate baseline is best labeled **about 1640 Chess.com 30+0
move-quality Elo**. The fitted point estimate is 1642. Its 95% whole-player
bootstrap interval is **at or below 1400 to 1780**; the lower endpoint is
censored by the predeclared calibration range and must not be read as exactly
1400.

This replaces the former 1675 label. Adding a total integer move-order
tie-break changed the engine's chosen-move digest from
`9469538452773833222` to the frozen `16258623573026552286`, so the old result
could not legitimately be carried forward.

The primary fit uses 100-rating bands. A 200-rating-band sensitivity fit gave
1644 (censored low endpoint through 1793); omitting the 2100 band gave 1645
(censored low endpoint through 1817). The point estimate is stable, while the
wide interval is reported as measured.

## Evidence

- 5,520 sampled positions split deterministically across four workers
- 2,703 informative analyzed positions from 2,540 games
- 2,703 represented human-player samples
- 20,000 whole-player bootstrap repetitions, seed 1
- Stockfish 17.1 at 100,000 nodes for each independently scored move
- Eight populated 100-point rating groups from 1400 through 2199
- The candidate and Stockfish both replayed the complete legal UCI prefix, so
  repetition state was identical rather than reconstructed from FEN alone
- The strict reporter replayed and legality-checked every stored prefix and
  rejected incomplete, duplicate, overlapping, or mixed-config shard sets

| Chess.com rating | Players | Games | Human loss | Bot loss | Human minus bot |
|---:|---:|---:|---:|---:|---:|
| 1400 to 1499 | 416 | 409 | 0.0904 | 0.0761 | +0.0142 |
| 1500 to 1599 | 263 | 262 | 0.0767 | 0.0823 | -0.0055 |
| 1600 to 1699 | 167 | 164 | 0.0876 | 0.0796 | +0.0081 |
| 1700 to 1799 | 259 | 255 | 0.0744 | 0.0792 | -0.0048 |
| 1800 to 1899 | 271 | 264 | 0.0620 | 0.0758 | -0.0138 |
| 1900 to 1999 | 161 | 158 | 0.0628 | 0.0775 | -0.0147 |
| 2000 to 2099 | 96 | 95 | 0.0500 | 0.0710 | -0.0210 |
| 2100 to 2199 | 68 | 67 | 0.0306 | 0.0679 | -0.0374 |

A positive difference means Minimax chose the better move on average. The
primary regression has R² 0.823; the 200-point sensitivity regression has R²
0.967.

The four early-v2 source files were left byte-for-byte unchanged and sealed
into separate reporting artifacts using the contemporaneously captured run
record. The attestation binds each source hash, the ordered corpus hashes, all
sampling/search settings, shard indexes, the frozen Minimax digest, Stockfish
binary SHA-256 `38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6`,
and producer SHA-256
`e0bee1c54876669eba4036dc9846af906b70fdd65beb796d29f3e943c3e46f9e`.
The shared capture-manifest SHA-256 is
`381e032c362aa513151a40473c3856f6db4741db189aad2bfccd63be4149bd0c`.

## Frozen engine and reference

- Release build of `MinimaxDepth3V1`, fixed depth 3
- Negamax alpha-beta with integer evaluation, deterministic total move order,
  and quiescence over captures, promotions, and check evasions
- Frozen chosen-move digest `16258623573026552286`
- Single-threaded search
- Stockfish 17.1 WDL reference at 100,000 nodes for the reference, human, and
  bot move
- Forced and essentially decided positions excluded
- Four player-disjoint workers

This is a historical-position move-quality estimate, not a rating earned by
playing full games. It does not measure clock management, resigning, or the
positions this engine creates over a complete game.
