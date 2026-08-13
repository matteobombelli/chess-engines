use std::env;
use std::process::ExitCode;
use std::time::Instant;

use arena::{
    MatchConfig, MatchReport, MinimaxEngine, RandomEngine, Record, run_match_with_progress,
};
use minimax::SearchLimits;

const HELP: &str = "\
Estimate Minimax's Elo relative to Random by playing a reproducible match.\n\
\n\
Usage: cargo run -p arena --release -- [OPTIONS]\n\
\n\
Options:\n\
  --games N       Number of games (default: 100)\n\
  --depth N       Fixed Minimax search depth (default: 3)\n\
  --seed N        Seed for Random's moves (default: 1)\n\
  --max-plies N   Adjudicate a draw after N half-moves (default: 1000)\n\
  -h, --help      Print this help\n";

#[derive(Clone, Copy, Debug)]
struct Args {
    games: u32,
    depth: u8,
    seed: u64,
    max_plies: u32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            games: 100,
            depth: 3,
            seed: 1,
            max_plies: 1_000,
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!("Try --help for usage.");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let Some(args) = parse_args(env::args().skip(1))? else {
        print!("{HELP}");
        return Ok(());
    };

    let limits = SearchLimits::fixed_depth(args.depth)?;
    let mut minimax = MinimaxEngine::new(limits)?;
    let mut random = RandomEngine::seeded(args.seed);
    let config = MatchConfig {
        games: args.games,
        max_plies: args.max_plies,
    };
    let progress_step = (args.games / 20).max(1);
    let started = Instant::now();

    println!(
        "Minimax (depth {}) vs Random: {} games, seed {}, max {} plies",
        args.depth, args.games, args.seed, args.max_plies
    );
    let report = run_match_with_progress(&mut minimax, &mut random, config, |completed, _, _| {
        if completed % progress_step == 0 || completed == args.games {
            eprintln!("completed {completed}/{} games", args.games);
        }
    })
    .map_err(|error| error.to_string())?;

    print_report(&report);
    println!("Elapsed: {:.1?}", started.elapsed());
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let mut parsed = Args::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "-h" || arg == "--help" {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--games" => parsed.games = parse_value(&arg, &value)?,
            "--depth" => parsed.depth = parse_value(&arg, &value)?,
            "--seed" => parsed.seed = parse_value(&arg, &value)?,
            "--max-plies" => parsed.max_plies = parse_value(&arg, &value)?,
            _ => return Err(format!("unknown option {arg:?}")),
        }
    }
    Ok(Some(parsed))
}

fn parse_value<T>(option: &str, value: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| format!("invalid {option} value {value:?}: {error}"))
}

fn print_report(report: &MatchReport) {
    let (elo_low, elo_high) = report.elo_95_interval();
    println!();
    println!(
        "{} W-D-L: {}-{}-{}",
        report.engine_a, report.overall.wins, report.overall.draws, report.overall.losses
    );
    println!("Score: {:.1}%", report.score() * 100.0);
    println!(
        "Relative Elo: {} ({} = 0)",
        format_elo(report.elo_difference()),
        report.engine_b
    );
    println!(
        "Approx. 95% interval: {} to {}",
        format_elo(elo_low),
        format_elo(elo_high)
    );
    print_record("As White", report.as_white);
    print_record("As Black", report.as_black);
    println!(
        "Draw causes: stalemate {}, repetition {}, 50-move {}, ply-limit {}",
        report.draws.stalemate,
        report.draws.threefold_repetition,
        report.draws.fifty_move_rule,
        report.draws.ply_limit
    );
}

fn print_record(label: &str, record: Record) {
    println!(
        "{label}: {}-{}-{} ({:.1}% over {} games)",
        record.wins,
        record.draws,
        record.losses,
        record.score() * 100.0,
        record.games()
    );
}

fn format_elo(elo: f64) -> String {
    if elo == f64::INFINITY {
        "+infinity".to_string()
    } else if elo == f64::NEG_INFINITY {
        "-infinity".to_string()
    } else {
        format!("{elo:+.0}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_options() {
        let args = parse_args(
            [
                "--games",
                "20",
                "--depth",
                "2",
                "--seed",
                "42",
                "--max-plies",
                "80",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();
        assert_eq!(args.games, 20);
        assert_eq!(args.depth, 2);
        assert_eq!(args.seed, 42);
        assert_eq!(args.max_plies, 80);
    }

    #[test]
    fn rejects_unknown_options() {
        let error = parse_args(["--wat", "1"].into_iter().map(str::to_string)).unwrap_err();
        assert!(error.contains("unknown option"));
    }
}
