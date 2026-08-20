use std::collections::HashSet;
use std::env;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use arena::{OPENING_SUITE_FORMAT_VERSION, OpeningSuite, OpeningSuiteEntry};
use chess_core::{Board, Status};
use minimax::{SearchLimits, find_best_move};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

const HELP: &str = "\
Generate the frozen, balanced opening suite used for paired engine evaluation.\n\
\n\
Usage: cargo run -p arena --release --bin generate-openings -- --output FILE [OPTIONS]\n\
\n\
Options:\n\
  --output FILE       New JSON file to create (required; never overwritten)\n\
  --count N           Number of accepted openings (default: 200)\n\
  --plies N           Random legal half-moves per opening (default: 8)\n\
  --seed N            Deterministic generator seed (default: 1)\n\
  --min-legal N       Minimum legal replies after the prefix (default: 8)\n\
  --max-score-cp N    Maximum absolute depth-3 score (default: 100)\n\
  --max-attempts N    Abort after this many candidates (default: 100000)\n\
  -h, --help          Print this help\n";

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    output: PathBuf,
    count: usize,
    plies: u16,
    seed: u64,
    minimum_legal_moves: usize,
    maximum_score_cp: i32,
    maximum_attempts: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let Some(args) = parse_args(env::args().skip(1))? else {
        print!("{HELP}");
        return Ok(());
    };
    let mut rng = StdRng::seed_from_u64(args.seed);
    let limits = SearchLimits::fixed_depth(3)?;
    let mut seen = HashSet::new();
    let mut openings = Vec::with_capacity(args.count);

    for attempt in 1..=args.maximum_attempts {
        let mut board =
            Board::import_san("").expect("the standard chess starting position must remain valid");
        for _ in 0..args.plies {
            let legal = board.get_legal_moves();
            let Some(mv) = legal.choose(&mut rng).copied() else {
                break;
            };
            board.make_move(mv);
        }
        if board.san_history.len() != usize::from(args.plies) || board.status() != Status::Ongoing {
            continue;
        }
        let legal_moves = board.get_legal_moves().len();
        if legal_moves < args.minimum_legal_moves {
            continue;
        }
        let fen = board.to_fen();
        if !seen.insert(fen.clone()) {
            continue;
        }
        let result = find_best_move(&board, limits).map_err(|error| error.to_string())?;
        if result.score.unsigned_abs() > args.maximum_score_cp as u32 {
            continue;
        }
        let number = openings.len() + 1;
        openings.push(OpeningSuiteEntry {
            id: format!("random-balanced-{number:04}"),
            san: board.san_history.join(" "),
            fen,
            legal_moves,
            depth_three_score_cp: result.score,
        });
        if openings.len() == args.count {
            eprintln!(
                "accepted {} openings after {attempt} candidates",
                openings.len()
            );
            break;
        }
        if openings.len() % 20 == 0 {
            eprintln!("accepted {}/{} openings", openings.len(), args.count);
        }
    }
    if openings.len() != args.count {
        return Err(format!(
            "accepted only {} of {} requested openings after {} attempts",
            openings.len(),
            args.count,
            args.maximum_attempts
        ));
    }

    let suite = OpeningSuite {
        format_version: OPENING_SUITE_FORMAT_VERSION,
        name: "alphamini-v1-balanced-openings".to_string(),
        seed: args.seed,
        plies: args.plies,
        minimum_legal_moves: args.minimum_legal_moves,
        maximum_absolute_depth_three_score_cp: args.maximum_score_cp,
        baseline: "MinimaxDepth3V1".to_string(),
        openings,
    };
    suite.validate().map_err(|error| error.to_string())?;

    if let Some(parent) = args
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .map_err(|error| format!("could not create {}: {error}", args.output.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &suite)
        .map_err(|error| format!("could not serialize suite: {error}"))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .map_err(|error| format!("could not finish {}: {error}", args.output.display()))?;
    println!(
        "Wrote {} openings to {}",
        suite.openings.len(),
        args.output.display()
    );
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let mut output = None;
    let mut count = 200;
    let mut plies = 8;
    let mut seed = 1;
    let mut minimum_legal_moves = 8;
    let mut maximum_score_cp = 100;
    let mut maximum_attempts = 100_000;
    let mut args = args.into_iter();
    while let Some(option) = args.next() {
        if matches!(option.as_str(), "-h" | "--help") {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {option}"))?;
        match option.as_str() {
            "--output" => output = Some(PathBuf::from(value)),
            "--count" => count = parse_value(&option, &value)?,
            "--plies" => plies = parse_value(&option, &value)?,
            "--seed" => seed = parse_value(&option, &value)?,
            "--min-legal" => minimum_legal_moves = parse_value(&option, &value)?,
            "--max-score-cp" => maximum_score_cp = parse_value(&option, &value)?,
            "--max-attempts" => maximum_attempts = parse_value(&option, &value)?,
            _ => return Err(format!("unknown option {option:?}")),
        }
    }
    let output = output.ok_or_else(|| "--output is required".to_string())?;
    if count == 0 || plies == 0 || minimum_legal_moves == 0 || maximum_attempts == 0 {
        return Err("counts, plies, legal moves, and attempts must be positive".to_string());
    }
    if maximum_score_cp < 0 {
        return Err("--max-score-cp must not be negative".to_string());
    }
    Ok(Some(Args {
        output,
        count,
        plies,
        seed,
        minimum_legal_moves,
        maximum_score_cp,
        maximum_attempts,
    }))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frozen_defaults_and_overrides() {
        let args = parse_args(
            ["--output", "suite.json", "--count", "3", "--seed", "9"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();
        assert_eq!(args.output, PathBuf::from("suite.json"));
        assert_eq!(args.count, 3);
        assert_eq!(args.plies, 8);
        assert_eq!(args.seed, 9);
        assert_eq!(args.maximum_score_cp, 100);
    }

    #[test]
    fn rejects_overwrite_prone_or_invalid_inputs_early() {
        assert!(parse_args(Vec::<String>::new()).is_err());
        assert!(
            parse_args(
                ["--output", "suite.json", "--count", "0"]
                    .into_iter()
                    .map(str::to_string)
            )
            .is_err()
        );
    }
}
