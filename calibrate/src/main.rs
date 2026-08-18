use std::collections::HashSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use arena::{Engine, MinimaxEngine, RandomEngine};
use calibrate::calibration::{CalibrationConfig, RatingEstimate, calibrate};
use calibrate::collect::{CollectConfig, collect};
use calibrate::evaluate::{EvaluationConfig, evaluate_samples};
use calibrate::pgn::{ChessComGame, SampleConfig, sample_game};
use calibrate::stockfish::Stockfish;
use calibrate::{AnalysisArtifact, AnalysisMetadata};
use minimax::SearchLimits;
use rand::SeedableRng;
use rand::rngs::StdRng;

const HELP: &str = "\
Estimate a bot's Chess.com 30+0 move-quality-equivalent rating.\n\
\n\
Usage:\n\
  calibrate collect --output FILE --seed-user NAME [OPTIONS]\n\
  calibrate analyze --corpus FILE [--corpus FILE ...] --output FILE --bot BOT [OPTIONS]\n\
  calibrate report --analysis FILE [--analysis FILE ...] [OPTIONS]\n\
\n\
Run `calibrate COMMAND --help` for command-specific options.\n";

const COLLECT_HELP: &str = "\
Download rated standard Chess.com games at exactly 30+0 (TimeControl 1800).\n\
\n\
Usage: calibrate collect --output FILE --seed-user NAME [OPTIONS]\n\
\n\
Options:\n\
  --output FILE       New JSONL corpus file (required)\n\
  --seed-user NAME    Initial 30+0 player; repeat for several seeds (required)\n\
  --max-users N       Maximum public player archives queried (default: 1000)\n\
  --max-games N       Stop after this many unique rated games (default: 5000)\n\
  --games-per-user N  Cap each queried player's contribution (default: 20)\n\
  --min-participant-rating N  Require at least one player at this rating (default: 0)\n\
  --seed N            Reproducible per-player game sampling seed (default: 1)\n\
  --user-agent TEXT   Chess.com API User-Agent; contact info is encouraged\n";

const ANALYZE_HELP: &str = "\
Compare a bot and the human move on identical positions using Stockfish WDL.\n\
\n\
Usage: calibrate analyze --corpus FILE [--corpus FILE ...] --output FILE --bot BOT [OPTIONS]\n\
\n\
Options:\n\
  --corpus FILE       JSONL file produced by collect; repeatable (required)\n\
  --exclude-corpus FILE  Exclude humans found in this corpus; repeatable\n\
  --output FILE       New JSON analysis artifact (required)\n\
  --bot BOT           random or minimax (required)\n\
  --minimax-depth N   Fixed depth when no time budget is set (default: 3)\n\
  --minimax-time-ms N Per-move Minimax budget; e.g. 9000 for production\n\
  --minimax-max-depth N  Timed-search depth ceiling (default: 64)\n\
  --bot-seed N        Seed for Random bot (default: 1)\n\
  --stockfish PATH    UCI engine path (default: /usr/games/stockfish)\n\
  --nodes N           Stockfish nodes per search (default: 100000)\n\
  --hash-mb N         Stockfish hash size in MiB (default: 128)\n\
  --positions N       Positions sampled per side per game (default: 1)\n\
  --positions-per-player N  Candidate-position cap per human (default: 3)\n\
  --analyzed-positions-per-player N  Informative rows per human (default: 1)\n\
  --max-positions N   Random candidate cap after player filtering (default: unlimited)\n\
  --shard-count N     Split humans deterministically across N jobs (default: 1)\n\
  --shard-index N     Zero-based shard handled by this job (default: 0)\n\
  --min-rating N      Lowest sampled human rating (default: 200)\n\
  --max-rating N      Highest sampled human rating (default: 3200)\n\
  --min-ply N         First eligible half-move (default: 12)\n\
  --max-ply N         Last eligible half-move (default: 60)\n\
  --sample-seed N     Reproducible position-sampling seed (default: 1)\n";

const REPORT_HELP: &str = "\
Fit the bot rating where human and bot expected-point loss are equal.\n\
\n\
Usage: calibrate report --analysis FILE [--analysis FILE ...] [OPTIONS]\n\
\n\
Options:\n\
  --analysis FILE     JSON artifact produced by analyze; repeatable (required)\n\
  --min-rating N      Lowest calibration rating (default: 400)\n\
  --max-rating N      Highest calibration rating (default: 2600)\n\
  --bin-width N       Rating band width (default: 200)\n\
  --min-samples N     Required samples in each included band (default: 25)\n\
  --bootstrap N       Whole-player bootstrap repetitions (default: 1000)\n\
  --seed N            Bootstrap seed (default: 1)\n";

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
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print!("{HELP}");
        return Ok(());
    };
    let args: Vec<String> = args.collect();
    match command.as_str() {
        "collect" => run_collect(&args),
        "analyze" => run_analyze(&args),
        "report" => run_report(&args),
        "-h" | "--help" => {
            print!("{HELP}");
            Ok(())
        }
        _ => Err(format!("unknown command {command:?}\n\n{HELP}")),
    }
}

fn run_collect(args: &[String]) -> Result<(), String> {
    if wants_help(args) {
        print!("{COLLECT_HELP}");
        return Ok(());
    }
    let parsed = Options::parse(args)?;
    parsed.reject_unknown(&[
        "output",
        "seed-user",
        "max-users",
        "max-games",
        "games-per-user",
        "min-participant-rating",
        "seed",
        "user-agent",
    ])?;
    let output_path = parsed.required_path("output")?;
    let seed_users = parsed.all("seed-user");
    if seed_users.is_empty() {
        return Err("--seed-user is required and may be repeated".to_string());
    }
    let config = CollectConfig {
        seed_users,
        max_users: parsed.value_or("max-users", 1_000_usize)?,
        max_games: parsed.value_or("max-games", 5_000_usize)?,
        max_games_per_user: parsed.value_or("games-per-user", 20_usize)?,
        minimum_participant_rating: parsed.value_or("min-participant-rating", 0_u32)?,
        seed: parsed.value_or("seed", 1_u64)?,
        user_agent: parsed
            .one("user-agent")
            .unwrap_or_else(|| "chess-engines-calibration/0.1".to_string()),
    };
    let file = create_new(&output_path)?;
    let mut writer = BufWriter::new(file);
    let stats = collect(&config, &mut writer, |username, stats| {
        eprintln!(
            "queried {}/{} users; saved {}/{} games (last: {})",
            stats.users_queried, config.max_users, stats.games_written, config.max_games, username
        );
    })?;
    writer
        .flush()
        .map_err(|error| format!("could not flush {}: {error}", output_path.display()))?;
    println!(
        "Saved {} unique rated Chess.com 30+0 games to {} ({} users queried, {} duplicates).",
        stats.games_written,
        output_path.display(),
        stats.users_queried,
        stats.duplicate_games
    );
    println!(
        "Skipped {} otherwise eligible games to enforce the global per-player cap.",
        stats.player_cap_skips
    );
    Ok(())
}

fn run_analyze(args: &[String]) -> Result<(), String> {
    if wants_help(args) {
        print!("{ANALYZE_HELP}");
        return Ok(());
    }
    let parsed = Options::parse(args)?;
    parsed.reject_unknown(&[
        "corpus",
        "exclude-corpus",
        "output",
        "bot",
        "minimax-depth",
        "minimax-time-ms",
        "minimax-max-depth",
        "bot-seed",
        "stockfish",
        "nodes",
        "hash-mb",
        "positions",
        "positions-per-player",
        "analyzed-positions-per-player",
        "max-positions",
        "shard-count",
        "shard-index",
        "min-rating",
        "max-rating",
        "min-ply",
        "max-ply",
        "sample-seed",
    ])?;
    let corpus_paths: Vec<PathBuf> = parsed
        .all("corpus")
        .into_iter()
        .map(PathBuf::from)
        .collect();
    if corpus_paths.is_empty() {
        return Err("--corpus is required and may be repeated".to_string());
    }
    let exclude_corpus_paths: Vec<PathBuf> = parsed
        .all("exclude-corpus")
        .into_iter()
        .map(PathBuf::from)
        .collect();
    let output_path = parsed.required_path("output")?;
    let bot_kind = parsed.required("bot")?;
    let minimax_depth = parsed.value_or("minimax-depth", 3_u8)?;
    let minimax_time_ms: Option<u64> = parsed.optional_value("minimax-time-ms")?;
    let minimax_max_depth = parsed.value_or("minimax-max-depth", 64_u8)?;
    let bot_seed = parsed.value_or("bot-seed", 1_u64)?;
    let stockfish_path = parsed
        .one("stockfish")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/games/stockfish"));
    let nodes = parsed.value_or("nodes", 100_000_u64)?;
    let hash_mb = parsed.value_or("hash-mb", 128_u32)?;
    let positions_per_side = parsed.value_or("positions", 1_usize)?;
    let positions_per_player = parsed.value_or("positions-per-player", 3_usize)?;
    if positions_per_player == 0 {
        return Err("--positions-per-player must be greater than zero".to_string());
    }
    let analyzed_positions_per_player =
        parsed.value_or("analyzed-positions-per-player", 1_usize)?;
    if analyzed_positions_per_player == 0 {
        return Err("--analyzed-positions-per-player must be greater than zero".to_string());
    }
    let max_positions = parsed.value_or("max-positions", usize::MAX)?;
    if max_positions == 0 {
        return Err("--max-positions must be greater than zero".to_string());
    }
    let shard_count = parsed.value_or("shard-count", 1_u64)?;
    let shard_index = parsed.value_or("shard-index", 0_u64)?;
    if shard_count == 0 || shard_index >= shard_count {
        return Err("--shard-count must be positive and --shard-index must be smaller".to_string());
    }
    let min_rating = parsed.value_or("min-rating", 200_u16)?;
    let max_rating = parsed.value_or("max-rating", 3_200_u16)?;
    if min_rating > max_rating {
        return Err("--min-rating must not exceed --max-rating".to_string());
    }
    let min_ply = parsed.value_or("min-ply", 12_u16)?;
    let max_ply = Some(parsed.value_or("max-ply", 60_u16)?);
    let sample_seed = parsed.value_or("sample-seed", 1_u64)?;

    let (mut bot, bot_name): (Box<dyn Engine>, String) = match bot_kind.as_str() {
        "random" => (
            Box::new(RandomEngine::seeded(bot_seed)),
            format!("Random (seed {bot_seed})"),
        ),
        "minimax" => {
            let limits = match minimax_time_ms {
                Some(milliseconds) => SearchLimits {
                    max_depth: minimax_max_depth,
                    move_time: Some(Duration::from_millis(milliseconds)),
                    max_nodes: None,
                },
                None => SearchLimits::fixed_depth(minimax_depth)?,
            };
            limits.validate()?;
            let name = match minimax_time_ms {
                Some(milliseconds) => {
                    format!("Minimax ({milliseconds} ms/move, depth ceiling {minimax_max_depth})")
                }
                None => format!("Minimax (depth {minimax_depth})"),
            };
            (Box::new(MinimaxEngine::new(limits)?), name)
        }
        _ => return Err("--bot must be random or minimax".to_string()),
    };

    let mut games = Vec::new();
    let mut seen_game_urls = HashSet::new();
    for corpus_path in &corpus_paths {
        for game in read_json_lines::<ChessComGame>(corpus_path)? {
            if seen_game_urls.insert(game.url.clone()) {
                games.push(game);
            }
        }
    }
    let mut excluded_players = HashSet::new();
    for corpus_path in &exclude_corpus_paths {
        for game in read_json_lines::<ChessComGame>(corpus_path)? {
            excluded_players.insert(game.white.username.to_ascii_lowercase());
            excluded_players.insert(game.black.username.to_ascii_lowercase());
        }
    }
    let mut rng = StdRng::seed_from_u64(sample_seed);
    let sample_config = SampleConfig {
        positions_per_side,
        min_ply,
        max_ply,
        min_rating,
        max_rating,
    };
    let mut samples = Vec::new();
    for game in &games {
        samples.extend(sample_game(game, sample_config, &mut rng)?);
    }
    samples.retain(|sample| {
        !excluded_players.contains(&sample.actor_username.to_ascii_lowercase())
            && stable_player_hash(&sample.actor_username) % shard_count == shard_index
    });
    // Spend expensive bot searches on more independent humans rather than many
    // correlated positions from the same prolific participant.
    use rand::seq::SliceRandom;
    samples.shuffle(&mut rng);
    let mut player_counts = std::collections::HashMap::new();
    samples.retain(|sample| {
        let count = player_counts
            .entry(sample.actor_username.to_ascii_lowercase())
            .or_insert(0_usize);
        if *count >= positions_per_player {
            return false;
        }
        *count += 1;
        true
    });
    samples.truncate(max_positions);
    if samples.is_empty() {
        return Err("the corpus produced no eligible positions".to_string());
    }
    let unique_games = samples
        .iter()
        .map(|sample| sample.game_id.as_str())
        .collect::<HashSet<_>>()
        .len();

    eprintln!(
        "Starting {} for {} positions from {} games...",
        stockfish_path.display(),
        samples.len(),
        unique_games
    );
    let mut stockfish = Stockfish::start(&stockfish_path, nodes, hash_mb)?;
    let reference_engine = stockfish.name().to_string();
    let progress_step = (samples.len() / 100).max(1);
    let evaluated = evaluate_samples(
        &samples,
        &mut *bot,
        &mut stockfish,
        EvaluationConfig {
            maximum_rows_per_player: analyzed_positions_per_player,
            ..EvaluationConfig::default()
        },
        |completed, total| {
            if completed % progress_step == 0 || completed == total {
                eprintln!("analyzed {completed}/{total} positions");
            }
        },
    )?;
    let artifact = AnalysisArtifact {
        metadata: AnalysisMetadata {
            format_version: 1,
            target: "Chess.com rated standard 30+0 (TimeControl 1800)".to_string(),
            bot: bot_name,
            reference_engine,
            reference_nodes_per_search: nodes,
            input_positions: samples.len(),
            unique_games,
            analyzed_unique_games: evaluated
                .rows
                .iter()
                .map(|row| row.game_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
        },
        skipped_uninformative: evaluated.skipped_uninformative,
        skipped_player_cap: evaluated.skipped_player_cap,
        rows: evaluated.rows,
    };
    let mut output = BufWriter::new(create_new(&output_path)?);
    serde_json::to_writer_pretty(&mut output, &artifact)
        .map_err(|error| format!("could not encode analysis: {error}"))?;
    output
        .write_all(b"\n")
        .and_then(|_| output.flush())
        .map_err(|error| format!("could not write {}: {error}", output_path.display()))?;
    println!(
        "Saved {} analyzed positions to {} ({} uninformative and {} player-cap positions skipped).",
        artifact.rows.len(),
        output_path.display(),
        artifact.skipped_uninformative,
        artifact.skipped_player_cap
    );
    Ok(())
}

/// FNV-1a gives reproducible sharding across processes and Rust versions.
fn stable_player_hash(username: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in username.bytes() {
        hash ^= u64::from(byte.to_ascii_lowercase());
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn run_report(args: &[String]) -> Result<(), String> {
    if wants_help(args) {
        print!("{REPORT_HELP}");
        return Ok(());
    }
    let parsed = Options::parse(args)?;
    parsed.reject_unknown(&[
        "analysis",
        "min-rating",
        "max-rating",
        "bin-width",
        "min-samples",
        "bootstrap",
        "seed",
    ])?;
    let analysis_paths: Vec<PathBuf> = parsed
        .all("analysis")
        .into_iter()
        .map(PathBuf::from)
        .collect();
    if analysis_paths.is_empty() {
        return Err("--analysis is required and may be repeated".to_string());
    }
    let mut artifacts: Vec<AnalysisArtifact> = Vec::new();
    for analysis_path in &analysis_paths {
        let artifact: AnalysisArtifact = serde_json::from_slice(
            &fs::read(analysis_path)
                .map_err(|error| format!("could not read {}: {error}", analysis_path.display()))?,
        )
        .map_err(|error| format!("invalid analysis file {}: {error}", analysis_path.display()))?;
        if artifact.metadata.format_version != 1 {
            return Err(format!(
                "unsupported analysis format version {} in {}",
                artifact.metadata.format_version,
                analysis_path.display()
            ));
        }
        if let Some(first) = artifacts.first()
            && (artifact.metadata.target != first.metadata.target
                || artifact.metadata.bot != first.metadata.bot
                || artifact.metadata.reference_engine != first.metadata.reference_engine
                || artifact.metadata.reference_nodes_per_search
                    != first.metadata.reference_nodes_per_search)
        {
            return Err(
                "analysis artifacts use different bot, target, or reference settings".to_string(),
            );
        }
        artifacts.push(artifact);
    }
    let metadata = &artifacts[0].metadata;
    let rows: Vec<_> = artifacts
        .iter()
        .flat_map(|artifact| artifact.rows.iter().cloned())
        .collect();
    let report = calibrate(
        &rows,
        CalibrationConfig {
            minimum_rating: parsed.value_or("min-rating", 400_u16)?,
            maximum_rating: parsed.value_or("max-rating", 2_600_u16)?,
            bin_width: parsed.value_or("bin-width", 200_u16)?,
            minimum_samples_per_bin: parsed.value_or("min-samples", 25_usize)?,
            bootstrap_repetitions: parsed.value_or("bootstrap", 1_000_usize)?,
            seed: parsed.value_or("seed", 1_u64)?,
        },
    )?;

    println!("Bot: {}", metadata.bot);
    println!("Target: {}", metadata.target);
    println!(
        "Reference: {} at {} nodes/search",
        metadata.reference_engine, metadata.reference_nodes_per_search
    );
    let analyzed_games = rows
        .iter()
        .map(|row| row.game_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    let sampled_positions = artifacts
        .iter()
        .map(|artifact| artifact.metadata.input_positions)
        .sum::<usize>();
    if artifacts.len() == 1 {
        println!(
            "Evidence: {} analyzed positions from {} games ({} sampled positions from {} games)",
            rows.len(),
            analyzed_games,
            sampled_positions,
            metadata.unique_games
        );
    } else {
        println!(
            "Evidence: {} analyzed positions from {} games ({} sampled positions across {} shards)",
            rows.len(),
            analyzed_games,
            sampled_positions,
            artifacts.len()
        );
    }
    println!(
        "Humans represented in analyzed positions: {}",
        rows.iter()
            .map(|row| row.actor_username.to_ascii_lowercase())
            .collect::<HashSet<_>>()
            .len()
    );
    let calibrated_low = report
        .bands
        .first()
        .expect("a calibration fit has populated bands")
        .minimum_rating;
    let calibrated_high = report
        .bands
        .last()
        .expect("a calibration fit has populated bands")
        .maximum_rating;
    match report.estimate {
        RatingEstimate::Estimated(rating) => {
            println!("Chess.com 30+0 equivalent: {:.0}", rating);
        }
        RatingEstimate::BelowRange(rating) => {
            println!("Chess.com 30+0 equivalent: below {rating}");
            println!("Uncertainty is censored by the lowest populated rating band.");
        }
        RatingEstimate::AboveRange(rating) => {
            println!("Chess.com 30+0 equivalent: above {rating}");
            println!("Uncertainty is censored by the highest populated rating band.");
        }
    }
    match report.interval_95 {
        Some((_report_low, report_high)) if report_high < f64::from(calibrated_low) => {
            println!(
                "Approx. 95% player-bootstrap interval: below {calibrated_low} (both endpoints censored)"
            );
        }
        Some((report_low, _report_high)) if report_low > f64::from(calibrated_high) => {
            println!(
                "Approx. 95% player-bootstrap interval: above {calibrated_high} (both endpoints censored)"
            );
        }
        Some((report_low, report_high)) => {
            let low_label = if report_low <= f64::from(calibrated_low) {
                format!("at or below {calibrated_low}")
            } else {
                format!("{report_low:.0}")
            };
            let high_label = if report_high >= f64::from(calibrated_high) {
                format!("at or above {calibrated_high}")
            } else {
                format!("{report_high:.0}")
            };
            println!("Approx. 95% player-bootstrap interval: {low_label} to {high_label}");
        }
        None => {
            println!(
                "Approx. 95% player-bootstrap interval: unavailable (only {} finite fits)",
                report.bootstrap_finite
            );
        }
    }
    println!(
        "Calibration fit: slope {:.4} expected points / 100 rating, R² {:.3}",
        report.slope_per_100_rating, report.r_squared
    );
    println!();
    println!("Band       N  Games  Human loss  Bot loss  Human-Bot");
    for band in report.bands {
        println!(
            "{:4}-{:4} {:4} {:6} {:11.4} {:9.4} {:+10.4}",
            band.minimum_rating,
            band.maximum_rating,
            band.samples,
            band.games,
            band.human_mean_loss,
            band.bot_mean_loss,
            band.human_minus_bot()
        );
    }
    if report.r_squared < 0.5 {
        println!(
            "\nWarning: rating explains little of the observed loss difference; collect more games before trusting this estimate."
        );
    }
    Ok(())
}

fn create_new(path: &Path) -> Result<File, String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            format!(
                "could not create {}: {error} (output files are never overwritten)",
                path.display()
            )
        })
}

fn read_json_lines<T>(path: &Path) -> Result<Vec<T>, String>
where
    T: serde::de::DeserializeOwned,
{
    let file =
        File::open(path).map_err(|error| format!("could not open {}: {error}", path.display()))?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            other => Some((index, other)),
        })
        .map(|(index, line)| {
            let line = line.map_err(|error| {
                format!(
                    "could not read {} line {}: {error}",
                    path.display(),
                    index + 1
                )
            })?;
            serde_json::from_str(&line).map_err(|error| {
                format!(
                    "invalid JSON in {} line {}: {error}",
                    path.display(),
                    index + 1
                )
            })
        })
        .collect()
}

fn wants_help(args: &[String]) -> bool {
    args.iter().any(|arg| arg == "-h" || arg == "--help")
}

#[derive(Debug, Default)]
struct Options {
    values: Vec<(String, String)>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut values = Vec::new();
        let mut index = 0;
        while index < args.len() {
            let option = args[index]
                .strip_prefix("--")
                .ok_or_else(|| format!("expected an option, got {:?}", args[index]))?;
            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("missing value for --{option}"))?;
            if value.starts_with("--") {
                return Err(format!("missing value for --{option}"));
            }
            values.push((option.to_string(), value.clone()));
            index += 2;
        }
        Ok(Self { values })
    }

    fn reject_unknown(&self, allowed: &[&str]) -> Result<(), String> {
        for (name, _) in &self.values {
            if !allowed.contains(&name.as_str()) {
                return Err(format!("unknown option --{name}"));
            }
        }
        Ok(())
    }

    fn one(&self, name: &str) -> Option<String> {
        self.values
            .iter()
            .rev()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, value)| value.clone())
    }

    fn all(&self, name: &str) -> Vec<String> {
        self.values
            .iter()
            .filter(|(candidate, _)| candidate == name)
            .map(|(_, value)| value.clone())
            .collect()
    }

    fn required(&self, name: &str) -> Result<String, String> {
        self.one(name)
            .ok_or_else(|| format!("--{name} is required"))
    }

    fn required_path(&self, name: &str) -> Result<PathBuf, String> {
        self.required(name).map(PathBuf::from)
    }

    fn value_or<T>(&self, name: &str, default: T) -> Result<T, String>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        match self.one(name) {
            Some(value) => value
                .parse()
                .map_err(|error| format!("invalid --{name} value {value:?}: {error}")),
            None => Ok(default),
        }
    }

    fn optional_value<T>(&self, name: &str) -> Result<Option<T>, String>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        self.one(name)
            .map(|value| {
                value
                    .parse()
                    .map_err(|error| format!("invalid --{name} value {value:?}: {error}"))
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_repeated_options() {
        let options = Options::parse(&[
            "--seed-user".to_string(),
            "Alice".to_string(),
            "--seed-user".to_string(),
            "Bob".to_string(),
        ])
        .unwrap();
        assert_eq!(options.all("seed-user"), ["Alice", "Bob"]);
    }

    #[test]
    fn reports_missing_values() {
        let error = Options::parse(&["--output".to_string()]).unwrap_err();
        assert!(error.contains("missing value"));
    }

    #[test]
    fn player_sharding_is_case_insensitive_and_stable() {
        assert_eq!(stable_player_hash("Alice"), stable_player_hash("alice"));
        assert_eq!(stable_player_hash("alice"), 5_803_779_529_149_266_183);
    }
}
