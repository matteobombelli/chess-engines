//! Full-game Elo calibration against a limited-strength Stockfish ladder.
//!
//! A bot plays complete games from the committed opening suite against
//! `UCI_Elo` rungs; its rating is the maximum-likelihood fit of the logistic
//! Elo model to every game played, with a 95% interval from a stratified
//! bootstrap over opening pairs. Every game is appended to a per-rung durable
//! log, so a run resumes at pair granularity after a crash.
//!
//! A bot too weak for the bottom rung is rated instead by cross-play against an
//! already-rated bot: `crossplay` records the games, and `fit` anchors them to
//! the opponent's rating, or reports a one-sided bound when it scores nothing.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arena::http::HttpEngine;
use arena::rating_log::{
    PairedLogHeader, RATING_LOG_HEADER_SCHEMA, StoredPair, append_pair_log,
    load_or_create_pair_log, read_pair_log_recovering_torn_tail,
};
use arena::uci::{MAX_UCI_ELO, MIN_UCI_ELO, UciEngine};
use arena::{
    BootstrapConfig, Engine, MINIMAX_V1_MOVE_DIGEST, MinimaxEngine, Opening, OpeningPairResult,
    OpeningSuite, PositionRandomEngine, Record, elo_from_score, paired_report_from_results,
    paired_score_bootstrap_95, run_paired_match_with_progress,
};
use minimax::SearchLimits;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const COMMITTED_SUITE: &str = include_str!("../../openings/alphamini-v1.json");
const SUMMARY_SCHEMA: &str = "full-game-elo-summary-v1";
const CROSSPLAY_LOG_HEADER_SCHEMA: &str = "full-game-elo-crossplay-v1";

/// Two-sided 95% z, which is also the one-sided 97.5% z used for the bound on a
/// bot that scored nothing.
const WILSON_Z: f64 = 1.959_963_984_540_054;

const DEFAULT_STOCKFISH: &str = "/usr/games/stockfish";
const DEFAULT_MOVETIME_MS: u64 = 100;
const DEFAULT_MAX_PLIES: u32 = 1_000;
const DEFAULT_BOOTSTRAP_SAMPLES: u32 = 20_000;
const DEFAULT_SEED: u64 = 1;
const DEFAULT_BLOCK_PAIRS: u32 = 10;
const DEFAULT_STEP: u32 = 150;
const DEFAULT_INFORMATIVE_LOW: f64 = 0.15;
const DEFAULT_INFORMATIVE_HIGH: f64 = 0.85;
const DEFAULT_MINIGPT_TEMPERATURE: f32 = 0.5;
const DEFAULT_MINIGPT_MODEL: &str = "artifacts/minigpt/current/model.onnx";

/// The fit is searched one ladder span outside the ladder itself, so a bot that
/// runs off either end lands strictly outside [1320, 3190] and is reported as
/// censored instead of being silently pinned to a rung.
const FIT_MIN_RATING: f64 = 320.0;
const FIT_MAX_RATING: f64 = 4_190.0;

const HELP: &str = "\
Rate a bot by full games against a limited-strength Stockfish ladder.\n\
\n\
Usage:\n\
  cargo run -p arena --release --bin rate -- play --bot NAME --elo N --pairs N [OPTIONS]\n\
  cargo run -p arena --release --bin rate -- fit --bot NAME [OPTIONS]\n\
  cargo run -p arena --release --bin rate -- auto --bot NAME [OPTIONS]\n\
  cargo run -p arena --release --bin rate -- crossplay --bot NAME --opponent NAME --pairs N [OPTIONS]\n\
\n\
Bots: random, depth3, minigpt (requires --features minigpt), alphamini-http, minimax9s-http\n\
\n\
Options:\n\
  --bot NAME             Bot to rate (required)\n\
  --elo N                Ladder rung for `play` (1320-3190)\n\
  --opponent NAME        Local bot to cross-play against: depth3 or minigpt\n\
  --opponent-rating X    Anchor rating for `fit` when the opponent has no summary.json\n\
  --pairs N              Opening pairs at the rung for `play` (default: 10)\n\
  --out-dir DIR          Log directory (default: runs/full-game-elo/<bot>/)\n\
  --seed N               Seed for the bot and the bootstrap (default: 1)\n\
  --max-plies N          Adjudicate a draw after N half-moves (default: 1000)\n\
  --bootstrap N          Bootstrap resamples (default: 20000)\n\
  --stockfish PATH       UCI opponent binary (default: /usr/games/stockfish)\n\
  --movetime-ms N        Stockfish movetime per move (default: 100)\n\
  --bot-url URL          Override the production URL of an http bot\n\
  --minigpt-model FILE   MiniGPT ONNX model (default: artifacts/minigpt/current/model.onnx)\n\
  --minigpt-manifest FILE  Model manifest; defaults to manifest.json beside the model\n\
  --minigpt-temperature X  Sampling temperature (default: 0.5)\n\
  --seed-elo N           First rung probed by `auto` (default: per bot)\n\
  --budget N             Game budget for `auto` (default: per bot)\n\
  --block-pairs N        Opening pairs per `auto` block (default: 10)\n\
  --step N               Elo step between probes (default: 150)\n\
  --target-half-width X  Stop `auto` at this 95% CI half-width (default: per bot)\n\
  --informative-low X    Lowest still-informative rung score (default: 0.15)\n\
  --informative-high X   Highest still-informative rung score (default: 0.85)\n\
  -h, --help             Print this help\n";

struct BotSpec {
    key: &'static str,
    name: &'static str,
    seed_elo: u32,
    game_budget: u32,
    target_half_width: f64,
    url: Option<&'static str>,
}

const BOTS: &[BotSpec] = &[
    BotSpec {
        key: "random",
        name: "Random",
        seed_elo: MIN_UCI_ELO,
        game_budget: 40,
        target_half_width: 75.0,
        url: None,
    },
    BotSpec {
        key: "depth3",
        name: "MinimaxDepth3V1",
        seed_elo: 1_650,
        game_budget: 400,
        target_half_width: 75.0,
        url: None,
    },
    BotSpec {
        key: "minigpt",
        name: "MiniGpt",
        seed_elo: 1_900,
        game_budget: 400,
        target_half_width: 75.0,
        url: None,
    },
    BotSpec {
        key: "alphamini-http",
        name: "AlphaMiniHttpV1",
        seed_elo: 1_950,
        game_budget: 160,
        target_half_width: 100.0,
        url: Some("https://apps.matteob.dev/projects/chessengines/api/alphamini/move"),
    },
    BotSpec {
        key: "minimax9s-http",
        name: "Minimax9sHttpV1",
        seed_elo: 2_050,
        game_budget: 160,
        target_half_width: 100.0,
        url: Some("https://apps.matteob.dev/projects/chessengines/api/minimax/move"),
    },
];

#[derive(Clone, Debug, PartialEq)]
enum Command {
    Play,
    Fit,
    Auto,
    Crossplay,
}

#[derive(Clone, Debug, PartialEq)]
struct Args {
    command: Command,
    bot: String,
    opponent: Option<String>,
    opponent_rating: Option<f64>,
    elo: Option<u32>,
    pairs: u32,
    out_dir: Option<PathBuf>,
    seed: u64,
    max_plies: u32,
    bootstrap_samples: u32,
    stockfish: PathBuf,
    movetime_ms: u64,
    bot_url: Option<String>,
    minigpt_model: PathBuf,
    minigpt_manifest: Option<PathBuf>,
    minigpt_temperature: f32,
    seed_elo: Option<u32>,
    budget: Option<u32>,
    block_pairs: u32,
    step: u32,
    target_half_width: Option<f64>,
    informative_low: f64,
    informative_high: f64,
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
    let spec = bot_spec(&args.bot)?;
    let out_dir = out_dir(&args, spec);
    let openings = committed_openings()?;
    match args.command {
        Command::Play => run_play(&args, spec, &out_dir, &openings),
        Command::Fit => run_fit(&args, spec, &out_dir),
        Command::Auto => run_auto(&args, spec, &out_dir, &openings),
        Command::Crossplay => run_crossplay(&args, spec, &out_dir, &openings),
    }
}

fn run_play(
    args: &Args,
    spec: &BotSpec,
    out_dir: &Path,
    openings: &[Opening],
) -> Result<(), String> {
    let uci_elo = args.elo.ok_or("--elo is required for `play`")?;
    let pairs = usize::try_from(args.pairs).map_err(|_| "--pairs is too large")?;
    play_rung(args, spec, out_dir, openings, uci_elo, pairs)
}

fn run_fit(args: &Args, spec: &BotSpec, out_dir: &Path) -> Result<(), String> {
    let ladder = read_ladder(out_dir)?;
    let crossplay = read_crossplay_logs(out_dir)?;
    if ladder.rungs.is_empty() && crossplay.is_empty() {
        return Err(format!(
            "no rung or cross-play logs in {}",
            out_dir.display()
        ));
    }
    let summary = fit_summary(args, spec, out_dir, &ladder, &crossplay)?;
    print_summary(&summary);
    write_summary(out_dir, &summary)
}

fn run_auto(
    args: &Args,
    spec: &BotSpec,
    out_dir: &Path,
    openings: &[Opening],
) -> Result<(), String> {
    let config = LadderConfig {
        seed_elo: args.seed_elo.unwrap_or(spec.seed_elo),
        block_pairs: args.block_pairs,
        step: args.step,
        game_budget: args.budget.unwrap_or(spec.game_budget),
        target_half_width: args.target_half_width.unwrap_or(spec.target_half_width),
        informative_low: args.informative_low,
        informative_high: args.informative_high,
        max_pairs_per_rung: u32::try_from(openings.len()).map_err(|_| "suite is too large")?,
    };
    loop {
        let ladder = read_ladder(out_dir)?;
        let states: Vec<RungState> = ladder.rungs.iter().map(RungState::from).collect();
        let fit = if ladder
            .rungs
            .iter()
            .any(|rung| config.is_informative(rung.score()))
        {
            Some(fit_outcome(
                &ladder.rungs,
                args.bootstrap_samples,
                args.seed,
            )?)
        } else {
            None
        };
        match next_action(&states, &config, fit) {
            LadderAction::Stop(reason) => {
                println!("Ladder stopped: {reason}");
                if ladder.rungs.is_empty() {
                    return Ok(());
                }
                let crossplay = read_crossplay_logs(out_dir)?;
                let summary = fit_summary(args, spec, out_dir, &ladder, &crossplay)?;
                print_summary(&summary);
                return write_summary(out_dir, &summary);
            }
            LadderAction::PlayBlock { uci_elo, pairs } => {
                let played = ladder
                    .rungs
                    .iter()
                    .find(|rung| rung.uci_elo == uci_elo)
                    .map_or(0, |rung| rung.pair_points.len());
                let target = played + pairs as usize;
                println!(
                    "Playing {pairs} opening pairs at UCI_Elo {uci_elo} ({played} already recorded)"
                );
                play_rung(args, spec, out_dir, openings, uci_elo, target)?;
            }
        }
    }
}

/// Play the rung's log up to `target_pairs` recorded opening pairs, appending
/// each completed pair before the next one starts.
fn play_rung(
    args: &Args,
    spec: &BotSpec,
    out_dir: &Path,
    openings: &[Opening],
    uci_elo: u32,
    target_pairs: usize,
) -> Result<(), String> {
    if target_pairs == 0 || target_pairs > openings.len() {
        return Err(format!(
            "a rung plays between 1 and {} opening pairs",
            openings.len()
        ));
    }
    let mut bot = build_bot(args, spec, args.seed)?;
    let mut stockfish = UciEngine::start(&args.stockfish, uci_elo, args.movetime_ms)?;
    let header = PairedLogHeader {
        schema: RATING_LOG_HEADER_SCHEMA.to_string(),
        engine_a: bot.name().to_string(),
        engine_b: stockfish.name().to_string(),
        depth: minimax_depth(spec),
        stockfish_version: Some(stockfish.version().to_string()),
        uci_elo: Some(uci_elo),
        movetime_ms: Some(args.movetime_ms),
        bot_url: bot_url(args, spec),
        ..blank_header(args, openings)
    };
    let path = rung_log_path(out_dir, uci_elo);
    play_pairs(
        args,
        &mut bot,
        &mut stockfish,
        &header,
        &path,
        &format!("UCI_Elo {uci_elo}"),
        openings,
        target_pairs,
    )
}

/// Record cross-play against an already-rated local bot. This is how a bot that
/// scores nothing against the bottom rung is still measured: the games anchor to
/// the opponent's own fitted rating in `fit`.
fn run_crossplay(
    args: &Args,
    spec: &BotSpec,
    out_dir: &Path,
    openings: &[Opening],
) -> Result<(), String> {
    let opponent_key = args
        .opponent
        .as_deref()
        .ok_or("--opponent is required for `crossplay`")?;
    if !matches!(opponent_key, "depth3" | "minigpt") {
        return Err(format!(
            "cross-play takes a local opponent: depth3 or minigpt, got {opponent_key:?}"
        ));
    }
    if opponent_key == spec.key {
        return Err("a bot cannot cross-play against itself".to_string());
    }
    let opponent_spec = bot_spec(opponent_key)?;
    let target_pairs = usize::try_from(args.pairs).map_err(|_| "--pairs is too large")?;
    if target_pairs == 0 || target_pairs > openings.len() {
        return Err(format!(
            "cross-play plays between 1 and {} opening pairs",
            openings.len()
        ));
    }
    let mut bot = build_bot(args, spec, args.seed)?;
    let mut opponent = build_bot(args, opponent_spec, args.seed ^ CROSSPLAY_OPPONENT_SEED_MIX)?;
    let header = PairedLogHeader {
        schema: CROSSPLAY_LOG_HEADER_SCHEMA.to_string(),
        engine_a: bot.name().to_string(),
        engine_b: opponent.name().to_string(),
        depth: minimax_depth(spec).or_else(|| minimax_depth(opponent_spec)),
        ..blank_header(args, openings)
    };
    let path = crossplay_log_path(out_dir, opponent_key);
    play_pairs(
        args,
        &mut bot,
        &mut opponent,
        &header,
        &path,
        opponent_key,
        openings,
        target_pairs,
    )
}

/// Play `openings` in suite order until the log holds `target_pairs` pairs,
/// committing each pair before the next game starts.
#[allow(clippy::too_many_arguments)]
fn play_pairs<A: Engine, B: Engine>(
    args: &Args,
    bot: &mut A,
    opponent: &mut B,
    header: &PairedLogHeader,
    path: &Path,
    label: &str,
    openings: &[Opening],
    target_pairs: usize,
) -> Result<(), String> {
    let mut pairs: Vec<OpeningPairResult> = load_or_create_pair_log(path, header)?
        .into_iter()
        .map(|(pair, _)| pair)
        .collect();
    for opening in openings.iter().take(target_pairs).skip(pairs.len()) {
        let completed = run_paired_match_with_progress(
            bot,
            opponent,
            std::slice::from_ref(opening),
            args.max_plies,
            |_, _, _| {},
        )
        .map_err(|error| error.to_string())?;
        let pair = completed
            .pairs
            .into_iter()
            .next()
            .expect("one opening produces one pair");
        append_pair_log(path, &StoredPair::from_pair(&pair, None))?;
        pairs.push(pair);
        let points: f64 = pairs.iter().map(|pair| pair.score).sum();
        eprintln!(
            "{label}: {}/{target_pairs} pairs; score {:.1}%",
            pairs.len(),
            points / pairs.len() as f64 * 100.0
        );
    }
    Ok(())
}

/// Keeps a cross-play opponent off the rated bot's RNG stream.
const CROSSPLAY_OPPONENT_SEED_MIX: u64 = 0x6372_6f73_7370_6c79;

fn build_bot(args: &Args, spec: &BotSpec, seed: u64) -> Result<Box<dyn Engine>, String> {
    match spec.key {
        "random" => Ok(Box::new(PositionRandomEngine::seeded(seed))),
        "depth3" => Ok(Box::new(MinimaxEngine::new(SearchLimits::fixed_depth(3)?)?)),
        "minigpt" => build_minigpt(args, seed),
        _ => {
            let url = bot_url(args, spec).ok_or_else(|| format!("{} has no URL", spec.key))?;
            Ok(Box::new(HttpEngine::new(spec.name, url)?))
        }
    }
}

fn minimax_depth(spec: &BotSpec) -> Option<u8> {
    (spec.key == "depth3").then_some(3)
}

#[cfg(feature = "minigpt")]
fn build_minigpt(args: &Args, seed: u64) -> Result<Box<dyn Engine>, String> {
    let manifest = args.minigpt_manifest.clone().unwrap_or_else(|| {
        args.minigpt_model
            .with_file_name(minigpt::MODEL_MANIFEST_FILE)
    });
    Ok(Box::new(arena::MiniGptEngine::load(
        &args.minigpt_model,
        manifest,
        Some(args.minigpt_temperature),
        seed,
    )?))
}

#[cfg(not(feature = "minigpt"))]
fn build_minigpt(_args: &Args, _seed: u64) -> Result<Box<dyn Engine>, String> {
    Err(
        "MiniGPT requires `cargo run -p arena --release --features minigpt --bin rate -- ...`"
            .to_string(),
    )
}

fn bot_url(args: &Args, spec: &BotSpec) -> Option<String> {
    args.bot_url
        .clone()
        .or_else(|| spec.url.map(str::to_string))
}

fn bot_spec(bot: &str) -> Result<&'static BotSpec, String> {
    BOTS.iter().find(|spec| spec.key == bot).ok_or_else(|| {
        let keys: Vec<&str> = BOTS.iter().map(|spec| spec.key).collect();
        format!("unknown bot {bot:?}; expected one of {}", keys.join(", "))
    })
}

fn out_dir(args: &Args, spec: &BotSpec) -> PathBuf {
    args.out_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("runs/full-game-elo/{}", spec.key)))
}

fn rung_log_path(out_dir: &Path, uci_elo: u32) -> PathBuf {
    out_dir.join(format!("elo-{uci_elo}.jsonl"))
}

fn crossplay_log_path(out_dir: &Path, opponent_key: &str) -> PathBuf {
    out_dir.join(format!("crossplay-{opponent_key}.jsonl"))
}

/// The identity every rating log shares. Callers fill in the engines and
/// whichever of the opponent fields their evaluation has.
fn blank_header(args: &Args, openings: &[Opening]) -> PairedLogHeader {
    PairedLogHeader {
        schema: RATING_LOG_HEADER_SCHEMA.to_string(),
        engine_a: String::new(),
        engine_b: String::new(),
        model_sha256: None,
        opponent_model_sha256: None,
        opening_suite_sha256: None,
        opening_ids: openings.iter().map(|opening| opening.id.clone()).collect(),
        depth: None,
        seed: args.seed,
        max_plies: args.max_plies,
        simulations: None,
        time_ms: None,
        batch_size: None,
        cpuct_ppm: None,
        fpu_reduction_ppm: None,
        bootstrap_samples: args.bootstrap_samples,
        required_lower_score_ppm: None,
        minimax_v1_move_digest: MINIMAX_V1_MOVE_DIGEST,
        evaluation_binary_sha256: None,
        target: format!("{}-{}", env::consts::ARCH, env::consts::OS),
        inference_device: None,
        exploratory: true,
        stockfish_version: None,
        uci_elo: None,
        movetime_ms: None,
        bot_url: None,
    }
}

fn committed_openings() -> Result<Vec<Opening>, String> {
    let suite: OpeningSuite = serde_json::from_str(COMMITTED_SUITE)
        .map_err(|error| format!("could not parse the committed opening suite: {error}"))?;
    suite.validate().map_err(|error| error.to_string())
}

/// One rung's recorded games. `pair_points` is the bot's points over the two
/// color-reversed games of each opening, so a bootstrap resamples openings, not
/// individually correlated games.
struct RungLog {
    uci_elo: u32,
    pair_points: Vec<f64>,
    pairs: Vec<OpeningPairResult>,
    engine_a: String,
    engine_b: String,
    stockfish_version: Option<String>,
    movetime_ms: Option<u64>,
}

impl RungLog {
    fn games(&self) -> u32 {
        2 * self.pair_points.len() as u32
    }

    fn points(&self) -> f64 {
        self.pair_points.iter().sum()
    }

    fn score(&self) -> f64 {
        if self.pair_points.is_empty() {
            return 0.0;
        }
        self.points() / f64::from(self.games())
    }
}

struct Ladder {
    rungs: Vec<RungLog>,
}

fn read_ladder(out_dir: &Path) -> Result<Ladder, String> {
    let mut rungs = Vec::new();
    if !out_dir.exists() {
        return Ok(Ladder { rungs });
    }
    let entries = fs::read_dir(out_dir)
        .map_err(|error| format!("could not read {}: {error}", out_dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("could not read {}: {error}", out_dir.display()))?
            .path();
        let is_rung_log = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("elo-") && name.ends_with(".jsonl"));
        if is_rung_log {
            let rung = read_rung_log(&path)?;
            // A header-only log is a rung a crash never got a game out of; it
            // carries no score for the ladder to read.
            if !rung.pair_points.is_empty() {
                rungs.push(rung);
            }
        }
    }
    rungs.sort_by_key(|rung| rung.uci_elo);
    Ok(Ladder { rungs })
}

fn read_rung_log(path: &Path) -> Result<RungLog, String> {
    let (header, pairs) = read_pair_log(path)?;
    let uci_elo = header
        .uci_elo
        .ok_or_else(|| format!("rung log {} records no UCI_Elo", path.display()))?;
    Ok(RungLog {
        uci_elo,
        pair_points: pairs.iter().map(|pair| pair.score * 2.0).collect(),
        pairs,
        engine_a: header.engine_a,
        engine_b: header.engine_b,
        stockfish_version: header.stockfish_version,
        movetime_ms: header.movetime_ms,
    })
}

fn read_pair_log(path: &Path) -> Result<(PairedLogHeader, Vec<OpeningPairResult>), String> {
    let committed = read_pair_log_recovering_torn_tail(path)?;
    let mut lines = committed.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| format!("log {} is empty", path.display()))?;
    let header: PairedLogHeader = serde_json::from_str(header_line)
        .map_err(|error| format!("invalid log header in {}: {error}", path.display()))?;
    let mut pairs = Vec::new();
    for (index, line) in lines.enumerate() {
        let stored: StoredPair = serde_json::from_str(line).map_err(|error| {
            format!(
                "invalid pair record on line {} of {}: {error}",
                index + 2,
                path.display()
            )
        })?;
        pairs.push(stored.into_pair()?.0);
    }
    Ok((header, pairs))
}

/// A recorded cross-play match: this bot against an already-rated local bot.
struct CrossplayLog {
    opponent_key: String,
    engine_a: String,
    engine_b: String,
    pairs: Vec<OpeningPairResult>,
}

fn read_crossplay_logs(out_dir: &Path) -> Result<Vec<CrossplayLog>, String> {
    let mut logs = Vec::new();
    if !out_dir.exists() {
        return Ok(logs);
    }
    let entries = fs::read_dir(out_dir)
        .map_err(|error| format!("could not read {}: {error}", out_dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("could not read {}: {error}", out_dir.display()))?
            .path();
        let opponent_key = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("crossplay-"))
            .and_then(|name| name.strip_suffix(".jsonl"))
            .map(str::to_string);
        let Some(opponent_key) = opponent_key else {
            continue;
        };
        let (header, pairs) = read_pair_log(&path)?;
        if pairs.is_empty() {
            continue;
        }
        logs.push(CrossplayLog {
            opponent_key,
            engine_a: header.engine_a,
            engine_b: header.engine_b,
            pairs,
        });
    }
    logs.sort_by(|left, right| left.opponent_key.cmp(&right.opponent_key));
    Ok(logs)
}

/// Expected score of a player rated `rating` against a rung rated `rung_elo`.
fn expected_score(rating: f64, rung_elo: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf((rung_elo - rating) / 400.0))
}

/// Maximum-likelihood rating under the logistic Elo model, with draws scored as
/// half a point. The log-likelihood is concave and its derivative is strictly
/// decreasing in the rating, so bisecting the derivative finds the maximum; a
/// bot that wins or loses everything pushes it to the search bound, which the
/// caller reports as censored.
fn fit_rating(observations: &[(f64, f64, f64)]) -> f64 {
    let derivative = |rating: f64| {
        observations
            .iter()
            .map(|(rung_elo, games, points)| points - games * expected_score(rating, *rung_elo))
            .sum::<f64>()
    };
    if derivative(FIT_MIN_RATING) <= 0.0 {
        return FIT_MIN_RATING;
    }
    if derivative(FIT_MAX_RATING) >= 0.0 {
        return FIT_MAX_RATING;
    }
    let mut low = FIT_MIN_RATING;
    let mut high = FIT_MAX_RATING;
    for _ in 0..80 {
        let middle = 0.5 * (low + high);
        if derivative(middle) > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    0.5 * (low + high)
}

fn observations(rungs: &[RungLog]) -> Vec<(f64, f64, f64)> {
    rungs
        .iter()
        .filter(|rung| rung.games() > 0)
        .map(|rung| {
            (
                f64::from(rung.uci_elo),
                f64::from(rung.games()),
                rung.points(),
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FitOutcome {
    rating: f64,
    low: f64,
    high: f64,
}

impl FitOutcome {
    fn half_width(&self) -> f64 {
        (self.high - self.low) / 2.0
    }

    /// A fit or interval endpoint outside the ladder is not a measurement: the
    /// ladder cannot distinguish "just past the end" from "far past the end".
    fn censoring(&self) -> Option<String> {
        let ends = [
            ("fit", self.rating),
            ("lower bound", self.low),
            ("upper bound", self.high),
        ];
        let notes: Vec<String> = ends
            .iter()
            .filter_map(|(label, value)| {
                if *value <= f64::from(MIN_UCI_ELO) {
                    Some(format!("{label} is at or below {MIN_UCI_ELO}"))
                } else if *value >= f64::from(MAX_UCI_ELO) {
                    Some(format!("{label} is at or above {MAX_UCI_ELO}"))
                } else {
                    None
                }
            })
            .collect();
        (!notes.is_empty()).then(|| notes.join("; "))
    }
}

/// Percentile interval from resampling opening pairs within each rung, so the
/// two games of a pair stay together and rungs keep their own sample sizes.
fn fit_outcome(rungs: &[RungLog], samples: u32, seed: u64) -> Result<FitOutcome, String> {
    if samples == 0 {
        return Err("--bootstrap must be greater than zero".to_string());
    }
    let played: Vec<&RungLog> = rungs.iter().filter(|rung| rung.games() > 0).collect();
    if played.is_empty() {
        return Err("no games recorded".to_string());
    }
    let rating = fit_rating(&observations(rungs));
    let mut rng = StdRng::seed_from_u64(seed);
    let mut estimates = Vec::with_capacity(samples as usize);
    let mut resampled = Vec::with_capacity(played.len());
    for _ in 0..samples {
        resampled.clear();
        for rung in &played {
            let count = rung.pair_points.len();
            let points: f64 = (0..count)
                .map(|_| rung.pair_points[rng.gen_range(0..count)])
                .sum();
            resampled.push((f64::from(rung.uci_elo), (2 * count) as f64, points));
        }
        estimates.push(fit_rating(&resampled));
    }
    estimates.sort_by(f64::total_cmp);
    Ok(FitOutcome {
        rating,
        low: estimates[percentile_index(estimates.len(), 0.025)],
        high: estimates[percentile_index(estimates.len(), 0.975)],
    })
}

fn percentile_index(len: usize, percentile: f64) -> usize {
    (((len - 1) as f64 * percentile).round() as usize).min(len - 1)
}

/// Upper end of the Wilson interval on a match score. With no points scored it
/// is the one-sided 97.5% bound, and the only thing the games can say.
fn wilson_upper_bound(points: f64, games: u32) -> f64 {
    let count = f64::from(games);
    let score = points / count;
    let denominator = 1.0 + WILSON_Z * WILSON_Z / count;
    let center = (score + WILSON_Z * WILSON_Z / (2.0 * count)) / denominator;
    let margin = WILSON_Z
        * ((score * (1.0 - score) + WILSON_Z * WILSON_Z / (4.0 * count)) / count).sqrt()
        / denominator;
    center + margin
}

/// How far below an opponent a bot that scored nothing must be, in Elo.
fn zero_score_deficit(games: u32) -> f64 {
    -elo_from_score(wilson_upper_bound(0.0, games))
}

/// A rating read off an already-rated opponent. The opponent's own interval is
/// carried into both ends, so the result is never more certain than the anchor.
fn crossplay_rating(opponent: FitOutcome, score: f64, low: f64, high: f64) -> FitOutcome {
    let clamp = |rating: f64| rating.clamp(FIT_MIN_RATING, FIT_MAX_RATING);
    FitOutcome {
        rating: clamp(opponent.rating + elo_from_score(score)),
        low: clamp(opponent.low + elo_from_score(low)),
        high: clamp(opponent.high + elo_from_score(high)),
    }
}

#[derive(Serialize)]
struct RungSummary {
    uci_elo: u32,
    pairs: usize,
    games: u32,
    wins: u32,
    draws: u32,
    losses: u32,
    score: f64,
}

/// One cross-play match, anchored to the opponent's own rating. A bot that
/// scored points gets a rating and interval; one that scored none gets the
/// one-sided bound instead, since its rating is unmeasurable from a shutout.
#[derive(Serialize)]
struct CrossplaySummary {
    opponent: String,
    opponent_engine: String,
    opponent_rating: f64,
    opponent_ci_low: f64,
    opponent_ci_high: f64,
    method: String,
    pairs: usize,
    games: u32,
    wins: u32,
    draws: u32,
    losses: u32,
    score: f64,
    rating: Option<f64>,
    ci_low: Option<f64>,
    ci_high: Option<f64>,
    censored: Option<String>,
    deficit_elo: Option<f64>,
    rating_upper_bound: Option<f64>,
}

#[derive(Serialize)]
struct RatingSummary {
    schema: String,
    bot: String,
    engine: String,
    opponent: String,
    movetime_ms: Option<u64>,
    games: u32,
    rating: Option<f64>,
    ci_low: Option<f64>,
    ci_high: Option<f64>,
    ci_half_width: Option<f64>,
    censored: Option<String>,
    bootstrap_samples: u32,
    bootstrap_seed: u64,
    rungs: Vec<RungSummary>,
    crossplay: Vec<CrossplaySummary>,
    created_unix_seconds: u64,
}

/// The fields `fit` reads back out of an opponent's own summary.
#[derive(Deserialize)]
struct AnchorSummary {
    rating: Option<f64>,
    ci_low: Option<f64>,
    ci_high: Option<f64>,
}

fn fit_summary(
    args: &Args,
    spec: &BotSpec,
    out_dir: &Path,
    ladder: &Ladder,
    crossplay: &[CrossplayLog],
) -> Result<RatingSummary, String> {
    let outcome = if ladder.rungs.is_empty() {
        None
    } else {
        Some(fit_outcome(
            &ladder.rungs,
            args.bootstrap_samples,
            args.seed,
        )?)
    };
    let mut rungs = Vec::with_capacity(ladder.rungs.len());
    for rung in &ladder.rungs {
        let record = rung_record(rung)?;
        rungs.push(RungSummary {
            uci_elo: rung.uci_elo,
            pairs: rung.pair_points.len(),
            games: rung.games(),
            wins: record.wins,
            draws: record.draws,
            losses: record.losses,
            score: rung.score(),
        });
    }
    let engine = ladder
        .rungs
        .first()
        .map_or_else(|| spec.name.to_string(), |rung| rung.engine_a.clone());
    let opponent = ladder
        .rungs
        .first()
        .and_then(|rung| rung.stockfish_version.clone())
        .unwrap_or_default();
    let mut cross = Vec::with_capacity(crossplay.len());
    for log in crossplay {
        cross.push(crossplay_summary(args, out_dir, log)?);
    }
    Ok(RatingSummary {
        schema: SUMMARY_SCHEMA.to_string(),
        bot: spec.key.to_string(),
        engine,
        opponent,
        movetime_ms: ladder.rungs.first().and_then(|rung| rung.movetime_ms),
        games: rungs.iter().map(|rung| rung.games).sum(),
        rating: outcome.map(|outcome| outcome.rating),
        ci_low: outcome.map(|outcome| outcome.low),
        ci_high: outcome.map(|outcome| outcome.high),
        ci_half_width: outcome.map(|outcome| outcome.half_width()),
        censored: outcome.and_then(|outcome| outcome.censoring()),
        bootstrap_samples: args.bootstrap_samples,
        bootstrap_seed: args.seed,
        rungs,
        crossplay: cross,
        created_unix_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| format!("system clock is before Unix epoch: {error}"))?
            .as_secs(),
    })
}

fn crossplay_summary(
    args: &Args,
    out_dir: &Path,
    log: &CrossplayLog,
) -> Result<CrossplaySummary, String> {
    let anchor = opponent_anchor(args, out_dir, &log.opponent_key)?;
    let report = paired_report_from_results(
        log.engine_a.clone(),
        log.engine_b.clone(),
        log.pairs.clone(),
    )
    .map_err(|error| error.to_string())?;
    let record = report.match_report.overall;
    let games = record.games();
    let score = record.score();
    let mut summary = CrossplaySummary {
        opponent: log.opponent_key.clone(),
        opponent_engine: log.engine_b.clone(),
        opponent_rating: anchor.rating,
        opponent_ci_low: anchor.low,
        opponent_ci_high: anchor.high,
        method: "opening-pair bootstrap off the opponent's rating".to_string(),
        pairs: log.pairs.len(),
        games,
        wins: record.wins,
        draws: record.draws,
        losses: record.losses,
        score,
        rating: None,
        ci_low: None,
        ci_high: None,
        censored: None,
        deficit_elo: None,
        rating_upper_bound: None,
    };
    if record.wins == 0 && record.draws == 0 {
        let bound = wilson_upper_bound(0.0, games);
        summary.method = "one-sided 97.5% Wilson bound".to_string();
        summary.deficit_elo = Some(zero_score_deficit(games));
        summary.rating_upper_bound =
            Some((anchor.high + elo_from_score(bound)).clamp(FIT_MIN_RATING, FIT_MAX_RATING));
        return Ok(summary);
    }
    let (low, high) = paired_score_bootstrap_95(
        &report,
        BootstrapConfig {
            samples: args.bootstrap_samples,
            seed: args.seed,
        },
    )
    .map_err(|error| error.to_string())?;
    let outcome = crossplay_rating(anchor, score, low, high);
    summary.rating = Some(outcome.rating);
    summary.ci_low = Some(outcome.low);
    summary.ci_high = Some(outcome.high);
    summary.censored = outcome.censoring();
    Ok(summary)
}

/// The opponent's own rating, from `--opponent-rating` or from the summary its
/// own ladder wrote. Without an anchor a cross-play score is only a comparison,
/// not a rating, so this refuses to guess.
fn opponent_anchor(args: &Args, out_dir: &Path, opponent_key: &str) -> Result<FitOutcome, String> {
    if let Some(rating) = args.opponent_rating {
        return Ok(FitOutcome {
            rating,
            low: rating,
            high: rating,
        });
    }
    let path = out_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(opponent_key)
        .join("summary.json");
    let bytes = fs::read(&path).map_err(|error| {
        format!(
            "could not read {} for {opponent_key}'s rating ({error}); rate it first or pass --opponent-rating",
            path.display()
        )
    })?;
    let summary: AnchorSummary = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid summary {}: {error}", path.display()))?;
    let rating = summary.rating.ok_or_else(|| {
        format!(
            "{} records no fitted rating; pass --opponent-rating",
            path.display()
        )
    })?;
    Ok(FitOutcome {
        rating,
        low: summary.ci_low.unwrap_or(rating),
        high: summary.ci_high.unwrap_or(rating),
    })
}

fn rung_record(rung: &RungLog) -> Result<Record, String> {
    if rung.pairs.is_empty() {
        return Ok(Record::default());
    }
    paired_report_from_results(
        rung.engine_a.clone(),
        rung.engine_b.clone(),
        rung.pairs.clone(),
    )
    .map(|report| report.match_report.overall)
    .map_err(|error| error.to_string())
}

fn print_summary(summary: &RatingSummary) {
    println!();
    if summary.rungs.is_empty() {
        println!("{}", summary.engine);
    } else {
        println!("{} vs {}", summary.engine, summary.opponent);
    }
    for rung in &summary.rungs {
        println!(
            "UCI_Elo {}: {}-{}-{} over {} games ({:.1}%)",
            rung.uci_elo,
            rung.wins,
            rung.draws,
            rung.losses,
            rung.games,
            rung.score * 100.0
        );
    }
    if let (Some(rating), Some(low), Some(high), Some(half_width)) = (
        summary.rating,
        summary.ci_low,
        summary.ci_high,
        summary.ci_half_width,
    ) {
        println!(
            "Fitted rating: {rating:.0} (95% CI {low:.0} to {high:.0}, half-width {half_width:.0}) over {} games",
            summary.games
        );
    }
    if let Some(censored) = &summary.censored {
        println!("Censored: {censored}");
    }
    for cross in &summary.crossplay {
        println!();
        println!(
            "vs {} ({}): {}-{}-{} over {} games ({:.1}%)",
            cross.opponent_engine,
            cross.opponent,
            cross.wins,
            cross.draws,
            cross.losses,
            cross.games,
            cross.score * 100.0
        );
        match (cross.rating, cross.deficit_elo) {
            (Some(rating), _) => println!(
                "Rating from cross-play: {rating:.0} (95% CI {:.0} to {:.0}, anchored to {:.0})",
                cross.ci_low.unwrap_or(rating),
                cross.ci_high.unwrap_or(rating),
                cross.opponent_rating
            ),
            (None, Some(deficit)) => println!(
                "Scored nothing: at least {deficit:.0} Elo below {} (fitted {:.0}); at or below {:.0}",
                cross.opponent,
                cross.opponent_rating,
                cross.rating_upper_bound.unwrap_or(f64::NAN)
            ),
            (None, None) => {}
        }
        if let Some(censored) = &cross.censored {
            println!("Censored: {censored}");
        }
    }
}

fn write_summary(out_dir: &Path, summary: &RatingSummary) -> Result<(), String> {
    fs::create_dir_all(out_dir)
        .map_err(|error| format!("could not create {}: {error}", out_dir.display()))?;
    let path = out_dir.join("summary.json");
    let mut bytes = serde_json::to_vec_pretty(summary)
        .map_err(|error| format!("could not encode the summary: {error}"))?;
    bytes.push(b'\n');
    fs::write(&path, bytes)
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    println!("Summary: {}", path.display());
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LadderConfig {
    seed_elo: u32,
    block_pairs: u32,
    step: u32,
    game_budget: u32,
    target_half_width: f64,
    informative_low: f64,
    informative_high: f64,
    max_pairs_per_rung: u32,
}

impl LadderConfig {
    fn is_informative(&self, score: f64) -> bool {
        (self.informative_low..=self.informative_high).contains(&score)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RungState {
    uci_elo: u32,
    pairs: u32,
    score: f64,
}

impl From<&RungLog> for RungState {
    fn from(rung: &RungLog) -> Self {
        Self {
            uci_elo: rung.uci_elo,
            pairs: rung.pair_points.len() as u32,
            score: rung.score(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum LadderAction {
    PlayBlock { uci_elo: u32, pairs: u32 },
    Stop(&'static str),
}

/// Decide the next block from the games recorded so far.
///
/// The decision reads only rung scores and sizes, never the order the rungs
/// were probed in, so a resumed `auto` run continues the same ladder it would
/// have followed without the interruption.
fn next_action(
    rungs: &[RungState],
    config: &LadderConfig,
    fit: Option<FitOutcome>,
) -> LadderAction {
    let games_played: u32 = rungs.iter().map(|rung| 2 * rung.pairs).sum();
    if games_played + 2 * config.block_pairs > config.game_budget {
        return LadderAction::Stop("game budget exhausted");
    }
    if rungs.is_empty() {
        return LadderAction::PlayBlock {
            uci_elo: config.seed_elo.clamp(MIN_UCI_ELO, MAX_UCI_ELO),
            pairs: config.block_pairs,
        };
    }

    // A crash can leave a rung part-way through a block. Finish it before any
    // rung score is read as the result of a whole block.
    if let Some(partial) = rungs
        .iter()
        .find(|rung| !rung.pairs.is_multiple_of(config.block_pairs))
    {
        return LadderAction::PlayBlock {
            uci_elo: partial.uci_elo,
            pairs: config.block_pairs - partial.pairs % config.block_pairs,
        };
    }

    let informative: Vec<&RungState> = rungs
        .iter()
        .filter(|rung| config.is_informative(rung.score))
        .collect();

    if informative.is_empty() {
        let won = rungs
            .iter()
            .filter(|rung| rung.score > config.informative_high)
            .map(|rung| rung.uci_elo)
            .max();
        let lost = rungs
            .iter()
            .filter(|rung| rung.score < config.informative_low)
            .map(|rung| rung.uci_elo)
            .min();
        return match (won, lost) {
            (Some(top), None) if top >= MAX_UCI_ELO => {
                LadderAction::Stop("the bot beats the top of the ladder")
            }
            (Some(top), None) => LadderAction::PlayBlock {
                uci_elo: (top + config.step).min(MAX_UCI_ELO),
                pairs: config.block_pairs,
            },
            (None, Some(bottom)) if bottom <= MIN_UCI_ELO => {
                LadderAction::Stop("the bot loses to the bottom of the ladder")
            }
            (None, Some(bottom)) => LadderAction::PlayBlock {
                uci_elo: bottom.saturating_sub(config.step).max(MIN_UCI_ELO),
                pairs: config.block_pairs,
            },
            (Some(top), Some(bottom)) if top + 1 < bottom => LadderAction::PlayBlock {
                uci_elo: (top + bottom) / 2,
                pairs: config.block_pairs,
            },
            _ => LadderAction::Stop("no informative rung between adjacent ladder rungs"),
        };
    }

    // One informative rung fixes the estimate but not its spread: probe the
    // other side of it before spending the budget on precision.
    if informative.len() < 2 {
        let anchor = informative[0];
        let target = if anchor.score >= 0.5 {
            (anchor.uci_elo + config.step).min(MAX_UCI_ELO)
        } else {
            anchor.uci_elo.saturating_sub(config.step).max(MIN_UCI_ELO)
        };
        if target != anchor.uci_elo && !rungs.iter().any(|rung| rung.uci_elo == target) {
            return LadderAction::PlayBlock {
                uci_elo: target,
                pairs: config.block_pairs,
            };
        }
    }

    let Some(fit) = fit else {
        return LadderAction::Stop("no fit for the recorded games");
    };
    if fit.censoring().is_some() {
        return LadderAction::Stop("the fit is censored at a ladder bound");
    }
    if fit.half_width() <= config.target_half_width {
        return LadderAction::Stop("target interval reached");
    }
    let next = informative
        .iter()
        .filter(|rung| rung.pairs + config.block_pairs <= config.max_pairs_per_rung)
        .min_by_key(|rung| (rung.pairs, rung.uci_elo));
    match next {
        Some(rung) => LadderAction::PlayBlock {
            uci_elo: rung.uci_elo,
            pairs: config.block_pairs,
        },
        None => LadderAction::Stop("the opening suite is exhausted at every informative rung"),
    }
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        return Ok(None);
    };
    let command = match command.as_str() {
        "-h" | "--help" => return Ok(None),
        "play" => Command::Play,
        "fit" => Command::Fit,
        "auto" => Command::Auto,
        "crossplay" => Command::Crossplay,
        other => return Err(format!("unknown subcommand {other:?}")),
    };
    let mut parsed = Args {
        command,
        bot: String::new(),
        opponent: None,
        opponent_rating: None,
        elo: None,
        pairs: DEFAULT_BLOCK_PAIRS,
        out_dir: None,
        seed: DEFAULT_SEED,
        max_plies: DEFAULT_MAX_PLIES,
        bootstrap_samples: DEFAULT_BOOTSTRAP_SAMPLES,
        stockfish: PathBuf::from(DEFAULT_STOCKFISH),
        movetime_ms: DEFAULT_MOVETIME_MS,
        bot_url: None,
        minigpt_model: PathBuf::from(DEFAULT_MINIGPT_MODEL),
        minigpt_manifest: None,
        minigpt_temperature: DEFAULT_MINIGPT_TEMPERATURE,
        seed_elo: None,
        budget: None,
        block_pairs: DEFAULT_BLOCK_PAIRS,
        step: DEFAULT_STEP,
        target_half_width: None,
        informative_low: DEFAULT_INFORMATIVE_LOW,
        informative_high: DEFAULT_INFORMATIVE_HIGH,
    };
    while let Some(option) = args.next() {
        if matches!(option.as_str(), "-h" | "--help") {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {option}"))?;
        match option.as_str() {
            "--bot" => parsed.bot = value,
            "--opponent" => parsed.opponent = Some(value),
            "--opponent-rating" => parsed.opponent_rating = Some(parse_value(&option, &value)?),
            "--elo" => parsed.elo = Some(parse_value(&option, &value)?),
            "--pairs" => parsed.pairs = parse_value(&option, &value)?,
            "--out-dir" => parsed.out_dir = Some(PathBuf::from(value)),
            "--seed" => parsed.seed = parse_value(&option, &value)?,
            "--max-plies" => parsed.max_plies = parse_value(&option, &value)?,
            "--bootstrap" => parsed.bootstrap_samples = parse_value(&option, &value)?,
            "--stockfish" => parsed.stockfish = PathBuf::from(value),
            "--movetime-ms" => parsed.movetime_ms = parse_value(&option, &value)?,
            "--bot-url" => parsed.bot_url = Some(value),
            "--minigpt-model" => parsed.minigpt_model = PathBuf::from(value),
            "--minigpt-manifest" => parsed.minigpt_manifest = Some(PathBuf::from(value)),
            "--minigpt-temperature" => parsed.minigpt_temperature = parse_value(&option, &value)?,
            "--seed-elo" => parsed.seed_elo = Some(parse_value(&option, &value)?),
            "--budget" => parsed.budget = Some(parse_value(&option, &value)?),
            "--block-pairs" => parsed.block_pairs = parse_value(&option, &value)?,
            "--step" => parsed.step = parse_value(&option, &value)?,
            "--target-half-width" => parsed.target_half_width = Some(parse_value(&option, &value)?),
            "--informative-low" => parsed.informative_low = parse_value(&option, &value)?,
            "--informative-high" => parsed.informative_high = parse_value(&option, &value)?,
            _ => return Err(format!("unknown option {option:?}")),
        }
    }
    if parsed.bot.is_empty() {
        return Err("--bot is required".to_string());
    }
    if parsed.pairs == 0 || parsed.block_pairs == 0 || parsed.step == 0 {
        return Err("--pairs, --block-pairs, and --step must be greater than zero".to_string());
    }
    if parsed.bootstrap_samples == 0 || parsed.max_plies == 0 || parsed.movetime_ms == 0 {
        return Err(
            "--bootstrap, --max-plies, and --movetime-ms must be greater than zero".to_string(),
        );
    }
    if !(0.0..=1.0).contains(&parsed.informative_low)
        || !(0.0..=1.0).contains(&parsed.informative_high)
        || parsed.informative_low >= parsed.informative_high
    {
        return Err(
            "--informative-low must be below --informative-high, both in [0,1]".to_string(),
        );
    }
    if parsed
        .opponent_rating
        .is_some_and(|rating| !rating.is_finite())
    {
        return Err("--opponent-rating must be finite".to_string());
    }
    if parsed.target_half_width.is_some_and(|width| width <= 0.0) {
        return Err("--target-half-width must be greater than zero".to_string());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> LadderConfig {
        LadderConfig {
            seed_elo: 1_650,
            block_pairs: 10,
            step: 150,
            game_budget: 400,
            target_half_width: 75.0,
            informative_low: DEFAULT_INFORMATIVE_LOW,
            informative_high: DEFAULT_INFORMATIVE_HIGH,
            max_pairs_per_rung: 200,
        }
    }

    fn rung(uci_elo: u32, pairs: u32, score: f64) -> RungState {
        RungState {
            uci_elo,
            pairs,
            score,
        }
    }

    fn wide_fit(rating: f64) -> FitOutcome {
        FitOutcome {
            rating,
            low: rating - 200.0,
            high: rating + 200.0,
        }
    }

    #[test]
    fn parses_a_subcommand_and_its_options() {
        let args = parse_args(
            ["play", "--bot", "depth3", "--elo", "1500", "--pairs", "2"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();
        assert_eq!(args.command, Command::Play);
        assert_eq!(args.bot, "depth3");
        assert_eq!(args.elo, Some(1_500));
        assert_eq!(args.pairs, 2);
        assert!(parse_args(["wat"].into_iter().map(str::to_string)).is_err());
        assert!(parse_args(["fit"].into_iter().map(str::to_string)).is_err());
    }

    #[test]
    fn every_bot_has_a_ladder_seed_inside_the_ladder() {
        for spec in BOTS {
            assert!(
                (MIN_UCI_ELO..=MAX_UCI_ELO).contains(&spec.seed_elo),
                "{}",
                spec.key
            );
            assert_eq!(
                spec.url.is_some(),
                spec.key.ends_with("-http"),
                "{}",
                spec.key
            );
        }
        assert!(bot_spec("depth3").is_ok());
        assert!(bot_spec("nope").is_err());
    }

    #[test]
    fn recovers_a_known_rating_from_synthetic_scores() {
        let truth = 1_837.0;
        let synthetic: Vec<(f64, f64, f64)> = [1_500.0, 1_800.0, 2_100.0]
            .into_iter()
            .map(|rung_elo| {
                let games = 1_000.0;
                (rung_elo, games, games * expected_score(truth, rung_elo))
            })
            .collect();
        assert!((fit_rating(&synthetic) - truth).abs() < 0.5);
    }

    #[test]
    fn a_single_rung_still_fits_its_own_score() {
        // Scoring 75% against 1600 is exactly 190.85 Elo above it.
        let fit = fit_rating(&[(1_600.0, 400.0, 300.0)]);
        assert!((fit - 1_790.85).abs() < 0.5, "unexpected fit {fit}");
    }

    #[test]
    fn detects_censoring_at_both_ladder_bounds() {
        let all_wins = fit_rating(&[(f64::from(MAX_UCI_ELO), 40.0, 40.0)]);
        assert_eq!(all_wins, FIT_MAX_RATING);
        let all_losses = fit_rating(&[(f64::from(MIN_UCI_ELO), 40.0, 0.0)]);
        assert_eq!(all_losses, FIT_MIN_RATING);

        let censored = FitOutcome {
            rating: all_losses,
            low: FIT_MIN_RATING,
            high: 1_400.0,
        };
        assert!(censored.censoring().unwrap().contains("at or below 1320"));
        let high = FitOutcome {
            rating: 3_000.0,
            low: 2_800.0,
            high: FIT_MAX_RATING,
        };
        assert!(high.censoring().unwrap().contains("at or above 3190"));
        assert_eq!(
            FitOutcome {
                rating: 2_000.0,
                low: 1_900.0,
                high: 2_100.0
            }
            .censoring(),
            None
        );
    }

    #[test]
    fn probes_the_seed_rung_first_then_steps_towards_the_bot() {
        assert_eq!(
            next_action(&[], &config(), None),
            LadderAction::PlayBlock {
                uci_elo: 1_650,
                pairs: 10
            }
        );
        assert_eq!(
            next_action(&[rung(1_650, 10, 0.95)], &config(), None),
            LadderAction::PlayBlock {
                uci_elo: 1_800,
                pairs: 10
            }
        );
        assert_eq!(
            next_action(&[rung(1_650, 10, 0.05)], &config(), None),
            LadderAction::PlayBlock {
                uci_elo: 1_500,
                pairs: 10
            }
        );
    }

    #[test]
    fn stops_when_the_bot_runs_off_either_end_of_the_ladder() {
        assert_eq!(
            next_action(&[rung(MIN_UCI_ELO, 10, 0.0)], &config(), None),
            LadderAction::Stop("the bot loses to the bottom of the ladder")
        );
        assert_eq!(
            next_action(&[rung(MAX_UCI_ELO, 10, 1.0)], &config(), None),
            LadderAction::Stop("the bot beats the top of the ladder")
        );
    }

    #[test]
    fn splits_a_bracket_that_contains_no_informative_rung() {
        let rungs = [rung(1_650, 10, 0.9), rung(1_950, 10, 0.05)];
        assert_eq!(
            next_action(&rungs, &config(), None),
            LadderAction::PlayBlock {
                uci_elo: 1_800,
                pairs: 10
            }
        );
        let adjacent = [rung(1_650, 10, 0.9), rung(1_651, 10, 0.05)];
        assert_eq!(
            next_action(&adjacent, &config(), None),
            LadderAction::Stop("no informative rung between adjacent ladder rungs")
        );
    }

    #[test]
    fn brackets_a_lone_informative_rung_before_refining_it() {
        assert_eq!(
            next_action(&[rung(1_650, 10, 0.7)], &config(), Some(wide_fit(1_800.0))),
            LadderAction::PlayBlock {
                uci_elo: 1_800,
                pairs: 10
            }
        );
        assert_eq!(
            next_action(&[rung(1_650, 10, 0.3)], &config(), Some(wide_fit(1_500.0))),
            LadderAction::PlayBlock {
                uci_elo: 1_500,
                pairs: 10
            }
        );
    }

    #[test]
    fn refines_the_smallest_informative_rung_until_the_interval_is_tight() {
        let rungs = [rung(1_650, 20, 0.7), rung(1_800, 10, 0.4)];
        assert_eq!(
            next_action(&rungs, &config(), Some(wide_fit(1_750.0))),
            LadderAction::PlayBlock {
                uci_elo: 1_800,
                pairs: 10
            }
        );
        let tight = FitOutcome {
            rating: 1_750.0,
            low: 1_690.0,
            high: 1_810.0,
        };
        assert_eq!(
            next_action(&rungs, &config(), Some(tight)),
            LadderAction::Stop("target interval reached")
        );
        let censored = FitOutcome {
            rating: 1_750.0,
            low: 1_200.0,
            high: 2_000.0,
        };
        assert_eq!(
            next_action(&rungs, &config(), Some(censored)),
            LadderAction::Stop("the fit is censored at a ladder bound")
        );
    }

    #[test]
    fn a_shutout_gives_a_one_sided_wilson_deficit() {
        // 0 points in 200 games: the Wilson 97.5% bound is z^2/(n+z^2), which
        // is 1.88% and puts the bot at least ~687 Elo below its opponent.
        let bound = wilson_upper_bound(0.0, 200);
        assert!((bound - 0.018_845).abs() < 1e-5, "unexpected bound {bound}");
        let deficit = zero_score_deficit(200);
        assert!(
            (deficit - 686.6).abs() < 1.0,
            "unexpected deficit {deficit}"
        );
        // More games can only tighten the bound, never loosen it.
        assert!(zero_score_deficit(400) > deficit);
        assert!(zero_score_deficit(40) < deficit);
        assert!(deficit.is_finite());
    }

    #[test]
    fn a_score_reads_a_rating_off_the_opponent_and_carries_its_interval() {
        let opponent = FitOutcome {
            rating: 1_650.0,
            low: 1_600.0,
            high: 1_700.0,
        };
        let outcome = crossplay_rating(opponent, 0.25, 0.20, 0.30);
        assert!((outcome.rating - (1_650.0 - 190.849)).abs() < 0.01);
        assert!((outcome.low - (1_600.0 - 240.824)).abs() < 0.01);
        assert!((outcome.high - (1_700.0 - 147.196)).abs() < 0.01);
        // An even score is the opponent's own rating, interval and all.
        let even = crossplay_rating(opponent, 0.5, 0.5, 0.5);
        assert_eq!(
            (even.rating, even.low, even.high),
            (1_650.0, 1_600.0, 1_700.0)
        );
        // A bootstrap end at zero would be minus infinity; it is reported at
        // the search floor instead, where `censoring` flags it.
        let censored = crossplay_rating(opponent, 0.25, 0.0, 0.30);
        assert_eq!(censored.low, FIT_MIN_RATING);
        assert!(censored.censoring().unwrap().contains("at or below 1320"));
    }

    #[test]
    fn crossplay_needs_a_local_opponent_that_is_not_the_bot_itself() {
        let args = parse_args(
            [
                "crossplay",
                "--bot",
                "random",
                "--opponent",
                "depth3",
                "--pairs",
                "50",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();
        assert_eq!(args.command, Command::Crossplay);
        assert_eq!(args.opponent.as_deref(), Some("depth3"));
        assert_eq!(args.pairs, 50);

        let openings = committed_openings().unwrap();
        let random = bot_spec("random").unwrap();
        let out_dir = PathBuf::from("/nonexistent");
        let reject = |opponent: Option<&str>| {
            let mut args = args.clone();
            args.opponent = opponent.map(str::to_string);
            run_crossplay(&args, random, &out_dir, &openings).unwrap_err()
        };
        assert!(reject(None).contains("--opponent is required"));
        assert!(reject(Some("alphamini-http")).contains("local opponent"));
        assert!(reject(Some("random")).contains("local opponent"));

        let depth3 = bot_spec("depth3").unwrap();
        let error = run_crossplay(&args, depth3, &out_dir, &openings).unwrap_err();
        assert!(error.contains("itself"), "unexpected error: {error}");
    }

    #[test]
    fn finishes_a_block_that_a_crash_left_part_way_through() {
        assert_eq!(
            next_action(&[rung(1_650, 3, 1.0)], &config(), None),
            LadderAction::PlayBlock {
                uci_elo: 1_650,
                pairs: 7
            }
        );
    }

    #[test]
    fn stops_a_block_short_of_the_game_budget() {
        let mut config = config();
        config.game_budget = 40;
        let rungs = [rung(1_650, 10, 0.7), rung(1_800, 10, 0.4)];
        assert_eq!(
            next_action(&rungs, &config, Some(wide_fit(1_750.0))),
            LadderAction::Stop("game budget exhausted")
        );
    }
}
