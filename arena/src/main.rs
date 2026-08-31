use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[cfg(feature = "alphamini")]
use std::time::Duration;

use alphamini::{
    FROZEN_GATE_BATCH_SIZE, FROZEN_GATE_BOOTSTRAP_SAMPLES, FROZEN_GATE_BOOTSTRAP_SEED,
    FROZEN_GATE_MAX_PLIES, FROZEN_GATE_REQUIRED_LOWER_SCORE, FROZEN_GATE_SIMULATIONS,
    FROZEN_GATE_TIME_MS,
};
#[cfg(feature = "alphamini")]
use alphamini::{
    FROZEN_GATE_CPUCT_PPM, FROZEN_GATE_FPU_REDUCTION_PPM, FROZEN_GATE_OPENING_PAIRS,
    FROZEN_GATE_OPENING_SUITE_SHA256, GateVerdictV1, SearchConfig, ValidatedModel,
};
#[cfg(feature = "alphamini")]
use arena::MINIMAX_V1_MOVE_DIGEST;
#[cfg(feature = "alphamini")]
use arena::paired_report_from_results;
#[cfg(feature = "alphamini")]
use arena::rating_log::{
    PAIRED_LOG_HEADER_SCHEMA, PairedLogHeader, StoredPair, append_pair_log, load_or_create_pair_log,
};
#[cfg(feature = "alphamini")]
use arena::{AlphaMiniEngine, AlphaMiniMetrics};
use arena::{
    BootstrapConfig, MatchConfig, MatchReport, MinimaxEngine, OpeningSuite, RandomEngine, Record,
    elo_from_score, paired_score_bootstrap_95, run_match_with_progress,
    run_paired_match_with_progress,
};
#[cfg(any(feature = "alphamini", feature = "minigpt"))]
use arena::{Engine, PositionRandomEngine};
#[cfg(feature = "minigpt")]
use arena::{MiniGptEngine, Opening};
#[cfg(feature = "alphamini")]
use artifact_io::{publish_bytes_new, sha256_bytes, sha256_file};
use minimax::SearchLimits;

const HELP: &str = "\
Run a reproducible Random/Minimax match or an AlphaMini release-gate match.\n\
\n\
Usage: cargo run -p arena --release --bin arena -- [OPTIONS]\n\
\n\
Options:\n\
  --games N       Games, or opening pairs with --openings (default: 100)\n\
  --depth N       Fixed Minimax search depth (default: 3)\n\
  --seed N        Seed for Random's moves (default: 1)\n\
  --max-plies N   Adjudicate a draw after N half-moves (default: 1000)\n\
  --openings FILE  Play a JSON opening suite twice with colors reversed\n\
  --bootstrap N   Pair-bootstrap samples with --openings (default: 20000)\n\
  --alphamini-model FILE     Compare this ONNX model against Minimax (requires feature)\n\
  --alphamini-manifest FILE  Required versioned model manifest\n\
  --opponent NAME            Rung: random, minimax, or minigpt (default: minimax)\n\
  --opponent-model FILE      Exploratory AlphaMini-vs-AlphaMini opponent ONNX\n\
  --opponent-manifest FILE   Matching opponent model manifest\n\
  --alphamini-simulations N  Search cap per move (default: 10000)\n\
  --alphamini-time-ms N      Wall-clock cap per move (default: 9000)\n\
  --alphamini-batch-size N   Leaf inference batch (default: 8)\n\
  --minigpt-model FILE       Play this MiniGPT ONNX model (requires feature)\n\
  --minigpt-manifest FILE    Model manifest; defaults to manifest.json beside it\n\
  --minigpt-temperature X    Override the manifest sampling temperature\n\
  --results FILE             Durable/resumable AlphaMini pair JSONL\n\
  --verdict FILE             Immutable AlphaMini gate verdict JSON\n\
  --require-lower-score X    Required bootstrap lower bound (default: 0.5)\n\
  --exploratory BOOL         Permit a non-frozen suite; not a gate (default: false)\n\
  -h, --help      Print this help\n";

#[derive(Clone, Debug)]
struct Args {
    games: u32,
    depth: u8,
    seed: u64,
    max_plies: u32,
    openings: Option<PathBuf>,
    bootstrap_samples: u32,
    alphamini_model: Option<PathBuf>,
    alphamini_manifest: Option<PathBuf>,
    opponent: String,
    opponent_model: Option<PathBuf>,
    opponent_manifest: Option<PathBuf>,
    alphamini_simulations: u32,
    alphamini_time_ms: u64,
    alphamini_batch_size: usize,
    minigpt_model: Option<PathBuf>,
    minigpt_manifest: Option<PathBuf>,
    minigpt_temperature: Option<f32>,
    results: Option<PathBuf>,
    verdict: Option<PathBuf>,
    required_lower_score: f64,
    exploratory: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            games: 100,
            depth: 3,
            seed: FROZEN_GATE_BOOTSTRAP_SEED,
            max_plies: FROZEN_GATE_MAX_PLIES,
            openings: None,
            bootstrap_samples: FROZEN_GATE_BOOTSTRAP_SAMPLES,
            alphamini_model: None,
            alphamini_manifest: None,
            opponent: "minimax".to_string(),
            opponent_model: None,
            opponent_manifest: None,
            alphamini_simulations: FROZEN_GATE_SIMULATIONS,
            alphamini_time_ms: FROZEN_GATE_TIME_MS,
            alphamini_batch_size: FROZEN_GATE_BATCH_SIZE,
            minigpt_model: None,
            minigpt_manifest: None,
            minigpt_temperature: None,
            results: None,
            verdict: None,
            required_lower_score: FROZEN_GATE_REQUIRED_LOWER_SCORE,
            exploratory: false,
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

    let started = Instant::now();

    if args.alphamini_model.is_some() {
        return run_alphamini_gate(&args, started);
    }
    if args.minigpt_model.is_some() {
        return run_minigpt_match(&args, started);
    }

    if args.alphamini_manifest.is_some() {
        return Err("--alphamini-manifest requires --alphamini-model".to_string());
    }
    if args.minigpt_manifest.is_some() || args.minigpt_temperature.is_some() {
        return Err(
            "--minigpt-manifest and --minigpt-temperature require --minigpt-model".to_string(),
        );
    }
    if args.opponent != "minimax" {
        return Err(
            "--opponent is only supported with --alphamini-model or --minigpt-model".to_string(),
        );
    }
    if args.opponent_model.is_some() || args.opponent_manifest.is_some() {
        return Err("--opponent-model is only supported with --alphamini-model".to_string());
    }
    if args.results.is_some() {
        return Err("--results is only supported with --alphamini-model".to_string());
    }
    if args.verdict.is_some() {
        return Err("--verdict is only supported with --alphamini-model".to_string());
    }

    let limits = SearchLimits::fixed_depth(args.depth)?;
    let mut minimax = MinimaxEngine::new(limits)?;
    let mut random = RandomEngine::seeded(args.seed);

    if let Some(path) = &args.openings {
        if args.bootstrap_samples == 0 {
            return Err("--bootstrap must be greater than zero".to_string());
        }
        let bytes = fs::read(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let suite: OpeningSuite = serde_json::from_slice(&bytes)
            .map_err(|error| format!("invalid opening suite {}: {error}", path.display()))?;
        let openings = suite.validate().map_err(|error| error.to_string())?;
        let pair_count = usize::try_from(args.games).map_err(|_| "--games is too large")?;
        if pair_count == 0 || pair_count > openings.len() {
            return Err(format!(
                "--games must select between 1 and {} opening pairs",
                openings.len()
            ));
        }
        let openings = &openings[..pair_count];
        println!(
            "Minimax (depth {}) vs Random: {} paired openings, seed {}, max {} plies",
            args.depth, pair_count, args.seed, args.max_plies
        );
        let paired = run_paired_match_with_progress(
            &mut minimax,
            &mut random,
            openings,
            args.max_plies,
            |completed, _, _| eprintln!("completed {completed}/{pair_count} opening pairs"),
        )
        .map_err(|error| error.to_string())?;
        print_report(&paired.match_report, false);
        let (low, high) = paired_score_bootstrap_95(
            &paired,
            BootstrapConfig {
                samples: args.bootstrap_samples,
                seed: args.seed,
            },
        )
        .map_err(|error| error.to_string())?;
        println!(
            "Opening-pair bootstrap 95% score interval: {:.1}% to {:.1}% ({} samples)",
            low * 100.0,
            high * 100.0,
            args.bootstrap_samples
        );
        println!(
            "Opening-pair bootstrap 95% relative-Elo interval: {} to {}",
            format_elo(elo_from_score(low)),
            format_elo(elo_from_score(high))
        );
        println!("Elapsed: {:.1?}", started.elapsed());
        return Ok(());
    }

    let config = MatchConfig {
        games: args.games,
        max_plies: args.max_plies,
    };
    let progress_step = (args.games / 20).max(1);

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

    print_report(&report, true);
    println!("Elapsed: {:.1?}", started.elapsed());
    Ok(())
}

#[cfg(feature = "alphamini")]
fn run_alphamini_gate(args: &Args, started: Instant) -> Result<(), String> {
    let model = args.alphamini_model.as_ref().expect("caller checked model");
    let manifest = args
        .alphamini_manifest
        .as_ref()
        .ok_or("--alphamini-manifest is required with --alphamini-model")?;
    let openings_path = args
        .openings
        .as_ref()
        .ok_or("--openings is required for an AlphaMini gate")?;
    if args.bootstrap_samples == 0
        || args.alphamini_simulations == 0
        || args.alphamini_time_ms == 0
        || args.alphamini_batch_size == 0
    {
        return Err("AlphaMini search and bootstrap limits must be greater than zero".to_string());
    }
    if !matches!(args.opponent.as_str(), "random" | "minimax" | "minigpt") {
        return Err("--opponent must be random, minimax, or minigpt".to_string());
    }
    if args.opponent == "minigpt" && !args.exploratory {
        return Err(
            "AlphaMini-vs-MiniGPT is exploratory and requires --exploratory true".to_string(),
        );
    }
    if args.opponent_model.is_some() && !args.exploratory {
        return Err(
            "AlphaMini-vs-AlphaMini is exploratory and requires --exploratory true".to_string(),
        );
    }
    if args.opponent_model.is_some() != args.opponent_manifest.is_some() {
        return Err(
            "--opponent-model and --opponent-manifest must be supplied together".to_string(),
        );
    }
    if args.opponent_model.is_some() && args.opponent != "minimax" {
        return Err("do not combine --opponent-model with --opponent".to_string());
    }
    if !args.required_lower_score.is_finite() || !(0.0..1.0).contains(&args.required_lower_score) {
        return Err("--require-lower-score must be finite and in [0,1)".to_string());
    }
    if args.exploratory && args.verdict.is_some() {
        return Err("--verdict is reserved for a frozen non-exploratory gate".to_string());
    }
    if !args.exploratory && (args.results.is_none() || args.verdict.is_none()) {
        return Err(
            "a release gate requires both --results (resumable pairs) and --verdict (immutable result)"
                .to_string(),
        );
    }
    let bytes = fs::read(openings_path)
        .map_err(|error| format!("could not read {}: {error}", openings_path.display()))?;
    let suite: OpeningSuite = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid opening suite {}: {error}", openings_path.display()))?;
    let suite_sha256 = sha256_bytes(&bytes);
    let openings = if args.exploratory {
        suite.validate().map_err(|error| error.to_string())?
    } else {
        if suite_sha256 != FROZEN_GATE_OPENING_SUITE_SHA256 {
            return Err(format!(
                "release gate requires committed suite SHA-256 {FROZEN_GATE_OPENING_SUITE_SHA256}; got {suite_sha256}"
            ));
        }
        if args.seed != FROZEN_GATE_BOOTSTRAP_SEED
            || args.bootstrap_samples != FROZEN_GATE_BOOTSTRAP_SAMPLES
        {
            return Err("release gate freezes --seed 1 and --bootstrap 20000".to_string());
        }
        if args.required_lower_score != FROZEN_GATE_REQUIRED_LOWER_SCORE {
            return Err("release gate freezes --require-lower-score 0.5".to_string());
        }
        if args.alphamini_time_ms != FROZEN_GATE_TIME_MS
            || args.alphamini_simulations != FROZEN_GATE_SIMULATIONS
            || args.alphamini_batch_size != FROZEN_GATE_BATCH_SIZE
            || args.max_plies != FROZEN_GATE_MAX_PLIES
        {
            return Err(
                "release gate freezes 9000 ms, 10000 simulations, batch size 8, and max 1000 plies"
                    .to_string(),
            );
        }
        if args.opponent == "minimax" && !(1..=3).contains(&args.depth) {
            return Err("only Minimax depths 1, 2, and 3 are frozen release rungs".to_string());
        }
        let expected_pairs = if args.opponent == "minimax" && args.depth == 3 {
            FROZEN_GATE_OPENING_PAIRS
        } else {
            100
        };
        if usize::try_from(args.games).map_err(|_| "--games is too large")? != expected_pairs {
            return Err(format!(
                "frozen depth-{} rung requires --games {expected_pairs}",
                args.depth
            ));
        }
        suite.validate_deep().map_err(|error| error.to_string())?
    };
    let pair_count = usize::try_from(args.games).map_err(|_| "--games is too large")?;
    if pair_count == 0 || pair_count > openings.len() {
        return Err(format!(
            "--games must select between 1 and {} opening pairs",
            openings.len()
        ));
    }
    let openings = &openings[..pair_count];
    let validated = ValidatedModel::load(model, manifest).map_err(|error| error.to_string())?;
    let model_sha256 = validated.manifest.model_sha256.clone();
    let opponent_model_sha256 = match (&args.opponent_model, &args.opponent_manifest) {
        (Some(model), Some(manifest)) => Some(
            ValidatedModel::load(model, manifest)
                .map_err(|error| error.to_string())?
                .manifest
                .model_sha256,
        ),
        (None, None) => None,
        _ => unreachable!("opponent model/manifest pairing validated"),
    };
    let search_config = if args.exploratory {
        SearchConfig::evaluation(
            args.alphamini_simulations,
            args.alphamini_batch_size,
            Duration::from_millis(args.alphamini_time_ms),
        )
    } else {
        SearchConfig::frozen_gate()
    };
    let mut alpha = AlphaMiniEngine::load(model, manifest, search_config, args.seed)?;
    let mut opponent: Box<dyn Engine> = match (&args.opponent_model, &args.opponent_manifest) {
        (Some(model), Some(manifest)) => Box::new(AlphaMiniEngine::load(
            model,
            manifest,
            search_config,
            args.seed ^ 0xa17a_1a1a_5eed_0001,
        )?),
        (None, None) => match args.opponent.as_str() {
            "random" => Box::new(PositionRandomEngine::seeded(args.seed)),
            "minimax" => Box::new(MinimaxEngine::new(SearchLimits::fixed_depth(args.depth)?)?),
            "minigpt" => load_minigpt(args, args.seed ^ MINIGPT_OPPONENT_SEED_MIX)?,
            _ => unreachable!("opponent validated"),
        },
        _ => unreachable!("opponent model/manifest pairing validated"),
    };
    let evaluation_binary_sha256 = sha256_file(
        &std::env::current_exe()
            .map_err(|error| format!("cannot resolve evaluation binary: {error}"))?,
    )
    .map_err(|error| format!("could not hash evaluation binary: {error}"))?;
    let header = PairedLogHeader {
        schema: PAIRED_LOG_HEADER_SCHEMA.to_string(),
        engine_a: alpha.name().to_string(),
        engine_b: opponent.name().to_string(),
        model_sha256: Some(model_sha256.clone()),
        opponent_model_sha256,
        opening_suite_sha256: Some(suite_sha256.clone()),
        opening_ids: openings.iter().map(|opening| opening.id.clone()).collect(),
        depth: Some(args.depth),
        seed: args.seed,
        max_plies: args.max_plies,
        simulations: Some(args.alphamini_simulations),
        time_ms: Some(args.alphamini_time_ms),
        batch_size: Some(args.alphamini_batch_size),
        cpuct_ppm: Some(FROZEN_GATE_CPUCT_PPM),
        fpu_reduction_ppm: Some(FROZEN_GATE_FPU_REDUCTION_PPM),
        bootstrap_samples: args.bootstrap_samples,
        required_lower_score_ppm: Some(score_to_ppm(args.required_lower_score)),
        minimax_v1_move_digest: MINIMAX_V1_MOVE_DIGEST,
        evaluation_binary_sha256: Some(evaluation_binary_sha256.clone()),
        target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        inference_device: Some("onnx-cpu".to_string()),
        exploratory: args.exploratory,
        stockfish_version: None,
        uci_elo: None,
        movetime_ms: None,
        bot_url: None,
    };
    println!(
        "AlphaMini vs {}: {} paired openings, {} ms / {} simulations",
        opponent.name(),
        openings.len(),
        args.alphamini_time_ms,
        args.alphamini_simulations
    );
    let mut pairs = Vec::new();
    let mut metrics = AlphaMiniMetrics::default();
    if let Some(path) = &args.results {
        for (pair, recorded) in load_or_create_pair_log(path, &header)? {
            metrics = metrics_sum(metrics, recorded.unwrap_or_default());
            pairs.push(pair);
        }
    }
    if !pairs.is_empty() {
        eprintln!(
            "resuming after {}/{} durably recorded opening pairs",
            pairs.len(),
            openings.len()
        );
    }
    for opening in openings.iter().skip(pairs.len()) {
        let completed = run_paired_match_with_progress(
            &mut alpha,
            &mut opponent,
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
        let delta = alpha.take_metrics();
        if let Some(path) = &args.results {
            append_pair_log(path, &StoredPair::from_pair(&pair, Some(delta)))?;
        }
        metrics = metrics_sum(metrics, delta);
        pairs.push(pair);
        let score = pairs.iter().map(|pair| pair.score).sum::<f64>() / pairs.len() as f64;
        eprintln!(
            "completed {}/{} opening pairs; score {:.1}%",
            pairs.len(),
            openings.len(),
            score * 100.0
        );
    }
    let paired =
        paired_report_from_results(header.engine_a.clone(), header.engine_b.clone(), pairs)
            .map_err(|error| error.to_string())?;
    print_report(&paired.match_report, false);
    print_alphamini_metrics(metrics);
    let (low, high) = paired_score_bootstrap_95(
        &paired,
        BootstrapConfig {
            samples: args.bootstrap_samples,
            seed: args.seed,
        },
    )
    .map_err(|error| error.to_string())?;
    println!(
        "Opening-pair bootstrap 95% score interval: {:.1}% to {:.1}% ({} samples)",
        low * 100.0,
        high * 100.0,
        args.bootstrap_samples
    );
    println!(
        "Opening-pair bootstrap 95% relative-Elo interval: {} to {}",
        format_elo(elo_from_score(low)),
        format_elo(elo_from_score(high))
    );
    let passed = low > args.required_lower_score;
    if args.exploratory {
        println!(
            "Exploratory result only (criterion would be {}): {}",
            format_score(args.required_lower_score),
            if passed { "above" } else { "not above" }
        );
    } else {
        let pair_log = args
            .results
            .as_ref()
            .expect("release gate requires pair log");
        let verdict_path = args
            .verdict
            .as_ref()
            .expect("release gate requires verdict");
        let verdict = GateVerdictV1 {
            schema: alphamini::manifest::GATE_VERDICT_VERSION.to_string(),
            passed,
            model_sha256,
            opening_suite_sha256: suite_sha256,
            opening_pairs: paired.pairs.len(),
            baseline: header.engine_b.clone(),
            minimax_v1_move_digest: header.minimax_v1_move_digest,
            simulations: args.alphamini_simulations,
            time_ms: args.alphamini_time_ms,
            batch_size: args.alphamini_batch_size,
            cpuct_ppm: FROZEN_GATE_CPUCT_PPM,
            fpu_reduction_ppm: FROZEN_GATE_FPU_REDUCTION_PPM,
            max_plies: header.max_plies,
            bootstrap_samples: args.bootstrap_samples,
            bootstrap_seed: args.seed,
            score: paired.match_report.score(),
            lower_score: low,
            upper_score: high,
            required_lower_score: args.required_lower_score,
            pair_log_sha256: sha256_file(pair_log)
                .map_err(|error| format!("could not hash {}: {error}", pair_log.display()))?,
            evaluation_binary_sha256,
            created_unix_seconds: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| format!("system clock is before Unix epoch: {error}"))?
                .as_secs(),
        };
        write_verdict_atomic(verdict_path, &verdict)?;
        println!(
            "Gate verdict: {} (lower bound {} required > {})",
            if passed { "PASSED" } else { "FAILED" },
            format_score(low),
            format_score(args.required_lower_score)
        );
        println!("Immutable verdict: {}", verdict_path.display());
    }
    println!("Elapsed: {:.1?}", started.elapsed());
    if !args.exploratory && !passed {
        Err("AlphaMini release gate failed; checkpoint must not be deployed".to_string())
    } else {
        Ok(())
    }
}

#[cfg(feature = "alphamini")]
fn metrics_sum(left: AlphaMiniMetrics, right: AlphaMiniMetrics) -> AlphaMiniMetrics {
    AlphaMiniMetrics {
        moves: left.moves + right.moves,
        completed_simulations: left.completed_simulations + right.completed_simulations,
        neural_evaluations: left.neural_evaluations + right.neural_evaluations,
        inference_batches: left.inference_batches + right.inference_batches,
        largest_batch: left.largest_batch.max(right.largest_batch),
        elapsed_micros: left.elapsed_micros + right.elapsed_micros,
        deadlines_reached: left.deadlines_reached + right.deadlines_reached,
    }
}

#[cfg(feature = "alphamini")]
fn print_alphamini_metrics(metrics: AlphaMiniMetrics) {
    let seconds = metrics.elapsed_micros as f64 / 1_000_000.0;
    let simulations_per_second = if seconds > 0.0 {
        metrics.completed_simulations as f64 / seconds
    } else {
        0.0
    };
    let mean_simulations = if metrics.moves > 0 {
        metrics.completed_simulations as f64 / metrics.moves as f64
    } else {
        0.0
    };
    println!(
        "AlphaMini search: {} moves, {} simulations ({mean_simulations:.1}/move, {simulations_per_second:.1}/s), {} neural evals in {} batches, largest batch {}, {} deadlines",
        metrics.moves,
        metrics.completed_simulations,
        metrics.neural_evaluations,
        metrics.inference_batches,
        metrics.largest_batch,
        metrics.deadlines_reached,
    );
}

#[cfg(feature = "alphamini")]
fn write_verdict_atomic(path: &std::path::Path, verdict: &GateVerdictV1) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(verdict)
        .map_err(|error| format!("could not encode gate verdict: {error}"))?;
    bytes.push(b'\n');
    publish_bytes_new(path, &bytes).map_err(|error| {
        format!(
            "could not publish immutable verdict {}: {error}",
            path.display()
        )
    })
}

#[cfg(feature = "alphamini")]
fn score_to_ppm(score: f64) -> u32 {
    (score * 1_000_000.0).round() as u32
}

#[cfg(feature = "alphamini")]
fn format_score(score: f64) -> String {
    format!("{:.3}%", score * 100.0)
}

#[cfg(not(feature = "alphamini"))]
fn run_alphamini_gate(_args: &Args, _started: Instant) -> Result<(), String> {
    Err(
        "AlphaMini model evaluation requires `cargo run -p arena --release --features alphamini --bin arena -- ...`"
            .to_string(),
    )
}

/// Keeps a MiniGPT opponent off the primary engine's RNG stream.
#[cfg(feature = "alphamini")]
const MINIGPT_OPPONENT_SEED_MIX: u64 = 0x6d69_6e69_6770_7401;

#[cfg(feature = "minigpt")]
fn load_minigpt(args: &Args, seed: u64) -> Result<Box<dyn Engine>, String> {
    let model = args
        .minigpt_model
        .as_ref()
        .ok_or("--minigpt-model is required to play MiniGPT")?;
    let manifest = args
        .minigpt_manifest
        .clone()
        .unwrap_or_else(|| model.with_file_name(minigpt::MODEL_MANIFEST_FILE));
    Ok(Box::new(MiniGptEngine::load(
        model,
        manifest,
        args.minigpt_temperature,
        seed,
    )?))
}

#[cfg(all(feature = "alphamini", not(feature = "minigpt")))]
fn load_minigpt(_args: &Args, _seed: u64) -> Result<Box<dyn Engine>, String> {
    Err(
        "MiniGPT requires `cargo run -p arena --release --features minigpt --bin arena -- ...`"
            .to_string(),
    )
}

/// MiniGPT evaluation is exploratory: there is no frozen rung, verdict, or
/// resumable pair log, so a suite simply selects paired openings over a plain
/// alternating-color match.
#[cfg(feature = "minigpt")]
fn run_minigpt_match(args: &Args, started: Instant) -> Result<(), String> {
    if args.opponent_model.is_some() || args.opponent_manifest.is_some() {
        return Err("--opponent-model is only supported with --alphamini-model".to_string());
    }
    if args.results.is_some() || args.verdict.is_some() {
        return Err("--results and --verdict are reserved for the AlphaMini gate".to_string());
    }
    if args.bootstrap_samples == 0 {
        return Err("--bootstrap must be greater than zero".to_string());
    }
    let mut engine = load_minigpt(args, args.seed)?;
    let mut opponent: Box<dyn Engine> = match args.opponent.as_str() {
        "random" => Box::new(PositionRandomEngine::seeded(args.seed)),
        "minimax" => Box::new(MinimaxEngine::new(SearchLimits::fixed_depth(args.depth)?)?),
        other => {
            return Err(format!(
                "a MiniGPT match takes --opponent random or minimax, got {other:?}"
            ));
        }
    };

    if let Some(path) = &args.openings {
        let openings = load_openings(path)?;
        let pair_count = usize::try_from(args.games).map_err(|_| "--games is too large")?;
        if pair_count == 0 || pair_count > openings.len() {
            return Err(format!(
                "--games must select between 1 and {} opening pairs",
                openings.len()
            ));
        }
        let openings = &openings[..pair_count];
        println!(
            "{} vs {}: {pair_count} paired openings, seed {}",
            engine.name(),
            opponent.name(),
            args.seed
        );
        let paired = run_paired_match_with_progress(
            &mut engine,
            &mut opponent,
            openings,
            args.max_plies,
            |completed, _, _| eprintln!("completed {completed}/{pair_count} opening pairs"),
        )
        .map_err(|error| error.to_string())?;
        print_report(&paired.match_report, false);
        let (low, high) = paired_score_bootstrap_95(
            &paired,
            BootstrapConfig {
                samples: args.bootstrap_samples,
                seed: args.seed,
            },
        )
        .map_err(|error| error.to_string())?;
        println!(
            "Opening-pair bootstrap 95% score interval: {:.1}% to {:.1}% ({} samples)",
            low * 100.0,
            high * 100.0,
            args.bootstrap_samples
        );
        println!(
            "Opening-pair bootstrap 95% relative-Elo interval: {} to {}",
            format_elo(elo_from_score(low)),
            format_elo(elo_from_score(high))
        );
        println!("Elapsed: {:.1?}", started.elapsed());
        return Ok(());
    }

    println!(
        "{} vs {}: {} games, seed {}, max {} plies",
        engine.name(),
        opponent.name(),
        args.games,
        args.seed,
        args.max_plies
    );
    let config = MatchConfig {
        games: args.games,
        max_plies: args.max_plies,
    };
    let report = run_match_with_progress(&mut engine, &mut opponent, config, |completed, _, _| {
        eprintln!("completed {completed}/{} games", args.games);
    })
    .map_err(|error| error.to_string())?;
    print_report(&report, true);
    println!("Elapsed: {:.1?}", started.elapsed());
    Ok(())
}

#[cfg(feature = "minigpt")]
fn load_openings(path: &std::path::Path) -> Result<Vec<Opening>, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let suite: OpeningSuite = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid opening suite {}: {error}", path.display()))?;
    suite.validate().map_err(|error| error.to_string())
}

#[cfg(not(feature = "minigpt"))]
fn run_minigpt_match(_args: &Args, _started: Instant) -> Result<(), String> {
    Err(
        "MiniGPT evaluation requires `cargo run -p arena --release --features minigpt --bin arena -- ...`"
            .to_string(),
    )
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
            "--openings" => parsed.openings = Some(PathBuf::from(value)),
            "--bootstrap" => parsed.bootstrap_samples = parse_value(&arg, &value)?,
            "--alphamini-model" => parsed.alphamini_model = Some(PathBuf::from(value)),
            "--alphamini-manifest" => parsed.alphamini_manifest = Some(PathBuf::from(value)),
            "--opponent" => parsed.opponent = value,
            "--opponent-model" => parsed.opponent_model = Some(PathBuf::from(value)),
            "--opponent-manifest" => parsed.opponent_manifest = Some(PathBuf::from(value)),
            "--alphamini-simulations" => parsed.alphamini_simulations = parse_value(&arg, &value)?,
            "--alphamini-time-ms" => parsed.alphamini_time_ms = parse_value(&arg, &value)?,
            "--alphamini-batch-size" => parsed.alphamini_batch_size = parse_value(&arg, &value)?,
            "--minigpt-model" => parsed.minigpt_model = Some(PathBuf::from(value)),
            "--minigpt-manifest" => parsed.minigpt_manifest = Some(PathBuf::from(value)),
            "--minigpt-temperature" => {
                parsed.minigpt_temperature = Some(parse_value(&arg, &value)?)
            }
            "--results" => parsed.results = Some(PathBuf::from(value)),
            "--verdict" => parsed.verdict = Some(PathBuf::from(value)),
            "--require-lower-score" => parsed.required_lower_score = parse_value(&arg, &value)?,
            "--exploratory" => parsed.exploratory = parse_value(&arg, &value)?,
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

fn print_report(report: &MatchReport, print_independent_game_interval: bool) {
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
    if print_independent_game_interval {
        let (elo_low, elo_high) = report.elo_95_interval();
        println!(
            "Approx. 95% interval: {} to {}",
            format_elo(elo_low),
            format_elo(elo_high)
        );
    }
    print_record("As White", report.as_white);
    print_record("As Black", report.as_black);
    println!(
        "Draw causes: stalemate {}, insufficient-material {}, repetition {}, 50-move {}, ply-limit {}",
        report.draws.stalemate,
        report.draws.insufficient_material,
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
    #[cfg(feature = "alphamini")]
    use arena::{GameResult, OpeningPairResult, Termination};
    #[cfg(feature = "alphamini")]
    use chess_core::Color;
    #[cfg(feature = "alphamini")]
    use std::fs::OpenOptions;
    #[cfg(feature = "alphamini")]
    use std::io::Write;

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
        assert_eq!(args.openings, None);
        assert_eq!(args.bootstrap_samples, 20_000);
    }

    #[test]
    fn parses_alphamini_release_gate_options() {
        let args = parse_args(
            [
                "--alphamini-model",
                "model.onnx",
                "--alphamini-manifest",
                "manifest.json",
                "--openings",
                "suite.json",
                "--results",
                "results.jsonl",
                "--verdict",
                "verdict.json",
                "--alphamini-simulations",
                "256",
                "--alphamini-time-ms",
                "9000",
                "--alphamini-batch-size",
                "16",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();
        assert_eq!(args.alphamini_model, Some(PathBuf::from("model.onnx")));
        assert_eq!(
            args.alphamini_manifest,
            Some(PathBuf::from("manifest.json"))
        );
        assert_eq!(args.results, Some(PathBuf::from("results.jsonl")));
        assert_eq!(args.verdict, Some(PathBuf::from("verdict.json")));
        assert_eq!(args.alphamini_simulations, 256);
        assert_eq!(args.alphamini_batch_size, 16);
    }

    #[cfg(feature = "alphamini")]
    #[test]
    fn paired_log_resumes_an_exact_prefix_and_rejects_identity_drift() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pairs.jsonl");
        let header = PairedLogHeader {
            schema: PAIRED_LOG_HEADER_SCHEMA.to_string(),
            engine_a: "AlphaMiniV1[abc]".to_string(),
            engine_b: "MinimaxDepth3V1".to_string(),
            model_sha256: Some("a".repeat(64)),
            opponent_model_sha256: None,
            opening_suite_sha256: Some("b".repeat(64)),
            opening_ids: vec!["opening-1".to_string()],
            depth: Some(3),
            seed: 1,
            max_plies: 512,
            simulations: Some(128),
            time_ms: Some(9_000),
            batch_size: Some(8),
            cpuct_ppm: Some(FROZEN_GATE_CPUCT_PPM),
            fpu_reduction_ppm: Some(FROZEN_GATE_FPU_REDUCTION_PPM),
            bootstrap_samples: 20_000,
            required_lower_score_ppm: Some(500_000),
            minimax_v1_move_digest: MINIMAX_V1_MOVE_DIGEST,
            evaluation_binary_sha256: Some("c".repeat(64)),
            target: "x86_64-linux".to_string(),
            inference_device: Some("onnx-cpu".to_string()),
            exploratory: false,
            stockfish_version: None,
            uci_elo: None,
            movetime_ms: None,
            bot_url: None,
        };
        assert_eq!(load_or_create_pair_log(&path, &header).unwrap(), Vec::new());
        let pair = OpeningPairResult {
            opening_id: "opening-1".to_string(),
            engine_a_as_white: GameResult {
                winner: Some(Color::White),
                termination: Termination::Checkmate,
                plies: 41,
            },
            engine_a_as_black: GameResult {
                winner: None,
                termination: Termination::ThreefoldRepetition,
                plies: 80,
            },
            score: 0.75,
        };
        let metrics = AlphaMiniMetrics {
            moves: 42,
            completed_simulations: 5_376,
            ..AlphaMiniMetrics::default()
        };
        append_pair_log(&path, &StoredPair::from_pair(&pair, Some(metrics))).unwrap();
        let loaded = load_or_create_pair_log(&path, &header).unwrap();
        assert_eq!(loaded, vec![(pair, Some(metrics))]);

        let committed = fs::read(&path).unwrap();
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(br#"{"schema":"alphamini-paired-opening"#)
            .unwrap();
        file.sync_all().unwrap();
        drop(file);
        let recovered = load_or_create_pair_log(&path, &header).unwrap();
        assert_eq!(recovered, loaded);
        assert_eq!(fs::read(&path).unwrap(), committed);

        let mut changed = header;
        changed.time_ms = Some(9_001);
        assert!(load_or_create_pair_log(&path, &changed).is_err());
    }

    #[cfg(feature = "alphamini")]
    #[test]
    fn deployment_and_arena_freeze_the_same_baseline_identity() {
        assert_eq!(
            MINIMAX_V1_MOVE_DIGEST,
            alphamini::FROZEN_GATE_MINIMAX_V1_MOVE_DIGEST
        );
        assert_eq!(
            sha256_bytes(include_bytes!("../openings/alphamini-v1.json")),
            alphamini::FROZEN_GATE_OPENING_SUITE_SHA256
        );
        let search = SearchConfig::frozen_gate();
        assert_eq!(search.simulations, FROZEN_GATE_SIMULATIONS);
        assert_eq!(search.batch_size, FROZEN_GATE_BATCH_SIZE);
        assert_eq!(
            search.move_time,
            Some(Duration::from_millis(FROZEN_GATE_TIME_MS))
        );
        assert_eq!(
            (search.cpuct * 1_000_000.0).round() as u32,
            FROZEN_GATE_CPUCT_PPM
        );
        assert_eq!(
            (search.fpu_reduction * 1_000_000.0).round() as u32,
            FROZEN_GATE_FPU_REDUCTION_PPM
        );
    }

    #[test]
    fn parses_minigpt_options() {
        let args = parse_args(
            [
                "--minigpt-model",
                "model.onnx",
                "--minigpt-manifest",
                "manifest.json",
                "--minigpt-temperature",
                "0.25",
                "--opponent",
                "minigpt",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();
        assert_eq!(args.minigpt_model, Some(PathBuf::from("model.onnx")));
        assert_eq!(args.minigpt_manifest, Some(PathBuf::from("manifest.json")));
        assert_eq!(args.minigpt_temperature, Some(0.25));
        assert_eq!(args.opponent, "minigpt");
    }

    #[test]
    fn rejects_unknown_options() {
        let error = parse_args(["--wat", "1"].into_iter().map(str::to_string)).unwrap_err();
        assert!(error.contains("unknown option"));
    }
}
