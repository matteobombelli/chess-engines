use std::path::{Path, PathBuf};

use alphamini::evaluator::{Evaluator, UniformEvaluator};
use alphamini::manifest::{ValidatedModel, sha256_bytes};
use alphamini::record::{
    COLLECTION_MANIFEST_VERSION, CollectionManifestV1, GameOutcomeV1, MAX_SELF_PLAY_PLIES_V1,
    SHARD_VERSION, SelfPlayShardV1, TerminationV1, derive_game_seed,
    write_collection_manifest_atomic, write_shard_atomic,
};
use alphamini::self_play::{SelfPlayConfig, play_games_batched_with_stats};
use alphamini::{encoding::ENCODER_VERSION, policy::POLICY_VERSION};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "alphamini-selfplay",
    about = "Collect or materialize AlphaMini self-play"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Collect(CollectArgs),
    Materialize(MaterializeArgs),
}

#[derive(Debug, Args)]
struct CollectArgs {
    #[arg(
        long,
        env = "ALPHAMINI_MODEL_PATH",
        required_unless_present = "uniform"
    )]
    model: Option<PathBuf>,
    #[arg(long, env = "ALPHAMINI_MANIFEST_PATH")]
    manifest: Option<PathBuf>,
    #[arg(long, conflicts_with = "model")]
    uniform: bool,
    #[arg(
        long,
        env = "ALPHAMINI_INFERENCE_DEVICE",
        value_enum,
        default_value = "cuda"
    )]
    device: InferenceDevice,
    #[arg(long, env = "ALPHAMINI_RUN_DIR")]
    run_dir: PathBuf,
    #[arg(long, env = "ALPHAMINI_RUN_ID")]
    run_id: String,
    #[arg(long, env = "ALPHAMINI_CYCLE_ID")]
    cycle_id: u64,
    #[arg(long, env = "ALPHAMINI_GAME_ID_START")]
    game_id_start: u64,
    #[arg(long, env = "ALPHAMINI_COLLECTION_DIR")]
    output_dir: PathBuf,
    #[arg(long, env = "ALPHAMINI_COLLECTION_MANIFEST")]
    collection_manifest: PathBuf,
    #[arg(long, env = "ALPHAMINI_CONFIG_SHA256")]
    config_sha256: String,
    /// Carried into process provenance by the orchestrator; the frozen file
    /// checksum above remains the collection identity.
    #[arg(long, env = "ALPHAMINI_CONFIG_JSON")]
    config_json: Option<String>,
    #[arg(long, default_value_t = 1_024)]
    games: u64,
    #[arg(long, default_value_t = 128)]
    shard_games: usize,
    #[arg(long, default_value_t = 128)]
    simulations: u32,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value_t = 512)]
    max_plies: u16,
    #[arg(long, env = "ALPHAMINI_DIRICHLET_ALPHA", default_value_t = 0.3)]
    dirichlet_alpha: f32,
    #[arg(long, env = "ALPHAMINI_DIRICHLET_EPSILON", default_value_t = 0.25)]
    dirichlet_epsilon: f32,
    #[arg(long, env = "ALPHAMINI_SAMPLE_UNTIL_PLY", default_value_t = 30)]
    sample_until_ply: u16,
    #[arg(long, env = "ALPHAMINI_CPUCT", default_value_t = 1.5)]
    cpuct: f32,
    #[arg(long, env = "ALPHAMINI_FPU_REDUCTION", default_value_t = 0.25)]
    fpu_reduction: f32,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InferenceDevice {
    Cpu,
    Cuda,
}

#[derive(Debug, Args)]
struct MaterializeArgs {
    #[arg(long, env = "ALPHAMINI_COLLECTION_MANIFEST")]
    collection_manifest: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    tensor_manifest: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Collect(args) => collect(args),
        Command::Materialize(args) => {
            let manifest = alphamini::record::materialize_collection(
                &args.collection_manifest,
                &args.output_dir,
                &args.tensor_manifest,
            )?;
            println!("materialized {} records", manifest.record_count);
            Ok(())
        }
    }
}

fn collect(args: CollectArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.games == 0
        || args.shard_games == 0
        || args.simulations == 0
        || args.max_plies == 0
        || args.max_plies > MAX_SELF_PLAY_PLIES_V1
    {
        return Err(format!(
            "games, shard-games, and simulations must be positive; max-plies must be in 1..={MAX_SELF_PLAY_PLIES_V1}"
        )
        .into());
    }
    let (mut evaluator, model_hash) = load_evaluator(&args)?;
    let _run_dir = &args.run_dir;
    let config = SelfPlayConfig {
        simulations: args.simulations,
        batch_size: 1,
        sample_through_ply: args.sample_until_ply,
        max_plies: args.max_plies,
        cpuct: args.cpuct,
        fpu_reduction: args.fpu_reduction,
        dirichlet_alpha: args.dirichlet_alpha,
        dirichlet_epsilon: args.dirichlet_epsilon,
    };
    let manifest_root = args
        .collection_manifest
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let _provenance_config = args.config_json.as_deref();
    let mut descriptors = Vec::new();
    let game_id_end = args
        .game_id_start
        .checked_add(args.games)
        .ok_or("game ID overflow")?;
    let specs: Vec<_> = (args.game_id_start..game_id_end)
        .map(|game_id| (game_id, derive_game_seed(args.seed, game_id)))
        .collect();
    // Scheduling spans the whole collection so physical shard boundaries do
    // not repeatedly drain the GPU worker cohort. Records remain sorted and
    // are partitioned into the same immutable shard sizes below.
    let (games, scheduler) = play_games_batched_with_stats(
        &specs,
        &model_hash,
        config,
        args.batch_size,
        evaluator.as_mut(),
    )?;
    let game_count = games.len() as u64;
    let position_count = games
        .iter()
        .map(|game| game.positions.len() as u64)
        .sum::<u64>();
    let mut outcomes = std::collections::BTreeMap::<&'static str, u64>::new();
    let mut terminations = std::collections::BTreeMap::<&'static str, u64>::new();
    for game in &games {
        *outcomes.entry(outcome_name(game.outcome)).or_default() += 1;
        *terminations
            .entry(termination_name(game.termination))
            .or_default() += 1;
    }
    let elapsed_seconds = scheduler.elapsed_micros as f64 / 1_000_000.0;
    eprintln!(
        "{}",
        serde_json::json!({
            // Keep the established event name for report readers. This is one
            // nonduplicative, collection-scoped scheduler aggregate even when
            // the records below are split into several physical shards.
            "event": "self_play_shard_complete",
            "telemetry_scope": "collection",
            "physical_shards": games.len().div_ceil(args.shard_games),
            "first_game_id": args.game_id_start,
            "last_game_id": game_id_end - 1,
            "games": games.len(),
            "positions": position_count,
            "elapsed_seconds": elapsed_seconds,
            "games_per_hour": if elapsed_seconds > 0.0 { games.len() as f64 * 3600.0 / elapsed_seconds } else { 0.0 },
            "worker_count": scheduler.worker_count,
            "inference_batches": scheduler.inference_batches,
            "neural_evaluations": scheduler.neural_evaluations,
            "maximum_batch": scheduler.maximum_batch,
            "batch_capacity": scheduler.requested_batch_capacity,
            "mean_batch_fill": scheduler.mean_batch_fill(),
            "batch_histogram": scheduler.batch_histogram,
            "inference_seconds": scheduler.inference_micros as f64 / 1_000_000.0,
            "completed_simulations": position_count * u64::from(args.simulations),
            "outcomes": outcomes,
            "terminations": terminations,
        })
    );

    let mut remaining_games = games.into_iter();
    loop {
        let shard_games: Vec<_> = remaining_games.by_ref().take(args.shard_games).collect();
        let Some(first_game) = shard_games.first() else {
            break;
        };
        let first_id = first_game.game_id;
        let last_id = shard_games
            .last()
            .expect("a nonempty shard has a last game")
            .game_id;
        let path = args
            .output_dir
            .join(format!("shard-{first_id:020}-{last_id:020}.msgpack.zst"));
        let mut descriptor = write_shard_atomic(
            &path,
            &SelfPlayShardV1 {
                schema: SHARD_VERSION.to_string(),
                encoder_schema: ENCODER_VERSION.to_string(),
                action_schema: POLICY_VERSION.to_string(),
                seed: args.seed,
                simulations: args.simulations,
                max_plies: args.max_plies,
                games: shard_games,
            },
        )?;
        descriptor.path = path
            .strip_prefix(manifest_root)
            .map_err(|_| "collection shards must be beneath the collection manifest directory")?
            .to_string_lossy()
            .into_owned();
        descriptors.push(descriptor);
    }

    let manifest = CollectionManifestV1 {
        schema: COLLECTION_MANIFEST_VERSION.to_string(),
        encoder_schema: ENCODER_VERSION.to_string(),
        action_schema: POLICY_VERSION.to_string(),
        run_id: args.run_id,
        cycle_id: args.cycle_id,
        game_id_start: args.game_id_start,
        model_sha256: model_hash,
        config_sha256: args.config_sha256,
        seed: args.seed,
        simulations: args.simulations,
        max_plies: args.max_plies,
        game_count,
        position_count,
        shards: descriptors,
    };
    write_collection_manifest_atomic(&args.collection_manifest, &manifest)?;
    println!(
        "sealed {} games / {} positions in {}",
        game_count,
        position_count,
        args.collection_manifest.display()
    );
    Ok(())
}

fn load_evaluator(
    args: &CollectArgs,
) -> Result<(Box<dyn Evaluator>, String), Box<dyn std::error::Error>> {
    if args.uniform {
        return Ok((
            Box::new(UniformEvaluator),
            sha256_bytes(b"alphamini-uniform-evaluator-v1"),
        ));
    }
    let model_path = args
        .model
        .as_ref()
        .ok_or("--model is required unless --uniform is explicit")?;
    let manifest_path = args
        .manifest
        .clone()
        .unwrap_or_else(|| model_path.with_file_name("manifest.json"));
    let validated = ValidatedModel::load(model_path, manifest_path)?;
    let hash = validated.manifest.model_sha256.clone();
    #[cfg(feature = "onnx")]
    {
        let evaluator: Box<dyn Evaluator> = match args.device {
            InferenceDevice::Cpu => {
                Box::new(alphamini::evaluator::OnnxEvaluator::load(&validated)?)
            }
            InferenceDevice::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Box::new(alphamini::evaluator::OnnxEvaluator::load_cuda(
                        &validated, 0,
                    )?)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(
                        "CUDA self-play requested, but binary lacks the `cuda` feature".into(),
                    );
                }
            }
        };
        Ok((evaluator, hash))
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = validated;
        let _ = hash;
        Err("this binary was built without the `onnx` feature; refusing to fall back".into())
    }
}

fn outcome_name(outcome: GameOutcomeV1) -> &'static str {
    match outcome {
        GameOutcomeV1::WhiteWin => "white_win",
        GameOutcomeV1::Draw => "draw",
        GameOutcomeV1::BlackWin => "black_win",
    }
}

fn termination_name(termination: TerminationV1) -> &'static str {
    match termination {
        TerminationV1::Checkmate => "checkmate",
        TerminationV1::Stalemate => "stalemate",
        TerminationV1::InsufficientMaterial => "insufficient_material",
        TerminationV1::ThreefoldRepetition => "threefold_repetition",
        TerminationV1::FiftyMoveRule => "fifty_move_rule",
        TerminationV1::PlyLimit => "ply_limit",
    }
}

#[cfg(test)]
mod tests {
    use alphamini::record::{materialize_collection, read_collection_manifest};

    use super::*;

    #[test]
    fn one_rolling_schedule_preserves_physical_shard_boundaries() {
        let temporary = tempfile::tempdir().unwrap();
        let collection_dir = temporary.path().join("collection");
        std::fs::create_dir(&collection_dir).unwrap();
        let collection_manifest = collection_dir.join("collection.json");
        collect(CollectArgs {
            model: None,
            manifest: None,
            uniform: true,
            device: InferenceDevice::Cpu,
            run_dir: temporary.path().to_path_buf(),
            run_id: "rolling-shard-test".to_string(),
            cycle_id: 3,
            game_id_start: 40,
            output_dir: collection_dir.clone(),
            collection_manifest: collection_manifest.clone(),
            config_sha256: "0".repeat(64),
            config_json: None,
            games: 5,
            shard_games: 2,
            simulations: 1,
            batch_size: 1,
            seed: 9,
            max_plies: 1,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            sample_until_ply: 0,
            cpuct: 1.5,
            fpu_reduction: 0.25,
        })
        .unwrap();

        let manifest = read_collection_manifest(&collection_manifest).unwrap();
        assert_eq!(manifest.game_count, 5);
        assert_eq!(manifest.position_count, 5);
        assert_eq!(
            manifest
                .shards
                .iter()
                .map(|shard| (shard.first_game_id, shard.last_game_id, shard.game_count))
                .collect::<Vec<_>>(),
            vec![(40, 41, 2), (42, 43, 2), (44, 44, 1)]
        );

        let tensor_dir = temporary.path().join("tensors");
        let tensor_manifest_path = tensor_dir.join("manifest.json");
        let tensor_manifest =
            materialize_collection(&collection_manifest, &tensor_dir, &tensor_manifest_path)
                .unwrap();
        assert_eq!(tensor_manifest.record_count, 5);
    }
}
