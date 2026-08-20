use std::path::PathBuf;

use alphamini::evaluator::UniformEvaluator;
use alphamini::manifest::sha256_bytes;
use alphamini::record::{
    COLLECTION_MANIFEST_VERSION, CollectionManifestV1, SHARD_VERSION, SelfPlayShardV1,
    derive_game_seed, materialize_collection, write_collection_manifest_atomic, write_shard_atomic,
};
use alphamini::self_play::{SelfPlayConfig, play_game};
use alphamini::{encoding::ENCODER_VERSION, policy::POLICY_VERSION};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "alphamini-fixtures",
    about = "Write a deterministic Rust/Python contract fixture"
)]
struct Args {
    output_dir: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;
    let model_hash = sha256_bytes(b"alphamini-fixture-model-v1");
    let config_hash = sha256_bytes(b"alphamini-fixture-config-v1");
    let mut evaluator = UniformEvaluator;
    let game = play_game(
        100,
        derive_game_seed(7, 100),
        &model_hash,
        SelfPlayConfig {
            simulations: 2,
            batch_size: 2,
            sample_through_ply: 1,
            max_plies: 2,
            ..SelfPlayConfig::default()
        },
        &mut evaluator,
    )?;
    let position_count = game.positions.len() as u64;
    let shard_path = args
        .output_dir
        .join("shard-00000000000000000100-00000000000000000100.msgpack.zst");
    let descriptor = write_shard_atomic(
        &shard_path,
        &SelfPlayShardV1 {
            schema: SHARD_VERSION.to_string(),
            encoder_schema: ENCODER_VERSION.to_string(),
            action_schema: POLICY_VERSION.to_string(),
            seed: 7,
            simulations: 2,
            max_plies: 2,
            games: vec![game],
        },
    )?;
    let collection_path = args.output_dir.join("collection.json");
    write_collection_manifest_atomic(
        &collection_path,
        &CollectionManifestV1 {
            schema: COLLECTION_MANIFEST_VERSION.to_string(),
            encoder_schema: ENCODER_VERSION.to_string(),
            action_schema: POLICY_VERSION.to_string(),
            run_id: "fixture".to_string(),
            cycle_id: 0,
            game_id_start: 100,
            model_sha256: model_hash,
            config_sha256: config_hash,
            seed: 7,
            simulations: 2,
            max_plies: 2,
            game_count: 1,
            position_count,
            shards: vec![descriptor],
        },
    )?;
    let tensors_dir = args.output_dir.join("tensors");
    let tensors_path = args.output_dir.join("tensors.json");
    materialize_collection(&collection_path, &tensors_dir, &tensors_path)?;
    println!("{}", tensors_path.display());
    Ok(())
}
