use std::path::PathBuf;

use clap::Parser;
use minigpt::ingest::{IngestOptions, run};
use minigpt::manifest::{SHARDS_MANIFEST_FILE, write_manifest_atomic};

#[derive(Debug, Parser)]
#[command(
    name = "minigpt-ingest",
    about = "Stream Lichess PGN dumps into MiniGPT token shards"
)]
struct Cli {
    /// A `.pgn.zst` dump; repeat to read several, in the order given.
    #[arg(long = "dump", required = true)]
    dumps: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 2_000)]
    min_elo: u32,
    #[arg(long, default_value_t = 10)]
    min_plies: u32,
    #[arg(long, default_value_t = 300)]
    max_plies: u32,
    /// Stop once this many tokens have been written across train and validation.
    #[arg(long, default_value_t = 1_000_000_000)]
    token_target: u64,
    #[arg(long, default_value_t = 0.005)]
    val_fraction: f64,
    #[arg(long, default_value_t = 50_000_000)]
    shard_tokens: u64,
    /// Parse/tokenize workers. Defaults to one per core beyond the reader.
    #[arg(long)]
    workers: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    if !(0.0..=1.0).contains(&args.val_fraction) {
        return Err("--val-fraction must be in 0..=1".into());
    }
    let options = IngestOptions {
        min_elo: args.min_elo,
        min_plies: args.min_plies,
        max_plies: args.max_plies,
        token_target: args.token_target,
        val_fraction_ppm: (args.val_fraction * 1_000_000.0).round() as u32,
        shard_tokens: args.shard_tokens,
        workers: args.workers.unwrap_or_else(default_workers),
    };

    let manifest = run(&args.dumps, &args.out, options)?;
    write_manifest_atomic(&args.out.join(SHARDS_MANIFEST_FILE), &manifest)?;
    println!(
        "{}",
        serde_json::json!({
            "event": "minigpt_ingest_complete",
            "manifest": args.out.join(SHARDS_MANIFEST_FILE).display().to_string(),
            "games_seen": manifest.counts.games_seen,
            "games_accepted": manifest.counts.games_accepted,
            "rejected": manifest.counts.rejected,
            "tokens_train": manifest.counts.tokens_train,
            "tokens_val": manifest.counts.tokens_val,
            "train_shards": manifest.train_shards.len(),
            "val_shards": manifest.val_shards.len(),
            "san_error_samples": manifest.san_error_samples,
        })
    );
    Ok(())
}

/// One core drives zstd decoding and game splitting; the rest tokenize.
fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(|cores| cores.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}
