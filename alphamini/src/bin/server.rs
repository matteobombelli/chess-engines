use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use alphamini::evaluator::{Evaluator, UniformEvaluator};
use alphamini::http::{AppState, router};
use alphamini::manifest::{
    FROZEN_GATE_BATCH_SIZE, FROZEN_GATE_SIMULATIONS, FROZEN_GATE_TIME_MS, GateVerdictV1,
    ValidatedModel,
};
use alphamini::mcts::SearchConfig;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "alphamini", about = "Serve legal AlphaMini moves over HTTP")]
struct Args {
    #[arg(
        long,
        env = "ALPHAMINI_MODEL_PATH",
        required_unless_present = "uniform"
    )]
    model: Option<PathBuf>,
    #[arg(long, env = "ALPHAMINI_MANIFEST_PATH")]
    manifest: Option<PathBuf>,
    #[arg(long, env = "ALPHAMINI_GATE_PATH")]
    gate: Option<PathBuf>,
    /// Explicit smoke-test mode; never selected as a model-loading fallback.
    #[arg(long, conflicts_with = "model")]
    uniform: bool,
    #[arg(long, env = "ALPHAMINI_BIND_ADDRESS", default_value = "127.0.0.1:3006")]
    bind: SocketAddr,
    #[arg(
        long,
        env = "ALPHAMINI_MAX_SIMULATIONS",
        default_value_t = FROZEN_GATE_SIMULATIONS
    )]
    simulations: u32,
    #[arg(
        long,
        env = "ALPHAMINI_BATCH_SIZE",
        default_value_t = FROZEN_GATE_BATCH_SIZE
    )]
    batch_size: usize,
    #[arg(
        long,
        env = "ALPHAMINI_MOVE_TIME_MS",
        default_value_t = FROZEN_GATE_TIME_MS
    )]
    move_time_ms: u64,
    #[arg(long, env = "ALPHAMINI_MAX_CONCURRENT_SEARCHES", default_value_t = 1)]
    max_concurrent_searches: usize,
    /// Validate model, manifest, ONNX session, and the gate verdict if one is
    /// given, then exit.
    #[arg(long)]
    verify_only: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.max_concurrent_searches != 1 {
        return Err(
            "v1 owns one evaluator and requires ALPHAMINI_MAX_CONCURRENT_SEARCHES=1".into(),
        );
    }
    // A gate verdict only certifies the budget it was measured at, so supplying
    // one pins serving to the frozen search parameters.
    let gated = !args.uniform && args.gate.is_some();
    if gated
        && (args.simulations != FROZEN_GATE_SIMULATIONS
            || args.move_time_ms != FROZEN_GATE_TIME_MS
            || args.batch_size != FROZEN_GATE_BATCH_SIZE)
    {
        return Err("gated serving freezes 10000 simulations, 9000 ms, and batch size 8".into());
    }
    let evaluator = load_evaluator(&args)?;
    if args.verify_only {
        eprintln!(
            "AlphaMini model, manifest,{} and inference session are valid",
            if gated { " gate verdict," } else { "" }
        );
        return Ok(());
    }
    let search = if gated {
        SearchConfig::frozen_gate()
    } else {
        SearchConfig::evaluation(
            args.simulations,
            args.batch_size,
            Duration::from_millis(args.move_time_ms),
        )
    };
    let state = AppState::new(evaluator, search)?;
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    eprintln!("alphamini listening on {}", listener.local_addr()?);
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

fn load_evaluator(args: &Args) -> Result<Box<dyn Evaluator>, Box<dyn std::error::Error>> {
    if args.uniform {
        return Ok(Box::new(UniformEvaluator));
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
    if let Some(gate_path) = args.gate.as_ref() {
        GateVerdictV1::load_for_deployment(gate_path, &validated.manifest.model_sha256)?;
    }
    #[cfg(feature = "onnx")]
    {
        Ok(Box::new(alphamini::evaluator::OnnxEvaluator::load(
            &validated,
        )?))
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = validated;
        Err("this binary was built without the `onnx` feature; refusing to fall back".into())
    }
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
}
