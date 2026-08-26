use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use minigpt::evaluator::OnnxEvaluator;
use minigpt::http::{AppState, DecodeConfig, router};
use minigpt::model_manifest::{MODEL_MANIFEST_FILE, ValidatedModel};

#[derive(Debug, Parser)]
#[command(name = "minigpt", about = "Serve legal MiniGPT moves over HTTP")]
struct Args {
    #[arg(long, env = "MINIGPT_MODEL_PATH")]
    model: PathBuf,
    /// Defaults to `manifest.json` beside the model.
    #[arg(long, env = "MINIGPT_MANIFEST_PATH")]
    manifest: Option<PathBuf>,
    #[arg(long, env = "MINIGPT_BIND_ADDRESS", default_value = "127.0.0.1:3008")]
    bind: SocketAddr,
    /// Override the manifest's published sampling temperature; zero is greedy.
    #[arg(long, env = "MINIGPT_TEMPERATURE")]
    temperature: Option<f32>,
    /// Validate the model, manifest, and ONNX session, then exit.
    #[arg(long)]
    verify_only: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let manifest_path = args
        .manifest
        .clone()
        .unwrap_or_else(|| args.model.with_file_name(MODEL_MANIFEST_FILE));
    let validated = ValidatedModel::load(&args.model, &manifest_path)?;
    let mut decode = DecodeConfig::from_manifest(&validated.manifest);
    if let Some(temperature) = args.temperature {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err("--temperature must be finite and non-negative".into());
        }
        decode.temperature = temperature;
    }
    let evaluator = OnnxEvaluator::load(&validated)?;
    if args.verify_only {
        eprintln!("minigpt model, manifest, and inference session are valid");
        return Ok(());
    }

    let state = AppState::new(Box::new(evaluator), decode);
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    eprintln!(
        "minigpt listening on {} (context {}, temperature {})",
        listener.local_addr()?,
        decode.context,
        decode.temperature
    );
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
}
