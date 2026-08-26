use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use minigpt::evaluator::OnnxEvaluator;
use minigpt::model_manifest::{MODEL_MANIFEST_FILE, ValidatedModel};
use minigpt::parity::{load_fixture, run_parity};

#[derive(Debug, Parser)]
#[command(
    name = "minigpt-fixtures",
    about = "Check the ONNX model against the frozen PyTorch parity fixtures"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Defaults to `manifest.json` beside the model.
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Directory written by `minigpt_train.parity.write_parity_fixture`.
    #[arg(long)]
    fixtures: PathBuf,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("minigpt-fixtures: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let args = Args::parse();
    let manifest_path = args
        .manifest
        .clone()
        .unwrap_or_else(|| args.model.with_file_name(MODEL_MANIFEST_FILE));
    let validated = ValidatedModel::load(&args.model, &manifest_path)?;
    let fixture = load_fixture(&args.fixtures)?;
    if fixture.context != validated.manifest.context {
        return Err(format!(
            "fixture context {} does not match manifest context {}",
            fixture.context, validated.manifest.context
        )
        .into());
    }
    let mut evaluator = OnnxEvaluator::load(&validated)?;
    let report = run_parity(
        &fixture,
        &args.fixtures,
        &validated.manifest.model_sha256,
        &mut evaluator,
    )?;
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    if !report.passed {
        eprintln!("minigpt-fixtures: ONNX logits drifted beyond the fixture tolerance");
    }
    Ok(report.passed)
}
