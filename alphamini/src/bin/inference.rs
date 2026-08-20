use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use alphamini::evaluator::OnnxEvaluator;
use alphamini::{ValidatedModel, run_inference_parity};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Device {
    Cpu,
    Cuda,
}

#[derive(Debug, Parser)]
#[command(
    name = "alphamini-inference",
    about = "Emit the frozen full-tensor Rust ONNX inference parity record"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long, value_enum)]
    device: Device,
    #[arg(long, default_value_t = 0)]
    cuda_device: i32,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("alphamini-inference: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.cuda_device < 0 {
        return Err("--cuda-device must be non-negative".into());
    }
    let model = ValidatedModel::load(&args.model, &args.manifest)?;
    let report = match args.device {
        Device::Cpu => {
            let mut evaluator = OnnxEvaluator::load(&model)?;
            run_inference_parity(&mut evaluator, &model.manifest.model_sha256, "cpu", None)?
        }
        Device::Cuda => run_cuda(&model, args.cuda_device)?,
    };
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, &report)?;
    output.write_all(b"\n")?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_cuda(
    model: &ValidatedModel,
    cuda_device: i32,
) -> Result<alphamini::InferenceParityV1, Box<dyn std::error::Error>> {
    let mut evaluator = OnnxEvaluator::load_cuda(model, cuda_device)?;
    Ok(run_inference_parity(
        &mut evaluator,
        &model.manifest.model_sha256,
        "cuda",
        Some(cuda_device),
    )?)
}

#[cfg(not(feature = "cuda"))]
fn run_cuda(
    _model: &ValidatedModel,
    _cuda_device: i32,
) -> Result<alphamini::InferenceParityV1, Box<dyn std::error::Error>> {
    Err("CUDA was requested, but this binary was not built with `--features cuda`".into())
}
