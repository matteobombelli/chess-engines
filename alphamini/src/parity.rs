//! Fixed-input, full-output inference parity contract.
//!
//! The emitted input is intentionally part of the report: Python can feed the
//! exact Rust-produced NCHW tensor to PyTorch and Python ONNX Runtime without
//! reimplementing chess state or the encoder.

use chess_core::Board;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::encoding::{ENCODER_VERSION, INPUT_PLANES, encode};
use crate::evaluator::{Evaluator, EvaluatorError};
use crate::policy::{POLICY_SIZE, POLICY_VERSION};

pub const INFERENCE_PARITY_SCHEMA: &str = "alphamini-inference-parity-v1";
pub const INFERENCE_PARITY_SAN: &str = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3";
pub const INFERENCE_PARITY_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 5 3";
pub const INFERENCE_PARITY_INPUT_SHA256: &str =
    "a3c8eb105e9af08a4bb13315141f289af83f1ebfc9059ca6c19070a6f6976d7a";

/// Machine-readable model-boundary evidence. Float arrays retain their raw
/// tensor order: NCHW input, then the flattened `[1, 73, 8, 8]` policy head.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceParityV1 {
    pub schema: String,
    pub device: String,
    pub cuda_device: Option<i32>,
    pub model_sha256: String,
    pub encoder_schema: String,
    pub action_schema: String,
    pub fen: String,
    pub input_shape: [usize; 4],
    /// SHA-256 over `input_values` encoded as contiguous little-endian f32.
    pub input_sha256: String,
    pub input_values: Vec<f32>,
    pub policy_shape: [usize; 4],
    pub policy_logits: Vec<f32>,
    pub wdl_shape: [usize; 2],
    pub wdl_logits: [f32; 3],
}

impl InferenceParityV1 {
    pub fn validate(&self) -> Result<(), ParityError> {
        if self.schema != INFERENCE_PARITY_SCHEMA
            || self.encoder_schema != ENCODER_VERSION
            || self.action_schema != POLICY_VERSION
            || self.fen != INFERENCE_PARITY_FEN
            || self.input_shape != [1, INPUT_PLANES, 8, 8]
            || self.policy_shape != [1, 73, 8, 8]
            || self.wdl_shape != [1, 3]
            || self.input_values.len() != INPUT_PLANES * 64
            || self.policy_logits.len() != POLICY_SIZE
        {
            return Err(ParityError::Contract(
                "inference parity report does not match the frozen tensor contract".to_string(),
            ));
        }
        let expected_input_sha256 = f32_le_sha256(&self.input_values);
        if self.input_sha256 != expected_input_sha256
            || self.input_sha256 != INFERENCE_PARITY_INPUT_SHA256
        {
            return Err(ParityError::Contract(format!(
                "input checksum mismatch: report={} actual={expected_input_sha256} frozen={INFERENCE_PARITY_INPUT_SHA256}",
                self.input_sha256,
            )));
        }
        if self.model_sha256.len() != 64
            || !self
                .model_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ParityError::Contract(
                "model_sha256 must be 64 hexadecimal characters".to_string(),
            ));
        }
        match (self.device.as_str(), self.cuda_device) {
            ("cpu", None) | ("cuda", Some(0..)) => {}
            _ => {
                return Err(ParityError::Contract(
                    "device must be cpu/null or cuda/non-negative-device-id".to_string(),
                ));
            }
        }
        if self
            .input_values
            .iter()
            .chain(self.policy_logits.iter())
            .chain(self.wdl_logits.iter())
            .any(|value| !value.is_finite())
        {
            return Err(ParityError::Contract(
                "parity report contains a non-finite float".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ParityError {
    #[error("could not construct the frozen inference position: {0}")]
    Position(String),
    #[error(transparent)]
    Evaluator(#[from] EvaluatorError),
    #[error("inference parity contract violation: {0}")]
    Contract(String),
}

pub fn run_inference_parity(
    evaluator: &mut dyn Evaluator,
    model_sha256: &str,
    device: &str,
    cuda_device: Option<i32>,
) -> Result<InferenceParityV1, ParityError> {
    let board = Board::import_san(INFERENCE_PARITY_SAN).map_err(ParityError::Position)?;
    if board.to_fen() != INFERENCE_PARITY_FEN {
        return Err(ParityError::Position(format!(
            "golden SAN resolved to {}, expected {INFERENCE_PARITY_FEN}",
            board.to_fen()
        )));
    }
    let input = encode(
        &board,
        crate::EncodingContext {
            prior_occurrences: board.prior_repetition_count().min(2) as u8,
        },
    );
    input.validate().map_err(ParityError::Contract)?;
    let mut evaluations = evaluator.evaluate_batch(std::slice::from_ref(&input))?;
    if evaluations.len() != 1 {
        return Err(ParityError::Contract(format!(
            "evaluator returned {} rows for a one-position batch",
            evaluations.len()
        )));
    }
    let evaluation = evaluations.pop().expect("length checked");
    evaluation.validate()?;
    let report = InferenceParityV1 {
        schema: INFERENCE_PARITY_SCHEMA.to_string(),
        device: device.to_string(),
        cuda_device,
        model_sha256: model_sha256.to_string(),
        encoder_schema: ENCODER_VERSION.to_string(),
        action_schema: POLICY_VERSION.to_string(),
        fen: board.to_fen(),
        input_shape: [1, INPUT_PLANES, 8, 8],
        input_sha256: f32_le_sha256(&input.values),
        input_values: input.values,
        policy_shape: [1, 73, 8, 8],
        policy_logits: evaluation.policy_logits.into_vec(),
        wdl_shape: [1, 3],
        wdl_logits: evaluation.wdl_logits,
    };
    report.validate()?;
    Ok(report)
}

fn f32_le_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    hex::encode(digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UniformEvaluator;

    #[test]
    fn frozen_report_exercises_black_canonical_repetition_input() {
        let mut evaluator = UniformEvaluator;
        let report = run_inference_parity(&mut evaluator, &"a".repeat(64), "cpu", None).unwrap();
        assert_eq!(report.policy_logits, vec![0.0; POLICY_SIZE]);
        assert_eq!(report.wdl_logits, [0.0; 3]);
        // The SAN returns to the Nf3/Nf6 position, so this is its second occurrence.
        assert!(
            report.input_values[12 * 64..13 * 64]
                .iter()
                .all(|&v| v == 1.0)
        );
        assert!(
            report.input_values[13 * 64..14 * 64]
                .iter()
                .all(|&v| v == 0.0)
        );
        // Black is the player to move and the halfmove clock is five.
        assert!(
            report.input_values[14 * 64..15 * 64]
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(
            report.input_values[20 * 64..21 * 64]
                .iter()
                .all(|&v| v == 0.05)
        );
        report.validate().unwrap();
    }

    #[test]
    fn report_rejects_device_or_input_drift() {
        let mut evaluator = UniformEvaluator;
        let mut report =
            run_inference_parity(&mut evaluator, &"a".repeat(64), "cuda", Some(0)).unwrap();
        report.input_values[0] = 0.5;
        assert!(report.validate().is_err());
        report.input_sha256 = f32_le_sha256(&report.input_values);
        report.cuda_device = None;
        assert!(report.validate().is_err());
    }
}
