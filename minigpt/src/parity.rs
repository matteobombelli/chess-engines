//! Checking the Rust ONNX session against the PyTorch logits frozen by
//! `minigpt_train.parity.write_parity_fixture`.
//!
//! The fixture is the authority: `parity.json` names every case, and each
//! `logits-tNNNN.f32` holds the expected `[1, T, VOCAB_SIZE]` block as
//! little-endian f32 in C order. Every recorded digest is re-checked before the
//! numbers are compared, so a corrupted fixture fails as loudly as a bad model.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::encoding::{TOKENIZER_VERSION, VOCAB_SIZE};
use crate::evaluator::{EvaluatorError, TokenEvaluator};
use crate::model_manifest::MODEL_INPUT_NAME;

pub const PARITY_FIXTURE_SCHEMA: &str = "minigpt.parity-fixture.v1";
pub const PARITY_FIXTURE_FILE: &str = "parity.json";
pub const PARITY_REPORT_SCHEMA: &str = "minigpt.parity-report.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParityFixtureV1 {
    pub schema: String,
    pub generated_at: String,
    pub tokenizer: String,
    pub vocab_size: usize,
    pub context: usize,
    pub model_sha256: String,
    pub input_name: String,
    pub input_dtype: String,
    pub logits_dtype: String,
    pub atol: f64,
    pub rtol: f64,
    pub cases: Vec<ParityCaseV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParityCaseV1 {
    pub name: String,
    pub sequence_length: usize,
    pub tokens: Vec<i64>,
    /// SHA-256 over the ONNX input exactly as the engine must build it:
    /// little-endian i64, C order, shape `[1, T]`.
    pub tokens_sha256: String,
    pub logits_path: String,
    pub logits_shape: [usize; 3],
    pub logits_sha256: String,
    pub python_ort_max_abs: f64,
}

impl ParityFixtureV1 {
    pub fn validate(&self) -> Result<(), ParityError> {
        exact("schema", &self.schema, PARITY_FIXTURE_SCHEMA)?;
        exact("tokenizer", &self.tokenizer, TOKENIZER_VERSION)?;
        exact("input_name", &self.input_name, MODEL_INPUT_NAME)?;
        exact("input_dtype", &self.input_dtype, "int64")?;
        exact("logits_dtype", &self.logits_dtype, "float32-le")?;
        if self.vocab_size != VOCAB_SIZE {
            return Err(ParityError::Contract(format!(
                "fixture vocab_size is {}, expected {VOCAB_SIZE}",
                self.vocab_size
            )));
        }
        if self.context < 2 {
            return Err(ParityError::Contract(format!(
                "fixture context is {}",
                self.context
            )));
        }
        if !self.atol.is_finite() || self.atol < 0.0 || !self.rtol.is_finite() || self.rtol < 0.0 {
            return Err(ParityError::Contract(format!(
                "fixture tolerances must be finite and non-negative: atol={} rtol={}",
                self.atol, self.rtol
            )));
        }
        if self.model_sha256.len() != 64
            || !self
                .model_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ParityError::Contract(
                "fixture model_sha256 must be 64 hexadecimal characters".to_string(),
            ));
        }
        if self.cases.is_empty() {
            return Err(ParityError::Contract("fixture has no cases".to_string()));
        }
        for case in &self.cases {
            case.validate(self.vocab_size, self.context)?;
        }
        Ok(())
    }
}

impl ParityCaseV1 {
    fn validate(&self, vocab_size: usize, context: usize) -> Result<(), ParityError> {
        let name = &self.name;
        if self.sequence_length == 0 || self.sequence_length > context {
            return Err(ParityError::Contract(format!(
                "case {name} has sequence length {} for a context of {context}",
                self.sequence_length
            )));
        }
        if self.tokens.len() != self.sequence_length {
            return Err(ParityError::Contract(format!(
                "case {name} lists {} tokens for sequence length {}",
                self.tokens.len(),
                self.sequence_length
            )));
        }
        if self.logits_shape != [1, self.sequence_length, vocab_size] {
            return Err(ParityError::Contract(format!(
                "case {name} declares logits shape {:?}",
                self.logits_shape
            )));
        }
        // The fixture directory is the only trusted root; a case must not name a
        // file outside it.
        if Path::new(&self.logits_path).components().count() != 1 {
            return Err(ParityError::Contract(format!(
                "case {name} logits_path {:?} is not a plain file name",
                self.logits_path
            )));
        }
        Ok(())
    }

    /// The input tensor bytes the digest is defined over.
    fn token_bytes(&self) -> Vec<u8> {
        self.tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect()
    }

    fn tokens_u16(&self) -> Result<Vec<u16>, ParityError> {
        self.tokens
            .iter()
            .map(|&token| {
                u16::try_from(token)
                    .ok()
                    .filter(|&token| usize::from(token) < VOCAB_SIZE)
                    .ok_or_else(|| {
                        ParityError::Contract(format!(
                            "case {} contains out-of-vocabulary token {token}",
                            self.name
                        ))
                    })
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParityReport {
    pub schema: String,
    pub model_sha256: String,
    pub atol: f64,
    pub rtol: f64,
    pub passed: bool,
    pub cases: Vec<ParityCaseReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParityCaseReport {
    pub name: String,
    pub sequence_length: usize,
    pub max_abs: f64,
    pub python_ort_max_abs: f64,
    pub passed: bool,
}

#[derive(Debug, Error)]
pub enum ParityError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid parity fixture JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parity fixture contract violation: {0}")]
    Contract(String),
    #[error(transparent)]
    Evaluator(#[from] EvaluatorError),
}

pub fn load_fixture(fixture_dir: impl AsRef<Path>) -> Result<ParityFixtureV1, ParityError> {
    let path = fixture_dir.as_ref().join(PARITY_FIXTURE_FILE);
    let bytes = read(&path)?;
    let fixture: ParityFixtureV1 = serde_json::from_slice(&bytes)?;
    fixture.validate()?;
    Ok(fixture)
}

/// Replay every fixture case through `evaluator` and compare all `T * vocab`
/// logits, not just the row serving would read.
pub fn run_parity(
    fixture: &ParityFixtureV1,
    fixture_dir: impl AsRef<Path>,
    model_sha256: &str,
    evaluator: &mut dyn TokenEvaluator,
) -> Result<ParityReport, ParityError> {
    if fixture.model_sha256 != model_sha256 {
        return Err(ParityError::Contract(format!(
            "fixture was written for model {}, not {model_sha256}",
            fixture.model_sha256
        )));
    }
    let fixture_dir = fixture_dir.as_ref();
    let mut cases = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let tokens = case.tokens_u16()?;
        let digest = hex::encode(Sha256::digest(case.token_bytes()));
        if digest != case.tokens_sha256 {
            return Err(ParityError::Contract(format!(
                "case {} token digest is {digest}, recorded as {}",
                case.name, case.tokens_sha256
            )));
        }
        let expected = read_logits(fixture_dir, case)?;
        let actual = evaluator.logits(&tokens)?;
        if actual.rows() != case.sequence_length {
            return Err(ParityError::Contract(format!(
                "case {} produced {} rows, expected {}",
                case.name,
                actual.rows(),
                case.sequence_length
            )));
        }
        let max_abs = expected
            .iter()
            .zip(actual.as_slice())
            .map(|(expected, actual)| f64::from((expected - actual).abs()))
            .fold(0.0_f64, f64::max);
        // The same tolerance rule NumPy's allclose applies on the Python side.
        let passed = expected
            .iter()
            .zip(actual.as_slice())
            .all(|(expected, actual)| {
                f64::from((expected - actual).abs())
                    <= fixture.atol + fixture.rtol * f64::from(expected.abs())
            });
        cases.push(ParityCaseReport {
            name: case.name.clone(),
            sequence_length: case.sequence_length,
            max_abs,
            python_ort_max_abs: case.python_ort_max_abs,
            passed,
        });
    }
    Ok(ParityReport {
        schema: PARITY_REPORT_SCHEMA.to_string(),
        model_sha256: model_sha256.to_string(),
        atol: fixture.atol,
        rtol: fixture.rtol,
        passed: cases.iter().all(|case| case.passed),
        cases,
    })
}

fn read_logits(fixture_dir: &Path, case: &ParityCaseV1) -> Result<Vec<f32>, ParityError> {
    let path = fixture_dir.join(&case.logits_path);
    let bytes = read(&path)?;
    let expected_len = case.sequence_length * VOCAB_SIZE * 4;
    if bytes.len() != expected_len {
        return Err(ParityError::Contract(format!(
            "case {} logit file is {} bytes, expected {expected_len}",
            case.name,
            bytes.len()
        )));
    }
    let digest = hex::encode(Sha256::digest(&bytes));
    if digest != case.logits_sha256 {
        return Err(ParityError::Contract(format!(
            "case {} logit digest is {digest}, recorded as {}",
            case.name, case.logits_sha256
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read(path: &Path) -> Result<Vec<u8>, ParityError> {
    std::fs::read(path).map_err(|source| ParityError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn exact(field: &str, actual: &str, expected: &str) -> Result<(), ParityError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ParityError::Contract(format!(
            "{field} must be {expected:?}, got {actual:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::encoding::BOS_TOKEN;
    use crate::evaluator::UniformEvaluator;

    const MODEL: &str = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";

    /// A fixture whose expected logits are the zeros `UniformEvaluator` returns.
    fn write_fixture(dir: &Path, lengths: &[usize], nudge: f32) -> ParityFixtureV1 {
        let mut cases = Vec::new();
        for &length in lengths {
            let tokens: Vec<i64> = std::iter::once(i64::from(BOS_TOKEN))
                .chain((1..length).map(|index| (index * 7 % 4_672) as i64))
                .collect();
            let mut values = vec![0.0_f32; length * VOCAB_SIZE];
            values[0] = nudge;
            let payload: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            let name = format!("t{length:04}");
            let logits_path = format!("logits-{name}.f32");
            fs::write(dir.join(&logits_path), &payload).unwrap();
            let token_bytes: Vec<u8> = tokens
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect();
            cases.push(ParityCaseV1 {
                name,
                sequence_length: length,
                tokens_sha256: hex::encode(Sha256::digest(&token_bytes)),
                tokens,
                logits_path,
                logits_shape: [1, length, VOCAB_SIZE],
                logits_sha256: hex::encode(Sha256::digest(&payload)),
                python_ort_max_abs: 0.0,
            });
        }
        let fixture = ParityFixtureV1 {
            schema: PARITY_FIXTURE_SCHEMA.to_string(),
            generated_at: "2026-08-26T00:00:00Z".to_string(),
            tokenizer: TOKENIZER_VERSION.to_string(),
            vocab_size: VOCAB_SIZE,
            context: 256,
            model_sha256: MODEL.to_string(),
            input_name: MODEL_INPUT_NAME.to_string(),
            input_dtype: "int64".to_string(),
            logits_dtype: "float32-le".to_string(),
            atol: 1e-3,
            rtol: 0.0,
            cases,
        };
        fs::write(
            dir.join(PARITY_FIXTURE_FILE),
            serde_json::to_vec(&fixture).unwrap(),
        )
        .unwrap();
        fixture
    }

    #[test]
    fn a_matching_model_passes_every_case() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &[1, 4, 64], 0.0);
        let fixture = load_fixture(dir.path()).unwrap();
        let report = run_parity(&fixture, dir.path(), MODEL, &mut UniformEvaluator).unwrap();
        assert!(report.passed);
        assert_eq!(report.cases.len(), 3);
        assert!(report.cases.iter().all(|case| case.max_abs == 0.0));
    }

    #[test]
    fn a_drifting_model_fails_the_case_that_drifted() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &[4], 0.5);
        let fixture = load_fixture(dir.path()).unwrap();
        let report = run_parity(&fixture, dir.path(), MODEL, &mut UniformEvaluator).unwrap();
        assert!(!report.passed);
        assert!((report.cases[0].max_abs - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_fixture_for_another_model_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &[1], 0.0);
        let fixture = load_fixture(dir.path()).unwrap();
        assert!(run_parity(&fixture, dir.path(), &"a".repeat(64), &mut UniformEvaluator).is_err());
    }

    #[test]
    fn corrupted_logit_files_and_digests_are_caught() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &[4], 0.0);
        let fixture = load_fixture(dir.path()).unwrap();
        let path = dir.path().join(&fixture.cases[0].logits_path);
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();
        assert!(run_parity(&fixture, dir.path(), MODEL, &mut UniformEvaluator).is_err());

        fs::write(&path, &bytes[..8]).unwrap();
        assert!(run_parity(&fixture, dir.path(), MODEL, &mut UniformEvaluator).is_err());
    }

    #[test]
    fn schema_and_path_escapes_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = write_fixture(dir.path(), &[1], 0.0);
        fixture.validate().unwrap();

        let mut escaping = fixture.clone();
        escaping.cases[0].logits_path = "../secret.f32".to_string();
        assert!(escaping.validate().is_err());

        let mut wrong_tokenizer = fixture.clone();
        wrong_tokenizer.tokenizer = "policy-v2".to_string();
        assert!(wrong_tokenizer.validate().is_err());

        fixture.cases[0].tokens.push(0);
        assert!(fixture.validate().is_err());
    }
}
