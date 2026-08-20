use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use artifact_io::{sha256_bytes as shared_sha256_bytes, sha256_file as shared_sha256_file};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::encoding::{ENCODER_VERSION, INPUT_PLANES};
use crate::policy::{POLICY_SIZE, POLICY_VERSION};

pub const MODEL_MANIFEST_VERSION: &str = "model-manifest-v1";
pub const GATE_VERDICT_VERSION: &str = "alphamini-gate-verdict-v1";
pub const FROZEN_GATE_OPENING_SUITE_SHA256: &str =
    "1ea08b1451a5650737f3c73418d91efd9f85f388c91de5239bf5f808ef0c50ac";
pub const FROZEN_GATE_MINIMAX_V1_MOVE_DIGEST: u64 = 16_258_623_573_026_552_286;
pub const FROZEN_GATE_BASELINE: &str = "MinimaxDepth3V1";
pub const FROZEN_GATE_OPENING_PAIRS: usize = 200;
pub const FROZEN_GATE_SIMULATIONS: u32 = 10_000;
pub const FROZEN_GATE_TIME_MS: u64 = 9_000;
pub const FROZEN_GATE_BATCH_SIZE: usize = 8;
pub const FROZEN_GATE_MAX_PLIES: u32 = 1_000;
pub const FROZEN_GATE_BOOTSTRAP_SAMPLES: u32 = 20_000;
pub const FROZEN_GATE_BOOTSTRAP_SEED: u64 = 1;
pub const FROZEN_GATE_REQUIRED_LOWER_SCORE: f64 = 0.5;
/// Exact millionths avoid using floating-point values as persisted identities.
pub const FROZEN_GATE_CPUCT_PPM: u32 = 1_500_000;
pub const FROZEN_GATE_FPU_REDUCTION_PPM: u32 = 250_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifestV1 {
    pub schema: String,
    pub encoder_schema: String,
    pub action_schema: String,
    pub onnx_opset: u32,
    pub input_name: String,
    pub policy_output_name: String,
    pub wdl_output_name: String,
    pub input_planes: usize,
    pub policy_size: usize,
    pub wdl_size: usize,
    pub residual_channels: usize,
    pub residual_blocks: usize,
    pub cycle: u64,
    pub parent_checkpoint_sha256: Option<String>,
    pub model_sha256: String,
}

impl ModelManifestV1 {
    pub fn validate_schema(&self) -> Result<(), ManifestError> {
        exact("schema", &self.schema, MODEL_MANIFEST_VERSION)?;
        exact("encoder_schema", &self.encoder_schema, ENCODER_VERSION)?;
        exact("action_schema", &self.action_schema, POLICY_VERSION)?;
        if self.onnx_opset != 17 {
            return Err(ManifestError::Schema(format!(
                "onnx_opset must be 17, got {}",
                self.onnx_opset
            )));
        }
        if self.input_planes != INPUT_PLANES
            || self.policy_size != POLICY_SIZE
            || self.wdl_size != 3
        {
            return Err(ManifestError::Schema(format!(
                "invalid tensor contract: input={} policy={} wdl={}",
                self.input_planes, self.policy_size, self.wdl_size
            )));
        }
        exact("input_name", &self.input_name, "input")?;
        exact(
            "policy_output_name",
            &self.policy_output_name,
            "policy_logits",
        )?;
        exact("wdl_output_name", &self.wdl_output_name, "wdl_logits")?;
        if self.residual_channels == 0 || self.residual_blocks == 0 {
            return Err(ManifestError::Schema(
                "network channel/block counts must be non-zero".to_string(),
            ));
        }
        validate_sha256("model_sha256", &self.model_sha256)?;
        if let Some(parent) = &self.parent_checkpoint_sha256 {
            validate_sha256("parent_checkpoint_sha256", parent)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ValidatedModel {
    pub model_path: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: ModelManifestV1,
}

/// Immutable arena result required before a checkpoint may be served as the
/// production AlphaMini. Earlier rungs can produce reports, but only the
/// frozen 200-pair Depth-3 contract validates here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateVerdictV1 {
    pub schema: String,
    pub passed: bool,
    pub model_sha256: String,
    pub opening_suite_sha256: String,
    pub opening_pairs: usize,
    pub baseline: String,
    pub minimax_v1_move_digest: u64,
    pub simulations: u32,
    pub time_ms: u64,
    pub batch_size: usize,
    pub cpuct_ppm: u32,
    pub fpu_reduction_ppm: u32,
    pub max_plies: u32,
    pub bootstrap_samples: u32,
    pub bootstrap_seed: u64,
    pub score: f64,
    pub lower_score: f64,
    pub upper_score: f64,
    pub required_lower_score: f64,
    pub pair_log_sha256: String,
    pub evaluation_binary_sha256: String,
    pub created_unix_seconds: u64,
}

impl GateVerdictV1 {
    pub fn load_for_deployment(
        path: impl AsRef<Path>,
        expected_model_sha256: &str,
    ) -> Result<Self, ManifestError> {
        let path = path.as_ref();
        let reader = BufReader::new(File::open(path).map_err(|source| ManifestError::Io {
            path: path.to_path_buf(),
            source,
        })?);
        let verdict: Self = serde_json::from_reader(reader)?;
        verdict.validate_for_deployment(expected_model_sha256)?;
        Ok(verdict)
    }

    pub fn validate_for_deployment(
        &self,
        expected_model_sha256: &str,
    ) -> Result<(), ManifestError> {
        exact("gate schema", &self.schema, GATE_VERDICT_VERSION)?;
        validate_sha256("gate model_sha256", &self.model_sha256)?;
        validate_sha256("opening_suite_sha256", &self.opening_suite_sha256)?;
        validate_sha256("pair_log_sha256", &self.pair_log_sha256)?;
        validate_sha256("evaluation_binary_sha256", &self.evaluation_binary_sha256)?;
        if self.model_sha256 != expected_model_sha256 {
            return Err(ManifestError::Schema(
                "gate verdict belongs to a different model".to_string(),
            ));
        }
        if !self.passed
            || self.opening_suite_sha256 != FROZEN_GATE_OPENING_SUITE_SHA256
            || self.opening_pairs != FROZEN_GATE_OPENING_PAIRS
            || self.baseline != FROZEN_GATE_BASELINE
            || self.minimax_v1_move_digest != FROZEN_GATE_MINIMAX_V1_MOVE_DIGEST
            || self.simulations != FROZEN_GATE_SIMULATIONS
            || self.time_ms != FROZEN_GATE_TIME_MS
            || self.batch_size != FROZEN_GATE_BATCH_SIZE
            || self.cpuct_ppm != FROZEN_GATE_CPUCT_PPM
            || self.fpu_reduction_ppm != FROZEN_GATE_FPU_REDUCTION_PPM
            || self.max_plies != FROZEN_GATE_MAX_PLIES
            || self.bootstrap_samples != FROZEN_GATE_BOOTSTRAP_SAMPLES
            || self.bootstrap_seed != FROZEN_GATE_BOOTSTRAP_SEED
            || self.created_unix_seconds == 0
        {
            return Err(ManifestError::Schema(
                "gate verdict does not match the frozen Depth-3 release contract".to_string(),
            ));
        }
        let scores = [
            self.score,
            self.lower_score,
            self.upper_score,
            self.required_lower_score,
        ];
        if scores
            .iter()
            .any(|score| !score.is_finite() || !(0.0..=1.0).contains(score))
            || self.lower_score > self.score
            || self.score > self.upper_score
            || self.required_lower_score != FROZEN_GATE_REQUIRED_LOWER_SCORE
            || self.lower_score <= self.required_lower_score
        {
            return Err(ManifestError::Schema(
                "gate score/interval does not prove the required lower bound".to_string(),
            ));
        }
        Ok(())
    }
}

impl ValidatedModel {
    pub fn load(
        model_path: impl AsRef<Path>,
        manifest_path: impl AsRef<Path>,
    ) -> Result<Self, ManifestError> {
        let model_path = model_path.as_ref();
        let manifest_path = manifest_path.as_ref();
        let reader =
            BufReader::new(
                File::open(manifest_path).map_err(|source| ManifestError::Io {
                    path: manifest_path.to_path_buf(),
                    source,
                })?,
            );
        let manifest: ModelManifestV1 = serde_json::from_reader(reader)?;
        manifest.validate_schema()?;
        let actual = sha256_file(model_path)?;
        if actual != manifest.model_sha256 {
            return Err(ManifestError::Checksum {
                expected: manifest.model_sha256.clone(),
                actual,
            });
        }
        Ok(Self {
            model_path: model_path.to_path_buf(),
            manifest_path: manifest_path.to_path_buf(),
            manifest,
        })
    }
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid model manifest JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("incompatible model manifest: {0}")]
    Schema(String),
    #[error("model checksum mismatch: expected {expected}, got {actual}")]
    Checksum { expected: String, actual: String },
}

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String, ManifestError> {
    let path = path.as_ref();
    shared_sha256_file(path).map_err(|source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    })
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    shared_sha256_bytes(bytes)
}

fn exact(field: &str, actual: &str, expected: &str) -> Result<(), ManifestError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ManifestError::Schema(format!(
            "{field} must be {expected:?}, got {actual:?}"
        )))
    }
}

fn validate_sha256(field: &str, value: &str) -> Result<(), ManifestError> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(ManifestError::Schema(format!(
            "{field} must be a 64-character hexadecimal SHA-256"
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn manifest(hash: String) -> ModelManifestV1 {
        ModelManifestV1 {
            schema: MODEL_MANIFEST_VERSION.into(),
            encoder_schema: ENCODER_VERSION.into(),
            action_schema: POLICY_VERSION.into(),
            onnx_opset: 17,
            input_name: "input".into(),
            policy_output_name: "policy_logits".into(),
            wdl_output_name: "wdl_logits".into(),
            input_planes: INPUT_PLANES,
            policy_size: POLICY_SIZE,
            wdl_size: 3,
            residual_channels: 64,
            residual_blocks: 6,
            cycle: 0,
            parent_checkpoint_sha256: None,
            model_sha256: hash,
        }
    }

    fn passing_gate(model_sha256: String) -> GateVerdictV1 {
        GateVerdictV1 {
            schema: GATE_VERDICT_VERSION.into(),
            passed: true,
            model_sha256,
            opening_suite_sha256: FROZEN_GATE_OPENING_SUITE_SHA256.into(),
            opening_pairs: FROZEN_GATE_OPENING_PAIRS,
            baseline: FROZEN_GATE_BASELINE.into(),
            minimax_v1_move_digest: FROZEN_GATE_MINIMAX_V1_MOVE_DIGEST,
            simulations: FROZEN_GATE_SIMULATIONS,
            time_ms: FROZEN_GATE_TIME_MS,
            batch_size: FROZEN_GATE_BATCH_SIZE,
            cpuct_ppm: FROZEN_GATE_CPUCT_PPM,
            fpu_reduction_ppm: FROZEN_GATE_FPU_REDUCTION_PPM,
            max_plies: FROZEN_GATE_MAX_PLIES,
            bootstrap_samples: FROZEN_GATE_BOOTSTRAP_SAMPLES,
            bootstrap_seed: FROZEN_GATE_BOOTSTRAP_SEED,
            score: 0.55,
            lower_score: 0.51,
            upper_score: 0.59,
            required_lower_score: FROZEN_GATE_REQUIRED_LOWER_SCORE,
            pair_log_sha256: "b".repeat(64),
            evaluation_binary_sha256: "c".repeat(64),
            created_unix_seconds: 1,
        }
    }

    #[test]
    fn validates_file_checksum_and_rejects_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        File::create(&model_path)
            .unwrap()
            .write_all(b"model")
            .unwrap();
        let manifest_path = dir.path().join("model.json");
        serde_json::to_writer(
            File::create(&manifest_path).unwrap(),
            &manifest(sha256_file(&model_path).unwrap()),
        )
        .unwrap();
        ValidatedModel::load(&model_path, &manifest_path).unwrap();

        File::create(&model_path)
            .unwrap()
            .write_all(b"changed")
            .unwrap();
        assert!(matches!(
            ValidatedModel::load(&model_path, &manifest_path),
            Err(ManifestError::Checksum { .. })
        ));
    }

    #[test]
    fn deployment_gate_is_bound_to_model_and_frozen_depth_three_contract() {
        let model = "a".repeat(64);
        let verdict = passing_gate(model.clone());
        verdict.validate_for_deployment(&model).unwrap();

        let mut failed = verdict.clone();
        failed.passed = false;
        assert!(failed.validate_for_deployment(&model).is_err());

        let mut weak = verdict.clone();
        weak.lower_score = 0.5;
        assert!(weak.validate_for_deployment(&model).is_err());

        let mut search_drift = verdict.clone();
        search_drift.cpuct_ppm += 1;
        assert!(search_drift.validate_for_deployment(&model).is_err());

        assert!(verdict.validate_for_deployment(&"d".repeat(64)).is_err());
    }
}
