//! The published model contract: `manifest.json` beside a `model.onnx`.
//!
//! Written by `minigpt_train.export.export_onnx`. Unknown fields are denied so a
//! training-side schema change fails loudly here instead of being served.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::encoding::{BOS_TOKEN, PAD_TOKEN, TOKENIZER_VERSION, VOCAB_SIZE};
use alphamini::policy::POLICY_SIZE;

pub const MODEL_MANIFEST_VERSION: &str = "minigpt.manifest.v1";
pub const MODEL_MANIFEST_FILE: &str = "manifest.json";
pub const MODEL_INPUT_NAME: &str = "tokens";
pub const MODEL_OUTPUT_NAME: &str = "logits";
pub const MODEL_ONNX_OPSET: u32 = 17;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifestV1 {
    pub schema: String,
    pub tokenizer: String,
    pub onnx_opset: u32,
    pub input_name: String,
    pub output_name: String,
    pub vocab_size: usize,
    /// Position-embedding rows, so the served sequence may never exceed it.
    pub context: usize,
    pub bos_token: u16,
    pub pad_token: u16,
    pub policy_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub decode_temperature: f32,
    pub model_sha256: String,
}

impl ModelManifestV1 {
    pub fn validate_schema(&self) -> Result<(), ManifestError> {
        exact("schema", &self.schema, MODEL_MANIFEST_VERSION)?;
        exact("tokenizer", &self.tokenizer, TOKENIZER_VERSION)?;
        exact("input_name", &self.input_name, MODEL_INPUT_NAME)?;
        exact("output_name", &self.output_name, MODEL_OUTPUT_NAME)?;
        if self.onnx_opset != MODEL_ONNX_OPSET {
            return Err(ManifestError::Schema(format!(
                "onnx_opset must be {MODEL_ONNX_OPSET}, got {}",
                self.onnx_opset
            )));
        }
        if self.vocab_size != VOCAB_SIZE
            || self.policy_size != POLICY_SIZE
            || self.bos_token != BOS_TOKEN
            || self.pad_token != PAD_TOKEN
        {
            return Err(ManifestError::Schema(format!(
                "invalid token contract: vocab={} policy={} bos={} pad={}",
                self.vocab_size, self.policy_size, self.bos_token, self.pad_token
            )));
        }
        // BOS plus at least one move must fit, or nothing can ever be decoded.
        if self.context < 2 {
            return Err(ManifestError::Schema(format!(
                "context must hold BOS plus a move, got {}",
                self.context
            )));
        }
        if self.d_model == 0
            || self.n_layers == 0
            || self.n_heads == 0
            || self.d_ff == 0
            || !self.d_model.is_multiple_of(self.n_heads)
        {
            return Err(ManifestError::Schema(format!(
                "invalid architecture: d_model={} n_layers={} n_heads={} d_ff={}",
                self.d_model, self.n_layers, self.n_heads, self.d_ff
            )));
        }
        if !self.decode_temperature.is_finite() || self.decode_temperature < 0.0 {
            return Err(ManifestError::Schema(format!(
                "decode_temperature must be finite and non-negative, got {}",
                self.decode_temperature
            )));
        }
        validate_sha256("model_sha256", &self.model_sha256)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ValidatedModel {
    pub model_path: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: ModelManifestV1,
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
    artifact_io::sha256_file(path).map_err(|source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    })
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

    fn manifest(model_sha256: String) -> ModelManifestV1 {
        ModelManifestV1 {
            schema: MODEL_MANIFEST_VERSION.into(),
            tokenizer: TOKENIZER_VERSION.into(),
            onnx_opset: MODEL_ONNX_OPSET,
            input_name: MODEL_INPUT_NAME.into(),
            output_name: MODEL_OUTPUT_NAME.into(),
            vocab_size: VOCAB_SIZE,
            context: 256,
            bos_token: BOS_TOKEN,
            pad_token: PAD_TOKEN,
            policy_size: POLICY_SIZE,
            d_model: 512,
            n_layers: 12,
            n_heads: 8,
            d_ff: 2_048,
            decode_temperature: 0.5,
            model_sha256,
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
        let manifest_path = dir.path().join(MODEL_MANIFEST_FILE);
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
    fn token_and_architecture_drift_is_rejected() {
        let good = manifest("a".repeat(64));
        good.validate_schema().unwrap();

        for mutate in [
            (|m: &mut ModelManifestV1| m.tokenizer = "policy-v2".into())
                as fn(&mut ModelManifestV1),
            |m| m.vocab_size = 4_737,
            |m| m.policy_size = 4_671,
            |m| m.bos_token = 0,
            |m| m.pad_token = 0,
            |m| m.onnx_opset = 18,
            |m| m.input_name = "input".into(),
            |m| m.output_name = "policy_logits".into(),
            |m| m.context = 1,
            |m| m.n_heads = 7,
            |m| m.d_ff = 0,
            |m| m.decode_temperature = f32::NAN,
            |m| m.model_sha256 = "nope".into(),
        ] {
            let mut broken = good.clone();
            mutate(&mut broken);
            assert!(
                broken.validate_schema().is_err(),
                "mutation should have been rejected: {broken:?}"
            );
        }
    }

    #[test]
    fn unknown_manifest_fields_fail_closed() {
        let mut value = serde_json::to_value(manifest("a".repeat(64))).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("quantized".into(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<ModelManifestV1>(value).is_err());
    }
}
