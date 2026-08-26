//! One forward pass over a token sequence.
//!
//! The graph published by `minigpt-train` takes `tokens: int64 [1, T]` and
//! returns `logits: float32 [1, T, VOCAB_SIZE]`. Only the last row matters for
//! serving, but parity compares every row, so the whole tensor is exposed.

use std::fmt;
use std::sync::Arc;

use thiserror::Error;

use crate::encoding::VOCAB_SIZE;

/// A `[T, VOCAB_SIZE]` logit block.
///
/// The ONNX backing keeps the runtime tensor alive instead of copying it: a
/// full-context pass is over a million floats, and serving only reads one row.
#[derive(Clone)]
enum LogitsBacking {
    Owned(Arc<[f32]>),
    #[cfg(feature = "onnx")]
    Onnx(Arc<ort::value::DynValue>),
}

impl LogitsBacking {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Owned(values) => values,
            #[cfg(feature = "onnx")]
            Self::Onnx(value) => {
                value
                    .try_extract_tensor::<f32>()
                    .expect("validated ONNX logit backing remains an f32 tensor")
                    .1
            }
        }
    }
}

#[derive(Clone)]
pub struct Logits {
    backing: LogitsBacking,
    rows: usize,
}

impl Logits {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn as_slice(&self) -> &[f32] {
        self.backing.as_slice()
    }

    pub fn row(&self, index: usize) -> &[f32] {
        &self.as_slice()[index * VOCAB_SIZE..(index + 1) * VOCAB_SIZE]
    }

    /// The next-move distribution: the prediction made from the whole prefix.
    pub fn last_row(&self) -> &[f32] {
        self.row(self.rows - 1)
    }

    pub fn from_values(values: Vec<f32>, rows: usize) -> Result<Self, EvaluatorError> {
        Self::from_backing(LogitsBacking::Owned(values.into()), rows)
    }

    #[cfg(feature = "onnx")]
    fn from_onnx(value: ort::value::DynValue, rows: usize) -> Result<Self, EvaluatorError> {
        value.try_extract_tensor::<f32>().map_err(|error| {
            EvaluatorError::Contract(format!("logits output is not an f32 tensor: {error}"))
        })?;
        Self::from_backing(LogitsBacking::Onnx(Arc::new(value)), rows)
    }

    fn from_backing(backing: LogitsBacking, rows: usize) -> Result<Self, EvaluatorError> {
        if rows == 0 {
            return Err(EvaluatorError::Contract(
                "model returned no logit rows".to_string(),
            ));
        }
        let expected = rows.checked_mul(VOCAB_SIZE).ok_or_else(|| {
            EvaluatorError::Contract("logit dimensions overflow usize".to_string())
        })?;
        let values = backing.as_slice();
        if values.len() != expected {
            return Err(EvaluatorError::Contract(format!(
                "model returned {} logits, expected {expected} for {rows} rows",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(EvaluatorError::Contract(
                "model returned a non-finite logit".to_string(),
            ));
        }
        Ok(Self { backing, rows })
    }
}

impl fmt::Debug for Logits {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Logits")
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum EvaluatorError {
    #[error("model contract violation: {0}")]
    Contract(String),
    #[error("model evaluation failed: {0}")]
    Runtime(String),
}

/// Serving evaluates one game at a time, so the graph's batch axis stays 1 and
/// this interface takes a single sequence.
pub trait TokenEvaluator: Send {
    fn logits(&mut self, tokens: &[u16]) -> Result<Logits, EvaluatorError>;
}

/// Deterministic smoke-test evaluator returning flat logits. It must be selected
/// explicitly; production ONNX loading never falls back to it.
#[derive(Clone, Debug, Default)]
pub struct UniformEvaluator;

impl TokenEvaluator for UniformEvaluator {
    fn logits(&mut self, tokens: &[u16]) -> Result<Logits, EvaluatorError> {
        validate_tokens(tokens)?;
        Logits::from_values(vec![0.0; tokens.len() * VOCAB_SIZE], tokens.len())
    }
}

pub(crate) fn validate_tokens(tokens: &[u16]) -> Result<(), EvaluatorError> {
    if tokens.is_empty() {
        return Err(EvaluatorError::Contract(
            "the token sequence must contain at least the BOS token".to_string(),
        ));
    }
    if let Some(&token) = tokens
        .iter()
        .find(|&&token| usize::from(token) >= VOCAB_SIZE)
    {
        return Err(EvaluatorError::Contract(format!(
            "token {token} is outside the {VOCAB_SIZE}-entry vocabulary"
        )));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
mod onnx {
    use ort::session::Session;
    use ort::value::Tensor;

    use super::*;
    use crate::model_manifest::ValidatedModel;

    pub struct OnnxEvaluator {
        session: Session,
        input_name: String,
        output_name: String,
        context: usize,
    }

    impl OnnxEvaluator {
        /// CPU only: MiniGPT serves a single game per request, where a 40M-parameter
        /// forward pass is already inside the move budget.
        pub fn load(model: &ValidatedModel) -> Result<Self, EvaluatorError> {
            let session = Session::builder()
                .map_err(runtime)?
                .commit_from_file(&model.model_path)
                .map_err(runtime)?;
            let input_name = model.manifest.input_name.clone();
            let output_name = model.manifest.output_name.clone();
            if !session
                .inputs()
                .iter()
                .any(|input| input.name() == input_name)
            {
                return Err(EvaluatorError::Contract(format!(
                    "ONNX graph does not contain manifest input {input_name:?}"
                )));
            }
            if !session
                .outputs()
                .iter()
                .any(|output| output.name() == output_name)
            {
                return Err(EvaluatorError::Contract(format!(
                    "ONNX graph does not contain manifest output {output_name:?}"
                )));
            }
            Ok(Self {
                session,
                input_name,
                output_name,
                context: model.manifest.context,
            })
        }
    }

    impl TokenEvaluator for OnnxEvaluator {
        fn logits(&mut self, tokens: &[u16]) -> Result<Logits, EvaluatorError> {
            validate_tokens(tokens)?;
            if tokens.len() > self.context {
                return Err(EvaluatorError::Contract(format!(
                    "sequence of {} tokens exceeds the {} position embeddings",
                    tokens.len(),
                    self.context
                )));
            }
            let values: Box<[i64]> = tokens.iter().map(|&token| i64::from(token)).collect();
            let input = Tensor::from_array(([1_usize, tokens.len()], values)).map_err(runtime)?;
            let mut outputs = self
                .session
                .run(ort::inputs![self.input_name.as_str() => input])
                .map_err(runtime)?;
            let output = outputs.remove(self.output_name.as_str()).ok_or_else(|| {
                EvaluatorError::Contract(format!(
                    "ONNX result omitted output {:?}",
                    self.output_name
                ))
            })?;
            let (shape, _) = output.try_extract_tensor::<f32>().map_err(runtime)?;
            let expected = [1_i64, tokens.len() as i64, VOCAB_SIZE as i64];
            if shape.as_ref() != expected {
                return Err(EvaluatorError::Contract(format!(
                    "logits have shape {:?}, expected {expected:?}",
                    shape.as_ref()
                )));
            }
            Logits::from_onnx(output, tokens.len())
        }
    }

    fn runtime(error: impl std::fmt::Display) -> EvaluatorError {
        EvaluatorError::Runtime(error.to_string())
    }
}

#[cfg(feature = "onnx")]
pub use onnx::OnnxEvaluator;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::BOS_TOKEN;

    #[test]
    fn rows_expose_only_their_own_slice() {
        let mut values = vec![0.0; 3 * VOCAB_SIZE];
        values[VOCAB_SIZE] = 1.0;
        values[3 * VOCAB_SIZE - 1] = 2.0;
        let logits = Logits::from_values(values, 3).unwrap();

        assert_eq!(logits.rows(), 3);
        assert_eq!(logits.row(1)[0], 1.0);
        assert_eq!(logits.row(0)[0], 0.0);
        assert_eq!(logits.last_row()[VOCAB_SIZE - 1], 2.0);
    }

    #[test]
    fn wrong_shapes_and_non_finite_logits_fail_closed() {
        assert!(Logits::from_values(vec![0.0; VOCAB_SIZE], 2).is_err());
        assert!(Logits::from_values(Vec::new(), 0).is_err());
        assert!(Logits::from_values(vec![f32::NAN; VOCAB_SIZE], 1).is_err());
    }

    #[test]
    fn uniform_evaluator_rejects_empty_and_out_of_vocabulary_input() {
        let mut evaluator = UniformEvaluator;
        assert_eq!(evaluator.logits(&[BOS_TOKEN]).unwrap().rows(), 1);
        assert!(evaluator.logits(&[]).is_err());
        assert!(evaluator.logits(&[VOCAB_SIZE as u16]).is_err());
    }
}
