use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use thiserror::Error;

use crate::encoding::EncodedPosition;
use crate::policy::POLICY_SIZE;

/// One logical policy row.
///
/// ONNX inference produces policy logits as one contiguous batched tensor.
/// Keeping that tensor behind an `Arc` lets each [`Evaluation`] retain only
/// its row range instead of allocating and copying 4,672 floats per leaf.
/// The type dereferences to `[f32]`, so callers can index and iterate it like
/// the former `Vec<f32>` field.
#[derive(Clone)]
enum PolicyBacking {
    Owned(Arc<[f32]>),
    #[cfg(feature = "onnx")]
    Onnx(Arc<ort::value::DynValue>),
}

impl PolicyBacking {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Owned(values) => values,
            #[cfg(feature = "onnx")]
            Self::Onnx(value) => {
                value
                    .try_extract_tensor::<f32>()
                    .expect("validated ONNX policy backing remains an f32 tensor")
                    .1
            }
        }
    }
}

#[derive(Clone)]
pub struct PolicyLogits {
    backing: PolicyBacking,
    start: usize,
    len: usize,
}

impl PolicyLogits {
    pub fn as_slice(&self) -> &[f32] {
        &self.backing.as_slice()[self.start..self.start + self.len]
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.as_slice().to_vec()
    }

    #[cfg(test)]
    fn from_batched_flat(values: Vec<f32>, row_count: usize) -> Result<Vec<Self>, EvaluatorError> {
        Self::from_batched_backing(PolicyBacking::Owned(values.into()), row_count)
    }

    #[cfg(feature = "onnx")]
    fn from_onnx_batched(
        value: ort::value::DynValue,
        row_count: usize,
    ) -> Result<Vec<Self>, EvaluatorError> {
        value.try_extract_tensor::<f32>().map_err(|error| {
            EvaluatorError::Contract(format!("policy output is not an f32 tensor: {error}"))
        })?;
        Self::from_batched_backing(PolicyBacking::Onnx(Arc::new(value)), row_count)
    }

    #[cfg(any(feature = "onnx", test))]
    fn from_batched_backing(
        backing: PolicyBacking,
        row_count: usize,
    ) -> Result<Vec<Self>, EvaluatorError> {
        let expected = row_count.checked_mul(POLICY_SIZE).ok_or_else(|| {
            EvaluatorError::Contract("policy batch dimensions overflow usize".to_string())
        })?;
        let actual = backing.as_slice().len();
        if actual != expected {
            return Err(EvaluatorError::Contract(format!(
                "policy batch has {} logits, expected {expected} for {row_count} rows",
                actual
            )));
        }

        Ok((0..row_count)
            .map(|row| Self {
                backing: backing.clone(),
                start: row * POLICY_SIZE,
                len: POLICY_SIZE,
            })
            .collect())
    }
}

impl From<Vec<f32>> for PolicyLogits {
    fn from(values: Vec<f32>) -> Self {
        let len = values.len();
        Self {
            backing: PolicyBacking::Owned(values.into()),
            start: 0,
            len,
        }
    }
}

impl AsRef<[f32]> for PolicyLogits {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl Deref for PolicyLogits {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl fmt::Debug for PolicyLogits {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(formatter)
    }
}

impl PartialEq for PolicyLogits {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<Vec<f32>> for PolicyLogits {
    fn eq(&self, other: &Vec<f32>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    pub policy_logits: PolicyLogits,
    /// Win/draw/loss logits for the side to move, in that order.
    pub wdl_logits: [f32; 3],
}

impl Evaluation {
    pub fn validate(&self) -> Result<(), EvaluatorError> {
        if self.policy_logits.len() != POLICY_SIZE {
            return Err(EvaluatorError::Contract(format!(
                "policy has {} logits, expected {POLICY_SIZE}",
                self.policy_logits.len()
            )));
        }
        if self
            .policy_logits
            .iter()
            .chain(self.wdl_logits.iter())
            .any(|value| !value.is_finite())
        {
            return Err(EvaluatorError::Contract(
                "model returned a non-finite logit".to_string(),
            ));
        }
        Ok(())
    }

    /// P(win)-P(loss), preserving the leaf side-to-move perspective.
    pub fn scalar_value(&self) -> f32 {
        let max = self
            .wdl_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let weights = self.wdl_logits.map(|logit| (logit - max).exp());
        let total = weights.iter().sum::<f32>();
        (weights[0] - weights[2]) / total
    }
}

#[derive(Debug, Error)]
pub enum EvaluatorError {
    #[error("model contract violation: {0}")]
    Contract(String),
    #[error("model evaluation failed: {0}")]
    Runtime(String),
}

/// A batch interface even when the current caller only has one leaf. This is
/// the seam used by the multi-game inference scheduler in the optimized core.
pub trait Evaluator: Send {
    fn evaluate_batch(
        &mut self,
        positions: &[EncodedPosition],
    ) -> Result<Vec<Evaluation>, EvaluatorError>;

    /// Evaluate an owned batch without forcing callers to retain or clone its
    /// input buffers. Evaluators that forward work across threads can override
    /// this method and move the positions directly into their request queue.
    fn evaluate_owned_batch(
        &mut self,
        positions: Vec<EncodedPosition>,
    ) -> Result<Vec<Evaluation>, EvaluatorError> {
        self.evaluate_batch(&positions)
    }
}

/// Deterministic smoke-test evaluator. It must be selected explicitly by CLI;
/// production ONNX loading never falls back to it.
#[derive(Clone, Debug, Default)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate_batch(
        &mut self,
        positions: &[EncodedPosition],
    ) -> Result<Vec<Evaluation>, EvaluatorError> {
        for position in positions {
            position.validate().map_err(EvaluatorError::Contract)?;
        }
        let policy_logits: PolicyLogits = vec![0.0; POLICY_SIZE].into();
        Ok(positions
            .iter()
            .map(|_| Evaluation {
                policy_logits: policy_logits.clone(),
                wdl_logits: [0.0, 0.0, 0.0],
            })
            .collect())
    }
}

#[cfg(feature = "onnx")]
mod onnx {
    use ort::session::Session;
    use ort::value::Tensor;

    use super::*;
    use crate::encoding::INPUT_PLANES;
    use crate::encoding::INPUT_VALUES;
    use crate::manifest::ValidatedModel;

    pub struct OnnxEvaluator {
        session: Session,
        input_name: String,
        policy_output_name: String,
        wdl_output_name: String,
    }

    impl OnnxEvaluator {
        pub fn load(model: &ValidatedModel) -> Result<Self, EvaluatorError> {
            let builder = Session::builder().map_err(runtime)?;
            Self::load_with_builder(model, builder)
        }

        #[cfg(feature = "cuda")]
        pub fn load_cuda(model: &ValidatedModel, device_id: i32) -> Result<Self, EvaluatorError> {
            use ort::ep::ExecutionProviderDispatch;

            let cuda: ExecutionProviderDispatch = ort::ep::CUDA::default()
                .with_device_id(device_id)
                .build()
                .error_on_failure();
            let builder = Session::builder()
                .map_err(runtime)?
                .with_execution_providers([cuda])
                .map_err(runtime)?
                .with_disable_cpu_fallback()
                .map_err(runtime)?;
            Self::load_with_builder(model, builder)
        }

        fn load_with_builder(
            model: &ValidatedModel,
            mut builder: ort::session::builder::SessionBuilder,
        ) -> Result<Self, EvaluatorError> {
            let session = builder
                .commit_from_file(&model.model_path)
                .map_err(runtime)?;

            let input_name = model.manifest.input_name.clone();
            let policy_output_name = model.manifest.policy_output_name.clone();
            let wdl_output_name = model.manifest.wdl_output_name.clone();
            if !session
                .inputs()
                .iter()
                .any(|input| input.name() == input_name)
            {
                return Err(EvaluatorError::Contract(format!(
                    "ONNX graph does not contain manifest input {input_name:?}"
                )));
            }
            for expected in [&policy_output_name, &wdl_output_name] {
                if !session
                    .outputs()
                    .iter()
                    .any(|output| output.name() == expected)
                {
                    return Err(EvaluatorError::Contract(format!(
                        "ONNX graph does not contain manifest output {expected:?}"
                    )));
                }
            }
            Ok(Self {
                session,
                input_name,
                policy_output_name,
                wdl_output_name,
            })
        }
    }

    impl Evaluator for OnnxEvaluator {
        fn evaluate_batch(
            &mut self,
            positions: &[EncodedPosition],
        ) -> Result<Vec<Evaluation>, EvaluatorError> {
            if positions.is_empty() {
                return Ok(Vec::new());
            }
            let mut flat = Vec::with_capacity(positions.len() * INPUT_VALUES);
            for position in positions {
                position.validate().map_err(EvaluatorError::Contract)?;
                flat.extend_from_slice(&position.values);
            }
            let input = Tensor::from_array((
                [positions.len(), INPUT_PLANES, 8, 8],
                flat.into_boxed_slice(),
            ))
            .map_err(runtime)?;
            let mut outputs = self
                .session
                .run(ort::inputs![self.input_name.as_str() => input])
                .map_err(runtime)?;
            let policy_output = outputs
                .remove(self.policy_output_name.as_str())
                .ok_or_else(|| {
                    EvaluatorError::Contract(format!(
                        "ONNX result omitted policy output {:?}",
                        self.policy_output_name
                    ))
                })?;
            let wdl_output = outputs
                .remove(self.wdl_output_name.as_str())
                .ok_or_else(|| {
                    EvaluatorError::Contract(format!(
                        "ONNX result omitted WDL output {:?}",
                        self.wdl_output_name
                    ))
                })?;
            let (_, policy) = policy_output.try_extract_tensor::<f32>().map_err(runtime)?;
            let (_, wdl) = wdl_output.try_extract_tensor::<f32>().map_err(runtime)?;
            if policy.len() != positions.len() * POLICY_SIZE || wdl.len() != positions.len() * 3 {
                return Err(EvaluatorError::Contract(format!(
                    "bad ONNX output sizes: policy={} wdl={} batch={}",
                    policy.len(),
                    wdl.len(),
                    positions.len()
                )));
            }
            let wdl_rows: Vec<_> = wdl
                .chunks_exact(3)
                .map(|row| [row[0], row[1], row[2]])
                .collect();
            let policy_rows = PolicyLogits::from_onnx_batched(policy_output, positions.len())?;
            policy_rows
                .into_iter()
                .zip(wdl_rows)
                .map(|(policy_logits, wdl_logits)| {
                    let evaluation = Evaluation {
                        policy_logits,
                        wdl_logits,
                    };
                    evaluation.validate()?;
                    Ok(evaluation)
                })
                .collect()
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

    #[test]
    fn wdl_value_uses_side_to_move_and_is_stable() {
        let win = Evaluation {
            policy_logits: vec![0.0; POLICY_SIZE].into(),
            wdl_logits: [1000.0, 0.0, -1000.0],
        };
        assert!((win.scalar_value() - 1.0).abs() < 1e-6);
        let loss = Evaluation {
            policy_logits: vec![0.0; POLICY_SIZE].into(),
            wdl_logits: [-4.0, 0.0, 4.0],
        };
        assert!(loss.scalar_value() < -0.96);
    }

    #[test]
    fn batched_policy_rows_share_storage_and_expose_only_their_row() {
        let mut flat = vec![0.0; 2 * POLICY_SIZE];
        flat[0] = 1.0;
        flat[POLICY_SIZE] = 2.0;
        flat[2 * POLICY_SIZE - 1] = 3.0;

        let rows = PolicyLogits::from_batched_flat(flat, 2).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].len(), POLICY_SIZE);
        assert_eq!(rows[1].len(), POLICY_SIZE);
        assert_eq!(rows[0][0], 1.0);
        assert_eq!(rows[0][POLICY_SIZE - 1], 0.0);
        assert_eq!(rows[1][0], 2.0);
        assert_eq!(rows[1][POLICY_SIZE - 1], 3.0);
        assert_eq!(
            rows[0].backing.as_slice().as_ptr(),
            rows[1].backing.as_slice().as_ptr()
        );
    }

    #[test]
    fn owned_policy_rows_remain_vec_ergonomic() {
        let logits: PolicyLogits = vec![4.0; POLICY_SIZE].into();
        assert_eq!(logits, vec![4.0; POLICY_SIZE]);
        assert_eq!(logits.iter().sum::<f32>(), 4.0 * POLICY_SIZE as f32);
        assert_eq!(logits.into_vec(), vec![4.0; POLICY_SIZE]);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_policy_rows_retain_the_runtime_tensor_without_copying() {
        use ort::value::Tensor;

        let mut flat = vec![0.0_f32; 2 * POLICY_SIZE];
        flat[POLICY_SIZE] = 7.0;
        let value = Tensor::from_array(([2, POLICY_SIZE], flat.into_boxed_slice()))
            .unwrap()
            .into_dyn();
        let original_pointer = value.try_extract_tensor::<f32>().unwrap().1.as_ptr();

        let rows = PolicyLogits::from_onnx_batched(value, 2).unwrap();
        assert_eq!(rows[0].backing.as_slice().as_ptr(), original_pointer);
        assert_eq!(rows[1].backing.as_slice().as_ptr(), original_pointer);
        assert_eq!(rows[1][0], 7.0);
    }
}
