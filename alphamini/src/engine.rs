use chess_core::{Board, Move};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::evaluator::Evaluator;
use crate::mcts::{Mcts, SearchConfig, SearchError, SearchStats};

/// Small adapter surface used by arena/calibration crates without making this
/// crate depend on their traits.
pub struct AlphaMiniEngine {
    name: String,
    search: Mcts,
    evaluator: Box<dyn Evaluator>,
    rng: ChaCha8Rng,
    last_stats: Option<SearchStats>,
}

impl AlphaMiniEngine {
    pub fn new(
        name: impl Into<String>,
        evaluator: Box<dyn Evaluator>,
        search: SearchConfig,
        seed: u64,
    ) -> Result<Self, SearchError> {
        Ok(Self {
            name: name.into(),
            search: Mcts::new(search)?,
            evaluator,
            rng: ChaCha8Rng::seed_from_u64(seed),
            last_stats: None,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    #[cfg(feature = "onnx")]
    pub fn from_onnx_cpu(
        name: impl Into<String>,
        model: &crate::manifest::ValidatedModel,
        search: SearchConfig,
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let evaluator = crate::evaluator::OnnxEvaluator::load(model)?;
        Ok(Self::new(name, Box::new(evaluator), search, seed)?)
    }

    #[cfg(feature = "cuda")]
    pub fn from_onnx_cuda(
        name: impl Into<String>,
        model: &crate::manifest::ValidatedModel,
        search: SearchConfig,
        seed: u64,
        device_id: i32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let evaluator = crate::evaluator::OnnxEvaluator::load_cuda(model, device_id)?;
        Ok(Self::new(name, Box::new(evaluator), search, seed)?)
    }

    pub fn choose_move(&mut self, board: &Board) -> Result<Move, SearchError> {
        let result = self
            .search
            .search(board, self.evaluator.as_mut(), &mut self.rng)?;
        self.last_stats = Some(result.stats);
        Ok(result.best_move)
    }

    pub fn last_search_stats(&self) -> Option<&SearchStats> {
        self.last_stats.as_ref()
    }
}
