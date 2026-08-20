//! AlphaMini's versioned Rust contracts and a complete, deliberately small
//! AlphaZero-style inference/search/self-play vertical slice.
//!
//! Chess legality remains owned by [`chess_core`]. Requests convert a
//! [`Board`](chess_core::Board) once, while MCTS and self-play traverse the
//! reversible [`SearchPosition`](chess_core::SearchPosition) without changing
//! model or data schemas.

pub mod encoding;
pub mod engine;
pub mod evaluator;
pub mod http;
pub mod manifest;
pub mod mcts;
pub mod parity;
pub mod policy;
pub mod record;
pub mod self_play;

pub use encoding::{EncodedPosition, EncodingContext, encode};
pub use engine::AlphaMiniEngine;
pub use evaluator::{Evaluation, Evaluator, UniformEvaluator};
pub use manifest::{
    FROZEN_GATE_BASELINE, FROZEN_GATE_BATCH_SIZE, FROZEN_GATE_BOOTSTRAP_SAMPLES,
    FROZEN_GATE_BOOTSTRAP_SEED, FROZEN_GATE_CPUCT_PPM, FROZEN_GATE_FPU_REDUCTION_PPM,
    FROZEN_GATE_MAX_PLIES, FROZEN_GATE_MINIMAX_V1_MOVE_DIGEST, FROZEN_GATE_OPENING_PAIRS,
    FROZEN_GATE_OPENING_SUITE_SHA256, FROZEN_GATE_REQUIRED_LOWER_SCORE, FROZEN_GATE_SIMULATIONS,
    FROZEN_GATE_TIME_MS, GateVerdictV1, ModelManifestV1, ValidatedModel,
};
pub use mcts::{Mcts, SearchConfig, SearchResult, SearchStats};
pub use parity::{
    INFERENCE_PARITY_INPUT_SHA256, INFERENCE_PARITY_SCHEMA, InferenceParityV1, run_inference_parity,
};
pub use record::{
    CollectionManifestV1, GameRecordV1, MAX_SELF_PLAY_PLIES_V1, PositionRecordV1, derive_game_seed,
};

/// FEN used whenever a complete game is reconstructed from SAN.
pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
