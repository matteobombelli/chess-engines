//! Token encoding, PGN ingest, and serving for a move-sequence GPT.
//!
//! Move tokens are exactly the AlphaMini `policy-v1` action indices, so a
//! sampled token is decoded back to a move through the legal move set that
//! [`chess_core`] owns. Ingest turns compressed Lichess dumps into flat `u16`
//! shards plus a manifest describing every filter and checksum behind them, and
//! the serving half runs the exported ONNX graph behind the same HTTP contract
//! the other engines answer on.

pub mod decode;
pub mod encoding;
pub mod evaluator;
pub mod http;
pub mod ingest;
pub mod manifest;
pub mod model_manifest;
pub mod parity;
pub mod pgn;
pub mod shard;

pub use decode::{DecodeError, choose_move, truncate_context};
pub use encoding::{
    BOS_TOKEN, GameEncoder, PAD_TOKEN, TOKENIZER_VERSION, VOCAB_SIZE, encode_movetext,
};
pub use evaluator::{EvaluatorError, Logits, TokenEvaluator, UniformEvaluator};
pub use http::{AppState, BotRequest, BotResponse, DecodeConfig, router};
pub use ingest::{IngestOptions, run};
pub use manifest::{SHARDS_MANIFEST_FILE, SHARDS_MANIFEST_VERSION, ShardsManifestV1};
pub use model_manifest::{
    MODEL_MANIFEST_FILE, MODEL_MANIFEST_VERSION, ModelManifestV1, ValidatedModel,
};
pub use parity::{PARITY_FIXTURE_SCHEMA, ParityFixtureV1, ParityReport, load_fixture, run_parity};
