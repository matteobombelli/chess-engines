pub mod calibration;
pub mod collect;
pub mod evaluate;
pub mod pgn;
pub mod stockfish;

use serde::{Deserialize, Serialize};

pub const CHESSCOM_TIME_CONTROL: &str = "1800";
pub const CHESSCOM_TIME_CLASS: &str = "rapid";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct PositionSample {
    pub game_id: String,
    pub actor_username: String,
    pub actor_rating: u16,
    pub ply: u16,
    pub fen: String,
    pub human_move: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct AnalysisRow {
    pub game_id: String,
    pub actor_username: String,
    pub actor_rating: u16,
    pub ply: u16,
    pub fen: String,
    pub human_move: String,
    pub bot_move: String,
    pub reference_move: String,
    pub best_expected_score: f64,
    pub human_expected_score: f64,
    pub bot_expected_score: f64,
    pub human_loss: f64,
    pub bot_loss: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct AnalysisMetadata {
    pub format_version: u32,
    pub target: String,
    pub bot: String,
    pub reference_engine: String,
    pub reference_nodes_per_search: u64,
    pub input_positions: usize,
    /// Games represented before uninformative positions are filtered.
    pub unique_games: usize,
    #[serde(default)]
    pub analyzed_unique_games: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct AnalysisArtifact {
    pub metadata: AnalysisMetadata,
    pub skipped_uninformative: usize,
    #[serde(default)]
    pub skipped_player_cap: usize,
    pub rows: Vec<AnalysisRow>,
}
