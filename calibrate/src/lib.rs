pub mod artifact;
pub mod attestation;
pub mod calibration;
pub mod collect;
pub mod evaluate;
pub mod identity;
pub mod pgn;
pub mod stockfish;

use serde::{Deserialize, Serialize};

pub const CHESSCOM_TIME_CONTROL: &str = "1800";
pub const CHESSCOM_TIME_CLASS: &str = "rapid";
pub const ANALYSIS_FORMAT_V1: u32 = 1;
pub const ANALYSIS_FORMAT_V2: u32 = 2;
pub const PLAYER_SHARD_SCHEMA_V1: &str = "fnv1a-ascii-casefold-v1";
pub const ANALYSIS_TARGET_V2: &str = "Chess.com rated standard 30+0 (TimeControl 1800)";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PositionSample {
    pub game_id: String,
    pub actor_username: String,
    pub actor_rating: u16,
    pub ply: u16,
    /// Legal UCI moves before the sampled move. Replaying this prefix preserves
    /// repetition history for both the candidate engine and Stockfish; FEN alone cannot.
    #[serde(default)]
    pub uci_prefix: Vec<String>,
    pub fen: String,
    pub human_move: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRow {
    pub game_id: String,
    pub actor_username: String,
    pub actor_rating: u16,
    pub ply: u16,
    /// Exact pre-move history for v2 artifacts; empty in legacy FEN-only rows.
    #[serde(default)]
    pub uci_prefix: Vec<String>,
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
#[serde(deny_unknown_fields)]
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
    /// Required experiment identity for format v2. It remains optional only so
    /// historical format-v1 artifacts continue to deserialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment: Option<AnalysisExperimentV2>,
    /// Required for format v2 and absent in legacy format v1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_index: Option<u64>,
    /// Present only when an early v2 artifact was sealed from independently
    /// captured run evidence after generation. Native v2 writers leave it absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation: Option<attestation::PostHocAttestationV2>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AnalysisArtifact {
    pub metadata: AnalysisMetadata,
    pub skipped_uninformative: usize,
    #[serde(default)]
    pub skipped_player_cap: usize,
    pub rows: Vec<AnalysisRow>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AnalysisExperimentV2 {
    /// Ordered file-content hashes. Ordering is part of position sampling.
    pub corpus_sha256: Vec<String>,
    pub exclude_corpus_sha256: Vec<String>,
    /// Domain-separated aggregate over both ordered digest lists, including
    /// their primary/exclusion roles and lengths.
    pub corpus_digest_sha256: String,
    /// Domain-separated digest of every effective analysis setting and engine
    /// identity. `shard_index` is deliberately excluded so all shards match.
    pub analysis_config_sha256: String,
    pub sampling: AnalysisSamplingV2,
    pub bot: AnalysisBotV2,
    pub reference: AnalysisReferenceV2,
    /// Hash of the exact `calibrate` executable, binding chess-core, arena, and
    /// Minimax implementations as linked into this run.
    pub calibration_binary_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AnalysisSamplingV2 {
    pub positions_per_side: usize,
    pub positions_per_player: usize,
    pub analyzed_positions_per_player: usize,
    /// `None` means no global candidate-position truncation.
    pub max_positions: Option<usize>,
    pub minimum_rating: u16,
    pub maximum_rating: u16,
    pub minimum_ply: u16,
    pub maximum_ply: u16,
    pub sample_seed: u64,
    pub shard_count: u64,
    pub player_shard_schema: String,
    pub minimum_best_expected_score_ppm: u32,
    pub maximum_best_expected_score_ppm: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AnalysisBotV2 {
    Random {
        seed: u64,
    },
    MinimaxFixed {
        depth: u8,
        baseline_move_digest: u64,
    },
    MinimaxTimed {
        move_time_ms: u64,
        maximum_depth: u8,
    },
    AlphaMini {
        model_sha256: String,
        manifest_sha256: String,
        simulations: u32,
        move_time_ms: u64,
        batch_size: usize,
        seed: u64,
        /// Exact effective MCTS constants in millionths, avoiding float JSON
        /// and equality ambiguity in the experiment identity.
        cpuct_ppm: u32,
        fpu_reduction_ppm: u32,
        root_dirichlet_alpha_ppm: Option<u32>,
        root_noise_fraction_ppm: u32,
        evaluator: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AnalysisReferenceV2 {
    pub engine_name: String,
    pub binary_sha256: String,
    pub nodes_per_search: u64,
    pub hash_mb: u32,
    pub threads: u16,
    pub show_wdl: bool,
}
