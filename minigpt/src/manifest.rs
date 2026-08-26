use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::pgn::RejectReason;

pub const SHARDS_MANIFEST_VERSION: &str = "minigpt.shards.v1";
pub const SHARDS_MANIFEST_FILE: &str = "shards.json";

/// One shard's token stream and its game index. `token_count * 2` is the size
/// of the `.bin`; `(game_count + 2) * 8` is the size of the `.idx`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardFileV1 {
    pub tokens_path: String,
    pub index_path: String,
    pub tokens_sha256: String,
    pub index_sha256: String,
    pub token_count: u64,
    pub game_count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiltersV1 {
    pub min_elo: u32,
    pub min_plies: u32,
    pub max_plies: u32,
    pub token_target: u64,
    /// Parts per million of games routed to validation; see
    /// [`crate::ingest::is_validation_game`] for the exact predicate.
    pub val_fraction_ppm: u32,
    pub shard_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceV1 {
    pub path: String,
    /// SHA-256 of the compressed dump, over the whole file even when the token
    /// target stopped decoding early.
    pub sha256: String,
    pub compressed_bytes: u64,
    pub games_seen: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedV1 {
    pub non_standard_start: u64,
    pub event: u64,
    pub elo: u64,
    pub termination: u64,
    pub variation: u64,
    pub ply_bounds: u64,
    pub san_error: u64,
}

impl RejectedV1 {
    pub fn record(&mut self, reason: RejectReason) {
        *match reason {
            RejectReason::NonStandardStart => &mut self.non_standard_start,
            RejectReason::Event => &mut self.event,
            RejectReason::Elo => &mut self.elo,
            RejectReason::Termination => &mut self.termination,
            RejectReason::Variation => &mut self.variation,
            RejectReason::PlyBounds => &mut self.ply_bounds,
            RejectReason::SanError => &mut self.san_error,
        } += 1;
    }

    pub fn total(&self) -> u64 {
        self.non_standard_start
            + self.event
            + self.elo
            + self.termination
            + self.variation
            + self.ply_bounds
            + self.san_error
    }
}

/// Every game read is either accepted or counted under exactly one reject
/// reason, so `games_seen == games_accepted + rejected.total()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountsV1 {
    pub games_seen: u64,
    pub games_accepted: u64,
    pub games_train: u64,
    pub games_val: u64,
    pub tokens_train: u64,
    pub tokens_val: u64,
    pub rejected: RejectedV1,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardsManifestV1 {
    pub schema: String,
    pub tokenizer: String,
    pub vocab_size: u64,
    pub bos_token: u16,
    pub pad_token: u16,
    pub filters: FiltersV1,
    pub sources: Vec<SourceV1>,
    pub counts: CountsV1,
    pub train_shards: Vec<ShardFileV1>,
    pub val_shards: Vec<ShardFileV1>,
    /// Site tags of the first few games whose movetext failed to replay, kept
    /// so a run that starts rejecting everything is diagnosable after the fact.
    pub san_error_samples: Vec<String>,
    pub started_unix_seconds: u64,
    pub completed_unix_seconds: u64,
}

pub fn write_manifest_atomic(path: &Path, manifest: &ShardsManifestV1) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    artifact_io::publish_bytes_new(path, &bytes)
}

pub fn read_manifest(path: &Path) -> io::Result<ShardsManifestV1> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}
