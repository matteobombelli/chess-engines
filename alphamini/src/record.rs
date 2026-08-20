use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Component, Path, PathBuf};

use artifact_io::{publish_bytes_idempotent, publish_bytes_new};
use chess_core::{Board, CastlingRights, Color, Piece, PieceKind, SearchPosition, Square, Status};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::encoding::{ENCODER_VERSION, INPUT_PLANES, encode};
use crate::manifest::{ManifestError, sha256_bytes};
use crate::policy::{POLICY_SIZE, POLICY_VERSION, move_to_action};

pub const POSITION_RECORD_VERSION: &str = "position-record-v1";
pub const GAME_RECORD_VERSION: &str = "game-record-v1";
pub const SHARD_VERSION: &str = "self-play-shard-v1";
pub const COLLECTION_MANIFEST_VERSION: &str = "collection-manifest-v1";
pub const TENSOR_CACHE_MANIFEST_VERSION: &str = "tensor-cache-manifest-v1";
pub const MAX_SELF_PLAY_PLIES_V1: u16 = 512;

/// Frozen per-game RNG stream derivation. `collection_seed` remains the base
/// seed recorded in the collection and shard; each game owns a stable stream
/// independent of scheduling and shard boundaries.
pub fn derive_game_seed(collection_seed: u64, game_id: u64) -> u64 {
    let mut value = collection_seed ^ game_id;
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    White,
    Black,
}

impl From<Color> for Side {
    fn from(color: Color) -> Self {
        match color {
            Color::White => Self::White,
            Color::Black => Self::Black,
        }
    }
}

impl From<Side> for Color {
    fn from(side: Side) -> Self {
        match side {
            Side::White => Self::White,
            Side::Black => Self::Black,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionV1 {
    Queen,
    Rook,
    Bishop,
    Knight,
}

impl PromotionV1 {
    fn piece_kind(self) -> PieceKind {
        match self {
            Self::Queen => PieceKind::Queen,
            Self::Rook => PieceKind::Rook,
            Self::Bishop => PieceKind::Bishop,
            Self::Knight => PieceKind::Knight,
        }
    }
}

impl From<PieceKind> for PromotionV1 {
    fn from(kind: PieceKind) -> Self {
        match kind {
            PieceKind::Queen => Self::Queen,
            PieceKind::Rook => Self::Rook,
            PieceKind::Bishop => Self::Bishop,
            PieceKind::Knight => Self::Knight,
            _ => unreachable!("only promotion kinds are converted"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyVisitV1 {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<PromotionV1>,
    pub visits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameOutcomeV1 {
    WhiteWin,
    Draw,
    BlackWin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationV1 {
    Checkmate,
    Stalemate,
    InsufficientMaterial,
    ThreefoldRepetition,
    FiftyMoveRule,
    PlyLimit,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionRecordV1 {
    pub schema: String,
    pub game_id: u64,
    pub ply: u16,
    /// White P/N/B/R/Q/K followed by Black P/N/B/R/Q/K.
    pub piece_bitboards: [u64; 12],
    pub side_to_move: Side,
    /// KQkq in bits 0..4.
    pub castling_rights: u8,
    pub en_passant_square: Option<u8>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    pub prior_occurrences: u8,
    pub previous_move_uci: Option<String>,
    /// Move actually sampled/selected from this position and committed to the game.
    pub selected_move_uci: String,
    pub policy: Vec<PolicyVisitV1>,
    pub outcome: GameOutcomeV1,
    pub termination: TerminationV1,
}

impl PositionRecordV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn from_search_position(
        position: &SearchPosition,
        game_id: u64,
        ply: u16,
        previous_move_uci: Option<String>,
        selected_move_uci: String,
        policy: Vec<PolicyVisitV1>,
        outcome: GameOutcomeV1,
        termination: TerminationV1,
    ) -> Self {
        let mut bitboards = [0_u64; 12];
        for index in 0..64 {
            if let Some(piece) = position.piece_at(Square(index as u8)) {
                bitboards[piece_slot(piece)] |= 1_u64 << index;
            }
        }
        let castling_rights = position.castling_rights();
        let mut castling = 0;
        castling |= u8::from(castling_rights.white_kingside);
        castling |= u8::from(castling_rights.white_queenside) << 1;
        castling |= u8::from(castling_rights.black_kingside) << 2;
        castling |= u8::from(castling_rights.black_queenside) << 3;
        Self {
            schema: POSITION_RECORD_VERSION.to_string(),
            game_id,
            ply,
            piece_bitboards: bitboards,
            side_to_move: position.side_to_move().into(),
            castling_rights: castling,
            en_passant_square: position.en_passant_target().map(|square| square.0),
            halfmove_clock: position.halfmove_clock(),
            fullmove_number: position.fullmove_number(),
            prior_occurrences: position.prior_repetition_count().min(2) as u8,
            previous_move_uci,
            selected_move_uci,
            policy,
            outcome,
            termination,
        }
    }

    pub fn to_board(&self) -> Result<Board, RecordError> {
        self.validate()?;
        let mut board = Board::empty();
        for (slot, bitboard) in self.piece_bitboards.iter().copied().enumerate() {
            let mut remaining = bitboard;
            while remaining != 0 {
                let square = remaining.trailing_zeros() as u8;
                remaining &= remaining - 1;
                board.set_piece(Square(square), Some(piece_for_slot(slot)));
            }
        }
        board.side_to_move = self.side_to_move.into();
        board.castling = CastlingRights {
            white_kingside: self.castling_rights & 1 != 0,
            white_queenside: self.castling_rights & 2 != 0,
            black_kingside: self.castling_rights & 4 != 0,
            black_queenside: self.castling_rights & 8 != 0,
        };
        board.en_passant = self.en_passant_square.map(Square);
        board.halfmove_clock = self.halfmove_clock;
        board.fullmove_number = self.fullmove_number;
        Board::from_fen(&board.to_fen()).map_err(|error| {
            RecordError::Schema(format!("record does not form a valid FEN: {error}"))
        })
    }

    pub fn validate(&self) -> Result<(), RecordError> {
        if self.schema != POSITION_RECORD_VERSION {
            return Err(RecordError::Schema(format!(
                "position schema is {}, expected {POSITION_RECORD_VERSION}",
                self.schema
            )));
        }
        if self.castling_rights & !0x0f != 0 {
            return Err(RecordError::Schema(
                "castling mask uses unknown bits".to_string(),
            ));
        }
        if self.en_passant_square.is_some_and(|square| square >= 64) {
            return Err(RecordError::Schema(
                "en-passant square is out of range".to_string(),
            ));
        }
        if self.prior_occurrences > 2 {
            return Err(RecordError::Schema(
                "prior occurrences must be clipped to 0..=2".to_string(),
            ));
        }
        if self.selected_move_uci.is_empty() {
            return Err(RecordError::Schema(
                "selected move UCI must not be empty".to_string(),
            ));
        }
        validate_result(self.outcome, self.termination)?;
        let mut occupied = 0_u64;
        for pieces in self.piece_bitboards {
            if occupied & pieces != 0 {
                return Err(RecordError::Schema("piece bitboards overlap".to_string()));
            }
            occupied |= pieces;
        }
        if self.piece_bitboards[5].count_ones() != 1 || self.piece_bitboards[11].count_ones() != 1 {
            return Err(RecordError::Schema(
                "position must contain exactly one king per color".to_string(),
            ));
        }
        if self.policy.is_empty() || self.policy.iter().any(|target| target.visits == 0) {
            return Err(RecordError::Schema(
                "every sparse policy target must contain positive visits".to_string(),
            ));
        }
        let mut targets = std::collections::HashSet::new();
        for target in &self.policy {
            if target.from >= 64 || target.to >= 64 {
                return Err(RecordError::Schema(
                    "policy square is out of range".to_string(),
                ));
            }
            let promotion = target.promotion.map(|kind| kind as u8).unwrap_or(u8::MAX);
            if !targets.insert((target.from, target.to, promotion)) {
                return Err(RecordError::Schema(
                    "duplicate sparse policy target".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameRecordV1 {
    pub schema: String,
    pub game_id: u64,
    pub seed: u64,
    pub model_sha256: String,
    pub outcome: GameOutcomeV1,
    pub termination: TerminationV1,
    pub plies: u16,
    pub positions: Vec<PositionRecordV1>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfPlayShardV1 {
    pub schema: String,
    pub encoder_schema: String,
    pub action_schema: String,
    pub seed: u64,
    pub simulations: u32,
    pub max_plies: u16,
    pub games: Vec<GameRecordV1>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardDescriptorV1 {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub first_game_id: u64,
    pub last_game_id: u64,
    pub game_count: u64,
    pub position_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionManifestV1 {
    pub schema: String,
    pub encoder_schema: String,
    pub action_schema: String,
    pub run_id: String,
    pub cycle_id: u64,
    pub game_id_start: u64,
    pub model_sha256: String,
    pub config_sha256: String,
    pub seed: u64,
    pub simulations: u32,
    pub max_plies: u16,
    pub game_count: u64,
    pub position_count: u64,
    pub shards: Vec<ShardDescriptorV1>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDescriptorV1 {
    pub path: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorCacheManifestV1 {
    pub schema: String,
    pub encoder_schema: String,
    pub action_schema: String,
    pub source_collection_sha256: String,
    pub record_count: u64,
    pub policy_size: u64,
    pub input_shape: Vec<u64>,
    pub inputs: TensorDescriptorV1,
    pub policy_offsets: TensorDescriptorV1,
    pub policy_indices: TensorDescriptorV1,
    pub policy_values: TensorDescriptorV1,
    pub wdl: TensorDescriptorV1,
    pub game_ids: TensorDescriptorV1,
}

#[derive(Debug, Error)]
pub enum RecordError {
    #[error("record schema violation: {0}")]
    Schema(String),
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("serialization failed: {0}")]
    Serialization(String),
    #[error("checksum mismatch for {path}: expected {expected}, got {actual}")]
    Checksum {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error("policy mapping failed: {0}")]
    Policy(String),
}

pub fn write_shard_atomic(
    path: &Path,
    shard: &SelfPlayShardV1,
) -> Result<ShardDescriptorV1, RecordError> {
    validate_shard(shard)?;
    let bytes = rmp_serde::to_vec_named(shard)
        .map_err(|error| RecordError::Serialization(error.to_string()))?;
    let compressed =
        zstd::stream::encode_all(bytes.as_slice(), 3).map_err(|source| RecordError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    write_atomic_new(path, &compressed)?;
    let first = shard.games.first().expect("validated nonempty").game_id;
    let last = shard.games.last().expect("validated nonempty").game_id;
    Ok(ShardDescriptorV1 {
        path: path
            .file_name()
            .expect("shard has filename")
            .to_string_lossy()
            .into_owned(),
        bytes: compressed.len() as u64,
        sha256: sha256_bytes(&compressed),
        first_game_id: first,
        last_game_id: last,
        game_count: shard.games.len() as u64,
        position_count: shard
            .games
            .iter()
            .map(|game| game.positions.len() as u64)
            .sum(),
    })
}

pub fn write_collection_manifest_atomic(
    path: &Path,
    manifest: &CollectionManifestV1,
) -> Result<(), RecordError> {
    validate_collection(manifest)?;
    validate_collection_files(manifest, path.parent().unwrap_or_else(|| Path::new(".")))?;
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|error| RecordError::Serialization(error.to_string()))?;
    write_atomic_new(path, &bytes)
}

fn validate_collection_files(
    manifest: &CollectionManifestV1,
    root: &Path,
) -> Result<(), RecordError> {
    for descriptor in &manifest.shards {
        let _ = read_verified_shard(root, descriptor, manifest)?;
    }
    Ok(())
}

/// Load one immutable shard and enforce the exact same descriptor/collection
/// identity at sealing and materialization. Keeping this as one path prevents
/// a future validation fix from protecting only one phase.
fn read_verified_shard(
    root: &Path,
    descriptor: &ShardDescriptorV1,
    manifest: &CollectionManifestV1,
) -> Result<SelfPlayShardV1, RecordError> {
    let path = checked_relative_join(root, &descriptor.path)?;
    let bytes = fs::read(&path).map_err(|source| RecordError::Io {
        path: path.clone(),
        source,
    })?;
    if bytes.len() as u64 != descriptor.bytes {
        return Err(RecordError::Schema(format!(
            "shard {} has {} bytes, descriptor says {}",
            path.display(),
            bytes.len(),
            descriptor.bytes
        )));
    }
    let actual = sha256_bytes(&bytes);
    if actual != descriptor.sha256 {
        return Err(RecordError::Checksum {
            path,
            expected: descriptor.sha256.clone(),
            actual,
        });
    }
    let decoded = zstd::stream::decode_all(bytes.as_slice()).map_err(|source| RecordError::Io {
        path: path.clone(),
        source,
    })?;
    let shard: SelfPlayShardV1 = rmp_serde::from_slice(&decoded)
        .map_err(|error| RecordError::Serialization(error.to_string()))?;
    validate_shard(&shard)?;
    let first = shard.games.first().expect("validated nonempty").game_id;
    let last = shard.games.last().expect("validated nonempty").game_id;
    let position_count = shard
        .games
        .iter()
        .map(|game| game.positions.len() as u64)
        .sum::<u64>();
    if first != descriptor.first_game_id
        || last != descriptor.last_game_id
        || shard.games.len() as u64 != descriptor.game_count
        || position_count != descriptor.position_count
        || shard.seed != manifest.seed
        || shard.simulations != manifest.simulations
        || shard.max_plies != manifest.max_plies
        || shard
            .games
            .iter()
            .any(|game| game.model_sha256 != manifest.model_sha256)
    {
        return Err(RecordError::Schema(format!(
            "decoded shard {} metadata disagrees with collection",
            path.display()
        )));
    }
    Ok(shard)
}

pub fn read_collection_manifest(path: &Path) -> Result<CollectionManifestV1, RecordError> {
    let reader = BufReader::new(File::open(path).map_err(|source| RecordError::Io {
        path: path.to_path_buf(),
        source,
    })?);
    let manifest = serde_json::from_reader(reader)
        .map_err(|error| RecordError::Serialization(error.to_string()))?;
    validate_collection(&manifest)?;
    Ok(manifest)
}

pub fn materialize_collection(
    collection_path: &Path,
    output_dir: &Path,
    tensor_manifest_path: &Path,
) -> Result<TensorCacheManifestV1, RecordError> {
    let collection = read_collection_manifest(collection_path)?;
    let collection_bytes = fs::read(collection_path).map_err(|source| RecordError::Io {
        path: collection_path.to_path_buf(),
        source,
    })?;
    let collection_root = collection_path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir).map_err(|source| RecordError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let mut inputs = Vec::new();
    let mut offsets = vec![0_u64];
    let mut indices = Vec::<u16>::new();
    let mut policy_values = Vec::<f32>::new();
    let mut wdl = Vec::<f32>::new();
    let mut game_ids = Vec::<u64>::new();

    for descriptor in &collection.shards {
        let shard = read_verified_shard(collection_root, descriptor, &collection)?;

        for game in shard.games {
            for record in game.positions {
                record.validate()?;
                let board = record.to_board()?;
                let encoded = encode(
                    &board,
                    crate::encoding::EncodingContext {
                        prior_occurrences: record.prior_occurrences,
                    },
                );
                inputs.extend_from_slice(&encoded.values);

                let total = record
                    .policy
                    .iter()
                    .map(|target| target.visits as u64)
                    .sum::<u64>();
                let legal_moves = board.get_legal_moves();
                let mut legal_by_key = std::collections::HashMap::with_capacity(legal_moves.len());
                for mv in legal_moves {
                    let promotion = mv.promotion.map(promotion_key).unwrap_or(u8::MAX);
                    legal_by_key.insert(
                        (mv.start_square.0, mv.end_square.0, promotion),
                        move_to_action(mv, board.side_to_move)
                            .map_err(|error| RecordError::Policy(error.to_string()))?,
                    );
                }
                let mut row_actions = std::collections::HashSet::new();
                for target in record.policy {
                    if target.visits == 0 {
                        continue;
                    }
                    let promotion = target
                        .promotion
                        .map(|kind| promotion_key(kind.piece_kind()))
                        .unwrap_or(u8::MAX);
                    let action = *legal_by_key
                        .get(&(target.from, target.to, promotion))
                        .ok_or_else(|| {
                            RecordError::Policy("stored target is not legal".to_string())
                        })?;
                    if !row_actions.insert(action) {
                        return Err(RecordError::Policy(
                            "two sparse targets map to one policy action".to_string(),
                        ));
                    }
                    indices.push(action as u16);
                    policy_values.push(target.visits as f32 / total as f32);
                }
                offsets.push(indices.len() as u64);
                wdl.extend_from_slice(&wdl_for(record.outcome, record.side_to_move));
                game_ids.push(record.game_id);
            }
        }
    }

    let count = game_ids.len() as u64;
    let mut inputs_desc = write_tensor(
        output_dir,
        "inputs.f32.bin",
        "f32-le",
        &[count, INPUT_PLANES as u64, 8, 8],
        &f32_bytes(&inputs),
    )?;
    let mut offsets_desc = write_tensor(
        output_dir,
        "policy-offsets.u64.bin",
        "u64-le",
        &[count + 1],
        &u64_bytes(&offsets),
    )?;
    let mut indices_desc = write_tensor(
        output_dir,
        "policy-indices.u16.bin",
        "u16-le",
        &[indices.len() as u64],
        &u16_bytes(&indices),
    )?;
    let mut values_desc = write_tensor(
        output_dir,
        "policy-values.f32.bin",
        "f32-le",
        &[policy_values.len() as u64],
        &f32_bytes(&policy_values),
    )?;
    let mut wdl_desc = write_tensor(
        output_dir,
        "wdl.f32.bin",
        "f32-le",
        &[count, 3],
        &f32_bytes(&wdl),
    )?;
    let mut games_desc = write_tensor(
        output_dir,
        "game-ids.u64.bin",
        "u64-le",
        &[count],
        &u64_bytes(&game_ids),
    )?;

    let tensor_root = tensor_manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    for descriptor in [
        &mut inputs_desc,
        &mut offsets_desc,
        &mut indices_desc,
        &mut values_desc,
        &mut wdl_desc,
        &mut games_desc,
    ] {
        let actual = output_dir.join(&descriptor.path);
        descriptor.path = actual
            .strip_prefix(tensor_root)
            .map_err(|_| {
                RecordError::Schema(
                    "tensor files must be beneath the tensor manifest directory".to_string(),
                )
            })?
            .to_string_lossy()
            .into_owned();
    }

    let manifest = TensorCacheManifestV1 {
        schema: TENSOR_CACHE_MANIFEST_VERSION.to_string(),
        encoder_schema: ENCODER_VERSION.to_string(),
        action_schema: POLICY_VERSION.to_string(),
        source_collection_sha256: sha256_bytes(&collection_bytes),
        record_count: count,
        policy_size: POLICY_SIZE as u64,
        input_shape: vec![INPUT_PLANES as u64, 8, 8],
        inputs: inputs_desc,
        policy_offsets: offsets_desc,
        policy_indices: indices_desc,
        policy_values: values_desc,
        wdl: wdl_desc,
        game_ids: games_desc,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| RecordError::Serialization(error.to_string()))?;
    write_atomic_idempotent(tensor_manifest_path, &manifest_bytes)?;
    Ok(manifest)
}

fn validate_shard(shard: &SelfPlayShardV1) -> Result<(), RecordError> {
    if shard.schema != SHARD_VERSION
        || shard.encoder_schema != ENCODER_VERSION
        || shard.action_schema != POLICY_VERSION
        || shard.simulations == 0
        || shard.max_plies == 0
        || shard.max_plies > MAX_SELF_PLAY_PLIES_V1
        || shard.games.is_empty()
    {
        return Err(RecordError::Schema(
            "invalid or empty shard header".to_string(),
        ));
    }
    let mut previous = None;
    for game in &shard.games {
        if game.schema != GAME_RECORD_VERSION
            || !valid_hash(&game.model_sha256)
            || game.positions.is_empty()
            || game.positions.len() > usize::from(shard.max_plies)
            || game.plies as usize != game.positions.len()
        {
            return Err(RecordError::Schema("invalid game record".to_string()));
        }
        validate_result(game.outcome, game.termination)?;
        if previous.is_some_and(|id| game.game_id <= id) {
            return Err(RecordError::Schema(
                "game IDs must be strictly increasing".to_string(),
            ));
        }
        let expected_seed = derive_game_seed(shard.seed, game.game_id);
        if game.seed != expected_seed {
            return Err(RecordError::Schema(format!(
                "game {} seed {} disagrees with derived seed {expected_seed}",
                game.game_id, game.seed
            )));
        }
        previous = Some(game.game_id);
        for (ply, record) in game.positions.iter().enumerate() {
            record.validate()?;
            if record.game_id != game.game_id
                || record.ply as usize != ply
                || record.outcome != game.outcome
                || record.termination != game.termination
            {
                return Err(RecordError::Schema(
                    "position/game metadata mismatch".to_string(),
                ));
            }
            let visits = record.policy.iter().try_fold(0_u64, |total, target| {
                total.checked_add(u64::from(target.visits))
            });
            if visits != Some(u64::from(shard.simulations)) {
                return Err(RecordError::Schema(format!(
                    "game {} ply {ply} policy visits do not equal the frozen {}-simulation budget",
                    game.game_id, shard.simulations
                )));
            }
        }
        validate_game_trajectory(game, shard.max_plies)?;
    }
    Ok(())
}

fn validate_game_trajectory(game: &GameRecordV1, max_plies: u16) -> Result<(), RecordError> {
    let mut replay = Board::from_fen(crate::START_FEN)
        .expect("the frozen AlphaMini starting FEN must remain valid");
    let mut previous_selected: Option<String> = None;

    for (ply, record) in game.positions.iter().enumerate() {
        let recorded_fen = record.to_board()?.to_fen();
        let replay_fen = replay.to_fen();
        if recorded_fen != replay_fen {
            return Err(RecordError::Schema(format!(
                "game {} ply {ply} state mismatch: recorded {recorded_fen}, replayed {replay_fen}",
                game.game_id
            )));
        }
        let expected_repetitions = replay.prior_repetition_count().min(2) as u8;
        if record.prior_occurrences != expected_repetitions {
            return Err(RecordError::Schema(format!(
                "game {} ply {ply} repetition mismatch: recorded {}, replayed {expected_repetitions}",
                game.game_id, record.prior_occurrences
            )));
        }
        if record.previous_move_uci != previous_selected {
            return Err(RecordError::Schema(format!(
                "game {} ply {ply} previous-move linkage is broken",
                game.game_id
            )));
        }

        let legal_moves = replay.get_legal_moves();
        let selected = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.to_uci() == record.selected_move_uci)
            .ok_or_else(|| {
                RecordError::Schema(format!(
                    "game {} ply {ply} selected move {} is illegal",
                    game.game_id, record.selected_move_uci
                ))
            })?;

        let mut selected_has_visits = false;
        for target in &record.policy {
            let target_promotion = target.promotion.map(PromotionV1::piece_kind);
            let target_move = legal_moves.iter().copied().find(|mv| {
                mv.start_square.0 == target.from
                    && mv.end_square.0 == target.to
                    && mv.promotion == target_promotion
            });
            let Some(target_move) = target_move else {
                return Err(RecordError::Schema(format!(
                    "game {} ply {ply} policy contains an illegal move",
                    game.game_id
                )));
            };
            if target_move == selected && target.visits > 0 {
                selected_has_visits = true;
            }
        }
        if !selected_has_visits {
            return Err(RecordError::Schema(format!(
                "game {} ply {ply} selected move is absent from positive-visit policy",
                game.game_id
            )));
        }

        previous_selected = Some(record.selected_move_uci.clone());
        replay.make_search_move(selected);
        if ply + 1 < game.positions.len() && replay.status() != Status::Ongoing {
            return Err(RecordError::Schema(format!(
                "game {} continues after a terminal result at ply {ply}",
                game.game_id
            )));
        }
    }

    let final_status = replay.status();
    if final_status == Status::Ongoing && game.plies != max_plies {
        return Err(RecordError::Schema(format!(
            "game {} claims a ply-limit draw after {} plies, frozen cap is {max_plies}",
            game.game_id, game.plies
        )));
    }
    let expected_termination = match final_status {
        Status::Ongoing => TerminationV1::PlyLimit,
        Status::Checkmate => TerminationV1::Checkmate,
        Status::Stalemate => TerminationV1::Stalemate,
        Status::InsufficientMaterial => TerminationV1::InsufficientMaterial,
        Status::ThreefoldRepetition => TerminationV1::ThreefoldRepetition,
        Status::FiftyMoveRule => TerminationV1::FiftyMoveRule,
    };
    if game.termination != expected_termination {
        return Err(RecordError::Schema(format!(
            "game {} final status {final_status:?} disagrees with termination {:?}",
            game.game_id, game.termination
        )));
    }
    let expected_outcome = if final_status == Status::Checkmate {
        match replay.side_to_move {
            Color::White => GameOutcomeV1::BlackWin,
            Color::Black => GameOutcomeV1::WhiteWin,
        }
    } else {
        GameOutcomeV1::Draw
    };
    if game.outcome != expected_outcome {
        return Err(RecordError::Schema(format!(
            "game {} final position implies {expected_outcome:?}, recorded {:?}",
            game.game_id, game.outcome
        )));
    }
    Ok(())
}

fn validate_result(outcome: GameOutcomeV1, termination: TerminationV1) -> Result<(), RecordError> {
    let valid = match termination {
        TerminationV1::Checkmate => outcome != GameOutcomeV1::Draw,
        TerminationV1::Stalemate
        | TerminationV1::InsufficientMaterial
        | TerminationV1::ThreefoldRepetition
        | TerminationV1::FiftyMoveRule
        | TerminationV1::PlyLimit => outcome == GameOutcomeV1::Draw,
    };
    if valid {
        Ok(())
    } else {
        Err(RecordError::Schema(
            "game outcome is inconsistent with termination".to_string(),
        ))
    }
}

fn validate_collection(manifest: &CollectionManifestV1) -> Result<(), RecordError> {
    if manifest.schema != COLLECTION_MANIFEST_VERSION
        || manifest.encoder_schema != ENCODER_VERSION
        || manifest.action_schema != POLICY_VERSION
        || manifest.simulations == 0
        || manifest.max_plies == 0
        || manifest.max_plies > MAX_SELF_PLAY_PLIES_V1
    {
        return Err(RecordError::Schema(
            "invalid collection schema header".to_string(),
        ));
    }
    if manifest.shards.is_empty() {
        return Err(RecordError::Schema(
            "collection must contain shards".to_string(),
        ));
    }
    if !valid_hash(&manifest.model_sha256) || !valid_hash(&manifest.config_sha256) {
        return Err(RecordError::Schema(
            "collection hashes must be hexadecimal SHA-256".to_string(),
        ));
    }
    let mut paths = std::collections::HashSet::new();
    let mut expected_first = manifest.game_id_start;
    for shard in &manifest.shards {
        checked_relative_join(Path::new("."), &shard.path)?;
        if !paths.insert(&shard.path) {
            return Err(RecordError::Schema(
                "collection contains duplicate shard paths".to_string(),
            ));
        }
        if !valid_hash(&shard.sha256)
            || shard.bytes == 0
            || shard.game_count == 0
            || shard.first_game_id != expected_first
            || shard.last_game_id < shard.first_game_id
            || shard.game_count != shard.last_game_id - shard.first_game_id + 1
        {
            return Err(RecordError::Schema(
                "invalid, gapped, or overlapping shard descriptor".to_string(),
            ));
        }
        expected_first = shard
            .last_game_id
            .checked_add(1)
            .ok_or_else(|| RecordError::Schema("game ID range overflows u64".to_string()))?;
    }
    let games = manifest
        .shards
        .iter()
        .map(|shard| shard.game_count)
        .sum::<u64>();
    let positions = manifest
        .shards
        .iter()
        .map(|shard| shard.position_count)
        .sum::<u64>();
    if games != manifest.game_count || positions != manifest.position_count {
        return Err(RecordError::Schema(
            "collection counts do not match shards".to_string(),
        ));
    }
    Ok(())
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn write_tensor(
    output_dir: &Path,
    name: &str,
    dtype: &str,
    shape: &[u64],
    bytes: &[u8],
) -> Result<TensorDescriptorV1, RecordError> {
    let path = output_dir.join(name);
    write_atomic_idempotent(&path, bytes)?;
    Ok(TensorDescriptorV1 {
        path: name.to_string(),
        dtype: dtype.to_string(),
        shape: shape.to_vec(),
        bytes: bytes.len() as u64,
        sha256: sha256_bytes(bytes),
    })
}

fn write_atomic_new(path: &Path, bytes: &[u8]) -> Result<(), RecordError> {
    publish_bytes_new(path, bytes).map_err(|source| RecordError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write_atomic_idempotent(path: &Path, bytes: &[u8]) -> Result<(), RecordError> {
    publish_bytes_idempotent(path, bytes).map_err(|source| RecordError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn checked_relative_join(root: &Path, relative: &str) -> Result<PathBuf, RecordError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path.components().any(|part| {
            matches!(
                part,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(RecordError::Schema(format!(
            "unsafe shard path {relative:?}"
        )));
    }
    Ok(root.join(path))
}

fn piece_slot(piece: Piece) -> usize {
    let offset = if piece.color == Color::White { 0 } else { 6 };
    offset
        + match piece.kind {
            PieceKind::Pawn => 0,
            PieceKind::Knight => 1,
            PieceKind::Bishop => 2,
            PieceKind::Rook => 3,
            PieceKind::Queen => 4,
            PieceKind::King => 5,
        }
}

fn piece_for_slot(slot: usize) -> Piece {
    Piece {
        color: if slot < 6 { Color::White } else { Color::Black },
        kind: match slot % 6 {
            0 => PieceKind::Pawn,
            1 => PieceKind::Knight,
            2 => PieceKind::Bishop,
            3 => PieceKind::Rook,
            4 => PieceKind::Queen,
            5 => PieceKind::King,
            _ => unreachable!(),
        },
    }
}

fn promotion_key(kind: PieceKind) -> u8 {
    match kind {
        PieceKind::Knight => 0,
        PieceKind::Bishop => 1,
        PieceKind::Rook => 2,
        PieceKind::Queen => 3,
        _ => u8::MAX,
    }
}

fn wdl_for(outcome: GameOutcomeV1, side: Side) -> [f32; 3] {
    match (outcome, side) {
        (GameOutcomeV1::Draw, _) => [0.0, 1.0, 0.0],
        (GameOutcomeV1::WhiteWin, Side::White) | (GameOutcomeV1::BlackWin, Side::Black) => {
            [1.0, 0.0, 0.0]
        }
        _ => [0.0, 0.0, 1.0],
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u64_bytes(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> PositionRecordV1 {
        let board = Board::from_fen(crate::START_FEN).unwrap();
        let position = SearchPosition::from_board(&board);
        PositionRecordV1::from_search_position(
            &position,
            10,
            0,
            None,
            "e2e4".to_string(),
            vec![PolicyVisitV1 {
                from: 12,
                to: 28,
                promotion: None,
                visits: 4,
            }],
            GameOutcomeV1::Draw,
            TerminationV1::PlyLimit,
        )
    }

    #[test]
    fn raw_position_round_trips_without_materialized_planes() {
        let record = record();
        record.validate().unwrap();
        assert_eq!(record.to_board().unwrap().to_fen(), crate::START_FEN);
    }

    #[test]
    fn rejects_unclipped_repetition_and_inconsistent_results() {
        let mut invalid_repetition = record();
        invalid_repetition.prior_occurrences = 3;
        assert!(invalid_repetition.validate().is_err());

        let mut decisive_ply_limit = record();
        decisive_ply_limit.outcome = GameOutcomeV1::WhiteWin;
        assert!(decisive_ply_limit.validate().is_err());

        let mut drawn_checkmate = record();
        drawn_checkmate.termination = TerminationV1::Checkmate;
        assert!(drawn_checkmate.validate().is_err());
    }

    #[test]
    fn shards_are_immutable_and_checksum_addressed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.msgpack.zst");
        let position = record();
        let game = GameRecordV1 {
            schema: GAME_RECORD_VERSION.into(),
            game_id: 10,
            seed: derive_game_seed(1, 10),
            model_sha256: "0".repeat(64),
            outcome: GameOutcomeV1::Draw,
            termination: TerminationV1::PlyLimit,
            plies: 1,
            positions: vec![position],
        };
        let shard = SelfPlayShardV1 {
            schema: SHARD_VERSION.into(),
            encoder_schema: ENCODER_VERSION.into(),
            action_schema: POLICY_VERSION.into(),
            seed: 1,
            simulations: 4,
            max_plies: 1,
            games: vec![game],
        };
        let descriptor = write_shard_atomic(&path, &shard).unwrap();
        assert_eq!(
            descriptor.sha256,
            crate::manifest::sha256_file(&path).unwrap()
        );
        assert!(write_shard_atomic(&path, &shard).is_err());

        let collection_path = dir.path().join("collection.json");
        let collection = CollectionManifestV1 {
            schema: COLLECTION_MANIFEST_VERSION.into(),
            encoder_schema: ENCODER_VERSION.into(),
            action_schema: POLICY_VERSION.into(),
            run_id: "retry-test".into(),
            cycle_id: 0,
            game_id_start: 10,
            model_sha256: "0".repeat(64),
            config_sha256: "1".repeat(64),
            seed: 1,
            simulations: 4,
            max_plies: 1,
            game_count: 1,
            position_count: 1,
            shards: vec![descriptor],
        };
        write_collection_manifest_atomic(&collection_path, &collection).unwrap();

        let mut wrong_model = collection.clone();
        wrong_model.model_sha256 = "2".repeat(64);
        assert!(
            write_collection_manifest_atomic(&dir.path().join("bad-model.json"), &wrong_model)
                .is_err()
        );
        let mut wrong_range = collection.clone();
        wrong_range.shards[0].first_game_id = 11;
        assert!(
            write_collection_manifest_atomic(&dir.path().join("bad-range.json"), &wrong_range)
                .is_err()
        );
        let mut wrong_seed = collection.clone();
        wrong_seed.seed ^= 1;
        assert!(
            write_collection_manifest_atomic(&dir.path().join("bad-seed.json"), &wrong_seed)
                .is_err()
        );
        let mut wrong_simulations = collection.clone();
        wrong_simulations.simulations += 1;
        assert!(
            write_collection_manifest_atomic(
                &dir.path().join("bad-simulations.json"),
                &wrong_simulations,
            )
            .is_err()
        );
        let mut wrong_cap = collection.clone();
        wrong_cap.max_plies += 1;
        assert!(
            write_collection_manifest_atomic(&dir.path().join("bad-cap.json"), &wrong_cap).is_err()
        );
        let tensor_dir = dir.path().join("tensors");
        let tensor_manifest = dir.path().join("tensors.json");
        let first =
            materialize_collection(&collection_path, &tensor_dir, &tensor_manifest).unwrap();
        let second =
            materialize_collection(&collection_path, &tensor_dir, &tensor_manifest).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn atomic_publish_never_clobbers_a_racing_winner() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("winner.bin");
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .map(|value| {
                let output = output.clone();
                let barrier = barrier.clone();
                let value = value.to_vec();
                thread::spawn(move || {
                    barrier.wait();
                    (value.clone(), write_atomic_new(&output, &value).is_ok())
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|(_, won)| *won).count(), 1);
        let winner = results.iter().find(|(_, won)| *won).unwrap();
        assert_eq!(fs::read(output).unwrap(), winner.0);
    }

    #[test]
    fn trajectory_replay_rejects_move_link_state_and_terminal_tampering() {
        use crate::evaluator::UniformEvaluator;
        use crate::self_play::{SelfPlayConfig, play_game};

        let mut evaluator = UniformEvaluator;
        let collection_seed = 9;
        let game = play_game(
            20,
            derive_game_seed(collection_seed, 20),
            &"0".repeat(64),
            SelfPlayConfig {
                simulations: 2,
                max_plies: 2,
                ..SelfPlayConfig::default()
            },
            &mut evaluator,
        )
        .unwrap();
        let shard = SelfPlayShardV1 {
            schema: SHARD_VERSION.into(),
            encoder_schema: ENCODER_VERSION.into(),
            action_schema: POLICY_VERSION.into(),
            seed: collection_seed,
            simulations: 2,
            max_plies: 2,
            games: vec![game],
        };
        validate_shard(&shard).unwrap();

        let mut wrong_seed = shard.clone();
        wrong_seed.games[0].seed ^= 1;
        assert!(validate_shard(&wrong_seed).is_err());

        let mut incomplete_policy = shard.clone();
        incomplete_policy.games[0].positions[0].policy[0].visits += 1;
        assert!(validate_shard(&incomplete_policy).is_err());

        let mut truncated = shard.clone();
        truncated.games[0].positions.truncate(1);
        truncated.games[0].plies = 1;
        assert!(validate_shard(&truncated).is_err());

        let mut illegal_selected = shard.clone();
        illegal_selected.games[0].positions[0].selected_move_uci = "e2e5".into();
        assert!(validate_shard(&illegal_selected).is_err());

        let mut broken_link = shard.clone();
        broken_link.games[0].positions[1].previous_move_uci = Some("a2a3".into());
        assert!(validate_shard(&broken_link).is_err());

        let mut wrong_state = shard.clone();
        wrong_state.games[0].positions[1].halfmove_clock += 1;
        assert!(validate_shard(&wrong_state).is_err());

        let mut false_terminal = shard;
        false_terminal.games[0].termination = TerminationV1::Stalemate;
        for position in &mut false_terminal.games[0].positions {
            position.termination = TerminationV1::Stalemate;
        }
        assert!(validate_shard(&false_terminal).is_err());
    }

    #[test]
    fn game_seed_derivation_is_frozen_splitmix64() {
        assert_eq!(derive_game_seed(0, 0), 0xe220_a839_7b1d_cdaf);
        assert_eq!(derive_game_seed(0x1234, 0x1234), 0xe220_a839_7b1d_cdaf);
    }
}
