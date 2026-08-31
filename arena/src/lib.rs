use std::fmt;

pub mod http;
pub mod rating_log;
pub mod uci;

use chess_core::{Board, Color, Move, Status};
use minimax::{SearchLimits, find_best_move};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[cfg(feature = "alphamini")]
use alphamini::{Evaluator, Mcts, SearchConfig, ValidatedModel};
#[cfg(feature = "minigpt")]
use minigpt::{GameEncoder, TokenEvaluator, truncate_context};

/// A chess engine that can be compared in the arena.
///
/// Future bots only need a small adapter implementing this trait. The arena
/// owns all game-state and result handling and verifies every engine move.
pub trait Engine {
    fn name(&self) -> &str;
    fn choose_move(&mut self, board: &Board) -> Result<Move, String>;
}

impl<T: Engine + ?Sized> Engine for Box<T> {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
        (**self).choose_move(board)
    }
}

pub const MINIMAX_V1_MOVE_DIGEST_POSITIONS: usize = 16;
pub const MINIMAX_V1_MOVE_DIGEST: u64 = 16_258_623_573_026_552_286;

/// Stable regression digest for the frozen depth-1/2/3 baseline ladder.
///
/// The digest is deliberately not a rating. It detects changes in search,
/// evaluation, move generation, or tie-breaking that require a new baseline
/// identity and a fresh calibration.
pub fn minimax_v1_move_digest() -> Result<u64, String> {
    let suite: OpeningSuite =
        serde_json::from_str(include_str!("../openings/alphamini-v1.json"))
            .map_err(|error| format!("could not parse committed opening suite: {error}"))?;
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for depth in 1..=3 {
        let limits = SearchLimits::fixed_depth(depth)?;
        for entry in suite.openings.iter().take(MINIMAX_V1_MOVE_DIGEST_POSITIONS) {
            let board = Board::import_san(&entry.san)?;
            let result = find_best_move(&board, limits).map_err(|error| error.to_string())?;
            for byte in depth
                .to_le_bytes()
                .into_iter()
                .chain(result.best_move.to_uci().bytes())
                .chain(result.score.to_le_bytes())
            {
                digest ^= u64::from(byte);
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    Ok(digest)
}

pub struct RandomEngine {
    rng: StdRng,
}

/// Stateless random-strength baseline for resumable paired evaluations. Its
/// choice is keyed by the complete position and seed, so skipping a durable
/// opening prefix cannot shift a hidden RNG stream.
pub struct PositionRandomEngine {
    seed: u64,
    name: String,
}

impl PositionRandomEngine {
    pub fn seeded(seed: u64) -> Self {
        Self {
            seed,
            name: format!("PositionRandomV1[seed={seed}]"),
        }
    }
}

impl Engine for PositionRandomEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
        let moves = board.get_legal_moves();
        if moves.is_empty() {
            return Err("no legal move in an ongoing position".to_string());
        }
        let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ self.seed;
        for byte in board.to_fen().bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let index = splitmix64(hash) as usize % moves.len();
        Ok(moves[index])
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl RandomEngine {
    pub fn seeded(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Engine for RandomEngine {
    fn name(&self) -> &str {
        "Random"
    }

    fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
        random::choose_move(board, &mut self.rng)
            .ok_or_else(|| "no legal move in an ongoing position".to_string())
    }
}

pub struct MinimaxEngine {
    limits: SearchLimits,
    name: String,
}

impl MinimaxEngine {
    pub fn new(limits: SearchLimits) -> Result<Self, String> {
        limits.validate()?;
        let name = if limits.move_time.is_none() && limits.max_nodes.is_none() {
            format!("MinimaxDepth{}V1", limits.max_depth)
        } else {
            "MinimaxTimedV1".to_string()
        };
        Ok(Self { limits, name })
    }
}

impl Engine for MinimaxEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
        find_best_move(board, self.limits)
            .map(|result| result.best_move)
            .map_err(|error| error.to_string())
    }
}

/// Adapter used by the paired arena and move-quality calibration. The model is
/// loaded and checksum/schema validated once; every move is still checked by
/// the arena against chess-core's legal move list.
#[cfg(feature = "alphamini")]
pub struct AlphaMiniEngine {
    evaluator: Box<dyn Evaluator>,
    search: Mcts,
    rng: StdRng,
    name: String,
    metrics: AlphaMiniMetrics,
}

/// Search counters recorded alongside an AlphaMini pair result. The type is
/// always available so a durable pair log can be read without the feature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlphaMiniMetrics {
    pub moves: u64,
    pub completed_simulations: u64,
    pub neural_evaluations: u64,
    pub inference_batches: u64,
    pub largest_batch: usize,
    pub elapsed_micros: u64,
    pub deadlines_reached: u64,
}

#[cfg(feature = "alphamini")]
impl AlphaMiniEngine {
    pub fn load(
        model_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        search: SearchConfig,
        seed: u64,
    ) -> Result<Self, String> {
        let validated =
            ValidatedModel::load(model_path, manifest_path).map_err(|error| error.to_string())?;
        let identity = validated.manifest.model_sha256[..12].to_string();
        let evaluator = alphamini::evaluator::OnnxEvaluator::load(&validated)
            .map_err(|error| error.to_string())?;
        Self::with_evaluator(Box::new(evaluator), search, seed, identity)
    }

    pub fn with_evaluator(
        evaluator: Box<dyn Evaluator>,
        search: SearchConfig,
        seed: u64,
        identity: impl Into<String>,
    ) -> Result<Self, String> {
        let search_engine = Mcts::new(search).map_err(|error| error.to_string())?;
        let identity = identity.into();
        if identity.trim().is_empty() {
            return Err("AlphaMini identity must not be empty".to_string());
        }
        Ok(Self {
            evaluator,
            search: search_engine,
            rng: StdRng::seed_from_u64(seed),
            name: format!("AlphaMiniV1[{identity}]"),
            metrics: AlphaMiniMetrics::default(),
        })
    }

    pub fn metrics(&self) -> AlphaMiniMetrics {
        self.metrics
    }

    /// Return this evaluation segment's counters and reset them, allowing each
    /// durable opening-pair record to carry exact (not cumulative) metrics.
    pub fn take_metrics(&mut self) -> AlphaMiniMetrics {
        std::mem::take(&mut self.metrics)
    }
}

#[cfg(feature = "alphamini")]
impl Engine for AlphaMiniEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
        let result = self
            .search
            .search(board, self.evaluator.as_mut(), &mut self.rng)
            .map_err(|error| error.to_string())?;
        self.metrics.moves += 1;
        self.metrics.completed_simulations += u64::from(result.stats.completed_simulations);
        self.metrics.neural_evaluations += u64::from(result.stats.neural_evaluations);
        self.metrics.inference_batches += u64::from(result.stats.inference_batches);
        self.metrics.largest_batch = self.metrics.largest_batch.max(result.stats.largest_batch);
        self.metrics.elapsed_micros += result.stats.elapsed_micros;
        self.metrics.deadlines_reached += u64::from(result.stats.deadline_reached);
        Ok(result.best_move)
    }
}

/// Adapter for the stateless move-sequence GPT. It keeps no state between
/// turns: the game's SAN history is re-encoded every move, so a board handed to
/// it out of order is answered from that board's own history.
#[cfg(feature = "minigpt")]
pub struct MiniGptEngine {
    evaluator: Box<dyn TokenEvaluator>,
    context: usize,
    temperature: f32,
    rng: StdRng,
    name: String,
}

#[cfg(feature = "minigpt")]
impl MiniGptEngine {
    /// `temperature` overrides the manifest's published sampling temperature.
    pub fn load(
        model_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        temperature: Option<f32>,
        seed: u64,
    ) -> Result<Self, String> {
        let validated = minigpt::ValidatedModel::load(model_path, manifest_path)
            .map_err(|error| error.to_string())?;
        let identity = validated.manifest.model_sha256[..12].to_string();
        let context = validated.manifest.context;
        let temperature = temperature.unwrap_or(validated.manifest.decode_temperature);
        let evaluator = minigpt::evaluator::OnnxEvaluator::load(&validated)
            .map_err(|error| error.to_string())?;
        Self::with_evaluator(Box::new(evaluator), context, temperature, seed, identity)
    }

    pub fn with_evaluator(
        evaluator: Box<dyn TokenEvaluator>,
        context: usize,
        temperature: f32,
        seed: u64,
        identity: impl Into<String>,
    ) -> Result<Self, String> {
        // BOS plus one move must fit, or nothing can ever be decoded.
        if context < 2 {
            return Err(format!(
                "MiniGPT context must hold BOS plus a move, got {context}"
            ));
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(format!(
                "MiniGPT temperature must be finite and non-negative, got {temperature}"
            ));
        }
        let identity = identity.into();
        if identity.trim().is_empty() {
            return Err("MiniGPT identity must not be empty".to_string());
        }
        Ok(Self {
            evaluator,
            context,
            temperature,
            rng: StdRng::seed_from_u64(seed),
            name: format!("MiniGptV1[{identity}]"),
        })
    }

    pub fn context(&self) -> usize {
        self.context
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

#[cfg(feature = "minigpt")]
impl Engine for MiniGptEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
        let mut encoder = GameEncoder::new();
        for san in &board.san_history {
            encoder.push_san(san).map_err(|error| error.to_string())?;
        }
        let tokens = truncate_context(encoder.tokens(), self.context);
        let logits = self
            .evaluator
            .logits(&tokens)
            .map_err(|error| error.to_string())?;
        minigpt::choose_move(logits.last_row(), board, self.temperature, &mut self.rng)
            .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchConfig {
    pub games: u32,
    /// Safety cap for engines that do not reach a core draw condition.
    pub max_plies: u32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            games: 100,
            max_plies: 1_000,
        }
    }
}

impl MatchConfig {
    fn validate(self) -> Result<(), ArenaError> {
        if self.games == 0 {
            return Err(ArenaError::InvalidConfig(
                "games must be greater than zero".to_string(),
            ));
        }
        if self.max_plies == 0 {
            return Err(ArenaError::InvalidConfig(
                "max plies must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    Checkmate,
    Stalemate,
    InsufficientMaterial,
    ThreefoldRepetition,
    FiftyMoveRule,
    PlyLimit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameResult {
    pub winner: Option<Color>,
    pub termination: Termination,
    pub plies: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Record {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

impl Record {
    pub fn games(self) -> u32 {
        self.wins + self.draws + self.losses
    }

    pub fn score(self) -> f64 {
        if self.games() == 0 {
            return 0.0;
        }
        (f64::from(self.wins) + 0.5 * f64::from(self.draws)) / f64::from(self.games())
    }

    fn add(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Win => self.wins += 1,
            Outcome::Draw => self.draws += 1,
            Outcome::Loss => self.losses += 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawCounts {
    pub stalemate: u32,
    pub insufficient_material: u32,
    pub threefold_repetition: u32,
    pub fifty_move_rule: u32,
    pub ply_limit: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchReport {
    /// All records are from engine A's perspective.
    pub engine_a: String,
    pub engine_b: String,
    pub overall: Record,
    pub as_white: Record,
    pub as_black: Record,
    pub draws: DrawCounts,
}

/// A deterministic, evaluation-only opening expressed as PGN movetext.
///
/// The same opening is played twice with engine colors reversed by
/// [`run_paired_match`]. Training code must not consume arena openings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Opening {
    pub id: String,
    pub san: String,
}

pub const OPENING_SUITE_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpeningSuiteEntry {
    pub id: String,
    pub san: String,
    pub fen: String,
    pub legal_moves: usize,
    pub depth_three_score_cp: i32,
}

impl OpeningSuiteEntry {
    pub fn opening(&self) -> Result<Opening, ArenaError> {
        let opening = Opening::new(self.id.clone(), self.san.clone())?;
        let actual_fen = Board::import_san(&self.san)
            .map_err(|error| ArenaError::InvalidOpening {
                id: self.id.clone(),
                error,
            })?
            .to_fen();
        if actual_fen != self.fen {
            return Err(ArenaError::InvalidOpening {
                id: self.id.clone(),
                error: format!(
                    "stored FEN does not match replayed SAN: expected {:?}, got {:?}",
                    self.fen, actual_fen
                ),
            });
        }
        Ok(opening)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpeningSuite {
    pub format_version: u32,
    pub name: String,
    pub seed: u64,
    pub plies: u16,
    pub minimum_legal_moves: usize,
    pub maximum_absolute_depth_three_score_cp: i32,
    pub baseline: String,
    pub openings: Vec<OpeningSuiteEntry>,
}

impl OpeningSuite {
    pub fn validate(&self) -> Result<Vec<Opening>, ArenaError> {
        if self.format_version != OPENING_SUITE_FORMAT_VERSION {
            return Err(ArenaError::InvalidConfig(format!(
                "unsupported opening suite format version {}; expected {}",
                self.format_version, OPENING_SUITE_FORMAT_VERSION
            )));
        }
        if self.openings.is_empty() {
            return Err(ArenaError::InvalidConfig(
                "opening suite must contain at least one opening".to_string(),
            ));
        }
        if self.maximum_absolute_depth_three_score_cp < 0 {
            return Err(ArenaError::InvalidConfig(
                "opening suite maximum score must not be negative".to_string(),
            ));
        }
        let mut ids = std::collections::HashSet::with_capacity(self.openings.len());
        self.openings
            .iter()
            .map(|entry| {
                if !ids.insert(entry.id.as_str()) {
                    return Err(ArenaError::InvalidOpening {
                        id: entry.id.clone(),
                        error: "duplicate opening id".to_string(),
                    });
                }
                if entry.legal_moves < self.minimum_legal_moves {
                    return Err(ArenaError::InvalidOpening {
                        id: entry.id.clone(),
                        error: "stored legal-move count violates suite filter".to_string(),
                    });
                }
                if entry.depth_three_score_cp.unsigned_abs()
                    > self.maximum_absolute_depth_three_score_cp as u32
                {
                    return Err(ArenaError::InvalidOpening {
                        id: entry.id.clone(),
                        error: "stored score violates suite balance filter".to_string(),
                    });
                }
                let opening = entry.opening()?;
                let actual_legal_moves = Board::import_san(&entry.san)
                    .map_err(|error| ArenaError::InvalidOpening {
                        id: entry.id.clone(),
                        error,
                    })?
                    .get_legal_moves()
                    .len();
                if actual_legal_moves != entry.legal_moves {
                    return Err(ArenaError::InvalidOpening {
                        id: entry.id.clone(),
                        error: format!(
                            "stored legal-move count {} does not match replayed count {}",
                            entry.legal_moves, actual_legal_moves
                        ),
                    });
                }
                Ok(opening)
            })
            .collect()
    }

    /// Re-run the frozen depth-3 filter as an expensive release/report check.
    pub fn validate_deep(&self) -> Result<Vec<Opening>, ArenaError> {
        let openings = self.validate()?;
        let limits = SearchLimits::fixed_depth(3).map_err(ArenaError::InvalidConfig)?;
        for entry in &self.openings {
            let board =
                Board::import_san(&entry.san).map_err(|error| ArenaError::InvalidOpening {
                    id: entry.id.clone(),
                    error,
                })?;
            let actual = find_best_move(&board, limits)
                .map_err(|error| ArenaError::InvalidOpening {
                    id: entry.id.clone(),
                    error: format!("depth-3 verification failed: {error}"),
                })?
                .score;
            if actual != entry.depth_three_score_cp {
                return Err(ArenaError::InvalidOpening {
                    id: entry.id.clone(),
                    error: format!(
                        "stored depth-3 score {} does not match current baseline score {}",
                        entry.depth_three_score_cp, actual
                    ),
                });
            }
        }
        Ok(openings)
    }
}

impl Opening {
    pub fn new(id: impl Into<String>, san: impl Into<String>) -> Result<Self, ArenaError> {
        let opening = Self {
            id: id.into(),
            san: san.into(),
        };
        if opening.id.trim().is_empty() {
            return Err(ArenaError::InvalidOpening {
                id: opening.id,
                error: "opening id must not be empty".to_string(),
            });
        }
        let board =
            Board::import_san(&opening.san).map_err(|error| ArenaError::InvalidOpening {
                id: opening.id.clone(),
                error,
            })?;
        if board.status() != Status::Ongoing {
            return Err(ArenaError::InvalidOpening {
                id: opening.id,
                error: "opening must end in an ongoing position".to_string(),
            });
        }
        Ok(opening)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpeningPairResult {
    pub opening_id: String,
    pub engine_a_as_white: GameResult,
    pub engine_a_as_black: GameResult,
    /// Mean score over the two color-reversed games, from engine A's perspective.
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairedMatchReport {
    pub match_report: MatchReport,
    pub pairs: Vec<OpeningPairResult>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootstrapConfig {
    pub samples: u32,
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            samples: 20_000,
            seed: 1,
        }
    }
}

impl MatchReport {
    pub fn score(&self) -> f64 {
        self.overall.score()
    }

    /// Engine A's Elo difference over engine B under the logistic Elo model.
    pub fn elo_difference(&self) -> f64 {
        elo_from_score(self.score())
    }

    /// Approximate 95% interval, derived from a Wilson interval on match score.
    /// Draws count as half a point.
    pub fn elo_95_interval(&self) -> (f64, f64) {
        let (low, high) = wilson_score_interval(
            f64::from(self.overall.wins) + 0.5 * f64::from(self.overall.draws),
            self.overall.games(),
        );
        (elo_from_score(low), elo_from_score(high))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ArenaError {
    InvalidConfig(String),
    InvalidOpening { id: String, error: String },
    Engine { name: String, error: String },
    IllegalMove { name: String, mv: Move },
}

impl fmt::Display for ArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => write!(formatter, "invalid match config: {error}"),
            Self::InvalidOpening { id, error } => {
                write!(formatter, "invalid opening {id:?}: {error}")
            }
            Self::Engine { name, error } => write!(formatter, "{name} failed: {error}"),
            Self::IllegalMove { name, mv } => {
                write!(formatter, "{name} returned an illegal move: {mv:?}")
            }
        }
    }
}

impl std::error::Error for ArenaError {}

#[derive(Clone, Copy)]
enum Outcome {
    Win,
    Draw,
    Loss,
}

/// Play a color-balanced match. Engine A is White in even-numbered games and
/// Black in odd-numbered games, so the color counts differ by at most one.
pub fn run_match<A, B>(
    engine_a: &mut A,
    engine_b: &mut B,
    config: MatchConfig,
) -> Result<MatchReport, ArenaError>
where
    A: Engine,
    B: Engine,
{
    run_match_with_progress(engine_a, engine_b, config, |_, _, _| {})
}

pub fn run_match_with_progress<A, B, F>(
    engine_a: &mut A,
    engine_b: &mut B,
    config: MatchConfig,
    mut progress: F,
) -> Result<MatchReport, ArenaError>
where
    A: Engine,
    B: Engine,
    F: FnMut(u32, &GameResult, &MatchReport),
{
    config.validate()?;

    let mut report = MatchReport {
        engine_a: engine_a.name().to_string(),
        engine_b: engine_b.name().to_string(),
        overall: Record::default(),
        as_white: Record::default(),
        as_black: Record::default(),
        draws: DrawCounts::default(),
    };

    for game_index in 0..config.games {
        let a_is_white = game_index % 2 == 0;
        let result = if a_is_white {
            play_game(engine_a, engine_b, config.max_plies)?
        } else {
            play_game(engine_b, engine_a, config.max_plies)?
        };
        let a_color = if a_is_white {
            Color::White
        } else {
            Color::Black
        };
        let outcome = match result.winner {
            Some(winner) if winner == a_color => Outcome::Win,
            Some(_) => Outcome::Loss,
            None => Outcome::Draw,
        };

        report.overall.add(outcome);
        if a_is_white {
            report.as_white.add(outcome);
        } else {
            report.as_black.add(outcome);
        }
        match result.termination {
            Termination::Stalemate => report.draws.stalemate += 1,
            Termination::InsufficientMaterial => report.draws.insufficient_material += 1,
            Termination::ThreefoldRepetition => report.draws.threefold_repetition += 1,
            Termination::FiftyMoveRule => report.draws.fifty_move_rule += 1,
            Termination::PlyLimit => report.draws.ply_limit += 1,
            Termination::Checkmate => {}
        }

        progress(game_index + 1, &result, &report);
    }

    Ok(report)
}

/// Play every opening twice, with engine A as White and then Black.
///
/// Openings are statistical clusters: callers should bootstrap pair scores,
/// rather than pretending the two games in a pair are independent.
pub fn run_paired_match<A, B>(
    engine_a: &mut A,
    engine_b: &mut B,
    openings: &[Opening],
    max_plies: u32,
) -> Result<PairedMatchReport, ArenaError>
where
    A: Engine,
    B: Engine,
{
    run_paired_match_with_progress(engine_a, engine_b, openings, max_plies, |_, _, _| {})
}

pub fn run_paired_match_with_progress<A, B, F>(
    engine_a: &mut A,
    engine_b: &mut B,
    openings: &[Opening],
    max_plies: u32,
    mut progress: F,
) -> Result<PairedMatchReport, ArenaError>
where
    A: Engine,
    B: Engine,
    F: FnMut(usize, &OpeningPairResult, &MatchReport),
{
    if openings.is_empty() {
        return Err(ArenaError::InvalidConfig(
            "at least one opening is required".to_string(),
        ));
    }
    if max_plies == 0 {
        return Err(ArenaError::InvalidConfig(
            "max plies must be greater than zero".to_string(),
        ));
    }

    let mut report = MatchReport {
        engine_a: engine_a.name().to_string(),
        engine_b: engine_b.name().to_string(),
        overall: Record::default(),
        as_white: Record::default(),
        as_black: Record::default(),
        draws: DrawCounts::default(),
    };
    let mut pairs = Vec::with_capacity(openings.len());
    let mut opening_ids = std::collections::HashSet::with_capacity(openings.len());

    for opening in openings {
        // Opening::new validates eagerly, but validate again at the boundary so
        // deserialized or directly-constructed values cannot bypass legality.
        if opening.id.trim().is_empty() || !opening_ids.insert(opening.id.as_str()) {
            return Err(ArenaError::InvalidOpening {
                id: opening.id.clone(),
                error: "opening ids must be non-empty and unique".to_string(),
            });
        }
        let board =
            Board::import_san(&opening.san).map_err(|error| ArenaError::InvalidOpening {
                id: opening.id.clone(),
                error,
            })?;
        if board.status() != Status::Ongoing {
            return Err(ArenaError::InvalidOpening {
                id: opening.id.clone(),
                error: "opening must end in an ongoing position".to_string(),
            });
        }

        let as_white = play_game_from_board(board.clone(), engine_a, engine_b, max_plies)?;
        let white_outcome = outcome_for_color(&as_white, Color::White);
        add_result(&mut report, white_outcome, true, &as_white);

        let as_black = play_game_from_board(board, engine_b, engine_a, max_plies)?;
        let black_outcome = outcome_for_color(&as_black, Color::Black);
        add_result(&mut report, black_outcome, false, &as_black);

        pairs.push(OpeningPairResult {
            opening_id: opening.id.clone(),
            engine_a_as_white: as_white,
            engine_a_as_black: as_black,
            score: (outcome_points(white_outcome) + outcome_points(black_outcome)) / 2.0,
        });
        progress(
            pairs.len(),
            pairs.last().expect("pair was just pushed"),
            &report,
        );
    }

    Ok(PairedMatchReport {
        match_report: report,
        pairs,
    })
}

/// Return the percentile 95% interval obtained by resampling opening pairs.
pub fn paired_score_bootstrap_95(
    report: &PairedMatchReport,
    config: BootstrapConfig,
) -> Result<(f64, f64), ArenaError> {
    if config.samples == 0 {
        return Err(ArenaError::InvalidConfig(
            "bootstrap samples must be greater than zero".to_string(),
        ));
    }
    if report.pairs.is_empty() {
        return Err(ArenaError::InvalidConfig(
            "cannot bootstrap an empty paired match".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut estimates = Vec::with_capacity(config.samples as usize);
    for _ in 0..config.samples {
        let sum: f64 = (0..report.pairs.len())
            .map(|_| {
                let index = rng.gen_range(0..report.pairs.len());
                report.pairs[index].score
            })
            .sum();
        estimates.push(sum / report.pairs.len() as f64);
    }
    estimates.sort_by(f64::total_cmp);
    let low = percentile_index(estimates.len(), 0.025);
    let high = percentile_index(estimates.len(), 0.975);
    Ok((estimates[low], estimates[high]))
}

/// Rebuild aggregate statistics from durably recorded opening-pair results.
/// This is used by the resumable CLI after validating its append-only log.
pub fn paired_report_from_results(
    engine_a: impl Into<String>,
    engine_b: impl Into<String>,
    pairs: Vec<OpeningPairResult>,
) -> Result<PairedMatchReport, ArenaError> {
    if pairs.is_empty() {
        return Err(ArenaError::InvalidConfig(
            "at least one completed opening pair is required".to_string(),
        ));
    }
    let mut report = MatchReport {
        engine_a: engine_a.into(),
        engine_b: engine_b.into(),
        overall: Record::default(),
        as_white: Record::default(),
        as_black: Record::default(),
        draws: DrawCounts::default(),
    };
    let mut ids = std::collections::HashSet::with_capacity(pairs.len());
    for pair in &pairs {
        if pair.opening_id.trim().is_empty() || !ids.insert(pair.opening_id.as_str()) {
            return Err(ArenaError::InvalidOpening {
                id: pair.opening_id.clone(),
                error: "recorded opening ids must be non-empty and unique".to_string(),
            });
        }
        let white = outcome_for_color(&pair.engine_a_as_white, Color::White);
        let black = outcome_for_color(&pair.engine_a_as_black, Color::Black);
        let score = (outcome_points(white) + outcome_points(black)) / 2.0;
        if (score - pair.score).abs() > f64::EPSILON {
            return Err(ArenaError::InvalidOpening {
                id: pair.opening_id.clone(),
                error: "recorded pair score disagrees with game results".to_string(),
            });
        }
        add_result(&mut report, white, true, &pair.engine_a_as_white);
        add_result(&mut report, black, false, &pair.engine_a_as_black);
    }
    Ok(PairedMatchReport {
        match_report: report,
        pairs,
    })
}

pub fn play_game(
    white: &mut dyn Engine,
    black: &mut dyn Engine,
    max_plies: u32,
) -> Result<GameResult, ArenaError> {
    if max_plies == 0 {
        return Err(ArenaError::InvalidConfig(
            "max plies must be greater than zero".to_string(),
        ));
    }

    let board = Board::import_san("").expect("the standard initial position is valid");
    play_game_from_board(board, white, black, max_plies)
}

pub fn play_game_from_opening(
    white: &mut dyn Engine,
    black: &mut dyn Engine,
    opening: &Opening,
    max_plies: u32,
) -> Result<GameResult, ArenaError> {
    let board = Board::import_san(&opening.san).map_err(|error| ArenaError::InvalidOpening {
        id: opening.id.clone(),
        error,
    })?;
    if board.status() != Status::Ongoing {
        return Err(ArenaError::InvalidOpening {
            id: opening.id.clone(),
            error: "opening must end in an ongoing position".to_string(),
        });
    }
    play_game_from_board(board, white, black, max_plies)
}

fn play_game_from_board(
    mut board: Board,
    white: &mut dyn Engine,
    black: &mut dyn Engine,
    max_plies: u32,
) -> Result<GameResult, ArenaError> {
    if max_plies == 0 {
        return Err(ArenaError::InvalidConfig(
            "max plies must be greater than zero".to_string(),
        ));
    }
    let mut plies = 0;

    loop {
        match board.status() {
            Status::Checkmate => {
                return Ok(GameResult {
                    winner: Some(board.side_to_move.opposite()),
                    termination: Termination::Checkmate,
                    plies,
                });
            }
            Status::Stalemate => return Ok(draw(Termination::Stalemate, plies)),
            Status::InsufficientMaterial => {
                return Ok(draw(Termination::InsufficientMaterial, plies));
            }
            Status::ThreefoldRepetition => {
                return Ok(draw(Termination::ThreefoldRepetition, plies));
            }
            Status::FiftyMoveRule => return Ok(draw(Termination::FiftyMoveRule, plies)),
            Status::Ongoing => {}
        }

        if plies >= max_plies {
            return Ok(draw(Termination::PlyLimit, plies));
        }

        let engine: &mut dyn Engine = match board.side_to_move {
            Color::White => white,
            Color::Black => black,
        };
        let name = engine.name().to_string();
        let mv = engine
            .choose_move(&board)
            .map_err(|error| ArenaError::Engine {
                name: name.clone(),
                error,
            })?;
        if !board.get_legal_moves().contains(&mv) {
            return Err(ArenaError::IllegalMove { name, mv });
        }
        board.make_move(mv);
        plies += 1;
    }
}

fn outcome_for_color(result: &GameResult, color: Color) -> Outcome {
    match result.winner {
        Some(winner) if winner == color => Outcome::Win,
        Some(_) => Outcome::Loss,
        None => Outcome::Draw,
    }
}

fn outcome_points(outcome: Outcome) -> f64 {
    match outcome {
        Outcome::Win => 1.0,
        Outcome::Draw => 0.5,
        Outcome::Loss => 0.0,
    }
}

fn add_result(report: &mut MatchReport, outcome: Outcome, as_white: bool, result: &GameResult) {
    report.overall.add(outcome);
    if as_white {
        report.as_white.add(outcome);
    } else {
        report.as_black.add(outcome);
    }
    match result.termination {
        Termination::Stalemate => report.draws.stalemate += 1,
        Termination::InsufficientMaterial => report.draws.insufficient_material += 1,
        Termination::ThreefoldRepetition => report.draws.threefold_repetition += 1,
        Termination::FiftyMoveRule => report.draws.fifty_move_rule += 1,
        Termination::PlyLimit => report.draws.ply_limit += 1,
        Termination::Checkmate => {}
    }
}

fn percentile_index(len: usize, percentile: f64) -> usize {
    (((len - 1) as f64 * percentile).round() as usize).min(len - 1)
}

fn draw(termination: Termination, plies: u32) -> GameResult {
    GameResult {
        winner: None,
        termination,
        plies,
    }
}

/// Convert expected match score to an Elo difference.
pub fn elo_from_score(score: f64) -> f64 {
    400.0 * (score / (1.0 - score)).log10()
}

fn wilson_score_interval(points: f64, games: u32) -> (f64, f64) {
    let n = f64::from(games);
    let score = points / n;
    let z = 1.959_963_984_540_054_f64;
    let denominator = 1.0 + z * z / n;
    let center = (score + z * z / (2.0 * n)) / denominator;
    let margin = z * ((score * (1.0 - score) + z * z / (4.0 * n)) / n).sqrt() / denominator;
    (center - margin, center + margin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_core::{Piece, PieceKind, Square};

    struct FirstLegal(&'static str);

    impl Engine for FirstLegal {
        fn name(&self) -> &str {
            self.0
        }

        fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
            board
                .get_legal_moves()
                .into_iter()
                .next()
                .ok_or_else(|| "no move".to_string())
        }
    }

    #[test]
    fn converts_score_to_elo() {
        let elo = elo_from_score(0.75);
        assert!((elo - 190.8485).abs() < 0.001);
        assert_eq!(elo_from_score(0.5), 0.0);
    }

    #[test]
    fn frozen_minimax_rungs_have_explicit_names() {
        for depth in 1..=3 {
            let engine = MinimaxEngine::new(SearchLimits::fixed_depth(depth).unwrap()).unwrap();
            assert_eq!(engine.name(), format!("MinimaxDepth{depth}V1"));
        }
    }

    #[test]
    fn position_random_is_reproducible_without_hidden_stream_state() {
        let board = Board::import_san("1. e4 e5 2. Nf3").unwrap();
        let mut first = PositionRandomEngine::seeded(7);
        let mut second = PositionRandomEngine::seeded(7);
        let expected = first.choose_move(&board).unwrap();
        assert_eq!(first.choose_move(&board).unwrap(), expected);
        assert_eq!(second.choose_move(&board).unwrap(), expected);
    }

    #[test]
    fn alternates_colors_and_records_adjudicated_draws() {
        let mut a = FirstLegal("A");
        let mut b = FirstLegal("B");
        let report = run_match(
            &mut a,
            &mut b,
            MatchConfig {
                games: 3,
                max_plies: 1,
            },
        )
        .unwrap();

        assert_eq!(
            report.overall,
            Record {
                wins: 0,
                draws: 3,
                losses: 0
            }
        );
        assert_eq!(report.as_white.games(), 2);
        assert_eq!(report.as_black.games(), 1);
        assert_eq!(report.draws.ply_limit, 3);
        assert_eq!(report.elo_difference(), 0.0);
    }

    struct IllegalEngine;

    impl Engine for IllegalEngine {
        fn name(&self) -> &str {
            "Illegal"
        }

        fn choose_move(&mut self, _board: &Board) -> Result<Move, String> {
            Ok(Move {
                piece: Piece {
                    color: Color::White,
                    kind: PieceKind::Queen,
                },
                start_square: Square::new(3, 0),
                end_square: Square::new(3, 7),
                promotion: None,
            })
        }
    }

    #[test]
    fn rejects_illegal_engine_moves() {
        let mut illegal = IllegalEngine;
        let mut legal = FirstLegal("Legal");
        let error = play_game(&mut illegal, &mut legal, 1).unwrap_err();
        assert!(matches!(error, ArenaError::IllegalMove { .. }));
    }

    #[test]
    fn paired_match_uses_each_opening_with_colors_reversed() {
        let openings = vec![
            Opening::new("king-pawn", "1. e4 e5 2. Nf3 Nc6").unwrap(),
            Opening::new("queen-pawn", "1. d4 d5 2. c4 e6").unwrap(),
        ];
        let mut a = FirstLegal("A");
        let mut b = FirstLegal("B");
        let paired = run_paired_match(&mut a, &mut b, &openings, 1).unwrap();

        assert_eq!(paired.pairs.len(), 2);
        assert_eq!(paired.match_report.overall.games(), 4);
        assert_eq!(paired.match_report.as_white.games(), 2);
        assert_eq!(paired.match_report.as_black.games(), 2);
        assert!(paired.pairs.iter().all(|pair| pair.score == 0.5));

        let interval = paired_score_bootstrap_95(
            &paired,
            BootstrapConfig {
                samples: 100,
                seed: 7,
            },
        )
        .unwrap();
        assert_eq!(interval, (0.5, 0.5));
    }

    #[test]
    fn rejects_invalid_or_terminal_openings() {
        let invalid = Opening::new("bad", "1. e5").unwrap_err();
        assert!(matches!(invalid, ArenaError::InvalidOpening { .. }));

        let terminal = Opening::new("mate", "1. f3 e5 2. g4 Qh4#").unwrap_err();
        assert!(matches!(terminal, ArenaError::InvalidOpening { .. }));

        let duplicate = Opening::new("same", "1. e4 e5").unwrap();
        let mut a = FirstLegal("A");
        let mut b = FirstLegal("B");
        let error =
            run_paired_match(&mut a, &mut b, &[duplicate.clone(), duplicate], 1).unwrap_err();
        assert!(matches!(error, ArenaError::InvalidOpening { .. }));
    }

    #[test]
    fn committed_alphamini_opening_suite_is_replayable_and_balanced() {
        let suite: OpeningSuite =
            serde_json::from_str(include_str!("../openings/alphamini-v1.json")).unwrap();
        let openings = suite.validate().unwrap();
        assert_eq!(suite.seed, 1);
        assert_eq!(suite.plies, 8);
        assert_eq!(openings.len(), 200);
    }

    #[test]
    #[ignore = "expensive frozen-depth regression; run before publishing a result"]
    fn committed_opening_scores_match_frozen_depth_three() {
        let suite: OpeningSuite =
            serde_json::from_str(include_str!("../openings/alphamini-v1.json")).unwrap();
        assert_eq!(suite.validate_deep().unwrap().len(), 200);
    }

    #[test]
    fn frozen_minimax_ladder_matches_move_digest() {
        assert_eq!(minimax_v1_move_digest().unwrap(), MINIMAX_V1_MOVE_DIGEST);
    }

    #[cfg(feature = "minigpt")]
    fn minigpt_engine(seed: u64) -> MiniGptEngine {
        MiniGptEngine::with_evaluator(
            Box::new(minigpt::UniformEvaluator),
            256,
            0.5,
            seed,
            "fixture",
        )
        .unwrap()
    }

    #[cfg(feature = "minigpt")]
    #[test]
    fn minigpt_plays_legal_moves_from_committed_openings() {
        let suite: OpeningSuite =
            serde_json::from_str(include_str!("../openings/alphamini-v1.json")).unwrap();
        let mut engine = minigpt_engine(1);
        for entry in suite.openings.iter().take(8) {
            let mut board = Board::import_san(&entry.san).unwrap();
            for _ in 0..12 {
                if board.status() != Status::Ongoing {
                    break;
                }
                let chosen = engine.choose_move(&board).unwrap();
                assert!(
                    board.get_legal_moves().contains(&chosen),
                    "opening {} produced {chosen:?}",
                    entry.id
                );
                board.make_move(chosen);
            }
        }
    }

    #[cfg(feature = "minigpt")]
    #[test]
    fn minigpt_is_reproducible_from_its_seed() {
        let play = |seed| {
            let mut engine = minigpt_engine(seed);
            let mut board = Board::import_san("1. e4 e5 2. Nf3 Nc6").unwrap();
            let mut opponent = PositionRandomEngine::seeded(9);
            let mut played = Vec::new();
            for ply in 0..24 {
                if board.status() != Status::Ongoing {
                    break;
                }
                let chosen = if ply % 2 == 0 {
                    engine.choose_move(&board).unwrap()
                } else {
                    opponent.choose_move(&board).unwrap()
                };
                board.make_move(chosen);
                played.push(chosen);
            }
            played
        };
        assert_eq!(play(7), play(7));
        assert_ne!(play(7), play(8));
    }
}
