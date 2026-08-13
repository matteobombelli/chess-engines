use std::fmt;

use chess_core::{Board, Color, Move, Status};
use minimax::{SearchLimits, find_best_move};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A chess engine that can be compared in the arena.
///
/// Future bots only need a small adapter implementing this trait. The arena
/// owns all game-state and result handling and verifies every engine move.
pub trait Engine {
    fn name(&self) -> &str;
    fn choose_move(&mut self, board: &Board) -> Result<Move, String>;
}

pub struct RandomEngine {
    rng: StdRng,
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
}

impl MinimaxEngine {
    pub fn new(limits: SearchLimits) -> Result<Self, String> {
        limits.validate()?;
        Ok(Self { limits })
    }
}

impl Engine for MinimaxEngine {
    fn name(&self) -> &str {
        "Minimax"
    }

    fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
        find_best_move(board, self.limits)
            .map(|result| result.best_move)
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
    Engine { name: String, error: String },
    IllegalMove { name: String, mv: Move },
}

impl fmt::Display for ArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => write!(formatter, "invalid match config: {error}"),
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
            Termination::ThreefoldRepetition => report.draws.threefold_repetition += 1,
            Termination::FiftyMoveRule => report.draws.fifty_move_rule += 1,
            Termination::PlyLimit => report.draws.ply_limit += 1,
            Termination::Checkmate => {}
        }

        progress(game_index + 1, &result, &report);
    }

    Ok(report)
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

    let mut board = Board::import_san("").expect("the standard initial position is valid");
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
}
