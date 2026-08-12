pub mod config;
pub mod evaluation;
pub mod move_ordering;
pub mod search;

use std::fmt;

use chess_core::{Board, Move, Status};
use serde::{Deserialize, Serialize};

pub use config::SearchLimits;
pub use search::{SearchError, SearchResult, SearchStats, find_best_move};

/// One request to the bot.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BotRequest {
    /// The game so far as PGN movetext.
    #[serde(default)]
    pub san: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BotResponse {
    pub san: String,
    pub fen: String,
}

#[derive(Debug)]
pub enum BotError {
    InvalidGame(String),
    GameOver(Status),
    Search(SearchError),
    IllegalEngineMove(Move),
}

impl fmt::Display for BotError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGame(error) => write!(formatter, "invalid game: {error}"),
            Self::GameOver(status) => write!(formatter, "game is over: {status:?}"),
            Self::Search(error) => error.fmt(formatter),
            Self::IllegalEngineMove(mv) => {
                write!(
                    formatter,
                    "search returned a move that is not legal: {mv:?}"
                )
            }
        }
    }
}

impl std::error::Error for BotError {}

impl From<SearchError> for BotError {
    fn from(error: SearchError) -> Self {
        Self::Search(error)
    }
}

pub fn respond(request: BotRequest) -> Result<BotResponse, BotError> {
    respond_with_limits(request, SearchLimits::default())
}

pub fn respond_with_limits(
    request: BotRequest,
    limits: SearchLimits,
) -> Result<BotResponse, BotError> {
    let board = position_from_request(&request)?;
    let status = board.status();
    if status != Status::Ongoing {
        return Err(BotError::GameOver(status));
    }
    let result = find_best_move(&board, limits)?;
    apply_engine_move(board, result.best_move)
}

/// Rebuild the position from the request.
pub fn position_from_request(request: &BotRequest) -> Result<Board, BotError> {
    Board::import_san(request.san.as_deref().unwrap_or("")).map_err(BotError::InvalidGame)
}

/// Apply the engine's move after checking it is legal.
pub fn apply_engine_move(mut board: Board, mv: Move) -> Result<BotResponse, BotError> {
    if !board.get_legal_moves().contains(&mv) {
        return Err(BotError::IllegalEngineMove(mv));
    }
    board.make_move(mv);
    Ok(BotResponse {
        san: board
            .san_history
            .last()
            .cloned()
            .expect("make_move records SAN"),
        fen: board.to_fen(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_core::{Color, Piece, PieceKind, Square};

    #[test]
    fn replay_and_respond() {
        let request = BotRequest {
            san: Some("1. e4 e5 2. Nf3".to_string()),
        };
        let board = position_from_request(&request).unwrap();
        let mv = board.get_legal_moves()[0];
        let response = apply_engine_move(board.clone(), mv).unwrap();

        let mut expected = board;
        expected.san_to_move(&response.san).unwrap();
        assert_eq!(response.fen, expected.to_fen());
    }

    #[test]
    fn rejects_illegal_request() {
        let error = respond(BotRequest {
            san: Some("1. e5".to_string()),
        })
        .unwrap_err();
        assert!(matches!(error, BotError::InvalidGame(_)));
    }

    #[test]
    fn reports_finished_game() {
        let error = respond(BotRequest {
            san: Some("1. f3 e5 2. g4 Qh4#".to_string()),
        })
        .unwrap_err();
        assert!(matches!(error, BotError::GameOver(Status::Checkmate)));
    }

    #[test]
    fn rejects_illegal_engine_move() {
        let board = Board::import_san("").unwrap();
        let fake = Move {
            piece: Piece {
                color: Color::White,
                kind: PieceKind::Queen,
            },
            start_square: Square::new(3, 0),
            end_square: Square::new(3, 7),
            promotion: None,
        };
        assert!(matches!(
            apply_engine_move(board, fake),
            Err(BotError::IllegalEngineMove(_))
        ));
    }
}
