//! Chess rules shared by the engines and the web front end.
//!
//! The crate owns the board, legal move generation, FEN and SAN, UCI move
//! strings, and a reversible [`SearchPosition`] for tree search.
//! [`Board::status`] ends a game on its own at the third repetition and at 100
//! halfmoves, so neither draw waits for a player to claim it.

mod board;
mod fen;
mod legal_moves;
mod san;
mod search;
mod uci;

pub use board::{Board, CastlingRights, Color, Piece, PieceKind, Square};
pub use legal_moves::{Move, Status};
pub use san::movetext_moves;
pub use search::{SearchPosition, SearchUndo};
