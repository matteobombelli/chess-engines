use chess_core::{Board, Color, Move, PieceKind, SearchPosition, Square};
use thiserror::Error;

use crate::encoding::canonical_square;

pub const POLICY_VERSION: &str = "policy-v1";
pub const MOVES_PER_SQUARE: usize = 73;
pub const POLICY_SIZE: usize = 64 * MOVES_PER_SQUARE;

const RAY_DIRECTIONS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];
const KNIGHT_DIRECTIONS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PolicyError {
    #[error("move {0} cannot be represented by {POLICY_VERSION}")]
    Unrepresentable(String),
    #[error("policy index {0} is out of range")]
    OutOfRange(usize),
    #[error("policy index {0} does not identify a legal move")]
    IllegalIndex(usize),
    #[error("two legal moves collide at policy index {0}")]
    Collision(usize),
}

pub fn move_to_action(mv: Move, side_to_move: Color) -> Result<usize, PolicyError> {
    let from = canonical_square(mv.start_square, side_to_move);
    let to = canonical_square(mv.end_square, side_to_move);
    let dx = to.file() as i8 - from.file() as i8;
    let dy = to.rank() as i8 - from.rank() as i8;

    let plane = if mv.piece.kind == PieceKind::Pawn
        && matches!(
            mv.promotion,
            Some(PieceKind::Knight | PieceKind::Bishop | PieceKind::Rook)
        ) {
        let direction = match dx {
            -1 => 0,
            0 => 1,
            1 => 2,
            _ => return Err(PolicyError::Unrepresentable(mv.to_uci())),
        };
        if dy != 1 {
            return Err(PolicyError::Unrepresentable(mv.to_uci()));
        }
        let promotion = match mv.promotion {
            Some(PieceKind::Knight) => 0,
            Some(PieceKind::Bishop) => 1,
            Some(PieceKind::Rook) => 2,
            _ => unreachable!("guarded above"),
        };
        64 + direction * 3 + promotion
    } else if let Some(direction) = KNIGHT_DIRECTIONS
        .iter()
        .position(|&(candidate_x, candidate_y)| (candidate_x, candidate_y) == (dx, dy))
    {
        56 + direction
    } else {
        let distance = dx.unsigned_abs().max(dy.unsigned_abs()) as usize;
        if distance == 0 || distance > 7 {
            return Err(PolicyError::Unrepresentable(mv.to_uci()));
        }
        let unit = (dx / distance as i8, dy / distance as i8);
        if unit.0 * distance as i8 != dx || unit.1 * distance as i8 != dy {
            return Err(PolicyError::Unrepresentable(mv.to_uci()));
        }
        let direction = RAY_DIRECTIONS
            .iter()
            .position(|&candidate| candidate == unit)
            .ok_or_else(|| PolicyError::Unrepresentable(mv.to_uci()))?;
        direction * 7 + (distance - 1)
    };

    // Plane-major C order matches flattening a [73, 8, 8] policy head.
    Ok(plane * 64 + from.index())
}

/// Resolve an action only through the legal move set; never invent a move from
/// an untrusted model index.
pub fn action_to_move(board: &Board, action: usize) -> Result<Move, PolicyError> {
    if action >= POLICY_SIZE {
        return Err(PolicyError::OutOfRange(action));
    }
    let mut found = None;
    for mv in board.get_legal_moves() {
        if move_to_action(mv, board.side_to_move)? == action {
            if found.is_some() {
                return Err(PolicyError::Collision(action));
            }
            found = Some(mv);
        }
    }
    found.ok_or(PolicyError::IllegalIndex(action))
}

pub fn action_to_search_move(
    position: &mut SearchPosition,
    action: usize,
) -> Result<Move, PolicyError> {
    if action >= POLICY_SIZE {
        return Err(PolicyError::OutOfRange(action));
    }
    let side = position.side_to_move();
    let mut found = None;
    for mv in position.legal_moves() {
        if move_to_action(mv, side)? == action {
            if found.is_some() {
                return Err(PolicyError::Collision(action));
            }
            found = Some(mv);
        }
    }
    found.ok_or(PolicyError::IllegalIndex(action))
}

pub fn legal_action_mask(board: &Board) -> Result<Vec<bool>, PolicyError> {
    let mut mask = vec![false; POLICY_SIZE];
    for mv in board.get_legal_moves() {
        let action = move_to_action(mv, board.side_to_move)?;
        if std::mem::replace(&mut mask[action], true) {
            return Err(PolicyError::Collision(action));
        }
    }
    Ok(mask)
}

/// Convert a canonical action origin back to an absolute square. Primarily used
/// by fixture tooling; legality-aware decoding should use [`action_to_move`].
pub fn action_origin(action: usize, side_to_move: Color) -> Result<Square, PolicyError> {
    if action >= POLICY_SIZE {
        return Err(PolicyError::OutOfRange(action));
    }
    let canonical = Square((action % 64) as u8);
    Ok(canonical_square(canonical, side_to_move))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn all_start_moves_are_unique_and_round_trip() {
        let board = Board::from_fen(crate::START_FEN).unwrap();
        let legal = board.get_legal_moves();
        let actions: HashSet<_> = legal
            .iter()
            .map(|&mv| move_to_action(mv, board.side_to_move).unwrap())
            .collect();
        assert_eq!(actions.len(), 20);
        for mv in legal {
            let action = move_to_action(mv, board.side_to_move).unwrap();
            assert_eq!(action_to_move(&board, action).unwrap(), mv);
        }
    }

    #[test]
    fn equivalent_white_and_black_pushes_share_canonical_action() {
        let white = Board::from_fen(crate::START_FEN).unwrap();
        let white_action =
            move_to_action(white.move_from_uci("e2e4").unwrap(), Color::White).unwrap();
        let black =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let black_action =
            move_to_action(black.move_from_uci("e7e5").unwrap(), Color::Black).unwrap();
        assert_eq!(white_action, black_action);
    }

    #[test]
    fn every_promotion_is_distinct_and_round_trips() {
        for fen in [
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            let promotions: Vec<_> = board
                .get_legal_moves()
                .into_iter()
                .filter(|mv| mv.promotion.is_some())
                .collect();
            let actions: HashSet<_> = promotions
                .iter()
                .map(|&mv| move_to_action(mv, board.side_to_move).unwrap())
                .collect();
            assert_eq!(actions.len(), 4);
            for mv in promotions {
                assert_eq!(
                    action_to_move(&board, move_to_action(mv, board.side_to_move).unwrap())
                        .unwrap(),
                    mv
                );
            }
        }
    }

    #[test]
    fn castling_and_en_passant_positions_are_collision_free() {
        for fen in [
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            let legal = board.get_legal_moves();
            let actions: HashSet<_> = legal
                .iter()
                .map(|&mv| move_to_action(mv, board.side_to_move).unwrap())
                .collect();
            assert_eq!(actions.len(), legal.len());
            for mv in legal {
                let action = move_to_action(mv, board.side_to_move).unwrap();
                assert_eq!(action_to_move(&board, action).unwrap(), mv);
            }
        }
    }
}
