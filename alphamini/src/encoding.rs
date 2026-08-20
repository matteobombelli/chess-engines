use chess_core::{Board, Color, PieceKind, SearchPosition, Square};
use serde::{Deserialize, Serialize};

pub const ENCODER_VERSION: &str = "encoder-v1";
pub const INPUT_PLANES: usize = 22;
pub const BOARD_SQUARES: usize = 64;
pub const INPUT_VALUES: usize = INPUT_PLANES * BOARD_SQUARES;

/// Information that is not recoverable from a standalone FEN/`Board`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodingContext {
    /// Number of earlier occurrences of the current repetition key, clipped to 2.
    pub prior_occurrences: u8,
}

/// Contiguous NCHW data for one position (C=22, H=8, W=8).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EncodedPosition {
    pub version: String,
    pub values: Vec<f32>,
}

impl EncodedPosition {
    pub fn zeros() -> Self {
        Self {
            version: ENCODER_VERSION.to_string(),
            values: vec![0.0; INPUT_VALUES],
        }
    }

    pub fn plane(&self, plane: usize) -> &[f32] {
        let start = plane * BOARD_SQUARES;
        &self.values[start..start + BOARD_SQUARES]
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != ENCODER_VERSION {
            return Err(format!(
                "unsupported encoder version {}, expected {ENCODER_VERSION}",
                self.version
            ));
        }
        if self.values.len() != INPUT_VALUES {
            return Err(format!(
                "encoder produced {} values, expected {INPUT_VALUES}",
                self.values.len()
            ));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err("encoded input contains a non-finite value".to_string());
        }
        Ok(())
    }
}

/// Encode a board in the frozen, history-free `encoder-v1` layout.
///
/// Plane layout:
/// 0..6 own P/N/B/R/Q/K, 6..12 opponent P/N/B/R/Q/K, 12/13 repetition,
/// 14 absolute side (1=White), 15..19 own/opponent K/Q castling, 19 EP,
/// 20 halfmove clock, 21 ones.
pub fn encode(board: &Board, context: EncodingContext) -> EncodedPosition {
    encode_search(&SearchPosition::from_board(board), context)
}

/// Encode search-native state without reconstructing a `Board` or string
/// history. The supplied context is explicit so raw records can be rematerialized.
pub fn encode_search(position: &SearchPosition, context: EncodingContext) -> EncodedPosition {
    let mut encoded = EncodedPosition::zeros();

    for index in 0..64 {
        let square = Square(index as u8);
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        let owner_offset = if piece.color == position.side_to_move() {
            0
        } else {
            6
        };
        let plane = owner_offset + piece_plane(piece.kind);
        let canonical = canonical_square(square, position.side_to_move()).index();
        encoded.values[plane * 64 + canonical] = 1.0;
    }

    fill_plane(
        &mut encoded,
        12,
        (context.prior_occurrences >= 1) as u8 as f32,
    );
    fill_plane(
        &mut encoded,
        13,
        (context.prior_occurrences >= 2) as u8 as f32,
    );
    fill_plane(
        &mut encoded,
        14,
        (position.side_to_move() == Color::White) as u8 as f32,
    );

    let castling = position.castling_rights();
    let (own_k, own_q, opponent_k, opponent_q) = match position.side_to_move() {
        Color::White => (
            castling.white_kingside,
            castling.white_queenside,
            castling.black_kingside,
            castling.black_queenside,
        ),
        Color::Black => (
            castling.black_kingside,
            castling.black_queenside,
            castling.white_kingside,
            castling.white_queenside,
        ),
    };
    for (plane, allowed) in [(15, own_k), (16, own_q), (17, opponent_k), (18, opponent_q)] {
        fill_plane(&mut encoded, plane, allowed as u8 as f32);
    }

    if let Some(square) = position.en_passant_target() {
        let canonical = canonical_square(square, position.side_to_move()).index();
        encoded.values[19 * 64 + canonical] = 1.0;
    }
    fill_plane(
        &mut encoded,
        20,
        position.halfmove_clock().min(100) as f32 / 100.0,
    );
    fill_plane(&mut encoded, 21, 1.0);
    encoded
}

/// Encode a live search path using its exact repetition count.
pub fn encode_search_current(position: &SearchPosition) -> EncodedPosition {
    encode_search(
        position,
        EncodingContext {
            prior_occurrences: position.prior_repetition_count().min(2) as u8,
        },
    )
}

/// Rank-flip Black positions while deliberately preserving files.
pub fn canonical_square(square: Square, side_to_move: Color) -> Square {
    match side_to_move {
        Color::White => square,
        Color::Black => Square::new(square.file(), 7 - square.rank()),
    }
}

fn piece_plane(kind: PieceKind) -> usize {
    match kind {
        PieceKind::Pawn => 0,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::Queen => 4,
        PieceKind::King => 5,
    }
}

fn fill_plane(encoded: &mut EncodedPosition, plane: usize, value: f32) {
    encoded.values[plane * 64..(plane + 1) * 64].fill(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_has_canonical_piece_planes() {
        let board = Board::from_fen(crate::START_FEN).unwrap();
        let input = encode(&board, EncodingContext::default());
        input.validate().unwrap();

        assert_eq!(input.plane(0).iter().sum::<f32>(), 8.0);
        assert_eq!(input.plane(5).iter().sum::<f32>(), 1.0);
        assert_eq!(input.plane(6).iter().sum::<f32>(), 8.0);
        assert!(input.plane(14).iter().all(|&value| value == 1.0));
        assert!(input.plane(15).iter().all(|&value| value == 1.0));
        assert!(input.plane(18).iter().all(|&value| value == 1.0));
        assert!(input.plane(21).iter().all(|&value| value == 1.0));
    }

    #[test]
    fn black_rank_flip_and_context_planes_are_exact() {
        let board = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 b - e3 100 80").unwrap();
        let input = encode(
            &board,
            EncodingContext {
                prior_occurrences: 2,
            },
        );

        // Black king e8 becomes own king on canonical e1.
        assert_eq!(input.plane(5)[Square::new(4, 0).index()], 1.0);
        // White pawn e2 becomes opponent pawn on canonical e7.
        assert_eq!(input.plane(6)[Square::new(4, 6).index()], 1.0);
        // e3 rank-flips to e6.
        assert_eq!(input.plane(19)[Square::new(4, 5).index()], 1.0);
        assert!(input.plane(12).iter().all(|&value| value == 1.0));
        assert!(input.plane(13).iter().all(|&value| value == 1.0));
        assert!(input.plane(14).iter().all(|&value| value == 0.0));
        assert!(input.plane(20).iter().all(|&value| value == 1.0));
    }
}
