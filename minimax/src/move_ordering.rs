use chess_core::{Board, Move, Piece, PieceKind};

const PV_MOVE: i32 = 1_000_000;
const CAPTURE: i32 = 100_000;
const PROMOTION: i32 = 80_000;
const CASTLING: i32 = 2_000;

/// Order moves for alpha-beta pruning.
pub fn order_moves(board: &Board, moves: &mut [Move], principal_variation: Option<Move>) {
    moves.sort_unstable_by_key(|mv| {
        std::cmp::Reverse(move_order_score(board, *mv, principal_variation))
    });
}

pub fn move_order_score(board: &Board, mv: Move, principal_variation: Option<Move>) -> i32 {
    let mut score = 0;
    if Some(mv) == principal_variation {
        score += PV_MOVE;
    }

    if let Some(victim) = captured_piece(board, mv) {
        score += CAPTURE + piece_value(victim.kind) * 16 - piece_value(mv.piece.kind);
    }
    if let Some(promoted) = mv.promotion {
        score += PROMOTION + piece_value(promoted);
    }
    if mv.piece.kind == PieceKind::King
        && mv.start_square.file().abs_diff(mv.end_square.file()) == 2
    {
        score += CASTLING;
    }
    score
}

pub fn captured_piece(board: &Board, mv: Move) -> Option<Piece> {
    if let Some(piece) = board.piece_at(mv.end_square) {
        return Some(piece);
    }
    let is_en_passant = mv.piece.kind == PieceKind::Pawn
        && Some(mv.end_square) == board.en_passant
        && mv.start_square.file() != mv.end_square.file();
    if is_en_passant {
        let square = chess_core::Square::new(mv.end_square.file(), mv.start_square.rank());
        return board.piece_at(square);
    }
    None
}

fn piece_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Knight => 320,
        PieceKind::Bishop => 335,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 20_000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_before_quiet_move() {
        let board = Board::from_fen("4k3/8/8/8/3q4/2P5/8/4K3 w - - 0 1").unwrap();
        let mut moves = board.get_legal_moves();
        order_moves(&board, &mut moves, None);
        assert_eq!(moves[0].to_uci(), "c3d4");
    }

    #[test]
    fn pv_move_first() {
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut moves = board.get_legal_moves();
        let pv = board.move_from_uci("a2a3").unwrap();
        order_moves(&board, &mut moves, Some(pv));
        assert_eq!(moves[0], pv);
    }

    #[test]
    fn en_passant_capture() {
        let board = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let mv = board.move_from_uci("e5d6").unwrap();
        assert_eq!(captured_piece(&board, mv).unwrap().kind, PieceKind::Pawn);
    }
}
