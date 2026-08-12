use chess_core::{Board, Color, PieceKind, Square};

/// Evaluation in centipawns.
pub type Score = i32;

pub const INFINITY: Score = 32_000;
pub const MATE_SCORE: Score = 30_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvaluationBreakdown {
    pub material: Score,
    pub piece_square: Score,
    pub pawn_structure: Score,
    pub bishop_pair: Score,
    pub rook_files: Score,
    pub king_safety: Score,
    pub tempo: Score,
}

impl EvaluationBreakdown {
    pub fn total(self) -> Score {
        self.material
            + self.piece_square
            + self.pawn_structure
            + self.bishop_pair
            + self.rook_files
            + self.king_safety
            + self.tempo
    }

    fn scaled(self, factor: Score) -> Self {
        Self {
            material: self.material * factor,
            piece_square: self.piece_square * factor,
            pawn_structure: self.pawn_structure * factor,
            bishop_pair: self.bishop_pair * factor,
            rook_files: self.rook_files * factor,
            king_safety: self.king_safety * factor,
            tempo: self.tempo * factor,
        }
    }
}

/// Positive scores favor the side to move.
pub fn evaluate(board: &Board) -> Score {
    evaluate_breakdown(board, board.side_to_move).total()
}

pub fn evaluate_breakdown(board: &Board, perspective: Color) -> EvaluationBreakdown {
    let mut score = white_perspective_breakdown(board);
    if perspective == Color::Black {
        score = score.scaled(-1);
    }
    score
}

fn white_perspective_breakdown(board: &Board) -> EvaluationBreakdown {
    let phase = game_phase(board);
    let mut result = EvaluationBreakdown::default();
    let mut bishops = [0_u8; 2];
    let mut pawns_by_file = [[0_u8; 8]; 2];

    for index in 0..64 {
        let square = Square(index);
        let Some(piece) = board.piece_at(square) else {
            continue;
        };
        let sign = color_sign(piece.color);
        result.material += sign * piece_value(piece.kind);
        result.piece_square += sign * positional_value(piece.kind, piece.color, square, phase);

        if piece.kind == PieceKind::Bishop {
            bishops[color_index(piece.color)] += 1;
        }
        if piece.kind == PieceKind::Pawn {
            pawns_by_file[color_index(piece.color)][square.file() as usize] += 1;
        }
    }

    for color in [Color::White, Color::Black] {
        let sign = color_sign(color);
        if bishops[color_index(color)] >= 2 {
            result.bishop_pair += sign * 30;
        }
        result.pawn_structure += sign * pawn_structure(board, color, &pawns_by_file);
        result.rook_files += sign * rook_file_score(board, color, &pawns_by_file);
        result.king_safety += sign * king_safety(board, color, phase);
    }

    result.tempo = color_sign(board.side_to_move) * 10;
    result
}

fn piece_value(kind: PieceKind) -> Score {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Knight => 320,
        PieceKind::Bishop => 335,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 0,
    }
}

/// `phase` is 0 in an ending and 24 in the starting position.
fn positional_value(kind: PieceKind, color: Color, square: Square, phase: Score) -> Score {
    let file = square.file() as Score;
    let rank = relative_rank(color, square) as Score;
    let file_center = 7 - (2 * file - 7).abs();
    let rank_center = 7 - (2 * rank - 7).abs();
    let centrality = file_center + rank_center;

    match kind {
        PieceKind::Pawn => rank * 7 + file_center * 2,
        PieceKind::Knight => centrality * 5 - edge_penalty(square) * 12,
        PieceKind::Bishop => centrality * 3 - edge_penalty(square) * 3,
        PieceKind::Rook => {
            let seventh_rank_bonus = if rank == 6 { 22 } else { 0 };
            seventh_rank_bonus + file_center
        }
        PieceKind::Queen => centrality * 2,
        PieceKind::King => {
            let middlegame = -centrality * 5 + castled_king_bonus(square);
            let endgame = centrality * 6;
            (middlegame * phase + endgame * (24 - phase)) / 24
        }
    }
}

fn pawn_structure(board: &Board, color: Color, pawns_by_file: &[[u8; 8]; 2]) -> Score {
    let mine = &pawns_by_file[color_index(color)];
    let mut score = 0;

    for file in 0..8 {
        if mine[file] > 1 {
            score -= 14 * Score::from(mine[file] - 1);
        }

        for rank in 0..8 {
            let square = Square::new(file as u8, rank);
            if !board
                .piece_at(square)
                .is_some_and(|piece| piece.color == color && piece.kind == PieceKind::Pawn)
            {
                continue;
            }

            let has_left = file > 0 && mine[file - 1] > 0;
            let has_right = file < 7 && mine[file + 1] > 0;
            if !has_left && !has_right {
                score -= 11;
            }
            if connected_pawn(board, color, square) {
                score += 5;
            }
            if passed_pawn(board, color, square) {
                let advancement = Score::from(relative_rank(color, square));
                score += 12 + advancement * advancement * 3;
            }
        }
    }
    score
}

fn connected_pawn(board: &Board, color: Color, square: Square) -> bool {
    for file_delta in [-1_i8, 1] {
        for rank_delta in [-1_i8, 0, 1] {
            if let Some(neighbor) = offset(square, file_delta, rank_delta) {
                if board
                    .piece_at(neighbor)
                    .is_some_and(|piece| piece.color == color && piece.kind == PieceKind::Pawn)
                {
                    return true;
                }
            }
        }
    }
    false
}

fn passed_pawn(board: &Board, color: Color, square: Square) -> bool {
    let direction: i8 = if color == Color::White { 1 } else { -1 };
    for file_delta in [-1_i8, 0, 1] {
        let mut next = offset(square, file_delta, direction);
        while let Some(target) = next {
            if board.piece_at(target).is_some_and(|piece| {
                piece.color == color.opposite() && piece.kind == PieceKind::Pawn
            }) {
                return false;
            }
            next = offset(target, 0, direction);
        }
    }
    true
}

fn rook_file_score(board: &Board, color: Color, pawns_by_file: &[[u8; 8]; 2]) -> Score {
    let mine = &pawns_by_file[color_index(color)];
    let theirs = &pawns_by_file[color_index(color.opposite())];
    let mut score = 0;
    for index in 0..64 {
        let square = Square(index);
        if board
            .piece_at(square)
            .is_some_and(|piece| piece.color == color && piece.kind == PieceKind::Rook)
        {
            let file = square.file() as usize;
            if mine[file] == 0 {
                score += 10;
                if theirs[file] == 0 {
                    score += 12;
                }
            }
        }
    }
    score
}

fn king_safety(board: &Board, color: Color, phase: Score) -> Score {
    if phase < 8 {
        return 0;
    }
    let Some(king) = (0..64).map(Square).find(|&square| {
        board
            .piece_at(square)
            .is_some_and(|piece| piece.color == color && piece.kind == PieceKind::King)
    }) else {
        return 0;
    };

    let direction: i8 = if color == Color::White { 1 } else { -1 };
    let mut score = 0;
    for file_delta in [-1_i8, 0, 1] {
        if let Some(shield) = offset(king, file_delta, direction) {
            if board
                .piece_at(shield)
                .is_some_and(|piece| piece.color == color && piece.kind == PieceKind::Pawn)
            {
                score += 8;
            }
        }
    }
    score
}

fn game_phase(board: &Board) -> Score {
    let phase = (0..64).fold(0, |phase, index| {
        phase
            + match board.piece_at(Square(index)).map(|piece| piece.kind) {
                Some(PieceKind::Knight | PieceKind::Bishop) => 1,
                Some(PieceKind::Rook) => 2,
                Some(PieceKind::Queen) => 4,
                _ => 0,
            }
    });
    phase.min(24)
}

fn relative_rank(color: Color, square: Square) -> u8 {
    match color {
        Color::White => square.rank(),
        Color::Black => 7 - square.rank(),
    }
}

fn castled_king_bonus(square: Square) -> Score {
    if matches!(square.file(), 2 | 6) && matches!(square.rank(), 0 | 7) {
        35
    } else {
        0
    }
}

fn edge_penalty(square: Square) -> Score {
    Score::from(u8::from(matches!(square.file(), 0 | 7)) + u8::from(matches!(square.rank(), 0 | 7)))
}

fn offset(square: Square, file_delta: i8, rank_delta: i8) -> Option<Square> {
    let file = square.file() as i8 + file_delta;
    let rank = square.rank() as i8 + rank_delta;
    ((0..8).contains(&file) && (0..8).contains(&rank)).then(|| Square::new(file as u8, rank as u8))
}

fn color_index(color: Color) -> usize {
    usize::from(color == Color::Black)
}

fn color_sign(color: Color) -> Score {
    if color == Color::White { 1 } else { -1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn starting_position() {
        let board = Board::from_fen(START).unwrap();
        assert_eq!(evaluate(&board), 10);
    }

    #[test]
    fn extra_queen() {
        let board = Board::from_fen("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1").unwrap();
        let white = evaluate_breakdown(&board, Color::White).total();
        let black = evaluate_breakdown(&board, Color::Black).total();
        assert!(white > 850, "white score was {white}");
        assert_eq!(black, -white);
    }

    #[test]
    fn passed_pawn_bonus() {
        let low = Board::from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1").unwrap();
        let high = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let low_score = evaluate_breakdown(&low, Color::White).pawn_structure;
        let high_score = evaluate_breakdown(&high, Color::White).pawn_structure;
        assert!(high_score > low_score);
    }
}
