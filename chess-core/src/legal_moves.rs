use crate::board::*;

/// A move on the board
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub piece: Piece,
    pub start_square: Square,
    pub end_square: Square,

    /// The piece a promoting pawn turns into, or None for any other move
    pub promotion: Option<PieceKind>,
}

/// The state of the game for the side to move
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Ongoing,
    Checkmate,
    Stalemate,
}

impl Board {
    /// Apply a move to the Board
    pub fn make_move(&mut self, mv: Move) {
        let color: Color = mv.piece.color;
        let from: Square = mv.start_square;
        let to: Square = mv.end_square;

        // Remember the en-passant target before we overwrite it below
        let ep_target: Option<Square> = self.en_passant;
        let is_capture: bool = self.piece_at(to).is_some()
            || (mv.piece.kind == PieceKind::Pawn && Some(to) == ep_target);

        // Lift the moving piece off its start square
        self.set_piece(from, None);

        match mv.piece.kind {
            PieceKind::Pawn => {
                // En-passant capture: a diagonal step onto the en-passant target
                // takes the pawn that just double-pushed, sitting behind `to`
                if Some(to) == ep_target && from.file() != to.file() {
                    self.set_piece(Square::new(to.file(), from.rank()), None);
                }

                // Promotion: a pawn reaching the back rank becomes the chosen
                // piece, defaulting to a queen when no choice was given
                if to.rank() == 0 || to.rank() == 7 {
                    let kind: PieceKind = mv.promotion.unwrap_or(PieceKind::Queen);
                    self.set_piece(to, Some(Piece { color, kind }));
                } else {
                    self.set_piece(to, Some(mv.piece));
                }
            }
            PieceKind::King => {
                self.set_piece(to, Some(mv.piece));

                // Castling: the king steps two files, so the rook jumps across it
                let rook: Piece = Piece { color, kind: PieceKind::Rook };
                if from.file() == 4 && to.file() == 6 {
                    // Kingside: h-rook to f
                    self.set_piece(Square::new(7, from.rank()), None);
                    self.set_piece(Square::new(5, from.rank()), Some(rook));
                } else if from.file() == 4 && to.file() == 2 {
                    // Queenside: a-rook to d
                    self.set_piece(Square::new(0, from.rank()), None);
                    self.set_piece(Square::new(3, from.rank()), Some(rook));
                }
            }
            _ => self.set_piece(to, Some(mv.piece)),
        }

        // Castling rights: a king move, or a rook leaving or being captured on
        // its home square, revokes the matching right
        if mv.piece.kind == PieceKind::King {
            match color {
                Color::White => {
                    self.castling.white_kingside = false;
                    self.castling.white_queenside = false;
                }
                Color::Black => {
                    self.castling.black_kingside = false;
                    self.castling.black_queenside = false;
                }
            }
        }
        for sq in [from, to] {
            if sq == Square::new(0, 0) { self.castling.white_queenside = false; }
            if sq == Square::new(7, 0) { self.castling.white_kingside = false; }
            if sq == Square::new(0, 7) { self.castling.black_queenside = false; }
            if sq == Square::new(7, 7) { self.castling.black_kingside = false; }
        }

        // En-passant target: a pawn double-push leaves the square it skipped over
        self.en_passant = match mv.piece.kind {
            PieceKind::Pawn if from.rank().abs_diff(to.rank()) == 2 => {
                Some(Square::new(from.file(), (from.rank() + to.rank()) / 2))
            }
            _ => None,
        };

        // Clocks: halfmove resets on a pawn move or capture, fullmove ticks after Black
        self.halfmove_clock = if mv.piece.kind == PieceKind::Pawn || is_capture {
            0
        } else {
            self.halfmove_clock + 1
        };
        if color == Color::Black {
            self.fullmove_number += 1;
        }

        self.side_to_move = color.opposite();
    }

    /// Get all legal moves from the current board position
    pub fn get_legal_moves(&self) -> Vec<Move> {
        let me: Color = self.side_to_move;
        let mut legal_moves: Vec<Move> = Vec::new();

        // A pseudo-legal move is legal only if it doesn't leave our king in check
        for mv in self.pseudo_legal_moves() {
            let mut next: Board = self.clone();
            next.make_move(mv);
            if let Some(king) = next.find_king(me) {
                if !next.is_attacked(king, me.opposite()) {
                    legal_moves.push(mv);
                }
            }
        }

        legal_moves
    }

    /// Whether the side to move is currently in check
    pub fn is_in_check(&self) -> bool {
        match self.find_king(self.side_to_move) {
            Some(king) => self.is_attacked(king, self.side_to_move.opposite()),
            None => false,
        }
    }

    /// Classify the position for the side to move: a side with no legal moves is
    /// checkmated if it is in check, otherwise stalemated
    pub fn status(&self) -> Status {
        if !self.get_legal_moves().is_empty() {
            Status::Ongoing
        } else if self.is_in_check() {
            Status::Checkmate
        } else {
            Status::Stalemate
        }
    }

    /// All pseudo-legal moves for the side to move, ignoring whether they leave
    /// our own king in check
    fn pseudo_legal_moves(&self) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::new();
        for rank in 0..8 {
            for file in 0..8 {
                let from: Square = Square::new(file, rank);
                let piece: Piece = match self.piece_at(from) {
                    Some(p) if p.color == self.side_to_move => p,
                    _ => continue,
                };
                match piece.kind {
                    PieceKind::Pawn => self.gen_pawn_moves(from, piece, &mut moves),
                    PieceKind::Knight => self.gen_step_moves(from, piece, &KNIGHT_OFFSETS, &mut moves),
                    PieceKind::Bishop => self.gen_slide_moves(from, piece, &BISHOP_DIRS, &mut moves),
                    PieceKind::Rook => self.gen_slide_moves(from, piece, &ROOK_DIRS, &mut moves),
                    PieceKind::Queen => self.gen_slide_moves(from, piece, &QUEEN_DIRS, &mut moves),
                    PieceKind::King => {
                        self.gen_step_moves(from, piece, &KING_OFFSETS, &mut moves);
                        self.gen_castling_moves(from, piece, &mut moves);
                    }
                }
            }
        }
        moves
    }

    /// Generate single-step moves (knight, king) from a list of offsets
    fn gen_step_moves(&self, from: Square, piece: Piece, offsets: &[(i8, i8)], moves: &mut Vec<Move>) {
        for &(df, dr) in offsets {
            if let Some(to) = offset_square(from, df, dr) {
                // We may land on an empty square or capture an enemy piece
                match self.piece_at(to) {
                    Some(target) if target.color == piece.color => continue,
                    _ => moves.push(Move { piece, start_square: from, end_square: to, promotion: None }),
                }
            }
        }
    }

    /// Generate sliding moves (bishop, rook, queen) along a list of directions
    fn gen_slide_moves(&self, from: Square, piece: Piece, dirs: &[(i8, i8)], moves: &mut Vec<Move>) {
        for &(df, dr) in dirs {
            let mut to: Option<Square> = offset_square(from, df, dr);
            while let Some(sq) = to {
                match self.piece_at(sq) {
                    Some(target) => {
                        // Stop at the first piece, capturing it if it's an enemy
                        if target.color != piece.color {
                            moves.push(Move { piece, start_square: from, end_square: sq, promotion: None });
                        }
                        break;
                    }
                    None => {
                        moves.push(Move { piece, start_square: from, end_square: sq, promotion: None });
                        to = offset_square(sq, df, dr);
                    }
                }
            }
        }
    }

    /// Generate pawn moves: pushes, double-pushes, captures, and en passant
    fn gen_pawn_moves(&self, from: Square, piece: Piece, moves: &mut Vec<Move>) {
        // White pawns march up the board (+1 rank), black pawns down (-1)
        let dir: i8 = match piece.color {
            Color::White => 1,
            Color::Black => -1,
        };
        let start_rank: u8 = match piece.color {
            Color::White => 1,
            Color::Black => 6,
        };

        // Single push onto an empty square
        if let Some(one) = offset_square(from, 0, dir) {
            if self.piece_at(one).is_none() {
                push_pawn_move(piece, from, one, moves);

                // Double push from the home rank, if both squares are empty
                if from.rank() == start_rank {
                    if let Some(two) = offset_square(from, 0, dir * 2) {
                        if self.piece_at(two).is_none() {
                            push_pawn_move(piece, from, two, moves);
                        }
                    }
                }
            }
        }

        // Diagonal captures, including en passant
        for df in [-1, 1] {
            if let Some(to) = offset_square(from, df, dir) {
                let is_enemy: bool = match self.piece_at(to) {
                    Some(target) => target.color != piece.color,
                    None => false,
                };
                if is_enemy || Some(to) == self.en_passant {
                    push_pawn_move(piece, from, to, moves);
                }
            }
        }
    }

    /// Generate castling moves when rights, empty squares, and king safety allow
    fn gen_castling_moves(&self, from: Square, piece: Piece, moves: &mut Vec<Move>) {
        let rank: u8 = match piece.color {
            Color::White => 0,
            Color::Black => 7,
        };
        let enemy: Color = piece.color.opposite();

        // The king must be on its home square and not currently in check
        if from != Square::new(4, rank) || self.is_attacked(from, enemy) {
            return;
        }

        let (kingside, queenside): (bool, bool) = match piece.color {
            Color::White => (self.castling.white_kingside, self.castling.white_queenside),
            Color::Black => (self.castling.black_kingside, self.castling.black_queenside),
        };

        // Kingside: f and g empty, and the king never crosses an attacked square
        if kingside
            && self.piece_at(Square::new(5, rank)).is_none()
            && self.piece_at(Square::new(6, rank)).is_none()
            && !self.is_attacked(Square::new(5, rank), enemy)
            && !self.is_attacked(Square::new(6, rank), enemy)
        {
            moves.push(Move { piece, start_square: from, end_square: Square::new(6, rank), promotion: None });
        }

        // Queenside: b, c, d empty, and the king never crosses an attacked square
        if queenside
            && self.piece_at(Square::new(1, rank)).is_none()
            && self.piece_at(Square::new(2, rank)).is_none()
            && self.piece_at(Square::new(3, rank)).is_none()
            && !self.is_attacked(Square::new(3, rank), enemy)
            && !self.is_attacked(Square::new(2, rank), enemy)
        {
            moves.push(Move { piece, start_square: from, end_square: Square::new(2, rank), promotion: None });
        }
    }

    /// Find the square of a given color's king, if one is on the board
    fn find_king(&self, color: Color) -> Option<Square> {
        for rank in 0..8 {
            for file in 0..8 {
                let sq: Square = Square::new(file, rank);
                if self.piece_at(sq) == Some(Piece { color, kind: PieceKind::King }) {
                    return Some(sq);
                }
            }
        }
        None
    }

    /// Whether `sq` is attacked by any piece of color `by`
    fn is_attacked(&self, sq: Square, by: Color) -> bool {
        // Pawns: a `by`-colored pawn attacking `sq` sits one rank toward its own
        // side, so we look back down its capture diagonals
        let pawn_dir: i8 = match by {
            Color::White => 1,
            Color::Black => -1,
        };
        for df in [-1, 1] {
            if let Some(p) = offset_square(sq, df, -pawn_dir) {
                if self.piece_at(p) == Some(Piece { color: by, kind: PieceKind::Pawn }) {
                    return true;
                }
            }
        }

        // Knights
        for &(df, dr) in &KNIGHT_OFFSETS {
            if let Some(p) = offset_square(sq, df, dr) {
                if self.piece_at(p) == Some(Piece { color: by, kind: PieceKind::Knight }) {
                    return true;
                }
            }
        }

        // Enemy king on an adjacent square
        for &(df, dr) in &KING_OFFSETS {
            if let Some(p) = offset_square(sq, df, dr) {
                if self.piece_at(p) == Some(Piece { color: by, kind: PieceKind::King }) {
                    return true;
                }
            }
        }

        // Sliding pieces: bishops/queens on diagonals, rooks/queens on lines
        self.attacked_along(sq, by, &BISHOP_DIRS, PieceKind::Bishop)
            || self.attacked_along(sq, by, &ROOK_DIRS, PieceKind::Rook)
    }

    /// Whether a slider of `kind` (or a queen) of color `by` attacks `sq` along `dirs`
    fn attacked_along(&self, sq: Square, by: Color, dirs: &[(i8, i8)], kind: PieceKind) -> bool {
        for &(df, dr) in dirs {
            let mut to: Option<Square> = offset_square(sq, df, dr);
            while let Some(p) = to {
                match self.piece_at(p) {
                    Some(piece) => {
                        if piece.color == by
                            && (piece.kind == kind || piece.kind == PieceKind::Queen)
                        {
                            return true;
                        }
                        break;
                    }
                    None => to = offset_square(p, df, dr),
                }
            }
        }
        false
    }
}

// Helpers
/// Offset a square by (file, rank), returning None if it leaves the board
fn offset_square(sq: Square, df: i8, dr: i8) -> Option<Square> {
    let file: i8 = sq.file() as i8 + df;
    let rank: i8 = sq.rank() as i8 + dr;
    if (0..8).contains(&file) && (0..8).contains(&rank) {
        Some(Square::new(file as u8, rank as u8))
    } else {
        None
    }
}

/// Push a pawn move, expanding it into all four promotions on the back rank
fn push_pawn_move(piece: Piece, from: Square, to: Square, moves: &mut Vec<Move>) {
    if to.rank() == 0 || to.rank() == 7 {
        for kind in [PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop, PieceKind::Knight] {
            moves.push(Move { piece, start_square: from, end_square: to, promotion: Some(kind) });
        }
    } else {
        moves.push(Move { piece, start_square: from, end_square: to, promotion: None });
    }
}

/// Knight jumps as (file, rank) offsets
const KNIGHT_OFFSETS: [(i8, i8); 8] = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
];

/// King steps as (file, rank) offsets
const KING_OFFSETS: [(i8, i8); 8] = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
];

/// Bishop directions as (file, rank) offsets
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Rook directions as (file, rank) offsets
const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Queen directions as (file, rank) offsets
const QUEEN_DIRS: [(i8, i8); 8] = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Count the leaf nodes of the move tree to a given depth
    fn perft(board: &Board, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mut nodes: u64 = 0;
        for mv in board.get_legal_moves() {
            let mut next: Board = board.clone();
            next.make_move(mv);
            nodes += perft(&next, depth - 1);
        }
        nodes
    }

    #[test]
    fn starting_position_move_counts() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(start).expect("start FEN should parse");

        // Well-known perft numbers for the initial position
        assert_eq!(perft(&board, 1), 20);
        assert_eq!(perft(&board, 2), 400);
        assert_eq!(perft(&board, 3), 8902);
    }

    #[test]
    fn make_move_pushes_pawn_and_flips_side() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let mut board = Board::from_fen(start).expect("start FEN should parse");

        let pawn = Piece { color: Color::White, kind: PieceKind::Pawn };
        let e2 = Square::new(4, 1);
        let e4 = Square::new(4, 3);
        board.make_move(Move { piece: pawn, start_square: e2, end_square: e4, promotion: None });

        assert_eq!(board.piece_at(e2), None);
        assert_eq!(board.piece_at(e4), Some(pawn));
        assert_eq!(board.side_to_move, Color::Black);
        // A double push leaves an en-passant target on e3
        assert_eq!(board.en_passant, Some(Square::new(4, 2)));
    }

    #[test]
    fn king_must_escape_check() {
        // Black king on h8 is checked by a white rook on h1; it can only step to
        // g7 or g8 (h7 stays on the attacked file)
        let board = Board::from_fen("7k/8/8/8/8/8/8/4K2R b K - 0 1")
            .expect("FEN should parse");

        let moves = board.get_legal_moves();
        assert_eq!(moves.len(), 2);
    }

    #[test]
    fn pawn_promotes_to_all_four_pieces() {
        // White pawn on a7 can promote on a8; that single push must expand into
        // queen, rook, bishop, and knight
        let board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
            .expect("FEN should parse");

        let promotions = board
            .get_legal_moves()
            .iter()
            .filter(|m| m.promotion.is_some())
            .count();
        assert_eq!(promotions, 4);
    }

    #[test]
    fn under_promotion_to_knight() {
        let mut board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
            .expect("FEN should parse");

        let pawn = Piece { color: Color::White, kind: PieceKind::Pawn };
        let a7 = Square::new(0, 6);
        let a8 = Square::new(0, 7);
        board.make_move(Move {
            piece: pawn,
            start_square: a7,
            end_square: a8,
            promotion: Some(PieceKind::Knight),
        });

        assert_eq!(board.piece_at(a8), Some(Piece { color: Color::White, kind: PieceKind::Knight }));
    }

    #[test]
    fn detects_checkmate() {
        // Fool's mate: White is mated after 1. f3 e5 2. g4 Qh4#
        let board = Board::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
            .expect("FEN should parse");

        assert_eq!(board.status(), Status::Checkmate);
    }

    #[test]
    fn detects_stalemate() {
        // Black king on h8 has no legal move but is not in check
        let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
            .expect("FEN should parse");

        assert_eq!(board.status(), Status::Stalemate);
    }
}
