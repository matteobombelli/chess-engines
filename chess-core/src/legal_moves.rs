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
    InsufficientMaterial,
    ThreefoldRepetition,
    FiftyMoveRule,
}

impl Board {
    /// Apply a move to the Board, recording it in `san_history` as SAN
    pub fn make_move(&mut self, mv: Move) {
        let legal_moves: Vec<Move> = self.get_legal_moves();
        self.make_move_with_legal_moves(mv, &legal_moves);
    }

    /// [`make_move`](Self::make_move) for a caller that already holds the legal
    /// moves of the pre-move position, which are what SAN disambiguation reads.
    pub(crate) fn make_move_with_legal_moves(&mut self, mv: Move, legal_moves: &[Move]) {
        // Compute the move text against the pre-move position (for capture,
        // disambiguation), apply the move, then add the check/mate suffix from
        // the resulting position where `side_to_move` is now the opponent
        let body: String = self.san_body_with_legal_moves(mv, legal_moves);
        self.apply_move(mv);
        self.record_current_position();
        let suffix: &str = match self.status() {
            Status::Checkmate => "#",
            _ if self.is_in_check() => "+",
            _ => "",
        };
        self.san_history.push(format!("{body}{suffix}"));
    }

    /// Apply a legal move without recording SAN. Used for search positions.
    pub fn make_search_move(&mut self, mv: Move) {
        self.apply_move(mv);
        self.record_current_position();
    }

    /// Apply a move to the Board without recording it
    pub(crate) fn apply_move(&mut self, mv: Move) {
        let color: Color = mv.piece.color;
        let from: Square = mv.start_square;
        let to: Square = mv.end_square;

        // Resolve en passant before overwriting the target below. Move
        // generation and both repetition-key implementations use this same
        // structural predicate, so malformed public Board state cannot make
        // their interpretations drift apart.
        let en_passant_capture = self.en_passant_capture_square(mv.piece, from, to);
        let is_capture: bool = self.piece_at(to).is_some() || en_passant_capture.is_some();

        // Lift the moving piece off its start square
        self.set_piece(from, None);

        match mv.piece.kind {
            PieceKind::Pawn => {
                // En-passant capture: a diagonal step onto the en-passant target
                // takes the pawn that just double-pushed, sitting behind `to`
                if let Some(captured_pawn) = en_passant_capture {
                    self.set_piece(captured_pawn, None);
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
                let rook: Piece = Piece {
                    color,
                    kind: PieceKind::Rook,
                };
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
            if sq == Square::new(0, 0) {
                self.castling.white_queenside = false;
            }
            if sq == Square::new(7, 0) {
                self.castling.white_kingside = false;
            }
            if sq == Square::new(0, 7) {
                self.castling.black_queenside = false;
            }
            if sq == Square::new(7, 7) {
                self.castling.black_kingside = false;
            }
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

    /// Return the en-passant target and captured-pawn square when the raw Board
    /// state describes a structurally valid right for the side to move.
    ///
    /// This deliberately does not test whether an adjacent friendly pawn exists
    /// or whether moving it would expose its king. Those move-specific checks
    /// are layered on top by `en_passant_capture_square` and
    /// `effective_en_passant_target` respectively.
    fn structurally_valid_en_passant(&self) -> Option<(Square, Square)> {
        let target = self.en_passant?;
        let expected_rank = match self.side_to_move {
            Color::White => 5,
            Color::Black => 2,
        };
        if target.rank() != expected_rank || self.piece_at(target).is_some() {
            return None;
        }

        let captured_pawn_rank = match self.side_to_move {
            Color::White => target.rank().checked_sub(1)?,
            Color::Black => target.rank().checked_add(1).filter(|rank| *rank < 8)?,
        };
        let captured_pawn = Square::new(target.file(), captured_pawn_rank);
        if self.piece_at(captured_pawn)
            != Some(Piece {
                color: self.side_to_move.opposite(),
                kind: PieceKind::Pawn,
            })
        {
            return None;
        }

        Some((target, captured_pawn))
    }

    /// Return the captured-pawn square when `(piece, from, to)` is a
    /// structurally valid en-passant capture in this position.
    pub(crate) fn en_passant_capture_square(
        &self,
        piece: Piece,
        from: Square,
        to: Square,
    ) -> Option<Square> {
        if piece
            != (Piece {
                color: self.side_to_move,
                kind: PieceKind::Pawn,
            })
        {
            return None;
        }
        let (target, captured_pawn) = self.structurally_valid_en_passant()?;
        (to == target
            && from.rank() == captured_pawn.rank()
            && from.file().abs_diff(target.file()) == 1)
            .then_some(captured_pawn)
    }

    /// Return the en-passant target only when at least one legal capture uses
    /// it. This is the shared FIDE repetition-identity interpretation used by
    /// both Board's string key and SearchPosition's Zobrist key.
    pub(crate) fn effective_en_passant_target(&self) -> Option<Square> {
        let (target, captured_pawn) = self.structurally_valid_en_passant()?;
        let moving_side = self.side_to_move;
        let pawn = Piece {
            color: moving_side,
            kind: PieceKind::Pawn,
        };

        for file_delta in [-1i8, 1] {
            let from_file = target.file() as i8 + file_delta;
            if !(0..8).contains(&from_file) {
                continue;
            }
            let from = Square::new(from_file as u8, captured_pawn.rank());
            if self.piece_at(from) != Some(pawn)
                || self.en_passant_capture_square(pawn, from, target) != Some(captured_pawn)
            {
                continue;
            }

            let mut next = self.without_history();
            next.apply_move(Move {
                piece: pawn,
                start_square: from,
                end_square: target,
                promotion: None,
            });
            if next
                .find_king(moving_side)
                .is_some_and(|king| !next.is_attacked(king, moving_side.opposite()))
            {
                return Some(target);
            }
        }

        None
    }

    /// Get all legal moves from the current board position
    pub fn get_legal_moves(&self) -> Vec<Move> {
        let me: Color = self.side_to_move;
        let mut legal_moves: Vec<Move> = Vec::new();

        // The check test below reads only piece placement, and `apply_move`
        // never touches the histories, so probe on a history-free copy. Cloning
        // the SAN and repetition strings once per candidate would make one call
        // cost O(plies played) for state nothing here looks at.
        let probe: Board = self.without_history();

        // A pseudo-legal move is legal only if it doesn't leave our king in check
        for mv in self.pseudo_legal_moves() {
            let mut next: Board = probe.clone();
            next.apply_move(mv);
            if let Some(king) = next.find_king(me) {
                if !next.is_attacked(king, me.opposite()) {
                    legal_moves.push(mv);
                }
            }
        }

        legal_moves
    }

    /// A copy of the chess state alone, with the SAN and repetition histories
    /// dropped rather than cloned. Only valid for questions that do not consult
    /// history, such as whether a move leaves the mover's king in check.
    fn without_history(&self) -> Board {
        Board {
            squares: self.squares,
            side_to_move: self.side_to_move,
            castling: self.castling,
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
            san_history: Vec::new(),
            position_history: Vec::new(),
        }
    }

    /// Whether the side to move is currently in check
    pub fn is_in_check(&self) -> bool {
        match self.find_king(self.side_to_move) {
            Some(king) => self.is_attacked(king, self.side_to_move.opposite()),
            None => false,
        }
    }

    /// Classify the position for the side to move. Checkmate and stalemate take
    /// precedence over automatic draw-rule handling.
    pub fn status(&self) -> Status {
        if self.get_legal_moves().is_empty() {
            return if self.is_in_check() {
                Status::Checkmate
            } else {
                Status::Stalemate
            };
        }

        if self.has_insufficient_material() {
            return Status::InsufficientMaterial;
        }

        if self.halfmove_clock >= 100 {
            return Status::FiftyMoveRule;
        }

        if self.current_position_repetition_count() >= 3 {
            return Status::ThreefoldRepetition;
        }

        Status::Ongoing
    }

    /// Conservative FIDE dead-position cases that depend only on material:
    /// bare kings, a single bishop/knight, or bishops confined to one color.
    /// Positions with pawns, rooks, queens, or multiple knight colors remain
    /// playable because some legal continuation can still end in checkmate.
    pub fn has_insufficient_material(&self) -> bool {
        let mut minor_count = 0;
        let mut knight_count = 0;
        let mut bishop_square_color = None;
        for (index, piece) in self.squares.iter().enumerate() {
            let Some(piece) = piece else {
                continue;
            };
            match piece.kind {
                PieceKind::King => {}
                PieceKind::Pawn | PieceKind::Rook | PieceKind::Queen => return false,
                PieceKind::Knight => {
                    minor_count += 1;
                    knight_count += 1;
                }
                PieceKind::Bishop => {
                    minor_count += 1;
                    let square = Square(index as u8);
                    let color = (square.file() + square.rank()) % 2;
                    if bishop_square_color.is_some_and(|existing| existing != color) {
                        bishop_square_color = Some(2);
                    } else if bishop_square_color.is_none() {
                        bishop_square_color = Some(color);
                    }
                }
            }
        }
        minor_count == 0
            || minor_count == 1
            || (knight_count == 0 && bishop_square_color.is_some_and(|color| color < 2))
    }

    pub(crate) fn reset_position_history(&mut self) {
        self.position_history.clear();
        self.record_current_position();
    }

    fn record_current_position(&mut self) {
        self.position_history.push(self.current_position_key());
    }

    /// Number of occurrences of the current repetition position, including now.
    pub fn current_position_repetition_count(&self) -> usize {
        let Some(current) = self.position_history.last() else {
            return 0;
        };
        self.position_history
            .iter()
            .filter(|position| *position == current)
            .count()
    }

    /// Number of earlier occurrences of the current repetition position.
    pub fn prior_repetition_count(&self) -> usize {
        self.current_position_repetition_count().saturating_sub(1)
    }

    /// Piece placement, side to move, castling rights, and any en-passant right
    /// that changes the legal moves are the only inputs to repetition identity.
    fn current_position_key(&self) -> String {
        let fen = self.to_fen();
        let fields: Vec<&str> = fen.split_whitespace().collect();
        let en_passant = if self.effective_en_passant_target().is_some() {
            fields[3]
        } else {
            "-"
        };
        format!("{} {} {} {}", fields[0], fields[1], fields[2], en_passant)
    }

    /// All pseudo-legal moves for the side to move, ignoring whether they leave
    /// our own king in check
    fn pseudo_legal_moves(&self) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::new();
        self.pseudo_legal_moves_into(&mut moves);
        moves
    }

    /// Fill a reusable buffer with pseudo-legal moves.
    pub(crate) fn pseudo_legal_moves_into(&self, moves: &mut Vec<Move>) {
        moves.clear();
        for rank in 0..8 {
            for file in 0..8 {
                let from: Square = Square::new(file, rank);
                let piece: Piece = match self.piece_at(from) {
                    Some(p) if p.color == self.side_to_move => p,
                    _ => continue,
                };
                match piece.kind {
                    PieceKind::Pawn => self.gen_pawn_moves(from, piece, moves),
                    PieceKind::Knight => self.gen_step_moves(from, piece, &KNIGHT_OFFSETS, moves),
                    PieceKind::Bishop => self.gen_slide_moves(from, piece, &BISHOP_DIRS, moves),
                    PieceKind::Rook => self.gen_slide_moves(from, piece, &ROOK_DIRS, moves),
                    PieceKind::Queen => self.gen_slide_moves(from, piece, &QUEEN_DIRS, moves),
                    PieceKind::King => {
                        self.gen_step_moves(from, piece, &KING_OFFSETS, moves);
                        self.gen_castling_moves(from, piece, moves);
                    }
                }
            }
        }
    }

    /// Generate single-step moves (knight, king) from a list of offsets
    fn gen_step_moves(
        &self,
        from: Square,
        piece: Piece,
        offsets: &[(i8, i8)],
        moves: &mut Vec<Move>,
    ) {
        for &(df, dr) in offsets {
            if let Some(to) = offset_square(from, df, dr) {
                // We may land on an empty square or capture an enemy piece
                match self.piece_at(to) {
                    Some(target) if target.color == piece.color => continue,
                    _ => moves.push(Move {
                        piece,
                        start_square: from,
                        end_square: to,
                        promotion: None,
                    }),
                }
            }
        }
    }

    /// Generate sliding moves (bishop, rook, queen) along a list of directions
    fn gen_slide_moves(
        &self,
        from: Square,
        piece: Piece,
        dirs: &[(i8, i8)],
        moves: &mut Vec<Move>,
    ) {
        for &(df, dr) in dirs {
            let mut to: Option<Square> = offset_square(from, df, dr);
            while let Some(sq) = to {
                match self.piece_at(sq) {
                    Some(target) => {
                        // Stop at the first piece, capturing it if it's an enemy
                        if target.color != piece.color {
                            moves.push(Move {
                                piece,
                                start_square: from,
                                end_square: sq,
                                promotion: None,
                            });
                        }
                        break;
                    }
                    None => {
                        moves.push(Move {
                            piece,
                            start_square: from,
                            end_square: sq,
                            promotion: None,
                        });
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
                let is_en_passant = self.en_passant_capture_square(piece, from, to).is_some();
                if is_enemy || is_en_passant {
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
            && self.piece_at(Square::new(7, rank))
                == Some(Piece {
                    color: piece.color,
                    kind: PieceKind::Rook,
                })
            && self.piece_at(Square::new(5, rank)).is_none()
            && self.piece_at(Square::new(6, rank)).is_none()
            && !self.is_attacked(Square::new(5, rank), enemy)
            && !self.is_attacked(Square::new(6, rank), enemy)
        {
            moves.push(Move {
                piece,
                start_square: from,
                end_square: Square::new(6, rank),
                promotion: None,
            });
        }

        // Queenside: b, c, d empty, and the king never crosses an attacked square
        if queenside
            && self.piece_at(Square::new(0, rank))
                == Some(Piece {
                    color: piece.color,
                    kind: PieceKind::Rook,
                })
            && self.piece_at(Square::new(1, rank)).is_none()
            && self.piece_at(Square::new(2, rank)).is_none()
            && self.piece_at(Square::new(3, rank)).is_none()
            && !self.is_attacked(Square::new(3, rank), enemy)
            && !self.is_attacked(Square::new(2, rank), enemy)
        {
            moves.push(Move {
                piece,
                start_square: from,
                end_square: Square::new(2, rank),
                promotion: None,
            });
        }
    }

    /// Find the square of a given color's king, if one is on the board
    pub(crate) fn find_king(&self, color: Color) -> Option<Square> {
        for rank in 0..8 {
            for file in 0..8 {
                let sq: Square = Square::new(file, rank);
                if self.piece_at(sq)
                    == Some(Piece {
                        color,
                        kind: PieceKind::King,
                    })
                {
                    return Some(sq);
                }
            }
        }
        None
    }

    /// Whether `sq` is attacked by any piece of color `by`
    pub(crate) fn is_attacked(&self, sq: Square, by: Color) -> bool {
        // Pawns: a `by`-colored pawn attacking `sq` sits one rank toward its own
        // side, so we look back down its capture diagonals
        let pawn_dir: i8 = match by {
            Color::White => 1,
            Color::Black => -1,
        };
        for df in [-1, 1] {
            if let Some(p) = offset_square(sq, df, -pawn_dir) {
                if self.piece_at(p)
                    == Some(Piece {
                        color: by,
                        kind: PieceKind::Pawn,
                    })
                {
                    return true;
                }
            }
        }

        // Knights
        for &(df, dr) in &KNIGHT_OFFSETS {
            if let Some(p) = offset_square(sq, df, dr) {
                if self.piece_at(p)
                    == Some(Piece {
                        color: by,
                        kind: PieceKind::Knight,
                    })
                {
                    return true;
                }
            }
        }

        // Enemy king on an adjacent square
        for &(df, dr) in &KING_OFFSETS {
            if let Some(p) = offset_square(sq, df, dr) {
                if self.piece_at(p)
                    == Some(Piece {
                        color: by,
                        kind: PieceKind::King,
                    })
                {
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
        for kind in [
            PieceKind::Queen,
            PieceKind::Rook,
            PieceKind::Bishop,
            PieceKind::Knight,
        ] {
            moves.push(Move {
                piece,
                start_square: from,
                end_square: to,
                promotion: Some(kind),
            });
        }
    } else {
        moves.push(Move {
            piece,
            start_square: from,
            end_square: to,
            promotion: None,
        });
    }
}

/// Knight jumps as (file, rank) offsets
const KNIGHT_OFFSETS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

/// King steps as (file, rank) offsets
const KING_OFFSETS: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// Bishop directions as (file, rank) offsets
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Rook directions as (file, rank) offsets
const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Queen directions as (file, rank) offsets
const QUEEN_DIRS: [(i8, i8); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
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
            next.apply_move(mv);
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

        let pawn = Piece {
            color: Color::White,
            kind: PieceKind::Pawn,
        };
        let e2 = Square::new(4, 1);
        let e4 = Square::new(4, 3);
        board.make_move(Move {
            piece: pawn,
            start_square: e2,
            end_square: e4,
            promotion: None,
        });

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
        let board = Board::from_fen("7k/8/8/8/8/8/8/4K2R b K - 0 1").expect("FEN should parse");

        let moves = board.get_legal_moves();
        assert_eq!(moves.len(), 2);
    }

    #[test]
    fn pawn_promotes_to_all_four_pieces() {
        // White pawn on a7 can promote on a8; that single push must expand into
        // queen, rook, bishop, and knight
        let board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").expect("FEN should parse");

        let promotions = board
            .get_legal_moves()
            .iter()
            .filter(|m| m.promotion.is_some())
            .count();
        assert_eq!(promotions, 4);
    }

    #[test]
    fn under_promotion_to_knight() {
        let mut board =
            Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").expect("FEN should parse");

        let pawn = Piece {
            color: Color::White,
            kind: PieceKind::Pawn,
        };
        let a7 = Square::new(0, 6);
        let a8 = Square::new(0, 7);
        board.make_move(Move {
            piece: pawn,
            start_square: a7,
            end_square: a8,
            promotion: Some(PieceKind::Knight),
        });

        assert_eq!(
            board.piece_at(a8),
            Some(Piece {
                color: Color::White,
                kind: PieceKind::Knight
            })
        );
    }

    #[test]
    fn detects_checkmate() {
        // Fool's mate: White is mated after 1. f3 e5 2. g4 Qh4#
        let board =
            Board::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
                .expect("FEN should parse");

        assert_eq!(board.status(), Status::Checkmate);
    }

    #[test]
    fn detects_stalemate() {
        // Black king on h8 has no legal move but is not in check
        let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").expect("FEN should parse");

        assert_eq!(board.status(), Status::Stalemate);
    }

    #[test]
    fn detects_threefold_repetition() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let mut board = Board::from_fen(start).expect("start FEN should parse");

        for san in ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"] {
            board
                .san_to_move(san)
                .expect("repetition move should be legal");
        }

        assert_eq!(board.status(), Status::ThreefoldRepetition);
        assert_eq!(board.current_position_repetition_count(), 3);
        assert_eq!(board.prior_repetition_count(), 2);
    }

    #[test]
    fn detects_fifty_move_rule_at_one_hundred_halfmoves() {
        let mut board =
            Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 99 50").expect("FEN should parse");

        assert_eq!(board.status(), Status::Ongoing);
        board.san_to_move("Rb1").expect("rook move should be legal");
        assert_eq!(board.halfmove_clock, 100);
        assert_eq!(board.status(), Status::FiftyMoveRule);
    }

    #[test]
    fn detects_only_conservative_insufficient_material_positions() {
        for fen in [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2N1K3 w - - 0 1",
            "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            assert!(board.has_insufficient_material(), "{fen}");
            assert_eq!(board.status(), Status::InsufficientMaterial, "{fen}");
        }

        for fen in [
            "2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/1NB1K3 w - - 0 1",
            "4k3/8/8/8/8/8/P7/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/1NN1K3 w - - 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            assert!(!board.has_insufficient_material(), "{fen}");
            assert_eq!(board.status(), Status::Ongoing, "{fen}");
        }
    }

    #[test]
    fn checkmate_takes_precedence_over_fifty_move_rule() {
        let board =
            Board::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 100 51")
                .expect("FEN should parse");

        assert_eq!(board.status(), Status::Checkmate);
    }

    #[test]
    fn search_move_skips_san() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(start).expect("start FEN should parse");
        let mv = board.move_from_uci("e2e4").expect("e2e4 should be legal");

        let mut regular = board.clone();
        regular.make_move(mv);
        let mut searched = board;
        searched.make_search_move(mv);

        assert_eq!(searched.to_fen(), regular.to_fen());
        assert!(searched.san_history.is_empty());
        assert_eq!(regular.san_history, ["e4"]);
        assert_eq!(searched.position_history, regular.position_history);
    }

    #[test]
    fn repetition_key_ignores_unusable_en_passant_target() {
        let without_target =
            Board::from_fen("4k3/8/8/3p4/8/8/8/4K3 w - - 0 1").expect("FEN should parse");
        let unusable_target =
            Board::from_fen("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1").expect("FEN should parse");
        let capturable_target =
            Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").expect("FEN should parse");
        let same_without_target =
            Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1").expect("FEN should parse");
        let occupied_target = Board::from_fen("4k3/8/3n4/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let occupied_without_target =
            Board::from_fen("4k3/8/3n4/3pP3/8/8/8/4K3 w - - 0 1").unwrap();

        assert_eq!(
            without_target.current_position_key(),
            unusable_target.current_position_key()
        );
        assert_ne!(
            capturable_target.current_position_key(),
            same_without_target.current_position_key()
        );
        assert_eq!(
            occupied_target.current_position_key(),
            occupied_without_target.current_position_key()
        );
    }

    #[test]
    fn stale_castling_rights_cannot_materialize_a_rook() {
        let no_rooks = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w KQ - 0 1").unwrap();
        let legal = no_rooks.legal_uci_moves();
        assert!(!legal.contains(&"e1g1".to_string()));
        assert!(!legal.contains(&"e1c1".to_string()));

        let enemy_rooks = Board::from_fen("4k3/8/8/8/8/8/8/r3K2r w KQ - 0 1").unwrap();
        let legal = enemy_rooks.legal_uci_moves();
        assert!(!legal.contains(&"e1g1".to_string()));
        assert!(!legal.contains(&"e1c1".to_string()));
    }

    #[test]
    fn malformed_en_passant_target_cannot_capture_a_missing_pawn() {
        let missing_pawn = Board::from_fen("4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1").unwrap();
        assert!(!missing_pawn.legal_uci_moves().contains(&"e5d6".to_string()));

        let occupied_target = Board::from_fen("4k3/8/3n4/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        // This is a normal capture of the knight on d6. Applying it must not
        // also remove the pawn on d5 as though it were en passant.
        let mut after = occupied_target.clone();
        after.uci_to_move("e5d6").unwrap();
        assert_eq!(
            after.piece_at(Square::new(3, 4)),
            Some(Piece {
                color: Color::Black,
                kind: PieceKind::Pawn,
            })
        );

        // Public Board state can be assembled without the FEN parser, so move
        // generation also rejects a target on an impossible rank.
        let mut wrong_rank = Board::from_fen("4k3/8/8/8/8/3Pp3/8/4K3 w - - 0 1").unwrap();
        wrong_rank.en_passant = Some(Square::new(4, 3));
        assert!(!wrong_rank.legal_uci_moves().contains(&"d3e4".to_string()));
    }

    #[test]
    fn board_and_search_share_en_passant_legality_and_identity() {
        struct Case {
            name: &'static str,
            board: Board,
            structurally_valid: bool,
            legal_capture: Option<&'static str>,
        }

        let mut wrong_rank = Board::from_fen("4k3/8/8/8/8/3Pp3/8/4K3 w - - 0 1").unwrap();
        wrong_rank.en_passant = Some(Square::new(4, 3));
        wrong_rank.reset_position_history();

        let cases = [
            Case {
                name: "valid white capture",
                board: Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap(),
                structurally_valid: true,
                legal_capture: Some("e5d6"),
            },
            Case {
                name: "valid black capture",
                board: Board::from_fen("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1").unwrap(),
                structurally_valid: true,
                legal_capture: Some("e4d3"),
            },
            Case {
                name: "missing captured pawn",
                board: Board::from_fen("4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1").unwrap(),
                structurally_valid: false,
                legal_capture: None,
            },
            Case {
                name: "occupied target",
                board: Board::from_fen("4k3/8/3n4/3pP3/8/8/8/4K3 w - d6 0 1").unwrap(),
                structurally_valid: false,
                legal_capture: None,
            },
            Case {
                name: "wrong target rank",
                board: wrong_rank,
                structurally_valid: false,
                legal_capture: None,
            },
            Case {
                name: "wrong captured piece kind",
                board: Board::from_fen("4k3/8/8/3nP3/8/8/8/4K3 w - d6 0 1").unwrap(),
                structurally_valid: false,
                legal_capture: None,
            },
            Case {
                name: "capturing pawn is pinned",
                board: Board::from_fen("4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap(),
                structurally_valid: true,
                legal_capture: None,
            },
            Case {
                name: "no adjacent capturing pawn",
                board: Board::from_fen("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1").unwrap(),
                structurally_valid: true,
                legal_capture: None,
            },
        ];

        for Case {
            name,
            board,
            structurally_valid,
            legal_capture,
        } in cases
        {
            assert_eq!(
                board.structurally_valid_en_passant().is_some(),
                structurally_valid,
                "{name}: structural validity"
            );

            let board_moves = board.get_legal_moves();
            let board_en_passant_moves: Vec<Move> = board_moves
                .iter()
                .copied()
                .filter(|mv| {
                    board
                        .en_passant_capture_square(mv.piece, mv.start_square, mv.end_square)
                        .is_some()
                })
                .collect();
            assert_eq!(
                board_en_passant_moves.len(),
                usize::from(legal_capture.is_some()),
                "{name}: legal en-passant count"
            );
            if let Some(uci) = legal_capture {
                assert_eq!(
                    board.move_from_uci(uci),
                    Ok(*board_en_passant_moves
                        .first()
                        .expect("expected en-passant move must exist")),
                    "{name}: expected legal en-passant move"
                );
            }

            let mut search = crate::SearchPosition::from_board(&board);
            assert_eq!(search.legal_moves(), board_moves, "{name}: legal moves");
            assert_eq!(search.status(), board.status(), "{name}: status");

            let mut without_target = board.clone();
            without_target.en_passant = None;
            without_target.reset_position_history();
            let board_identity_uses_target =
                board.current_position_key() != without_target.current_position_key();
            let search_identity_uses_target = search.position_key()
                != crate::SearchPosition::from_board(&without_target).position_key();
            assert_eq!(
                board_identity_uses_target, search_identity_uses_target,
                "{name}: Board/SearchPosition identity parity"
            );
            assert_eq!(
                board_identity_uses_target,
                legal_capture.is_some(),
                "{name}: only a legal capture affects repetition identity"
            );
            assert_eq!(
                board.effective_en_passant_target().is_some(),
                legal_capture.is_some(),
                "{name}: effective target"
            );

            if let Some(mv) = board_en_passant_moves.first().copied() {
                let mut board_after = board.clone();
                board_after.make_search_move(mv);
                let undo = search.make_move(mv);
                assert_eq!(search.squares(), &board_after.squares, "{name}: pieces");
                assert_eq!(
                    search.position_key(),
                    crate::SearchPosition::from_board(&board_after).position_key(),
                    "{name}: post-capture identity"
                );
                search.unmake_move(undo);
                assert_eq!(
                    search,
                    crate::SearchPosition::from_board(&board),
                    "{name}: reversible search state"
                );
            }
        }
    }
}
