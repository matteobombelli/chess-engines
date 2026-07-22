/// One of the two players
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

impl Color {
    /// Returns the other side
    pub fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}


/// The kind of a piece, ignoring its color
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

/// A piece on the board: color + kind
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

/// A square, indexed 0..=63 as a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Square(pub u8);

impl Square {
    /// Build a square from file and rank
    pub fn new(file: u8, rank: u8) -> Square {
        Square(rank * 8 + file)
    }

    /// The file
    pub fn file(self) -> u8 {
        self.0 % 8
    }

    /// The rank
    pub fn rank(self) -> u8 {
        self.0 / 8
    }
    
    /// The 0..=63 index
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Which castles each side may still legally make
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CastlingRights {
    pub white_kingside: bool,
    pub white_queenside: bool,
    pub black_kingside: bool,
    pub black_queenside: bool,
}

/// A full chess position
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Board {
    /// All 64 squares
    pub squares: [Option<Piece>; 64],
    pub side_to_move: Color,
    pub castling: CastlingRights,
    
    /// The square a pawn may be captured on by en passant
    pub en_passant: Option<Square>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,

    /// The moves played on this board so far, in Standard Algebraic Notation
    pub san_history: Vec<String>,

    /// Canonical positions reached during this game, including the initial one.
    /// Move clocks are excluded because they do not affect repetition.
    pub(crate) position_history: Vec<String>,
}

impl Board {
    /// An empty board. White to move, no castling rights, no pieces
    pub fn empty() -> Board {
        let mut board = Board {
            squares: [None; 64],
            side_to_move: Color::White,
            castling: CastlingRights {
                white_kingside: false,
                white_queenside: false,
                black_kingside: false,
                black_queenside: false,
            },
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            san_history: Vec::new(),
            position_history: Vec::new(),
        };
        board.reset_position_history();
        board
    }

    /// The piece on sq, or None if the square is empty
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.squares[sq.index()]
    }

    /// Place a piece on sq, or clear it by passing None
    pub fn set_piece(&mut self, sq: Square, piece: Option<Piece>) {
        self.squares[sq.index()] = piece;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opposite_color() {
        assert_eq!(Color::White.opposite(), Color::Black);
        assert_eq!(Color::Black.opposite(), Color::White);
    }

    #[test]
    fn place_and_read() {
        let mut board = Board::empty();
        let e4 = Square::new(4, 3);
        let pawn = Piece { color: Color::White, kind: PieceKind::Pawn };
        
        board.set_piece(e4, Some(pawn));

        assert_eq!(board.piece_at(e4), Some(pawn));
        assert_eq!(board.piece_at(Square::new(0, 0)), None);
    }
}
