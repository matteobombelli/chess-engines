use crate::board::*;

impl Board {
    /// Parse a position from Forsyth-Edwards Notation
    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(format!("expected 6 FEN fields, got {}", fields.len()));
        }

        let mut board = Board::empty();

        // Field 1: Piece placement
        let ranks: Vec<&str> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(format!("expected 8 ranks, got {}", ranks.len()));
        }
        
        // Iterate over ranks 1...8 represented as 0...7 in Board
        for (row, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - row as u8;
            let mut file: u8 = 0;
            for ch in rank_str.chars() {
                if let Some(digit) = ch.to_digit(10) {
                    file += digit as u8;
                } else {
                    let piece = piece_from_char(ch)?;
                    board.set_piece(Square::new(file, rank), Some(piece));
                    file += 1
                }
            }
        }

        // Field 2: Color to move
        board.side_to_move = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(format!("expected 'w' or 'b', got {}", fields[1])),
        };

        // Field 3: Castling rights
        board.castling = CastlingRights {
            white_kingside: fields[2].contains('K'),
            white_queenside: fields[2].contains('Q'),
            black_kingside: fields[2].contains('k'),
            black_queenside: fields[2].contains('q'),
        };

        // Field 4: En-passant targets
        board.en_passant = match fields[3] {
            "-" => None,
            _ => Some(square_from_str(fields[3])?),
        };

        // Field 5: Half-move clock (Moves since the last pawn advance or capture)
        board.halfmove_clock = fields[4]
            .parse()
            .map_err(|_| "bad halfmove clock".to_string())?;

        // Field 6: Fullmove Number
        board.fullmove_number = fields[5]
            .parse()
            .map_err(|_| "bad fullmove number".to_string())?;

        Ok(board)
    }

    /// Parse a position to Forsyth-Edwards Notation
    pub fn to_fen(&self) -> String {
        // Field 1: Piece Placement
        let mut ranks: Vec<String> = Vec::new();
        for rank in (0..8).rev() {
            let mut row = String::new();
            let mut empty = 0;
            for file in 0..8 {
                match self.piece_at(Square::new(file, rank)) {
                    Some(p) => {
                        if empty > 0 {
                            row.push_str(&empty.to_string());
                            empty = 0
                        }
                        row.push(char_from_piece(p));
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                row.push_str(&empty.to_string());
            }
            ranks.push(row);
        }
        let placement: String = ranks.join("/");

        // Field 2: Color to move
        let side: &str = match self.side_to_move {
            Color::White => "w",
            Color::Black => "b",
        };
        
        // Field 3: Castling rights
        let mut castling_rights: Vec<&str> = Vec::new();
        if self.castling.white_kingside { castling_rights.push("K"); }
        if self.castling.white_queenside { castling_rights.push("Q"); }
        if self.castling.black_kingside { castling_rights.push("k"); }
        if self.castling.black_queenside { castling_rights.push("q"); }
        let castling: String = if castling_rights.is_empty() {
            "-".to_string()
        } else {
            castling_rights.join("")
        };

        // Field 4: En-passant targets
        let en_passant: String = if let Some(sq) = self.en_passant {
            str_from_square(sq)
        } else {
            "-".to_string()
        };

        // Field 5: Half-move clock
        let halfmove_clock: String = self.halfmove_clock.to_string();

        // Field 6: Full move number
        let fullmove_number: String = self.fullmove_number.to_string();

        // Construct FEN string
        format!("{} {} {} {} {} {}", placement, side, castling, en_passant, halfmove_clock, fullmove_number)
    }
}

// Helpers
/// Turn a FEN piece letter into a piece
fn piece_from_char(ch: char) -> Result<Piece, String> {
    let color: Color = if ch.is_ascii_uppercase() {
        Color::White  
    } else {
        Color::Black
    };

    let kind = match ch.to_ascii_lowercase() {
        'p' => PieceKind::Pawn,
        'n' => PieceKind::Knight,
        'b' => PieceKind::Bishop,
        'r' => PieceKind::Rook,
        'q' => PieceKind::Queen,
        'k' => PieceKind::King,
        _ => return Err(format!("expected 'p', 'n', 'b', 'r', 'q', or 'k', got {}", ch)),
    };

    Ok(Piece { color, kind })
}

/// Parse a square like "e3" into a Square
fn square_from_str(s: &str) -> Result<Square, String> {
    let chs: Vec<char> = s.chars().collect();
    if chs.len() != 2 {
        return Err(format!("expected string of length 2, got {}", chs.len()));
    }

    let file: u8 = match chs[0] {
        'a' => 0,
        'b' => 1,
        'c' => 2,
        'd' => 3,
        'e' => 4,
        'f' => 5,
        'g' => 6,
        'h' => 7,
        _ => return Err(format!("expected valid file, got {}", chs[0])),
    };
    let rank:u8 = match chs[1].to_digit(10) {
        Some (d) if (1..=8).contains(&d) => (d - 1) as u8,
        _ => return Err(format!("expected rank 1-8, got {}", chs[1])),
    };

    Ok(Square::new(file, rank))
}

fn char_from_piece(p: Piece) -> char {
    let ch: char = match p.kind {
        PieceKind::Pawn => 'p',
        PieceKind::Knight => 'n',
        PieceKind::Bishop => 'b',
        PieceKind::Rook => 'r',
        PieceKind::Queen => 'q',
        PieceKind::King => 'k',
    };

    match p.color {
        Color::White => ch.to_ascii_uppercase(),
        Color::Black => ch,
    }
}

pub(crate) fn str_from_square(sq: Square) -> String {
    let rank: u8 = sq.rank() + 1;
    format!("{}{}", file_letter(sq.file()), rank.to_string())
}

/// Turn a 0..=7 file index into its letter a..h
pub(crate) fn file_letter(file: u8) -> char {
    match file {
        0 => 'a',
        1 => 'b',
        2 => 'c',
        3 => 'd',
        4 => 'e',
        5 => 'f',
        6 => 'g',
        7 => 'h',
        _ => '-',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_starting_position() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(start).expect("start FEN should parse");

        let w_rook = Piece { color: Color::White, kind: PieceKind::Rook };
        let b_king = Piece { color: Color::Black, kind: PieceKind::King };

        assert_eq!(board.piece_at(Square::new(0, 0)), Some(w_rook));
        assert_eq!(board.piece_at(Square::new(4, 7)), Some(b_king));
        assert_eq!(board.piece_at(Square::new(4, 3)), None);

        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.castling.white_kingside, true);
        assert_eq!(board.halfmove_clock, 0);
    }

    #[test]
    fn fen_idempotency() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(start).expect("start FEN should parse");

        assert_eq!(board.to_fen(), start);
    }
}
