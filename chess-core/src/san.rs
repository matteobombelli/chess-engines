use crate::board::*;
use crate::fen::{ file_letter, str_from_square };
use crate::legal_moves::Move;

/// The standard starting position, used as the base for `import_san`
const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

impl Board {
    /// The Standard Algebraic Notation for a move, without any check/mate suffix,
    /// computed against the pre-move position `self`
    pub(crate) fn san_body(&self, mv: Move) -> String {
        let from: Square = mv.start_square;
        let to: Square = mv.end_square;
        let kind: PieceKind = mv.piece.kind;

        // Castling: the king steps two files off its home square
        if kind == PieceKind::King && from.file() == 4 {
            if to.file() == 6 { return "O-O".to_string(); }
            if to.file() == 2 { return "O-O-O".to_string(); }
        }

        // A capture is a move onto an occupied square, or a pawn onto the
        // en-passant target
        let is_capture: bool = self.piece_at(to).is_some()
            || (kind == PieceKind::Pawn && Some(to) == self.en_passant);

        let mut san = String::new();

        if kind == PieceKind::Pawn {
            // Pawn captures name the origin file, e.g. "exd5"; pushes name only
            // the destination
            if is_capture {
                san.push(file_letter(from.file()));
                san.push('x');
            }
            san.push_str(&str_from_square(to));

            // Promotion on the back rank, defaulting to a queen like `make_move`
            if to.rank() == 0 || to.rank() == 7 {
                let promo: PieceKind = mv.promotion.unwrap_or(PieceKind::Queen);
                san.push('=');
                san.push_str(kind_letter(promo));
            }
        } else {
            san.push_str(kind_letter(kind));
            san.push_str(&self.disambiguation(mv));
            if is_capture { san.push('x'); }
            san.push_str(&str_from_square(to));
        }

        san
    }

    /// The disambiguation part of a piece move: the extra origin file and/or rank
    /// needed when another same-kind piece could also land on the destination
    fn disambiguation(&self, mv: Move) -> String {
        let from: Square = mv.start_square;

        // Other legal moves of the same piece kind reaching the same square.
        // Using legal (not pseudo-legal) moves resolves pinned-piece cases.
        let others: Vec<Square> = self
            .get_legal_moves()
            .into_iter()
            .filter(|m| {
                m.piece.kind == mv.piece.kind
                    && m.end_square == mv.end_square
                    && m.start_square != from
            })
            .map(|m| m.start_square)
            .collect();

        if others.is_empty() {
            return String::new();
        }

        let same_file: bool = others.iter().any(|s| s.file() == from.file());
        let same_rank: bool = others.iter().any(|s| s.rank() == from.rank());

        // Prefer the file; fall back to the rank; use both when neither is unique
        if !same_file {
            file_letter(from.file()).to_string()
        } else if !same_rank {
            (from.rank() + 1).to_string()
        } else {
            format!("{}{}", file_letter(from.file()), from.rank() + 1)
        }
    }

    /// The recorded moves as PGN movetext, e.g. "1. e4 e5 2. Nf3 Nc6".
    ///
    /// Move numbering assumes the game began with White to move.
    pub fn export_san(&self) -> String {
        let mut out = String::new();
        for (i, san) in self.san_history.iter().enumerate() {
            if i % 2 == 0 {
                if !out.is_empty() { out.push(' '); }
                out.push_str(&format!("{}. ", i / 2 + 1));
            } else {
                out.push(' ');
            }
            out.push_str(san);
        }
        out
    }

    /// Play a single SAN move if it is legal in the current position, recording
    /// it in the history. Returns an error if the token matches no legal move.
    pub fn san_to_move(&mut self, san: &str) -> Result<(), String> {
        let target: String = normalize_san(san);
        for mv in self.get_legal_moves() {
            if normalize_san(&self.san_body(mv)) == target {
                self.make_move(mv);
                return Ok(());
            }
        }
        Err(format!("illegal or unrecognized SAN: {san}"))
    }

    /// Replay a movetext string from the standard starting position, the inverse
    /// of `export_san`. Move numbers and a trailing result token are ignored.
    pub fn import_san(movetext: &str) -> Result<Board, String> {
        let mut board = Board::from_fen(START_FEN)?;
        for token in movetext.split_whitespace() {
            // Skip move-number tokens ("1.", "1...") and game results
            if token.contains('.') || matches!(token, "1-0" | "0-1" | "1/2-1/2" | "*") {
                continue;
            }
            board.san_to_move(token)?;
        }
        Ok(board)
    }
}

// Helpers
/// The SAN letter for a piece kind: pawns have none, others are N/B/R/Q/K
fn kind_letter(kind: PieceKind) -> &'static str {
    match kind {
        PieceKind::Pawn => "",
        PieceKind::Knight => "N",
        PieceKind::Bishop => "B",
        PieceKind::Rook => "R",
        PieceKind::Queen => "Q",
        PieceKind::King => "K",
    }
}

/// Normalize a SAN token for comparison: drop check/mate/annotation suffixes and
/// accept zeros for castling (e.g. "0-0")
fn normalize_san(san: &str) -> String {
    san.trim_end_matches(|c| matches!(c, '+' | '#' | '!' | '?'))
        .replace('0', "O")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SAN body of a hand-built move on a position
    fn body(fen: &str, mv: Move) -> String {
        Board::from_fen(fen).expect("FEN should parse").san_body(mv)
    }

    fn pawn(color: Color) -> Piece {
        Piece { color, kind: PieceKind::Pawn }
    }

    #[test]
    fn pawn_push_and_piece_move() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let e4 = Move {
            piece: pawn(Color::White),
            start_square: Square::new(4, 1),
            end_square: Square::new(4, 3),
            promotion: None,
        };
        assert_eq!(body(start, e4), "e4");

        let nf3 = Move {
            piece: Piece { color: Color::White, kind: PieceKind::Knight },
            start_square: Square::new(6, 0),
            end_square: Square::new(5, 2),
            promotion: None,
        };
        assert_eq!(body(start, nf3), "Nf3");
    }

    #[test]
    fn piece_and_pawn_captures() {
        // White bishop on c4 takes on f7
        let fen = "rnbqk1nr/pppp1ppp/8/2b1p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1";
        let bxf7 = Move {
            piece: Piece { color: Color::White, kind: PieceKind::Bishop },
            start_square: Square::new(2, 3),
            end_square: Square::new(5, 6),
            promotion: None,
        };
        assert_eq!(body(fen, bxf7), "Bxf7");

        // White pawn on e4 takes on d5
        let fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1";
        let exd5 = Move {
            piece: pawn(Color::White),
            start_square: Square::new(4, 3),
            end_square: Square::new(3, 4),
            promotion: None,
        };
        assert_eq!(body(fen, exd5), "exd5");
    }

    #[test]
    fn disambiguation_by_file_rank_and_both() {
        // Two knights on the same rank (c3, e3) both reach d5 -> file
        let fen = "4k3/8/8/8/8/2N1N3/8/4K3 w - - 0 1";
        let ncd5 = Move {
            piece: Piece { color: Color::White, kind: PieceKind::Knight },
            start_square: Square::new(2, 2),
            end_square: Square::new(3, 4),
            promotion: None,
        };
        assert_eq!(body(fen, ncd5), "Ncd5");

        // Two knights on the same file (c3, c5) both reach e4 -> rank
        let fen = "4k3/8/8/2N5/8/2N5/8/4K3 w - - 0 1";
        let n3e4 = Move {
            piece: Piece { color: Color::White, kind: PieceKind::Knight },
            start_square: Square::new(2, 2),
            end_square: Square::new(4, 3),
            promotion: None,
        };
        assert_eq!(body(fen, n3e4), "N3e4");

        // Three knights (b5 shares file with b3; f3 shares rank) all reach d4 -> both
        let fen = "4k3/8/8/1N6/8/1N3N2/8/4K3 w - - 0 1";
        let nb3d4 = Move {
            piece: Piece { color: Color::White, kind: PieceKind::Knight },
            start_square: Square::new(1, 2),
            end_square: Square::new(3, 3),
            promotion: None,
        };
        assert_eq!(body(fen, nb3d4), "Nb3d4");
    }

    #[test]
    fn castling_both_sides() {
        let fen = "rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1";
        let king = Piece { color: Color::White, kind: PieceKind::King };

        let short = Move {
            piece: king,
            start_square: Square::new(4, 0),
            end_square: Square::new(6, 0),
            promotion: None,
        };
        assert_eq!(body(fen, short), "O-O");

        let fen = "r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w KQkq - 0 1";
        let long = Move {
            piece: king,
            start_square: Square::new(4, 0),
            end_square: Square::new(2, 0),
            promotion: None,
        };
        assert_eq!(body(fen, long), "O-O-O");
    }

    #[test]
    fn promotion_and_promotion_capture() {
        let fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1";
        let promote = Move {
            piece: pawn(Color::White),
            start_square: Square::new(0, 6),
            end_square: Square::new(0, 7),
            promotion: Some(PieceKind::Queen),
        };
        assert_eq!(body(fen, promote), "a8=Q");

        let under = Move { promotion: Some(PieceKind::Knight), ..promote };
        assert_eq!(body(fen, under), "a8=N");

        // Pawn on a7 captures a knight on b8, promoting to a knight
        let fen = "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1";
        let capture_promote = Move {
            piece: pawn(Color::White),
            start_square: Square::new(0, 6),
            end_square: Square::new(1, 7),
            promotion: Some(PieceKind::Knight),
        };
        assert_eq!(body(fen, capture_promote), "axb8=N");
    }

    #[test]
    fn make_move_records_check_suffix() {
        // White rook lifts to e7, checking the black king on e8 (not mate)
        let mut board = Board::from_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")
            .expect("FEN should parse");
        board.make_move(Move {
            piece: Piece { color: Color::White, kind: PieceKind::Rook },
            start_square: Square::new(4, 1),
            end_square: Square::new(4, 6),
            promotion: None,
        });
        assert_eq!(board.san_history, vec!["Re7+".to_string()]);
    }

    #[test]
    fn make_move_records_checkmate_suffix() {
        // Fool's mate: Black plays Qd8-h4#
        let mut board = Board::from_fen("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
            .expect("FEN should parse");
        board.make_move(Move {
            piece: Piece { color: Color::Black, kind: PieceKind::Queen },
            start_square: Square::new(3, 7),
            end_square: Square::new(7, 3),
            promotion: None,
        });
        assert_eq!(board.san_history, vec!["Qh4#".to_string()]);
    }

    #[test]
    fn export_movetext_with_numbers() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let mut board = Board::from_fen(start).expect("start FEN should parse");

        board.make_move(Move {
            piece: pawn(Color::White),
            start_square: Square::new(4, 1),
            end_square: Square::new(4, 3),
            promotion: None,
        });
        board.make_move(Move {
            piece: pawn(Color::Black),
            start_square: Square::new(4, 6),
            end_square: Square::new(4, 4),
            promotion: None,
        });
        board.make_move(Move {
            piece: Piece { color: Color::White, kind: PieceKind::Knight },
            start_square: Square::new(6, 0),
            end_square: Square::new(5, 2),
            promotion: None,
        });
        board.make_move(Move {
            piece: Piece { color: Color::Black, kind: PieceKind::Knight },
            start_square: Square::new(1, 7),
            end_square: Square::new(2, 5),
            promotion: None,
        });

        assert_eq!(board.export_san(), "1. e4 e5 2. Nf3 Nc6");
    }

    #[test]
    fn san_to_move_plays_legal_and_rejects_illegal() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let mut board = Board::from_fen(start).expect("start FEN should parse");

        assert!(board.san_to_move("e4").is_ok());
        assert_eq!(board.piece_at(Square::new(4, 3)), Some(pawn(Color::White)));
        assert_eq!(board.side_to_move, Color::Black);

        // A black move and a garbage token are both rejected, leaving the board as-is
        let before = board.clone();
        assert!(board.san_to_move("e3").is_err());
        assert!(board.san_to_move("zz").is_err());
        assert_eq!(board, before);
    }

    #[test]
    fn import_export_round_trip() {
        let movetext = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6";
        let board = Board::import_san(movetext).expect("movetext should replay");
        assert_eq!(board.export_san(), movetext);

        // A bad move in the middle surfaces an error
        assert!(Board::import_san("1. e4 e9").is_err());
    }
}
