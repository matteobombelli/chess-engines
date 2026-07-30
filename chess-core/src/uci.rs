use crate::board::*;
use crate::fen::str_from_square;
use crate::legal_moves::Move;

/// The standard starting position, used as the base for `import_uci`
const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

impl Move {
    /// The move in long algebraic notation: origin square, destination square, and
    /// a lowercase promotion letter, e.g. "e2e4", "e1g1", "e7e8q".
    ///
    /// This is the canonical key the learned engines store and exchange. Unlike
    /// SAN it needs no knowledge of the position to write, so it survives being
    /// passed between Rust, Python, and a stored count table.
    ///
    /// Castling is encoded as the king's own two-square step ("e1g1"), matching
    /// how this crate represents the move. A pawn arriving on the back rank
    /// always carries a promotion letter: `promotion: None` there is defaulted to
    /// a queen exactly as `make_move` and `san_body` do, so the three never
    /// disagree about what a move means.
    pub fn to_uci(self) -> String {
        let mut uci: String = format!(
            "{}{}",
            str_from_square(self.start_square),
            str_from_square(self.end_square)
        );

        if self.piece.kind == PieceKind::Pawn
            && (self.end_square.rank() == 0 || self.end_square.rank() == 7)
        {
            let promo: PieceKind = self.promotion.unwrap_or(PieceKind::Queen);
            uci.push(promotion_letter(promo));
        }

        uci
    }
}

impl Board {
    /// Find the legal move matching a UCI string, without trusting the string to
    /// describe a real move.
    ///
    /// The lookup is deliberately a search over `get_legal_moves`: the returned
    /// `Move` is one this position actually generated, so a caller can apply it
    /// without re-deriving a piece, a capture, or a castling side from text. An
    /// unparsable, illegal, or merely unrecognized string is an error, never a
    /// guess.
    pub fn move_from_uci(&self, uci: &str) -> Result<Move, String> {
        // Reject obvious junk before scanning the legal set so the error message
        // distinguishes "not a UCI string" from "not legal here".
        if !is_uci_shaped(uci) {
            return Err(format!("malformed UCI move: {uci}"));
        }

        self.get_legal_moves()
            .into_iter()
            .find(|mv| mv.to_uci() == uci)
            .ok_or_else(|| format!("illegal UCI move in this position: {uci}"))
    }

    /// Play a single UCI move if it is legal, recording it in `san_history`.
    /// The UCI counterpart of `san_to_move`.
    pub fn uci_to_move(&mut self, uci: &str) -> Result<Move, String> {
        let mv: Move = self.move_from_uci(uci)?;
        self.make_move(mv);
        Ok(mv)
    }

    /// Every legal move in this position as canonical UCI.
    ///
    /// This is the candidate set the gateway sends to a model service. A model
    /// may rank these strings; it may never introduce another one.
    pub fn legal_uci_moves(&self) -> Vec<String> {
        self.get_legal_moves()
            .into_iter()
            .map(|mv| mv.to_uci())
            .collect()
    }

    /// Replay a whole-game UCI sequence from the standard starting position.
    ///
    /// Used to rebuild and verify an untrusted move history: replay it, then
    /// compare the resulting FEN with the position the client claims to be in.
    pub fn import_uci(moves: &[String]) -> Result<Board, String> {
        let mut board: Board = Board::from_fen(START_FEN)?;
        for uci in moves {
            board.uci_to_move(uci)?;
        }
        Ok(board)
    }
}

// Helpers
/// The lowercase UCI letter for a promotion piece
fn promotion_letter(kind: PieceKind) -> char {
    match kind {
        PieceKind::Knight => 'n',
        PieceKind::Bishop => 'b',
        PieceKind::Rook => 'r',
        // A pawn cannot stay a pawn or become a king; both are written as the
        // queen the move generator would have produced.
        _ => 'q',
    }
}

/// Whether a string has the shape of a UCI move: two squares plus an optional
/// promotion letter. Shape only; legality is decided against the position.
fn is_uci_shaped(uci: &str) -> bool {
    let chars: Vec<char> = uci.chars().collect();
    if chars.len() != 4 && chars.len() != 5 {
        return false;
    }

    let square_ok = |file: char, rank: char| {
        ('a'..='h').contains(&file) && ('1'..='8').contains(&rank)
    };

    square_ok(chars[0], chars[1])
        && square_ok(chars[2], chars[3])
        && chars.get(4).is_none_or(|c| matches!(c, 'q' | 'r' | 'b' | 'n'))
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn pawn_push_and_knight_move_round_trip() {
        let board = Board::from_fen(START).expect("start FEN should parse");

        let e4 = board.move_from_uci("e2e4").expect("e2e4 is legal at the start");
        assert_eq!(e4.to_uci(), "e2e4");
        assert_eq!(board.san_body(e4), "e4");

        let nf3 = board.move_from_uci("g1f3").expect("g1f3 is legal at the start");
        assert_eq!(nf3.to_uci(), "g1f3");
        assert_eq!(board.san_body(nf3), "Nf3");
    }

    #[test]
    fn castling_is_the_king_two_square_step() {
        let board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
            .expect("FEN should parse");

        let short = board.move_from_uci("e1g1").expect("kingside castling is legal");
        let long = board.move_from_uci("e1c1").expect("queenside castling is legal");

        assert_eq!(board.san_body(short), "O-O");
        assert_eq!(board.san_body(long), "O-O-O");
        assert_eq!(short.to_uci(), "e1g1");
        assert_eq!(long.to_uci(), "e1c1");
    }

    #[test]
    fn castling_moves_the_rook_when_applied_from_uci() {
        let mut board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
            .expect("FEN should parse");
        board.uci_to_move("e1g1").expect("kingside castling is legal");

        let rook = Piece { color: Color::White, kind: PieceKind::Rook };
        let king = Piece { color: Color::White, kind: PieceKind::King };
        assert_eq!(board.piece_at(Square::new(6, 0)), Some(king));
        assert_eq!(board.piece_at(Square::new(5, 0)), Some(rook));
        assert_eq!(board.piece_at(Square::new(7, 0)), None);
    }

    #[test]
    fn en_passant_capture_round_trips() {
        // Black has just played d7d5; White's e5 pawn may take on d6.
        let mut board = Board::from_fen(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        )
        .expect("FEN should parse");

        let capture = board.move_from_uci("e5d6").expect("en passant is legal here");
        assert_eq!(capture.to_uci(), "e5d6");

        board.uci_to_move("e5d6").expect("en passant is legal here");
        // The captured pawn sat on d5, behind the destination square.
        assert_eq!(board.piece_at(Square::new(3, 4)), None);
        assert_eq!(board.san_history, vec!["exd6".to_string()]);
    }

    #[test]
    fn every_promotion_choice_has_a_distinct_uci() {
        let board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
            .expect("FEN should parse");

        let promotions: Vec<String> = board
            .legal_uci_moves()
            .into_iter()
            .filter(|uci| uci.starts_with("a7a8"))
            .collect();

        assert_eq!(promotions, vec!["a7a8q", "a7a8r", "a7a8b", "a7a8n"]);

        let knight = board.move_from_uci("a7a8n").expect("underpromotion is legal");
        assert_eq!(knight.promotion, Some(PieceKind::Knight));
        assert_eq!(board.san_body(knight), "a8=N");
    }

    #[test]
    fn a_promotion_without_its_letter_is_not_a_legal_candidate() {
        // "a7a8" is UCI-shaped but ambiguous: the move generator only ever emits
        // the four explicit promotions, so the bare string must not resolve.
        let board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
            .expect("FEN should parse");

        let error = board.move_from_uci("a7a8").unwrap_err();
        assert!(error.contains("illegal"), "unexpected error: {error}");
    }

    #[test]
    fn malformed_and_illegal_strings_are_rejected_differently() {
        let board = Board::from_fen(START).expect("start FEN should parse");

        assert!(board.move_from_uci("e2e9").unwrap_err().contains("malformed"));
        assert!(board.move_from_uci("").unwrap_err().contains("malformed"));
        assert!(board.move_from_uci("e2e4e4").unwrap_err().contains("malformed"));
        assert!(board.move_from_uci("e2e4k").unwrap_err().contains("malformed"));
        // Well formed, but not White's move to make from the start position.
        assert!(board.move_from_uci("e7e5").unwrap_err().contains("illegal"));
    }

    #[test]
    fn an_illegal_move_leaves_the_board_untouched() {
        let mut board = Board::from_fen(START).expect("start FEN should parse");
        let before = board.clone();

        assert!(board.uci_to_move("e7e5").is_err());
        assert_eq!(board, before);
    }

    #[test]
    fn legal_uci_moves_matches_the_legal_move_count() {
        let board = Board::from_fen(START).expect("start FEN should parse");
        let uci = board.legal_uci_moves();

        assert_eq!(uci.len(), 20);
        assert_eq!(uci.len(), board.get_legal_moves().len());
        // Canonical UCI is unique per legal move, which is what lets a score map
        // be keyed by it.
        let mut sorted = uci.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), uci.len());
    }

    #[test]
    fn import_uci_replays_a_game_and_agrees_with_san() {
        let moves: Vec<String> = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let board = Board::import_uci(&moves).expect("the Ruy Lopez should replay");

        assert_eq!(board.export_san(), "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6");
        assert_eq!(
            Board::import_san("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6").unwrap().to_fen(),
            board.to_fen()
        );
    }

    #[test]
    fn import_uci_rejects_an_illegal_continuation() {
        let moves: Vec<String> = ["e2e4", "e2e4"].iter().map(|s| s.to_string()).collect();
        assert!(Board::import_uci(&moves).is_err());
    }

    #[test]
    fn every_legal_move_in_a_busy_position_round_trips() {
        // A position with castling, promotions, en passant, and captures all
        // available at once: to_uci and move_from_uci must be exact inverses for
        // the whole legal set, since that pairing is what keeps a model's score
        // keys attached to real moves.
        for fen in [
            START,
            "r3k2r/pPpppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "8/2k5/8/8/8/8/5k2/6R1 b - - 0 1",
        ] {
            let board = Board::from_fen(fen).expect("FEN should parse");
            for mv in board.get_legal_moves() {
                let uci = mv.to_uci();
                assert_eq!(
                    board.move_from_uci(&uci),
                    Ok(mv),
                    "{uci} did not round trip in {fen}"
                );
            }
        }
    }
}
