//! Reversible position state for tree search.
//!
//! [`Board`] remains the user-facing game representation and records SAN plus
//! canonical position strings. Search only needs the current chess state, so
//! [`SearchPosition`] converts that history once, maintains compact repetition
//! keys, and applies moves with a small delta undo record.

use std::collections::HashMap;

use crate::{Board, CastlingRights, Color, Move, Piece, PieceKind, Square, Status};

/// A chess position optimized for reversible tree traversal.
///
/// The position owns no SAN or string history. Clone this once per independent
/// search worker, then use [`make_move`](Self::make_move) and
/// [`unmake_move`](Self::unmake_move) while traversing the tree.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SearchPosition {
    board: Board,
    bitboards: [u64; 12],
    key: u128,
    effective_en_passant: Option<Square>,
    key_history: Vec<u128>,
    repetitions: HashMap<u128, u16>,
}

/// Opaque state required to undo one search move.
///
/// Undo records are stack-disciplined: the most recently returned value must
/// be the next one passed to [`SearchPosition::unmake_move`].
#[derive(Debug)]
#[must_use = "the undo record is required to restore the search position"]
pub struct SearchUndo {
    board: BoardUndo,
    previous_key: u128,
    resulting_key: u128,
    previous_effective_en_passant: Option<Square>,
    history_len_before: usize,
}

#[derive(Debug)]
struct BoardUndo {
    // A legal chess move changes at most four squares (castling).
    changed: [Option<(Square, Option<Piece>)>; 4],
    changed_len: usize,
    side_to_move: Color,
    castling: CastlingRights,
    en_passant: Option<Square>,
    halfmove_clock: u32,
    fullmove_number: u32,
}

impl SearchPosition {
    /// Convert a game board into search state.
    ///
    /// Existing repetition history is preserved, but its string keys are
    /// converted to stable 128-bit keys and strings/SAN are then discarded.
    pub fn from_board(source: &Board) -> Self {
        let mut key_history = Vec::with_capacity(source.position_history.len().max(1));
        for position in &source.position_history {
            let historical = Board::from_fen(&format!("{position} 0 1"))
                .expect("Board position history must contain canonical FEN fields");
            key_history.push(position_key(&historical).0);
        }

        let mut board = source.clone();
        board.san_history = Vec::new();
        board.position_history = Vec::new();

        let (key, effective_en_passant) = position_key(&board);
        if key_history.last().copied() != Some(key) {
            // Public Board fields can be edited directly. Treat such a board as
            // having reached the edited position after its recorded history.
            key_history.push(key);
        }
        if key_history.is_empty() {
            key_history.push(key);
        }

        let mut repetitions = HashMap::with_capacity(key_history.len());
        for &historical_key in &key_history {
            let count = repetitions.entry(historical_key).or_insert(0u16);
            *count = count.saturating_add(1);
        }

        Self {
            bitboards: make_bitboards(&board),
            board,
            key,
            effective_en_passant,
            key_history,
            repetitions,
        }
    }

    /// The piece currently occupying `square`.
    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        self.board.piece_at(square)
    }

    /// All squares in a1-to-h8 order.
    pub fn squares(&self) -> &[Option<Piece>; 64] {
        &self.board.squares
    }

    /// Occupancy bitboard for one colored piece kind (a1 is the low bit).
    pub fn piece_bitboard(&self, color: Color, kind: PieceKind) -> u64 {
        self.bitboards[piece_index(Piece { color, kind })]
    }

    /// Occupancy bitboard for all pieces.
    pub fn occupancy(&self) -> u64 {
        self.bitboards
            .iter()
            .copied()
            .fold(0, |all, pieces| all | pieces)
    }

    /// The player whose turn it is.
    pub fn side_to_move(&self) -> Color {
        self.board.side_to_move
    }

    /// Current castling rights.
    pub fn castling_rights(&self) -> CastlingRights {
        self.board.castling
    }

    /// Raw FEN en-passant target, whether or not a legal capture exists.
    pub fn en_passant_target(&self) -> Option<Square> {
        self.board.en_passant
    }

    /// Halfmoves since the last pawn move or capture.
    pub fn halfmove_clock(&self) -> u32 {
        self.board.halfmove_clock
    }

    /// FEN fullmove number.
    pub fn fullmove_number(&self) -> u32 {
        self.board.fullmove_number
    }

    /// Stable 128-bit Zobrist identity for chess repetition state.
    ///
    /// Move clocks are excluded. An en-passant target is included only when it
    /// enables a legal en-passant capture, matching repetition semantics.
    pub fn position_key(&self) -> u128 {
        self.key
    }

    /// Number of occurrences of the current position, including this one.
    pub fn repetition_count(&self) -> u16 {
        self.repetition_count_for(self.key)
    }

    /// Number of earlier occurrences of the current position.
    pub fn prior_repetition_count(&self) -> u16 {
        self.repetition_count().saturating_sub(1)
    }

    /// Number of occurrences of `key` on the current game path.
    pub fn repetition_count_for(&self, key: u128) -> u16 {
        self.repetitions.get(&key).copied().unwrap_or(0)
    }

    /// Number of positions on the game path, including the current position.
    pub fn history_len(&self) -> usize {
        self.key_history.len()
    }

    /// Whether the side to move is in check.
    pub fn is_in_check(&self) -> bool {
        self.board.is_in_check()
    }

    /// Fill `moves` with all legal moves, reusing its allocation.
    pub fn legal_moves_into(&mut self, moves: &mut Vec<Move>) {
        let moving_side = self.board.side_to_move;
        self.board.pseudo_legal_moves_into(moves);

        let candidate_count = moves.len();
        let mut accepted = 0;
        for candidate in 0..candidate_count {
            let mv = moves[candidate];
            let undo = self.apply_board_move(mv);
            let legal = self
                .board
                .find_king(moving_side)
                .is_some_and(|king| !self.board.is_attacked(king, moving_side.opposite()));
            self.restore_board(undo);

            if legal {
                moves[accepted] = mv;
                accepted += 1;
            }
        }
        moves.truncate(accepted);
    }

    /// Allocate and return all legal moves.
    pub fn legal_moves(&mut self) -> Vec<Move> {
        let mut moves = Vec::new();
        self.legal_moves_into(&mut moves);
        moves
    }

    /// Classify the current position, generating legal moves once.
    pub fn status(&mut self) -> Status {
        let mut legal_moves = Vec::new();
        self.legal_moves_into(&mut legal_moves);
        self.status_with_legal_moves(&legal_moves)
    }

    /// Classify the position using legal moves already generated for it.
    ///
    /// `legal_moves` must belong to this exact position. Mate/stalemate retain
    /// precedence over automatic draw rules, matching [`Board::status`].
    pub fn status_with_legal_moves(&self, legal_moves: &[Move]) -> Status {
        if legal_moves.is_empty() {
            return if self.is_in_check() {
                Status::Checkmate
            } else {
                Status::Stalemate
            };
        }
        if self.board.has_insufficient_material() {
            return Status::InsufficientMaterial;
        }
        if self.board.halfmove_clock >= 100 {
            return Status::FiftyMoveRule;
        }
        if self.repetition_count() >= 3 {
            return Status::ThreefoldRepetition;
        }
        Status::Ongoing
    }

    /// Apply a legal move and return the opaque record needed to undo it.
    pub fn make_move(&mut self, mv: Move) -> SearchUndo {
        assert_eq!(
            self.board.piece_at(mv.start_square),
            Some(mv.piece),
            "search move's piece must occupy its start square"
        );

        let previous_key = self.key;
        let previous_effective_en_passant = self.effective_en_passant;
        let history_len_before = self.key_history.len();
        let board = self.apply_board_move(mv);
        let new_effective_en_passant = self.board.effective_en_passant_target();

        self.update_bitboards_and_key(&board, new_effective_en_passant);
        self.effective_en_passant = new_effective_en_passant;

        let resulting_key = self.key;
        self.key_history.push(resulting_key);
        let count = self.repetitions.entry(resulting_key).or_insert(0);
        *count = count.saturating_add(1);

        SearchUndo {
            board,
            previous_key,
            resulting_key,
            previous_effective_en_passant,
            history_len_before,
        }
    }

    /// Undo the most recently applied search move.
    pub fn unmake_move(&mut self, undo: SearchUndo) {
        assert_eq!(
            self.key, undo.resulting_key,
            "search undos must be applied in reverse move order"
        );
        assert_eq!(
            self.key_history.len(),
            undo.history_len_before + 1,
            "search undo does not belong to the current path"
        );

        let popped = self.key_history.pop();
        assert_eq!(popped, Some(undo.resulting_key));
        let remove_count = {
            let count = self
                .repetitions
                .get_mut(&undo.resulting_key)
                .expect("current search key must have a repetition entry");
            *count -= 1;
            *count == 0
        };
        if remove_count {
            self.repetitions.remove(&undo.resulting_key);
        }

        self.restore_bitboards(&undo.board);
        self.restore_board(undo.board);
        self.key = undo.previous_key;
        self.effective_en_passant = undo.previous_effective_en_passant;

        debug_assert_eq!(self.key_history.last().copied(), Some(self.key));
    }

    fn apply_board_move(&mut self, mv: Move) -> BoardUndo {
        let mut undo = BoardUndo {
            changed: [None; 4],
            changed_len: 0,
            side_to_move: self.board.side_to_move,
            castling: self.board.castling,
            en_passant: self.board.en_passant,
            halfmove_clock: self.board.halfmove_clock,
            fullmove_number: self.board.fullmove_number,
        };

        remember_square(&self.board, &mut undo, mv.start_square);
        remember_square(&self.board, &mut undo, mv.end_square);

        if let Some(captured_pawn) =
            self.board
                .en_passant_capture_square(mv.piece, mv.start_square, mv.end_square)
        {
            remember_square(&self.board, &mut undo, captured_pawn);
        }

        if mv.piece.kind == PieceKind::King
            && mv.start_square.file() == 4
            && mv.end_square.file().abs_diff(mv.start_square.file()) == 2
        {
            let rank = mv.start_square.rank();
            let (rook_from, rook_to) = if mv.end_square.file() == 6 {
                (Square::new(7, rank), Square::new(5, rank))
            } else {
                (Square::new(0, rank), Square::new(3, rank))
            };
            remember_square(&self.board, &mut undo, rook_from);
            remember_square(&self.board, &mut undo, rook_to);
        }

        self.board.apply_move(mv);
        undo
    }

    fn restore_board(&mut self, undo: BoardUndo) {
        for &(square, piece) in undo.changed[..undo.changed_len].iter().flatten() {
            self.board.set_piece(square, piece);
        }
        self.board.side_to_move = undo.side_to_move;
        self.board.castling = undo.castling;
        self.board.en_passant = undo.en_passant;
        self.board.halfmove_clock = undo.halfmove_clock;
        self.board.fullmove_number = undo.fullmove_number;
    }

    fn update_bitboards_and_key(
        &mut self,
        undo: &BoardUndo,
        new_effective_en_passant: Option<Square>,
    ) {
        for &(square, old_piece) in undo.changed[..undo.changed_len].iter().flatten() {
            if let Some(piece) = old_piece {
                self.bitboards[piece_index(piece)] &= !(1u64 << square.0);
                self.key ^= piece_feature(piece, square);
            }
            if let Some(piece) = self.board.piece_at(square) {
                self.bitboards[piece_index(piece)] |= 1u64 << square.0;
                self.key ^= piece_feature(piece, square);
            }
        }

        xor_side(&mut self.key, undo.side_to_move);
        xor_side(&mut self.key, self.board.side_to_move);
        xor_castling(&mut self.key, undo.castling);
        xor_castling(&mut self.key, self.board.castling);
        xor_en_passant(&mut self.key, self.effective_en_passant);
        xor_en_passant(&mut self.key, new_effective_en_passant);
    }

    fn restore_bitboards(&mut self, undo: &BoardUndo) {
        for &(square, old_piece) in undo.changed[..undo.changed_len].iter().flatten() {
            if let Some(piece) = self.board.piece_at(square) {
                self.bitboards[piece_index(piece)] &= !(1u64 << square.0);
            }
            if let Some(piece) = old_piece {
                self.bitboards[piece_index(piece)] |= 1u64 << square.0;
            }
        }
    }
}

impl From<&Board> for SearchPosition {
    fn from(board: &Board) -> Self {
        Self::from_board(board)
    }
}

fn remember_square(board: &Board, undo: &mut BoardUndo, square: Square) {
    if undo.changed[..undo.changed_len]
        .iter()
        .flatten()
        .any(|(remembered, _)| *remembered == square)
    {
        return;
    }
    assert!(undo.changed_len < undo.changed.len());
    undo.changed[undo.changed_len] = Some((square, board.piece_at(square)));
    undo.changed_len += 1;
}

fn make_bitboards(board: &Board) -> [u64; 12] {
    let mut bitboards = [0u64; 12];
    for index in 0..64 {
        if let Some(piece) = board.squares[index] {
            bitboards[piece_index(piece)] |= 1u64 << index;
        }
    }
    bitboards
}

fn piece_index(piece: Piece) -> usize {
    let color = match piece.color {
        Color::White => 0,
        Color::Black => 6,
    };
    let kind = match piece.kind {
        PieceKind::Pawn => 0,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::Queen => 4,
        PieceKind::King => 5,
    };
    color + kind
}

fn position_key(board: &Board) -> (u128, Option<Square>) {
    let mut key = 0u128;
    for index in 0..64 {
        if let Some(piece) = board.squares[index] {
            key ^= piece_feature(piece, Square(index as u8));
        }
    }
    xor_side(&mut key, board.side_to_move);
    xor_castling(&mut key, board.castling);
    let effective = board.effective_en_passant_target();
    xor_en_passant(&mut key, effective);
    (key, effective)
}

// Feature numbering is an on-disk/public identity contract. Keep it stable.
// 0..768: colored piece/square, 768: black to move, 769..773: castling,
// 773..837: effective en-passant square.
fn piece_feature(piece: Piece, square: Square) -> u128 {
    zobrist_feature((piece_index(piece) * 64 + square.index()) as u64)
}

fn xor_side(key: &mut u128, side: Color) {
    if side == Color::Black {
        *key ^= zobrist_feature(768);
    }
}

fn xor_castling(key: &mut u128, rights: CastlingRights) {
    for (offset, enabled) in [
        rights.white_kingside,
        rights.white_queenside,
        rights.black_kingside,
        rights.black_queenside,
    ]
    .into_iter()
    .enumerate()
    {
        if enabled {
            *key ^= zobrist_feature(769 + offset as u64);
        }
    }
}

fn xor_en_passant(key: &mut u128, square: Option<Square>) {
    if let Some(square) = square {
        *key ^= zobrist_feature(773 + square.0 as u64);
    }
}

fn zobrist_feature(feature: u64) -> u128 {
    let low = splitmix64(feature ^ 0x243f_6a88_85a3_08d3);
    let high = splitmix64(feature ^ 0x1319_8a2e_0370_7344);
    (u128::from(high) << 64) | u128::from(low)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    fn move_from_uci(board: &Board, uci: &str) -> Move {
        board.move_from_uci(uci).expect("test move must be legal")
    }

    fn assert_round_trip(fen: &str, uci: &str) {
        let board = Board::from_fen(fen).expect("test FEN must parse");
        let mv = move_from_uci(&board, uci);
        let mut search = SearchPosition::from_board(&board);
        let original = search.clone();

        let undo = search.make_move(mv);
        search.unmake_move(undo);

        assert_eq!(search, original);
    }

    #[test]
    fn exposes_position_state_and_bitboards() {
        let board = Board::from_fen(START).unwrap();
        let search = SearchPosition::from_board(&board);

        assert_eq!(search.side_to_move(), Color::White);
        assert_eq!(search.castling_rights(), board.castling);
        assert_eq!(search.en_passant_target(), None);
        assert_eq!(search.halfmove_clock(), 0);
        assert_eq!(search.fullmove_number(), 1);
        assert_eq!(search.squares(), &board.squares);
        assert_eq!(
            search.piece_bitboard(Color::White, PieceKind::Pawn),
            0x0000_0000_0000_ff00
        );
        assert_eq!(search.occupancy(), 0xffff_0000_0000_ffff);
    }

    #[test]
    fn reusable_legal_move_buffer_matches_board() {
        let board = Board::from_fen(START).unwrap();
        let mut search = SearchPosition::from_board(&board);
        let mut moves = vec![Move {
            piece: Piece {
                color: Color::White,
                kind: PieceKind::King,
            },
            start_square: Square::new(0, 0),
            end_square: Square::new(0, 0),
            promotion: None,
        }];

        search.legal_moves_into(&mut moves);
        assert_eq!(moves, board.get_legal_moves());

        let capacity = moves.capacity();
        search.legal_moves_into(&mut moves);
        assert_eq!(moves.len(), 20);
        assert!(moves.capacity() >= capacity);
    }

    fn perft(search: &mut SearchPosition, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = search.legal_moves();
        let mut nodes = 0;
        for mv in moves {
            let undo = search.make_move(mv);
            nodes += perft(search, depth - 1);
            search.unmake_move(undo);
        }
        nodes
    }

    #[test]
    fn reversible_move_tree_matches_starting_perft() {
        let board = Board::from_fen(START).unwrap();
        let mut search = SearchPosition::from_board(&board);
        let original = search.clone();

        assert_eq!(perft(&mut search, 3), 8_902);
        assert_eq!(search, original);
    }

    #[test]
    fn legal_generation_does_not_change_position() {
        let board =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        let mut search = SearchPosition::from_board(&board);
        let original = search.clone();

        let _ = search.legal_moves();
        assert_eq!(search, original);
    }

    #[test]
    fn ordinary_capture_castling_en_passant_and_promotion_round_trip() {
        assert_round_trip(START, "e2e4");
        assert_round_trip("4k3/8/8/8/8/8/4r3/4R1K1 w - - 7 20", "e1e2");
        assert_round_trip("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 12 34", "e1g1");
        assert_round_trip("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 20", "e5d6");
        assert_round_trip("4k3/P7/8/8/8/8/8/4K3 w - - 0 40", "a7a8n");
    }

    #[test]
    fn nested_moves_restore_key_history_and_all_state() {
        let board = Board::from_fen(START).unwrap();
        let mut search = SearchPosition::from_board(&board);
        let original = search.clone();

        let e2e4 = search
            .legal_moves()
            .into_iter()
            .find(|mv| mv.start_square == Square::new(4, 1) && mv.end_square == Square::new(4, 3))
            .unwrap();
        let first = search.make_move(e2e4);
        let after_first = search.clone();
        let e7e5 = search
            .legal_moves()
            .into_iter()
            .find(|mv| mv.start_square == Square::new(4, 6) && mv.end_square == Square::new(4, 4))
            .unwrap();
        let second = search.make_move(e7e5);

        assert_ne!(search.position_key(), after_first.position_key());
        search.unmake_move(second);
        assert_eq!(search, after_first);
        search.unmake_move(first);
        assert_eq!(search, original);
    }

    #[test]
    fn incremental_state_matches_fresh_conversion_along_a_game() {
        let mut board = Board::from_fen(START).unwrap();
        let mut search = SearchPosition::from_board(&board);

        for uci in [
            "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1",
            "b7b5", "a4b3", "d7d6",
        ] {
            let mv = move_from_uci(&board, uci);
            let _undo = search.make_move(mv);
            board.make_search_move(mv);

            let fresh = SearchPosition::from_board(&board);
            assert_eq!(search, fresh, "incremental state diverged after {uci}");
            assert_eq!(position_key(&search.board).0, search.position_key());
        }
    }

    #[test]
    fn repetition_history_is_imported_and_updated_reversibly() {
        let mut board = Board::from_fen(START).unwrap();
        for san in ["Nf3", "Nf6", "Ng1", "Ng8"] {
            board.san_to_move(san).unwrap();
        }
        let mut search = SearchPosition::from_board(&board);
        assert_eq!(search.repetition_count(), 2);
        assert_eq!(search.prior_repetition_count(), 1);

        let before = search.clone();
        let mv = search
            .legal_moves()
            .into_iter()
            .find(|mv| mv.start_square == Square::new(6, 0) && mv.end_square == Square::new(5, 2))
            .unwrap();
        let undo = search.make_move(mv);
        assert_eq!(search.repetition_count(), 2);
        search.unmake_move(undo);
        assert_eq!(search, before);
    }

    #[test]
    fn status_uses_existing_moves_and_preserves_terminal_precedence() {
        for (fen, expected) in [
            (
                "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 100 51",
                Status::Checkmate,
            ),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", Status::Stalemate),
            ("4k3/8/8/8/8/8/8/R3K3 w - - 100 50", Status::FiftyMoveRule),
        ] {
            let board = Board::from_fen(fen).unwrap();
            let mut search = SearchPosition::from_board(&board);
            let moves = search.legal_moves();
            assert_eq!(search.status_with_legal_moves(&moves), expected);
            assert_eq!(search.status(), expected);
        }
    }

    #[test]
    fn threefold_status_survives_board_conversion() {
        let mut board = Board::from_fen(START).unwrap();
        for san in ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"] {
            board.san_to_move(san).unwrap();
        }
        let mut search = SearchPosition::from_board(&board);
        let moves = search.legal_moves();
        assert_eq!(search.repetition_count(), 3);
        assert_eq!(
            search.status_with_legal_moves(&moves),
            Status::ThreefoldRepetition
        );
    }

    #[test]
    fn position_key_has_repetition_semantics_and_is_clock_independent() {
        let plain = Board::from_fen("4k3/8/8/3p4/8/8/8/4K3 w - - 0 1").unwrap();
        let unusable = Board::from_fen("4k3/8/8/3p4/8/8/8/4K3 w - d6 99 40").unwrap();
        let capturable = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let no_target = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1").unwrap();

        assert_eq!(
            SearchPosition::from_board(&plain).position_key(),
            SearchPosition::from_board(&unusable).position_key()
        );
        assert_ne!(
            SearchPosition::from_board(&capturable).position_key(),
            SearchPosition::from_board(&no_target).position_key()
        );

        let pinned = Board::from_fen("4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let pinned_without_target = Board::from_fen("4r1k1/8/8/3pP3/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            SearchPosition::from_board(&pinned).position_key(),
            SearchPosition::from_board(&pinned_without_target).position_key(),
            "an en-passant capture forbidden by a pin cannot affect repetition identity"
        );
    }

    #[test]
    fn impossible_en_passant_rank_does_not_change_search_identity() {
        let without = Board::from_fen("4k3/8/8/8/8/3Pp3/8/4K3 w - - 0 1").unwrap();
        let mut malformed = without.clone();
        malformed.en_passant = Some(Square::new(4, 3));

        let without = SearchPosition::from_board(&without);
        let malformed = SearchPosition::from_board(&malformed);
        assert_eq!(without.position_key(), malformed.position_key());
    }

    #[test]
    fn position_key_changes_for_side_piece_and_castling_rights() {
        let base = SearchPosition::from_board(&Board::from_fen(START).unwrap()).position_key();
        let black = SearchPosition::from_board(
            &Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap(),
        )
        .position_key();
        let no_castling = SearchPosition::from_board(
            &Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1").unwrap(),
        )
        .position_key();
        let missing_pawn = SearchPosition::from_board(
            &Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap(),
        )
        .position_key();

        assert_ne!(base, black);
        assert_ne!(base, no_castling);
        assert_ne!(base, missing_pawn);
        // Freeze one known key so accidental feature/PRNG changes are caught.
        assert_eq!(base, 0x50e8_c360_e0e3_f845_c803_81e6_edbd_46ed);
    }
}
