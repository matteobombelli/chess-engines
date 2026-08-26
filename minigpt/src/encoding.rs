use alphamini::START_FEN;
use alphamini::policy::{POLICY_SIZE, POLICY_VERSION, PolicyError, move_to_action};
use chess_core::{Board, movetext_moves};
use thiserror::Error;

/// The move vocabulary is exactly the AlphaMini action space, so a token id
/// below [`BOS_TOKEN`] can always be decoded with `alphamini::policy`.
pub const TOKENIZER_VERSION: &str = POLICY_VERSION;
pub const BOS_TOKEN: u16 = POLICY_SIZE as u16;
pub const PAD_TOKEN: u16 = BOS_TOKEN + 1;
/// Rounded up from the 4674 used ids so embedding and output matmuls land on a
/// GPU-friendly multiple of 64. Ids 4674..4736 are never emitted.
pub const VOCAB_SIZE: usize = 4_736;

const _: () = assert!(POLICY_SIZE == 4_672);
const _: () = assert!(VOCAB_SIZE.is_multiple_of(64) && VOCAB_SIZE > PAD_TOKEN as usize);

#[derive(Debug, Error)]
pub enum EncodingError {
    #[error("ply {ply}: {message}")]
    San { ply: usize, message: String },
    #[error("ply {ply}: {source}")]
    Policy {
        ply: usize,
        #[source]
        source: PolicyError,
    },
}

/// Incremental SAN-to-token encoder holding the replayed position.
///
/// A failed [`push_san`](Self::push_san) may leave the board a move ahead of
/// the token stream, so an encoder that returned an error must be discarded.
pub struct GameEncoder {
    board: Board,
    tokens: Vec<u16>,
}

impl GameEncoder {
    pub fn new() -> Self {
        Self {
            board: Board::from_fen(START_FEN).expect("frozen start FEN is valid"),
            tokens: vec![BOS_TOKEN],
        }
    }

    /// Play one SAN move and append its action token. Legality is owned by
    /// `chess_core`: the token can only come from a move in the legal set.
    pub fn push_san(&mut self, san: &str) -> Result<(), EncodingError> {
        let ply = self.tokens.len() - 1;
        let side_to_move = self.board.side_to_move;
        let mv = self
            .board
            .san_to_move(san)
            .map_err(|message| EncodingError::San { ply, message })?;
        let action = move_to_action(mv, side_to_move)
            .map_err(|source| EncodingError::Policy { ply, source })?;
        self.tokens.push(action as u16);
        Ok(())
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn tokens(&self) -> &[u16] {
        &self.tokens
    }

    pub fn into_tokens(self) -> Vec<u16> {
        self.tokens
    }
}

impl Default for GameEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Tokenize PGN movetext replayed from the standard starting position.
///
/// The returned stream is `BOS` followed by one action token per ply. Comments,
/// variations, and NAGs are rejected by `movetext_moves`; strip them before
/// calling (see [`crate::pgn::sanitize_movetext`]).
pub fn encode_movetext(movetext: &str) -> Result<Vec<u16>, EncodingError> {
    let mut encoder = GameEncoder::new();
    for san in movetext_moves(movetext) {
        encoder.push_san(san)?;
    }
    Ok(encoder.into_tokens())
}

#[cfg(test)]
mod tests {
    use alphamini::policy::action_to_move;
    use chess_core::Status;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// Play random legal moves and return the movetext plus the moves played.
    fn random_game(seed: u64, max_plies: usize) -> Board {
        let mut board = Board::from_fen(START_FEN).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..max_plies {
            if board.status() != Status::Ongoing {
                break;
            }
            let legal = board.get_legal_moves();
            let chosen = legal[rng.gen_range(0..legal.len())];
            board.make_move(chosen);
        }
        board
    }

    #[test]
    fn random_games_round_trip_through_the_action_vocabulary() {
        for seed in 0..8 {
            let played = random_game(seed, 80);
            let movetext = played.export_san();
            let tokens = encode_movetext(&movetext).expect("a legal game encodes");
            assert_eq!(tokens.len(), played.san_history.len() + 1);
            assert_eq!(tokens[0], BOS_TOKEN);

            let mut replay = Board::from_fen(START_FEN).unwrap();
            for (ply, &token) in tokens[1..].iter().enumerate() {
                let mv = action_to_move(&replay, usize::from(token))
                    .expect("an emitted token identifies exactly one legal move");
                assert_eq!(
                    replay.san_for(mv),
                    played.san_history[ply],
                    "seed {seed} ply {ply}"
                );
                replay.make_move(mv);
            }
            assert_eq!(replay.san_history, played.san_history);
        }
    }

    #[test]
    fn every_move_token_is_a_policy_action() {
        for seed in 0..8 {
            let tokens = encode_movetext(&random_game(seed, 80).export_san()).unwrap();
            assert!(
                tokens[1..]
                    .iter()
                    .all(|&token| usize::from(token) < POLICY_SIZE)
            );
        }
    }

    #[test]
    fn vocabulary_constants_are_frozen() {
        assert_eq!(BOS_TOKEN, 4_672);
        assert_eq!(PAD_TOKEN, 4_673);
        assert_eq!(VOCAB_SIZE, 4_736);
        assert_eq!(TOKENIZER_VERSION, "policy-v1");
    }

    #[test]
    fn a_known_opening_has_stable_token_ids() {
        // Frozen: retokenizing an existing dataset must not silently change ids.
        assert_eq!(
            encode_movetext("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6").unwrap(),
            vec![BOS_TOKEN, 76, 76, 4_038, 3_585, 3_333, 8]
        );
    }

    #[test]
    fn an_empty_movetext_is_just_the_start_token() {
        assert_eq!(encode_movetext("").unwrap(), vec![BOS_TOKEN]);
    }

    #[test]
    fn illegal_and_malformed_san_name_the_ply() {
        let error = encode_movetext("1. e4 e9").unwrap_err();
        assert!(
            matches!(error, EncodingError::San { ply: 1, .. }),
            "{error}"
        );

        let error = encode_movetext("1. e4 e5 2. e5").unwrap_err();
        assert!(
            matches!(error, EncodingError::San { ply: 2, .. }),
            "{error}"
        );
    }

    #[test]
    fn black_and_white_share_canonical_actions() {
        let mirrored = encode_movetext("1. e4 e5").unwrap();
        assert_eq!(mirrored[1], mirrored[2]);
    }
}
