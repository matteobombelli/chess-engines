use chess_core::{Board, Move};
use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// One newline-delimited JSON request to the bot.
///
/// The whole game is sent as PGN movetext and the bot rebuilds the position by
/// replaying it. One field, one source of truth: there is no separate FEN that
/// could disagree with the moves, and no separate "move to apply" that could be
/// applied to the wrong position.
///
/// The cost of this shape is that a game must begin from the standard starting
/// position, since movetext has no way to express any other. That is already
/// what `Board::import_san` assumes.
#[derive(Debug, Serialize, Deserialize)]
pub struct BotRequest {
    /// The game so far, e.g. `"1. e4 e5 2. Nf3"`. Absent or empty means the
    /// game has not started and the bot moves first.
    #[serde(default)]
    pub san: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BotResponse {
    /// The bot's reply, as a single SAN move.
    pub san: String,
    /// The position after it, so the caller can check its own replay agrees.
    pub fen: String,
}

/// Replay the movetext, then play a random legal move.
pub fn respond(request: BotRequest) -> Result<BotResponse, String> {
    let mut board = position_from_request(&request)?;

    let mv = choose_move(&board, &mut rand::thread_rng())
        .ok_or_else(|| "game is over: no legal moves".to_string())?;
    board.make_move(mv);

    Ok(BotResponse {
        san: board
            .san_history
            .last()
            .cloned()
            .expect("move was recorded"),
        fen: board.to_fen(),
    })
}

/// Choose uniformly from all legal moves using the supplied random source.
///
/// Supplying the RNG lets evaluators reproduce a match from its seed while the
/// HTTP bot can continue to use fresh randomness for normal games.
pub fn choose_move<R: Rng + ?Sized>(board: &Board, rng: &mut R) -> Option<Move> {
    candidate_moves(board).choose(rng).copied()
}

fn position_from_request(request: &BotRequest) -> Result<Board, String> {
    Board::import_san(request.san.as_deref().unwrap_or(""))
}

/// Keep promotion variants as distinct candidates so bots can underpromote.
fn candidate_moves(board: &Board) -> Vec<Move> {
    board.get_legal_moves()
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    /// A short legal game ending in an underpromotion to a knight on a8.
    const UNDERPROMOTION: &str = "1. e4 d5 2. exd5 c6 3. dxc6 Nf6 4. cxb7 a6 5. bxa8=N";

    fn request(movetext: &str) -> BotRequest {
        BotRequest {
            san: Some(movetext.to_string()),
        }
    }

    #[test]
    fn bot_can_reply_as_black() {
        let response = respond(request("1. e4")).expect("bot should reply to a legal game");

        let mut expected = Board::import_san("1. e4").unwrap();
        expected
            .san_to_move(&response.san)
            .expect("bot response must be legal SAN");
        assert_eq!(response.fen, expected.to_fen());
        assert_eq!(expected.side_to_move, chess_core::Color::White);
    }

    #[test]
    fn bot_can_make_the_opening_move_as_white() {
        // An absent or empty game means nothing has been played yet, so the bot
        // moves first. This is how the frontend opens when the player is Black.
        for movetext in [None, Some(String::new())] {
            let response = respond(BotRequest { san: movetext }).expect("bot should open the game");

            let mut expected = Board::from_fen(START).unwrap();
            expected
                .san_to_move(&response.san)
                .expect("bot response must be a legal White move");
            assert_eq!(response.fen, expected.to_fen());
            assert_eq!(expected.side_to_move, chess_core::Color::Black);
        }
    }

    #[test]
    fn illegal_input_is_rejected() {
        // Black cannot play e5 as the first move of the game.
        let error = respond(request("1. e5")).unwrap_err();
        assert!(error.contains("illegal"), "unexpected error: {error}");

        // A move that is legal in isolation but not at this point in the game.
        let error = respond(request("1. e4 e5 2. e4")).unwrap_err();
        assert!(error.contains("ply 3"), "unexpected error: {error}");
    }

    #[test]
    fn a_replayed_game_matches_the_moves_that_were_sent() {
        let board = position_from_request(&request("1. e4 e5 2. Nf3 Nc6")).unwrap();
        assert_eq!(board.export_san(), "1. e4 e5 2. Nf3 Nc6");
    }

    #[test]
    fn bot_considers_all_four_promotion_choices() {
        let board = Board::from_fen("4k3/8/8/8/8/8/p7/4K3 b - - 0 1").unwrap();
        let promotions: Vec<_> = candidate_moves(&board)
            .into_iter()
            .filter_map(|mv| mv.promotion)
            .collect();

        assert_eq!(promotions.len(), 4);
        assert!(promotions.contains(&chess_core::PieceKind::Rook));
        assert!(promotions.contains(&chess_core::PieceKind::Bishop));
        assert!(promotions.contains(&chess_core::PieceKind::Knight));
    }

    #[test]
    fn bot_accepts_an_underpromotion_from_its_opponent() {
        let response =
            respond(request(UNDERPROMOTION)).expect("bot should accept a legal underpromotion");

        let mut expected = Board::import_san(UNDERPROMOTION).unwrap();
        expected
            .san_to_move(&response.san)
            .expect("bot response must be legal SAN");
        assert_eq!(response.fen, expected.to_fen());
    }

    #[test]
    fn bot_reports_a_finished_game_rather_than_inventing_a_move() {
        // Fool's mate: White is checkmated and has nothing to play.
        let error = respond(request("1. f3 e5 2. g4 Qh4#")).unwrap_err();
        assert!(error.contains("game is over"), "unexpected error: {error}");
    }
}
