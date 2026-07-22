use chess_core::{Board, Move};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// One newline-delimited JSON request to the bot.
#[derive(Debug, Serialize, Deserialize)]
pub struct BotRequest {
    pub fen: String,
    /// Optional opponent move to apply before the bot replies.
    pub san: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BotResponse {
    pub san: String,
    pub fen: String,
}

/// Apply an optional SAN move to a FEN position, then play a random legal move.
pub fn respond(request: BotRequest) -> Result<BotResponse, String> {
    let mut board = position_after_request(request)?;

    let mv = *candidate_moves(&board)
        .choose(&mut rand::thread_rng())
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

fn position_after_request(request: BotRequest) -> Result<Board, String> {
    let mut board = Board::from_fen(&request.fen)?;
    if let Some(san) = request.san {
        board.san_to_move(&san)?;
    }
    Ok(board)
}

/// Keep promotion variants as distinct candidates so bots can underpromote.
fn candidate_moves(board: &Board) -> Vec<Move> {
    board.get_legal_moves()
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn bot_can_reply_as_black() {
        let response = respond(BotRequest {
            fen: START.to_string(),
            san: Some("e4".to_string()),
        })
        .expect("bot should reply to a legal move");

        let mut expected = Board::from_fen(START).unwrap();
        expected.san_to_move("e4").unwrap();
        expected
            .san_to_move(&response.san)
            .expect("bot response must be legal SAN");
        assert_eq!(response.fen, expected.to_fen());
    }

    #[test]
    fn bot_can_make_the_opening_move_as_white() {
        let response = respond(BotRequest {
            fen: START.to_string(),
            san: None,
        })
        .expect("bot should play from the side to move");

        let mut expected = Board::from_fen(START).unwrap();
        expected
            .san_to_move(&response.san)
            .expect("bot response must be a legal White move");
        assert_eq!(response.fen, expected.to_fen());
        assert_eq!(expected.side_to_move, chess_core::Color::Black);
    }

    #[test]
    fn illegal_input_is_rejected() {
        let error = respond(BotRequest {
            fen: START.to_string(),
            san: Some("e5".to_string()),
        })
        .unwrap_err();
        assert!(error.contains("illegal"));
    }

    #[test]
    fn bot_considers_all_four_promotion_choices() {
        let board = position_after_request(BotRequest {
            fen: "4k3/8/8/8/8/8/p7/4K3 b - - 0 1".to_string(),
            san: None,
        })
        .unwrap();
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
        let response = respond(BotRequest {
            fen: "4k3/P7/8/8/8/8/8/4K3 w - - 0 1".to_string(),
            san: Some("a8=N".to_string()),
        })
        .expect("bot should accept a legal underpromotion");

        let mut expected = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        expected.san_to_move("a8=N").unwrap();
        expected.san_to_move(&response.san).unwrap();
        assert_eq!(response.fen, expected.to_fen());
    }
}
