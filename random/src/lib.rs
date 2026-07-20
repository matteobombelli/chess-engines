use chess_core::Board;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// One newline-delimited JSON request to the bot.
#[derive(Debug, Deserialize)]
pub struct BotRequest {
    pub fen: String,
    /// Optional opponent move to apply before the bot replies.
    pub san: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BotResponse {
    pub san: String,
    pub fen: String,
}

/// Apply an optional SAN move to a FEN position, then play a random legal move.
pub fn respond(request: BotRequest) -> Result<BotResponse, String> {
    let mut board = Board::from_fen(&request.fen)?;
    if let Some(san) = request.san {
        board.san_to_move(&san)?;
    }

    let mv = *board
        .get_legal_moves()
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

#[cfg(test)]
mod tests {
    use super::*;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn fen_in_san_in_legal_san_and_fen_out() {
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
    fn illegal_input_is_rejected() {
        let error = respond(BotRequest {
            fen: START.to_string(),
            san: Some("e5".to_string()),
        })
        .unwrap_err();
        assert!(error.contains("illegal"));
    }
}
