use chess_core::{Board, Color, movetext_moves};
use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::{CHESSCOM_TIME_CLASS, CHESSCOM_TIME_CONTROL, PositionSample};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChessComArchive {
    pub games: Vec<ChessComGame>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChessComPlayer {
    pub username: String,
    pub rating: u32,
    pub result: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChessComGame {
    pub url: String,
    pub pgn: String,
    pub time_control: String,
    pub time_class: String,
    pub rated: bool,
    pub rules: String,
    pub white: ChessComPlayer,
    pub black: ChessComPlayer,
}

impl ChessComGame {
    pub fn is_rated_standard_30_0(&self) -> bool {
        self.rated
            && self.rules == "chess"
            && self.time_class == CHESSCOM_TIME_CLASS
            && self.time_control == CHESSCOM_TIME_CONTROL
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleConfig {
    pub positions_per_side: usize,
    pub min_ply: u16,
    pub max_ply: Option<u16>,
    pub min_rating: u16,
    pub max_rating: u16,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            positions_per_side: 1,
            min_ply: 12,
            max_ply: Some(60),
            min_rating: 200,
            max_rating: 3_200,
        }
    }
}

pub fn sample_game<R: Rng + ?Sized>(
    game: &ChessComGame,
    config: SampleConfig,
    rng: &mut R,
) -> Result<Vec<PositionSample>, String> {
    if !game.is_rated_standard_30_0() {
        return Ok(Vec::new());
    }
    let rating_is_eligible = |rating: u32| {
        u16::try_from(rating)
            .is_ok_and(|rating| (config.min_rating..=config.max_rating).contains(&rating))
    };
    if !rating_is_eligible(game.white.rating) && !rating_is_eligible(game.black.rating) {
        return Ok(Vec::new());
    }

    let movetext = mainline_movetext(&game.pgn)?;
    let mut board = Board::import_san("")?;
    let mut uci_prefix = Vec::new();
    let mut white = Vec::new();
    let mut black = Vec::new();

    for (index, san) in movetext_moves(&movetext).enumerate() {
        let ply = u16::try_from(index + 1).map_err(|_| "game has too many moves".to_string())?;
        let actor = board.side_to_move;
        let player = match actor {
            Color::White => &game.white,
            Color::Black => &game.black,
        };
        let fen = board.to_fen();
        let legal_move_count = board.get_legal_moves().len();
        let actor_rating = u16::try_from(player.rating).ok();
        let in_ply_range = ply >= config.min_ply && config.max_ply.is_none_or(|max| ply <= max);
        let in_rating_range = actor_rating
            .is_some_and(|rating| (config.min_rating..=config.max_rating).contains(&rating));
        let sample_prefix =
            (in_ply_range && in_rating_range && legal_move_count > 1).then(|| uci_prefix.clone());
        let human_move = board
            .san_to_move(san)
            .map_err(|error| format!("{} ply {ply}: {error}", game.url))?;
        let human_move = human_move.to_uci();
        uci_prefix.push(human_move.clone());

        let (Some(actor_rating), Some(sample_prefix)) = (actor_rating, sample_prefix) else {
            continue;
        };

        let sample = PositionSample {
            game_id: game.url.clone(),
            actor_username: player.username.clone(),
            actor_rating,
            ply,
            uci_prefix: sample_prefix,
            fen,
            human_move,
        };
        match actor {
            Color::White => white.push(sample),
            Color::Black => black.push(sample),
        }
    }

    white.shuffle(rng);
    black.shuffle(rng);
    white.truncate(config.positions_per_side);
    black.truncate(config.positions_per_side);
    white.extend(black);
    Ok(white)
}

/// Extract the main line from a Chess.com PGN. Clock comments, NAGs, and
/// parenthesized analysis variations are intentionally excluded.
pub fn mainline_movetext(pgn: &str) -> Result<String, String> {
    let body = pgn
        .lines()
        .filter(|line| !line.trim_start().starts_with('['))
        .collect::<Vec<_>>()
        .join("\n");

    let mut cleaned = String::with_capacity(body.len());
    let mut comment_depth = 0_u32;
    let mut variation_depth = 0_u32;
    let mut semicolon_comment = false;

    for ch in body.chars() {
        if semicolon_comment {
            if ch == '\n' {
                semicolon_comment = false;
                cleaned.push(' ');
            }
            continue;
        }
        if comment_depth > 0 {
            match ch {
                '{' => comment_depth += 1,
                '}' => comment_depth -= 1,
                _ => {}
            }
            continue;
        }
        if variation_depth > 0 {
            match ch {
                '(' => variation_depth += 1,
                ')' => variation_depth -= 1,
                _ => {}
            }
            continue;
        }

        match ch {
            '{' => comment_depth = 1,
            '(' => variation_depth = 1,
            ';' => semicolon_comment = true,
            '\n' | '\r' => cleaned.push(' '),
            _ => cleaned.push(ch),
        }
    }

    if comment_depth != 0 {
        return Err("unterminated PGN comment".to_string());
    }
    if variation_depth != 0 {
        return Err("unterminated PGN variation".to_string());
    }

    Ok(cleaned
        .split_whitespace()
        .filter(|token| !token.starts_with('$'))
        .collect::<Vec<_>>()
        .join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn game() -> ChessComGame {
        ChessComGame {
            url: "https://example.test/game/1".to_string(),
            pgn: "[Event \"Live Chess\"]\n[TimeControl \"1800\"]\n\n\
                  1. e4 {[%clk 0:29:59]} e5 2. Nf3 (2. Bc4) Nc6 \
                  3. Bb5 $1 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 \
                  7. Bb3 d6 8. c3 O-O 9. h3 1/2-1/2"
                .to_string(),
            time_control: "1800".to_string(),
            time_class: "rapid".to_string(),
            rated: true,
            rules: "chess".to_string(),
            white: ChessComPlayer {
                username: "white".to_string(),
                rating: 1_200,
                result: "agreed".to_string(),
            },
            black: ChessComPlayer {
                username: "black".to_string(),
                rating: 1_250,
                result: "agreed".to_string(),
            },
        }
    }

    #[test]
    fn removes_comments_variations_and_nags() {
        let clean = mainline_movetext(&game().pgn).unwrap();
        assert!(!clean.contains("clk"));
        assert!(!clean.contains("Bc4"));
        assert!(!clean.contains("$1"));
        Board::import_san(&clean).unwrap();
    }

    #[test]
    fn samples_one_position_for_each_side() {
        let mut rng = StdRng::seed_from_u64(7);
        let samples = sample_game(
            &game(),
            SampleConfig {
                min_ply: 1,
                ..SampleConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(samples.len(), 2);
        assert_ne!(samples[0].ply % 2, samples[1].ply % 2);
        for sample in samples {
            assert_eq!(sample.uci_prefix.len(), usize::from(sample.ply - 1));
            let replayed = Board::import_uci(&sample.uci_prefix).unwrap();
            assert_eq!(replayed.to_fen(), sample.fen);
            replayed.move_from_uci(&sample.human_move).unwrap();
        }
    }

    #[test]
    fn rejects_the_wrong_time_control() {
        let mut wrong = game();
        wrong.time_control = "600".to_string();
        let mut rng = StdRng::seed_from_u64(1);
        assert!(
            sample_game(&wrong, SampleConfig::default(), &mut rng)
                .unwrap()
                .is_empty()
        );
    }
}
