use arena::Engine;
use chess_core::Board;
use std::collections::HashMap;

use crate::stockfish::Stockfish;
use crate::{AnalysisRow, PositionSample};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvaluationConfig {
    /// Ignore positions that are effectively decided even with best play.
    pub minimum_best_expected_score: f64,
    pub maximum_best_expected_score: f64,
    /// Keep at most this many informative rows for one human player.
    pub maximum_rows_per_player: usize,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            minimum_best_expected_score: 0.05,
            maximum_best_expected_score: 0.95,
            maximum_rows_per_player: 1,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvaluationResult {
    pub rows: Vec<AnalysisRow>,
    pub skipped_uninformative: usize,
    pub skipped_player_cap: usize,
}

pub fn evaluate_samples<F>(
    samples: &[PositionSample],
    bot: &mut dyn Engine,
    stockfish: &mut Stockfish,
    config: EvaluationConfig,
    mut progress: F,
) -> Result<EvaluationResult, String>
where
    F: FnMut(usize, usize),
{
    if config.maximum_rows_per_player == 0 {
        return Err("maximum rows per player must be greater than zero".to_string());
    }
    if !(0.0..=1.0).contains(&config.minimum_best_expected_score)
        || !(0.0..=1.0).contains(&config.maximum_best_expected_score)
        || config.minimum_best_expected_score >= config.maximum_best_expected_score
    {
        return Err("expected-score bounds must satisfy 0 <= minimum < maximum <= 1".to_string());
    }

    let mut result = EvaluationResult::default();
    let mut player_rows: HashMap<String, usize> = HashMap::new();
    for (index, sample) in samples.iter().enumerate() {
        let player = sample.actor_username.to_ascii_lowercase();
        if player_rows.get(&player).copied().unwrap_or(0) >= config.maximum_rows_per_player {
            result.skipped_player_cap += 1;
            progress(index + 1, samples.len());
            continue;
        }
        if let Some(row) = evaluate_sample(sample, bot, stockfish, config)? {
            result.rows.push(row);
            *player_rows.entry(player).or_default() += 1;
        } else {
            result.skipped_uninformative += 1;
        }
        progress(index + 1, samples.len());
    }
    Ok(result)
}

pub fn evaluate_sample(
    sample: &PositionSample,
    bot: &mut dyn Engine,
    stockfish: &mut Stockfish,
    config: EvaluationConfig,
) -> Result<Option<AnalysisRow>, String> {
    let board = Board::from_fen(&sample.fen)
        .map_err(|error| format!("bad sample FEN in {}: {error}", sample.game_id))?;
    board
        .move_from_uci(&sample.human_move)
        .map_err(|error| format!("bad human move in {}: {error}", sample.game_id))?;

    // First discover Stockfish's reference move, then force that move in a
    // second search. Human, bot, and reference moves consequently receive the
    // same node budget instead of the reference score sharing its budget across
    // every root move.
    let reference_move = stockfish.analyze(&sample.fen, None)?.best_move;
    let best_expected_score = stockfish.expected_score(&sample.fen, Some(&reference_move))?;
    if best_expected_score < config.minimum_best_expected_score
        || best_expected_score > config.maximum_best_expected_score
    {
        return Ok(None);
    }

    // Choose the bot move only after the cheap reference screen. This matters
    // for time-limited engines: a discarded position should not consume its
    // full move budget.
    let bot_move = bot
        .choose_move(&board)
        .map_err(|error| format!("{} failed in {}: {error}", bot.name(), sample.game_id))?;
    if !board.get_legal_moves().contains(&bot_move) {
        return Err(format!(
            "{} returned illegal move {:?} in {}",
            bot.name(),
            bot_move,
            sample.game_id
        ));
    }
    let bot_move = bot_move.to_uci();

    let human_expected_score = if sample.human_move == reference_move {
        best_expected_score
    } else {
        stockfish.expected_score(&sample.fen, Some(&sample.human_move))?
    };
    let bot_expected_score = if bot_move == reference_move {
        best_expected_score
    } else if bot_move == sample.human_move {
        human_expected_score
    } else {
        stockfish.expected_score(&sample.fen, Some(&bot_move))?
    };

    Ok(Some(AnalysisRow {
        game_id: sample.game_id.clone(),
        actor_username: sample.actor_username.clone(),
        actor_rating: sample.actor_rating,
        ply: sample.ply,
        fen: sample.fen.clone(),
        human_move: sample.human_move.clone(),
        bot_move,
        reference_move,
        best_expected_score,
        human_expected_score,
        bot_expected_score,
        // Independent finite-node searches can occasionally score a candidate
        // above the selected reference move. Clamp that search noise to zero.
        human_loss: (best_expected_score - human_expected_score).max(0.0),
        bot_loss: (best_expected_score - bot_expected_score).max(0.0),
    }))
}
