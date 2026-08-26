//! Turning next-token logits into a move.
//!
//! Legality is owned by `chess_core`: the model only ranks the actions that
//! [`legal_action_mask`] already proved legal, so no logit — including the ones
//! on BOS, PAD, and the unused padding ids — can produce an illegal move.

use alphamini::policy::{PolicyError, action_to_move, legal_action_mask};
use chess_core::{Board, Move};
use rand::Rng;
use thiserror::Error;

use crate::encoding::VOCAB_SIZE;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("the position has no legal moves")]
    Terminal,
    #[error(transparent)]
    Policy(#[from] PolicyError),
    #[error("decode contract violation: {0}")]
    Contract(String),
}

/// The sequence the model may see: BOS plus the most recent moves.
///
/// The graph has exactly `context` position embeddings, so a longer game keeps
/// its start token and drops the oldest moves. `context` is manifest-validated
/// to be at least two.
pub fn truncate_context(tokens: &[u16], context: usize) -> Vec<u16> {
    if tokens.len() <= context {
        return tokens.to_vec();
    }
    let mut kept = Vec::with_capacity(context);
    kept.push(tokens[0]);
    kept.extend_from_slice(&tokens[tokens.len() - (context - 1)..]);
    kept
}

/// Sample a legal move from one position's next-token logits.
///
/// `temperature` of zero is greedy; otherwise the legal logits are divided by it
/// and sampled as a softmax. Illegal actions are dropped rather than set to
/// `-inf`, which is the same distribution without the `exp(-inf)` edge cases.
pub fn choose_move(
    logits: &[f32],
    board: &Board,
    temperature: f32,
    rng: &mut impl Rng,
) -> Result<Move, DecodeError> {
    let action = choose_action(logits, board, temperature, rng)?;
    Ok(action_to_move(board, action)?)
}

fn choose_action(
    logits: &[f32],
    board: &Board,
    temperature: f32,
    rng: &mut impl Rng,
) -> Result<usize, DecodeError> {
    if logits.len() != VOCAB_SIZE {
        return Err(DecodeError::Contract(format!(
            "expected {VOCAB_SIZE} logits, got {}",
            logits.len()
        )));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(DecodeError::Contract(format!(
            "temperature must be finite and non-negative, got {temperature}"
        )));
    }

    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for (action, &is_legal) in legal_action_mask(board)?.iter().enumerate() {
        if !is_legal {
            continue;
        }
        let logit = logits[action];
        if !logit.is_finite() {
            return Err(DecodeError::Contract(format!(
                "the logit for legal action {action} is not finite"
            )));
        }
        candidates.push((action, logit));
    }
    let (best, highest) = candidates
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.1.total_cmp(&right.1))
        .map(|(index, &(_, logit))| (index, logit))
        .ok_or(DecodeError::Terminal)?;
    if temperature == 0.0 {
        return Ok(candidates[best].0);
    }

    // Shift by the maximum before exponentiating so a large logit cannot overflow.
    let mut total = 0.0_f32;
    let weights: Vec<f32> = candidates
        .iter()
        .map(|&(_, logit)| {
            let weight = ((logit - highest) / temperature).exp();
            total += weight;
            weight
        })
        .collect();
    if !total.is_finite() || total <= 0.0 {
        return Err(DecodeError::Contract(format!(
            "the softmax over {} legal actions summed to {total}",
            candidates.len()
        )));
    }
    let target = rng.gen_range(0.0..total);
    let mut cumulative = 0.0_f32;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += weight;
        if target < cumulative {
            return Ok(candidates[index].0);
        }
    }
    // Only reachable when rounding leaves `target` at or past the final sum.
    Ok(candidates[candidates.len() - 1].0)
}

#[cfg(test)]
mod tests {
    use alphamini::START_FEN;
    use alphamini::policy::POLICY_SIZE;
    use chess_core::Status;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::encoding::{BOS_TOKEN, PAD_TOKEN, encode_movetext};

    fn random_position(seed: u64, plies: usize) -> Board {
        let mut board = Board::from_fen(START_FEN).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..plies {
            if board.status() != Status::Ongoing {
                break;
            }
            let legal = board.get_legal_moves();
            board.make_move(legal[rng.gen_range(0..legal.len())]);
        }
        board
    }

    fn random_logits(rng: &mut ChaCha8Rng) -> Vec<f32> {
        (0..VOCAB_SIZE)
            .map(|_| rng.gen_range(-20.0..20.0))
            .collect()
    }

    #[test]
    fn sampling_never_leaves_the_legal_move_set() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
        for seed in 0..24 {
            let board = random_position(seed, (seed as usize * 7) % 60);
            if board.status() != Status::Ongoing {
                continue;
            }
            let legal = board.get_legal_moves();
            for temperature in [0.0, 0.05, 0.5, 1.0, 8.0] {
                for _ in 0..8 {
                    let logits = random_logits(&mut rng);
                    let chosen = choose_move(&logits, &board, temperature, &mut rng).unwrap();
                    assert!(legal.contains(&chosen), "seed {seed} produced {chosen:?}");
                }
            }
        }
    }

    #[test]
    fn zero_temperature_takes_the_highest_legal_logit() {
        let board = Board::from_fen(START_FEN).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut logits = vec![-1.0; VOCAB_SIZE];
        // Reserved ids outrank every move but can never be chosen.
        logits[POLICY_SIZE..].fill(1_000.0);
        logits[encode_movetext("1. Nf3").unwrap()[1] as usize] = 5.0;
        assert_eq!(
            board.san_for(choose_move(&logits, &board, 0.0, &mut rng).unwrap()),
            "Nf3"
        );

        logits[encode_movetext("1. e4").unwrap()[1] as usize] = 5.000_1;
        assert_eq!(
            board.san_for(choose_move(&logits, &board, 0.0, &mut rng).unwrap()),
            "e4"
        );
    }

    #[test]
    fn reserved_and_padding_ids_are_unreachable() {
        let board = Board::import_san("1. e4 e5 2. Nf3").unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut logits = vec![0.0; VOCAB_SIZE];
        logits[POLICY_SIZE..].fill(1_000.0);
        assert!(logits[BOS_TOKEN as usize] > 0.0 && logits[PAD_TOKEN as usize] > 0.0);
        let legal = board.get_legal_moves();
        for temperature in [0.0, 0.2, 1.0] {
            for _ in 0..64 {
                let chosen = choose_move(&logits, &board, temperature, &mut rng).unwrap();
                assert!(legal.contains(&chosen));
            }
        }
    }

    #[test]
    fn a_terminal_position_has_nothing_to_decode() {
        let board = Board::import_san("1. f3 e5 2. g4 Qh4#").unwrap();
        assert_ne!(board.status(), Status::Ongoing);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        assert!(matches!(
            choose_move(&vec![0.0; VOCAB_SIZE], &board, 1.0, &mut rng),
            Err(DecodeError::Terminal)
        ));
    }

    #[test]
    fn malformed_logits_and_temperatures_fail_closed() {
        let board = Board::from_fen(START_FEN).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        assert!(choose_move(&[0.0; 8], &board, 1.0, &mut rng).is_err());
        assert!(choose_move(&vec![0.0; VOCAB_SIZE], &board, -1.0, &mut rng).is_err());
        assert!(choose_move(&vec![0.0; VOCAB_SIZE], &board, f32::NAN, &mut rng).is_err());
        assert!(choose_move(&vec![f32::NAN; VOCAB_SIZE], &board, 1.0, &mut rng).is_err());
    }

    #[test]
    fn truncation_keeps_the_start_token_and_the_newest_moves() {
        let mut tokens = vec![BOS_TOKEN];
        tokens.extend((0..350_usize).map(|ply| (ply * 13 % POLICY_SIZE) as u16));
        assert_eq!(tokens.len(), 351);

        let kept = truncate_context(&tokens, 256);
        assert_eq!(kept.len(), 256);
        assert_eq!(kept[0], BOS_TOKEN);
        assert_eq!(&kept[1..], &tokens[tokens.len() - 255..]);
        assert_eq!(kept.last(), tokens.last());

        // Anything that already fits is passed through unchanged.
        assert_eq!(truncate_context(&tokens[..10], 256), tokens[..10]);
        assert_eq!(truncate_context(&tokens[..256], 256), tokens[..256]);
        assert_eq!(truncate_context(&[BOS_TOKEN], 256), vec![BOS_TOKEN]);
    }
}
