use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::time::{Duration, Instant};

use chess_core::{Board, Color, SearchPosition, Status};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::encoding::EncodedPosition;
use crate::evaluator::{Evaluation, Evaluator, EvaluatorError};
use crate::mcts::{Mcts, SearchConfig, SearchError, SearchResult};
use crate::record::{
    GAME_RECORD_VERSION, GameOutcomeV1, GameRecordV1, MAX_SELF_PLAY_PLIES_V1, PolicyVisitV1,
    PositionRecordV1, PromotionV1, TerminationV1,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfPlayConfig {
    pub simulations: u32,
    pub batch_size: usize,
    pub sample_through_ply: u16,
    pub max_plies: u16,
    pub cpuct: f32,
    pub fpu_reduction: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            simulations: 128,
            batch_size: 8,
            sample_through_ply: 30,
            max_plies: 512,
            cpuct: 1.5,
            fpu_reduction: 0.25,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }
}

#[derive(Debug, Error)]
pub enum SelfPlayError {
    #[error(transparent)]
    Search(#[from] SearchError),
    #[error("self-play generated invalid policy: {0}")]
    Policy(String),
    #[error("invalid self-play configuration: {0}")]
    Config(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSchedulerStats {
    /// Number of long-lived game workers in the bounded rolling pool.
    pub worker_count: usize,
    pub inference_batches: u64,
    pub neural_evaluations: u64,
    pub maximum_batch: usize,
    pub requested_batch_capacity: usize,
    pub elapsed_micros: u64,
    pub inference_micros: u64,
    /// Index is realized batch size; value is number of such batches.
    pub batch_histogram: Vec<u64>,
}

impl BatchSchedulerStats {
    pub fn mean_batch_fill(&self) -> f64 {
        if self.inference_batches == 0 || self.requested_batch_capacity == 0 {
            0.0
        } else {
            self.neural_evaluations as f64
                / (self.inference_batches as f64 * self.requested_batch_capacity as f64)
        }
    }
}

/// Run games through a bounded rolling worker pool while one owner batches
/// their inference requests. The pool has at most twice the requested
/// inference capacity, and workers claim the next ordered `(game_id, seed)` as
/// soon as they finish. Each tree asks for one leaf at a time; requests from
/// different games are flattened into batches up to `inference_batch_size`,
/// then split back to the waiting search threads in game-local order.
pub fn play_games_batched(
    games: &[(u64, u64)],
    model_sha256: &str,
    config: SelfPlayConfig,
    inference_batch_size: usize,
    evaluator: &mut dyn Evaluator,
) -> Result<Vec<GameRecordV1>, SelfPlayError> {
    play_games_batched_with_stats(games, model_sha256, config, inference_batch_size, evaluator)
        .map(|(games, _)| games)
}

pub fn play_games_batched_with_stats(
    games: &[(u64, u64)],
    model_sha256: &str,
    config: SelfPlayConfig,
    inference_batch_size: usize,
    evaluator: &mut dyn Evaluator,
) -> Result<(Vec<GameRecordV1>, BatchSchedulerStats), SelfPlayError> {
    play_games_batched_with_stats_impl(
        games,
        model_sha256,
        config,
        inference_batch_size,
        evaluator,
        None,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GameWorkerEvent {
    Started(u64),
    Finished(u64),
}

type GameWorkerObserver = Arc<dyn Fn(GameWorkerEvent) + Send + Sync>;

fn play_games_batched_with_stats_impl(
    games: &[(u64, u64)],
    model_sha256: &str,
    mut config: SelfPlayConfig,
    inference_batch_size: usize,
    evaluator: &mut dyn Evaluator,
    worker_observer: Option<GameWorkerObserver>,
) -> Result<(Vec<GameRecordV1>, BatchSchedulerStats), SelfPlayError> {
    if games.is_empty() || inference_batch_size == 0 {
        return Err(SelfPlayError::Config(
            "batched collection needs games and a positive inference batch".to_string(),
        ));
    }
    let mut game_ids = HashSet::with_capacity(games.len());
    if let Some(duplicate) = games
        .iter()
        .map(|(game_id, _)| *game_id)
        .find(|game_id| !game_ids.insert(*game_id))
    {
        return Err(SelfPlayError::Config(format!(
            "batched collection contains duplicate game ID {duplicate}"
        )));
    }
    let mut expected_games = games.to_vec();
    expected_games.sort_unstable_by_key(|(game_id, _)| *game_id);
    // One pending leaf per game maximizes cross-game diversity and avoids a
    // single tree monopolizing a GPU batch.
    config.batch_size = 1;
    let worker_count = games
        .len()
        .min(inference_batch_size.saturating_mul(2).max(1));
    let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>();
    let (game_tx, game_rx) = mpsc::channel::<(u64, Result<GameRecordV1, String>)>();
    let next_game = AtomicUsize::new(0);
    let started_games = AtomicUsize::new(0);

    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            let request_tx = request_tx.clone();
            let game_tx = game_tx.clone();
            let model_sha256 = model_sha256.to_string();
            let next_game = &next_game;
            let started_games = &started_games;
            let worker_observer = worker_observer.clone();
            scope.spawn(move || {
                let mut proxy = ChannelEvaluator {
                    requests: request_tx,
                };
                loop {
                    let index = next_game.fetch_add(1, Ordering::Relaxed);
                    let Some(&(game_id, seed)) = games.get(index) else {
                        break;
                    };
                    started_games.fetch_add(1, Ordering::Relaxed);
                    if let Some(observer) = &worker_observer {
                        observer(GameWorkerEvent::Started(game_id));
                    }
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        play_game(game_id, seed, &model_sha256, config, &mut proxy)
                    }))
                    .map_err(|_| "self-play worker panicked".to_string())
                    .and_then(|result| result.map_err(|error| error.to_string()));
                    if let Some(observer) = &worker_observer {
                        observer(GameWorkerEvent::Finished(game_id));
                    }
                    if game_tx.send((game_id, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(request_tx);
        drop(game_tx);

        let mut completed = 0;
        let mut results = Vec::with_capacity(games.len());
        let mut first_error = None;
        let mut stats = BatchSchedulerStats {
            worker_count,
            requested_batch_capacity: inference_batch_size,
            batch_histogram: vec![0; inference_batch_size + 1],
            ..BatchSchedulerStats::default()
        };
        let mut last_heartbeat = Instant::now();
        while completed < games.len() {
            while let Ok((_, result)) = game_rx.try_recv() {
                completed += 1;
                match result {
                    Ok(game) => results.push(game),
                    Err(error) if first_error.is_none() => first_error = Some(error),
                    Err(_) => {}
                }
            }
            if completed == games.len() {
                break;
            }
            if last_heartbeat.elapsed() >= Duration::from_secs(30) {
                let elapsed = started.elapsed().as_secs_f64();
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "event": "self_play_heartbeat",
                        "games_started": started_games.load(Ordering::Relaxed).min(games.len()),
                        "games_completed": completed,
                        "games_total": games.len(),
                        "worker_count": worker_count,
                        "elapsed_seconds": elapsed,
                        "inference_batches": stats.inference_batches,
                        "neural_evaluations": stats.neural_evaluations,
                        "maximum_batch": stats.maximum_batch,
                        "mean_batch_fill": stats.mean_batch_fill(),
                    })
                );
                last_heartbeat = Instant::now();
            }

            let first = match request_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(request) => request,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    match game_rx.recv_timeout(Duration::from_millis(10)) {
                        Ok((_, result)) => {
                            completed += 1;
                            match result {
                                Ok(game) => results.push(game),
                                Err(error) if first_error.is_none() => first_error = Some(error),
                                Err(_) => {}
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            return Err(SelfPlayError::Policy(
                                "self-play workers disconnected before reporting completion"
                                    .to_string(),
                            ));
                        }
                    }
                    continue;
                }
            };
            let mut requests = vec![first];
            let mut position_count = requests[0].positions.len();
            while position_count < inference_batch_size {
                match request_rx.recv_timeout(Duration::from_micros(200)) {
                    Ok(request) => {
                        position_count += request.positions.len();
                        requests.push(request);
                    }
                    Err(_) => break,
                }
            }
            stats.inference_batches += 1;
            stats.neural_evaluations += position_count as u64;
            stats.maximum_batch = stats.maximum_batch.max(position_count);
            if position_count >= stats.batch_histogram.len() {
                stats.batch_histogram.resize(position_count + 1, 0);
            }
            stats.batch_histogram[position_count] += 1;
            let inference_started = Instant::now();
            dispatch_inference(evaluator, requests);
            stats.inference_micros += inference_started
                .elapsed()
                .as_micros()
                .min(u64::MAX as u128) as u64;
        }

        if let Some(error) = first_error {
            return Err(SelfPlayError::Policy(format!(
                "concurrent self-play worker failed: {error}"
            )));
        }
        results.sort_by_key(|game| game.game_id);
        let actual_games: Vec<_> = results
            .iter()
            .map(|game| (game.game_id, game.seed))
            .collect();
        if actual_games != expected_games {
            return Err(SelfPlayError::Policy(format!(
                "rolling worker pool returned the wrong game aggregate: expected {expected_games:?}, got {actual_games:?}"
            )));
        }
        stats.elapsed_micros = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        Ok((results, stats))
    })
}

struct InferenceRequest {
    positions: Vec<EncodedPosition>,
    response: SyncSender<Result<Vec<Evaluation>, String>>,
}

struct ChannelEvaluator {
    requests: mpsc::Sender<InferenceRequest>,
}

impl Evaluator for ChannelEvaluator {
    fn evaluate_batch(
        &mut self,
        positions: &[EncodedPosition],
    ) -> Result<Vec<Evaluation>, EvaluatorError> {
        self.evaluate_owned_batch(positions.to_vec())
    }

    fn evaluate_owned_batch(
        &mut self,
        positions: Vec<EncodedPosition>,
    ) -> Result<Vec<Evaluation>, EvaluatorError> {
        let (response, receive) = mpsc::sync_channel(1);
        self.requests
            .send(InferenceRequest {
                positions,
                response,
            })
            .map_err(|_| EvaluatorError::Runtime("inference scheduler stopped".to_string()))?;
        receive
            .recv()
            .map_err(|_| EvaluatorError::Runtime("inference response was dropped".to_string()))?
            .map_err(EvaluatorError::Runtime)
    }
}

fn dispatch_inference(evaluator: &mut dyn Evaluator, mut requests: Vec<InferenceRequest>) {
    let counts: Vec<_> = requests
        .iter()
        .map(|request| request.positions.len())
        .collect();
    let position_count = counts.iter().sum();
    let mut positions = Vec::with_capacity(position_count);
    for request in &mut requests {
        positions.append(&mut request.positions);
    }
    match evaluator.evaluate_owned_batch(positions) {
        Ok(evaluations) if evaluations.len() == position_count => {
            let mut evaluations = evaluations.into_iter();
            for (request, count) in requests.into_iter().zip(counts) {
                let response = evaluations.by_ref().take(count).collect();
                let _ = request.response.send(Ok(response));
            }
            debug_assert!(evaluations.next().is_none());
        }
        Ok(evaluations) => {
            let message = format!(
                "central evaluator returned {} rows for {} positions",
                evaluations.len(),
                position_count
            );
            for request in requests {
                let _ = request.response.send(Err(message.clone()));
            }
        }
        Err(error) => {
            let message = error.to_string();
            for request in requests {
                let _ = request.response.send(Err(message.clone()));
            }
        }
    }
}

pub fn play_game(
    game_id: u64,
    seed: u64,
    model_sha256: &str,
    config: SelfPlayConfig,
    evaluator: &mut dyn Evaluator,
) -> Result<GameRecordV1, SelfPlayError> {
    if config.max_plies == 0
        || config.max_plies > MAX_SELF_PLAY_PLIES_V1
        || config.simulations == 0
        || config.batch_size == 0
    {
        return Err(SelfPlayError::Config(format!(
            "max_plies must be in 1..={MAX_SELF_PLAY_PLIES_V1}; simulations and batch_size must be positive"
        )));
    }
    let board = Board::from_fen(crate::START_FEN).expect("frozen start FEN is valid");
    let mut position = SearchPosition::from_board(&board);
    let search = Mcts::new(SearchConfig {
        simulations: config.simulations,
        batch_size: config.batch_size,
        cpuct: config.cpuct,
        fpu_reduction: config.fpu_reduction,
        root_dirichlet_alpha: Some(config.dirichlet_alpha),
        root_noise_fraction: config.dirichlet_epsilon,
        move_time: None,
    })?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut positions = Vec::new();
    let mut previous_move = None;

    let (outcome, termination) = loop {
        let ply = positions.len() as u16;
        if ply >= config.max_plies {
            break (GameOutcomeV1::Draw, TerminationV1::PlyLimit);
        }
        let result = search.search_position(&mut position, evaluator, &mut rng)?;
        let chosen = choose_move(&result, ply < config.sample_through_ply, &mut rng)?;
        let policy = sparse_policy(&result);
        let selected_move_uci = chosen.to_uci();
        positions.push(PositionRecordV1::from_search_position(
            &position,
            game_id,
            ply,
            previous_move.clone(),
            selected_move_uci.clone(),
            policy,
            GameOutcomeV1::Draw,
            TerminationV1::PlyLimit,
        ));
        previous_move = Some(selected_move_uci);
        let _committed = position.make_move(chosen);

        let mut legal_moves = Vec::new();
        position.legal_moves_into(&mut legal_moves);
        match position.status_with_legal_moves(&legal_moves) {
            Status::Ongoing => {}
            Status::Checkmate => {
                let winner = position.side_to_move().opposite();
                break (
                    if winner == Color::White {
                        GameOutcomeV1::WhiteWin
                    } else {
                        GameOutcomeV1::BlackWin
                    },
                    TerminationV1::Checkmate,
                );
            }
            Status::Stalemate => break (GameOutcomeV1::Draw, TerminationV1::Stalemate),
            Status::InsufficientMaterial => {
                break (GameOutcomeV1::Draw, TerminationV1::InsufficientMaterial);
            }
            Status::ThreefoldRepetition => {
                break (GameOutcomeV1::Draw, TerminationV1::ThreefoldRepetition);
            }
            Status::FiftyMoveRule => {
                break (GameOutcomeV1::Draw, TerminationV1::FiftyMoveRule);
            }
        }
    };

    for position in &mut positions {
        position.outcome = outcome;
        position.termination = termination;
    }
    Ok(GameRecordV1 {
        schema: GAME_RECORD_VERSION.to_string(),
        game_id,
        seed,
        model_sha256: model_sha256.to_string(),
        outcome,
        termination,
        plies: positions.len() as u16,
        positions,
    })
}

fn sparse_policy(result: &SearchResult) -> Vec<PolicyVisitV1> {
    result
        .action_visits
        .iter()
        .filter(|target| target.visits > 0)
        .map(|target| PolicyVisitV1 {
            from: target.mv.start_square.0,
            to: target.mv.end_square.0,
            promotion: target.mv.promotion.map(PromotionV1::from),
            visits: target.visits,
        })
        .collect()
}

fn choose_move<R: Rng + ?Sized>(
    result: &SearchResult,
    sample: bool,
    rng: &mut R,
) -> Result<chess_core::Move, SelfPlayError> {
    if sample {
        let total = result
            .action_visits
            .iter()
            .map(|target| target.visits as u64)
            .sum::<u64>();
        if total == 0 {
            return Ok(result.best_move);
        }
        let mut draw = rng.gen_range(0..total);
        for target in &result.action_visits {
            if draw < target.visits as u64 {
                return Ok(target.mv);
            }
            draw -= target.visits as u64;
        }
        unreachable!("sample draw is strictly below the total root visits");
    } else {
        result
            .action_visits
            .iter()
            .max_by_key(|target| (target.visits, std::cmp::Reverse(target.action)))
            .map(|target| target.mv)
            .ok_or_else(|| SelfPlayError::Policy("search returned no root actions".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Condvar, Mutex};

    use super::*;
    use crate::encoding::encode_search_current;
    use crate::evaluator::UniformEvaluator;
    use crate::policy::action_to_search_move;

    fn decoded_policy(position: &mut SearchPosition, result: &SearchResult) -> Vec<PolicyVisitV1> {
        result
            .action_visits
            .iter()
            .filter(|target| target.visits > 0)
            .map(|target| {
                let mv = action_to_search_move(position, target.action as usize).unwrap();
                PolicyVisitV1 {
                    from: mv.start_square.0,
                    to: mv.end_square.0,
                    promotion: mv.promotion.map(PromotionV1::from),
                    visits: target.visits,
                }
            })
            .collect()
    }

    fn decoded_choice<R: Rng + ?Sized>(
        position: &mut SearchPosition,
        result: &SearchResult,
        sample: bool,
        rng: &mut R,
    ) -> chess_core::Move {
        let action = if sample {
            let total = result
                .action_visits
                .iter()
                .map(|target| target.visits as u64)
                .sum::<u64>();
            if total == 0 {
                return result.best_move;
            }
            let mut draw = rng.gen_range(0..total);
            let mut selected = result.action_visits[0].action;
            for target in &result.action_visits {
                if draw < target.visits as u64 {
                    selected = target.action;
                    break;
                }
                draw -= target.visits as u64;
            }
            selected
        } else {
            result
                .action_visits
                .iter()
                .max_by_key(|target| (target.visits, std::cmp::Reverse(target.action)))
                .unwrap()
                .action
        };
        action_to_search_move(position, action as usize).unwrap()
    }

    #[test]
    fn ply_cap_is_an_explicit_draw_target() {
        let mut evaluator = UniformEvaluator;
        let game = play_game(
            1,
            42,
            &"0".repeat(64),
            SelfPlayConfig {
                simulations: 2,
                batch_size: 2,
                sample_through_ply: 1,
                max_plies: 2,
                ..SelfPlayConfig::default()
            },
            &mut evaluator,
        )
        .unwrap();
        assert_eq!(game.termination, TerminationV1::PlyLimit);
        assert_eq!(game.outcome, GameOutcomeV1::Draw);
        assert_eq!(game.positions.len(), 2);
        assert!(game.positions.iter().all(|position| {
            position.outcome == GameOutcomeV1::Draw
                && position.termination == TerminationV1::PlyLimit
        }));
    }

    #[test]
    fn concurrent_games_share_central_inference_batches() {
        struct BatchSpy {
            largest: Arc<Mutex<usize>>,
        }
        impl Evaluator for BatchSpy {
            fn evaluate_batch(
                &mut self,
                positions: &[EncodedPosition],
            ) -> Result<Vec<Evaluation>, EvaluatorError> {
                let mut largest = self.largest.lock().unwrap();
                *largest = (*largest).max(positions.len());
                drop(largest);
                UniformEvaluator.evaluate_batch(positions)
            }
        }

        let largest = Arc::new(Mutex::new(0));
        let mut evaluator = BatchSpy {
            largest: largest.clone(),
        };
        let games = play_games_batched(
            &[(10, 1), (11, 2), (12, 3), (13, 4)],
            &"0".repeat(64),
            SelfPlayConfig {
                simulations: 2,
                max_plies: 1,
                ..SelfPlayConfig::default()
            },
            4,
            &mut evaluator,
        )
        .unwrap();
        assert_eq!(
            games.iter().map(|game| game.game_id).collect::<Vec<_>>(),
            vec![10, 11, 12, 13]
        );
        assert!(*largest.lock().unwrap() > 1);
    }

    #[test]
    fn rolling_pool_refills_before_the_initial_cohort_drains() {
        let events = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
        let observer_events = events.clone();
        let observer: GameWorkerObserver = Arc::new(move |event| {
            let (events, changed) = &*observer_events;
            let mut events = events.lock().unwrap();
            events.push(event);
            changed.notify_all();

            // With batch capacity one, IDs 30 and 10 are the two initial
            // workers. Keep ID 10 alive until the worker finishing ID 30 has
            // claimed the ordered refill (ID 20).
            if event == GameWorkerEvent::Started(10) {
                while !events.contains(&GameWorkerEvent::Started(20)) {
                    events = changed.wait(events).unwrap();
                }
            }
        });
        let specs = [(30, 101), (10, 102), (20, 103)];
        let mut evaluator = UniformEvaluator;
        let (games, stats) = play_games_batched_with_stats_impl(
            &specs,
            &"0".repeat(64),
            SelfPlayConfig {
                simulations: 1,
                max_plies: 1,
                sample_through_ply: 0,
                ..SelfPlayConfig::default()
            },
            1,
            &mut evaluator,
            Some(observer),
        )
        .unwrap();

        assert_eq!(stats.worker_count, 2);
        assert_eq!(
            games
                .iter()
                .map(|game| (game.game_id, game.seed))
                .collect::<Vec<_>>(),
            vec![(10, 102), (20, 103), (30, 101)]
        );
        assert_eq!(
            games.iter().map(|game| game.positions.len()).sum::<usize>(),
            specs.len()
        );
        assert_eq!(
            games
                .iter()
                .map(|game| game.game_id)
                .collect::<HashSet<_>>()
                .len(),
            specs.len()
        );

        let events = events.0.lock().unwrap();
        let refill_started = events
            .iter()
            .position(|event| *event == GameWorkerEvent::Started(20))
            .unwrap();
        let initial_worker_finished = events
            .iter()
            .position(|event| *event == GameWorkerEvent::Finished(10))
            .unwrap();
        assert!(refill_started < initial_worker_finished);
    }

    #[test]
    fn rolling_pool_rejects_duplicate_game_ids_before_search() {
        let mut evaluator = UniformEvaluator;
        let error = play_games_batched(
            &[(7, 11), (7, 12)],
            &"0".repeat(64),
            SelfPlayConfig::default(),
            2,
            &mut evaluator,
        )
        .unwrap_err();
        assert!(
            matches!(error, SelfPlayError::Config(message) if message.contains("duplicate game ID 7"))
        );
    }

    #[test]
    fn root_moves_preserve_decoded_policy_and_selection_semantics() {
        let board = Board::from_fen(crate::START_FEN).unwrap();
        let mut position = SearchPosition::from_board(&board);
        let mut evaluator = UniformEvaluator;
        let mut search_rng = ChaCha8Rng::seed_from_u64(19);
        let result = Mcts::new(SearchConfig {
            simulations: 31,
            batch_size: 4,
            ..SearchConfig::default()
        })
        .unwrap()
        .search_position(&mut position, &mut evaluator, &mut search_rng)
        .unwrap();

        assert_eq!(
            result
                .action_visits
                .iter()
                .map(|target| target.visits)
                .sum::<u32>(),
            31
        );
        assert_eq!(
            sparse_policy(&result),
            decoded_policy(&mut position, &result)
        );

        let mut optimized_rng = ChaCha8Rng::seed_from_u64(23);
        let mut decoded_rng = ChaCha8Rng::seed_from_u64(23);
        for _ in 0..32 {
            assert_eq!(
                choose_move(&result, true, &mut optimized_rng).unwrap(),
                decoded_choice(&mut position, &result, true, &mut decoded_rng)
            );
        }
        assert_eq!(
            choose_move(&result, false, &mut optimized_rng).unwrap(),
            decoded_choice(&mut position, &result, false, &mut decoded_rng)
        );
    }

    #[test]
    fn central_dispatch_moves_encoded_payloads_without_cloning() {
        struct OwnedPointerSpy {
            seen: Arc<Mutex<Vec<usize>>>,
        }
        impl Evaluator for OwnedPointerSpy {
            fn evaluate_batch(
                &mut self,
                _positions: &[EncodedPosition],
            ) -> Result<Vec<Evaluation>, EvaluatorError> {
                panic!("central dispatch must use the owned evaluator path")
            }

            fn evaluate_owned_batch(
                &mut self,
                positions: Vec<EncodedPosition>,
            ) -> Result<Vec<Evaluation>, EvaluatorError> {
                *self.seen.lock().unwrap() = positions
                    .iter()
                    .map(|position| position.values.as_ptr() as usize)
                    .collect();
                UniformEvaluator.evaluate_batch(&positions)
            }
        }

        let board = Board::from_fen(crate::START_FEN).unwrap();
        let position = SearchPosition::from_board(&board);
        let first = encode_search_current(&position);
        let mut second_position = position.clone();
        let first_move = second_position.legal_moves()[0];
        let _undo = second_position.make_move(first_move);
        let second = encode_search_current(&second_position);
        let expected_pointers = vec![
            first.values.as_ptr() as usize,
            second.values.as_ptr() as usize,
        ];
        let (first_tx, first_rx) = mpsc::sync_channel(1);
        let (second_tx, second_rx) = mpsc::sync_channel(1);
        let requests = vec![
            InferenceRequest {
                positions: vec![first],
                response: first_tx,
            },
            InferenceRequest {
                positions: vec![second],
                response: second_tx,
            },
        ];
        let seen = Arc::new(Mutex::new(Vec::new()));
        dispatch_inference(&mut OwnedPointerSpy { seen: seen.clone() }, requests);

        assert_eq!(*seen.lock().unwrap(), expected_pointers);
        assert_eq!(first_rx.recv().unwrap().unwrap().len(), 1);
        assert_eq!(second_rx.recv().unwrap().unwrap().len(), 1);
    }
}
