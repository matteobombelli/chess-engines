use std::cmp::Ordering;
use std::time::{Duration, Instant};

use chess_core::{Board, Move, SearchPosition, SearchUndo, Status};
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use thiserror::Error;

use crate::encoding::{EncodedPosition, encode_search_current};
use crate::evaluator::{Evaluation, Evaluator, EvaluatorError};
use crate::manifest::{
    FROZEN_GATE_BATCH_SIZE, FROZEN_GATE_CPUCT_PPM, FROZEN_GATE_FPU_REDUCTION_PPM,
    FROZEN_GATE_SIMULATIONS, FROZEN_GATE_TIME_MS,
};
use crate::policy::{PolicyError, move_to_action};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    pub simulations: u32,
    pub batch_size: usize,
    pub cpuct: f32,
    pub fpu_reduction: f32,
    pub root_dirichlet_alpha: Option<f32>,
    pub root_noise_fraction: f32,
    /// Checked between completed inference batches; root inference is mandatory.
    pub move_time: Option<Duration>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 128,
            batch_size: 8,
            cpuct: ppm_as_f32(FROZEN_GATE_CPUCT_PPM),
            fpu_reduction: ppm_as_f32(FROZEN_GATE_FPU_REDUCTION_PPM),
            root_dirichlet_alpha: None,
            root_noise_fraction: 0.0,
            move_time: None,
        }
    }
}

impl SearchConfig {
    /// Deterministic, noise-free evaluation search. PUCT and FPU remain part
    /// of the frozen evaluation identity even when resource limits vary for an
    /// exploratory rung.
    pub fn evaluation(simulations: u32, batch_size: usize, move_time: Duration) -> Self {
        Self {
            simulations,
            batch_size,
            move_time: Some(move_time),
            ..Self::default()
        }
    }

    /// Exact search contract accepted by a production gate verdict.
    pub fn frozen_gate() -> Self {
        Self::evaluation(
            FROZEN_GATE_SIMULATIONS,
            FROZEN_GATE_BATCH_SIZE,
            Duration::from_millis(FROZEN_GATE_TIME_MS),
        )
    }

    pub fn self_play(simulations: u32, batch_size: usize) -> Self {
        Self {
            simulations,
            batch_size,
            root_dirichlet_alpha: Some(0.3),
            root_noise_fraction: 0.25,
            ..Self::default()
        }
    }

    pub fn validate(self) -> Result<(), SearchError> {
        if self.simulations == 0 || self.batch_size == 0 {
            return Err(SearchError::Config(
                "simulations and batch_size must be greater than zero".to_string(),
            ));
        }
        if !self.cpuct.is_finite() || self.cpuct <= 0.0 {
            return Err(SearchError::Config(
                "cpuct must be finite and positive".to_string(),
            ));
        }
        if !self.fpu_reduction.is_finite() || self.fpu_reduction < 0.0 {
            return Err(SearchError::Config(
                "fpu_reduction must be finite and non-negative".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.root_noise_fraction) {
            return Err(SearchError::Config(
                "root_noise_fraction must be in [0,1]".to_string(),
            ));
        }
        if self.root_noise_fraction > 0.0
            && self
                .root_dirichlet_alpha
                .is_none_or(|alpha| !alpha.is_finite() || alpha <= 0.0)
        {
            return Err(SearchError::Config(
                "positive root noise requires positive finite Dirichlet alpha".to_string(),
            ));
        }
        if self.move_time.is_some_and(|duration| duration.is_zero()) {
            return Err(SearchError::Config(
                "move_time must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

const fn ppm_as_f32(value: u32) -> f32 {
    value as f32 / 1_000_000.0
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub completed_simulations: u32,
    pub inference_batches: u32,
    pub neural_evaluations: u32,
    pub largest_batch: usize,
    pub elapsed_micros: u64,
    pub deadline_reached: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionVisit {
    /// Legal root move associated with `action`. Keeping the move alongside
    /// its statistics lets self-play serialize and sample the root directly,
    /// without regenerating the legal move list for every action.
    pub mv: Move,
    pub action: u16,
    pub visits: u32,
    /// Mean backed-up value from the root side-to-move perspective.
    pub mean_value: f32,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best_move: Move,
    pub root_value: f32,
    pub action_visits: Vec<ActionVisit>,
    pub stats: SearchStats,
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("invalid search configuration: {0}")]
    Config(String),
    #[error("cannot search terminal position: {0:?}")]
    Terminal(Status),
    #[error(transparent)]
    Evaluation(#[from] EvaluatorError),
    #[error(transparent)]
    Policy(#[from] PolicyError),
    #[error("inference returned {actual} evaluations for a batch of {expected}")]
    BatchCardinality { expected: usize, actual: usize },
}

pub struct Mcts {
    config: SearchConfig,
}

impl Mcts {
    pub fn new(config: SearchConfig) -> Result<Self, SearchError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> SearchConfig {
        self.config
    }

    pub fn search<R: Rng + ?Sized>(
        &self,
        board: &Board,
        evaluator: &mut dyn Evaluator,
        rng: &mut R,
    ) -> Result<SearchResult, SearchError> {
        let mut position = SearchPosition::from_board(board);
        self.search_position(&mut position, evaluator, rng)
    }

    /// Search an already-constructed reversible position. Every speculative
    /// move is unmade before return, so callers may commit only the chosen move
    /// and reuse the same position for the next ply.
    pub fn search_position<R: Rng + ?Sized>(
        &self,
        position: &mut SearchPosition,
        evaluator: &mut dyn Evaluator,
        rng: &mut R,
    ) -> Result<SearchResult, SearchError> {
        let started = Instant::now();
        let mut root_moves = Vec::new();
        position.legal_moves_into(&mut root_moves);
        let root_status = position.status_with_legal_moves(&root_moves);
        if root_status != Status::Ongoing {
            return Err(SearchError::Terminal(root_status));
        }

        let root_input = encode_search_current(position);
        let mut initial = evaluator.evaluate_owned_batch(vec![root_input])?;
        if initial.len() != 1 {
            return Err(SearchError::BatchCardinality {
                expected: 1,
                actual: initial.len(),
            });
        }
        let root_evaluation = initial.pop().expect("length checked");
        root_evaluation.validate()?;

        let mut nodes = vec![Node::new()];
        expand_node(
            &mut nodes[0],
            root_moves,
            position.side_to_move(),
            &root_evaluation,
        )?;
        self.apply_root_noise(&mut nodes[0], rng);

        let mut stats = SearchStats {
            inference_batches: 1,
            neural_evaluations: 1,
            largest_batch: 1,
            ..SearchStats::default()
        };

        while stats.completed_simulations < self.config.simulations {
            if self
                .config
                .move_time
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                stats.deadline_reached = true;
                break;
            }
            let remaining = (self.config.simulations - stats.completed_simulations) as usize;
            let target_batch = remaining.min(self.config.batch_size);
            let completed_at_batch_start = stats.completed_simulations;
            let mut inputs = Vec::with_capacity(target_batch);
            let mut pending = Vec::with_capacity(target_batch);

            while (stats.completed_simulations - completed_at_batch_start) as usize + pending.len()
                < target_batch
            {
                match self.select(&mut nodes, position)? {
                    Selection::Leaf(leaf) => {
                        let PendingLeaf {
                            node,
                            path,
                            input,
                            legal_moves,
                            side_to_move,
                        } = leaf;
                        inputs.push(input);
                        pending.push(PendingExpansion {
                            node,
                            path,
                            legal_moves,
                            side_to_move,
                        });
                    }
                    Selection::Terminal { path, value } => {
                        backup(&mut nodes, &path, value);
                        stats.completed_simulations += 1;
                    }
                    Selection::Collision => break,
                }
                if stats.completed_simulations >= self.config.simulations {
                    break;
                }
            }

            if pending.is_empty() {
                // Only possible if terminal backups consumed the remaining budget.
                continue;
            }
            let evaluations = evaluator.evaluate_owned_batch(inputs)?;
            if evaluations.len() != pending.len() {
                return Err(SearchError::BatchCardinality {
                    expected: pending.len(),
                    actual: evaluations.len(),
                });
            }
            stats.inference_batches += 1;
            stats.neural_evaluations += evaluations.len() as u32;
            stats.largest_batch = stats.largest_batch.max(evaluations.len());

            for (leaf, evaluation) in pending.into_iter().zip(evaluations) {
                evaluation.validate()?;
                let value = evaluation.scalar_value();
                let side = leaf.side_to_move;
                let node = &mut nodes[leaf.node];
                node.pending = false;
                expand_node(node, leaf.legal_moves, side, &evaluation)?;
                backup(&mut nodes, &leaf.path, value);
                stats.completed_simulations += 1;
            }
        }

        let root = &nodes[0];
        let chosen = if root.edges.iter().all(|edge| edge.visits == 0) {
            root.edges
                .iter()
                .max_by(|left, right| {
                    left.prior
                        .partial_cmp(&right.prior)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| right.action.cmp(&left.action))
                })
                .expect("ongoing root has legal moves")
        } else {
            root.edges
                .iter()
                .max_by(|left, right| {
                    left.visits
                        .cmp(&right.visits)
                        .then_with(|| right.action.cmp(&left.action))
                })
                .expect("ongoing root has legal moves")
        };
        let action_visits = root
            .edges
            .iter()
            .map(|edge| ActionVisit {
                mv: edge.mv,
                action: edge.action as u16,
                visits: edge.visits,
                mean_value: if edge.visits == 0 {
                    0.0
                } else {
                    edge.value_sum / edge.visits as f32
                },
            })
            .collect();
        stats.elapsed_micros = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        Ok(SearchResult {
            best_move: chosen.mv,
            root_value: root.network_value,
            action_visits,
            stats,
        })
    }

    fn select(
        &self,
        nodes: &mut Vec<Node>,
        position: &mut SearchPosition,
    ) -> Result<Selection, SearchError> {
        let mut node_index = 0;
        let mut path = Vec::new();
        let mut undos: Vec<SearchUndo> = Vec::new();

        let selected = loop {
            if let Some(value) = nodes[node_index].terminal_value {
                break Selection::Terminal { path, value };
            }
            if !nodes[node_index].expanded {
                if nodes[node_index].pending {
                    revert_virtual(&mut nodes[..], &path);
                    break Selection::Collision;
                }
                let mut legal_moves = Vec::new();
                position.legal_moves_into(&mut legal_moves);
                match position.status_with_legal_moves(&legal_moves) {
                    Status::Ongoing => {
                        nodes[node_index].pending = true;
                        break Selection::Leaf(PendingLeaf {
                            node: node_index,
                            path,
                            input: encode_search_current(position),
                            legal_moves,
                            side_to_move: position.side_to_move(),
                        });
                    }
                    status => {
                        let value = terminal_value(status);
                        let node = &mut nodes[node_index];
                        node.expanded = true;
                        node.terminal_value = Some(value);
                        break Selection::Terminal { path, value };
                    }
                }
            }

            let edge_index = select_edge(&nodes[node_index], node_index == 0, self.config);
            let (mv, child) = {
                let edge = &mut nodes[node_index].edges[edge_index];
                edge.virtual_visits += 1;
                edge.virtual_value_sum -= 1.0;
                (edge.mv, edge.child)
            };
            let child_index = match child {
                Some(index) => index,
                None => {
                    let index = nodes.len();
                    nodes.push(Node::new());
                    nodes[node_index].edges[edge_index].child = Some(index);
                    index
                }
            };
            path.push((node_index, edge_index));
            undos.push(position.make_move(mv));
            node_index = child_index;
        };

        while let Some(undo) = undos.pop() {
            position.unmake_move(undo);
        }
        Ok(selected)
    }

    fn apply_root_noise<R: Rng + ?Sized>(&self, root: &mut Node, rng: &mut R) {
        let Some(alpha) = self.config.root_dirichlet_alpha else {
            return;
        };
        if self.config.root_noise_fraction == 0.0 || root.edges.is_empty() {
            return;
        }
        let gamma = Gamma::new(alpha as f64, 1.0).expect("validated positive alpha");
        let mut noise: Vec<f64> = root.edges.iter().map(|_| gamma.sample(rng)).collect();
        let total = noise.iter().sum::<f64>();
        if total == 0.0 || !total.is_finite() {
            noise.fill(1.0 / root.edges.len() as f64);
        } else {
            for value in &mut noise {
                *value /= total;
            }
        }
        let fraction = self.config.root_noise_fraction;
        for (edge, noise) in root.edges.iter_mut().zip(noise) {
            edge.prior = (1.0 - fraction) * edge.prior + fraction * noise as f32;
        }
    }
}

struct PendingLeaf {
    node: usize,
    path: Vec<(usize, usize)>,
    input: EncodedPosition,
    legal_moves: Vec<Move>,
    side_to_move: chess_core::Color,
}

struct PendingExpansion {
    node: usize,
    path: Vec<(usize, usize)>,
    legal_moves: Vec<Move>,
    side_to_move: chess_core::Color,
}

enum Selection {
    Leaf(PendingLeaf),
    Terminal {
        path: Vec<(usize, usize)>,
        value: f32,
    },
    Collision,
}

#[derive(Debug)]
struct Node {
    expanded: bool,
    pending: bool,
    network_value: f32,
    terminal_value: Option<f32>,
    edges: Vec<Edge>,
}

impl Node {
    fn new() -> Self {
        Self {
            expanded: false,
            pending: false,
            network_value: 0.0,
            terminal_value: None,
            edges: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct Edge {
    mv: Move,
    action: usize,
    prior: f32,
    visits: u32,
    value_sum: f32,
    virtual_visits: u32,
    virtual_value_sum: f32,
    child: Option<usize>,
}

fn expand_node(
    node: &mut Node,
    legal_moves: Vec<Move>,
    side_to_move: chess_core::Color,
    evaluation: &Evaluation,
) -> Result<(), PolicyError> {
    let actions: Vec<_> = legal_moves
        .iter()
        .map(|&mv| move_to_action(mv, side_to_move))
        .collect::<Result<_, _>>()?;
    let policy_logits = evaluation.policy_logits.as_slice();
    let max_logit = actions
        .iter()
        .map(|&action| policy_logits[action])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = actions
        .iter()
        .map(|&action| (policy_logits[action] - max_logit).exp())
        .collect();
    let total = weights.iter().sum::<f32>();
    if total == 0.0 || !total.is_finite() {
        let uniform = 1.0 / weights.len() as f32;
        weights.fill(uniform);
    } else {
        for weight in &mut weights {
            *weight /= total;
        }
    }
    node.edges = legal_moves
        .into_iter()
        .zip(actions)
        .zip(weights)
        .map(|((mv, action), prior)| Edge {
            mv,
            action,
            prior,
            visits: 0,
            value_sum: 0.0,
            virtual_visits: 0,
            virtual_value_sum: 0.0,
            child: None,
        })
        .collect();
    node.network_value = evaluation.scalar_value();
    node.expanded = true;
    Ok(())
}

fn select_edge(node: &Node, is_root: bool, config: SearchConfig) -> usize {
    let parent_visits = node
        .edges
        .iter()
        .map(|edge| edge.visits + edge.virtual_visits)
        .sum::<u32>()
        .max(1) as f32;
    node.edges
        .iter()
        .enumerate()
        .map(|(index, edge)| {
            let effective_visits = edge.visits + edge.virtual_visits;
            let q = if effective_visits == 0 {
                if is_root {
                    node.network_value
                } else {
                    (node.network_value - config.fpu_reduction).clamp(-1.0, 1.0)
                }
            } else {
                (edge.value_sum + edge.virtual_value_sum) / effective_visits as f32
            };
            let u =
                config.cpuct * edge.prior * parent_visits.sqrt() / (1 + effective_visits) as f32;
            (index, q + u, edge.action)
        })
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| right.2.cmp(&left.2))
        })
        .map(|(index, _, _)| index)
        .expect("expanded ongoing node has edges")
}

fn backup(nodes: &mut [Node], path: &[(usize, usize)], mut value: f32) {
    for &(node_index, edge_index) in path.iter().rev() {
        let parent_value = -value;
        let edge = &mut nodes[node_index].edges[edge_index];
        debug_assert!(edge.virtual_visits > 0);
        edge.virtual_visits -= 1;
        edge.virtual_value_sum += 1.0;
        edge.visits += 1;
        edge.value_sum += parent_value;
        value = parent_value;
    }
}

fn revert_virtual(nodes: &mut [Node], path: &[(usize, usize)]) {
    for &(node_index, edge_index) in path.iter().rev() {
        let edge = &mut nodes[node_index].edges[edge_index];
        debug_assert!(edge.virtual_visits > 0);
        edge.virtual_visits -= 1;
        edge.virtual_value_sum += 1.0;
    }
}

fn terminal_value(status: Status) -> f32 {
    match status {
        Status::Checkmate => -1.0,
        Status::Stalemate
        | Status::InsufficientMaterial
        | Status::ThreefoldRepetition
        | Status::FiftyMoveRule => 0.0,
        Status::Ongoing => unreachable!("ongoing is not terminal"),
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::evaluator::UniformEvaluator;

    #[test]
    fn frozen_gate_search_matches_persisted_integer_identity() {
        let config = SearchConfig::frozen_gate();
        assert_eq!(config.simulations, FROZEN_GATE_SIMULATIONS);
        assert_eq!(config.batch_size, FROZEN_GATE_BATCH_SIZE);
        assert_eq!(config.cpuct, ppm_as_f32(FROZEN_GATE_CPUCT_PPM));
        assert_eq!(
            config.fpu_reduction,
            ppm_as_f32(FROZEN_GATE_FPU_REDUCTION_PPM)
        );
        assert_eq!(
            config.move_time,
            Some(Duration::from_millis(FROZEN_GATE_TIME_MS))
        );
        assert_eq!(config.root_dirichlet_alpha, None);
        assert_eq!(config.root_noise_fraction, 0.0);
    }

    #[test]
    fn completes_exact_budget_and_leaves_no_virtual_loss() {
        struct OwnedOnlyEvaluator {
            batches: u32,
        }
        impl Evaluator for OwnedOnlyEvaluator {
            fn evaluate_batch(
                &mut self,
                _positions: &[EncodedPosition],
            ) -> Result<Vec<Evaluation>, EvaluatorError> {
                panic!("MCTS must hand off owned input buffers")
            }

            fn evaluate_owned_batch(
                &mut self,
                positions: Vec<EncodedPosition>,
            ) -> Result<Vec<Evaluation>, EvaluatorError> {
                self.batches += 1;
                UniformEvaluator.evaluate_batch(&positions)
            }
        }

        let board = Board::from_fen(crate::START_FEN).unwrap();
        let mut evaluator = OwnedOnlyEvaluator { batches: 0 };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let result = Mcts::new(SearchConfig {
            simulations: 17,
            batch_size: 4,
            ..SearchConfig::default()
        })
        .unwrap()
        .search(&board, &mut evaluator, &mut rng)
        .unwrap();
        assert_eq!(result.stats.completed_simulations, 17);
        assert_eq!(result.stats.inference_batches, evaluator.batches);
        assert_eq!(
            result
                .action_visits
                .iter()
                .map(|item| item.visits)
                .sum::<u32>(),
            17
        );
        assert!(result.action_visits.iter().all(|item| {
            move_to_action(item.mv, board.side_to_move).unwrap() == item.action as usize
        }));
        assert!(board.move_from_uci(&result.best_move.to_uci()).is_ok());
    }

    #[test]
    fn refuses_terminal_root() {
        let board = Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let mut evaluator = UniformEvaluator;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        assert!(matches!(
            Mcts::new(SearchConfig::default())
                .unwrap()
                .search(&board, &mut evaluator, &mut rng),
            Err(SearchError::Terminal(Status::Checkmate))
        ));
    }

    #[test]
    fn fpu_is_unreduced_at_root_and_loss_biased_elsewhere() {
        let node = Node {
            expanded: true,
            pending: false,
            network_value: 0.5,
            terminal_value: None,
            edges: vec![Edge {
                mv: Board::from_fen(crate::START_FEN).unwrap().get_legal_moves()[0],
                action: 0,
                prior: 1.0,
                visits: 0,
                value_sum: 0.0,
                virtual_visits: 0,
                virtual_value_sum: 0.0,
                child: None,
            }],
        };
        // The single edge is selected in either case; explicitly verify the formula.
        let config = SearchConfig::default();
        let root_q = node.network_value;
        let nonroot_q = (node.network_value - config.fpu_reduction).clamp(-1.0, 1.0);
        assert_eq!(root_q, 0.5);
        assert_eq!(nonroot_q, 0.25);
        assert_eq!(select_edge(&node, true, config), 0);
        assert_eq!(select_edge(&node, false, config), 0);
    }

    #[test]
    fn mixed_terminal_and_neural_leaves_keep_budget_and_mate_sign() {
        let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let mut evaluator = UniformEvaluator;
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let result = Mcts::new(SearchConfig {
            simulations: 96,
            batch_size: 8,
            ..SearchConfig::default()
        })
        .unwrap()
        .search(&board, &mut evaluator, &mut rng)
        .unwrap();
        assert_eq!(result.stats.completed_simulations, 96);
        assert_eq!(
            result
                .action_visits
                .iter()
                .map(|edge| edge.visits)
                .sum::<u32>(),
            96
        );

        let mating_edges: Vec<_> = result
            .action_visits
            .iter()
            .filter(|edge| {
                let mut child = board.clone();
                child.make_search_move(
                    crate::policy::action_to_move(&board, edge.action as usize).unwrap(),
                );
                child.status() == Status::Checkmate
            })
            .collect();
        assert!(!mating_edges.is_empty());
        assert!(
            mating_edges
                .iter()
                .any(|edge| edge.visits > 0 && edge.mean_value > 0.99)
        );
    }
}
