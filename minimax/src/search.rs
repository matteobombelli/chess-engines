//! Minimax search using negamax and alpha-beta pruning.
//!
//! `evaluate` scores a position for the side to move. After making a move, the
//! other side is to move, so a child score must be negated. This lets one
//! maximizing function handle both White and Black.
//!
//! Implement this file in four passes:
//! 1. fixed-depth `search_depth`;
//! 2. recursive `alpha_beta`;
//! 3. iterative deepening in `find_best_move`;
//! 4. `quiescence`.

use std::fmt;
use std::time::{Duration, Instant};

use chess_core::{Board, Move, Status};

use crate::config::SearchLimits;
use crate::evaluation::{MATE_SCORE, Score};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchResult {
    /// The move chosen at the root.
    pub best_move: Move,
    /// Centipawns from the root side's perspective.
    pub score: Score,
    /// Best line found, starting with `best_move`.
    pub principal_variation: Vec<Move>,
    pub stats: SearchStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Positions visited during this search.
    pub nodes: u64,
    /// Branches skipped because their score reached beta.
    pub cutoffs: u64,
    /// Last depth that finished before a limit stopped the search.
    pub completed_depth: u8,
    pub elapsed: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchError {
    AlgorithmNotImplemented,
    GameOver(Status),
    InvalidLimits(String),
    Stopped,
}

impl fmt::Display for SearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlgorithmNotImplemented => write!(
                formatter,
                "minimax search is not implemented yet; complete minimax/src/search.rs"
            ),
            Self::GameOver(status) => write!(formatter, "game is over: {status:?}"),
            Self::InvalidLimits(error) => write!(formatter, "invalid search limits: {error}"),
            Self::Stopped => write!(formatter, "search stopped at its time or node limit"),
        }
    }
}

impl std::error::Error for SearchError {}

/// Find the best legal move.
pub fn find_best_move(board: &Board, limits: SearchLimits) -> Result<SearchResult, SearchError> {
    // Validate limits
    limits.validate().map_err(SearchError::InvalidLimits)?;

    // Reject positions where the game has ended
    let status = board.status();
    if status != Status::Ongoing {
        return Err(SearchError::GameOver(status));
    }

    // Simple search
    let mut context = SearchContext::new(limits);
    let mut last: Option<SearchResult> = None;

    for depth in 1..=limits.max_depth {
        let previous_best = last.as_ref().map(|result| result.best_move);

        match search_depth(board, depth, previous_best, &mut context) {
            Ok(result) => {
                last = Some(result);
            }
            Err(SearchError::Stopped) if last.is_some() => {
                break;
            }
            Err(error) => {
                return Err(error);
            }
        }
    }

    last.ok_or(SearchError::Stopped)
}

/// Search all root moves to one depth.
#[allow(dead_code, unused_variables)]
fn search_depth(
    board: &Board,
    depth: u8,
    previous_best: Option<Move>,
    context: &mut SearchContext,
) -> Result<SearchResult, SearchError> {
    // Generate and order moves
    let mut legal_moves: Vec<Move> = board.get_legal_moves();
    crate::move_ordering::order_moves(board, &mut legal_moves, previous_best);

    // Initialize search state
    let mut alpha = -crate::evaluation::INFINITY;
    let beta = crate::evaluation::INFINITY;
    let mut best_move: Option<Move> = None;
    let mut best_pv: Vec<Move> = Vec::new();

    // Evaluate branches
    for mv in legal_moves {
        let mut child = (*board).clone();
        child.make_search_move(mv);

        let child_result = alpha_beta(&child, depth - 1, 1, -beta, -alpha, context)?;

        let score = -child_result.score;

        // Store best move
        if score > alpha {
            alpha = score;
            best_move = Some(mv);

            let mut pv = vec![mv];
            pv.extend(child_result.principal_variation);
            best_pv = pv;
        }
    }

    let best_move = best_move.expect("ongoing position has moves");

    Ok(SearchResult {
        best_move,
        score: alpha,
        principal_variation: best_pv,
        stats: context.stats(depth),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NodeResult {
    /// Score from this node's side-to-move perspective.
    score: Score,
    /// Best moves below this node. Empty at a leaf.
    principal_variation: Vec<Move>,
}

/// Recursive negamax with alpha-beta pruning.
#[allow(dead_code, unused_mut, unused_variables)]
fn alpha_beta(
    board: &Board,
    depth: u8,
    ply: u16,
    mut alpha: Score,
    beta: Score,
    context: &mut SearchContext,
) -> Result<NodeResult, SearchError> {
    // Continue tactical moves at the depth limit
    if depth == 0 {
        return quiescence(board, ply, alpha, beta, context);
    }

    // Count this node and stop if a search limit was reached
    context.visit_node()?;

    // Return the correct score for checkmate or a draw
    if let Some(score) = terminal_score(board.status(), ply) {
        return Ok(NodeResult {
            score,
            principal_variation: Vec::new(),
        });
    }

    // Generate and order the legal moves
    let mut legal_moves = board.get_legal_moves();
    crate::move_ordering::order_moves(board, &mut legal_moves, None);

    // Keep the best line found from this position
    let mut best_pv = Vec::new();

    // Search each child position
    for mv in legal_moves {
        let mut child = (*board).clone();
        child.make_search_move(mv);
        let child_result = alpha_beta(&child, depth - 1, ply + 1, -beta, -alpha, context)?;

        let score = -child_result.score;

        // Stop searching when the parent will reject this branch
        if score >= beta {
            context.cutoffs += 1;

            let mut pv = vec![mv];
            pv.extend(child_result.principal_variation);

            return Ok(NodeResult {
                score,
                principal_variation: pv,
            });
        }

        // Save a new best score and principal variation
        if score > alpha {
            alpha = score;

            let mut pv = vec![mv];
            pv.extend(child_result.principal_variation);
            best_pv = pv;
        }
    }

    // Return the best result found at this node
    Ok(NodeResult {
        score: alpha,
        principal_variation: best_pv,
    })
}

/// Continue captures and promotions at leaf nodes.
#[allow(dead_code, unused_mut, unused_variables)]
fn quiescence(
    board: &Board,
    ply: u16,
    mut alpha: Score,
    beta: Score,
    context: &mut SearchContext,
) -> Result<NodeResult, SearchError> {
    // Count this node and stop if a search limit was reached
    context.visit_node()?;

    // Return the correct score for checkmate or a draw
    if let Some(score) = terminal_score(board.status(), ply) {
        return Ok(NodeResult {
            score,
            principal_variation: Vec::new(),
        });
    }

    // Check whether every legal move must escape check
    let in_check = board.is_in_check();

    // Use static evaluation when the side to move may stand pat
    if !in_check {
        let stand_pat = crate::evaluation::evaluate(board);

        // Stop when standing pat already reaches beta
        if stand_pat >= beta {
            return Ok(NodeResult {
                score: stand_pat,
                principal_variation: Vec::new(),
            });
        }

        // Save a better static score
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    // Generate the legal continuations
    let mut legal_moves = board.get_legal_moves();

    // In quiet positions, keep only captures and promotions
    if !in_check {
        legal_moves.retain(|mv| {
            crate::move_ordering::captured_piece(board, *mv).is_some() || mv.promotion.is_some()
        });
    }

    // Search the most promising tactical moves first
    crate::move_ordering::order_moves(board, &mut legal_moves, None);

    // Keep the best tactical line found from this position
    let mut best_pv = Vec::new();

    // Search each tactical continuation
    for mv in legal_moves {
        let mut child = (*board).clone();
        child.make_search_move(mv);
        let child_result = quiescence(&child, ply + 1, -beta, -alpha, context)?;

        let score = -child_result.score;

        // Stop searching when the parent will reject this branch
        if score >= beta {
            context.cutoffs += 1;

            let mut pv = vec![mv];
            pv.extend(child_result.principal_variation);

            return Ok(NodeResult {
                score,
                principal_variation: pv,
            });
        }

        // Save a new best score and tactical line
        if score > alpha {
            alpha = score;

            let mut pv = vec![mv];
            pv.extend(child_result.principal_variation);
            best_pv = pv;
        }
    }

    // Return the best quiet or tactical result found
    Ok(NodeResult {
        score: alpha,
        principal_variation: best_pv,
    })
}

/// Score a finished position for the side to move.
pub fn terminal_score(status: Status, ply: u16) -> Option<Score> {
    match status {
        // The side to move has been checkmated, so the score is negative. Adding
        // ply prefers delivering mate sooner and receiving mate later.
        Status::Checkmate => Some(-MATE_SCORE + Score::from(ply)),
        Status::Stalemate | Status::ThreefoldRepetition | Status::FiftyMoveRule => Some(0),
        Status::Ongoing => None,
    }
}

#[allow(dead_code)]
struct SearchContext {
    /// Time at which this search started.
    started: Instant,
    /// None means there is no time limit.
    deadline: Option<Instant>,
    /// None means there is no node limit.
    max_nodes: Option<u64>,
    nodes: u64,
    cutoffs: u64,
}

#[allow(dead_code)]
impl SearchContext {
    fn new(limits: SearchLimits) -> Self {
        let started = Instant::now();
        Self {
            started,
            deadline: limits.move_time.map(|duration| started + duration),
            max_nodes: limits.max_nodes,
            nodes: 0,
            cutoffs: 0,
        }
    }

    /// Count a node and check the search limits.
    fn visit_node(&mut self) -> Result<(), SearchError> {
        self.nodes += 1;
        if self.max_nodes.is_some_and(|limit| self.nodes > limit) {
            return Err(SearchError::Stopped);
        }
        // Reading the clock is relatively expensive, so check it every 1,024
        // nodes. The first-node check handles an already expired deadline.
        if (self.nodes == 1 || self.nodes & 1023 == 0)
            && self.deadline.is_some_and(|limit| Instant::now() >= limit)
        {
            return Err(SearchError::Stopped);
        }
        Ok(())
    }

    fn stats(&self, completed_depth: u8) -> SearchStats {
        SearchStats {
            nodes: self.nodes,
            cutoffs: self.cutoffs,
            completed_depth,
            elapsed: self.started.elapsed(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mate_score() {
        let immediate = terminal_score(Status::Checkmate, 2).unwrap();
        let later = terminal_score(Status::Checkmate, 6).unwrap();
        assert!(immediate < later);
        assert!(immediate < -29_000);
    }

    #[test]
    fn draw_score() {
        for status in [
            Status::Stalemate,
            Status::ThreefoldRepetition,
            Status::FiftyMoveRule,
        ] {
            assert_eq!(terminal_score(status, 12), Some(0));
        }
        assert_eq!(terminal_score(Status::Ongoing, 0), None);
    }

    #[test]
    fn node_limit() {
        let mut context = SearchContext::new(SearchLimits {
            max_depth: 1,
            move_time: None,
            max_nodes: Some(1),
        });
        assert_eq!(context.visit_node(), Ok(()));
        assert_eq!(context.visit_node(), Err(SearchError::Stopped));
    }

    #[test]
    fn search_result_is_legal() {
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        match find_best_move(&board, SearchLimits::fixed_depth(1).unwrap()) {
            Err(SearchError::AlgorithmNotImplemented) => {}
            Ok(result) => assert!(board.get_legal_moves().contains(&result.best_move)),
            Err(error) => panic!("unexpected search error: {error}"),
        }
    }
}
