use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use chess_core::{Board, Status};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::evaluator::Evaluator;
use crate::mcts::{Mcts, SearchConfig, SearchError};

#[derive(Clone)]
pub struct AppState {
    evaluator: Arc<Mutex<Box<dyn Evaluator>>>,
    search: SearchConfig,
    gate: Arc<tokio::sync::Semaphore>,
}

impl AppState {
    pub fn new(evaluator: Box<dyn Evaluator>, search: SearchConfig) -> Result<Self, SearchError> {
        search.validate()?;
        Ok(Self {
            evaluator: Arc::new(Mutex::new(evaluator)),
            search,
            gate: Arc::new(tokio::sync::Semaphore::new(1)),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BotRequest {
    /// Complete PGN movetext. Empty means the initial position.
    #[serde(default)]
    pub san: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BotResponse {
    pub san: String,
    pub fen: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub fn router(state: AppState) -> axum::Router {
    axum::Router::new()
        .route("/move", post(move_handler))
        .with_state(state)
}

async fn move_handler(
    State(state): State<AppState>,
    payload: Result<Json<BotRequest>, JsonRejection>,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(rejection) => return error(StatusCode::BAD_REQUEST, rejection.body_text()),
    };
    let permit = match state.gate.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "AlphaMini is already searching",
            );
        }
    };
    let evaluator = state.evaluator.clone();
    let search = state.search;
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut evaluator = evaluator
            .lock()
            .map_err(|_| ApiError::Search("evaluator lock is poisoned".to_string()))?;
        choose_response(&request, search, evaluator.as_mut())
    });
    match task.await {
        Err(error_value) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("search worker failed: {error_value}"),
        ),
        Ok(Ok(response)) => (StatusCode::OK, Json(response)).into_response(),
        Ok(Err(ApiError::Invalid(message))) => error(StatusCode::BAD_REQUEST, message),
        Ok(Err(ApiError::Terminal(message))) => error(StatusCode::CONFLICT, message),
        Ok(Err(ApiError::Search(message))) => error(StatusCode::INTERNAL_SERVER_ERROR, message),
    }
}

pub fn choose_response(
    request: &BotRequest,
    search: SearchConfig,
    evaluator: &mut dyn Evaluator,
) -> Result<BotResponse, ApiError> {
    let mut board = Board::import_san(&request.san).map_err(ApiError::Invalid)?;
    if board.status() != Status::Ongoing {
        return Err(ApiError::Terminal("game is already over".to_string()));
    }
    let seed = deterministic_seed(board.to_fen().as_bytes());
    let mut rng = ChaCha8Rng::from_seed(seed);
    let result = Mcts::new(search)
        .map_err(|error| ApiError::Search(error.to_string()))?
        .search(&board, evaluator, &mut rng)
        .map_err(|error| ApiError::Search(error.to_string()))?;
    board.make_move(result.best_move);
    let san = board
        .san_history
        .last()
        .cloned()
        .expect("make_move records SAN");
    Ok(BotResponse {
        san,
        fen: board.to_fen(),
    })
}

#[derive(Debug)]
pub enum ApiError {
    Invalid(String),
    Terminal(String),
    Search(String),
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
        .into_response()
}

fn deterministic_seed(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::evaluator::UniformEvaluator;

    #[test]
    fn adapter_preserves_san_fen_contract_and_legality() {
        let mut evaluator = UniformEvaluator;
        let response = choose_response(
            &BotRequest { san: String::new() },
            SearchConfig {
                simulations: 2,
                batch_size: 2,
                move_time: Some(Duration::from_secs(1)),
                ..SearchConfig::default()
            },
            &mut evaluator,
        )
        .unwrap();
        let replayed = Board::import_san(&format!("1. {}", response.san)).unwrap();
        assert_eq!(replayed.to_fen(), response.fen);
    }

    #[test]
    fn malformed_history_fails_closed() {
        let mut evaluator = UniformEvaluator;
        let result = choose_response(
            &BotRequest {
                san: "1. e5".into(),
            },
            SearchConfig {
                simulations: 1,
                batch_size: 1,
                ..SearchConfig::default()
            },
            &mut evaluator,
        );
        assert!(matches!(result, Err(ApiError::Invalid(_))));
    }

    #[test]
    fn absent_san_defaults_to_start_but_unknown_fields_fail() {
        let request: BotRequest = serde_json::from_str("{}").unwrap();
        assert!(request.san.is_empty());
        assert!(serde_json::from_str::<BotRequest>(r#"{"san":"","fen":"forged"}"#).is_err());
    }

    #[tokio::test]
    async fn concurrent_search_is_rejected_immediately() {
        let state = AppState::new(
            Box::new(UniformEvaluator),
            SearchConfig {
                simulations: 1,
                batch_size: 1,
                ..SearchConfig::default()
            },
        )
        .unwrap();
        let _permit = state.gate.clone().try_acquire_owned().unwrap();
        let response =
            move_handler(State(state), Ok(Json(BotRequest { san: String::new() }))).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
