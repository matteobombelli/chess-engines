//! The bot HTTP contract, identical in shape to the AlphaMini server: a whole
//! game's movetext in, one legal SAN move and the resulting FEN out.

use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use chess_core::{Status, movetext_moves};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::decode::{choose_move, truncate_context};
use crate::encoding::GameEncoder;
use crate::evaluator::TokenEvaluator;
use crate::model_manifest::ModelManifestV1;

/// What the manifest fixes about serving: the graph's position budget and the
/// published sampling temperature.
#[derive(Clone, Copy, Debug)]
pub struct DecodeConfig {
    pub context: usize,
    pub temperature: f32,
}

impl DecodeConfig {
    pub fn from_manifest(manifest: &ModelManifestV1) -> Self {
        Self {
            context: manifest.context,
            temperature: manifest.decode_temperature,
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    evaluator: Arc<Mutex<Box<dyn TokenEvaluator>>>,
    decode: DecodeConfig,
    gate: Arc<tokio::sync::Semaphore>,
}

impl AppState {
    pub fn new(evaluator: Box<dyn TokenEvaluator>, decode: DecodeConfig) -> Self {
        Self {
            evaluator: Arc::new(Mutex::new(evaluator)),
            decode,
            gate: Arc::new(tokio::sync::Semaphore::new(1)),
        }
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
                "MiniGPT is already choosing a move",
            );
        }
    };
    let evaluator = state.evaluator.clone();
    let decode = state.decode;
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut evaluator = evaluator
            .lock()
            .map_err(|_| ApiError::Model("evaluator lock is poisoned".to_string()))?;
        choose_response(&request, decode, evaluator.as_mut())
    });
    match task.await {
        Err(error_value) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("decode worker failed: {error_value}"),
        ),
        Ok(Ok(response)) => (StatusCode::OK, Json(response)).into_response(),
        Ok(Err(ApiError::Invalid(message))) => error(StatusCode::BAD_REQUEST, message),
        Ok(Err(ApiError::Terminal(message))) => error(StatusCode::CONFLICT, message),
        Ok(Err(ApiError::Model(message))) => error(StatusCode::INTERNAL_SERVER_ERROR, message),
    }
}

pub fn choose_response(
    request: &BotRequest,
    decode: DecodeConfig,
    evaluator: &mut dyn TokenEvaluator,
) -> Result<BotResponse, ApiError> {
    // One replay produces both the tokens the model sees and the position whose
    // legal moves bound what it may answer.
    let mut encoder = GameEncoder::new();
    for san in movetext_moves(&request.san) {
        encoder
            .push_san(san)
            .map_err(|error| ApiError::Invalid(error.to_string()))?;
    }
    let mut board = encoder.board().clone();
    if board.status() != Status::Ongoing {
        return Err(ApiError::Terminal("game is already over".to_string()));
    }
    let tokens = truncate_context(encoder.tokens(), decode.context);
    let logits = evaluator
        .logits(&tokens)
        .map_err(|error| ApiError::Model(error.to_string()))?;
    let mut rng = ChaCha8Rng::from_seed(deterministic_seed(board.to_fen().as_bytes()));
    let chosen = choose_move(logits.last_row(), &board, decode.temperature, &mut rng)
        .map_err(|error| ApiError::Model(error.to_string()))?;
    board.make_move(chosen);
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
    Model(String),
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
    use chess_core::Board;

    use super::*;
    use crate::evaluator::UniformEvaluator;

    fn decode() -> DecodeConfig {
        DecodeConfig {
            context: 256,
            temperature: 0.5,
        }
    }

    #[test]
    fn adapter_preserves_san_fen_contract_and_legality() {
        let mut evaluator = UniformEvaluator;
        let response =
            choose_response(&BotRequest { san: String::new() }, decode(), &mut evaluator).unwrap();
        let replayed = Board::import_san(&format!("1. {}", response.san)).unwrap();
        assert_eq!(replayed.to_fen(), response.fen);
    }

    #[test]
    fn the_same_position_always_gets_the_same_answer() {
        let mut evaluator = UniformEvaluator;
        let request = BotRequest {
            san: "1. e4 e5 2. Nf3".into(),
        };
        let first = choose_response(&request, decode(), &mut evaluator).unwrap();
        let second = choose_response(&request, decode(), &mut evaluator).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn malformed_history_fails_closed() {
        let mut evaluator = UniformEvaluator;
        let result = choose_response(
            &BotRequest {
                san: "1. e5".into(),
            },
            decode(),
            &mut evaluator,
        );
        assert!(matches!(result, Err(ApiError::Invalid(_))));
    }

    #[test]
    fn a_finished_game_is_rejected_rather_than_answered() {
        let mut evaluator = UniformEvaluator;
        let result = choose_response(
            &BotRequest {
                san: "1. f3 e5 2. g4 Qh4#".into(),
            },
            decode(),
            &mut evaluator,
        );
        assert!(matches!(result, Err(ApiError::Terminal(_))));
    }

    #[test]
    fn absent_san_defaults_to_start_but_unknown_fields_fail() {
        let request: BotRequest = serde_json::from_str("{}").unwrap();
        assert!(request.san.is_empty());
        assert!(serde_json::from_str::<BotRequest>(r#"{"san":"","fen":"forged"}"#).is_err());
    }

    #[tokio::test]
    async fn concurrent_requests_are_rejected_immediately() {
        let state = AppState::new(Box::new(UniformEvaluator), decode());
        let _permit = state.gate.clone().try_acquire_owned().unwrap();
        let response =
            move_handler(State(state), Ok(Json(BotRequest { san: String::new() }))).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
