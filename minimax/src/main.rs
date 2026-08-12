use axum::{
    Router,
    extract::{Json, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{options, post},
};
use minimax::{BotError, BotRequest, SearchError, SearchLimits, respond_with_limits};
use serde::Serialize;

#[derive(Clone, Copy)]
struct AppState {
    limits: SearchLimits,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

async fn move_handler(State(state): State<AppState>, Json(request): Json<BotRequest>) -> Response {
    let mut response = match respond_with_limits(request, state.limits) {
        Ok(reply) => (StatusCode::OK, Json(reply)).into_response(),
        Err(error) => (
            status_for_error(&error),
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    };
    add_cors(&mut response);
    response
}

fn status_for_error(error: &BotError) -> StatusCode {
    match error {
        BotError::InvalidGame(_) => StatusCode::BAD_REQUEST,
        BotError::GameOver(_) => StatusCode::CONFLICT,
        BotError::Search(SearchError::AlgorithmNotImplemented) => StatusCode::NOT_IMPLEMENTED,
        BotError::Search(SearchError::InvalidLimits(_)) => StatusCode::INTERNAL_SERVER_ERROR,
        BotError::Search(SearchError::GameOver(_)) => StatusCode::CONFLICT,
        BotError::Search(SearchError::Stopped) => StatusCode::SERVICE_UNAVAILABLE,
        BotError::IllegalEngineMove(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

async fn preflight() -> Response {
    let mut response = StatusCode::NO_CONTENT.into_response();
    add_cors(&mut response);
    response
}

fn add_cors(response: &mut Response) {
    let headers = response.headers_mut();
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("content-type"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("POST, OPTIONS"),
    );
}

#[tokio::main]
async fn main() {
    let limits = SearchLimits::from_env()
        .unwrap_or_else(|error| panic!("invalid minimax server configuration: {error}"));
    let app = Router::new()
        .route("/move", post(move_handler))
        .route("/move", options(preflight))
        .with_state(AppState { limits });
    let bind_address =
        std::env::var("MINIMAX_BIND_ADDRESS").unwrap_or_else(|_| "127.0.0.1:3002".to_string());
    let listener = tokio::net::TcpListener::bind(&bind_address)
        .await
        .unwrap_or_else(|error| panic!("bind minimax bot on {bind_address}: {error}"));
    println!(
        "Minimax bot listening on http://{bind_address} (depth {})",
        limits.max_depth
    );
    axum::serve(listener, app).await.expect("serve minimax bot");
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_core::Status;

    #[test]
    fn error_statuses() {
        assert_eq!(
            status_for_error(&BotError::Search(SearchError::AlgorithmNotImplemented)),
            StatusCode::NOT_IMPLEMENTED
        );
        assert_eq!(
            status_for_error(&BotError::GameOver(Status::Checkmate)),
            StatusCode::CONFLICT
        );
    }
}
