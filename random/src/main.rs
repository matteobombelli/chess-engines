use axum::{
    extract::Json,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{options, post},
    Router,
};
use random::{respond, BotRequest};
use serde::Serialize;

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

async fn move_handler(Json(request): Json<BotRequest>) -> Response {
    let mut response = match respond(request) {
        Ok(reply) => (StatusCode::OK, Json(reply)).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response(),
    };
    add_cors(&mut response);
    response
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
    let app = Router::new()
        .route("/move", post(move_handler))
        .route("/move", options(preflight));
    let bind_address = std::env::var("BIND_ADDRESS")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string());
    let listener = tokio::net::TcpListener::bind(&bind_address)
        .await
        .unwrap_or_else(|error| panic!("bind random bot on {bind_address}: {error}"));
    println!("Random bot listening on http://{bind_address}");
    axum::serve(listener, app).await.expect("serve random bot");
}
