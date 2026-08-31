//! A deployed bot played over the production HTTP API.
//!
//! The time-budgeted bots get their strength from the machine they run on, so
//! rating them means playing the deployment rather than a local rebuild. The
//! request is the same one the site's board sends: the whole game as PGN
//! movetext, answered with one SAN move and the FEN it produces.

use std::thread::sleep;
use std::time::Duration;

use chess_core::{Board, Move};
use serde::{Deserialize, Serialize};

use crate::Engine;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const RETRY_BACKOFF: Duration = Duration::from_secs(5);
const MAX_RETRIES: u32 = 3;

#[derive(Serialize)]
struct BotRequest<'a> {
    san: &'a str,
}

#[derive(Deserialize)]
struct BotResponse {
    san: String,
    fen: String,
}

#[derive(Deserialize)]
struct BotErrorResponse {
    error: String,
}

pub struct HttpEngine {
    agent: ureq::Agent,
    url: String,
    name: String,
}

struct RequestFailure {
    message: String,
    retryable: bool,
}

impl HttpEngine {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Result<Self, String> {
        let url = url.into();
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(format!("bot URL must be http(s), got {url:?}"));
        }
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .http_status_as_error(false)
            .build()
            .new_agent();
        Ok(Self {
            agent,
            url,
            name: name.into(),
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    fn request_move(&self, san: &str) -> Result<BotResponse, String> {
        let mut attempts = 0;
        loop {
            match self.try_request_move(san) {
                Ok(reply) => return Ok(reply),
                Err(failure) if failure.retryable && attempts < MAX_RETRIES => {
                    attempts += 1;
                    eprintln!(
                        "{}: {} (retry {attempts}/{MAX_RETRIES} in {:?})",
                        self.name, failure.message, RETRY_BACKOFF
                    );
                    sleep(RETRY_BACKOFF);
                }
                Err(failure) => return Err(failure.message),
            }
        }
    }

    fn try_request_move(&self, san: &str) -> Result<BotResponse, RequestFailure> {
        let mut response = self
            .agent
            .post(&self.url)
            .send_json(BotRequest { san })
            .map_err(|error| RequestFailure {
                message: format!("{} request failed: {error}", self.url),
                retryable: matches!(
                    error,
                    ureq::Error::Timeout(_) | ureq::Error::Io(_) | ureq::Error::ConnectionFailed
                ),
            })?;
        let status = response.status().as_u16();
        if is_gateway_error(status) {
            return Err(RequestFailure {
                message: format!("{} is temporarily unavailable (HTTP {status})", self.url),
                retryable: true,
            });
        }
        if !(200..300).contains(&status) {
            let message = match response.body_mut().read_json::<BotErrorResponse>() {
                Ok(body) => body.error,
                Err(_) => format!("the bot request failed (HTTP {status})"),
            };
            return Err(RequestFailure {
                message: format!("{}: {message}", self.url),
                retryable: false,
            });
        }
        response
            .body_mut()
            .read_json::<BotResponse>()
            .map_err(|error| RequestFailure {
                message: format!("{} returned an unreadable body: {error}", self.url),
                retryable: false,
            })
    }
}

impl Engine for HttpEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
        let reply = self.request_move(&board.export_san())?;
        apply_reply(board, &reply.san, &reply.fen)
    }
}

fn is_gateway_error(status: u16) -> bool {
    matches!(status, 502..=504)
}

/// Play the returned SAN and confirm it produced the position the bot claims.
/// The FEN check is what catches a reply that is legal here but answers a
/// different game than the one that was sent.
fn apply_reply(board: &Board, san: &str, fen: &str) -> Result<Move, String> {
    let mut next = board.clone();
    let chosen = next.san_to_move(san)?;
    if next.to_fen() != fen {
        return Err(format!(
            "bot returned a mismatched position: expected {}, got {fen}",
            next.to_fen()
        ));
    }
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_an_http_url() {
        assert!(HttpEngine::new("bot", "apps.example.test/move").is_err());
        assert!(HttpEngine::new("bot", "https://apps.example.test/move").is_ok());
    }

    #[test]
    fn retries_only_gateway_failures() {
        assert!(is_gateway_error(502));
        assert!(is_gateway_error(504));
        assert!(!is_gateway_error(500));
        assert!(!is_gateway_error(200));
    }

    #[test]
    fn accepts_a_reply_only_when_its_fen_matches() {
        let board = Board::import_san("1. e4 e5").unwrap();
        let mut expected = board.clone();
        let played = expected.san_to_move("Nf3").unwrap();

        assert_eq!(
            apply_reply(&board, "Nf3", &expected.to_fen()).unwrap(),
            played
        );
        assert!(apply_reply(&board, "Nf3", &board.to_fen()).is_err());
        assert!(apply_reply(&board, "Nf6", &expected.to_fen()).is_err());
    }
}
