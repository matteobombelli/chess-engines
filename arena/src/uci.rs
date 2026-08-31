//! A limited-strength UCI engine used as a rating ladder opponent.
//!
//! The process protocol is deliberately the same shape as the move-quality
//! calibration's Stockfish driver, but this one plays whole games: it holds no
//! opinion about scores, only about the move to play under a fixed movetime and
//! a fixed `UCI_Elo` rung.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use chess_core::{Board, Move};

use crate::Engine;

/// Stockfish's own `UCI_Elo` limits. A rung outside them is silently clamped by
/// the engine, which would quietly mislabel every game played against it.
pub const MIN_UCI_ELO: u32 = 1_320;
pub const MAX_UCI_ELO: u32 = 3_190;

pub struct UciEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    version: String,
    name: String,
    movetime_ms: u64,
    /// The move history this process was last given, including its own reply.
    /// A position that does not extend it belongs to a new game.
    last_history: Option<Vec<String>>,
}

impl UciEngine {
    pub fn start(path: &Path, uci_elo: u32, movetime_ms: u64) -> Result<Self, String> {
        if !(MIN_UCI_ELO..=MAX_UCI_ELO).contains(&uci_elo) {
            return Err(format!(
                "UCI_Elo {uci_elo} is outside the {MIN_UCI_ELO}-{MAX_UCI_ELO} ladder"
            ));
        }
        if movetime_ms == 0 {
            return Err("movetime must be greater than zero".to_string());
        }
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("could not start {}: {error}", path.display()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "could not open engine stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "could not open engine stdout".to_string())?;
        let mut engine = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            version: "UCI engine".to_string(),
            name: String::new(),
            movetime_ms,
            last_history: None,
        };

        engine.send("uci")?;
        loop {
            let line = engine.read_line()?;
            if let Some(name) = line.strip_prefix("id name ") {
                engine.version = name.to_string();
            }
            if line == "uciok" {
                break;
            }
        }
        engine.name = format!("{}[UCI_Elo {uci_elo}]", engine.version);
        engine.send("setoption name Threads value 1")?;
        engine.send("setoption name Hash value 16")?;
        engine.send("setoption name UCI_LimitStrength value true")?;
        engine.send(&format!("setoption name UCI_Elo value {uci_elo}"))?;
        engine.ready()?;
        Ok(engine)
    }

    /// The engine's `id name`, without the rung suffix carried by `name`.
    pub fn version(&self) -> &str {
        &self.version
    }

    fn new_game(&mut self) -> Result<(), String> {
        self.send("ucinewgame")?;
        self.ready()
    }

    fn ready(&mut self) -> Result<(), String> {
        self.send("isready")?;
        loop {
            if self.read_line()? == "readyok" {
                return Ok(());
            }
        }
    }

    fn send(&mut self, command: &str) -> Result<(), String> {
        writeln!(self.stdin, "{command}")
            .and_then(|_| self.stdin.flush())
            .map_err(|error| format!("could not write to the UCI engine: {error}"))
    }

    fn read_line(&mut self) -> Result<String, String> {
        let mut line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut line)
            .map_err(|error| format!("could not read UCI engine output: {error}"))?;
        if bytes == 0 {
            return Err("the UCI engine exited unexpectedly".to_string());
        }
        Ok(line.trim().to_string())
    }
}

impl Engine for UciEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(&mut self, board: &Board) -> Result<Move, String> {
        let history = uci_history(board)?;
        let continues_game = self
            .last_history
            .as_ref()
            .is_some_and(|last| history.starts_with(last));
        if !continues_game {
            self.new_game()?;
        }
        if history.is_empty() {
            self.send("position startpos")?;
        } else {
            self.send(&format!("position startpos moves {}", history.join(" ")))?;
        }
        self.send(&format!("go movetime {}", self.movetime_ms))?;
        let best_move = loop {
            let line = self.read_line()?;
            if let Some(best_move) = parse_bestmove(&line) {
                break best_move.to_string();
            }
        };
        let chosen = board
            .move_from_uci(&best_move)
            .map_err(|error| format!("{} answered with {best_move:?}: {error}", self.name))?;
        let mut played = history;
        played.push(best_move);
        self.last_history = Some(played);
        Ok(chosen)
    }
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

/// The game so far as UCI moves, replayed from the recorded SAN.
///
/// The engine is always given `position startpos moves ...` so that it sees the
/// same repetition and fifty-move history the arena is scoring.
pub fn uci_history(board: &Board) -> Result<Vec<String>, String> {
    let mut replay = Board::import_san("").expect("the standard initial position is valid");
    let mut history = Vec::with_capacity(board.san_history.len());
    for san in &board.san_history {
        history.push(replay.san_to_move(san)?.to_uci());
    }
    Ok(history)
}

fn parse_bestmove(line: &str) -> Option<&str> {
    line.strip_prefix("bestmove ")?
        .split_whitespace()
        .next()
        .filter(|best_move| *best_move != "(none)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_move_out_of_a_bestmove_line() {
        assert_eq!(parse_bestmove("bestmove e2e4 ponder e7e5"), Some("e2e4"));
        assert_eq!(parse_bestmove("bestmove a7a8q"), Some("a7a8q"));
        assert_eq!(parse_bestmove("bestmove (none)"), None);
        assert_eq!(parse_bestmove("info depth 4 score cp 12"), None);
    }

    #[test]
    fn maps_a_bestmove_onto_the_legal_move_it_names() {
        let board = Board::import_san("1. e4 e5 2. Nf3").unwrap();
        let chosen = board.move_from_uci("b8c6").unwrap();
        assert_eq!(chosen.to_uci(), "b8c6");
        assert!(board.get_legal_moves().contains(&chosen));
        assert!(board.move_from_uci("e2e4").is_err());
        assert!(board.move_from_uci("wat").is_err());
    }

    #[test]
    fn replays_the_san_history_as_the_uci_prefix() {
        let board = Board::import_san("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6").unwrap();
        assert_eq!(
            uci_history(&board).unwrap(),
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
        );
        assert!(
            uci_history(&Board::import_san("").unwrap())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn rejects_rungs_outside_the_ladder() {
        let path = Path::new("/usr/games/stockfish");
        assert!(UciEngine::start(path, MIN_UCI_ELO - 1, 100).is_err());
        assert!(UciEngine::start(path, MAX_UCI_ELO + 1, 100).is_err());
        assert!(UciEngine::start(path, 1_500, 0).is_err());
    }
}
