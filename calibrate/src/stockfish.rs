use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub struct Stockfish {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    name: String,
    nodes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UciAnalysis {
    pub best_move: String,
    pub expected_score: f64,
}

impl Stockfish {
    pub fn start(path: &Path, nodes: u64, hash_mb: u32) -> Result<Self, String> {
        if nodes == 0 || hash_mb == 0 {
            return Err("Stockfish nodes and hash size must be greater than zero".to_string());
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
            .ok_or_else(|| "could not open Stockfish stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "could not open Stockfish stdout".to_string())?;
        let mut engine = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            name: "Stockfish".to_string(),
            nodes,
        };

        engine.send("uci")?;
        loop {
            let line = engine.read_line()?;
            if let Some(name) = line.strip_prefix("id name ") {
                engine.name = name.to_string();
            }
            if line == "uciok" {
                break;
            }
        }
        engine.send("setoption name Threads value 1")?;
        engine.send(&format!("setoption name Hash value {hash_mb}"))?;
        engine.send("setoption name UCI_ShowWDL value true")?;
        engine.ready()?;
        Ok(engine)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Expected score for the root side, optionally with one forced root move.
    pub fn expected_score(
        &mut self,
        fen: &str,
        uci_prefix: &[String],
        search_move: Option<&str>,
    ) -> Result<f64, String> {
        self.analyze(fen, uci_prefix, search_move)
            .map(|analysis| analysis.expected_score)
    }

    pub fn analyze(
        &mut self,
        fen: &str,
        uci_prefix: &[String],
        search_move: Option<&str>,
    ) -> Result<UciAnalysis, String> {
        if fen.contains(['\n', '\r']) {
            return Err("FEN contains a newline".to_string());
        }
        if uci_prefix
            .iter()
            .any(|mv| mv.contains(char::is_whitespace) || !(4..=5).contains(&mv.len()))
        {
            return Err("invalid UCI move in position prefix".to_string());
        }
        if search_move
            .is_some_and(|mv| mv.contains(['\n', '\r', ' ']) || !(4..=5).contains(&mv.len()))
        {
            return Err("invalid forced UCI move".to_string());
        }

        // Each move gets an independent node budget. Clearing the transposition
        // table prevents an earlier reference/human/bot search from donating
        // cached work to a later move or even to the next sampled position.
        self.send("setoption name Clear Hash")?;
        self.ready()?;
        if uci_prefix.is_empty() {
            self.send(&format!("position fen {fen}"))?;
        } else {
            // Replaying from startpos gives the reference engine the same
            // repetition state as the human and candidate engine.
            self.send(&format!("position startpos moves {}", uci_prefix.join(" ")))?;
        }
        let command = match search_move {
            Some(mv) => format!("go nodes {} searchmoves {mv}", self.nodes),
            None => format!("go nodes {}", self.nodes),
        };
        self.send(&command)?;

        let mut last_wdl = None;
        let best_move = loop {
            let line = self.read_line()?;
            if line.starts_with("info ") {
                if let Some(wdl) = parse_wdl(&line) {
                    last_wdl = Some(wdl);
                }
            } else if let Some(rest) = line.strip_prefix("bestmove ") {
                let mv = rest
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| "Stockfish returned an empty bestmove".to_string())?;
                break mv.to_string();
            }
        };
        let (wins, draws, losses) = last_wdl.ok_or_else(|| {
            "Stockfish returned no WDL score; use a build supporting UCI_ShowWDL".to_string()
        })?;
        let total = f64::from(wins + draws + losses);
        if total == 0.0 {
            return Err("Stockfish returned an empty WDL distribution".to_string());
        }
        Ok(UciAnalysis {
            best_move,
            expected_score: (f64::from(wins) + 0.5 * f64::from(draws)) / total,
        })
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
            .map_err(|error| format!("could not write to Stockfish: {error}"))
    }

    fn read_line(&mut self) -> Result<String, String> {
        let mut line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut line)
            .map_err(|error| format!("could not read Stockfish output: {error}"))?;
        if bytes == 0 {
            return Err("Stockfish exited unexpectedly".to_string());
        }
        Ok(line.trim().to_string())
    }
}

impl Drop for Stockfish {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

fn parse_wdl(line: &str) -> Option<(u32, u32, u32)> {
    let tokens: Vec<_> = line.split_whitespace().collect();
    let index = tokens.iter().rposition(|token| *token == "wdl")?;
    Some((
        tokens.get(index + 1)?.parse().ok()?,
        tokens.get(index + 2)?.parse().ok()?,
        tokens.get(index + 3)?.parse().ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_last_wdl_triplet_on_an_info_line() {
        assert_eq!(
            parse_wdl("info depth 12 score cp 31 wdl 125 850 25 nodes 10 pv e2e4"),
            Some((125, 850, 25))
        );
        assert_eq!(parse_wdl("info depth 1 score cp 0"), None);
    }
}
