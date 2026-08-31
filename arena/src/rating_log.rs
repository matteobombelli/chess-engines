//! Durable, resumable JSONL log of completed opening pairs.
//!
//! One header line pins the identity and configuration of an evaluation; every
//! following line is one completed color-reversed opening pair. A pair is
//! durable only once its terminating newline reached the disk, so a run that
//! crashes mid-pair resumes from the last committed record.
//!
//! The header is the union of the AlphaMini release gate's identity and the
//! full-game rating ladder's, so each of them writes only the fields it has.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use chess_core::Color;
use serde::{Deserialize, Serialize};

use crate::{AlphaMiniMetrics, GameResult, OpeningPairResult, Termination};

pub const PAIRED_LOG_HEADER_SCHEMA: &str = "alphamini-paired-evaluation-v1";
pub const PAIRED_LOG_PAIR_SCHEMA: &str = "alphamini-paired-opening-result-v1";
pub const RATING_LOG_HEADER_SCHEMA: &str = "full-game-elo-evaluation-v1";

/// A field that does not apply to an evaluation is left out of its header
/// rather than written as null, which keeps a release-gate header exactly the
/// field set the training tooling reads
/// (`alphamini-train/src/alphamini_train/evaluation.py`). The gate's own
/// `opponent_model_sha256` is the one exception: that reader requires the key
/// even for a single-model run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedLogHeader {
    pub schema: String,
    pub engine_a: String,
    pub engine_b: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    pub opponent_model_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opening_suite_sha256: Option<String>,
    pub opening_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<u8>,
    pub seed: u64,
    pub max_plies: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulations: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpuct_ppm: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fpu_reduction_ppm: Option<u32>,
    pub bootstrap_samples: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_lower_score_ppm: Option<u32>,
    pub minimax_v1_move_digest: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_binary_sha256: Option<String>,
    pub target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_device: Option<String>,
    pub exploratory: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stockfish_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uci_elo: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub movetime_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bot_url: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredGame {
    pub winner: Option<String>,
    pub termination: String,
    pub plies: u32,
}

impl StoredGame {
    pub fn from_game(game: &GameResult) -> Self {
        Self {
            winner: game.winner.map(|color| match color {
                Color::White => "white".to_string(),
                Color::Black => "black".to_string(),
            }),
            termination: match game.termination {
                Termination::Checkmate => "checkmate",
                Termination::Stalemate => "stalemate",
                Termination::InsufficientMaterial => "insufficient_material",
                Termination::ThreefoldRepetition => "threefold_repetition",
                Termination::FiftyMoveRule => "fifty_move_rule",
                Termination::PlyLimit => "ply_limit",
            }
            .to_string(),
            plies: game.plies,
        }
    }

    pub fn into_game(self) -> Result<GameResult, String> {
        let winner = match self.winner.as_deref() {
            None => None,
            Some("white") => Some(Color::White),
            Some("black") => Some(Color::Black),
            Some(value) => return Err(format!("invalid stored winner {value:?}")),
        };
        let termination = match self.termination.as_str() {
            "checkmate" => Termination::Checkmate,
            "stalemate" => Termination::Stalemate,
            "insufficient_material" => Termination::InsufficientMaterial,
            "threefold_repetition" => Termination::ThreefoldRepetition,
            "fifty_move_rule" => Termination::FiftyMoveRule,
            "ply_limit" => Termination::PlyLimit,
            value => return Err(format!("invalid stored termination {value:?}")),
        };
        if (winner.is_some()) != (termination == Termination::Checkmate) {
            return Err("stored winner must be present exactly for checkmate".to_string());
        }
        Ok(GameResult {
            winner,
            termination,
            plies: self.plies,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredPair {
    pub schema: String,
    pub opening_id: String,
    pub engine_a_as_white: StoredGame,
    pub engine_a_as_black: StoredGame,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<AlphaMiniMetrics>,
}

impl StoredPair {
    pub fn from_pair(pair: &OpeningPairResult, metrics: Option<AlphaMiniMetrics>) -> Self {
        Self {
            schema: PAIRED_LOG_PAIR_SCHEMA.to_string(),
            opening_id: pair.opening_id.clone(),
            engine_a_as_white: StoredGame::from_game(&pair.engine_a_as_white),
            engine_a_as_black: StoredGame::from_game(&pair.engine_a_as_black),
            metrics,
        }
    }

    pub fn into_pair(self) -> Result<(OpeningPairResult, Option<AlphaMiniMetrics>), String> {
        if self.schema != PAIRED_LOG_PAIR_SCHEMA {
            return Err(format!("unsupported pair log schema {:?}", self.schema));
        }
        let as_white = self.engine_a_as_white.into_game()?;
        let as_black = self.engine_a_as_black.into_game()?;
        let points = |game: &GameResult, color: Color| match game.winner {
            Some(winner) if winner == color => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        };
        let score = (points(&as_white, Color::White) + points(&as_black, Color::Black)) / 2.0;
        Ok((
            OpeningPairResult {
                opening_id: self.opening_id,
                engine_a_as_white: as_white,
                engine_a_as_black: as_black,
                score,
            },
            self.metrics,
        ))
    }
}

/// Create the log with `expected` as its header, or reopen an existing log and
/// return its committed pairs. Reopening refuses any header drift, so two
/// different evaluations can never be mixed into one log.
pub fn load_or_create_pair_log(
    path: &Path,
    expected: &PairedLogHeader,
) -> Result<Vec<(OpeningPairResult, Option<AlphaMiniMetrics>)>, String> {
    if !path.exists() {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("could not create {}: {error}", path.display()))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, expected)
            .map_err(|error| format!("could not encode pair-log header: {error}"))?;
        writer
            .write_all(b"\n")
            .and_then(|_| writer.flush())
            .and_then(|_| writer.get_ref().sync_all())
            .map_err(|error| format!("could not commit {}: {error}", path.display()))?;
        sync_parent_directory(path)?;
        return Ok(Vec::new());
    }

    let committed = read_pair_log_recovering_torn_tail(path)?;
    let mut lines = committed.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| format!("pair log {} is empty", path.display()))?;
    let actual: PairedLogHeader = serde_json::from_str(header_line)
        .map_err(|error| format!("invalid pair-log header in {}: {error}", path.display()))?;
    if actual != *expected {
        return Err(format!(
            "pair-log identity/config mismatch in {}; refuse to mix evaluations",
            path.display()
        ));
    }

    let mut records = Vec::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            return Err(format!(
                "blank line {} in pair log {}; refusing ambiguous recovery",
                line_index + 2,
                path.display()
            ));
        }
        let stored: StoredPair = serde_json::from_str(line).map_err(|error| {
            format!(
                "invalid durable pair record on line {} of {}: {error}",
                line_index + 2,
                path.display()
            )
        })?;
        let (pair, metrics) = stored.into_pair()?;
        let expected_id = expected.opening_ids.get(records.len()).ok_or_else(|| {
            format!(
                "pair log {} contains more pairs than its suite",
                path.display()
            )
        })?;
        if &pair.opening_id != expected_id {
            return Err(format!(
                "pair log {} is not the exact opening-suite prefix at record {}",
                path.display(),
                records.len()
            ));
        }
        records.push((pair, metrics));
    }
    if !records.is_empty() {
        crate::paired_report_from_results(
            actual.engine_a,
            actual.engine_b,
            records.iter().map(|(pair, _)| pair.clone()).collect(),
        )
        .map_err(|error| format!("invalid recorded pair results: {error}"))?;
    }
    Ok(records)
}

/// A pair is durable only after its terminating newline and `sync_data` have
/// completed. A crash can therefore leave bytes after the last newline; drop
/// exactly that uncommitted suffix while preserving and validating every
/// newline-terminated record. Corruption inside the committed prefix remains
/// a hard error in the JSON/schema checks of the caller.
pub fn read_pair_log_recovering_torn_tail(path: &Path) -> Result<String, String> {
    let mut bytes = fs::read(path)
        .map_err(|error| format!("could not read pair log {}: {error}", path.display()))?;
    let header_end = bytes
        .iter()
        .position(|&byte| byte == b'\n')
        .ok_or_else(|| {
            format!(
                "pair log {} has no committed header newline",
                path.display()
            )
        })?;
    if bytes.last() != Some(&b'\n') {
        let committed_len = bytes
            .iter()
            .rposition(|&byte| byte == b'\n')
            .map(|index| index + 1)
            .expect("header newline exists");
        debug_assert!(committed_len > header_end);
        let file = OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(|error| format!("could not open {} for recovery: {error}", path.display()))?;
        file.set_len(committed_len as u64)
            .and_then(|_| file.sync_all())
            .map_err(|error| {
                format!(
                    "could not discard torn final pair record from {}: {error}",
                    path.display()
                )
            })?;
        bytes.truncate(committed_len);
    }
    String::from_utf8(bytes)
        .map_err(|error| format!("pair log {} is not UTF-8: {error}", path.display()))
}

pub fn append_pair_log(path: &Path, pair: &StoredPair) -> Result<(), String> {
    let file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|error| format!("could not append {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, pair)
        .map_err(|error| format!("could not encode pair result: {error}"))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .and_then(|_| writer.get_ref().sync_data())
        .map_err(|error| {
            format!(
                "could not commit pair result to {}: {error}",
                path.display()
            )
        })
}

fn sync_parent_directory(path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            format!(
                "could not fsync parent directory {}: {error}",
                parent.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MINIMAX_V1_MOVE_DIGEST;

    fn header(opening_ids: &[&str]) -> PairedLogHeader {
        PairedLogHeader {
            schema: RATING_LOG_HEADER_SCHEMA.to_string(),
            engine_a: "MinimaxDepth3V1".to_string(),
            engine_b: "Stockfish 17.1".to_string(),
            model_sha256: None,
            opponent_model_sha256: None,
            opening_suite_sha256: None,
            opening_ids: opening_ids.iter().map(|id| id.to_string()).collect(),
            depth: Some(3),
            seed: 1,
            max_plies: 1_000,
            simulations: None,
            time_ms: None,
            batch_size: None,
            cpuct_ppm: None,
            fpu_reduction_ppm: None,
            bootstrap_samples: 20_000,
            required_lower_score_ppm: None,
            minimax_v1_move_digest: MINIMAX_V1_MOVE_DIGEST,
            evaluation_binary_sha256: None,
            target: "x86_64-linux".to_string(),
            inference_device: None,
            exploratory: true,
            stockfish_version: Some("Stockfish 17.1".to_string()),
            uci_elo: Some(1_500),
            movetime_ms: Some(100),
            bot_url: None,
        }
    }

    fn pair(opening_id: &str) -> OpeningPairResult {
        OpeningPairResult {
            opening_id: opening_id.to_string(),
            engine_a_as_white: GameResult {
                winner: Some(Color::White),
                termination: Termination::Checkmate,
                plies: 41,
            },
            engine_a_as_black: GameResult {
                winner: None,
                termination: Termination::FiftyMoveRule,
                plies: 200,
            },
            score: 0.75,
        }
    }

    /// `alphamini-train` rejects a gate log whose header carries any field it
    /// does not know, so the rating ladder's fields must stay out of one.
    #[test]
    fn a_gate_header_keeps_exactly_the_fields_the_training_tooling_reads() {
        let mut gate = header(&["opening-1"]);
        gate.schema = PAIRED_LOG_HEADER_SCHEMA.to_string();
        gate.model_sha256 = Some("a".repeat(64));
        gate.opening_suite_sha256 = Some("b".repeat(64));
        gate.simulations = Some(10_000);
        gate.time_ms = Some(9_000);
        gate.batch_size = Some(8);
        gate.cpuct_ppm = Some(1_250_000);
        gate.fpu_reduction_ppm = Some(300_000);
        gate.required_lower_score_ppm = Some(500_000);
        gate.evaluation_binary_sha256 = Some("c".repeat(64));
        gate.inference_device = Some("onnx-cpu".to_string());
        gate.stockfish_version = None;
        gate.uci_elo = None;
        gate.movetime_ms = None;
        gate.bot_url = None;

        let encoded: serde_json::Value = serde_json::to_value(&gate).unwrap();
        let mut fields: Vec<&str> = encoded
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        fields.sort_unstable();
        assert_eq!(
            fields,
            [
                "batch_size",
                "bootstrap_samples",
                "cpuct_ppm",
                "depth",
                "engine_a",
                "engine_b",
                "evaluation_binary_sha256",
                "exploratory",
                "fpu_reduction_ppm",
                "inference_device",
                "max_plies",
                "minimax_v1_move_digest",
                "model_sha256",
                "opening_ids",
                "opening_suite_sha256",
                "opponent_model_sha256",
                "required_lower_score_ppm",
                "schema",
                "seed",
                "simulations",
                "target",
                "time_ms",
            ]
        );
        assert_eq!(
            serde_json::from_value::<PairedLogHeader>(encoded).unwrap(),
            gate
        );
    }

    #[test]
    fn rating_header_round_trips_and_pins_identity() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("elo-1500.jsonl");
        let header = header(&["opening-1", "opening-2"]);
        assert_eq!(load_or_create_pair_log(&path, &header).unwrap(), Vec::new());

        let first = pair("opening-1");
        append_pair_log(&path, &StoredPair::from_pair(&first, None)).unwrap();
        assert_eq!(
            load_or_create_pair_log(&path, &header).unwrap(),
            vec![(first.clone(), None)]
        );

        let mut other_rung = header.clone();
        other_rung.uci_elo = Some(1_650);
        assert!(load_or_create_pair_log(&path, &other_rung).is_err());
    }

    #[test]
    fn a_torn_final_record_is_discarded_and_the_prefix_survives() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("elo-1500.jsonl");
        let header = header(&["opening-1", "opening-2"]);
        load_or_create_pair_log(&path, &header).unwrap();
        let first = pair("opening-1");
        append_pair_log(&path, &StoredPair::from_pair(&first, None)).unwrap();
        let committed = fs::read(&path).unwrap();

        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(br#"{"schema":"alphamini-paired-openi"#)
            .unwrap();
        file.sync_all().unwrap();
        drop(file);

        assert_eq!(
            load_or_create_pair_log(&path, &header).unwrap(),
            vec![(first, None)]
        );
        assert_eq!(fs::read(&path).unwrap(), committed);
    }

    #[test]
    fn a_log_that_is_not_the_suite_prefix_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("elo-1500.jsonl");
        let header = header(&["opening-1", "opening-2"]);
        load_or_create_pair_log(&path, &header).unwrap();
        append_pair_log(&path, &StoredPair::from_pair(&pair("opening-2"), None)).unwrap();
        assert!(load_or_create_pair_log(&path, &header).is_err());
    }
}
