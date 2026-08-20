//! Pure, post-hoc sealing of calibration artifacts produced by the early v2
//! writer.
//!
//! The first v2 writer captured replayable rows but did not embed the complete
//! experiment or shard identity. This module upgrades such an artifact without
//! editing it in place. The returned artifact records the exact source bytes,
//! the contemporaneous capture manifest, and the binary that performed the
//! attestation. Callers are responsible for writing the returned value to a
//! new path with create-new semantics.

use std::collections::HashSet;

use artifact_io::sha256_bytes;
use chess_core::Board;
use serde::{Deserialize, Serialize};

use crate::artifact::validate_analysis_artifact;
use crate::{ANALYSIS_FORMAT_V2, AnalysisArtifact, AnalysisExperimentV2, AnalysisRow};

pub const POST_HOC_ATTESTATION_METHOD_V1: &str = "captured-run-metadata-v1";
pub const CALIBRATION_CAPTURE_MANIFEST_SCHEMA_V1: &str = "calibration-capture-v1";

/// Contemporaneous evidence required to reconstruct a missing v2 experiment
/// identity. Commands remain literal strings so the operator can audit the
/// exact invocation; machine-checked identities are stored separately.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationCaptureManifestV1 {
    pub schema: String,
    /// Exact early-v2 `calibrate` executable used to produce the source shard.
    pub producer_binary_sha256: String,
    pub stockfish_binary_sha256: String,
    /// Ordered content hashes, preserving the sampling order and primary versus
    /// exclusion roles used by the run.
    pub corpus_sha256: Vec<String>,
    pub exclude_corpus_sha256: Vec<String>,
    /// One nonempty shell invocation for every declared shard, in shard order.
    pub exact_commands: Vec<String>,
}

/// Provenance added only when an early, otherwise immutable v2 artifact is
/// sealed after its analysis process has completed.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PostHocAttestationV2 {
    /// Always true. Keeping the marker explicit makes native and post-hoc v2
    /// artifacts visibly different in both JSON and downstream reports.
    pub post_hoc_attested: bool,
    /// Versioned procedure used to bind captured run evidence to the source.
    pub method: String,
    /// SHA-256 of the exact, pre-attestation artifact bytes.
    pub source_artifact_sha256: String,
    /// The format declared by those source bytes. This procedure accepts only
    /// the early format-v2 writer, never legacy FEN-only v1 data.
    pub source_format_version: u32,
    /// SHA-256 of the executable performing this transformation.
    pub attestor_binary_sha256: String,
    /// SHA-256 of the immutable evidence record from which the missing run
    /// identity was recovered.
    pub capture_manifest_sha256: String,
}

/// Byte-level evidence supplied to [`attest_legacy_v2_artifact`].
///
/// Both claimed digests are checked against their corresponding byte slices.
/// The capture bytes must deserialize as [`CalibrationCaptureManifestV1`] and
/// its machine identities are cross-checked against [`AnalysisExperimentV2`].
#[derive(Clone, Copy, Debug)]
pub struct PostHocAttestationInputV2<'a> {
    pub capture_manifest_bytes: &'a [u8],
    pub expected_capture_manifest_sha256: &'a str,
    pub attestor_binary_sha256: &'a str,
}

/// Seal an artifact emitted by the early v2 writer and return a new value.
///
/// `source_json` must be the exact bytes whose digest is supplied. They are
/// deserialized again and required to equal `source`, preventing a caller from
/// binding provenance for one file to a different in-memory artifact. The
/// input is borrowed and cloned only after all immutable-source checks pass.
pub fn attest_legacy_v2_artifact(
    source: &AnalysisArtifact,
    source_json: &[u8],
    expected_source_artifact_sha256: &str,
    experiment: AnalysisExperimentV2,
    shard_index: u64,
    input: PostHocAttestationInputV2<'_>,
) -> Result<AnalysisArtifact, String> {
    require_canonical_sha256("expected source artifact", expected_source_artifact_sha256)?;
    let actual_source_sha256 = sha256_bytes(source_json);
    if actual_source_sha256 != expected_source_artifact_sha256 {
        return Err(format!(
            "source artifact SHA-256 mismatch: expected {expected_source_artifact_sha256}, got {actual_source_sha256}"
        ));
    }

    require_canonical_sha256(
        "expected capture manifest",
        input.expected_capture_manifest_sha256,
    )?;
    if input.capture_manifest_bytes.is_empty() {
        return Err("capture manifest is empty".to_string());
    }
    let actual_capture_manifest_sha256 = sha256_bytes(input.capture_manifest_bytes);
    if actual_capture_manifest_sha256 != input.expected_capture_manifest_sha256 {
        return Err(format!(
            "capture manifest SHA-256 mismatch: expected {}, got {actual_capture_manifest_sha256}",
            input.expected_capture_manifest_sha256
        ));
    }
    require_canonical_sha256("attestor binary", input.attestor_binary_sha256)?;

    let capture_manifest: CalibrationCaptureManifestV1 =
        serde_json::from_slice(input.capture_manifest_bytes).map_err(|error| {
            format!("capture manifest is not valid calibration-capture-v1 JSON: {error}")
        })?;
    validate_capture_manifest(&capture_manifest, &experiment)?;

    let parsed_source: AnalysisArtifact = serde_json::from_slice(source_json)
        .map_err(|error| format!("source artifact is not valid analysis JSON: {error}"))?;
    if &parsed_source != source {
        return Err(
            "source artifact bytes do not deserialize to the supplied source value".to_string(),
        );
    }
    require_identity_fields_absent_from_source_json(source_json)?;
    validate_unsealed_v2_source(source)?;

    let attestation = PostHocAttestationV2 {
        post_hoc_attested: true,
        method: POST_HOC_ATTESTATION_METHOD_V1.to_string(),
        source_artifact_sha256: actual_source_sha256,
        source_format_version: ANALYSIS_FORMAT_V2,
        attestor_binary_sha256: input.attestor_binary_sha256.to_string(),
        capture_manifest_sha256: actual_capture_manifest_sha256,
    };
    validate_post_hoc_attestation(&attestation)?;

    let mut sealed = source.clone();
    sealed.metadata.experiment = Some(experiment);
    sealed.metadata.shard_index = Some(shard_index);
    sealed.metadata.attestation = Some(attestation);
    validate_analysis_artifact(&sealed)
        .map_err(|error| format!("attested artifact failed strict v2 validation: {error}"))?;
    Ok(sealed)
}

/// Validate the fixed fields and canonical hashes of stored post-hoc
/// provenance. Strict artifact validation calls this for attested v2 data.
pub fn validate_post_hoc_attestation(attestation: &PostHocAttestationV2) -> Result<(), String> {
    if !attestation.post_hoc_attested {
        return Err("post-hoc attestation marker must be true".to_string());
    }
    if attestation.method != POST_HOC_ATTESTATION_METHOD_V1 {
        return Err(format!(
            "unsupported post-hoc attestation method {:?}",
            attestation.method
        ));
    }
    if attestation.source_format_version != ANALYSIS_FORMAT_V2 {
        return Err(format!(
            "post-hoc attestation source format must be v2, got v{}",
            attestation.source_format_version
        ));
    }
    require_canonical_sha256(
        "post-hoc source artifact",
        &attestation.source_artifact_sha256,
    )?;
    require_canonical_sha256(
        "post-hoc attestor binary",
        &attestation.attestor_binary_sha256,
    )?;
    require_canonical_sha256(
        "post-hoc capture manifest",
        &attestation.capture_manifest_sha256,
    )?;
    Ok(())
}

/// Validate captured run evidence and require it to describe the experiment
/// identity that will be attached to the source artifact.
pub fn validate_capture_manifest(
    manifest: &CalibrationCaptureManifestV1,
    experiment: &AnalysisExperimentV2,
) -> Result<(), String> {
    if manifest.schema != CALIBRATION_CAPTURE_MANIFEST_SCHEMA_V1 {
        return Err(format!(
            "unsupported calibration capture schema {:?}",
            manifest.schema
        ));
    }
    require_canonical_sha256("capture producer binary", &manifest.producer_binary_sha256)?;
    require_canonical_sha256(
        "capture Stockfish binary",
        &manifest.stockfish_binary_sha256,
    )?;
    if manifest.corpus_sha256.is_empty() {
        return Err("capture manifest must contain at least one primary corpus hash".to_string());
    }
    for digest in manifest
        .corpus_sha256
        .iter()
        .chain(manifest.exclude_corpus_sha256.iter())
    {
        require_canonical_sha256("capture corpus", digest)?;
    }
    let shard_count = usize::try_from(experiment.sampling.shard_count)
        .map_err(|_| "experiment shard count does not fit this platform".to_string())?;
    if manifest.exact_commands.len() != shard_count
        || manifest
            .exact_commands
            .iter()
            .any(|command| command.trim().is_empty())
    {
        return Err(format!(
            "capture manifest must contain exactly {} nonempty shard commands",
            experiment.sampling.shard_count
        ));
    }
    if manifest.producer_binary_sha256 != experiment.calibration_binary_sha256 {
        return Err(
            "capture producer binary does not match the experiment calibration binary".to_string(),
        );
    }
    if manifest.stockfish_binary_sha256 != experiment.reference.binary_sha256 {
        return Err(
            "capture Stockfish binary does not match the experiment reference binary".to_string(),
        );
    }
    if manifest.corpus_sha256 != experiment.corpus_sha256
        || manifest.exclude_corpus_sha256 != experiment.exclude_corpus_sha256
    {
        return Err("capture ordered corpus identities do not match the experiment".to_string());
    }
    Ok(())
}

fn require_identity_fields_absent_from_source_json(source_json: &[u8]) -> Result<(), String> {
    let value: serde_json::Value = serde_json::from_slice(source_json)
        .map_err(|error| format!("source artifact is not valid JSON: {error}"))?;
    let metadata = value
        .as_object()
        .and_then(|root| root.get("metadata"))
        .and_then(serde_json::Value::as_object)
        .ok_or("source artifact metadata is not a JSON object")?;
    for field in ["experiment", "shard_index", "attestation"] {
        if metadata.contains_key(field) {
            return Err(format!(
                "source artifact already contains reserved v2 identity field {field:?}"
            ));
        }
    }
    Ok(())
}

fn validate_unsealed_v2_source(source: &AnalysisArtifact) -> Result<(), String> {
    let metadata = &source.metadata;
    if metadata.format_version != ANALYSIS_FORMAT_V2 {
        return Err(format!(
            "post-hoc attestation requires an early format-v2 artifact, got v{}",
            metadata.format_version
        ));
    }
    if metadata.experiment.is_some()
        || metadata.shard_index.is_some()
        || metadata.attestation.is_some()
    {
        return Err("source artifact is already sealed or attested".to_string());
    }
    if metadata.target.trim().is_empty()
        || metadata.bot.trim().is_empty()
        || metadata.reference_engine.trim().is_empty()
        || metadata.reference_nodes_per_search == 0
    {
        return Err("source artifact has an empty identity or reference node budget".to_string());
    }
    if metadata.input_positions == 0
        || metadata.unique_games == 0
        || metadata.unique_games > metadata.input_positions
        || metadata.analyzed_unique_games > metadata.unique_games
    {
        return Err("source artifact has impossible game/position counts".to_string());
    }
    let accounted = source
        .rows
        .len()
        .checked_add(source.skipped_uninformative)
        .and_then(|count| count.checked_add(source.skipped_player_cap))
        .ok_or("source artifact position counts overflow")?;
    if accounted != metadata.input_positions {
        return Err(format!(
            "source artifact accounts for {accounted} positions, metadata says {}",
            metadata.input_positions
        ));
    }

    let analyzed_games = source
        .rows
        .iter()
        .map(|row| row.game_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    if analyzed_games != metadata.analyzed_unique_games {
        return Err(format!(
            "source artifact analyzed_unique_games is {}, rows contain {analyzed_games}",
            metadata.analyzed_unique_games
        ));
    }

    let mut row_ids = HashSet::new();
    for (row_index, row) in source.rows.iter().enumerate() {
        validate_immutable_row(row).map_err(|error| format!("source row {row_index}: {error}"))?;
        if !row_ids.insert((row.game_id.as_str(), row.ply)) {
            return Err(format!(
                "duplicate source row identity: game {:?} ply {}",
                row.game_id, row.ply
            ));
        }
    }
    Ok(())
}

fn validate_immutable_row(row: &AnalysisRow) -> Result<(), String> {
    if row.game_id.trim().is_empty() || row.actor_username.trim().is_empty() {
        return Err("row has an empty game or player identity".to_string());
    }
    if row.ply == 0 || row.uci_prefix.len() != usize::from(row.ply - 1) {
        return Err(format!(
            "game {:?} ply {} has {} prefix moves",
            row.game_id,
            row.ply,
            row.uci_prefix.len()
        ));
    }
    let board = Board::import_uci(&row.uci_prefix).map_err(|error| {
        format!(
            "game {:?} ply {} has an illegal prefix: {error}",
            row.game_id, row.ply
        )
    })?;
    let replayed_fen = board.to_fen();
    if replayed_fen != row.fen {
        return Err(format!(
            "game {:?} ply {} prefix/FEN mismatch: replayed {replayed_fen}, stored {}",
            row.game_id, row.ply, row.fen
        ));
    }
    for (label, mv) in [
        ("human", &row.human_move),
        ("bot", &row.bot_move),
        ("reference", &row.reference_move),
    ] {
        board.move_from_uci(mv).map_err(|error| {
            format!(
                "game {:?} ply {} has illegal {label} move {mv:?}: {error}",
                row.game_id, row.ply
            )
        })?;
    }

    let scores = [
        row.best_expected_score,
        row.human_expected_score,
        row.bot_expected_score,
        row.human_loss,
        row.bot_loss,
    ];
    if scores
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(format!(
            "game {:?} has a non-finite or out-of-range score",
            row.game_id
        ));
    }
    let human_loss = (row.best_expected_score - row.human_expected_score).max(0.0);
    let bot_loss = (row.best_expected_score - row.bot_expected_score).max(0.0);
    if (row.human_loss - human_loss).abs() > 1e-12 || (row.bot_loss - bot_loss).abs() > 1e-12 {
        return Err(format!(
            "game {:?} has loss values inconsistent with its scores",
            row.game_id
        ));
    }
    Ok(())
}

fn require_canonical_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!(
            "{label} SHA-256 must be exactly 64 lowercase hexadecimal characters"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::validate_analysis_artifact;
    use crate::identity::{aggregate_corpus_sha256, analysis_config_sha256};
    use crate::{AnalysisBotV2, AnalysisReferenceV2, AnalysisSamplingV2, PLAYER_SHARD_SCHEMA_V1};

    fn source() -> AnalysisArtifact {
        let prefix = vec!["e2e4".to_string(), "e7e5".to_string()];
        let fen = Board::import_uci(&prefix).unwrap().to_fen();
        serde_json::from_value(serde_json::json!({
            "metadata": {
                "format_version": ANALYSIS_FORMAT_V2,
                "target": "Chess.com rated standard 30+0 (TimeControl 1800)",
                "bot": "Minimax (depth 3)",
                "reference_engine": "Stockfish 17.1",
                "reference_nodes_per_search": 100_000,
                "input_positions": 1,
                "unique_games": 1,
                "analyzed_unique_games": 1
            },
            "skipped_uninformative": 0,
            "skipped_player_cap": 0,
            "rows": [{
                "game_id": "game-1",
                "actor_username": "alice",
                "actor_rating": 1500,
                "ply": 3,
                "uci_prefix": prefix,
                "fen": fen,
                "human_move": "g1f3",
                "bot_move": "f1c4",
                "reference_move": "d2d4",
                "best_expected_score": 0.6,
                "human_expected_score": 0.5,
                "bot_expected_score": 0.4,
                "human_loss": 0.1,
                "bot_loss": 0.2
            }]
        }))
        .unwrap()
    }

    fn experiment() -> AnalysisExperimentV2 {
        let mut experiment = AnalysisExperimentV2 {
            corpus_sha256: vec!["a".repeat(64)],
            exclude_corpus_sha256: vec![],
            corpus_digest_sha256: String::new(),
            analysis_config_sha256: String::new(),
            sampling: AnalysisSamplingV2 {
                positions_per_side: 1,
                positions_per_player: 3,
                analyzed_positions_per_player: 1,
                max_positions: None,
                minimum_rating: 200,
                maximum_rating: 3_200,
                minimum_ply: 1,
                maximum_ply: 60,
                sample_seed: 1,
                shard_count: 1,
                player_shard_schema: PLAYER_SHARD_SCHEMA_V1.to_string(),
                minimum_best_expected_score_ppm: 50_000,
                maximum_best_expected_score_ppm: 950_000,
            },
            bot: AnalysisBotV2::MinimaxFixed {
                depth: 3,
                baseline_move_digest: 1,
            },
            reference: AnalysisReferenceV2 {
                engine_name: "Stockfish 17.1".to_string(),
                binary_sha256: "b".repeat(64),
                nodes_per_search: 100_000,
                hash_mb: 128,
                threads: 1,
                show_wdl: true,
            },
            calibration_binary_sha256: "c".repeat(64),
        };
        experiment.corpus_digest_sha256 =
            aggregate_corpus_sha256(&experiment.corpus_sha256, &experiment.exclude_corpus_sha256)
                .unwrap();
        experiment.analysis_config_sha256 = analysis_config_sha256(&experiment).unwrap();
        experiment
    }

    fn capture_manifest(experiment: &AnalysisExperimentV2) -> Vec<u8> {
        serde_json::to_vec_pretty(&CalibrationCaptureManifestV1 {
            schema: CALIBRATION_CAPTURE_MANIFEST_SCHEMA_V1.to_string(),
            producer_binary_sha256: experiment.calibration_binary_sha256.clone(),
            stockfish_binary_sha256: experiment.reference.binary_sha256.clone(),
            corpus_sha256: experiment.corpus_sha256.clone(),
            exclude_corpus_sha256: experiment.exclude_corpus_sha256.clone(),
            exact_commands: vec!["calibrate analyze --shard-count 1 --shard-index 0".to_string()],
        })
        .unwrap()
    }

    fn seal(source: &AnalysisArtifact) -> Result<AnalysisArtifact, String> {
        let source_json = serde_json::to_vec_pretty(source).unwrap();
        let source_sha256 = sha256_bytes(&source_json);
        let experiment = experiment();
        let capture_manifest = capture_manifest(&experiment);
        let capture_sha256 = sha256_bytes(&capture_manifest);
        attest_legacy_v2_artifact(
            source,
            &source_json,
            &source_sha256,
            experiment,
            0,
            PostHocAttestationInputV2 {
                capture_manifest_bytes: &capture_manifest,
                expected_capture_manifest_sha256: &capture_sha256,
                attestor_binary_sha256: &"d".repeat(64),
            },
        )
    }

    #[test]
    fn seals_a_clone_and_preserves_the_source() {
        let source = source();
        let original = source.clone();
        let sealed = seal(&source).unwrap();

        assert_eq!(source, original);
        assert!(sealed.metadata.experiment.is_some());
        assert_eq!(sealed.metadata.shard_index, Some(0));
        let provenance = sealed.metadata.attestation.as_ref().unwrap();
        assert!(provenance.post_hoc_attested);
        assert_eq!(provenance.method, POST_HOC_ATTESTATION_METHOD_V1);
        assert_eq!(provenance.source_format_version, ANALYSIS_FORMAT_V2);
        let experiment = experiment();
        let capture_manifest = capture_manifest(&experiment);
        assert_eq!(
            provenance.capture_manifest_sha256,
            sha256_bytes(&capture_manifest)
        );
        validate_analysis_artifact(&sealed).unwrap();
    }

    #[test]
    fn requires_source_bytes_digest_and_value_to_agree() {
        let source = source();
        let source_json = serde_json::to_vec_pretty(&source).unwrap();
        let source_sha256 = sha256_bytes(&source_json);
        let experiment = experiment();
        let capture_manifest = capture_manifest(&experiment);
        let capture_sha256 = sha256_bytes(&capture_manifest);
        let attestor_sha256 = "d".repeat(64);
        let evidence = || PostHocAttestationInputV2 {
            capture_manifest_bytes: &capture_manifest,
            expected_capture_manifest_sha256: &capture_sha256,
            attestor_binary_sha256: &attestor_sha256,
        };

        assert!(
            attest_legacy_v2_artifact(
                &source,
                &source_json,
                &"0".repeat(64),
                experiment.clone(),
                0,
                evidence(),
            )
            .unwrap_err()
            .contains("source artifact SHA-256 mismatch")
        );

        let mut different_value = source.clone();
        different_value.metadata.target.push_str(" changed");
        assert!(
            attest_legacy_v2_artifact(
                &different_value,
                &source_json,
                &source_sha256,
                experiment,
                0,
                evidence(),
            )
            .unwrap_err()
            .contains("do not deserialize to the supplied source")
        );
    }

    #[test]
    fn rejects_bad_capture_digest_and_previously_sealed_source() {
        let source = source();
        let source_json = serde_json::to_vec_pretty(&source).unwrap();
        let source_sha256 = sha256_bytes(&source_json);
        let experiment = experiment();
        let capture_manifest = capture_manifest(&experiment);
        assert!(
            attest_legacy_v2_artifact(
                &source,
                &source_json,
                &source_sha256,
                experiment.clone(),
                0,
                PostHocAttestationInputV2 {
                    capture_manifest_bytes: &capture_manifest,
                    expected_capture_manifest_sha256: &"0".repeat(64),
                    attestor_binary_sha256: &"d".repeat(64),
                },
            )
            .unwrap_err()
            .contains("capture manifest SHA-256 mismatch")
        );

        let sealed = seal(&source).unwrap();
        let sealed_json = serde_json::to_vec_pretty(&sealed).unwrap();
        let sealed_sha256 = sha256_bytes(&sealed_json);
        let capture_sha256 = sha256_bytes(&capture_manifest);
        assert!(
            attest_legacy_v2_artifact(
                &sealed,
                &sealed_json,
                &sealed_sha256,
                experiment,
                0,
                PostHocAttestationInputV2 {
                    capture_manifest_bytes: &capture_manifest,
                    expected_capture_manifest_sha256: &capture_sha256,
                    attestor_binary_sha256: &"d".repeat(64),
                },
            )
            .unwrap_err()
            .contains("reserved v2 identity field")
        );
    }

    #[test]
    fn rejects_invalid_accounting_and_row_replay_before_sealing() {
        let mut bad_counts = source();
        bad_counts.metadata.input_positions = 2;
        assert!(seal(&bad_counts).unwrap_err().contains("accounts for"));

        let mut bad_row = source();
        bad_row.rows[0].uci_prefix.pop();
        assert!(seal(&bad_row).unwrap_err().contains("prefix moves"));

        let mut duplicate = source();
        duplicate.rows.push(duplicate.rows[0].clone());
        duplicate.metadata.input_positions = 2;
        assert!(
            seal(&duplicate)
                .unwrap_err()
                .contains("duplicate source row")
        );
    }

    #[test]
    fn capture_manifest_is_strict_and_bound_to_the_experiment() {
        let experiment = experiment();
        let bytes = capture_manifest(&experiment);
        let manifest: CalibrationCaptureManifestV1 = serde_json::from_slice(&bytes).unwrap();
        validate_capture_manifest(&manifest, &experiment).unwrap();

        let mut wrong_binary = manifest.clone();
        wrong_binary.stockfish_binary_sha256 = "e".repeat(64);
        assert!(
            validate_capture_manifest(&wrong_binary, &experiment)
                .unwrap_err()
                .contains("reference binary")
        );

        let mut wrong_order = manifest.clone();
        wrong_order.corpus_sha256.insert(0, "f".repeat(64));
        assert!(
            validate_capture_manifest(&wrong_order, &experiment)
                .unwrap_err()
                .contains("ordered corpus identities")
        );

        let mut missing_command = manifest;
        missing_command.exact_commands.clear();
        assert!(
            validate_capture_manifest(&missing_command, &experiment)
                .unwrap_err()
                .contains("exactly 1 nonempty shard commands")
        );

        let mut unknown_field: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        unknown_field
            .as_object_mut()
            .unwrap()
            .insert("unknown".to_string(), serde_json::json!(true));
        assert!(serde_json::from_value::<CalibrationCaptureManifestV1>(unknown_field).is_err());
    }

    #[test]
    fn validates_fixed_provenance_fields() {
        let valid = PostHocAttestationV2 {
            post_hoc_attested: true,
            method: POST_HOC_ATTESTATION_METHOD_V1.to_string(),
            source_artifact_sha256: "a".repeat(64),
            source_format_version: ANALYSIS_FORMAT_V2,
            attestor_binary_sha256: "b".repeat(64),
            capture_manifest_sha256: "c".repeat(64),
        };
        validate_post_hoc_attestation(&valid).unwrap();

        let mut invalid = valid.clone();
        invalid.post_hoc_attested = false;
        assert!(validate_post_hoc_attestation(&invalid).is_err());
        invalid = valid.clone();
        invalid.method = "unknown".to_string();
        assert!(validate_post_hoc_attestation(&invalid).is_err());
        invalid = valid;
        invalid.capture_manifest_sha256 = "A".repeat(64);
        assert!(validate_post_hoc_attestation(&invalid).is_err());
    }
}
