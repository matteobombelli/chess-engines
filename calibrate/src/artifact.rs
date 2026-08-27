use std::collections::{HashMap, HashSet};

use chess_core::Board;

use crate::attestation::validate_post_hoc_attestation;
use crate::identity::{aggregate_corpus_sha256, analysis_config_sha256};
use crate::{
    ANALYSIS_FORMAT_V1, ANALYSIS_FORMAT_V2, ANALYSIS_TARGET_V2, AnalysisArtifact, AnalysisBotV2,
    AnalysisExperimentV2, AnalysisRow, PLAYER_SHARD_SCHEMA_V1,
};

/// Validate one artifact independently before publication or merging.
pub fn validate_analysis_artifact(artifact: &AnalysisArtifact) -> Result<(), String> {
    let metadata = &artifact.metadata;
    if !matches!(
        metadata.format_version,
        ANALYSIS_FORMAT_V1 | ANALYSIS_FORMAT_V2
    ) {
        return Err(format!(
            "unsupported analysis format version {}",
            metadata.format_version
        ));
    }
    if metadata.target.trim().is_empty()
        || metadata.bot.trim().is_empty()
        || metadata.reference_engine.trim().is_empty()
        || metadata.reference_nodes_per_search == 0
    {
        return Err("analysis metadata has an empty identity or node budget".to_string());
    }

    if metadata.format_version == ANALYSIS_FORMAT_V1 {
        if metadata.experiment.is_some()
            || metadata.shard_index.is_some()
            || metadata.attestation.is_some()
        {
            return Err("legacy format-v1 artifact contains format-v2 identity fields".to_string());
        }
        return Ok(());
    }

    let experiment = metadata
        .experiment
        .as_ref()
        .ok_or("format-v2 artifact is missing its experiment identity")?;
    validate_experiment(experiment)?;
    if metadata.target != ANALYSIS_TARGET_V2 {
        return Err("format-v2 artifact has the wrong calibration target".to_string());
    }
    let expected_bot = expected_bot_name(&experiment.bot);
    if metadata.bot != expected_bot {
        return Err(format!(
            "format-v2 bot label {:?} disagrees with experiment identity {:?}",
            metadata.bot, expected_bot
        ));
    }
    let shard_index = metadata
        .shard_index
        .ok_or("format-v2 artifact is missing shard_index")?;
    if shard_index >= experiment.sampling.shard_count {
        return Err(format!(
            "shard index {shard_index} is outside shard count {}",
            experiment.sampling.shard_count
        ));
    }
    if metadata.reference_engine != experiment.reference.engine_name
        || metadata.reference_nodes_per_search != experiment.reference.nodes_per_search
    {
        return Err("legacy reference metadata disagrees with v2 experiment identity".to_string());
    }
    if let Some(attestation) = &metadata.attestation {
        validate_post_hoc_attestation(attestation)?;
    }

    if metadata.input_positions == 0
        || metadata.unique_games == 0
        || metadata.unique_games > metadata.input_positions
        || metadata.analyzed_unique_games > metadata.unique_games
    {
        return Err("format-v2 artifact has impossible game/position counts".to_string());
    }
    let accounted = artifact
        .rows
        .len()
        .checked_add(artifact.skipped_uninformative)
        .and_then(|count| count.checked_add(artifact.skipped_player_cap))
        .ok_or("analysis position counts overflow")?;
    if accounted != metadata.input_positions {
        return Err(format!(
            "format-v2 artifact accounts for {accounted} positions, metadata says {}",
            metadata.input_positions
        ));
    }
    let analyzed_games = artifact
        .rows
        .iter()
        .map(|row| row.game_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    if metadata.analyzed_unique_games != analyzed_games {
        return Err(format!(
            "format-v2 analyzed_unique_games is {}, rows contain {analyzed_games}",
            metadata.analyzed_unique_games
        ));
    }

    let mut row_ids = HashSet::new();
    for (row_index, row) in artifact.rows.iter().enumerate() {
        validate_v2_row(row, experiment, shard_index)
            .map_err(|error| format!("row {row_index}: {error}"))?;
        if !row_ids.insert((row.game_id.as_str(), row.ply)) {
            return Err(format!(
                "duplicate row identity in shard {shard_index}: game {:?} ply {}",
                row.game_id, row.ply
            ));
        }
    }
    Ok(())
}

/// Validate a complete report input. V2 requires every declared shard exactly
/// once; v1 remains readable but can never be combined with v2.
pub fn validate_analysis_artifacts(artifacts: &[AnalysisArtifact]) -> Result<(), String> {
    let first = artifacts
        .first()
        .ok_or("at least one analysis artifact is required")?;
    for (artifact_index, artifact) in artifacts.iter().enumerate() {
        validate_analysis_artifact(artifact)
            .map_err(|error| format!("analysis artifact {artifact_index}: {error}"))?;
        if artifact.metadata.format_version != first.metadata.format_version {
            return Err("analysis artifacts mix format v1 and v2".to_string());
        }
        if artifact.metadata.target != first.metadata.target
            || artifact.metadata.bot != first.metadata.bot
            || artifact.metadata.reference_engine != first.metadata.reference_engine
            || artifact.metadata.reference_nodes_per_search
                != first.metadata.reference_nodes_per_search
        {
            return Err(
                "analysis artifacts use different bot, target, or reference settings".to_string(),
            );
        }
    }

    let mut row_ids = HashSet::new();
    for (artifact_index, artifact) in artifacts.iter().enumerate() {
        for (row_index, row) in artifact.rows.iter().enumerate() {
            if !row_ids.insert((row.game_id.as_str(), row.ply)) {
                return Err(format!(
                    "duplicate/overlapping row identity at artifact {artifact_index} row {row_index}: game {:?} ply {}",
                    row.game_id, row.ply,
                ));
            }
        }
    }

    if first.metadata.format_version == ANALYSIS_FORMAT_V1 {
        return Ok(());
    }
    let experiment = first
        .metadata
        .experiment
        .as_ref()
        .expect("individual v2 validation requires experiment");
    let shard_count = experiment.sampling.shard_count;
    if u64::try_from(artifacts.len()).ok() != Some(shard_count) {
        return Err(format!(
            "format-v2 report requires all {shard_count} shards exactly once; got {}",
            artifacts.len()
        ));
    }

    let mut shard_indices = HashSet::new();
    let mut player_shards = HashMap::<String, u64>::new();
    let attested = first.metadata.attestation.is_some();
    let mut source_artifact_hashes = HashSet::new();
    for artifact in artifacts {
        if artifact.metadata.experiment.as_ref() != Some(experiment) {
            return Err(
                "format-v2 analysis artifacts have different experiment configs".to_string(),
            );
        }
        let shard_index = artifact
            .metadata
            .shard_index
            .expect("individual v2 validation requires shard index");
        if !shard_indices.insert(shard_index) {
            return Err(format!("duplicate format-v2 shard index {shard_index}"));
        }
        if artifact.metadata.attestation.is_some() != attested {
            return Err("format-v2 report mixes native and post-hoc-attested shards".to_string());
        }
        if let (Some(first_attestation), Some(attestation)) = (
            first.metadata.attestation.as_ref(),
            artifact.metadata.attestation.as_ref(),
        ) {
            if attestation.method != first_attestation.method
                || attestation.source_format_version != first_attestation.source_format_version
                || attestation.attestor_binary_sha256 != first_attestation.attestor_binary_sha256
                || attestation.capture_manifest_sha256 != first_attestation.capture_manifest_sha256
            {
                return Err(
                    "post-hoc-attested shards use different attestation evidence or tools"
                        .to_string(),
                );
            }
            if !source_artifact_hashes.insert(attestation.source_artifact_sha256.as_str()) {
                return Err("post-hoc-attested shards repeat a source artifact hash".to_string());
            }
        }
        for row in &artifact.rows {
            let player = row.actor_username.to_ascii_lowercase();
            match player_shards.insert(player.clone(), shard_index) {
                Some(previous) if previous != shard_index => {
                    return Err(format!(
                        "player {player:?} overlaps shards {previous} and {shard_index}"
                    ));
                }
                _ => {}
            }
        }
    }
    if (0..shard_count).any(|index| !shard_indices.contains(&index)) {
        return Err("format-v2 report shard indexes are incomplete".to_string());
    }
    Ok(())
}

fn validate_experiment(experiment: &AnalysisExperimentV2) -> Result<(), String> {
    if experiment.corpus_sha256.is_empty()
        || experiment
            .corpus_sha256
            .iter()
            .chain(experiment.exclude_corpus_sha256.iter())
            .any(|digest| !valid_sha256(digest))
        || !valid_sha256(&experiment.corpus_digest_sha256)
        || !valid_sha256(&experiment.analysis_config_sha256)
        || !valid_sha256(&experiment.calibration_binary_sha256)
    {
        return Err("format-v2 experiment contains an invalid content hash".to_string());
    }
    let corpus_digest =
        aggregate_corpus_sha256(&experiment.corpus_sha256, &experiment.exclude_corpus_sha256)?;
    if experiment.corpus_digest_sha256 != corpus_digest {
        return Err("format-v2 corpus digest does not match its ordered file hashes".to_string());
    }
    let config_digest = analysis_config_sha256(experiment)?;
    if experiment.analysis_config_sha256 != config_digest {
        return Err("format-v2 config digest does not match its effective settings".to_string());
    }
    let sampling = &experiment.sampling;
    if sampling.positions_per_side == 0
        || sampling.positions_per_player == 0
        || sampling.analyzed_positions_per_player == 0
        || sampling.max_positions == Some(0)
        || sampling.minimum_rating > sampling.maximum_rating
        || sampling.minimum_ply == 0
        || sampling.minimum_ply > sampling.maximum_ply
        || sampling.shard_count == 0
        || sampling.player_shard_schema != PLAYER_SHARD_SCHEMA_V1
        || sampling.minimum_best_expected_score_ppm >= sampling.maximum_best_expected_score_ppm
        || sampling.maximum_best_expected_score_ppm > 1_000_000
    {
        return Err("format-v2 experiment contains an invalid sampling config".to_string());
    }
    match &experiment.bot {
        AnalysisBotV2::Random { .. } => {}
        AnalysisBotV2::MinimaxFixed {
            depth,
            baseline_move_digest,
        } if *depth > 0 && *baseline_move_digest != 0 => {}
        AnalysisBotV2::MinimaxTimed {
            move_time_ms,
            maximum_depth,
        } if *move_time_ms > 0 && *maximum_depth > 0 => {}
        AnalysisBotV2::AlphaMini {
            model_sha256,
            manifest_sha256,
            simulations,
            move_time_ms,
            batch_size,
            cpuct_ppm,
            fpu_reduction_ppm,
            root_dirichlet_alpha_ppm,
            root_noise_fraction_ppm,
            evaluator,
            ..
        } if valid_sha256(model_sha256)
            && valid_sha256(manifest_sha256)
            && *simulations > 0
            && *move_time_ms > 0
            && *batch_size > 0
            && *cpuct_ppm > 0
            && *fpu_reduction_ppm <= 1_000_000
            && root_dirichlet_alpha_ppm.is_none()
            && *root_noise_fraction_ppm == 0
            && evaluator == "onnxruntime-cpu-v1" => {}
        AnalysisBotV2::MiniGpt {
            model_sha256,
            manifest_sha256,
            context,
            evaluator,
            ..
        } if valid_sha256(model_sha256)
            && valid_sha256(manifest_sha256)
            && *context >= 2
            && evaluator == "onnxruntime-cpu-v1" => {}
        _ => return Err("format-v2 experiment contains an invalid bot config".to_string()),
    }
    let reference = &experiment.reference;
    if reference.engine_name.trim().is_empty()
        || !valid_sha256(&reference.binary_sha256)
        || reference.nodes_per_search == 0
        || reference.hash_mb == 0
        || reference.threads == 0
        || !reference.show_wdl
    {
        return Err("format-v2 experiment contains an invalid reference config".to_string());
    }
    Ok(())
}

fn expected_bot_name(bot: &AnalysisBotV2) -> String {
    match bot {
        AnalysisBotV2::Random { seed } => format!("Random (seed {seed})"),
        AnalysisBotV2::MinimaxFixed { depth, .. } => format!("Minimax (depth {depth})"),
        AnalysisBotV2::MinimaxTimed {
            move_time_ms,
            maximum_depth,
        } => format!("Minimax ({move_time_ms} ms/move, depth ceiling {maximum_depth})"),
        AnalysisBotV2::AlphaMini {
            model_sha256,
            move_time_ms,
            simulations,
            batch_size,
            ..
        } => format!(
            "AlphaMiniV1[{}] ({move_time_ms} ms/move, {simulations} simulation cap, batch {batch_size})",
            &model_sha256[..12]
        ),
        AnalysisBotV2::MiniGpt {
            model_sha256,
            context,
            temperature_ppm,
            ..
        } => minigpt_bot_name(&model_sha256[..12], *context, *temperature_ppm),
    }
}

/// The temperature is rendered from its exact millionths so the recorded name
/// can be rebuilt from the experiment identity alone.
pub fn minigpt_bot_name(model_identity: &str, context: usize, temperature_ppm: u32) -> String {
    format!(
        "MiniGptV1[{model_identity}] (temperature {:.6}, context {context})",
        f64::from(temperature_ppm) / 1_000_000.0
    )
}

fn validate_v2_row(
    row: &AnalysisRow,
    experiment: &AnalysisExperimentV2,
    shard_index: u64,
) -> Result<(), String> {
    if row.game_id.trim().is_empty() || row.actor_username.trim().is_empty() {
        return Err("format-v2 row has an empty game or player identity".to_string());
    }
    let sampling = &experiment.sampling;
    if stable_player_hash(&row.actor_username) % sampling.shard_count != shard_index {
        return Err(format!(
            "format-v2 row {:?} player {:?} is assigned to a different shard",
            row.game_id, row.actor_username
        ));
    }
    if !(sampling.minimum_rating..=sampling.maximum_rating).contains(&row.actor_rating)
        || !(sampling.minimum_ply..=sampling.maximum_ply).contains(&row.ply)
    {
        return Err(format!(
            "format-v2 row {:?} rating/ply is outside the sampling config",
            row.game_id
        ));
    }
    if row.ply == 0 || row.uci_prefix.len() != usize::from(row.ply - 1) {
        return Err(format!(
            "format-v2 row {:?} ply {} has {} prefix moves",
            row.game_id,
            row.ply,
            row.uci_prefix.len()
        ));
    }
    let board = Board::import_uci(&row.uci_prefix).map_err(|error| {
        format!(
            "format-v2 row {:?} ply {} has an illegal prefix: {error}",
            row.game_id, row.ply
        )
    })?;
    let replayed_fen = board.to_fen();
    if replayed_fen != row.fen {
        return Err(format!(
            "format-v2 row {:?} ply {} prefix/FEN mismatch: replayed {replayed_fen}, stored {}",
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
                "format-v2 row {:?} ply {} has illegal {label} move {mv:?}: {error}",
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
            "format-v2 row {:?} has a non-finite or out-of-range score",
            row.game_id
        ));
    }
    let minimum_best = f64::from(sampling.minimum_best_expected_score_ppm) / 1_000_000.0;
    let maximum_best = f64::from(sampling.maximum_best_expected_score_ppm) / 1_000_000.0;
    if !(minimum_best..=maximum_best).contains(&row.best_expected_score) {
        return Err(format!(
            "format-v2 row {:?} best score is outside the informative-position bounds",
            row.game_id
        ));
    }
    let human_loss = (row.best_expected_score - row.human_expected_score).max(0.0);
    let bot_loss = (row.best_expected_score - row.bot_expected_score).max(0.0);
    if (row.human_loss - human_loss).abs() > 1e-12 || (row.bot_loss - bot_loss).abs() > 1e-12 {
        return Err(format!(
            "format-v2 row {:?} has loss values inconsistent with its scores",
            row.game_id
        ));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Stable case-insensitive player partition used by both analysis and seal
/// validation. Keeping it in the library prevents generator/validator drift.
pub fn stable_player_hash(username: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in username.bytes() {
        hash ^= u64::from(byte.to_ascii_lowercase());
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnalysisMetadata, AnalysisReferenceV2, AnalysisSamplingV2, PLAYER_SHARD_SCHEMA_V1,
    };

    fn row(game: &str, player: &str) -> AnalysisRow {
        let prefix = vec!["e2e4".to_string(), "e7e5".to_string()];
        let board = Board::import_uci(&prefix).unwrap();
        AnalysisRow {
            game_id: game.to_string(),
            actor_username: player.to_string(),
            actor_rating: 1_500,
            ply: 3,
            uci_prefix: prefix,
            fen: board.to_fen(),
            human_move: "g1f3".to_string(),
            bot_move: "f1c4".to_string(),
            reference_move: "d2d4".to_string(),
            best_expected_score: 0.6,
            human_expected_score: 0.5,
            bot_expected_score: 0.4,
            human_loss: 0.1,
            bot_loss: 0.2,
        }
    }

    fn experiment(shard_count: u64) -> AnalysisExperimentV2 {
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
                shard_count,
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

    fn player_for_shard(shard_index: u64, shard_count: u64, ordinal: usize) -> String {
        (0_u64..)
            .map(|suffix| format!("player-{suffix}"))
            .filter(|name| stable_player_hash(name) % shard_count == shard_index)
            .nth(ordinal)
            .unwrap()
    }

    fn artifact(shard_count: u64, shard_index: u64, row: AnalysisRow) -> AnalysisArtifact {
        AnalysisArtifact {
            metadata: AnalysisMetadata {
                format_version: ANALYSIS_FORMAT_V2,
                target: ANALYSIS_TARGET_V2.to_string(),
                bot: "Minimax (depth 3)".to_string(),
                reference_engine: "Stockfish 17.1".to_string(),
                reference_nodes_per_search: 100_000,
                input_positions: 1,
                unique_games: 1,
                analyzed_unique_games: 1,
                experiment: Some(experiment(shard_count)),
                shard_index: Some(shard_index),
                attestation: None,
            },
            skipped_uninformative: 0,
            skipped_player_cap: 0,
            rows: vec![row],
        }
    }

    #[test]
    fn validates_replayed_v2_row_and_all_legal_alternative_moves() {
        validate_analysis_artifact(&artifact(1, 0, row("game-1", "alice"))).unwrap();

        for field in ["prefix", "fen", "human", "bot", "reference"] {
            let mut invalid = artifact(1, 0, row("game-1", "alice"));
            match field {
                "prefix" => {
                    invalid.rows[0].uci_prefix.pop();
                }
                "fen" => {
                    invalid.rows[0].fen =
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string();
                }
                "human" => invalid.rows[0].human_move = "e2e5".to_string(),
                "bot" => invalid.rows[0].bot_move = "e2e5".to_string(),
                "reference" => invalid.rows[0].reference_move = "e2e5".to_string(),
                _ => unreachable!(),
            }
            assert!(validate_analysis_artifact(&invalid).is_err(), "{field}");
        }
    }

    #[test]
    fn merge_requires_exact_config_and_complete_nonoverlapping_shards() {
        let shard_zero = player_for_shard(0, 2, 0);
        let shard_zero_two = player_for_shard(0, 2, 1);
        let shard_one = player_for_shard(1, 2, 0);
        let first = artifact(2, 0, row("game-1", &shard_zero));
        let second = artifact(2, 1, row("game-2", &shard_one));
        validate_analysis_artifacts(&[first.clone(), second.clone()]).unwrap();

        let mut duplicate_index = second.clone();
        duplicate_index.metadata.shard_index = Some(0);
        duplicate_index.rows[0].actor_username = shard_zero_two;
        assert!(validate_analysis_artifacts(&[first.clone(), duplicate_index]).is_err());
        assert!(validate_analysis_artifacts(std::slice::from_ref(&first)).is_err());

        let mut config_drift = second.clone();
        config_drift
            .metadata
            .experiment
            .as_mut()
            .unwrap()
            .sampling
            .sample_seed += 1;
        let experiment = config_drift.metadata.experiment.as_mut().unwrap();
        experiment.analysis_config_sha256 = analysis_config_sha256(experiment).unwrap();
        assert!(validate_analysis_artifacts(&[first.clone(), config_drift]).is_err());

        let mut duplicate_row = second.clone();
        duplicate_row.rows[0].game_id = first.rows[0].game_id.clone();
        duplicate_row.rows[0].ply = first.rows[0].ply;
        assert!(validate_analysis_artifacts(&[first.clone(), duplicate_row]).is_err());

        let mut overlapping_player = second;
        overlapping_player.rows[0].actor_username = first.rows[0].actor_username.to_uppercase();
        assert!(validate_analysis_artifacts(&[first, overlapping_player]).is_err());
    }

    #[test]
    fn rejects_digest_shard_accounting_and_score_tampering() {
        let player = player_for_shard(0, 1, 0);
        let baseline = artifact(1, 0, row("game-1", &player));

        let mut bad = baseline.clone();
        bad.metadata
            .experiment
            .as_mut()
            .unwrap()
            .corpus_digest_sha256 = "0".repeat(64);
        assert!(validate_analysis_artifact(&bad).is_err());

        let mut bad = baseline.clone();
        bad.metadata
            .experiment
            .as_mut()
            .unwrap()
            .analysis_config_sha256 = "A".repeat(64);
        assert!(validate_analysis_artifact(&bad).is_err());

        let mut bad = baseline.clone();
        bad.metadata.input_positions += 1;
        assert!(validate_analysis_artifact(&bad).is_err());

        let mut bad = baseline.clone();
        bad.rows[0].human_loss += 0.01;
        assert!(validate_analysis_artifact(&bad).is_err());

        let mut bad = baseline.clone();
        bad.rows[0].bot_expected_score = f64::NAN;
        assert!(validate_analysis_artifact(&bad).is_err());
    }

    #[test]
    fn accepts_empty_prefix_only_at_first_ply_and_enforces_player_partition() {
        let player = player_for_shard(0, 1, 0);
        let mut first_ply = row("game-1", &player);
        first_ply.ply = 1;
        first_ply.uci_prefix.clear();
        first_ply.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string();
        first_ply.human_move = "e2e4".to_string();
        first_ply.bot_move = "d2d4".to_string();
        first_ply.reference_move = "g1f3".to_string();
        validate_analysis_artifact(&artifact(1, 0, first_ply.clone())).unwrap();

        first_ply.ply = 2;
        assert!(validate_analysis_artifact(&artifact(1, 0, first_ply)).is_err());

        let wrong_player = player_for_shard(1, 2, 0);
        assert!(validate_analysis_artifact(&artifact(2, 0, row("game-2", &wrong_player))).is_err());
    }

    #[test]
    fn legacy_v1_is_supported_but_never_mixed_with_v2() {
        let v2 = artifact(1, 0, row("game-2", "bob"));
        let mut v1 = artifact(1, 0, row("game-1", "alice"));
        v1.metadata.format_version = ANALYSIS_FORMAT_V1;
        v1.metadata.experiment = None;
        v1.metadata.shard_index = None;
        v1.rows[0].uci_prefix.clear();
        validate_analysis_artifacts(std::slice::from_ref(&v1)).unwrap();
        assert!(validate_analysis_artifacts(&[v1, v2]).is_err());
    }
}
