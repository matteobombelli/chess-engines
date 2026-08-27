//! Stable content and configuration identities for calibration artifacts.
//!
//! These hashes deliberately do not use JSON serialization. JSON map ordering,
//! integer formatting, and future serde changes must not silently change an
//! experiment identity. Every value below has an explicit, versioned binary
//! framing instead.
//!
//! Framing begins with a NUL-terminated ASCII domain. Each named field is a
//! two-byte big-endian name length followed by the name. Integers are fixed
//! width and big-endian; strings/variants add an eight-byte byte length;
//! options add a one-byte presence tag; SHA-256 values are raw 32-byte values.
//! Digest lists add an eight-byte count followed by raw digests in order.

use std::path::{Path, PathBuf};

use artifact_io::sha256_file;
use sha2::{Digest, Sha256};

use crate::{AnalysisBotV2, AnalysisExperimentV2, AnalysisReferenceV2, AnalysisSamplingV2};

const CORPUS_DOMAIN_V2: &[u8] = b"alphamini.calibration.corpus-identity.v2\0";
const CONFIG_DOMAIN_V2: &[u8] = b"alphamini.calibration.analysis-config.v2\0";

/// Return the lowercase SHA-256 digest of the exact bytes at `path`.
pub fn sha256_file_hex(path: &Path) -> Result<String, String> {
    sha256_file(path).map_err(|error| format!("failed to hash {}: {error}", path.display()))
}

/// Hash each path independently, preserving caller-supplied order.
pub fn sha256_paths(paths: &[PathBuf]) -> Result<Vec<String>, String> {
    paths.iter().map(|path| sha256_file_hex(path)).collect()
}

/// Bind the ordered primary and exclusion corpus file hashes into one digest.
///
/// Individual file hashes are decoded to their raw 32-byte form before they
/// are framed. The aggregate therefore depends on file content, file order,
/// and whether a file was primary or exclusion, but not on hex case or paths.
pub fn aggregate_corpus_sha256(
    primary_file_sha256: &[String],
    exclusion_file_sha256: &[String],
) -> Result<String, String> {
    let mut framing = CanonicalHash::new(CORPUS_DOMAIN_V2);
    framing.digest_list(b"primary", primary_file_sha256)?;
    framing.digest_list(b"exclusion", exclusion_file_sha256)?;
    Ok(framing.finish())
}

/// Return the canonical identity of the effective non-corpus v2 settings.
///
/// This deliberately excludes `shard_index`, the ordered per-file corpus
/// hashes, `corpus_digest_sha256`, and `analysis_config_sha256` itself. Corpus
/// identity is independently bound by [`aggregate_corpus_sha256`]. It includes
/// shard count/schema, sampling, engine, reference-engine, model, manifest,
/// and linked calibration-binary identities.
pub fn analysis_config_sha256(experiment: &AnalysisExperimentV2) -> Result<String, String> {
    let mut framing = CanonicalHash::new(CONFIG_DOMAIN_V2);

    // Exhaustive destructuring makes a newly added experiment field a compile
    // error here instead of silently omitting it from provenance. The four
    // explicitly ignored identities are independently bound or self-referential.
    let AnalysisExperimentV2 {
        corpus_sha256: _,
        exclude_corpus_sha256: _,
        corpus_digest_sha256: _,
        analysis_config_sha256: _,
        sampling,
        bot,
        reference,
        calibration_binary_sha256,
    } = experiment;
    let AnalysisSamplingV2 {
        positions_per_side,
        positions_per_player,
        analyzed_positions_per_player,
        max_positions,
        minimum_rating,
        maximum_rating,
        minimum_ply,
        maximum_ply,
        sample_seed,
        shard_count,
        player_shard_schema,
        minimum_best_expected_score_ppm,
        maximum_best_expected_score_ppm,
    } = sampling;

    framing.usize(b"positions_per_side", *positions_per_side)?;
    framing.usize(b"positions_per_player", *positions_per_player)?;
    framing.usize(
        b"analyzed_positions_per_player",
        *analyzed_positions_per_player,
    )?;
    framing.optional_usize(b"max_positions", *max_positions)?;
    framing.u16(b"minimum_rating", *minimum_rating);
    framing.u16(b"maximum_rating", *maximum_rating);
    framing.u16(b"minimum_ply", *minimum_ply);
    framing.u16(b"maximum_ply", *maximum_ply);
    framing.u64(b"sample_seed", *sample_seed);
    framing.u64(b"shard_count", *shard_count);
    framing.string(b"player_shard_schema", player_shard_schema)?;
    framing.u32(
        b"minimum_best_expected_score_ppm",
        *minimum_best_expected_score_ppm,
    );
    framing.u32(
        b"maximum_best_expected_score_ppm",
        *maximum_best_expected_score_ppm,
    );

    match bot {
        AnalysisBotV2::Random { seed } => {
            framing.variant(b"bot", b"random")?;
            framing.u64(b"bot.seed", *seed);
        }
        AnalysisBotV2::MinimaxFixed {
            depth,
            baseline_move_digest,
        } => {
            framing.variant(b"bot", b"minimax_fixed")?;
            framing.u8(b"bot.depth", *depth);
            framing.u64(b"bot.baseline_move_digest", *baseline_move_digest);
        }
        AnalysisBotV2::MinimaxTimed {
            move_time_ms,
            maximum_depth,
        } => {
            framing.variant(b"bot", b"minimax_timed")?;
            framing.u64(b"bot.move_time_ms", *move_time_ms);
            framing.u8(b"bot.maximum_depth", *maximum_depth);
        }
        AnalysisBotV2::AlphaMini {
            model_sha256,
            manifest_sha256,
            simulations,
            move_time_ms,
            batch_size,
            seed,
            cpuct_ppm,
            fpu_reduction_ppm,
            root_dirichlet_alpha_ppm,
            root_noise_fraction_ppm,
            evaluator,
        } => {
            framing.variant(b"bot", b"alphamini")?;
            framing.digest(b"bot.model_sha256", model_sha256)?;
            framing.digest(b"bot.manifest_sha256", manifest_sha256)?;
            framing.u32(b"bot.simulations", *simulations);
            framing.u64(b"bot.move_time_ms", *move_time_ms);
            framing.usize(b"bot.batch_size", *batch_size)?;
            framing.u64(b"bot.seed", *seed);
            framing.u32(b"bot.cpuct_ppm", *cpuct_ppm);
            framing.u32(b"bot.fpu_reduction_ppm", *fpu_reduction_ppm);
            match root_dirichlet_alpha_ppm {
                None => framing.optional_u32(b"bot.root_dirichlet_alpha_ppm", None),
                Some(value) => framing.optional_u32(b"bot.root_dirichlet_alpha_ppm", Some(*value)),
            }
            framing.u32(b"bot.root_noise_fraction_ppm", *root_noise_fraction_ppm);
            framing.string(b"bot.evaluator", evaluator)?;
        }
        AnalysisBotV2::MiniGpt {
            model_sha256,
            manifest_sha256,
            context,
            temperature_ppm,
            seed,
            evaluator,
        } => {
            framing.variant(b"bot", b"mini_gpt")?;
            framing.digest(b"bot.model_sha256", model_sha256)?;
            framing.digest(b"bot.manifest_sha256", manifest_sha256)?;
            framing.usize(b"bot.context", *context)?;
            framing.u32(b"bot.temperature_ppm", *temperature_ppm);
            framing.u64(b"bot.seed", *seed);
            framing.string(b"bot.evaluator", evaluator)?;
        }
    }

    let AnalysisReferenceV2 {
        engine_name,
        binary_sha256,
        nodes_per_search,
        hash_mb,
        threads,
        show_wdl,
    } = reference;
    framing.string(b"reference.engine_name", engine_name)?;
    framing.digest(b"reference.binary_sha256", binary_sha256)?;
    framing.u64(b"reference.nodes_per_search", *nodes_per_search);
    framing.u32(b"reference.hash_mb", *hash_mb);
    framing.u16(b"reference.threads", *threads);
    framing.boolean(b"reference.show_wdl", *show_wdl);
    framing.digest(b"calibration_binary_sha256", calibration_binary_sha256)?;

    Ok(framing.finish())
}

struct CanonicalHash(Sha256);

impl CanonicalHash {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        Self(hasher)
    }

    fn field(&mut self, name: &[u8]) -> Result<(), String> {
        let length = u16::try_from(name.len())
            .map_err(|_| "canonical hash field name is too long".to_string())?;
        self.0.update(length.to_be_bytes());
        self.0.update(name);
        Ok(())
    }

    fn bytes(&mut self, name: &[u8], value: &[u8]) -> Result<(), String> {
        self.field(name)?;
        let length = u64::try_from(value.len()).map_err(|_| {
            format!(
                "canonical field {} is too long",
                String::from_utf8_lossy(name)
            )
        })?;
        self.0.update(length.to_be_bytes());
        self.0.update(value);
        Ok(())
    }

    fn string(&mut self, name: &[u8], value: &str) -> Result<(), String> {
        self.bytes(name, value.as_bytes())
    }

    fn variant(&mut self, name: &[u8], value: &[u8]) -> Result<(), String> {
        self.bytes(name, value)
    }

    fn digest(&mut self, name: &[u8], value: &str) -> Result<(), String> {
        self.field(name)?;
        self.0.update(decode_sha256(value)?);
        Ok(())
    }

    fn digest_list(&mut self, name: &[u8], values: &[String]) -> Result<(), String> {
        self.field(name)?;
        let count = u64::try_from(values.len())
            .map_err(|_| "canonical digest list has too many entries".to_string())?;
        self.0.update(count.to_be_bytes());
        for value in values {
            self.0.update(decode_sha256(value)?);
        }
        Ok(())
    }

    fn u8(&mut self, name: &[u8], value: u8) {
        self.field(name).expect("static field name fits in u16");
        self.0.update([value]);
    }

    fn u16(&mut self, name: &[u8], value: u16) {
        self.field(name).expect("static field name fits in u16");
        self.0.update(value.to_be_bytes());
    }

    fn u32(&mut self, name: &[u8], value: u32) {
        self.field(name).expect("static field name fits in u16");
        self.0.update(value.to_be_bytes());
    }

    fn u64(&mut self, name: &[u8], value: u64) {
        self.field(name).expect("static field name fits in u16");
        self.0.update(value.to_be_bytes());
    }

    fn usize(&mut self, name: &[u8], value: usize) -> Result<(), String> {
        let value = u64::try_from(value).map_err(|_| {
            format!(
                "canonical field {} does not fit in u64",
                String::from_utf8_lossy(name)
            )
        })?;
        self.u64(name, value);
        Ok(())
    }

    fn optional_usize(&mut self, name: &[u8], value: Option<usize>) -> Result<(), String> {
        self.field(name)?;
        match value {
            None => self.0.update([0]),
            Some(value) => {
                self.0.update([1]);
                let value = u64::try_from(value).map_err(|_| {
                    format!(
                        "canonical field {} does not fit in u64",
                        String::from_utf8_lossy(name)
                    )
                })?;
                self.0.update(value.to_be_bytes());
            }
        }
        Ok(())
    }

    fn optional_u32(&mut self, name: &[u8], value: Option<u32>) {
        self.field(name).expect("static field name fits in u16");
        match value {
            None => self.0.update([0]),
            Some(value) => {
                self.0.update([1]);
                self.0.update(value.to_be_bytes());
            }
        }
    }

    fn boolean(&mut self, name: &[u8], value: bool) {
        self.u8(name, u8::from(value));
    }

    fn finish(self) -> String {
        let digest = self.0.finalize();
        hex_digest(&digest)
    }
}

fn decode_sha256(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 {
        return Err(format!(
            "expected a 64-character SHA-256 digest, got {} characters",
            value.len()
        ));
    }

    let bytes = value.as_bytes();
    let mut decoded = [0_u8; 32];
    for (index, output) in decoded.iter_mut().enumerate() {
        let high = hex_nibble(bytes[index * 2])
            .ok_or_else(|| format!("invalid SHA-256 hex at byte offset {}", index * 2))?;
        let low = hex_nibble(bytes[index * 2 + 1])
            .ok_or_else(|| format!("invalid SHA-256 hex at byte offset {}", index * 2 + 1))?;
        *output = (high << 4) | low;
    }
    Ok(decoded)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PLAYER_SHARD_SCHEMA_V1;

    fn fixture(bot: AnalysisBotV2) -> AnalysisExperimentV2 {
        AnalysisExperimentV2 {
            corpus_sha256: vec!["00".repeat(32), "12".repeat(32)],
            exclude_corpus_sha256: vec!["ff".repeat(32)],
            corpus_digest_sha256: "aa".repeat(32),
            analysis_config_sha256: "bb".repeat(32),
            sampling: AnalysisSamplingV2 {
                positions_per_side: 73,
                positions_per_player: 9,
                analyzed_positions_per_player: 3,
                max_positions: Some(12_345),
                minimum_rating: 1_600,
                maximum_rating: 2_000,
                minimum_ply: 9,
                maximum_ply: 80,
                sample_seed: 0x0123_4567_89ab_cdef,
                shard_count: 4,
                player_shard_schema: PLAYER_SHARD_SCHEMA_V1.to_string(),
                minimum_best_expected_score_ppm: 20_000,
                maximum_best_expected_score_ppm: 980_000,
            },
            bot,
            reference: AnalysisReferenceV2 {
                engine_name: "Stockfish 18 test".to_string(),
                binary_sha256: "34".repeat(32),
                nodes_per_search: 50_000,
                hash_mb: 128,
                threads: 1,
                show_wdl: true,
            },
            calibration_binary_sha256: "56".repeat(32),
        }
    }

    #[test]
    fn corpus_identity_has_a_frozen_vector() {
        let digest =
            aggregate_corpus_sha256(&["00".repeat(32), "12".repeat(32)], &["ff".repeat(32)])
                .unwrap();
        assert_eq!(
            digest,
            "d9a9164d0470edf4b6a71c8fe99afa31d83468db018d2f11bda4c8380f434d3a"
        );

        let reordered =
            aggregate_corpus_sha256(&["12".repeat(32), "00".repeat(32)], &["ff".repeat(32)])
                .unwrap();
        assert_ne!(digest, reordered);
        let role_changed =
            aggregate_corpus_sha256(&["00".repeat(32)], &["12".repeat(32), "ff".repeat(32)])
                .unwrap();
        assert_ne!(digest, role_changed);
    }

    #[test]
    fn file_hash_has_a_frozen_vector() {
        let path = std::env::temp_dir().join(format!(
            "alphamini-calibration-sha256-fixture-{}",
            std::process::id()
        ));
        std::fs::write(&path, b"abc").unwrap();
        let digest = sha256_file_hex(&path).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn analysis_config_has_frozen_vectors_for_each_bot_kind() {
        let cases = [
            (
                AnalysisBotV2::Random { seed: 99 },
                "bd97c7bf9156f2ebf8e9173e4ab2f2fd126cdbea4a988f61a3fc7658bb9f9a0f",
            ),
            (
                AnalysisBotV2::MinimaxFixed {
                    depth: 3,
                    baseline_move_digest: 0xfeed_beef_cafe_babe,
                },
                "fa41b70ee291095217e310e7be090c9c4e30a817028532a599d52d07950ea029",
            ),
            (
                AnalysisBotV2::MinimaxTimed {
                    move_time_ms: 250,
                    maximum_depth: 64,
                },
                "a7f84b7f30e9290859f67120bc10d88ea1224aaaf29c6a4b16811228ebcd4c6b",
            ),
            (
                AnalysisBotV2::AlphaMini {
                    model_sha256: "78".repeat(32),
                    manifest_sha256: "9a".repeat(32),
                    simulations: 128,
                    move_time_ms: 2_000,
                    batch_size: 16,
                    seed: 7,
                    cpuct_ppm: 1_500_000,
                    fpu_reduction_ppm: 250_000,
                    root_dirichlet_alpha_ppm: None,
                    root_noise_fraction_ppm: 0,
                    evaluator: "onnxruntime-cpu-v1".to_string(),
                },
                "f60c4f09ab2f26147d847b915650c0303df7e84bbaf8c5ebd8fb4d8742d14cfa",
            ),
            (
                AnalysisBotV2::MiniGpt {
                    model_sha256: "78".repeat(32),
                    manifest_sha256: "9a".repeat(32),
                    context: 256,
                    temperature_ppm: 500_000,
                    seed: 7,
                    evaluator: "onnxruntime-cpu-v1".to_string(),
                },
                "d83a326fb7348eb600df8d4b906328422ce7b848789177ecce0536af7c5e3a25",
            ),
        ];

        for (bot, expected) in cases {
            assert_eq!(analysis_config_sha256(&fixture(bot)).unwrap(), expected);
        }
    }

    #[test]
    fn digest_framing_is_hex_case_independent_and_rejects_invalid_values() {
        let lower = aggregate_corpus_sha256(&["ab".repeat(32)], &[]).unwrap();
        let upper = aggregate_corpus_sha256(&["AB".repeat(32)], &[]).unwrap();
        assert_eq!(lower, upper);

        let error = aggregate_corpus_sha256(&["not-a-digest".to_string()], &[]).unwrap_err();
        assert!(error.contains("64-character SHA-256"));
    }

    #[test]
    fn config_identity_changes_when_an_effective_setting_changes() {
        let baseline = fixture(AnalysisBotV2::Random { seed: 99 });
        let baseline_digest = analysis_config_sha256(&baseline).unwrap();

        let mut changed = baseline.clone();
        changed.sampling.sample_seed += 1;
        assert_ne!(baseline_digest, analysis_config_sha256(&changed).unwrap());

        let mut changed = baseline.clone();
        changed.reference.hash_mb *= 2;
        assert_ne!(baseline_digest, analysis_config_sha256(&changed).unwrap());

        let mut identity_only_changes = baseline;
        identity_only_changes.corpus_sha256.reverse();
        identity_only_changes.exclude_corpus_sha256.clear();
        identity_only_changes.corpus_digest_sha256 = "cd".repeat(32);
        identity_only_changes.analysis_config_sha256 = "ef".repeat(32);
        assert_eq!(
            baseline_digest,
            analysis_config_sha256(&identity_only_changes).unwrap()
        );
    }
}
