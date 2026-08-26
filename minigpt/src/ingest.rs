//! Streaming ingest of compressed PGN dumps into token shards.
//!
//! One thread decodes zstd and splits the stream into games, applying the
//! tag-only filters; a pool of workers sanitizes, replays, and tokenizes the
//! survivors; the calling thread writes shards. Jobs are dispatched to workers
//! round-robin and collected in the same order, so shards hold games in the
//! order they appear in the dumps.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chess_core::movetext_moves;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::encoding::{BOS_TOKEN, PAD_TOKEN, TOKENIZER_VERSION, VOCAB_SIZE, encode_movetext};
use crate::manifest::{CountsV1, FiltersV1, SHARDS_MANIFEST_VERSION, ShardsManifestV1, SourceV1};
use crate::pgn::{PgnReader, RejectReason, SanitizeError, header_reject, sanitize_movetext};
use crate::shard::ShardWriter;

/// Games buffered per worker in each direction. The reader blocks once these
/// fill, which is what bounds memory over a 30 GB dump.
const QUEUE_CAPACITY: usize = 64;
const PROGRESS_INTERVAL: Duration = Duration::from_secs(10);
const SAN_ERROR_SAMPLES: usize = 20;
/// `ZSTD_WINDOWLOG_MAX_64`, the largest window a 64-bit decoder can accept.
const ZSTD_MAX_WINDOW_LOG: u32 = 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IngestOptions {
    pub min_elo: u32,
    pub min_plies: u32,
    pub max_plies: u32,
    pub token_target: u64,
    pub val_fraction_ppm: u32,
    pub shard_tokens: u64,
    pub workers: usize,
}

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid ingest options: {0}")]
    Options(String),
    #[error("ingest lost track of {0} games between reading and writing")]
    Accounting(i64),
}

/// Decide the train/validation split from the game id alone, so a game always
/// lands on the same side whatever subset of dumps a run covers.
///
/// The first eight bytes of `SHA-256(site)` are read big-endian; the game is
/// validation when that value modulo 1_000_000 is below `val_fraction_ppm`.
pub fn is_validation_game(site: &str, val_fraction_ppm: u32) -> bool {
    let digest = Sha256::digest(site.as_bytes());
    let value = u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 is 32 bytes"));
    value % 1_000_000 < u64::from(val_fraction_ppm)
}

pub fn run(
    dumps: &[PathBuf],
    output_dir: &Path,
    options: IngestOptions,
) -> Result<ShardsManifestV1, IngestError> {
    validate(dumps, options)?;
    std::fs::create_dir_all(output_dir).map_err(|source| IngestError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;
    let started = unix_seconds();
    let mut state = State {
        train: ShardWriter::new(output_dir, "shard", options.shard_tokens),
        val: ShardWriter::new(output_dir, "val", options.shard_tokens),
        counts: CountsV1::default(),
        san_error_samples: Vec::new(),
    };

    let mut sources = Vec::with_capacity(dumps.len());
    for dump in dumps {
        if state.total_tokens() >= options.token_target {
            break;
        }
        sources.push(ingest_dump(dump, &mut state, options)?);
    }

    let counts = state.counts;
    let train_shards = state.train.finish().map_err(|source| IngestError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;
    let val_shards = state.val.finish().map_err(|source| IngestError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let accounted = counts.games_accepted + counts.rejected.total();
    if accounted != counts.games_seen {
        return Err(IngestError::Accounting(
            counts.games_seen as i64 - accounted as i64,
        ));
    }

    Ok(ShardsManifestV1 {
        schema: SHARDS_MANIFEST_VERSION.to_string(),
        tokenizer: TOKENIZER_VERSION.to_string(),
        vocab_size: VOCAB_SIZE as u64,
        bos_token: BOS_TOKEN,
        pad_token: PAD_TOKEN,
        filters: FiltersV1 {
            min_elo: options.min_elo,
            min_plies: options.min_plies,
            max_plies: options.max_plies,
            token_target: options.token_target,
            val_fraction_ppm: options.val_fraction_ppm,
            shard_tokens: options.shard_tokens,
        },
        sources,
        counts,
        train_shards,
        val_shards,
        san_error_samples: state.san_error_samples,
        started_unix_seconds: started,
        completed_unix_seconds: unix_seconds(),
    })
}

fn validate(dumps: &[PathBuf], options: IngestOptions) -> Result<(), IngestError> {
    if dumps.is_empty() {
        return Err(IngestError::Options("no dumps given".to_string()));
    }
    if options.workers == 0 {
        return Err(IngestError::Options("workers must be positive".to_string()));
    }
    if options.shard_tokens == 0 || options.token_target == 0 {
        return Err(IngestError::Options(
            "shard-tokens and token-target must be positive".to_string(),
        ));
    }
    if options.min_plies > options.max_plies {
        return Err(IngestError::Options(
            "min-plies must not exceed max-plies".to_string(),
        ));
    }
    if options.val_fraction_ppm > 1_000_000 {
        return Err(IngestError::Options(
            "val-fraction must be in 0..=1".to_string(),
        ));
    }
    Ok(())
}

struct State {
    train: ShardWriter,
    val: ShardWriter,
    counts: CountsV1,
    san_error_samples: Vec<String>,
}

impl State {
    fn total_tokens(&self) -> u64 {
        self.train.total_tokens() + self.val.total_tokens()
    }
}

struct Job {
    site: String,
    movetext: String,
}

enum Encoded {
    Tokens { site: String, tokens: Vec<u16> },
    Rejected { site: String, reason: RejectReason },
}

/// Counters owned by the reader thread but read by the progress line.
#[derive(Default)]
struct ReaderStats {
    games_seen: AtomicU64,
    non_standard_start: AtomicU64,
    event: AtomicU64,
    elo: AtomicU64,
    termination: AtomicU64,
}

impl ReaderStats {
    fn record(&self, reason: RejectReason) {
        match reason {
            RejectReason::NonStandardStart => &self.non_standard_start,
            RejectReason::Event => &self.event,
            RejectReason::Elo => &self.elo,
            RejectReason::Termination => &self.termination,
            other => unreachable!("{other:?} is decided by a worker, not the reader"),
        }
        .fetch_add(1, Ordering::Relaxed);
    }
}

fn ingest_dump(
    path: &Path,
    state: &mut State,
    options: IngestOptions,
) -> Result<SourceV1, IngestError> {
    let io_error = |source| IngestError::Io {
        path: path.to_path_buf(),
        source,
    };
    let compressed_bytes = Arc::new(AtomicU64::new(0));
    let file = File::open(path).map_err(io_error)?;
    let mut decoder = zstd::stream::read::Decoder::new(HashingReader {
        inner: file,
        digest: Sha256::new(),
        bytes: Arc::clone(&compressed_bytes),
    })
    .map_err(io_error)?;
    // Lichess compresses with a long-range window. libzstd's default cap of 128
    // MiB would refuse those frames outright; the window is only allocated to
    // the size a frame actually declares.
    decoder
        .window_log_max(ZSTD_MAX_WINDOW_LOG)
        .map_err(io_error)?;
    let reader = PgnReader::new(BufReader::with_capacity(1 << 20, decoder));

    let stop = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(ReaderStats::default());
    let workers = options.workers;
    let mut job_senders = Vec::with_capacity(workers);
    let mut result_receivers = Vec::with_capacity(workers);

    let (reader, write_result) = std::thread::scope(|scope| {
        for _ in 0..workers {
            let (job_tx, job_rx) = mpsc::sync_channel::<Job>(QUEUE_CAPACITY);
            let (result_tx, result_rx) = mpsc::sync_channel::<Encoded>(QUEUE_CAPACITY);
            scope.spawn(move || {
                for job in job_rx {
                    if result_tx
                        .send(encode_job(job, options.min_plies, options.max_plies))
                        .is_err()
                    {
                        break;
                    }
                }
            });
            job_senders.push(job_tx);
            result_receivers.push(result_rx);
        }

        let reading = scope.spawn({
            let stop = Arc::clone(&stop);
            let stats = Arc::clone(&stats);
            let mut reader = reader;
            move || {
                let mut dispatched = 0_usize;
                while !stop.load(Ordering::Relaxed) {
                    let Some(game) = reader.next_game()? else {
                        break;
                    };
                    stats.games_seen.fetch_add(1, Ordering::Relaxed);
                    if let Some(reason) = header_reject(&game, options.min_elo) {
                        stats.record(reason);
                        continue;
                    }
                    let job = Job {
                        site: game.site,
                        movetext: game.movetext,
                    };
                    if job_senders[dispatched % workers].send(job).is_err() {
                        break;
                    }
                    dispatched += 1;
                }
                drop(job_senders);
                io::Result::Ok(reader)
            }
        });

        let write_result = collect(
            state,
            &result_receivers,
            &stop,
            &stats,
            path,
            &compressed_bytes,
            options,
        );
        let reader = reading
            .join()
            .unwrap_or_else(|_| Err(io::Error::other("the PGN reader thread panicked")));
        (reader, write_result)
    });

    // Finish hashing the dump over its whole length: the compressed tail is
    // read raw, without decoding, so an early stop still yields the file's
    // real checksum.
    let mut hashing = reader
        .map_err(io_error)?
        .into_inner()
        .into_inner()
        .finish()
        .into_inner();
    write_result.map_err(io_error)?;
    io::copy(&mut hashing, &mut io::sink()).map_err(io_error)?;

    state.counts.rejected.non_standard_start += stats.non_standard_start.load(Ordering::Relaxed);
    state.counts.rejected.event += stats.event.load(Ordering::Relaxed);
    state.counts.rejected.elo += stats.elo.load(Ordering::Relaxed);
    state.counts.rejected.termination += stats.termination.load(Ordering::Relaxed);
    let games_seen = stats.games_seen.load(Ordering::Relaxed);
    state.counts.games_seen += games_seen;

    Ok(SourceV1 {
        path: path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string()),
        sha256: hex::encode(hashing.digest.finalize()),
        compressed_bytes: compressed_bytes.load(Ordering::Relaxed),
        games_seen,
    })
}

/// Drain worker results in dispatch order until every queue closes.
///
/// A write failure or the token target stops the reader but never stops
/// draining: leaving results unread would deadlock the pool at scope exit.
fn collect(
    state: &mut State,
    result_receivers: &[mpsc::Receiver<Encoded>],
    stop: &AtomicBool,
    stats: &ReaderStats,
    path: &Path,
    compressed_bytes: &AtomicU64,
    options: IngestOptions,
) -> io::Result<()> {
    let workers = result_receivers.len();
    let started = Instant::now();
    let mut last_progress = Instant::now();
    let mut collected = 0_usize;
    let mut failure = None;

    while let Ok(encoded) = result_receivers[collected % workers].recv() {
        collected += 1;
        match encoded {
            Encoded::Tokens { site, tokens } => {
                state.counts.games_accepted += 1;
                let validation = is_validation_game(&site, options.val_fraction_ppm);
                if validation {
                    state.counts.games_val += 1;
                    state.counts.tokens_val += tokens.len() as u64;
                } else {
                    state.counts.games_train += 1;
                    state.counts.tokens_train += tokens.len() as u64;
                }
                let writer = if validation {
                    &mut state.val
                } else {
                    &mut state.train
                };
                if failure.is_none()
                    && let Err(error) = writer.push_game(&tokens)
                {
                    failure = Some(error);
                    stop.store(true, Ordering::Relaxed);
                }
            }
            Encoded::Rejected { site, reason } => {
                state.counts.rejected.record(reason);
                if reason == RejectReason::SanError
                    && state.san_error_samples.len() < SAN_ERROR_SAMPLES
                {
                    state.san_error_samples.push(site);
                }
            }
        }
        if state.total_tokens() >= options.token_target {
            stop.store(true, Ordering::Relaxed);
        }
        if last_progress.elapsed() >= PROGRESS_INTERVAL {
            report_progress(state, stats, path, compressed_bytes, started);
            last_progress = Instant::now();
        }
    }

    report_progress(state, stats, path, compressed_bytes, started);
    match failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn report_progress(
    state: &State,
    stats: &ReaderStats,
    path: &Path,
    compressed_bytes: &AtomicU64,
    started: Instant,
) {
    eprintln!(
        "{}",
        serde_json::json!({
            "event": "minigpt_ingest_progress",
            "dump": path.display().to_string(),
            "games_seen": stats.games_seen.load(Ordering::Relaxed),
            "games_accepted": state.counts.games_accepted,
            "tokens_train": state.counts.tokens_train,
            "tokens_val": state.counts.tokens_val,
            "compressed_mib_read": compressed_bytes.load(Ordering::Relaxed) as f64
                / (1024.0 * 1024.0),
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        })
    );
}

fn encode_job(job: Job, min_plies: u32, max_plies: u32) -> Encoded {
    let clean = match sanitize_movetext(&job.movetext) {
        Ok(clean) => clean,
        Err(SanitizeError::Variation) => {
            return Encoded::Rejected {
                site: job.site,
                reason: RejectReason::Variation,
            };
        }
        Err(SanitizeError::UnterminatedComment) => {
            return Encoded::Rejected {
                site: job.site,
                reason: RejectReason::SanError,
            };
        }
    };
    let plies = movetext_moves(&clean).count() as u32;
    if plies < min_plies || plies > max_plies {
        return Encoded::Rejected {
            site: job.site,
            reason: RejectReason::PlyBounds,
        };
    }
    match encode_movetext(&clean) {
        Ok(tokens) => Encoded::Tokens {
            site: job.site,
            tokens,
        },
        Err(_) => Encoded::Rejected {
            site: job.site,
            reason: RejectReason::SanError,
        },
    }
}

/// Hashes and counts the compressed bytes on their way into the zstd decoder,
/// so the dump checksum costs no second pass.
struct HashingReader<R> {
    inner: R,
    digest: Sha256,
    bytes: Arc<AtomicU64>,
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.digest.update(&buffer[..read]);
        self.bytes.fetch_add(read as u64, Ordering::Relaxed);
        Ok(read)
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::shard::read_shard;

    use super::*;

    const REJECTED_GAMES: &str = concat!(
        // Bullet.
        "[Event \"Rated Bullet game\"]\n[Site \"https://lichess.org/rej00001\"]\n",
        "[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[Termination \"Normal\"]\n\n",
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0\n\n",
        // Low Elo.
        "[Event \"Rated Blitz game\"]\n[Site \"https://lichess.org/rej00002\"]\n",
        "[WhiteElo \"1400\"]\n[BlackElo \"2600\"]\n[Termination \"Normal\"]\n\n",
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0\n\n",
        // Abandoned.
        "[Event \"Rated Rapid game\"]\n[Site \"https://lichess.org/rej00003\"]\n",
        "[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[Termination \"Abandoned\"]\n\n",
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0\n\n",
        // Set-up position.
        "[Event \"Rated Blitz game\"]\n[Site \"https://lichess.org/rej00004\"]\n",
        "[FEN \"8/8/8/8/8/8/8/K6k w - - 0 1\"]\n",
        "[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[Termination \"Normal\"]\n\n",
        "1. Kb2 Kh2 2. Kc2 Kg2 3. Kd2 Kf2 4. Ke2 Ke3 5. Kd1 Kd3 1-0\n\n",
        // Variation.
        "[Event \"Rated Blitz game\"]\n[Site \"https://lichess.org/rej00005\"]\n",
        "[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[Termination \"Normal\"]\n\n",
        "1. e4 e5 (1... c5 2. Nf3) 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0\n\n",
        // Too few plies.
        "[Event \"Rated Blitz game\"]\n[Site \"https://lichess.org/rej00006\"]\n",
        "[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[Termination \"Normal\"]\n\n",
        "1. e4 e5 2. Nf3 0-1\n\n",
        // Illegal SAN.
        "[Event \"Rated Blitz game\"]\n[Site \"https://lichess.org/rej00007\"]\n",
        "[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[Termination \"Normal\"]\n\n",
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Qz9 1-0\n\n",
    );

    const ACCEPTED_GAMES: &str = concat!(
        // Clock comments spanning lines, NAGs, and annotation suffixes.
        "[Event \"Rated Blitz game\"]\n[Site \"https://lichess.org/acc00001\"]\n",
        "[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[Termination \"Normal\"]\n\n",
        "1. e4?! { [%clk 0:03:00]\n",
        "[%eval 0.24] } e5 $2 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0\n\n",
        "[Event \"Rated Classical game\"]\n[Site \"https://lichess.org/acc00002\"]\n",
        "[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[Termination \"Time forfeit\"]\n\n",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 1/2-1/2\n\n",
        "[Event \"Rated Rapid game\"]\n[Site \"https://lichess.org/acc00003\"]\n",
        "[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[Termination \"Normal\"]\n\n",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 *\n",
    );

    fn options() -> IngestOptions {
        IngestOptions {
            min_elo: 2_000,
            min_plies: 10,
            max_plies: 300,
            token_target: u64::MAX,
            val_fraction_ppm: 0,
            shard_tokens: 1_000_000,
            workers: 3,
        }
    }

    fn write_dump(directory: &Path, name: &str, pgn: &str) -> PathBuf {
        let path = directory.join(name);
        std::fs::write(&path, zstd::stream::encode_all(pgn.as_bytes(), 3).unwrap()).unwrap();
        path
    }

    #[test]
    fn a_dump_is_filtered_tokenized_and_sealed_into_verifiable_shards() {
        let directory = tempfile::tempdir().unwrap();
        let pgn = format!("{REJECTED_GAMES}{ACCEPTED_GAMES}");
        let dump = write_dump(directory.path(), "dump.pgn.zst", &pgn);
        let output = directory.path().join("shards");

        let manifest = run(std::slice::from_ref(&dump), &output, options()).unwrap();

        assert_eq!(manifest.schema, "minigpt.shards.v1");
        assert_eq!(manifest.tokenizer, "policy-v1");
        assert_eq!(manifest.counts.games_seen, 10);
        assert_eq!(manifest.counts.games_accepted, 3);
        assert_eq!(manifest.counts.rejected.event, 1);
        assert_eq!(manifest.counts.rejected.elo, 1);
        assert_eq!(manifest.counts.rejected.termination, 1);
        assert_eq!(manifest.counts.rejected.non_standard_start, 1);
        assert_eq!(manifest.counts.rejected.variation, 1);
        assert_eq!(manifest.counts.rejected.ply_bounds, 1);
        assert_eq!(manifest.counts.rejected.san_error, 1);
        assert_eq!(
            manifest.san_error_samples,
            vec!["https://lichess.org/rej00007"]
        );
        assert_eq!(
            manifest.counts.games_seen,
            manifest.counts.games_accepted + manifest.counts.rejected.total()
        );

        // Every accepted game is ten plies plus its BOS token.
        assert_eq!(manifest.counts.tokens_train, 33);
        assert_eq!(manifest.counts.tokens_val, 0);
        assert!(manifest.val_shards.is_empty());
        assert_eq!(manifest.train_shards.len(), 1);

        let shard = &manifest.train_shards[0];
        assert_eq!(shard.game_count, 3);
        assert_eq!(shard.token_count, 33);
        assert_eq!(
            artifact_io::sha256_file(output.join(&shard.tokens_path)).unwrap(),
            shard.tokens_sha256
        );
        assert_eq!(
            artifact_io::sha256_file(output.join(&shard.index_path)).unwrap(),
            shard.index_sha256
        );

        let games = read_shard(
            &output.join(&shard.tokens_path),
            &output.join(&shard.index_path),
        )
        .unwrap();
        assert_eq!(games.len(), 3);
        assert_eq!(
            games[0],
            encode_movetext("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7").unwrap()
        );
        assert!(
            games
                .iter()
                .all(|game| game[0] == BOS_TOKEN && game.len() == 11)
        );

        assert_eq!(manifest.sources.len(), 1);
        assert_eq!(manifest.sources[0].path, "dump.pgn.zst");
        assert_eq!(
            manifest.sources[0].sha256,
            artifact_io::sha256_file(&dump).unwrap()
        );
        assert_eq!(
            manifest.sources[0].compressed_bytes,
            std::fs::metadata(&dump).unwrap().len()
        );

        // The manifest is serde round-trippable under deny_unknown_fields.
        let path = output.join(crate::manifest::SHARDS_MANIFEST_FILE);
        crate::manifest::write_manifest_atomic(&path, &manifest).unwrap();
        assert_eq!(crate::manifest::read_manifest(&path).unwrap(), manifest);
    }

    #[test]
    fn several_dumps_are_read_in_order_and_share_the_shard_sequence() {
        let directory = tempfile::tempdir().unwrap();
        let first = write_dump(directory.path(), "first.pgn.zst", ACCEPTED_GAMES);
        let second = write_dump(directory.path(), "second.pgn.zst", ACCEPTED_GAMES);
        let output = directory.path().join("shards");

        let manifest = run(
            &[first, second],
            &output,
            IngestOptions {
                shard_tokens: 22,
                ..options()
            },
        )
        .unwrap();

        assert_eq!(manifest.counts.games_accepted, 6);
        assert_eq!(
            manifest
                .sources
                .iter()
                .map(|source| source.path.as_str())
                .collect::<Vec<_>>(),
            vec!["first.pgn.zst", "second.pgn.zst"]
        );
        assert_eq!(
            manifest
                .train_shards
                .iter()
                .map(|shard| (shard.tokens_path.clone(), shard.game_count))
                .collect::<Vec<_>>(),
            vec![
                ("shard-0000.bin".to_string(), 2),
                ("shard-0001.bin".to_string(), 2),
                ("shard-0002.bin".to_string(), 2),
            ]
        );

        let games: Vec<Vec<u16>> = manifest
            .train_shards
            .iter()
            .flat_map(|shard| {
                read_shard(
                    &output.join(&shard.tokens_path),
                    &output.join(&shard.index_path),
                )
                .unwrap()
            })
            .collect();
        assert_eq!(games[..3], games[3..]);
    }

    #[test]
    fn the_token_target_stops_the_run_and_still_checksums_the_whole_dump() {
        let directory = tempfile::tempdir().unwrap();
        let pgn = ACCEPTED_GAMES.repeat(200);
        let dump = write_dump(directory.path(), "dump.pgn.zst", &pgn);
        let output = directory.path().join("shards");

        let manifest = run(
            std::slice::from_ref(&dump),
            &output,
            IngestOptions {
                token_target: 100,
                ..options()
            },
        )
        .unwrap();

        assert!(manifest.counts.tokens_train >= 100);
        assert!(
            manifest.counts.games_seen < 600,
            "the reader should stop early"
        );
        assert_eq!(
            manifest.sources[0].sha256,
            artifact_io::sha256_file(&dump).unwrap(),
            "an early stop must still hash the whole file"
        );
        assert_eq!(
            manifest.counts.games_seen,
            manifest.counts.games_accepted + manifest.counts.rejected.total()
        );
    }

    #[test]
    fn the_split_routes_games_by_id_and_is_stable_across_runs() {
        let ids: Vec<String> = (0..20_000)
            .map(|index| format!("https://lichess.org/{index:08}"))
            .collect();
        let validation: HashSet<&String> = ids
            .iter()
            .filter(|site| is_validation_game(site, 100_000))
            .collect();

        // Ten percent of ids, within sampling noise, and the same set every time.
        assert!(
            (1_500..2_500).contains(&validation.len()),
            "{} routed to validation",
            validation.len()
        );
        assert!(
            ids.iter()
                .all(|site| validation.contains(site) == is_validation_game(site, 100_000))
        );
        assert!(!ids.iter().any(|site| is_validation_game(site, 0)));
        assert!(ids.iter().all(|site| is_validation_game(site, 1_000_000)));
    }

    #[test]
    fn a_validation_fraction_splits_shards_by_prefix() {
        let directory = tempfile::tempdir().unwrap();
        let dump = write_dump(directory.path(), "dump.pgn.zst", ACCEPTED_GAMES);
        let output = directory.path().join("shards");

        let manifest = run(
            &[dump],
            &output,
            IngestOptions {
                val_fraction_ppm: 1_000_000,
                ..options()
            },
        )
        .unwrap();

        assert_eq!(manifest.counts.games_val, 3);
        assert_eq!(manifest.counts.games_train, 0);
        assert!(manifest.train_shards.is_empty());
        assert_eq!(manifest.val_shards[0].tokens_path, "val-0000.bin");
        assert_eq!(manifest.val_shards[0].index_path, "val-0000.idx");
    }
}
