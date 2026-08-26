//! On-disk token shards.
//!
//! `<prefix>-NNNN.bin` is the raw token stream: little-endian `u16`, games
//! concatenated in the order they were read, each game being its `BOS` token
//! followed by one action token per ply. No padding is stored.
//!
//! `<prefix>-NNNN.idx` locates those games. It is little-endian `u64`
//! throughout: a game count `G`, then `G + 1` offsets. Offsets are measured in
//! **tokens**, not bytes; multiply by two for a byte offset into the `.bin`.
//! Game `i` occupies `offsets[i]..offsets[i + 1]`, `offsets[0]` is always zero,
//! and `offsets[G]` is the shard's total token count. The file is therefore
//! exactly `(G + 2) * 8` bytes.

use std::io;
use std::path::{Path, PathBuf};

use artifact_io::{publish_bytes_new, sha256_bytes};

use crate::manifest::ShardFileV1;

/// Accumulates games in memory and seals a shard once it reaches its token
/// budget. Shards are published atomically and never overwritten.
pub struct ShardWriter {
    directory: PathBuf,
    prefix: String,
    shard_tokens: u64,
    tokens: Vec<u16>,
    offsets: Vec<u64>,
    files: Vec<ShardFileV1>,
    total_tokens: u64,
    total_games: u64,
}

impl ShardWriter {
    pub fn new(
        directory: impl Into<PathBuf>,
        prefix: impl Into<String>,
        shard_tokens: u64,
    ) -> Self {
        Self {
            directory: directory.into(),
            prefix: prefix.into(),
            shard_tokens: shard_tokens.max(1),
            tokens: Vec::new(),
            offsets: Vec::new(),
            files: Vec::new(),
            total_tokens: 0,
            total_games: 0,
        }
    }

    pub fn push_game(&mut self, tokens: &[u16]) -> io::Result<()> {
        self.offsets.push(self.tokens.len() as u64);
        self.tokens.extend_from_slice(tokens);
        self.total_tokens += tokens.len() as u64;
        self.total_games += 1;
        if self.tokens.len() as u64 >= self.shard_tokens {
            self.seal()?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<Vec<ShardFileV1>> {
        self.seal()?;
        Ok(self.files)
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    pub fn total_games(&self) -> u64 {
        self.total_games
    }

    fn seal(&mut self) -> io::Result<()> {
        if self.offsets.is_empty() {
            return Ok(());
        }
        let stem = format!("{}-{:04}", self.prefix, self.files.len());
        let tokens_path = self.directory.join(format!("{stem}.bin"));
        let index_path = self.directory.join(format!("{stem}.idx"));

        let mut tokens = Vec::with_capacity(self.tokens.len() * 2);
        for token in self.tokens.drain(..) {
            tokens.extend_from_slice(&token.to_le_bytes());
        }
        let mut index = Vec::with_capacity((self.offsets.len() + 2) * 8);
        index.extend_from_slice(&(self.offsets.len() as u64).to_le_bytes());
        for offset in self.offsets.drain(..) {
            index.extend_from_slice(&offset.to_le_bytes());
        }
        index.extend_from_slice(&((tokens.len() / 2) as u64).to_le_bytes());

        publish_bytes_new(&tokens_path, &tokens)?;
        publish_bytes_new(&index_path, &index)?;
        self.files.push(ShardFileV1 {
            tokens_path: file_name(&tokens_path),
            index_path: file_name(&index_path),
            tokens_sha256: sha256_bytes(&tokens),
            index_sha256: sha256_bytes(&index),
            token_count: (tokens.len() / 2) as u64,
            game_count: (index.len() / 8 - 2) as u64,
        });
        Ok(())
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .expect("shard paths are built with a file name")
        .to_string_lossy()
        .into_owned()
}

/// Read one shard back into per-game token streams, the reference decoding of
/// the layout documented above.
pub fn read_shard(tokens_path: &Path, index_path: &Path) -> io::Result<Vec<Vec<u16>>> {
    let tokens = std::fs::read(tokens_path)?;
    let index = std::fs::read(index_path)?;
    if tokens.len() % 2 != 0 || index.len() % 8 != 0 || index.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shard files are not a whole number of tokens and offsets",
        ));
    }
    let words: Vec<u64> = index
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8) yields 8 bytes")))
        .collect();
    let game_count = words[0] as usize;
    let offsets = &words[1..];
    if offsets.len() != game_count + 1 || offsets[game_count] != (tokens.len() / 2) as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shard index does not describe the token stream",
        ));
    }
    let stream: Vec<u16> = tokens
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("chunks_exact(2) yields 2 bytes")))
        .collect();
    (0..game_count)
        .map(|game| {
            let (start, end) = (offsets[game] as usize, offsets[game + 1] as usize);
            if start > end || end > stream.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("shard index entry {game} is out of range"),
                ));
            }
            Ok(stream[start..end].to_vec())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::BOS_TOKEN;

    fn game(length: u16) -> Vec<u16> {
        let mut tokens = vec![BOS_TOKEN];
        tokens.extend(1..length);
        tokens
    }

    #[test]
    fn games_round_trip_through_shards_that_roll_over_a_token_budget() {
        let directory = tempfile::tempdir().unwrap();
        let games: Vec<Vec<u16>> = (3..9).map(game).collect();

        let mut writer = ShardWriter::new(directory.path(), "shard", 10);
        for tokens in &games {
            writer.push_game(tokens).unwrap();
        }
        assert_eq!(writer.total_games(), 6);
        let total_tokens = writer.total_tokens();
        let files = writer.finish().unwrap();

        assert!(files.len() > 1, "the budget should have rolled a shard");
        assert_eq!(
            files.iter().map(|file| file.token_count).sum::<u64>(),
            total_tokens
        );
        assert_eq!(files.iter().map(|file| file.game_count).sum::<u64>(), 6);

        let mut read_back = Vec::new();
        for (index, file) in files.iter().enumerate() {
            assert_eq!(file.tokens_path, format!("shard-{index:04}.bin"));
            let tokens_path = directory.path().join(&file.tokens_path);
            let index_path = directory.path().join(&file.index_path);
            assert_eq!(
                artifact_io::sha256_file(&tokens_path).unwrap(),
                file.tokens_sha256
            );
            assert_eq!(
                artifact_io::sha256_file(&index_path).unwrap(),
                file.index_sha256
            );
            assert_eq!(
                std::fs::metadata(&tokens_path).unwrap().len(),
                file.token_count * 2
            );
            assert_eq!(
                std::fs::metadata(&index_path).unwrap().len(),
                (file.game_count + 2) * 8
            );
            read_back.extend(read_shard(&tokens_path, &index_path).unwrap());
        }
        assert_eq!(read_back, games);
    }

    #[test]
    fn a_writer_that_saw_no_games_publishes_nothing() {
        let directory = tempfile::tempdir().unwrap();
        let files = ShardWriter::new(directory.path(), "val", 10)
            .finish()
            .unwrap();
        assert!(files.is_empty());
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }
}
