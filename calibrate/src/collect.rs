use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::process::Command;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::pgn::{ChessComArchive, ChessComGame};

#[derive(Clone, Debug)]
pub struct CollectConfig {
    pub seed_users: Vec<String>,
    pub max_users: usize,
    pub max_games: usize,
    pub max_games_per_user: usize,
    /// Save a game only if at least one participant meets this rating.
    pub minimum_participant_rating: u32,
    pub seed: u64,
    pub user_agent: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectStats {
    pub users_queried: usize,
    pub games_written: usize,
    pub duplicate_games: usize,
    pub player_cap_skips: usize,
    pub users_without_games: usize,
}

pub fn collect<W, F>(
    config: &CollectConfig,
    output: &mut W,
    mut progress: F,
) -> Result<CollectStats, String>
where
    W: Write,
    F: FnMut(&str, CollectStats),
{
    if config.seed_users.is_empty() {
        return Err("at least one seed username is required".to_string());
    }
    if config.max_users == 0 || config.max_games == 0 || config.max_games_per_user == 0 {
        return Err(
            "max users, max games, and max games per user must be greater than zero".to_string(),
        );
    }

    let mut queued = HashSet::new();
    let mut queue = VecDeque::new();
    for username in &config.seed_users {
        enqueue(username, &mut queued, &mut queue)?;
    }

    let mut seen_games = HashSet::new();
    let mut player_game_counts: HashMap<String, usize> = HashMap::new();
    let mut stats = CollectStats::default();
    let mut rng = StdRng::seed_from_u64(config.seed);
    while let Some(username) = queue.pop_front() {
        if stats.users_queried >= config.max_users || stats.games_written >= config.max_games {
            break;
        }
        let games = fetch_30_0_games(&username, &config.user_agent)?;
        stats.users_queried += 1;
        if games.is_empty() {
            stats.users_without_games += 1;
        }

        for game in &games {
            // Following both players, including from unrated games, grows a
            // time-control-specific opponent graph. Only rated games are saved.
            enqueue(&game.white.username, &mut queued, &mut queue)?;
            enqueue(&game.black.username, &mut queued, &mut queue)?;
        }

        let mut eligible: Vec<_> = games
            .into_iter()
            .filter(|game| {
                game.is_rated_standard_30_0()
                    && (game.white.rating >= config.minimum_participant_rating
                        || game.black.rating >= config.minimum_participant_rating)
            })
            .collect();
        eligible.shuffle(&mut rng);
        for game in eligible {
            if !seen_games.insert(game.url.clone()) {
                stats.duplicate_games += 1;
                continue;
            }
            let white = game.white.username.to_ascii_lowercase();
            let black = game.black.username.to_ascii_lowercase();
            if participant_is_capped(&player_game_counts, &white, config.max_games_per_user)
                || participant_is_capped(&player_game_counts, &black, config.max_games_per_user)
            {
                stats.player_cap_skips += 1;
                continue;
            }

            serde_json::to_writer(&mut *output, &game)
                .map_err(|error| format!("could not encode game: {error}"))?;
            output
                .write_all(b"\n")
                .map_err(|error| format!("could not write corpus: {error}"))?;
            stats.games_written += 1;
            *player_game_counts.entry(white).or_default() += 1;
            *player_game_counts.entry(black).or_default() += 1;
            if stats.games_written >= config.max_games {
                break;
            }
        }
        progress(&username, stats);
    }

    Ok(stats)
}

fn participant_is_capped(
    counts: &HashMap<String, usize>,
    username: &str,
    max_games: usize,
) -> bool {
    counts.get(username).copied().unwrap_or(0) >= max_games
}

fn enqueue(
    username: &str,
    queued: &mut HashSet<String>,
    queue: &mut VecDeque<String>,
) -> Result<(), String> {
    if username.is_empty()
        || !username
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
    {
        return Err(format!(
            "username contains unsupported characters: {username:?}"
        ));
    }
    let normalized = username.to_ascii_lowercase();
    if queued.insert(normalized.clone()) {
        queue.push_back(normalized);
    }
    Ok(())
}

fn fetch_30_0_games(username: &str, user_agent: &str) -> Result<Vec<ChessComGame>, String> {
    let url = format!("https://api.chess.com/pub/player/{username}/games/live/1800/0");
    let response = Command::new("curl")
        .args([
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "60",
            "--user-agent",
            user_agent,
            "--write-out",
            "\n%{http_code}",
            &url,
        ])
        .output()
        .map_err(|error| format!("could not start curl: {error}"))?;
    if !response.status.success() {
        return Err(format!(
            "curl failed for {username}: {}",
            String::from_utf8_lossy(&response.stderr).trim()
        ));
    }

    let text = String::from_utf8(response.stdout)
        .map_err(|error| format!("Chess.com returned non-UTF-8 data: {error}"))?;
    let (body, status) = text
        .rsplit_once('\n')
        .ok_or_else(|| "curl response did not contain an HTTP status".to_string())?;
    match status {
        "200" => serde_json::from_str::<ChessComArchive>(body)
            .map(|archive| archive.games)
            .map_err(|error| format!("invalid Chess.com response for {username}: {error}")),
        "404" => Ok(Vec::new()),
        "429" => Err("Chess.com rate limited the collector; wait and resume later".to_string()),
        _ => Err(format!(
            "Chess.com returned HTTP {status} for {username}: {}",
            body.trim()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usernames_are_normalized_and_deduplicated() {
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        enqueue("Player_One", &mut seen, &mut queue).unwrap();
        enqueue("player_one", &mut seen, &mut queue).unwrap();
        assert_eq!(queue.into_iter().collect::<Vec<_>>(), vec!["player_one"]);
    }

    #[test]
    fn unsafe_username_characters_are_rejected() {
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        assert!(enqueue("not/a/user", &mut seen, &mut queue).is_err());
    }

    #[test]
    fn participant_cap_is_global_and_case_insensitive_after_normalization() {
        let counts = HashMap::from([("player_one".to_string(), 20)]);
        assert!(participant_is_capped(&counts, "player_one", 20));
        assert!(!participant_is_capped(&counts, "player_two", 20));
    }
}
