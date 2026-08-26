use std::io::{self, BufRead};

use thiserror::Error;

/// Why a game was not written to a shard. Every game read from a dump either
/// lands in a shard or is counted under exactly one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// A `[FEN]` tag, or a `[Variant]` tag naming something other than standard.
    NonStandardStart,
    /// Not a Blitz/Rapid/Classical `[Event]`.
    Event,
    /// A missing, unparseable, or too-low `[WhiteElo]`/`[BlackElo]`.
    Elo,
    /// A `[Termination]` other than Normal or Time forfeit.
    Termination,
    /// Movetext containing a variation, which is not a single game.
    Variation,
    /// Ply count outside the configured bounds.
    PlyBounds,
    /// Movetext that failed to sanitize, parse, or replay legally.
    SanError,
}

/// The tags this pipeline filters or splits on, plus the raw movetext.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RawGame {
    pub site: String,
    pub event: String,
    pub white_elo: Option<u32>,
    pub black_elo: Option<u32>,
    pub termination: String,
    pub non_standard_start: bool,
    pub movetext: String,
}

/// Split a PGN stream into games.
///
/// A game runs from its first tag line to the blank line that closes its
/// movetext, or to end of input. Tag lines are only recognized before the
/// movetext starts, so a `{ [%clk ...] }` comment continuing onto its own line
/// stays part of the movetext.
pub struct PgnReader<R> {
    reader: R,
    line: String,
}

impl<R: BufRead> PgnReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line: String::new(),
        }
    }

    pub fn into_inner(self) -> R {
        self.reader
    }

    pub fn next_game(&mut self) -> io::Result<Option<RawGame>> {
        let mut game = RawGame::default();
        let mut saw_tag = false;
        let mut in_movetext = false;
        loop {
            self.line.clear();
            if self.reader.read_line(&mut self.line)? == 0 {
                return Ok(if saw_tag || in_movetext {
                    Some(game)
                } else {
                    None
                });
            }
            let trimmed: &str = self.line.trim();
            if trimmed.is_empty() {
                if in_movetext {
                    return Ok(Some(game));
                }
                continue;
            }
            if !in_movetext && trimmed.starts_with('[') {
                saw_tag = true;
                apply_tag(&mut game, trimmed);
                continue;
            }
            in_movetext = true;
            if !game.movetext.is_empty() {
                game.movetext.push(' ');
            }
            game.movetext.push_str(trimmed);
        }
    }
}

fn apply_tag(game: &mut RawGame, line: &str) {
    let body: &str = line
        .strip_prefix('[')
        .unwrap_or(line)
        .strip_suffix(']')
        .unwrap_or(line);
    let Some((key, rest)) = body.split_once(' ') else {
        return;
    };
    let value: &str = rest.trim().trim_matches('"');
    match key {
        "Site" => value.clone_into(&mut game.site),
        "Event" => value.clone_into(&mut game.event),
        "WhiteElo" => game.white_elo = value.parse().ok(),
        "BlackElo" => game.black_elo = value.parse().ok(),
        "Termination" => value.clone_into(&mut game.termination),
        "FEN" => game.non_standard_start = true,
        "Variant" => game.non_standard_start |= !value.eq_ignore_ascii_case("Standard"),
        _ => {}
    }
}

/// Apply the tag-only filters, which is everything that can be decided without
/// replaying the movetext.
pub fn header_reject(game: &RawGame, min_elo: u32) -> Option<RejectReason> {
    if game.non_standard_start {
        return Some(RejectReason::NonStandardStart);
    }
    if !is_allowed_event(&game.event) {
        return Some(RejectReason::Event);
    }
    match (game.white_elo, game.black_elo) {
        (Some(white), Some(black)) if white >= min_elo && black >= min_elo => {}
        _ => return Some(RejectReason::Elo),
    }
    if !is_allowed_termination(&game.termination) {
        return Some(RejectReason::Termination);
    }
    None
}

/// Lichess events name the time control inside a longer phrase, e.g.
/// "Rated Blitz game" or "Blitz Arena Titled Arena".
fn is_allowed_event(event: &str) -> bool {
    let event: String = event.to_ascii_lowercase();
    !event.contains("bullet")
        && (event.contains("blitz") || event.contains("rapid") || event.contains("classical"))
}

fn is_allowed_termination(termination: &str) -> bool {
    termination.eq_ignore_ascii_case("Normal") || termination.eq_ignore_ascii_case("Time forfeit")
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SanitizeError {
    #[error("movetext has a `{{` comment that is never closed")]
    UnterminatedComment,
    #[error("movetext contains a variation")]
    Variation,
}

/// Reduce annotated PGN movetext to bare SAN tokens.
///
/// `movetext_moves` treats anything it does not recognize as bookkeeping as a
/// move, so comments and NAGs have to be removed here rather than being
/// tolerated downstream. Comments are stripped first: a `(` inside one is
/// annotation, not a variation. A game that really does carry a variation is
/// rejected rather than flattened into a different game.
pub fn sanitize_movetext(movetext: &str) -> Result<String, SanitizeError> {
    let mut clean = String::with_capacity(movetext.len());
    let mut rest: &str = movetext;
    while let Some(index) = rest.find(['{', '$', '(']) {
        clean.push_str(&rest[..index]);
        clean.push(' ');
        match &rest[index..index + 1] {
            "{" => {
                let end: usize = rest[index..]
                    .find('}')
                    .ok_or(SanitizeError::UnterminatedComment)?;
                rest = &rest[index + end + 1..];
            }
            "$" => {
                let end: usize = rest[index..]
                    .find(char::is_whitespace)
                    .unwrap_or(rest.len() - index);
                rest = &rest[index + end..];
            }
            _ => return Err(SanitizeError::Variation),
        }
    }
    clean.push_str(rest);
    Ok(clean)
}

#[cfg(test)]
mod tests {
    use chess_core::movetext_moves;

    use super::*;

    fn games(pgn: &str) -> Vec<RawGame> {
        let mut reader = PgnReader::new(pgn.as_bytes());
        let mut out = Vec::new();
        while let Some(game) = reader.next_game().unwrap() {
            out.push(game);
        }
        out
    }

    fn headers(event: &str, elo: u32, termination: &str) -> RawGame {
        RawGame {
            site: "https://lichess.org/abcd1234".to_string(),
            event: event.to_string(),
            white_elo: Some(elo),
            black_elo: Some(elo),
            termination: termination.to_string(),
            non_standard_start: false,
            movetext: String::new(),
        }
    }

    #[test]
    fn a_stream_splits_into_games_with_the_tags_we_filter_on() {
        let pgn = concat!(
            "[Event \"Rated Blitz game\"]\n",
            "[Site \"https://lichess.org/aaaa1111\"]\n",
            "[WhiteElo \"2410\"]\n",
            "[BlackElo \"2388\"]\n",
            "[Termination \"Normal\"]\n",
            "\n",
            "1. e4 e5 2. Nf3 1-0\n",
            "\n",
            "[Event \"Rated Bullet game\"]\n",
            "[Site \"https://lichess.org/bbbb2222\"]\n",
            "[WhiteElo \"?\"]\n",
            "[BlackElo \"1200\"]\n",
            "[Termination \"Abandoned\"]\n",
            "\n",
            "1. d4 0-1\n",
        );
        let games = games(pgn);
        assert_eq!(games.len(), 2);
        assert_eq!(games[0].site, "https://lichess.org/aaaa1111");
        assert_eq!(games[0].event, "Rated Blitz game");
        assert_eq!(games[0].white_elo, Some(2_410));
        assert_eq!(games[0].movetext, "1. e4 e5 2. Nf3 1-0");
        assert_eq!(games[1].white_elo, None);
        assert_eq!(games[1].termination, "Abandoned");
    }

    #[test]
    fn movetext_spanning_lines_and_comments_stays_one_game() {
        let pgn = concat!(
            "[Event \"Rated Rapid game\"]\n",
            "[Site \"https://lichess.org/cccc3333\"]\n",
            "\n",
            "1. e4 { [%clk 0:03:00]\n",
            "[%eval 0.17] } e5\n",
            "2. Nf3 { comment } 1/2-1/2\n",
            "\n",
        );
        let games = games(pgn);
        assert_eq!(games.len(), 1);
        assert_eq!(
            games[0].movetext,
            "1. e4 { [%clk 0:03:00] [%eval 0.17] } e5 2. Nf3 { comment } 1/2-1/2"
        );
        let clean = sanitize_movetext(&games[0].movetext).unwrap();
        assert_eq!(
            movetext_moves(&clean).collect::<Vec<_>>(),
            vec!["e4", "e5", "Nf3"]
        );
    }

    #[test]
    fn a_final_game_without_a_trailing_blank_line_is_still_returned() {
        let pgn = "[Event \"Rated Classical game\"]\n\n1. e4 *";
        assert_eq!(games(pgn).len(), 1);
    }

    #[test]
    fn nags_and_annotation_suffixes_survive_sanitizing() {
        let clean = sanitize_movetext("1. e4?! $2 e5!? $146 2. Nf3 1-0").unwrap();
        assert_eq!(
            movetext_moves(&clean).collect::<Vec<_>>(),
            vec!["e4?!", "e5!?", "Nf3"]
        );
    }

    #[test]
    fn result_tokens_are_not_moves() {
        for result in ["1-0", "0-1", "1/2-1/2", "*"] {
            let clean = sanitize_movetext(&format!("1. e4 e5 {result}")).unwrap();
            assert_eq!(
                movetext_moves(&clean).collect::<Vec<_>>(),
                vec!["e4", "e5"],
                "{result}"
            );
        }
    }

    #[test]
    fn variations_are_rejected_and_parentheses_in_comments_are_not() {
        assert_eq!(
            sanitize_movetext("1. e4 e5 (1... c5 2. Nf3) 2. Nf3 1-0"),
            Err(SanitizeError::Variation)
        );
        let clean = sanitize_movetext("1. e4 { a (good) move } e5 1-0").unwrap();
        assert_eq!(movetext_moves(&clean).collect::<Vec<_>>(), vec!["e4", "e5"]);
        assert_eq!(
            sanitize_movetext("1. e4 { never closed e5"),
            Err(SanitizeError::UnterminatedComment)
        );
    }

    #[test]
    fn header_filters_accept_the_time_controls_we_train_on() {
        for event in [
            "Rated Blitz game",
            "Rated Rapid game",
            "Rated Classical game",
            "rated blitz tournament https://lichess.org/tournament/x",
        ] {
            assert_eq!(header_reject(&headers(event, 2_400, "Normal"), 2_000), None);
        }
        assert_eq!(
            header_reject(&headers("Rated Blitz game", 2_400, "Time forfeit"), 2_000),
            None
        );
    }

    #[test]
    fn header_filters_reject_bullet_low_elo_and_odd_terminations() {
        assert_eq!(
            header_reject(&headers("Rated Bullet game", 2_400, "Normal"), 2_000),
            Some(RejectReason::Event)
        );
        assert_eq!(
            header_reject(&headers("Rated UltraBullet game", 2_400, "Normal"), 2_000),
            Some(RejectReason::Event)
        );
        assert_eq!(
            header_reject(
                &headers("Rated Correspondence game", 2_400, "Normal"),
                2_000
            ),
            Some(RejectReason::Event)
        );
        assert_eq!(
            header_reject(&headers("Rated Blitz game", 1_999, "Normal"), 2_000),
            Some(RejectReason::Elo)
        );
        assert_eq!(
            header_reject(&headers("Rated Blitz game", 2_400, "Abandoned"), 2_000),
            Some(RejectReason::Termination)
        );
        assert_eq!(
            header_reject(
                &headers("Rated Blitz game", 2_400, "Rules infraction"),
                2_000
            ),
            Some(RejectReason::Termination)
        );

        let mut unrated = headers("Rated Blitz game", 2_400, "Normal");
        unrated.black_elo = None;
        assert_eq!(header_reject(&unrated, 2_000), Some(RejectReason::Elo));
    }

    #[test]
    fn a_setup_position_is_rejected_but_an_explicit_standard_variant_is_not() {
        let pgn = concat!(
            "[Event \"Rated Blitz game\"]\n",
            "[Variant \"Standard\"]\n",
            "\n",
            "1. e4 1-0\n",
            "\n",
            "[Event \"Rated Blitz game\"]\n",
            "[FEN \"8/8/8/8/8/8/8/K6k w - - 0 1\"]\n",
            "\n",
            "1. Kb2 1-0\n",
            "\n",
            "[Event \"Rated Blitz game\"]\n",
            "[Variant \"Chess960\"]\n",
            "\n",
            "1. e4 1-0\n",
        );
        let games = games(pgn);
        assert!(!games[0].non_standard_start);
        assert!(games[1].non_standard_start);
        assert!(games[2].non_standard_start);
        assert_eq!(
            header_reject(&games[1], 0),
            Some(RejectReason::NonStandardStart)
        );
    }
}
