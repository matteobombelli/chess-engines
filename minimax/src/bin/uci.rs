use std::io::{self, BufRead, Write};
use std::time::Duration;

use chess_core::{Board, Color};
use minimax::{SearchLimits, find_best_move};

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut board = Board::from_fen(START_FEN).expect("valid starting FEN");
    let defaults = SearchLimits::from_env()
        .unwrap_or_else(|error| panic!("invalid minimax configuration: {error}"));

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let command = line.trim();
        let result = if command == "uci" {
            writeln!(stdout, "id name Minimax")
                .and_then(|_| writeln!(stdout, "uciok"))
                .map_err(|error| error.to_string())
        } else if command == "isready" {
            writeln!(stdout, "readyok").map_err(|error| error.to_string())
        } else if command == "ucinewgame" {
            board = Board::from_fen(START_FEN).expect("valid starting FEN");
            Ok(())
        } else if command.starts_with("position ") {
            set_position(&mut board, command)
        } else if command == "go" || command.starts_with("go ") {
            let limits = limits_from_go(command, board.side_to_move, defaults);
            match find_best_move(&board, limits) {
                Ok(result) => {
                    let elapsed_ms = result.stats.elapsed.as_millis().max(1);
                    let nps = u128::from(result.stats.nodes) * 1000 / elapsed_ms;
                    writeln!(
                        stdout,
                        "info depth {} score cp {} nodes {} nps {} pv {}",
                        result.stats.completed_depth,
                        result.score,
                        result.stats.nodes,
                        nps,
                        result
                            .principal_variation
                            .iter()
                            .map(|mv| mv.to_uci())
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                    .and_then(|_| writeln!(stdout, "bestmove {}", result.best_move.to_uci()))
                    .map_err(|error| error.to_string())
                }
                Err(error) => writeln!(stdout, "info string {error}")
                    .and_then(|_| writeln!(stdout, "bestmove 0000"))
                    .map_err(|write_error| write_error.to_string()),
            }
        } else if command == "quit" {
            break;
        } else if command == "stop" || command.is_empty() {
            Ok(())
        } else {
            writeln!(stdout, "info string unsupported command: {command}")
                .map_err(|error| error.to_string())
        };

        if let Err(error) = result {
            let _ = writeln!(stdout, "info string error: {error}");
        }
        let _ = stdout.flush();
    }
}

fn set_position(board: &mut Board, command: &str) -> Result<(), String> {
    let fields: Vec<&str> = command.split_whitespace().collect();
    let mut cursor = 1;
    let mut next = match fields.get(cursor).copied() {
        Some("startpos") => {
            cursor += 1;
            Board::from_fen(START_FEN)?
        }
        Some("fen") => {
            cursor += 1;
            if fields.len() < cursor + 6 {
                return Err("position fen requires all six FEN fields".to_string());
            }
            let fen = fields[cursor..cursor + 6].join(" ");
            cursor += 6;
            Board::from_fen(&fen)?
        }
        _ => return Err("expected `position startpos` or `position fen ...`".to_string()),
    };

    if fields.get(cursor) == Some(&"moves") {
        cursor += 1;
        for uci in &fields[cursor..] {
            next.uci_to_move(uci)?;
        }
    } else if cursor != fields.len() {
        return Err("expected `moves` after the base position".to_string());
    }
    *board = next;
    Ok(())
}

fn limits_from_go(command: &str, side: Color, defaults: SearchLimits) -> SearchLimits {
    let fields: Vec<&str> = command.split_whitespace().collect();
    let value_after = |name: &str| {
        fields
            .iter()
            .position(|field| *field == name)
            .and_then(|index| fields.get(index + 1))
            .and_then(|value| value.parse::<u64>().ok())
    };

    let max_depth = value_after("depth")
        .and_then(|depth| u8::try_from(depth).ok())
        .filter(|depth| (1..=SearchLimits::MAX_SUPPORTED_DEPTH).contains(depth))
        .unwrap_or(defaults.max_depth);
    let max_nodes = value_after("nodes").or(defaults.max_nodes);
    let explicit_time = value_after("movetime").map(Duration::from_millis);

    let (clock_name, increment_name) = match side {
        Color::White => ("wtime", "winc"),
        Color::Black => ("btime", "binc"),
    };
    let clock_time = value_after(clock_name).map(|remaining| {
        let increment = value_after(increment_name).unwrap_or(0);
        let budget = (remaining / 30 + increment * 3 / 4)
            .max(1)
            .min((remaining / 2).max(1));
        Duration::from_millis(budget)
    });

    SearchLimits {
        max_depth,
        move_time: explicit_time.or(clock_time).or(defaults.move_time),
        max_nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_position() {
        let mut board = Board::from_fen(START_FEN).unwrap();
        set_position(&mut board, "position startpos moves e2e4 e7e5").unwrap();
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.san_history, ["e4", "e5"]);

        set_position(&mut board, "position fen 4k3/8/8/8/8/8/8/4K3 b - - 0 7").unwrap();
        assert_eq!(board.side_to_move, Color::Black);
        assert_eq!(board.fullmove_number, 7);
    }

    #[test]
    fn parses_go_limits() {
        let defaults = SearchLimits::fixed_depth(5).unwrap();
        let limits = limits_from_go(
            "go depth 7 wtime 30000 btime 90000 winc 1000 nodes 1234",
            Color::White,
            defaults,
        );
        assert_eq!(limits.max_depth, 7);
        assert_eq!(limits.max_nodes, Some(1234));
        assert_eq!(limits.move_time, Some(Duration::from_millis(1750)));
    }
}
