use chess_core::{Board, Color, Piece, PieceKind, Square, Status};
use gloo_net::http::Request;
use leptos::mount::mount_to_body;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const BOT_URL: &str = "http://127.0.0.1:3000/move";

#[derive(Serialize)]
struct BotRequest {
    fen: String,
    san: Option<String>,
}

#[derive(Deserialize)]
struct BotResponse {
    san: String,
    fen: String,
}

#[component]
fn App() -> impl IntoView {
    let board = RwSignal::new(Board::from_fen(START_FEN).expect("valid start position"));
    let selected = RwSignal::new(None::<Square>);
    let history = RwSignal::new(Vec::<String>::new());
    let thinking = RwSignal::new(false);
    let error = RwSignal::new(None::<String>);
    let game_id = RwSignal::new(0_u32);

    let reset = move |_| {
        board.set(Board::from_fen(START_FEN).expect("valid start position"));
        selected.set(None);
        history.set(Vec::new());
        thinking.set(false);
        error.set(None);
        game_id.update(|id| *id += 1);
    };

    let play_square = move |square: Square| {
        if thinking.get_untracked() || board.get_untracked().status() != Status::Ongoing {
            return;
        }

        let current = board.get_untracked();
        if current.side_to_move != Color::White {
            return;
        }

        let Some(from) = selected.get_untracked() else {
            if current
                .piece_at(square)
                .is_some_and(|piece| piece.color == Color::White)
            {
                selected.set(Some(square));
            }
            return;
        };

        let chosen = current
            .get_legal_moves()
            .into_iter()
            .filter(|mv| mv.start_square == from && mv.end_square == square)
            .find(|mv| mv.promotion == Some(PieceKind::Queen))
            .or_else(|| {
                current
                    .get_legal_moves()
                    .into_iter()
                    .find(|mv| mv.start_square == from && mv.end_square == square)
            });

        let Some(mv) = chosen else {
            selected.set(
                current
                    .piece_at(square)
                    .and_then(|piece| (piece.color == Color::White).then_some(square)),
            );
            return;
        };

        let request_fen = current.to_fen();
        let mut after_player = current;
        after_player.make_move(mv);
        let player_san = after_player
            .san_history
            .last()
            .cloned()
            .expect("move recorded");

        board.set(after_player.clone());
        history.update(|moves| moves.push(player_san.clone()));
        selected.set(None);
        error.set(None);

        if after_player.status() != Status::Ongoing {
            return;
        }

        thinking.set(true);
        let request_id = game_id.get_untracked();
        spawn_local(async move {
            let result = request_bot(request_fen, player_san, after_player).await;
            if game_id.get_untracked() != request_id {
                return;
            }
            thinking.set(false);
            match result {
                Ok((next, bot_san)) => {
                    board.set(next);
                    history.update(|moves| moves.push(bot_san));
                }
                Err(message) => error.set(Some(message)),
            }
        });
    };

    view! {
        <main class="app-shell">
            <header>
                <div>
                    <p class="eyebrow">"CHESS ENGINES"</p>
                    <h1>"Play a bot"</h1>
                </div>
                <button class="reset" on:click=reset>"New game"</button>
            </header>

            <section class="game-layout">
                <div class="board-wrap" aria-label="Chess board">
                    <div class="board">
                        {(0..64).map(|index| {
                            let file = (index % 8) as u8;
                            let rank = 7 - (index / 8) as u8;
                            let square = Square::new(file, rank);
                            view! {
                                <button
                                    class=move || square_class(board.get(), selected.get(), square)
                                    on:click=move |_| play_square(square)
                                    aria-label=move || square_name(square)
                                >
                                    {move || board.get().piece_at(square).map(|piece| view! {
                                        <img src=piece_src(piece) alt=piece_name(piece) draggable="false" />
                                    })}
                                    {(file == 0).then(|| view! { <span class="rank-label">{rank + 1}</span> })}
                                    {(rank == 0).then(|| view! { <span class="file-label">{(b'a' + file) as char}</span> })}
                                </button>
                            }
                        }).collect_view()}
                    </div>
                </div>

                <aside>
                    <div class="panel bot-panel">
                        <label for="bot">"Opponent"</label>
                        <select id="bot">
                            <option value="random">"Random"</option>
                        </select>
                        <p class="bot-note">"Chooses uniformly from every legal move."</p>
                    </div>

                    <div class="panel status-panel">
                        <span class=move || if thinking.get() { "status-dot thinking" } else { "status-dot" }></span>
                        <div>
                            <small>"STATUS"</small>
                            <strong>{move || status_text(&board.get(), thinking.get())}</strong>
                        </div>
                    </div>

                    {move || error.get().map(|message| view! {
                        <div class="error">{message}</div>
                    })}

                    <div class="panel moves-panel">
                        <div class="moves-heading">
                            <span>"Moves"</span>
                            <small>{move || history.get().len()}</small>
                        </div>
                        <div class="moves-list">
                            {move || move_pairs(&history.get()).into_iter().map(|(number, white, black)| view! {
                                <div class="move-row">
                                    <span>{number}</span>
                                    <b>{white}</b>
                                    <b>{black}</b>
                                </div>
                            }).collect_view()}
                            {move || history.get().is_empty().then(|| view! {
                                <p class="empty-moves">"Your moves will appear here."</p>
                            })}
                        </div>
                    </div>
                </aside>
            </section>
        </main>
    }
}

async fn request_bot(
    fen: String,
    san: String,
    mut after_player: Board,
) -> Result<(Board, String), String> {
    let response = Request::post(BOT_URL)
        .json(&BotRequest {
            fen,
            san: Some(san),
        })
        .map_err(|error| error.to_string())?
        .send()
        .await
        .map_err(|_| {
            "Could not reach the random bot. Is `cargo run -p random` running?".to_string()
        })?;

    if !response.ok() {
        return Err(response
            .text()
            .await
            .unwrap_or_else(|_| "Bot rejected the move".into()));
    }

    let reply: BotResponse = response.json().await.map_err(|error| error.to_string())?;
    after_player.san_to_move(&reply.san)?;
    if after_player.to_fen() != reply.fen {
        return Err("Bot returned a mismatched position".to_string());
    }
    Ok((after_player, reply.san))
}

fn square_class(board: Board, selected: Option<Square>, square: Square) -> String {
    let mut classes = vec!["square"];
    if (square.file() + square.rank()).is_multiple_of(2) {
        classes.push("dark");
    } else {
        classes.push("light");
    }
    if selected == Some(square) {
        classes.push("selected");
    } else if selected.is_some_and(|from| {
        board
            .get_legal_moves()
            .iter()
            .any(|mv| mv.start_square == from && mv.end_square == square)
    }) {
        classes.push(if board.piece_at(square).is_some() {
            "capture"
        } else {
            "legal"
        });
    }
    classes.join(" ")
}

fn piece_src(piece: Piece) -> &'static str {
    match (piece.color, piece.kind) {
        (Color::White, PieceKind::Pawn) => "/public/white-pawn.png",
        (Color::White, PieceKind::Knight) => "/public/white-knight.png",
        (Color::White, PieceKind::Bishop) => "/public/white-bishop.png",
        (Color::White, PieceKind::Rook) => "/public/white-rook.png",
        (Color::White, PieceKind::Queen) => "/public/white-queen.png",
        (Color::White, PieceKind::King) => "/public/white-king.png",
        (Color::Black, PieceKind::Pawn) => "/public/black-pawn.png",
        (Color::Black, PieceKind::Knight) => "/public/black-knight.png",
        (Color::Black, PieceKind::Bishop) => "/public/black-bishop.png",
        (Color::Black, PieceKind::Rook) => "/public/black-rook.png",
        (Color::Black, PieceKind::Queen) => "/public/black-queen.png",
        (Color::Black, PieceKind::King) => "/public/black-king.png",
    }
}

fn piece_name(piece: Piece) -> &'static str {
    match (piece.color, piece.kind) {
        (Color::White, PieceKind::Pawn) => "White pawn",
        (Color::White, PieceKind::Knight) => "White knight",
        (Color::White, PieceKind::Bishop) => "White bishop",
        (Color::White, PieceKind::Rook) => "White rook",
        (Color::White, PieceKind::Queen) => "White queen",
        (Color::White, PieceKind::King) => "White king",
        (Color::Black, PieceKind::Pawn) => "Black pawn",
        (Color::Black, PieceKind::Knight) => "Black knight",
        (Color::Black, PieceKind::Bishop) => "Black bishop",
        (Color::Black, PieceKind::Rook) => "Black rook",
        (Color::Black, PieceKind::Queen) => "Black queen",
        (Color::Black, PieceKind::King) => "Black king",
    }
}

fn square_name(square: Square) -> String {
    format!("{}{}", (b'a' + square.file()) as char, square.rank() + 1)
}

fn status_text(board: &Board, thinking: bool) -> &'static str {
    if thinking {
        "Random is thinking…"
    } else {
        match board.status() {
            Status::Checkmate if board.side_to_move == Color::White => "Checkmate — Random wins",
            Status::Checkmate => "Checkmate — You win",
            Status::Stalemate => "Draw by stalemate",
            Status::Ongoing if board.is_in_check() => "Your king is in check",
            Status::Ongoing => "Your move",
        }
    }
}

fn move_pairs(moves: &[String]) -> Vec<(usize, String, String)> {
    moves
        .chunks(2)
        .enumerate()
        .map(|(index, pair)| {
            (
                index + 1,
                pair[0].clone(),
                pair.get(1).cloned().unwrap_or_default(),
            )
        })
        .collect()
}

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(App);
}
