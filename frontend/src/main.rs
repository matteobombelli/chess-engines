use chess_core::{Board, Color, Move, Piece, PieceKind, Square, Status};
use gloo_net::http::Request;
use leptos::mount::mount_to_body;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const BOT_URL: &str = "/projects/chessengines/api/move";

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
    let dragged = RwSignal::new(None::<Square>);
    let history = RwSignal::new(Vec::<String>::new());
    let thinking = RwSignal::new(false);
    let error = RwSignal::new(None::<String>);
    let game_id = RwSignal::new(0_u32);
    let pending_promotion = RwSignal::new(None::<(Square, Square)>);

    let reset = move |_| {
        board.set(Board::from_fen(START_FEN).expect("valid start position"));
        selected.set(None);
        dragged.set(None);
        history.set(Vec::new());
        thinking.set(false);
        error.set(None);
        pending_promotion.set(None);
        game_id.update(|id| *id += 1);
    };

    let play_move = move |current: Board, mv| {
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
        pending_promotion.set(None);
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

    let play_square = move |square: Square| {
        if pending_promotion.get_untracked().is_some()
            || thinking.get_untracked()
            || board.get_untracked().status() != Status::Ongoing
        {
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

        let candidates = moves_between(&current, from, square);

        if candidates.len() > 1 && candidates.iter().all(|mv| mv.promotion.is_some()) {
            pending_promotion.set(Some((from, square)));
            dragged.set(None);
            return;
        }

        let Some(mv) = candidates.into_iter().next() else {
            selected.set(
                current
                    .piece_at(square)
                    .and_then(|piece| (piece.color == Color::White).then_some(square)),
            );
            return;
        };

        play_move(current, mv);
    };

    let choose_promotion = move |kind: PieceKind| {
        let Some((from, to)) = pending_promotion.get_untracked() else {
            return;
        };
        let current = board.get_untracked();
        if let Some(mv) = moves_between(&current, from, to)
            .into_iter()
            .find(|mv| mv.promotion == Some(kind))
        {
            play_move(current, mv);
        } else {
            pending_promotion.set(None);
            selected.set(None);
            error.set(Some("That promotion is no longer legal".to_string()));
        }
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
                                    on:dragover=move |event| {
                                        if dragged.get_untracked().is_some() {
                                            event.prevent_default();
                                            if let Some(transfer) = event.data_transfer() {
                                                transfer.set_drop_effect("move");
                                            }
                                        }
                                    }
                                    on:drop=move |event| {
                                        event.prevent_default();
                                        if dragged.get_untracked().is_some() {
                                            play_square(square);
                                        }
                                        dragged.set(None);
                                    }
                                    aria-label=move || square_name(square)
                                >
                                    {move || board.get().piece_at(square).map(|piece| view! {
                                        <img
                                            src=piece_src(piece)
                                            alt=piece_name(piece)
                                            draggable=if piece.color == Color::White { "true" } else { "false" }
                                            on:dragstart=move |event| {
                                                let current = board.get_untracked();
                                                let can_drag = !thinking.get_untracked()
                                                    && current.status() == Status::Ongoing
                                                    && current.side_to_move == Color::White
                                                    && current.piece_at(square)
                                                        .is_some_and(|piece| piece.color == Color::White);

                                                if !can_drag {
                                                    event.prevent_default();
                                                    return;
                                                }

                                                selected.set(Some(square));
                                                dragged.set(Some(square));
                                                if let Some(transfer) = event.data_transfer() {
                                                    transfer.set_effect_allowed("move");
                                                    let _ = transfer.set_data("text/plain", &square_name(square));
                                                }
                                            }
                                            on:dragend=move |_| {
                                                dragged.set(None);
                                                selected.set(None);
                                            }
                                        />
                                    })}
                                    {(file == 0).then(|| view! { <span class="rank-label">{rank + 1}</span> })}
                                    {(rank == 0).then(|| view! { <span class="file-label">{(b'a' + file) as char}</span> })}
                                </button>
                            }
                        }).collect_view()}
                    </div>
                    {move || pending_promotion.get().map(|(from, _)| {
                        let color = board.get().piece_at(from)
                            .map(|piece| piece.color)
                            .unwrap_or(Color::White);
                        view! {
                            <div
                                class="promotion-overlay"
                                role="dialog"
                                aria-modal="true"
                                aria-label="Choose promotion piece"
                            >
                                <div class="promotion-dialog">
                                    <strong>"Promote pawn to"</strong>
                                    <div class="promotion-options">
                                        {[PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop, PieceKind::Knight]
                                            .into_iter()
                                            .map(|kind| {
                                                let piece = Piece { color, kind };
                                                view! {
                                                    <button
                                                        class="promotion-option"
                                                        aria-label=format!("Promote to {}", piece_kind_name(kind))
                                                        on:click=move |_| choose_promotion(kind)
                                                    >
                                                        <img src=piece_src(piece) alt="" />
                                                        <span>{piece_kind_name(kind)}</span>
                                                    </button>
                                                }
                                            })
                                            .collect_view()}
                                    </div>
                                    <button
                                        class="promotion-cancel"
                                        on:click=move |_| {
                                            pending_promotion.set(None);
                                            selected.set(None);
                                        }
                                    >
                                        "Cancel"
                                    </button>
                                </div>
                            </div>
                        }
                    })}
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

fn moves_between(board: &Board, from: Square, to: Square) -> Vec<Move> {
    board
        .get_legal_moves()
        .into_iter()
        .filter(|mv| mv.start_square == from && mv.end_square == to)
        .collect()
}

fn piece_src(piece: Piece) -> &'static str {
    match (piece.color, piece.kind) {
        (Color::White, PieceKind::Pawn) => "/projects/chessengines/public/white-pawn.png",
        (Color::White, PieceKind::Knight) => "/projects/chessengines/public/white-knight.png",
        (Color::White, PieceKind::Bishop) => "/projects/chessengines/public/white-bishop.png",
        (Color::White, PieceKind::Rook) => "/projects/chessengines/public/white-rook.png",
        (Color::White, PieceKind::Queen) => "/projects/chessengines/public/white-queen.png",
        (Color::White, PieceKind::King) => "/projects/chessengines/public/white-king.png",
        (Color::Black, PieceKind::Pawn) => "/projects/chessengines/public/black-pawn.png",
        (Color::Black, PieceKind::Knight) => "/projects/chessengines/public/black-knight.png",
        (Color::Black, PieceKind::Bishop) => "/projects/chessengines/public/black-bishop.png",
        (Color::Black, PieceKind::Rook) => "/projects/chessengines/public/black-rook.png",
        (Color::Black, PieceKind::Queen) => "/projects/chessengines/public/black-queen.png",
        (Color::Black, PieceKind::King) => "/projects/chessengines/public/black-king.png",
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

fn piece_kind_name(kind: PieceKind) -> &'static str {
    match kind {
        PieceKind::Pawn => "pawn",
        PieceKind::Knight => "knight",
        PieceKind::Bishop => "bishop",
        PieceKind::Rook => "rook",
        PieceKind::Queen => "queen",
        PieceKind::King => "king",
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
            Status::Checkmate if board.side_to_move == Color::White => "Checkmate - Random wins",
            Status::Checkmate => "Checkmate - You win",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_destination_keeps_all_four_choices() {
        let board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let moves = moves_between(&board, Square::new(0, 6), Square::new(0, 7));
        let promotions: Vec<_> = moves.into_iter().filter_map(|mv| mv.promotion).collect();

        assert_eq!(
            promotions,
            vec![
                PieceKind::Queen,
                PieceKind::Rook,
                PieceKind::Bishop,
                PieceKind::Knight,
            ]
        );
    }
}
