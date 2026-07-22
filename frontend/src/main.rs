use chess_core::{Board, Color, Move, Piece, PieceKind, Square, Status};
use gloo_net::http::Request;
use leptos::mount::mount_to_body;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const BOT_URL: &str = "/projects/chessengines/api/move";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Model {
    Random,
}

impl Model {
    fn from_value(value: &str) -> Self {
        match value {
            "random" => Self::Random,
            _ => Self::Random,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Random => "Random",
        }
    }

    fn note(self) -> &'static str {
        match self {
            Self::Random => "Chooses uniformly from every legal move.",
        }
    }
}

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

#[derive(Deserialize)]
struct BotErrorResponse {
    error: String,
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
    let player_color = RwSignal::new(Color::White);
    let selected_model = RwSignal::new(Model::Random);

    let reset = move |_| {
        start_game(
            player_color.get_untracked(),
            board,
            selected,
            dragged,
            history,
            thinking,
            error,
            game_id,
            pending_promotion,
        );
    };

    let switch_sides = move |_| {
        if side_switch_locked(history.get_untracked().len()) {
            return;
        }

        let color = match player_color.get_untracked() {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };
        player_color.set(color);
        start_game(
            color,
            board,
            selected,
            dragged,
            history,
            thinking,
            error,
            game_id,
            pending_promotion,
        );
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

        start_bot_turn(
            board,
            history,
            thinking,
            error,
            game_id,
            request_fen,
            Some(player_san),
            after_player,
        );
    };

    let play_square = move |square: Square| {
        if pending_promotion.get_untracked().is_some()
            || thinking.get_untracked()
            || board.get_untracked().status() != Status::Ongoing
        {
            return;
        }

        let current = board.get_untracked();
        let color = player_color.get_untracked();
        if current.side_to_move != color {
            return;
        }

        let Some(from) = selected.get_untracked() else {
            if current
                .piece_at(square)
                .is_some_and(|piece| piece.color == color)
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
                    .and_then(|piece| (piece.color == color).then_some(square)),
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
                <div class="header-actions">
                    <button
                        class="switch-side"
                        disabled=move || side_switch_locked(history.get().len())
                        on:click=switch_sides
                    >
                        "Switch sides"
                    </button>
                    <button class="reset" on:click=reset>"New game"</button>
                </div>
            </header>

            <section class="game-layout">
                <div class="board-wrap" aria-label="Chess board">
                    <div class="board">
                        {move || {
                            (0..64).map(|index| {
                                let column = (index % 8) as u8;
                                let row = (index / 8) as u8;
                                let square = oriented_square(index, player_color.get());
                                let file = square.file();
                                let rank = square.rank();
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
                                            draggable=if piece.color == player_color.get() { "true" } else { "false" }
                                            on:dragstart=move |event| {
                                                let current = board.get_untracked();
                                                let color = player_color.get_untracked();
                                                let can_drag = !thinking.get_untracked()
                                                    && current.status() == Status::Ongoing
                                                    && current.side_to_move == color
                                                    && current.piece_at(square)
                                                        .is_some_and(|piece| piece.color == color);

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
                                    {(column == 0).then(|| view! { <span class="rank-label">{rank + 1}</span> })}
                                    {(row == 7).then(|| view! { <span class="file-label">{(b'a' + file) as char}</span> })}
                                </button>
                            }
                            }).collect_view()
                        }}
                    </div>
                    {move || pending_promotion.get().map(|(from, _)| {
                        let color = board.get().piece_at(from)
                            .map(|piece| piece.color)
                            .unwrap_or_else(|| player_color.get());
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
                        <select
                            id="bot"
                            on:change=move |event| {
                                selected_model.set(Model::from_value(&event_target_value(&event)));
                            }
                        >
                            <option value="random">"Random"</option>
                        </select>
                        <p class="bot-note">{move || selected_model.get().note()}</p>
                        <a class="about-link" href="#about-model">
                            "About "
                            {move || selected_model.get().name()}
                            <span aria-hidden="true">" ↓"</span>
                        </a>
                        <p class="player-side">
                            "You're playing "
                            <strong>{move || color_name(player_color.get())}</strong>
                        </p>
                    </div>

                    <div class="panel status-panel">
                        <span class=move || if thinking.get() { "status-dot thinking" } else { "status-dot" }></span>
                        <div>
                            <small>"STATUS"</small>
                            <strong>{move || status_text(&board.get(), thinking.get(), player_color.get())}</strong>
                        </div>
                    </div>

                    {move || error.get().map(|message| view! {
                        <div class="error">{message}</div>
                    })}

                    <div class="panel moves-panel">
                        <div class="moves-heading">
                            <span>"Moves"</span>
                            <small>{move || move_count(history.get().len())}</small>
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
                                <p class="empty-moves">"Moves will appear here."</p>
                            })}
                        </div>
                    </div>
                </aside>
            </section>

            {move || match selected_model.get() {
                Model::Random => view! {
                    <section class="about-model" id="about-model" aria-labelledby="about-model-title">
                        <div class="about-heading">
                            <div>
                                <p class="eyebrow">"ABOUT THE MODEL"</p>
                                <h2 id="about-model-title">"About Random"</h2>
                            </div>
                            <p class="about-intro">
                                "Random knows the rules of chess but has no strategy. Each legal move has an equal chance of being selected."
                            </p>
                        </div>

                        <div class="about-steps">
                            <article>
                                <span class="step-number">"01"</span>
                                <h3>"Read the position"</h3>
                                <p>
                                    "The bot receives the current position in Forsyth Edwards Notation (FEN). After you move, it receives that move in Standard Algebraic Notation (SAN)."
                                </p>
                            </article>
                            <article>
                                <span class="step-number">"02"</span>
                                <h3>"Find every legal move"</h3>
                                <p>
                                    "It generates all moves allowed in that position, including castling, en passant, and each promotion choice. Moves that leave its king in check are excluded."
                                </p>
                            </article>
                            <article>
                                <span class="step-number">"03"</span>
                                <h3>"Pick one at random"</h3>
                                <p>
                                    "One candidate is selected uniformly, then returned as SAN together with the resulting FEN. The same position can produce a different reply each time."
                                </p>
                            </article>
                        </div>

                        <div class="about-summary">
                            <strong>"Limits"</strong>
                            <p>
                                "There is no search, position evaluation, training, or game memory. Checkmates and blunders are both accidental."
                            </p>
                        </div>
                    </section>
                },
            }}
        </main>
    }
}

#[allow(clippy::too_many_arguments)]
fn start_game(
    player_color: Color,
    board: RwSignal<Board>,
    selected: RwSignal<Option<Square>>,
    dragged: RwSignal<Option<Square>>,
    history: RwSignal<Vec<String>>,
    thinking: RwSignal<bool>,
    error: RwSignal<Option<String>>,
    game_id: RwSignal<u32>,
    pending_promotion: RwSignal<Option<(Square, Square)>>,
) {
    let starting_board = Board::from_fen(START_FEN).expect("valid start position");
    board.set(starting_board.clone());
    selected.set(None);
    dragged.set(None);
    history.set(Vec::new());
    thinking.set(false);
    error.set(None);
    pending_promotion.set(None);
    game_id.update(|id| *id += 1);

    if player_color == Color::Black {
        start_bot_turn(
            board,
            history,
            thinking,
            error,
            game_id,
            START_FEN.to_string(),
            None,
            starting_board,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn start_bot_turn(
    board: RwSignal<Board>,
    history: RwSignal<Vec<String>>,
    thinking: RwSignal<bool>,
    error: RwSignal<Option<String>>,
    game_id: RwSignal<u32>,
    request_fen: String,
    player_san: Option<String>,
    before_bot: Board,
) {
    thinking.set(true);
    let request_id = game_id.get_untracked();
    spawn_local(async move {
        let result = request_bot(request_fen, player_san, before_bot).await;
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
}

async fn request_bot(
    fen: String,
    player_san: Option<String>,
    mut before_bot: Board,
) -> Result<(Board, String), String> {
    let payload = BotRequest {
        fen,
        san: player_san,
    };
    let mut gateway_retries = 1;

    let response = loop {
        let response = Request::post(BOT_URL)
            .json(&payload)
            .map_err(|error| error.to_string())?
            .send()
            .await
            .map_err(|_| "Could not reach the bot. Please try again.".to_string())?;

        if is_gateway_error(response.status()) && gateway_retries > 0 {
            gateway_retries -= 1;
            continue;
        }
        break response;
    };

    if !response.ok() {
        let status = response.status();
        if is_gateway_error(status) {
            return Err("The bot is temporarily unavailable. Please try again.".into());
        }

        return Err(match response.json::<BotErrorResponse>().await {
            Ok(body) => body.error,
            Err(_) => format!("The bot request failed (HTTP {status})."),
        });
    }

    let reply: BotResponse = response.json().await.map_err(|error| error.to_string())?;
    before_bot.san_to_move(&reply.san)?;
    if before_bot.to_fen() != reply.fen {
        return Err("Bot returned a mismatched position".to_string());
    }
    Ok((before_bot, reply.san))
}

fn is_gateway_error(status: u16) -> bool {
    matches!(status, 502..=504)
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

fn oriented_square(index: usize, player_color: Color) -> Square {
    let column = (index % 8) as u8;
    let row = (index / 8) as u8;
    match player_color {
        Color::White => Square::new(column, 7 - row),
        Color::Black => Square::new(7 - column, row),
    }
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

fn color_name(color: Color) -> &'static str {
    match color {
        Color::White => "White",
        Color::Black => "Black",
    }
}

fn status_text(board: &Board, thinking: bool, player_color: Color) -> &'static str {
    if thinking {
        "Bot is thinking..."
    } else {
        match board.status() {
            Status::Checkmate if board.side_to_move == player_color => "Checkmate - Bot wins",
            Status::Checkmate => "Checkmate - You win",
            Status::Stalemate => "Draw by stalemate",
            Status::ThreefoldRepetition => "Draw by threefold repetition",
            Status::FiftyMoveRule => "Draw by the 50-move rule",
            Status::Ongoing if board.is_in_check() => "Your king is in check",
            Status::Ongoing => "Your move",
        }
    }
}

fn move_count(half_moves: usize) -> String {
    if half_moves.is_multiple_of(2) {
        (half_moves / 2).to_string()
    } else {
        format!("{}.5", half_moves / 2)
    }
}

fn side_switch_locked(half_moves: usize) -> bool {
    half_moves >= 2
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

    #[test]
    fn gateway_failures_are_retryable() {
        assert!(is_gateway_error(502));
        assert!(is_gateway_error(503));
        assert!(is_gateway_error(504));
        assert!(!is_gateway_error(400));
        assert!(!is_gateway_error(500));
    }

    #[test]
    fn move_count_uses_half_moves() {
        assert_eq!(move_count(0), "0");
        assert_eq!(move_count(1), "0.5");
        assert_eq!(move_count(2), "1");
        assert_eq!(move_count(3), "1.5");
    }

    #[test]
    fn side_switch_locks_after_one_full_move() {
        assert!(!side_switch_locked(0));
        assert!(!side_switch_locked(1));
        assert!(side_switch_locked(2));
        assert!(side_switch_locked(3));
    }

    #[test]
    fn checkmate_winner_respects_the_players_side() {
        let board = Board::from_fen(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        )
        .unwrap();

        assert_eq!(
            status_text(&board, false, Color::White),
            "Checkmate - Bot wins"
        );
        assert_eq!(
            status_text(&board, false, Color::Black),
            "Checkmate - You win"
        );
    }

    #[test]
    fn board_orientation_puts_the_players_pieces_at_the_bottom() {
        assert_eq!(oriented_square(0, Color::White), Square::new(0, 7));
        assert_eq!(oriented_square(63, Color::White), Square::new(7, 0));
        assert_eq!(oriented_square(0, Color::Black), Square::new(7, 0));
        assert_eq!(oriented_square(63, Color::Black), Square::new(0, 7));
    }
}
