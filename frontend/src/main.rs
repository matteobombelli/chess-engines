use chess_core::{Board, Color, Move, Piece, PieceKind, Square, Status};
use gloo_net::http::Request;
use leptos::mount::mount_to_body;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};

mod minigpt_training;
mod training;
use minigpt_training::MiniGptTrainingProgress;
use training::TrainingProgress;

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Model {
    Random,
    MinimaxDepth3,
    MinimaxNineSeconds,
    AlphaMini,
    MiniGpt,
}

impl Model {
    fn name(self) -> &'static str {
        match self {
            Self::Random => "Random",
            Self::MinimaxDepth3 => "Depth-3 Minimax",
            Self::MinimaxNineSeconds => "9-second Minimax",
            Self::AlphaMini => "AlphaMini",
            Self::MiniGpt => "MiniGPT",
        }
    }

    fn note(self) -> &'static str {
        match self {
            Self::Random => "Chooses a legal move at random.",
            Self::MinimaxDepth3 => "Searches three moves ahead and usually responds quickly.",
            Self::MinimaxNineSeconds => "Searches for up to nine seconds before each move.",
            Self::AlphaMini => "Uses a compact neural network and Monte Carlo tree search.",
            Self::MiniGpt => "Predicts the next move from the whole game, without searching.",
        }
    }

    fn selector_label(self) -> &'static str {
        match self {
            Self::Random => "Random",
            Self::MinimaxDepth3 => "Minimax · depth 3",
            Self::MinimaxNineSeconds => "Minimax · 9 seconds",
            Self::AlphaMini => "AlphaMini",
            Self::MiniGpt => "MiniGPT",
        }
    }

    fn elo_label(self) -> &'static str {
        match self {
            Self::Random => "<400 Elo",
            Self::MinimaxDepth3 => "≈1640 Elo",
            Self::MinimaxNineSeconds => "≈2050 Elo",
            Self::AlphaMini => "≈1970 Elo",
            Self::MiniGpt => "≈1930 Elo",
        }
    }

    fn url(self) -> &'static str {
        match self {
            Self::Random => "/projects/chessengines/api/random/move",
            Self::MinimaxDepth3 => "/projects/chessengines/api/minimax/depth-3/move",
            Self::MinimaxNineSeconds => "/projects/chessengines/api/minimax/move",
            Self::AlphaMini => "/projects/chessengines/api/alphamini/move",
            Self::MiniGpt => "/projects/chessengines/api/minigpt/move",
        }
    }
}

#[derive(Serialize)]
/// The whole game as PGN movetext.
/// An empty string means the game has not started and the bot moves first.
struct BotRequest {
    san: String,
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
    let bot_selector = NodeRef::<leptos::html::Details>::new();

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
            selected_model.get_untracked(),
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
            selected_model.get_untracked(),
        );
    };

    let play_move = move |current: Board, mv| {
        let mut after_player = current;
        after_player.make_move(mv);
        // The game so far, taken from the board itself rather than the display
        // list, so what we send can never disagree with the position we are in.
        let movetext = after_player.export_san();
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
            movetext,
            after_player,
            selected_model.get_untracked(),
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
                        <p class="bot-label" id="opponent-label">"Opponent"</p>
                        <details class="bot-selector" node_ref=bot_selector>
                            <summary aria-labelledby="opponent-label">
                                <span>{move || selected_model.get().selector_label()}</span>
                                <span class="bot-option-elo">
                                    {move || selected_model.get().elo_label()}
                                </span>
                            </summary>
                            <div class="bot-options" aria-labelledby="opponent-label">
                                {[
                                    Model::Random,
                                    Model::MinimaxDepth3,
                                    Model::MinimaxNineSeconds,
                                    Model::AlphaMini,
                                    Model::MiniGpt,
                                ]
                                    .into_iter()
                                    .map(|model| view! {
                                        <button
                                            type="button"
                                            class=move || if selected_model.get() == model {
                                                "bot-option selected"
                                            } else {
                                                "bot-option"
                                            }
                                            aria-pressed=move || selected_model.get() == model
                                            on:click=move |_| {
                                                selected_model.set(model);
                                                if let Some(selector) = bot_selector.get() {
                                                    let _ = selector.remove_attribute("open");
                                                }
                                            }
                                        >
                                            <span>{model.selector_label()}</span>
                                            <span class="bot-option-elo">{model.elo_label()}</span>
                                        </button>
                                    })
                                    .collect_view()}
                            </div>
                        </details>
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
                                <p class="eyebrow">"ABOUT THE ENGINE"</p>
                                <h2 id="about-model-title">"Random"</h2>
                            </div>
                            <p class="about-intro">
                                "Random knows the rules of chess, but it has no strategy. It chooses from all legal moves with equal odds."
                            </p>
                        </div>

                        <div class="about-details">
                            <article>
                                <h3>"Implementation"</h3>
                                <p>
                                    "The browser sends the moves played so far. The API rebuilds the board, finds every legal move, and returns one at random."
                                </p>
                            </article>
                            <article>
                                <h3>"Rules"</h3>
                                <p>
                                    "The shared rules engine handles castling, en passant, promotion, check, checkmate, and draws. It leaves out any move that would expose its king."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "On historical 30+0 positions, its move quality fell below 400 Chess.com Elo. The 95% player-bootstrap CI is also below 400, with both endpoints censored by the calibration floor."
                                </p>
                            </article>
                        </div>
                    </section>
                }.into_any(),
                Model::MinimaxDepth3 => view! {
                    <section class="about-model" id="about-model" aria-labelledby="about-model-title">
                        <div class="about-heading">
                            <div>
                                <p class="eyebrow">"ABOUT THE ENGINE"</p>
                                <h2 id="about-model-title">"Depth-3 Minimax"</h2>
                            </div>
                            <p class="about-intro">
                                "Depth-3 Minimax looks three moves ahead and chooses the line with the best score. Since the search always stops at the same depth, it usually replies quickly."
                            </p>
                        </div>

                        <div class="about-details">
                            <article>
                                <h3>"Search"</h3>
                                <p>
                                    "It assumes both sides choose their strongest move. If the third move leads to more captures or leaves a king in check, the engine follows those replies before it scores the position."
                                </p>
                            </article>
                            <article>
                                <h3>"Evaluation"</h3>
                                <p>
                                    "The score weighs material, piece placement, pawn structure, open files, and king safety. It gives a small bonus to the side to move. Checkmate is decisive and draws are even."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "On historical 30+0 positions, its move quality was closest to players rated around 1640 Chess.com Elo. The 95% whole-player bootstrap interval is at or below 1400 to 1780; its lower endpoint is censored by the calibrated range. It did not play full games on Chess.com."
                                </p>
                            </article>
                        </div>
                    </section>
                }.into_any(),
                Model::MinimaxNineSeconds => view! {
                    <section class="about-model" id="about-model" aria-labelledby="about-model-title">
                        <div class="about-heading">
                            <div>
                                <p class="eyebrow">"ABOUT THE ENGINE"</p>
                                <h2 id="about-model-title">"9-second Minimax"</h2>
                            </div>
                            <p class="about-intro">
                                "The timed engine searches for up to nine seconds before each move. It starts shallow and keeps searching deeper until time runs out."
                            </p>
                        </div>

                        <div class="about-details">
                            <article>
                                <h3>"Search"</h3>
                                <p>
                                    "It saves the best move after every finished search, so the next, deeper search can use all the time left. Move ordering and alpha-beta pruning skip lines that cannot change its choice."
                                </p>
                            </article>
                            <article>
                                <h3>"Evaluation"</h3>
                                <p>
                                    "The score weighs material, piece placement, pawn structure, open files, and king safety. It gives a small bonus to the side to move. Checkmate is decisive and draws are even."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "On historical 30+0 positions, its move quality was closest to players rated around 2050 Chess.com Elo. The 95% player-bootstrap CI starts at 1889. Its upper endpoint is at or above 2199, the highest well-populated rating group."
                                </p>
                            </article>
                        </div>
                    </section>
                }.into_any(),
                Model::AlphaMini => view! {
                    <section class="about-model" id="about-model" aria-labelledby="about-model-title">
                        <div class="about-heading">
                            <div>
                                <p class="eyebrow">"ABOUT THE ENGINE"</p>
                                <h2 id="about-model-title">"AlphaMini"</h2>
                            </div>
                            <p class="about-intro">
                                "AlphaMini learns only from self-play. A compact convolutional network predicts promising moves and the likely result, while Monte Carlo tree search tests those predictions against the legal replies."
                            </p>
                        </div>

                        <div class="about-details">
                            <article>
                                <h3>"Search"</h3>
                                <p>
                                    "Rust generates every legal move and searches a tree for up to nine seconds. The neural network ranks positions; it never invents or directly applies a move."
                                </p>
                            </article>
                            <article>
                                <h3>"Training"</h3>
                                <p>
                                    "The deployed network trained for 72 hours of seeded, versioned self-play on one RTX 3070. Checkpoints, data shards, software versions, and interrupted-run recovery state are recorded so training can be audited and continued."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "On historical 30+0 positions, its move quality was closest to players rated around 1970 Chess.com Elo. The 95% player-bootstrap CI starts at 1758; its upper endpoint is at or above 1999. An independent sample of players rated 1700 and up agreed, crossing in the same band. It did not play full games on Chess.com."
                                </p>
                            </article>
                        </div>

                        <TrainingProgress/>
                    </section>
                }.into_any(),
                Model::MiniGpt => view! {
                    <section class="about-model" id="about-model" aria-labelledby="about-model-title">
                        <div class="about-heading">
                            <div>
                                <p class="eyebrow">"ABOUT THE ENGINE"</p>
                                <h2 id="about-model-title">"MiniGPT"</h2>
                            </div>
                            <p class="about-intro">
                                "MiniGPT is a 40-million-parameter GPT that predicts the next move from the whole game, trained on 11 million strong Lichess games. It does not search: it reads the moves played so far and answers with the move it expects to come next."
                            </p>
                        </div>

                        <div class="about-details">
                            <article>
                                <h3>"Prediction"</h3>
                                <p>
                                    "Every game is a sequence of move tokens. A decoder-only transformer reads that sequence and scores each possible next move in one forward pass, so a reply takes milliseconds rather than seconds."
                                </p>
                            </article>
                            <article>
                                <h3>"Rules"</h3>
                                <p>
                                    "Rust still generates the legal moves and keeps every illegal option out of the choice. The model only ranks moves the rules already allow; it never invents or applies one itself."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "Calibrated at about 1930 on the Chess.com 30+0 move-quality scale, using the same Stockfish reference method as the other engines — within the confidence interval of AlphaMini, from a single millisecond forward pass instead of a nine-second search."
                                </p>
                            </article>
                        </div>

                        <MiniGptTrainingProgress/>
                    </section>
                }.into_any(),
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
    model: Model,
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
            // Nothing has been played yet, so the bot opens.
            String::new(),
            starting_board,
            model,
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
    movetext: String,
    before_bot: Board,
    model: Model,
) {
    thinking.set(true);
    let request_id = game_id.get_untracked();
    spawn_local(async move {
        let result = request_bot(model, movetext, before_bot).await;
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
    model: Model,
    movetext: String,
    mut before_bot: Board,
) -> Result<(Board, String), String> {
    let payload = BotRequest { san: movetext };
    let mut gateway_retries = 1;

    let response = loop {
        let response = Request::post(model.url())
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
            Status::InsufficientMaterial => "Draw by insufficient material",
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
    fn minimax_options_use_separate_routes() {
        assert_eq!(
            Model::MinimaxDepth3.url(),
            "/projects/chessengines/api/minimax/depth-3/move"
        );
        assert_eq!(
            Model::MinimaxNineSeconds.url(),
            "/projects/chessengines/api/minimax/move"
        );
    }

    #[test]
    fn model_labels_include_elo() {
        assert_eq!(Model::Random.elo_label(), "<400 Elo");
        assert_eq!(Model::MinimaxDepth3.elo_label(), "≈1640 Elo");
        assert_eq!(Model::MinimaxNineSeconds.elo_label(), "≈2050 Elo");
        assert_eq!(Model::AlphaMini.elo_label(), "≈1970 Elo");
        assert_eq!(Model::MiniGpt.elo_label(), "≈1930 Elo");
        assert_eq!(
            Model::AlphaMini.url(),
            "/projects/chessengines/api/alphamini/move"
        );
    }

    #[test]
    fn minigpt_is_offered_with_its_calibrated_rating() {
        assert_eq!(Model::MiniGpt.elo_label(), "≈1930 Elo");
        assert_eq!(
            Model::MiniGpt.url(),
            "/projects/chessengines/api/minigpt/move"
        );
        assert_eq!(Model::MiniGpt.selector_label(), "MiniGPT");
    }

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
        let board =
            Board::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
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
