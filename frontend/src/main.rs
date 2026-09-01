use chess_core::{Board, Color, Move, Piece, PieceKind, Square, Status};
use gloo_net::http::Request;
use leptos::mount::mount_to_body;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};

mod explainers;
mod minigpt_training;
mod training;
use explainers::{
    AlphaMiniHowItWorks, MiniGptHowItWorks, MinimaxHowItWorks, RandomHowItWorks,
    TimedMinimaxHowItWorks,
};
use minigpt_training::MiniGptTrainingProgress;
use training::TrainingProgress;

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Where the in-progress game is kept between page loads. The version suffix
/// means a future format change can ignore old values instead of misreading
/// them.
const SAVE_KEY: &str = "chessengines.game.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
            Self::MinimaxDepth3 => "≈1700 Elo",
            Self::MinimaxNineSeconds => "≈2060 Elo",
            Self::AlphaMini => "≈2290 Elo",
            Self::MiniGpt => "≈1400 Elo",
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

    /// Stable name used in saved games. Changing one of these strings orphans
    /// every game already in a visitor's browser, so they are kept apart from
    /// the display labels.
    fn id(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::MinimaxDepth3 => "minimax-depth-3",
            Self::MinimaxNineSeconds => "minimax-nine-seconds",
            Self::AlphaMini => "alphamini",
            Self::MiniGpt => "minigpt",
        }
    }

    fn from_id(id: &str) -> Option<Self> {
        match id {
            "random" => Some(Self::Random),
            "minimax-depth-3" => Some(Self::MinimaxDepth3),
            "minimax-nine-seconds" => Some(Self::MinimaxNineSeconds),
            "alphamini" => Some(Self::AlphaMini),
            "minigpt" => Some(Self::MiniGpt),
            _ => None,
        }
    }
}

fn color_id(color: Color) -> &'static str {
    match color {
        Color::White => "white",
        Color::Black => "black",
    }
}

fn color_from_id(id: &str) -> Option<Color> {
    match id {
        "white" => Some(Color::White),
        "black" => Some(Color::Black),
        _ => None,
    }
}

/// The saved game exactly as it is written to localStorage.
#[derive(Serialize, Deserialize)]
struct SavedGame {
    model: String,
    color: String,
    san: Vec<String>,
}

/// A saved game that has been checked and replayed.
struct RestoredGame {
    model: Model,
    color: Color,
    san: Vec<String>,
    board: Board,
}

/// Replay a list of SAN moves from the starting position.
///
/// `None` covers every way the list can be wrong: a token that is not a move,
/// a move that is not legal in the position it reaches, or a move played after
/// the game has already ended.
fn replay_san(moves: &[String]) -> Option<Board> {
    let mut board = Board::from_fen(START_FEN).ok()?;
    for san in moves {
        if board.status() != Status::Ongoing {
            return None;
        }
        board.san_to_move(san).ok()?;
    }
    Some(board)
}

/// Turn a stored value into a game. Pure, so the discard rules are testable
/// without a browser.
fn parse_saved_game(raw: &str) -> Option<RestoredGame> {
    let saved: SavedGame = serde_json::from_str(raw).ok()?;
    let model = Model::from_id(&saved.model)?;
    let color = color_from_id(&saved.color)?;
    let board = replay_san(&saved.san)?;
    Some(RestoredGame {
        model,
        color,
        san: saved.san,
        board,
    })
}

/// localStorage, or `None` where it is unavailable. Reading the property
/// throws in a private window, so every caller treats storage as optional.
fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

fn save_game(model: Model, color: Color, moves: &[String]) {
    let Some(storage) = local_storage() else {
        return;
    };
    let saved = SavedGame {
        model: model.id().to_string(),
        color: color_id(color).to_string(),
        san: moves.to_vec(),
    };
    if let Ok(encoded) = serde_json::to_string(&saved) {
        let _ = storage.set_item(SAVE_KEY, &encoded);
    }
}

/// Read the saved game, dropping the key when it cannot be replayed.
fn load_game() -> Option<RestoredGame> {
    let storage = local_storage()?;
    let raw = storage.get_item(SAVE_KEY).ok().flatten()?;
    let restored = parse_saved_game(&raw);
    if restored.is_none() {
        let _ = storage.remove_item(SAVE_KEY);
    }
    restored
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
    // Seed the signals from the saved game before the first render, so a
    // refresh paints the position it left off at instead of the start position.
    let restored = load_game();
    let (start_board, start_history, start_model, start_color) = match &restored {
        Some(game) => (game.board.clone(), game.san.clone(), game.model, game.color),
        None => (
            Board::from_fen(START_FEN).expect("valid start position"),
            Vec::new(),
            Model::Random,
            Color::White,
        ),
    };

    let board = RwSignal::new(start_board);
    let selected = RwSignal::new(None::<Square>);
    let dragged = RwSignal::new(None::<Square>);
    let history = RwSignal::new(start_history);
    let thinking = RwSignal::new(false);
    let error = RwSignal::new(None::<String>);
    let game_id = RwSignal::new(0_u32);
    let pending_promotion = RwSignal::new(None::<(Square, Square)>);
    let player_color = RwSignal::new(start_color);
    let selected_model = RwSignal::new(start_model);
    let bot_selector = NodeRef::<leptos::html::Details>::new();

    // A refresh can land in the middle of the bot's turn, which loses the
    // in-flight request. Ask again from the restored position. `start_game` is
    // deliberately not used here: it would reset the board it just restored.
    if let Some(game) = restored
        && game.board.status() == Status::Ongoing
        && game.board.side_to_move != game.color
    {
        let movetext = game.board.export_san();
        start_bot_turn(
            board, history, thinking, error, game_id, movetext, game.board, game.model,
        );
    }

    Effect::new(move |_| {
        save_game(selected_model.get(), player_color.get(), &history.get());
    });

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

    let locked = move || game_locked(history.get().len());

    // The dropdown can be open when the bot's reply lands and locks the game.
    // Shut it so the panel cannot be left showing options that do nothing.
    Effect::new(move |_| {
        if locked()
            && let Some(selector) = bot_selector.get()
        {
            let _ = selector.remove_attribute("open");
        }
    });

    let switch_sides = move |_| {
        if game_locked(history.get_untracked().len()) {
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

    let choose_model = move |model: Model| {
        if let Some(selector) = bot_selector.get_untracked() {
            let _ = selector.remove_attribute("open");
        }
        if game_locked(history.get_untracked().len()) || selected_model.get_untracked() == model {
            return;
        }

        selected_model.set(model);
        // Restart, the same way switching sides does, so an opening move made
        // by the previous bot never carries into the new one's game.
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
            model,
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
                        disabled=locked
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
                        <details
                            class=move || if locked() { "bot-selector locked" } else { "bot-selector" }
                            node_ref=bot_selector
                        >
                            <summary
                                aria-labelledby="opponent-label"
                                aria-disabled=move || if locked() { "true" } else { "false" }
                                on:click=move |event| {
                                    // Keep the dropdown shut while the game runs.
                                    // Enter and Space on the summary arrive here too.
                                    if locked() {
                                        event.prevent_default();
                                    }
                                }
                            >
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
                                            disabled=locked
                                            aria-pressed=move || selected_model.get() == model
                                            on:click=move |_| choose_model(model)
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
                                "Random knows the rules of chess and nothing else. It asks for the legal moves and picks one with equal odds. Nothing in it can tell a good move from a bad one. I wrote it first, as the floor the other four engines are measured against."
                            </p>
                        </div>

                        <RandomHowItWorks/>

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
                                    "All five bots share one rules engine written in Rust. It handles castling, en passant, promotion, check, checkmate, and draws, and it never offers a move that would leave its own king in check."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "Random went winless in 220 full games against Stockfish, Depth-3, and MiniGPT. Its only half points were seven stalemates MiniGPT stumbled into from winning positions. Every measurement is consistent with a rating below 400 on the approximate Chess.com scale used for the other four engines."
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
                                "Depth-3 Minimax looks three moves ahead and plays the line with the best score. It assumes you answer with your strongest reply at every step, scores the positions it lands in with seven hand-written chess rules, and carries those scores back up the tree. The depth is fixed, so its reply time barely changes from one move to the next. This is the first engine here that plans at all."
                            </p>
                        </div>

                        <MinimaxHowItWorks/>

                        <div class="about-details">
                            <article>
                                <h3>"Search"</h3>
                                <p>
                                    "One function handles both sides. It scores a position for whoever is to move, then negates the score coming back up, so maximizing at every level is the same as assuming you always take your best reply. Alpha-beta pruning drops a branch as soon as one of your replies is good enough that the bot would never send the game that way. The skipped work cannot change the move it picks, so the pruning is free depth."
                                </p>
                            </article>
                            <article>
                                <h3>"Quiescence"</h3>
                                <p>
                                    "Stopping the count in the middle of a trade would let the bot record a queen it just took and never see the recapture. So at the third move it keeps going, playing out captures and promotions until nothing tactical is left. When a king is in check it searches every legal move, because every one of them is forced."
                                </p>
                            </article>
                            <article>
                                <h3>"Evaluation"</h3>
                                <p>
                                    "The leaf score is in hundredths of a pawn. Material runs 100 for a pawn, 320 for a knight, 335 for a bishop, 500 for a rook, and 900 for a queen. On top of that: a placement term for central squares and advanced pawns, a king that wants to hide in the middlegame and walk forward in the ending, penalties for doubled and isolated pawns, bonuses for connected and passed pawns, 30 for the bishop pair, 10 for a rook on a file with no friendly pawn and 12 more if the file is fully open, 8 for each pawn shielding the king, and 10 for having the move. Checkmate scores 30,000 minus how deep the search had to go to reach it, so a mate found sooner outranks the same mate found later. Every draw scores zero."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "Eighty full games against Stockfish 17.1 at fixed strength levels fit a rating of 1698, with a 95% interval of 1627 to 1766. That scale is Stockfish's human-anchored Elo, close to a Chess.com rating, and engines cannot play rated games on Chess.com, so every number here is an estimate on it."
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
                                "This engine gets nine seconds per move. It searches to depth 1, starts over at depth 2, and keeps restarting one level deeper until the time is gone. It runs the same code and the same evaluation as Depth-3. The only difference is what stops the search: the depth ceiling is set to 64, so in practice the clock decides."
                            </p>
                        </div>

                        <TimedMinimaxHowItWorks/>

                        <div class="about-details">
                            <article>
                                <h3>"Search"</h3>
                                <p>
                                    "Every depth that finishes leaves a best move behind, so there is always an answer ready when the clock stops. Reading the clock costs time of its own, so the deadline is checked once every 1,024 positions. Whatever the depth in progress had worked out is thrown away, because it has only looked at some of the moves and would rank them unfairly."
                                </p>
                            </article>
                            <article>
                                <h3>"Move ordering"</h3>
                                <p>
                                    "Restarting from depth 1 sounds wasteful, and it pays for itself. The best move from the last depth is tried first at the next one, then captures ranked by what they win against what they risk, then promotions, then castling. Alpha-beta prunes hardest when the strongest move comes first, so the cheap shallow searches are what buy the deep one."
                                </p>
                            </article>
                            <article>
                                <h3>"Evaluation"</h3>
                                <p>
                                    "Identical to Depth-3: material, piece placement, pawn structure, rook files, king safety, and 10 hundredths of a pawn for having the move. Checkmate scores 30,000 minus how deep the search had to go to reach it. Every draw scores zero."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "Sixty full games against Stockfish 17.1 at fixed strength levels fit a rating of 2057, with a 95% interval of 1961 to 2142. The games ran through the live site, so the number reflects the server this engine really runs on. Same approximate Chess.com scale as the other engines."
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
                                "AlphaMini learns from self-play alone. It has never seen a human game. The board reaches it as 22 stacked grids of numbers, and a small convolutional network turns those into two answers: a score for every move it could play, and its odds of winning, drawing, and losing. Monte Carlo tree search spends nine seconds testing those answers against real replies. I trained it for 72 hours on one RTX 3070."
                            </p>
                        </div>

                        <AlphaMiniHowItWorks/>

                        <div class="about-details">
                            <article>
                                <h3>"What it sees"</h3>
                                <p>
                                    "The position becomes 22 planes of 8 by 8 numbers. The first six mark where its own pawns, knights, bishops, rooks, queens, and king stand, and the next six do the same for yours. Two say whether this exact position has already happened once or twice, which is how it knows a repetition is coming. Four hold castling rights and one marks the en passant square. One says which colour is to move, one carries the halfmove clock as the count divided by 100, so it can feel the fifty-move rule approaching, and the last is all ones, a constant the convolutions use to find the edge of the board. When AlphaMini plays Black the board is flipped so the side to move always looks up the board, and no move history is in there at all."
                                </p>
                            </article>
                            <article>
                                <h3>"The network"</h3>
                                <p>
                                    "A 64-channel stem reads the planes and six 64-channel residual blocks refine them. Each block ends with squeeze and excitation, which weighs the whole board at once and turns channels up or down, so a threat on one wing can quiet a plan on the other. Two heads read the result. The policy head returns 4,672 numbers, one for every move the action space can name, which is 73 ways a piece can travel from each of 64 starting squares. The value head returns three, the chances of a win, a draw, and a loss. Search collapses those into one number by subtracting the loss chance from the win chance."
                                </p>
                            </article>
                            <article>
                                <h3>"Search"</h3>
                                <p>
                                    "The network alone is a hunch. Search is what checks it. One simulation walks down the tree picking the move with the best mix of what the tree has already scored and what the policy head expected, using an exploration constant of 1.5 to weigh the second against the first. At a position it has not seen, Rust lists the legal moves and the network scores them, and the value comes back up the path, updating every move along it. AlphaMini runs 10,000 of those or nine seconds, whichever ends first, eight positions to a batch on the GPU. The move it visited most often is the move it plays, which is a sturdier signal than the one it scored highest, because a move only collects visits by surviving repeated looks."
                                </p>
                            </article>
                            <article>
                                <h3>"Training"</h3>
                                <p>
                                    "One cycle plays 1,024 games against itself at 128 simulations a move, with random noise mixed into the root priors so it keeps trying openings it has already dismissed, then trains the network on the visit counts and the results those games produced. The next cycle plays with the network the last one left behind. That loop ran across three chained runs for 72.09 active hours. Every cycle is seeded and its checkpoint is recorded with the exact code and data behind it, so an interrupted run resumes where it stopped and any cycle can be checked later."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "160 full games against Stockfish 17.1 at fixed strength levels fit a rating of 2289, with a 95% interval of 2229 to 2353. The games ran through the live site, so the number reflects the production server. An older estimate that judged single moves put it at 1970; playing whole games moved it up, because a searching engine converts the small advantages its judgment finds. Same approximate Chess.com scale as the other engines."
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
                                "MiniGPT has 40.3 million parameters and does no search at all. It is the same kind of model that predicts the next word in a sentence, pointed at chess: every move is one token, and the model reads the tokens played so far and answers with the move it expects to come next. It has never been shown a board. All it knows is which moves tend to follow which. I trained it on 11.2 million Lichess games where both players were rated at least 2000."
                            </p>
                        </div>

                        <MiniGptHowItWorks/>

                        <div class="about-details">
                            <article>
                                <h3>"Prediction"</h3>
                                <p>
                                    "The game so far becomes a start token followed by one token per move, drawn from a vocabulary of 4,736 entries. Twelve transformer layers read that sequence, each carrying 512 numbers per token through 8 attention heads. Attention is the part that matters here: it lets every move in the sequence look back at all the moves before it, so the token for the last move arrives at the top carrying the shape of the whole game. Reading off that final token gives one score for every entry in the vocabulary, in a single forward pass that takes about nine milliseconds on CPU. Only the last 256 tokens fit, and a longer game loses its own opening from view."
                                </p>
                            </article>
                            <article>
                                <h3>"Rules"</h3>
                                <p>
                                    "The model is free to score a move that would leave its king in check, or one the pieces cannot even make. So the Rust rules engine generates the legal moves first and throws away the score of every token that is not one of them. What survives is turned into probabilities at temperature 0.5, which widens the gap between strong and weak moves so the weak ones rarely come through, and the reply is drawn from those, which keeps some variety in the openings. The sampler is seeded from the position itself, so the same position always draws the same answer."
                                </p>
                            </article>
                            <article>
                                <h3>"What it learned from"</h3>
                                <p>
                                    "11.2 million Lichess blitz, rapid, and classical games, kept only when both players were rated at least 2000, the game started from the normal position, and it ran between 10 and 300 half-moves. A single unreadable move throws out the whole game. The model saw those games about 1.8 times over 77,000 optimizer steps. There is no rating token and no result token in the input, so it has one playing strength and no dial to turn."
                                </p>
                            </article>
                            <article>
                                <h3>"Rating"</h3>
                                <p>
                                    "160 full games against Stockfish 17.1 at fixed strength levels fit a rating of 1395, with a 95% interval of 1322 to 1455. Single-move tests once scored it near 1930, and the gap is the finding: its moves look strong one at a time, but with no search it walks into tactics across a whole game. Same approximate Chess.com scale as the other engines."
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

/// Once both sides have moved, the opponent and the side you play are fixed
/// for the rest of the game. "New game" is what clears it.
fn game_locked(half_moves: usize) -> bool {
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
        assert_eq!(Model::MinimaxDepth3.elo_label(), "≈1700 Elo");
        assert_eq!(Model::MinimaxNineSeconds.elo_label(), "≈2060 Elo");
        assert_eq!(Model::AlphaMini.elo_label(), "≈2290 Elo");
        assert_eq!(Model::MiniGpt.elo_label(), "≈1400 Elo");
        assert_eq!(
            Model::AlphaMini.url(),
            "/projects/chessengines/api/alphamini/move"
        );
    }

    #[test]
    fn minigpt_is_offered_with_its_calibrated_rating() {
        assert_eq!(Model::MiniGpt.elo_label(), "≈1400 Elo");
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
    fn game_locks_after_one_full_move() {
        assert!(!game_locked(0));
        assert!(!game_locked(1));
        assert!(game_locked(2));
        assert!(game_locked(3));
    }

    #[test]
    fn model_ids_round_trip() {
        for model in [
            Model::Random,
            Model::MinimaxDepth3,
            Model::MinimaxNineSeconds,
            Model::AlphaMini,
            Model::MiniGpt,
        ] {
            assert_eq!(Model::from_id(model.id()), Some(model));
        }
        assert_eq!(Model::from_id("stockfish"), None);
        assert_eq!(Model::from_id(""), None);
    }

    #[test]
    fn colors_round_trip() {
        assert_eq!(color_from_id(color_id(Color::White)), Some(Color::White));
        assert_eq!(color_from_id(color_id(Color::Black)), Some(Color::Black));
        assert_eq!(color_from_id("green"), None);
    }

    #[test]
    fn replay_rebuilds_a_short_game() {
        let moves = ["e4", "e5", "Nf3", "Nc6"].map(String::from).to_vec();
        let board = replay_san(&moves).expect("a legal game replays");

        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.san_history, moves);
        assert_eq!(board.status(), Status::Ongoing);
        assert_eq!(board.export_san(), "1. e4 e5 2. Nf3 Nc6");
    }

    #[test]
    fn replay_of_an_empty_game_is_the_start_position() {
        let board = replay_san(&[]).expect("no moves replays");
        assert_eq!(board.to_fen(), START_FEN);
    }

    #[test]
    fn replay_rejects_garbage() {
        // Not a move at all.
        assert!(replay_san(&["e4".into(), "hello".into()]).is_none());
        // Well formed, illegal in the position it reaches.
        assert!(replay_san(&["e4".into(), "e5".into(), "e5".into()]).is_none());
        // Illegal on the very first ply.
        assert!(replay_san(&["Qd5".into()]).is_none());
        // Playing on after checkmate.
        assert!(
            replay_san(&[
                "f3".into(),
                "e5".into(),
                "g4".into(),
                "Qh4#".into(),
                "a3".into()
            ])
            .is_none()
        );
    }

    #[test]
    fn a_saved_game_parses_back_into_its_signals() {
        let raw = r#"{"model":"alphamini","color":"black","san":["e4","e5"]}"#;
        let game = parse_saved_game(raw).expect("a well formed value restores");

        assert_eq!(game.model, Model::AlphaMini);
        assert_eq!(game.color, Color::Black);
        assert_eq!(game.san, vec!["e4".to_string(), "e5".to_string()]);
        assert_eq!(game.board.san_history.len(), 2);
    }

    #[test]
    fn a_saved_game_round_trips_through_json() {
        let saved = SavedGame {
            model: Model::MiniGpt.id().to_string(),
            color: color_id(Color::White).to_string(),
            san: vec!["d4".to_string()],
        };
        let encoded = serde_json::to_string(&saved).expect("serializes");
        let game = parse_saved_game(&encoded).expect("restores");

        assert_eq!(game.model, Model::MiniGpt);
        assert_eq!(game.color, Color::White);
        assert_eq!(game.san, vec!["d4".to_string()]);
    }

    #[test]
    fn a_damaged_saved_game_is_discarded() {
        // Not JSON.
        assert!(parse_saved_game("").is_none());
        assert!(parse_saved_game("{oops").is_none());
        // JSON of the wrong shape.
        assert!(parse_saved_game(r#"{"model":"random"}"#).is_none());
        assert!(parse_saved_game(r#"[1, 2, 3]"#).is_none());
        // A model or colour this build does not know.
        assert!(parse_saved_game(r#"{"model":"leela","color":"white","san":[]}"#).is_none());
        assert!(parse_saved_game(r#"{"model":"random","color":"teal","san":[]}"#).is_none());
        // Moves that do not replay.
        assert!(parse_saved_game(r#"{"model":"random","color":"white","san":["e9"]}"#).is_none());
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
