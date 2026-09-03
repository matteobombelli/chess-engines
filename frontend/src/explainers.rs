//! One "How it works" figure set per engine.
//!
//! Every number that appears in a diagram comes from the repository: search
//! limits from the deployed unit files, network shapes from the design docs,
//! and legal-move counts from `chess_core` itself.

use leptos::prelude::*;

/// A small start-position board. Used by the Random and AlphaMini figures.
fn board_glyph(x0: f64, y0: f64, cell: f64) -> impl IntoView {
    let squares = (0..64)
        .filter(|index| (index % 8 + index / 8) % 2 == 1)
        .map(move |index| {
            let file = (index % 8) as f64;
            let rank = (index / 8) as f64;
            view! {
                <rect
                    class="cell"
                    x={x0 + file * cell}
                    y={y0 + rank * cell}
                    width={cell}
                    height={cell}
                >
                </rect>
            }
        })
        .collect_view();

    // Ranks 8 and 7 hold the dark pieces, ranks 2 and 1 the light ones.
    let pieces = [(0_usize, true), (1, true), (6, false), (7, false)]
        .into_iter()
        .flat_map(|(rank, dark)| (0..8).map(move |file| (rank, dark, file)))
        .map(move |(rank, dark, file)| {
            view! {
                <circle
                    class={if dark { "piece dark" } else { "piece" }}
                    cx={x0 + (file as f64 + 0.5) * cell}
                    cy={y0 + (rank as f64 + 0.5) * cell}
                    r={cell * 0.28}
                >
                </circle>
            }
        })
        .collect_view();

    view! {
        <g>
            <rect class="box" x={x0} y={y0} width={cell * 8.0} height={cell * 8.0}></rect>
            {squares}
            {pieces}
        </g>
    }
}

/// The 20 legal moves of the opening position, in the order `chess_core`
/// reports them once sorted, laid out four to a row.
const START_MOVES: [&str; 20] = [
    "a3", "a4", "b3", "b4", "c3", "c4", "d3", "d4", "e3", "e4", "f3", "f4", "g3", "g4", "h3", "h4",
    "Na3", "Nc3", "Nf3", "Nh3",
];
const START_PICK: usize = 11;

#[component]
pub fn RandomHowItWorks() -> impl IntoView {
    let chips = START_MOVES
        .into_iter()
        .enumerate()
        .map(|(index, san)| {
            let x = 244.0 + (index % 4) as f64 * 60.0;
            let y = 66.0 + (index / 4) as f64 * 28.0;
            let picked = index == START_PICK;
            view! {
                <g>
                    <rect
                        class={if picked { "box accent" } else { "box" }}
                        x={x}
                        y={y}
                        width="54"
                        height="22"
                        rx="3"
                    >
                    </rect>
                    <text
                        class={if picked { "label accent" } else { "label" }}
                        x={x + 27.0}
                        y={y + 15.0}
                        text-anchor="middle"
                    >
                        {san}
                    </text>
                </g>
            }
        })
        .collect_view();

    view! {
        <div class="how-it-works">
            <h3>"How it works"</h3>

            <figure class="chart diagram">
                <svg
                    viewBox="0 0 720 260"
                    role="img"
                    aria-label="Random replays the game, asks the shared Rust rules engine for every legal move, and then picks one of the 20 legal opening moves with equal odds."
                >
                    <defs>
                        <marker
                            id="random-arrow"
                            viewBox="0 0 8 8"
                            refX="7"
                            refY="4"
                            markerWidth="6"
                            markerHeight="6"
                            orient="auto"
                        >
                            <path class="head" d="M 0 0 L 8 4 L 0 8 z"></path>
                        </marker>
                    </defs>

                    {board_glyph(20.0, 73.0, 15.0)}
                    <text class="tick" x="80" y="60" text-anchor="middle">"the position"</text>

                    <text class="tick" x="190" y="106" text-anchor="middle">"shared Rust"</text>
                    <text class="tick" x="190" y="120" text-anchor="middle">"rules engine"</text>
                    <line
                        class="edge"
                        x1="148"
                        y1="133"
                        x2="232"
                        y2="133"
                        marker-end="url(#random-arrow)"
                    >
                    </line>

                    <text class="tick" x="361" y="52" text-anchor="middle">"20 legal moves"</text>
                    {chips}

                    <text class="tick" x="516" y="106" text-anchor="middle">"one drawn"</text>
                    <text class="tick" x="516" y="120" text-anchor="middle">"at random"</text>
                    <line
                        class="edge"
                        x1="486"
                        y1="133"
                        x2="546"
                        y2="133"
                        marker-end="url(#random-arrow)"
                    >
                    </line>

                    <text class="tick" x="624" y="96" text-anchor="middle">"the reply"</text>
                    <rect class="box accent" x="554" y="109" width="140" height="48" rx="4"></rect>
                    <text class="label accent" x="624" y="138" text-anchor="middle">"f4"</text>
                </svg>
                <figcaption>
                    <strong>"One move, drawn at random."</strong>
                    " The browser sends the moves played so far. Random replays them, asks the shared
                    rules engine for every move that is legal in the position it reaches, and picks
                    one. The opening position has 20 legal moves, so each one has a 1 in 20 chance.
                    Nothing in Random scores moves, so it is as likely to hang its queen as to
                    take yours."
                </figcaption>
            </figure>
        </div>
    }
}

/// Leaf centres, then the scores under them. The two blank leaves are the ones
/// the cutoff skipped.
const TREE_LEAF_X: [f64; 9] = [
    160.0, 226.0, 293.0, 359.0, 425.0, 491.0, 558.0, 624.0, 690.0,
];
const TREE_LEAF_SCORE: [&str; 9] = [
    "+0.1", "-0.8", "+0.5", "", "", "+1.2", "+0.7", "+0.6", "+0.9",
];
const TREE_LEAF_SKIPPED: [bool; 9] = [false, false, false, true, true, false, false, false, false];
const TREE_LEAF_Y: f64 = 244.0;

fn triangle_up(cx: f64, cy: f64) -> String {
    format!(
        "{},{} {},{} {},{}",
        cx,
        cy - 13.0,
        cx - 14.0,
        cy + 11.0,
        cx + 14.0,
        cy + 11.0
    )
}

fn triangle_down(cx: f64, cy: f64) -> String {
    format!(
        "{},{} {},{} {},{}",
        cx - 14.0,
        cy - 11.0,
        cx + 14.0,
        cy - 11.0,
        cx,
        cy + 13.0
    )
}

/// The caption's "about 2,000 positions" is `find_best_move` at depth 3 from
/// `1. e4 e5 2. Nf3`, which reports 2,124 nodes.
#[component]
pub fn MinimaxHowItWorks() -> impl IntoView {
    let leaves = (0..9)
        .map(|index| {
            let x = TREE_LEAF_X[index];
            let skipped = TREE_LEAF_SKIPPED[index];
            let chosen = index == 8;
            view! {
                <g class={if skipped { "muted" } else { "" }}>
                    <rect
                        class={if chosen { "box accent" } else { "box" }}
                        x={x - 22.0}
                        y={TREE_LEAF_Y - 12.0}
                        width="44"
                        height="24"
                        rx="3"
                    >
                    </rect>
                    <text
                        class={if chosen { "label accent" } else { "label" }}
                        x={x}
                        y={TREE_LEAF_Y + 5.0}
                        text-anchor="middle"
                    >
                        {TREE_LEAF_SCORE[index]}
                    </text>
                </g>
            }
        })
        .collect_view();

    view! {
        <div class="how-it-works">
            <h3>"How it works"</h3>

            <figure class="chart diagram">
                <svg
                    viewBox="0 0 720 300"
                    role="img"
                    aria-label="Depth-3 Minimax builds a tree of its move, your reply, and its answer, scores the positions at the bottom, and carries the scores back up, skipping one branch that alpha-beta pruning shows cannot change the answer."
                >
                    <text class="tick" x="126" y="51" text-anchor="end">"my move"</text>
                    <text class="tick" x="126" y="117" text-anchor="end">"your best reply"</text>
                    <text class="tick" x="126" y="183" text-anchor="end">"my reply"</text>
                    <text class="tick" x="126" y="249" text-anchor="end">"score"</text>

                    <line class="edge" x1="434" y1="57" x2="276" y2="101"></line>
                    <line class="edge accent" x1="434" y1="57" x2="591" y2="101"></line>
                    <line class="edge" x1="276" y1="125" x2="193" y2="165"></line>
                    <line class="edge" x1="276" y1="125" x2="359" y2="165"></line>
                    <line class="edge" x1="591" y1="125" x2="524" y2="165"></line>
                    <line class="edge accent" x1="591" y1="125" x2="657" y2="165"></line>
                    <line class="edge" x1="193" y1="189" x2="160" y2="232"></line>
                    <line class="edge" x1="193" y1="189" x2="226" y2="232"></line>
                    <line class="edge" x1="359" y1="189" x2="293" y2="232"></line>
                    <g class="muted">
                        <line class="edge" x1="359" y1="189" x2="359" y2="232"></line>
                        <line class="edge" x1="359" y1="189" x2="425" y2="232"></line>
                    </g>
                    <line class="edge" x1="524" y1="189" x2="491" y2="232"></line>
                    <line class="edge" x1="524" y1="189" x2="558" y2="232"></line>
                    <line class="edge" x1="657" y1="189" x2="624" y2="232"></line>
                    <line class="edge accent" x1="657" y1="189" x2="690" y2="232"></line>

                    <polygon class="box accent" points={triangle_up(434.0, 46.0)}></polygon>
                    <polygon class="box" points={triangle_down(276.0, 112.0)}></polygon>
                    <polygon class="box accent" points={triangle_down(591.0, 112.0)}></polygon>
                    <polygon class="box" points={triangle_up(193.0, 178.0)}></polygon>
                    <polygon class="box" points={triangle_up(359.0, 178.0)}></polygon>
                    <polygon class="box" points={triangle_up(524.0, 178.0)}></polygon>
                    <polygon class="box accent" points={triangle_up(657.0, 178.0)}></polygon>

                    {leaves}

                    <text class="label accent" x="434" y="22" text-anchor="middle">"+0.9"</text>
                    <text class="tick" x="213" y="183">"+0.1"</text>
                    <text class="tick" x="379" y="183">"already +0.5"</text>
                    <text class="tick" x="392" y="278" text-anchor="middle">
                        "alpha-beta cutoff"
                    </text>
                </svg>
                <figcaption>
                    <strong>"Three moves ahead."</strong>
                    " The labels are written from the engine's side of the board. It tries each of its
                    own moves, then each of your replies, then each of its answers, and scores the
                    position it lands in. Reading back up, it assumes you take the reply it likes
                    least and it takes the answer it likes most. That makes the right-hand branch
                    worth +0.9, and that is the branch it plays. The greyed branch was never scored.
                    Its first answer there was already worth +0.5, more than the +0.1 you can hold
                    the engine to on the branch beside it, so you would never choose that branch, and
                    alpha-beta pruning stopped looking. The tree drawn here branches two ways at each
                    level. The real position after 1. e4 e5 2. Nf3 is much wider, and the depth-3
                    search visits about 2,000 positions before it answers."
                </figcaption>
            </figure>
        </div>
    }
}

/// Iterative deepening from the opening position, measured once on my own
/// machine with the deployed release build. Cumulative seconds at the end of
/// each depth, mapped onto a 0 to 9 second track running from x 60 to x 690.
const TIMELINE_X0: f64 = 60.0;
const TIMELINE_PER_SECOND: f64 = 70.0;
const TIMELINE_DEPTH_4_END: f64 = 0.031;
const TIMELINE_DEPTH_5_END: f64 = 0.851;
const TIMELINE_DEPTH_6_END: f64 = 3.776;
const TIMELINE_BUDGET: f64 = 9.0;

fn timeline_x(seconds: f64) -> f64 {
    TIMELINE_X0 + seconds * TIMELINE_PER_SECOND
}

#[component]
pub fn TimedMinimaxHowItWorks() -> impl IntoView {
    let end_4 = timeline_x(TIMELINE_DEPTH_4_END);
    let end_5 = timeline_x(TIMELINE_DEPTH_5_END);
    let end_6 = timeline_x(TIMELINE_DEPTH_6_END);
    let end = timeline_x(TIMELINE_BUDGET);

    let ticks = (0..=9)
        .map(|second| {
            let x = timeline_x(second as f64);
            view! { <line class="edge" x1={x} y1="210" x2={x} y2="216"></line> }
        })
        .collect_view();

    let saves = [end_4, end_5, end_6]
        .into_iter()
        .map(|x| view! { <line class="edge accent" x1={x} y1="148" x2={x} y2="162"></line> })
        .collect_view();

    view! {
        <div class="how-it-works">
            <h3>"How it works"</h3>

            <figure class="chart diagram">
                <svg
                    viewBox="0 0 720 300"
                    role="img"
                    aria-label="The 9-second Minimax searches one depth after another inside a nine-second budget, saving a best move after every depth it finishes and discarding the depth that is still running when the time runs out."
                >
                    <line class="edge" x1={end_4} y1="96" x2={end_4} y2="62"></line>
                    <text class="tick" x="68" y="58">
                        "depths 1 to 4 finish inside the first 0.03 seconds"
                    </text>

                    <rect
                        class="box"
                        x={TIMELINE_X0}
                        y="96"
                        width={end_4 - TIMELINE_X0}
                        height="52"
                    >
                    </rect>
                    <rect class="box" x={end_4} y="96" width={end_5 - end_4} height="52"></rect>
                    <rect class="box" x={end_5} y="96" width={end_6 - end_5} height="52"></rect>
                    <rect
                        class="box muted"
                        x={end_6}
                        y="96"
                        width={end - end_6}
                        height="52"
                    >
                    </rect>
                    <text class="label" x={(end_4 + end_5) / 2.0} y="127" text-anchor="middle">
                        "5"
                    </text>
                    <text class="label" x={(end_5 + end_6) / 2.0} y="127" text-anchor="middle">
                        "6"
                    </text>
                    <text class="label" x={(end_6 + end) / 2.0} y="127" text-anchor="middle">
                        "depth 7, cut off"
                    </text>

                    {saves}
                    <text class="tick" x="193" y="180" text-anchor="middle">"best move saved"</text>

                    <line class="baseline" x1={TIMELINE_X0} y1="210" x2={end} y2="210"></line>
                    {ticks}
                    <text class="tick" x={TIMELINE_X0} y="234" text-anchor="middle">"0 s"</text>
                    <text class="tick" x={timeline_x(3.0)} y="234" text-anchor="middle">"3 s"</text>
                    <text class="tick" x={timeline_x(6.0)} y="234" text-anchor="middle">"6 s"</text>
                    <text class="tick" x={end} y="234" text-anchor="middle">"9 s"</text>

                    <line
                        class="edge accent"
                        x1={end}
                        y1="60"
                        x2={end}
                        y2="210"
                        stroke-dasharray="4 5"
                    >
                    </line>
                    <text class="label accent" x={end - 6.0} y="52" text-anchor="end">
                        "time is up"
                    </text>

                    <text class="tick" x={(end_6 + end) / 2.0} y="262" text-anchor="middle">
                        "this depth never finishes, so its work is thrown away"
                    </text>
                </svg>
                <figcaption>
                    <strong>"Nine seconds, one depth at a time."</strong>
                    " The engine searches to depth 1, then starts over at depth 2, and so on until the
                    clock stops it. Every depth it finishes leaves a best move behind, so it always
                    has an answer ready when the nine seconds end. Whatever the last search had
                    worked out is thrown away, because a half-searched depth has only looked at some
                    of the moves and would rank them unfairly. The widths above come from one search
                    of the opening position on my own machine, where the first four depths together
                    cost 0.03 seconds and depth 6 alone cost 2.9 seconds. The deployed server is
                    different hardware and the cost also changes with the position, so the depth it
                    reaches moves around."
                </figcaption>
            </figure>
        </div>
    }
}

#[component]
pub fn AlphaMiniHowItWorks() -> impl IntoView {
    // The front plane is 76 units square, so its eight rows and columns are
    // 9.5 apart.
    let plane_grid = (1..8)
        .flat_map(|step| {
            let offset = step as f64 * 9.5;
            [
                view! {
                    <line
                        class="grid"
                        x1={184.0 + offset}
                        y1={100.0}
                        x2={184.0 + offset}
                        y2={176.0}
                    >
                    </line>
                },
                view! {
                    <line
                        class="grid"
                        x1={184.0}
                        y1={100.0 + offset}
                        x2={260.0}
                        y2={100.0 + offset}
                    >
                    </line>
                },
            ]
        })
        .collect_view();

    view! {
        <div class="how-it-works">
            <h3>"How it works"</h3>

            <figure class="chart diagram">
                <svg
                    viewBox="0 0 720 260"
                    role="img"
                    aria-label="AlphaMini encodes the position as 22 planes of 8 by 8, runs them through a 64-channel trunk of six residual blocks, and reads two heads, one that scores all 4,672 moves and one that predicts a win, a draw, or a loss."
                >
                    <defs>
                        <marker
                            id="alphamini-arrow"
                            viewBox="0 0 8 8"
                            refX="7"
                            refY="4"
                            markerWidth="6"
                            markerHeight="6"
                            orient="auto"
                        >
                            <path class="head" d="M 0 0 L 8 4 L 0 8 z"></path>
                        </marker>
                    </defs>

                    {board_glyph(20.0, 74.0, 14.0)}
                    <text class="tick" x="76" y="60" text-anchor="middle">"the position"</text>
                    <line
                        class="edge"
                        x1="140"
                        y1="130"
                        x2="176"
                        y2="130"
                        marker-end="url(#alphamini-arrow)"
                    >
                    </line>

                    <rect class="box" x="200" y="84" width="76" height="76"></rect>
                    <rect class="box" x="192" y="92" width="76" height="76"></rect>
                    <rect class="box" x="184" y="100" width="76" height="76"></rect>
                    {plane_grid}
                    <text class="tick" x="230" y="198" text-anchor="middle">"22 planes, 8 by 8"</text>
                    <line
                        class="edge"
                        x1="284"
                        y1="130"
                        x2="312"
                        y2="130"
                        marker-end="url(#alphamini-arrow)"
                    >
                    </line>

                    <rect class="box" x="320" y="74" width="158" height="112" rx="4"></rect>
                    <text class="label accent" x="399" y="104" text-anchor="middle">"conv trunk"</text>
                    <text class="tick" x="399" y="128" text-anchor="middle">"64-channel stem"</text>
                    <text class="tick" x="399" y="146" text-anchor="middle">"6 residual blocks"</text>
                    <text class="tick" x="399" y="164" text-anchor="middle">
                        "squeeze and excitation"
                    </text>

                    <line
                        class="edge"
                        x1="486"
                        y1="126"
                        x2="534"
                        y2="94"
                        marker-end="url(#alphamini-arrow)"
                    >
                    </line>
                    <line
                        class="edge"
                        x1="486"
                        y1="134"
                        x2="534"
                        y2="166"
                        marker-end="url(#alphamini-arrow)"
                    >
                    </line>

                    <rect class="box" x="540" y="62" width="166" height="56" rx="4"></rect>
                    <text class="label accent" x="623" y="86" text-anchor="middle">"policy head"</text>
                    <text class="tick" x="623" y="106" text-anchor="middle">"4,672 move scores"</text>

                    <rect class="box" x="540" y="142" width="166" height="56" rx="4"></rect>
                    <text class="label accent" x="623" y="166" text-anchor="middle">"value head"</text>
                    <text class="tick" x="623" y="186" text-anchor="middle">"win, draw, or loss"</text>
                </svg>
                <figcaption>
                    <strong>"One small network, two answers."</strong>
                    " The board becomes 22 planes of 8 by 8 numbers. Twelve hold the pieces, six for
                    its own and six for yours, and the remaining ten carry state that the piece
                    placement does not show. Those are castling rights, the en passant square, how
                    often this position has already occurred, the halfmove clock, which colour is
                    to move, and one plane of ones that lets the convolutions find the edge of the
                    board.
                    When AlphaMini plays Black the whole board is flipped, so the side to move always
                    faces up the board and the network only has to learn chess from one direction. A
                    64-channel trunk of six residual blocks reads those planes. The policy head then
                    scores every one of the 4,672 moves the action space can express, and the value
                    head gives the odds of a win, a draw, and a loss, which search reads as the win
                    chance minus the loss chance. The network is given no move history and no opening
                    book, and it only scores moves the rules engine has already listed."
                </figcaption>
            </figure>

            <figure class="chart diagram">
                <svg
                    viewBox="0 0 720 300"
                    role="img"
                    aria-label="Monte Carlo tree search repeats a loop of select, expand, evaluate with the network, and back up, until it has run 10,000 simulations or spent nine seconds, and then plays the move it visited most often."
                >
                    <defs>
                        <marker
                            id="mcts-arrow"
                            viewBox="0 0 8 8"
                            refX="7"
                            refY="4"
                            markerWidth="6"
                            markerHeight="6"
                            orient="auto"
                        >
                            <path class="head" d="M 0 0 L 8 4 L 0 8 z"></path>
                        </marker>
                    </defs>

                    <path class="edge" d="M 130 106 Q 130 46 231 46" marker-end="url(#mcts-arrow)"></path>
                    <path class="edge" d="M 369 46 Q 470 46 470 106" marker-end="url(#mcts-arrow)"></path>
                    <path class="edge" d="M 470 154 Q 470 214 369 214" marker-end="url(#mcts-arrow)"></path>
                    <path class="edge" d="M 231 214 Q 130 214 130 154" marker-end="url(#mcts-arrow)"></path>

                    <rect class="box" x="65" y="110" width="130" height="40" rx="4"></rect>
                    <text class="label" x="130" y="128" text-anchor="middle">"select"</text>
                    <text class="tick" x="130" y="144" text-anchor="middle">"walk to a new leaf"</text>

                    <rect class="box" x="235" y="26" width="130" height="40" rx="4"></rect>
                    <text class="label" x="300" y="44" text-anchor="middle">"expand"</text>
                    <text class="tick" x="300" y="60" text-anchor="middle">"list the legal moves"</text>

                    <rect class="box" x="405" y="110" width="130" height="40" rx="4"></rect>
                    <text class="label" x="470" y="128" text-anchor="middle">"evaluate"</text>
                    <text class="tick" x="470" y="144" text-anchor="middle">"run the network"</text>

                    <rect class="box" x="235" y="194" width="130" height="40" rx="4"></rect>
                    <text class="label" x="300" y="212" text-anchor="middle">"back up"</text>
                    <text class="tick" x="300" y="228" text-anchor="middle">"update the path"</text>

                    <text class="label accent" x="300" y="124" text-anchor="middle">
                        "up to 10,000 times"
                    </text>
                    <text class="tick" x="300" y="144" text-anchor="middle">
                        "or nine seconds"
                    </text>

                    <line
                        class="edge"
                        x1="340"
                        y1="238"
                        x2="420"
                        y2="262"
                        marker-end="url(#mcts-arrow)"
                    >
                    </line>
                    <rect class="box accent" x="428" y="246" width="278" height="46" rx="4"></rect>
                    <text class="label accent" x="567" y="274" text-anchor="middle">
                        "the most visited move is played"
                    </text>
                </svg>
                <figcaption>
                    <strong>"The search around it."</strong>
                    " One simulation walks down the tree to a position the search has not seen, lists
                    the legal moves there, asks the network how the game looks and which moves
                    deserve attention, then carries that answer back up the path it came down. Each
                    step down weighs the value the tree has already measured against the policy
                    head's prior, so a move the policy head rated highly is visited early and keeps
                    being visited only while its measured value holds up. The move with the most
                    visits at the end is the one played. Visit counts are a steadier signal than
                    the value estimate, because a move only collects visits by scoring well as the
                    tree grows. The Rust rules engine lists the legal moves at every node, so the
                    network only ever ranks moves the rules allow."
                </figcaption>
            </figure>
        </div>
    }
}

/// A plausible ranking shape over candidate replies after 1. e4 e5 2. Nf3.
/// The third field records whether `chess_core` calls the move legal there.
const MINIGPT_BARS: [(&str, f64, bool); 12] = [
    ("Nc6", 150.0, true),
    ("Nf6", 126.0, true),
    ("d6", 86.0, true),
    ("Nd7", 30.0, false),
    ("d5", 74.0, true),
    ("Bc5", 64.0, true),
    ("Qd5", 22.0, false),
    ("Qh4", 46.0, true),
    ("Kd7", 14.0, false),
    ("Be7", 38.0, true),
    ("a6", 26.0, true),
    ("O-O", 18.0, false),
];
const MINIGPT_BASELINE: f64 = 220.0;

#[component]
pub fn MiniGptHowItWorks() -> impl IntoView {
    let tokens = ["BOS", "e4", "e5", "Nf3", "…"]
        .into_iter()
        .enumerate()
        .map(|(index, token)| {
            let x = 20.0 + index as f64 * 56.0;
            view! {
                <g>
                    <rect class="box" x={x} y="106" width="52" height="32" rx="3"></rect>
                    <text class="label" x={x + 26.0} y="127" text-anchor="middle">{token}</text>
                </g>
            }
        })
        .collect_view();

    let bars = MINIGPT_BARS
        .into_iter()
        .enumerate()
        .map(|(index, (san, height, legal))| {
            let x = 30.0 + index as f64 * 56.0;
            view! {
                <g class={if legal { "" } else { "muted" }}>
                    <rect
                        class={if index == 0 { "fill warm" } else { "fill" }}
                        x={x}
                        y={MINIGPT_BASELINE - height}
                        width="38"
                        height={height}
                    >
                    </rect>
                    <text class="tick" x={x + 19.0} y="238" text-anchor="middle">{san}</text>
                </g>
            }
        })
        .collect_view();

    view! {
        <div class="how-it-works">
            <h3>"How it works"</h3>

            <figure class="chart diagram">
                <svg
                    viewBox="0 0 720 260"
                    role="img"
                    aria-label="MiniGPT turns each move played into one token and runs the sequence through 12 transformer layers, which return one score for every token in its 4,736-entry vocabulary."
                >
                    <defs>
                        <marker
                            id="minigpt-arrow"
                            viewBox="0 0 8 8"
                            refX="7"
                            refY="4"
                            markerWidth="6"
                            markerHeight="6"
                            orient="auto"
                        >
                            <path class="head" d="M 0 0 L 8 4 L 0 8 z"></path>
                        </marker>
                    </defs>

                    <text class="tick" x="158" y="90" text-anchor="middle">"the game so far"</text>
                    {tokens}
                    <text class="tick" x="158" y="160" text-anchor="middle">"one token per move"</text>

                    <line
                        class="edge"
                        x1="306"
                        y1="122"
                        x2="348"
                        y2="122"
                        marker-end="url(#minigpt-arrow)"
                    >
                    </line>

                    <rect class="box" x="356" y="56" width="200" height="134" rx="4"></rect>
                    <text class="label accent" x="456" y="84" text-anchor="middle">"MiniGPT"</text>
                    <text class="tick" x="456" y="110" text-anchor="middle">"12 layers"</text>
                    <text class="tick" x="456" y="128" text-anchor="middle">"512 values per token"</text>
                    <text class="tick" x="456" y="146" text-anchor="middle">"8 attention heads"</text>
                    <text class="tick" x="456" y="164" text-anchor="middle">"256 tokens of context"</text>

                    <line
                        class="edge"
                        x1="564"
                        y1="122"
                        x2="600"
                        y2="122"
                        marker-end="url(#minigpt-arrow)"
                    >
                    </line>

                    <text class="tick" x="657" y="78" text-anchor="middle">"4,736 tokens"</text>
                    <rect class="box" x="608" y="92" width="98" height="60" rx="4"></rect>
                    <text class="label" x="657" y="116" text-anchor="middle">"one score"</text>
                    <text class="label" x="657" y="136" text-anchor="middle">"per token"</text>
                </svg>
                <figcaption>
                    <strong>"A language model over moves."</strong>
                    " Every move played becomes a single token, after one token that marks the start
                    of the game. Twelve transformer layers read the whole sequence in one forward
                    pass and return a score for each of the 4,736 tokens in the vocabulary. Inside
                    every layer, each token attends to all the tokens before it, so the output at
                    the last token depends on the whole game, including the move you just played.
                    Nothing caches between moves, so the whole game is pushed through
                    the model again every time it is MiniGPT's turn. That vocabulary is AlphaMini's
                    action space reused, so both models name moves the same way. The window holds 256
                    tokens, the start token plus the newest 255 moves, so a longer game loses its own
                    opening from view."
                </figcaption>
            </figure>

            <figure class="chart diagram">
                <svg
                    viewBox="0 0 720 260"
                    role="img"
                    aria-label="The rules engine drops every token that is not a legal move in the current position, and the reply is sampled from the legal scores that remain."
                >
                    <text class="tick" x="24" y="34">"after 1. e4 e5 2. Nf3"</text>
                    <text class="tick" x="700" y="34" text-anchor="end">
                        "greyed moves are illegal here"
                    </text>
                    <text class="label warm" x="30" y="58">"usually played"</text>
                    {bars}
                    <line class="baseline" x1="24" y1={MINIGPT_BASELINE} x2="700" y2={MINIGPT_BASELINE}></line>
                </svg>
                <figcaption>
                    <strong>"Legal moves only."</strong>
                    " The model has no board, so nothing stops it scoring a move that cannot be
                    played. After 1. e4 e5 2. Nf3 there are 29 legal replies. Rust works them out and
                    drops every other token before anything is chosen, so an illegal move cannot
                    reach the board however highly the model scored it. The reply is then sampled
                    from what survives at temperature 0.5, which keeps some variety in the openings
                    without letting weak moves through often. The draw is seeded from the position,
                    so MiniGPT answers the same position the same way every time. The bars show the
                    shape of a ranking and are drawn by hand, not measured."
                </figcaption>
            </figure>
        </div>
    }
}
