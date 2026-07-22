# ChessBots

Run the random bot API and frontend in separate terminals:

```sh
cargo run -p random
cd frontend && trunk serve --open
```

`POST http://127.0.0.1:3000/move` accepts a FEN and an optional opponent move,
then returns the bot's legal SAN move and resulting FEN:

```json
{"fen":"...","san":"e4"}
```

Set `san` to `null` when the bot should move immediately from the supplied FEN,
such as when it opens a game as White. Bots always play the FEN's side to move,
so the same contract supports bots playing either color.

## Deploy

Deploy the committed `main` working tree to production with:

```sh
./scripts/deploy.sh
```

The script runs the workspace tests, builds the random bot and frontend in release
mode, publishes the frontend to `/srv/chessengines`, restarts the
`chessengines-random` user service, and verifies the live page and API.
