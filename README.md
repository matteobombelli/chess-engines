# ChessBots

Run the random bot API and frontend in separate terminals:

```sh
cargo run -p random
cd frontend && trunk serve --open
```

`POST http://127.0.0.1:3000/move` accepts `{"fen":"...","san":"e4"}` and
returns the bot's legal SAN move and resulting FEN.
