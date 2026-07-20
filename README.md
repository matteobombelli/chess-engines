# ChessBots

Run the random bot as a newline-delimited JSON backend:

```sh
echo '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","san":"e4"}' | cargo run -q -p random
```

It returns the bot's legal SAN move and the resulting FEN. Omit `san` to ask
the bot to move directly from the supplied position.
