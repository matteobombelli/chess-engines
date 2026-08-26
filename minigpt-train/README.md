# minigpt-train

Reproducible training and run operations for MiniGPT: a decoder-only GPT trained by
next-move prediction over the pre-tokenized game shards written by the Rust
`minigpt-ingest` binary.

The package mirrors `alphamini-train`: immutable run ledger, hash-frozen configuration,
fork-to-change, and RNG-complete resume. The unit of progress is a training **segment**
of `training.segment_steps` optimizer updates rather than a self-play cycle.

```
uv sync --extra train --extra test
uv run minigpt-train doctor --config ../configs/minigpt/pilot.toml
uv run minigpt-train start  --config ../configs/minigpt/pilot.toml --run-dir ../runs/minigpt-pilot-001
uv run minigpt-train resume --run-dir ../runs/minigpt-pilot-001
uv run minigpt-train export --run-dir ../runs/minigpt-pilot-001
```
