# AlphaMini training operations

This package owns model training, checkpointing, artifact export, and the
transactional run ledger. Rust owns chess rules, self-play, and conversion of
raw positions into versioned tensor caches.

Use Python 3.12 and install the locked training environment:

```bash
cd alphamini-train
uv sync --extra train --extra test --locked
uv run alphamini-train doctor --config ../configs/alphamini/pilot.toml
```

The complete operational procedure is in
[`docs/alphamini/training-runbook.md`](../docs/alphamini/training-runbook.md).
