# ruvector-dag/examples

Runnable examples (`cargo run --example <name> -p ruvector-dag`).

- `basic_usage.rs` — minimal DAG construction + topological attention.
- `attention_demo.rs` — compare each attention mechanism on a sample DAG.
- `attention_selection.rs` — `selector` picking an attention strategy per query.
- `learning_workflow.rs` — SONA MicroLoRA adaptation loop (requires `full` feature).
- `self_healing.rs` — anomaly detection → repair strategy pipeline.
- `synthetic_haptic.rs` — synthetic-data scenario for the haptic / sensor use case.
- `exotic/` — advanced research scenarios; see `exotic/CLAUDE.md`.

See `../CLAUDE.md`.
