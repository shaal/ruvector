# ruvector-dag/tests/integration

Per-subsystem integration tests.

- `mod.rs` — wires the submodules into a single test binary.
- `attention_tests.rs` — exercises the seven attention mechanisms.
- `dag_tests.rs` — `QueryDag` construction / traversal / serialization.
- `healing_tests.rs` — self-healing orchestrator + strategies.
- `mincut_tests.rs` — mincut engine and bottleneck analysis.
- `sona_tests.rs` — SONA engine, EWC++, MicroLoRA, reasoning bank, trajectories.

Uses helpers from `../fixtures/`. See `../CLAUDE.md`.
