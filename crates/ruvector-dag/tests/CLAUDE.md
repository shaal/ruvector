# ruvector-dag/tests

Integration tests for the DAG / mincut / attention / SONA / healing stack.

- `mincut_tests.rs` — top-level mincut integration scenarios.
- `integration/` — additional grouped integration tests (`attention_tests.rs`, `dag_tests.rs`, `healing_tests.rs`, `mincut_tests.rs`, `sona_tests.rs`, `mod.rs`).
- `fixtures/` — shared test helpers (DAG generators, mock QuDAG, pattern + trajectory generators).
- `data/` — sample inputs (`sample_dags.json`).

See `../CLAUDE.md`.
