Top-level integration tests that cross crate boundaries or exercise published packages end-to-end. Workspace unit tests live alongside their crates under `../crates/*/src/` and `../crates/*/tests/`.

Top-level Rust integration files:
- `advanced_tests.rs` - exercises `ruvector_core::advanced` (Hypergraph, TemporalHyperedge, CausalMemory, LearnedIndex, RecursiveModelIndex, HybridIndex).
- `graph_integration.rs`, `graph_full_integration.rs` - end-to-end tests for `ruvector-graph` (GraphDB, Node, Edge, Properties, queries).
- `hyperbolic_attention_tests.rs` - `ruvector-attention` hyperbolic mechanisms.
- `sandbox_security_tests.rs` - C5 sandbox path-restriction contract for `rvagent-backends`. Run with `cargo test -p rvagent-backends --test sandbox_security_tests`.
- `security_verification_test.rs` - verifies fixes for known security issues.
- `test_agenticdb.rs` - AgenticDB API surface (all 5 tables, compatibility with upstream agenticDB).

Top-level script:
- `test-all-packages.sh` - runs the broad smoke suite across all packages.

Subdirectories (each has its own CLAUDE.md):
- `agentic-jujutsu/` - Jest/TS test suite for the agentic-jujutsu version control system.
- `docker-integration/` - integration tests for published `ruvector-attention` (NAPI, WASM) running inside Docker.
- `integration/distributed/` - Raft, replication, sharding tests using a docker-compose'd 5-node cluster.
- `rvf-integration/` - RVF lifecycle smoke tests (CLI-driven and Rust).
- `wasm-integration/` - wasm-bindgen-test suites for the edge-net WASM crates.

Run a single integration test with `cargo test --test <file_stem>`. Most subdirectories have their own run scripts/configs.
