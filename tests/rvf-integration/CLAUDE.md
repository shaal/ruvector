End-to-end smoke tests for the RVF (RuVector Format) lifecycle: create -> ingest -> query -> close -> reopen -> query -> verify match, plus delete/compact/derive.

Files:
- `smoke-test.js` - Node CLI driver that exercises `npx ruvector rvf` commands with dim=128, cosine metric, 20 vectors, k=5. Exits 0 on success.
- `tests/` - Rust version of the same lifecycle (15-step coverage).

Run the JS test with `node tests/rvf-integration/smoke-test.js` from the repo root. Run the Rust test via `cargo test -p <crate> --test rvf_smoke_test` (see the file header for details). Underlying crates live at `../../crates/rvf/`.
