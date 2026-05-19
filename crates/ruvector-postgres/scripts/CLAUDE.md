# ruvector-postgres/scripts

Auxiliary scripts.

## Files

- `docker-test.sh` — Wraps the docker test-runner image for local CI parity.
- `download_models.rs` — Rust helper to download embedding models (MiniLM, BGE, MPNet) used by `../src/embeddings/`. Invoke via `cargo run --bin download_models` or similar.
