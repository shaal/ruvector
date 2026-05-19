# OSpipe / src / bin

Executable entry points for the OSpipe crate.

## Important files
- `ospipe-server.rs` - main binary `ospipe-server`. Boots the Axum HTTP server defined in `../server/mod.rs` over the OSpipe pipeline (capture -> dedup -> embed -> store -> hybrid search). Native-only (cfg-gated away from `wasm32`).

## Build / run
- `cargo run -p ospipe --bin ospipe-server` (release: add `--release`).
- Prebuilt binaries live in `../../dist/`.

## Related
- HTTP layer: `../server/`.
- Pipeline being served: `../pipeline/`, `../storage/`, `../search/`.
