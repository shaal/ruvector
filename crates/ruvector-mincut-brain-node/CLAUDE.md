# ruvector-mincut-brain-node

Minimal WASM binary for pi.ruv.io brain node publication (ADR-063). Re-exports canonical min-cut `extern "C"` functions from `ruvector-mincut::wasm::canonical` and provides V1 ABI stub exports (`memory`, `malloc`, `feature_extract_dim`, `feature_extract`) required by the brain server's node-publish endpoint.

## Important files

- `Cargo.toml` — `[workspace]` standalone, `publish = false`. `crate-type = ["cdylib"]`. Depends on `ruvector-mincut` (path, `wasm` + `canonical` features) and `getrandom` (js feature). Release profile: `opt-level = "z"`, `lto = true`, `strip = true`, `panic = "abort"` — tuned for the smallest possible WASM artifact.
- `Cargo.lock` — Standalone lockfile.
- `src/lib.rs` — Re-exports canonical min-cut WASM functions + V1 ABI stubs: `malloc` (bump allocator), `feature_extract_dim` (returns 0), `feature_extract` (no-op). `memory` is auto-exported by the WASM linker.

## V1 ABI

The brain server requires four exports for all WASM nodes:
1. `memory` — auto-exported linear memory.
2. `malloc(size) -> u32` — bump allocator starting at 64KB offset.
3. `feature_extract_dim() -> u32` — embedding dim (0 for graph nodes).
4. `feature_extract(...)` — no-op for graph nodes.

Plus the canonical min-cut entry points re-exported from `ruvector-mincut::wasm::canonical`.

## Build

```
cargo build --release --target wasm32-unknown-unknown
```

## Related

- `ruvector-mincut` — provides the canonical min-cut implementation.
- ADR-063 — Brain node publication protocol.
