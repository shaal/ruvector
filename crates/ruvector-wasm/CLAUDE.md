# ruvector-wasm

WASM bindings for Ruvector. Provides the full VectorDB API (insert, search,
delete, batch ops) in the browser, plus the kernel pack system (ADR-005)
for sandboxed execution of ML compute kernels.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib", "rlib"]`. Pulls
  `ruvector-core` (memory-only + uuid-support); optional `ruvector-collections`
  and `ruvector-filter`. `getrandom02 = "0.2"` aliased to inject `js` for
  WASM. `web-sys` enables IndexedDB. Features include `kernel-pack` to
  enable the manifest/sandbox system.
- `INTEGRATION_STATUS.md` - tracking notes for kernel pack integration.
- `package.json` - npm metadata for the published package.
- `.cargo/` - target-specific cargo overrides.
- `src/lib.rs` - top-level `#[wasm_bindgen]` exports; module wiring.
- `src/kernel/` - kernel pack system (when `kernel-pack` feature is on).
- `src/worker.js`, `src/worker-pool.js` - Web Worker scripts for parallel
  ops.
- `src/indexeddb.js` - IndexedDB persistence helpers used by the Rust side.
- `kernels/` - first-party WASM ML kernels (rmsnorm, rope, swiglu).
- `tests/wasm.rs` - `wasm-bindgen-test` integration tests.

## Public API surface
- `init()` (#[wasm_bindgen(start)]) - panic hook + tracing.
- `WasmVectorDB` (and friends) - VectorDB CRUD, search (`SearchQuery`,
  `SearchResult`), batch ops, IndexedDB persistence, zero-copy transfers.
- Behind `kernel-pack`: manifest parsing, Ed25519 signature verification,
  SHA256 hash verification, trusted-kernel allowlist, epoch-based
  execution budgets, shared-memory tensor protocol.

## Related
- `../ruvector-core`, `../ruvector-collections`, `../ruvector-filter`.
- `../ruvector-graph-wasm`, `../ruvector-solver-wasm`, `../ruvector-delta-wasm`.
- `../rvlite` builds on this for its single-binary front-end.
