# ruvector-diskann-node

NAPI-RS Node.js native bindings for `ruvector-diskann`. Builds as a `cdylib` consumable from Node via napi 3.x.

## Files

- `Cargo.toml` — `crate-type = ["cdylib"]`. Depends on `ruvector-diskann`, `napi`, `napi-derive`, serde, tokio, `parking_lot`.
- `build.rs` — minimal `napi_build::setup()` shim.
- `src/lib.rs` — `DiskAnnOptions`, `DiskAnnSearchResult`, and the `DiskAnn` `#[napi]` class wrapping
  `Arc<RwLock<ruvector_diskann::DiskAnnIndex>>` with constructor, build, insert, and search methods exposed to JavaScript.

## Related

- `../ruvector-diskann` — the underlying Rust DiskANN implementation.
- `../../npm/packages/ruvector-diskann` (if present) — the npm wrapper that consumes the produced `.node` binary.
- Other NAPI bindings: search for sibling `ruvector-*-node` crates.
