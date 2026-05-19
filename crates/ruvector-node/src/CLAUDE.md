# ruvector-node/src

Single-file NAPI binding crate.

## Files

- `lib.rs` — all `#[napi]` types and methods. Distance enum, config wrappers, async `VectorDB` API, collection / filter / metrics surfaces. Allows clippy::all + pedantic locally (generated NAPI bindings are noisy).
