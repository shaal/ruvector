# ruvector-rabitq-wasm/src

Single-file WASM binding crate.

## Files

- `lib.rs` — wraps `ruvector_rabitq::{AnnIndex, RabitqPlusIndex}`. Defines the JS `RabitqIndex` class (`build` + `search`), the `SearchResult` struct, and the `init()` panic-hook entry point. Sequential on wasm32 by design.
