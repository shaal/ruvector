# ruvector-router-ffi/src

Single-file source for the NAPI-RS bindings.

## Files

- `lib.rs` — All `#[napi]` exports: `DistanceMetric` enum (with `From` conversions to/from `CoreDistanceMetric`), `DbOptions` config struct, `VectorDB` Node-facing class wrapping `Arc<CoreVectorDB>`. Module-level `INSTANCE_COUNTER: AtomicU64` for diagnostic identity.

## Notes

- Strict: `#![deny(clippy::all)]`.
- Async paths use `tokio` per the workspace runtime.
