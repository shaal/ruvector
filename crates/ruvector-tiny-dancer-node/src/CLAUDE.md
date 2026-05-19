# ruvector-tiny-dancer-node/src

Sole source dir.

## Files

- `lib.rs` — `#[napi(object)] RouterConfig` and `#[napi]`-annotated `Router` wrapping `ruvector_tiny_dancer_core::Router`. Bridges core types (`CoreRoutingRequest`, `CoreRoutingResponse`, `CoreRoutingDecision`, `CoreCandidate`) to JS. Uses `parking_lot::RwLock` for interior mutability.
