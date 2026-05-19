# ruvector-tiny-dancer-node

Node.js / TypeScript bindings (NAPI-RS) for `../ruvector-tiny-dancer-core` — high-performance neural routing with zero-copy buffer sharing, async/await support, and TypeScript types.

## Layout

- `Cargo.toml` — `cdylib`. Deps: `ruvector-tiny-dancer-core`, `napi`/`napi-derive`, `tokio`, `thiserror`/`anyhow`, `serde`/`serde_json`, `chrono`, `parking_lot`. Build-dep: `napi-build`.
- `build.rs` — `napi_build::setup()`.
- `package.json` — npm metadata for `ruvector-tiny-dancer-node` v0.1.16.
- `src/lib.rs` — `#[napi(object)] RouterConfig`, plus wrappers around `CoreRouter`/`CoreRouterConfig`/`CoreRoutingRequest`/`CoreRoutingResponse`/`CoreRoutingDecision`/`CoreCandidate`. Uses `Arc` + `parking_lot::RwLock` for shared state.

## Public API (JS)

`Router` class with config (confidence threshold, uncertainty cap, circuit breaker, quantization, database path), routing requests, and decision responses.

## Related

- `../ruvector-tiny-dancer-core` — pure-Rust router implementation
- npm: `ruvector-tiny-dancer-node` (this dir's package.json)
