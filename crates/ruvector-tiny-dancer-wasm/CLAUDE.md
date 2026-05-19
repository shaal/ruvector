# ruvector-tiny-dancer-wasm

WASM bindings for Tiny Dancer neural routing — exposes `ruvector-tiny-dancer-core` (FastGRNN router) to the browser/Node.

## Important files

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Depends on `ruvector-tiny-dancer-core` (path), `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `web-sys`. Release profile: `opt-level = "z"`, LTO, `panic = "abort"` (minimal WASM size).
- `package.json` — npm packaging metadata.
- `src/lib.rs` — `init()` panic hook + `RouterConfig` builder (`model_path`, `confidence_threshold`, `max_uncertainty`, `enable_circuit_breaker`, `circuit_breaker_threshold`, `enable_quantization`). Wraps `CoreRouter`, `CoreCandidate`, `CoreRoutingRequest`, `CoreRoutingResponse`.

## Public API (JS surface)

- `class RouterConfig` — fluent setters for routing knobs.
- Wrappers around `Router`, `Candidate`, `RoutingRequest`, `RoutingResponse` from the core crate.

## Build

```
wasm-pack build crates/ruvector-tiny-dancer-wasm --target web
```

## Related

- Backbone: `ruvector-tiny-dancer-core`.
- Node bindings of the related router: `ruvector-router-ffi` (different router-core).
- PostgreSQL-side routing: `ruvector-postgres/src/routing/`.
