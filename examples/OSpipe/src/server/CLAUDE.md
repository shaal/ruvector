# OSpipe / src / server

HTTP server module. Wraps the OSpipe pipeline + search behind an Axum app exposed by the `ospipe-server` binary in `../bin/`.

## Important files
- `mod.rs` - module root: route definitions, handlers, app state, CORS layer (via `tower-http`).

## Related
- Binary entry: `../bin/ospipe-server.rs`.
- Stages it wires together: `../pipeline/`, `../storage/`, `../search/`, `../graph/`, `../learning/`.
- Native-only (cfg-gated away from `wasm32`); WASM users use `../wasm/` instead.
