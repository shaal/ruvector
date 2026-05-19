# exo-node

Node.js bindings for the EXO-AI cognitive substrate via NAPI-RS.
Produces a `cdylib` loadable from Node so JS/TS consumers can drive the
classical backend without re-implementing the trait set.

## Files

- `Cargo.toml` — depends on `exo-core`, `exo-backend-classical`, and
  `napi` with features `napi9`, `async`, `tokio_rt`. `crate-type =
  ["cdylib"]`.
- `src/lib.rs` — `#[napi]` exports for the substrate API.
- `src/types.rs` — JS-friendly wrappers around exo-core types.

## Build

```bash
# Build the cdylib (use napi-rs CLI for a real npm-publishable artifact)
cargo build -p exo-node --release
```

NAPI-RS scaffolding (`package.json`, `napi.config`, `index.d.ts`) is
not checked in here — this crate is the Rust half only.

## Related

- `../exo-backend-classical/` — backend exposed to Node
- `../exo-wasm/` — sibling binding for browsers/edge
