# exo-wasm

WASM bindings for the EXO-AI cognitive substrate so it can run in
browsers / edge runtimes. Builds as both `cdylib` (for
`wasm-bindgen`) and `rlib` (for native consumers wanting the same API
surface).

## Files

- `Cargo.toml` — depends on `ruvector-core` (default-features off,
  `uuid-support`), wasm-bindgen, wasm-bindgen-futures.
- `src/lib.rs` — `#[wasm_bindgen]` exports.
- `src/types.rs` — JS-friendly wrapper types.
- `src/utils.rs` — utility helpers (panic hooks, console logging).
- `examples/browser_demo.html` — minimal browser harness consuming the
  generated JS glue.

## Build

```bash
# Requires wasm-pack
wasm-pack build crates/exo-wasm --target web --release
# Then open examples/browser_demo.html
```

## Related

- `../exo-node/` — sibling Node.js binding
- `../../../wasm-react/` — React-based example using a different WASM
  artifact (`ruvector-wasm`)
- `../../docs/EXAMPLES.md`
