# ruvector-acorn-wasm

WebAssembly bindings for `ruvector-acorn` (predicate-agnostic filtered HNSW). Exposes
`AcornIndex` as a JS-friendly class for browsers, Cloudflare Workers, Deno, and Bun.

## Layout

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Depends on `ruvector-acorn`,
  `wasm-bindgen`, `js-sys`, optional `console_error_panic_hook`. `[profile.release]
  opt-level = "s", lto = true`.
- `build.sh` — wasm-pack build script.
- `src/lib.rs` — `AcornIndex` class, `SearchResult` struct, `init()` panic hook entry.

## Public JS API

```js
import init, { AcornIndex } from "@ruvector/acorn-wasm";
await init();
const idx = AcornIndex.build(vectors, dim, 2);  // gamma=2 = ACORN-gamma
idx.search(query, k, (id) => id % 2 === 0);
```

Returns `[{id, distance}, ...]`. `SearchResult` mirrors `@ruvector/rabitq-wasm`.

## Related

- `crates/ruvector-acorn` — native Rust implementation.
