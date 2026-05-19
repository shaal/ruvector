# ruvector-decompiler-wasm

WASM bindings for `../ruvector-decompiler`, exposing the Louvain-based JavaScript bundle decompilation pipeline (parse → graph → partition → infer → witness) to Node.js and browsers.

## Layout

- `Cargo.toml` — `cdylib` + `rlib`. Depends on `ruvector-decompiler` with `wasm` feature, plus `wasm-bindgen`, `serde-wasm-bindgen`, `serde_json`, `console_error_panic_hook`, `getrandom` (with `js` feature). Release profile: `opt-level = "s"`, LTO on.
- `src/lib.rs` — sole source. Exports `init()` (panic hook) and `decompile(source, config_json) -> String` returning a JSON-serialised `DecompileResult` or an `{"error": ...}` object.

## Public API (JS)

- `init()` — call once at module load
- `decompile(source: string, config_json: string) -> string`

## Related

- `../ruvector-decompiler` — pure-Rust pipeline
- `../ruvector-mincut` — Louvain implementation used downstream
