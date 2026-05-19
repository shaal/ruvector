# ruvector-decompiler-wasm/src

Sole source dir.

## Files

- `lib.rs` — thin wasm-bindgen shim around `ruvector_decompiler::decompile`. Parses caller-supplied `config_json` into `DecompileConfig` (defaulted on parse failure), runs the pipeline, returns JSON string. `init()` installs `console_error_panic_hook`.
