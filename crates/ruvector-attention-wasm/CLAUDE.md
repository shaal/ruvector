# ruvector-attention-wasm

WebAssembly bindings for high-performance attention mechanisms: Multi-Head, Flash, Hyperbolic, Linear, Local-Global, MoE, and CGT Sheaf attention — with browser GPU acceleration for transformers and LLMs. Wraps `ruvector-attention`.

## Important files

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Depends on `ruvector-attention` (path, `wasm` feature). Release profile `opt-level = "s"`, LTO on.
- `package.json` — npm packaging for the generated bindings.
- `tsconfig.json` — TypeScript config for the JS wrapper.
- `src/lib.rs` — WASM entry, `init()` panic hook, `version()`, `available_mechanisms()`.

## Source modules (`src/`)

- `lib.rs` — Module declarations + bootstrap.
- `attention.rs` — Bindings around the attention operators (scaled dot product, multi-head, flash, hyperbolic, linear, local-global, moe, cgt-sheaf).
- `training.rs` — Training-mode hooks (gradient passes, mask construction).
- `utils.rs` — Helpers shared by attention/training.

## Subdirectories

- `js/` — TS wrapper (`index.ts`, `types.ts`).
- `pkg/` — Generated `wasm-pack` output (committed `.wasm`, `.js`, `.d.ts`, `package.json`).
- `tests/web.rs` — `wasm-bindgen-test` browser tests.

## Public API (JS)

- `init()`, `version()`, `available_mechanisms()` — capability discovery.
- Attention class constructors from `attention.rs` (e.g. `MultiHeadAttention`, `FlashAttention`, etc.).

## Related

- Backbone: `ruvector-attention`.
- Other WASM siblings: `ruvector-economy-wasm`, `ruvector-sparse-inference-wasm`, `ruvector-tiny-dancer-wasm`.
