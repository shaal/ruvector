# ruvector-verified-wasm/src

- `lib.rs` — main module. Defines `init`, `version`, and the `#[wasm_bindgen] JsProofEnv` that wraps `ruvector_verified::{ProofEnvironment, cache::ConversionCache, fast_arena::FastTermArena, gated::{ProofKind, ProofTier}, proof_store, vector_types}`.
- `utils.rs` — `set_panic_hook` (via `console_error_panic_hook` style) and `console_log` helper for browser `console.log`.

See `../CLAUDE.md`.
