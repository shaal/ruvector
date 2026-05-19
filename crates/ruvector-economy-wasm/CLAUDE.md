# ruvector-economy-wasm

CRDT-based autonomous credit economy for distributed compute networks — WASM-optimized. Provides a P2P-safe credit ledger (G-Counter, PN-Counter), contribution curve (10x early adopter -> 1x baseline), stake/slash mechanics, reputation scoring, and Merkle state-root verification.

## Important files

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Features `qdag`, `reputation`, `full = ["qdag","reputation"]`. Uses `wasm-bindgen`, `js-sys`, `serde`, `rustc-hash`, `sha2`, optional `console_error_panic_hook`.
- `src/lib.rs` — Top-level crate doc with JS quick-start. Declares the five flat modules.

## Source modules (`src/`)

- `ledger.rs` — `CreditLedger` (G-Counter + PN-Counter for P2P credit tracking).
- `curve.rs` — `contribution_multiplier(...)` early-adopter curve.
- `stake.rs` — Stake / slash mechanics for participation.
- `reputation.rs` — `ReputationScore` multi-factor (accuracy, uptime, stake) reputation.
- `lib.rs` — Module declarations + WASM bootstrap.

## Public API (JS surface)

- `class CreditLedger("node-id")` — `.credit(amount, tag)`, `.balance()`.
- `contribution_multiplier(compute_hours)` — early-adopter multiplier.
- `class ReputationScore(accuracy, uptime, stake)` — `.composite_score()`.

## Subdirectory

- `pkg/` — Generated `wasm-pack` output (committed).

## Related

- Built for distributed compute / agent economies inside the broader ruvector ecosystem.
- WASM siblings: `ruvector-attention-wasm`, `ruvector-sparse-inference-wasm`, `ruvector-tiny-dancer-wasm`.
