# ruvector-economy-wasm/src

Five flat modules implementing the WASM credit economy.

## Files

- `lib.rs` — Crate doc + module declarations + WASM init.
- `ledger.rs` — `CreditLedger` CRDT — G-Counter for credits earned, PN-Counter for net balance.
- `curve.rs` — `contribution_multiplier(network_hours)` — early-adopter decay curve (10x -> 1x).
- `stake.rs` — Participation stake + slashing primitives.
- `reputation.rs` — `ReputationScore` weighted by accuracy, uptime, stake.
