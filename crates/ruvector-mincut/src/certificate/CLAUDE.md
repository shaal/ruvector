# ruvector-mincut/src/certificate

Cryptographic / structural proof certificates for min-cut decisions.

- `mod.rs` — `Certificate` type and verification API.
- `audit.rs` — audit log helpers for replaying certificate chains.

Pairs with `src/witness/` (witness chain) — together they form the audit trail consumed by `mcp-gate` / coherence gate clients.
