# ruvix-proof/tests

## Files

- `security_integration.rs` — covers single-use nonces (no replay), time-bounded validity (proofs expire), capability gating
  (PROVE right enforced), and cache bounds (TTL + max entries) per ADR-087 Section 20.4.
