# ruvix-proof

Proof engine for the RuVix Cognition Kernel (ADR-087) with 3-tier proof routing. Enforces time-bounded validity, single-use
nonces, capability-gated (`PROVE` right) verification, and a bounded proof cache (max 64 entries, 100ms TTL).

## Proof tiers

| Tier | Name | Latency target | Use case |
|---|---|---|---|
| 0 | Reflex | <100ns | High-frequency vector updates |
| 1 | Standard | <100us | Graph mutations with Merkle witness |
| 2 | Deep | <10ms | Full coherence verification with mincut |

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-cap`. Dev: criterion, proptest.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `benches/proof_bench.rs` — per-tier verification latency.
- `tests/security_integration.rs` — security-property integration tests.
