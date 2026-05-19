# rvm-proof

Proof-gated state transitions for the RVM microhypervisor (ADR-135). Every mutation to partition state requires a valid proof that is then recorded in the witness trail.

## Proof tiers

| Tier | Verification | Cost | Use case |
|------|--------------|------|----------|
| `Hash` | SHA-256 preimage | O(1) | Routine transitions |
| `Witness` | Witness-chain verification | O(n) | Cross-partition ops |
| `Zk` | Zero-knowledge proof | Expensive | Privacy-preserving |

`#![no_std] #![forbid(unsafe_code)] #![deny(missing_docs)]`.

## Layout

- `Cargo.toml` — `rlib`; deps `rvm-types`, `rvm-cap`, `rvm-witness`, `spin`, `subtle`; optional `sha2`, `hmac`, `ed25519-dalek`.
- `src/lib.rs` — crate root.
- `src/context.rs` — proof context with builder pattern for P2 validation.
- `src/engine.rs` — unified proof engine (P1 → P2 → witness pipeline).
- `src/policy.rs` — P2 policy rules with constant-time evaluation.
- `src/constant_time.rs` — constant-time comparison utilities.
- `src/signer.rs` — witness signing traits + impls (ADR-142).
- `src/tee.rs` — TEE attestation trait definitions (ADR-142).
- `src/tee_provider.rs` — software TEE quote provider (ADR-142 Phase 3).
- `src/tee_verifier.rs` — software TEE quote verifier (ADR-142 Phase 3).
- `src/tee_signer.rs` — TEE-backed witness-signer pipeline (ADR-142 Phase 3).

See `../CLAUDE.md`.
