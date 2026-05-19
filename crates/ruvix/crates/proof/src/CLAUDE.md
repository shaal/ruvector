# ruvix-proof/src

## Files

- `lib.rs` — crate root; re-exports `ProofEngine`, `ProofEngineConfig`, `ProofVerifier`, `ProofCache`.
- `engine.rs` — `ProofEngine`: top-level coordinator.
- `verifier.rs` — `ProofVerifier` per-tier verification logic.
- `routing.rs` — 3-tier routing decisions (Reflex / Standard / Deep).
- `cache.rs` — bounded `ProofCache` (max 64 entries, 100ms TTL) with single-use nonce tracking.
- `attestation.rs` — attestation token construction + binding to objects.
- `witness.rs` — witness-log entry types emitted on successful verification.
- `integration.rs` — high-level integration helpers for callers (e.g. syscall dispatch in nucleus).
- `error.rs` — proof error enum.
