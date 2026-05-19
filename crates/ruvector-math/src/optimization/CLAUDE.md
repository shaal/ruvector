# ruvector-math/src/optimization

Polynomial optimisation and sum-of-squares (SOS) certificates. Provides certifiable optimisation: prove non-negativity of polynomials and certify bounds on attention / routing policies.

- `mod.rs` — re-exports.
- `polynomial.rs` — polynomial representation and arithmetic.
- `sos.rs` — Sum-of-Squares decomposition.
- `sdp.rs` — semidefinite-programming backend for SOS.
- `certificates.rs` — `BoundsCertificate`, `NonnegativityCertificate` (Positivstellensatz-style proofs).

Used by the mincut-governance layer to provide provable guardrails on permission rules and attention-policy stability. See `../CLAUDE.md`.
