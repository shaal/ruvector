# prime-radiant/src/coherence

Core coherence computation engine. Implements the formula `E(S) = sum(w_e * |r_e|^2)` where `r_e = rho_u(x_u) - rho_v(x_v)`.

## Files

- `mod.rs` — module entrypoint with the engine architecture diagram.
- `engine.rs` — `CoherenceEngine`: aggregates residuals, holds residual cache, drives full + incremental computation.
- `energy.rs` — energy value object + aggregation helpers.
- `spectral.rs` — spectral analyzer (eigenvalue / Fiedler-style probes).
- `incremental.rs` — delta-based recomputation when only a few edges change.
- `history.rs` — rolling history of energy + decisions for replay / audit.

## Related

- Operates over the substrate from `substrate/`.
- Feeds into `execution/gate.rs` and `governance/`.
