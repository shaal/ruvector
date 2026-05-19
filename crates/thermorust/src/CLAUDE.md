# thermorust/src

Source for the thermodynamic neural-motif engine.

## Files

- `lib.rs` — top-level crate docs (modules table + quick-start) and module decls.
- `state.rs` — `State { activations, dissipated_joules }`.
- `energy.rs` — `EnergyModel` trait, `Ising`, `SoftSpin`, `Couplings`.
- `dynamics.rs` — `Params`, `step_discrete` (Metropolis–Hastings), `step_continuous` (Langevin), `anneal_discrete`, `anneal_continuous`, `inject_spikes`.
- `noise.rs` — Langevin and Poisson spike noise sources.
- `metrics.rs` — `magnetisation`, `overlap`, `binary_entropy`, free energy, `Trace`.
- `motifs.rs` — `IsingMotif` (ring/fully-connected/Hopfield) and `SoftSpinMotif` pre-wired starters.
