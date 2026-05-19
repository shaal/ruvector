# thermorust

Minimal thermodynamic neural-motif crate. Treats computation as energy-driven state transitions with Landauer-style dissipation and Langevin/Metropolis noise baked in. Provides Ising / soft-spin / Hopfield motifs plus discrete (Metropolis) and continuous (Langevin) dynamics with annealing schedules.

## Layout

- `Cargo.toml` — deps: `rand` (with `small_rng`), `rand_distr`. Dev: `criterion`. Single Criterion bench `motif_bench`. Lints relaxed (research-tier).
- `src/lib.rs` — top-level docs with quick-start example using `IsingMotif::ring` + `anneal_discrete`. Module decls.
- `src/state.rs` — `State` (activation vector + dissipated-joules counter).
- `src/energy.rs` — `EnergyModel` trait + `Ising`, `SoftSpin`, `Couplings`.
- `src/dynamics.rs` — `Params`, `step_discrete` (Metropolis–Hastings), `step_continuous` (Langevin), `anneal_discrete`, `anneal_continuous`, `inject_spikes`.
- `src/noise.rs` — Langevin Gaussian and Poisson spike noise sources.
- `src/metrics.rs` — magnetisation, overlap, binary entropy, free energy, `Trace`.
- `src/motifs.rs` — pre-wired motifs (`IsingMotif`, `SoftSpinMotif`); ring / fully-connected / Hopfield / soft-spin constructors.
- `benches/motif_bench.rs` — Criterion microbenchmarks over the motifs and dynamics.
- `tests/correctness.rs` — invariants (entropy bounds, overlap, magnetisation, spike injection).

## Public API

`State`, `EnergyModel` (+ `Ising`, `SoftSpin`, `Couplings`), `Params`, `step_*` / `anneal_*` / `inject_spikes`, metrics (`magnetisation`, `overlap`, `binary_entropy`, `free_energy`, `Trace`), `IsingMotif`, `SoftSpinMotif`.

## Related

- `../ruvector-coherence`, `../ruvector-mincut` — sibling math/research crates
- `../ruvector-bench` — heavier benchmark harness
