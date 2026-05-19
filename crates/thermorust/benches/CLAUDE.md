# thermorust/benches

Criterion microbenches.

## Files

- `motif_bench.rs` — measures `step_discrete`, `anneal_discrete`, `anneal_continuous` over `IsingMotif` and `SoftSpinMotif` instances with seeded RNG. Registered as `[[bench]] name = "motif_bench"` in `../Cargo.toml`.
