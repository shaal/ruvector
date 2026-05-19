# exo-exotic

Research-tier crate collecting "exotic" cognitive experiments built on
top of exo-core + exo-temporal: strange loops, dream cycles, free
energy minimization, morphogenesis, collective consciousness, temporal
qualia, multiple selves, cognitive thermodynamics, emergence detection,
and "cognitive black holes". Most modules are sketches / hypothesis
implementations, not stabilized APIs.

## Files

- `Cargo.toml` — depends on `exo-core`, `exo-temporal`,
  `ruvector-domain-expansion`, serde, thiserror.
- `src/lib.rs` — module re-exports.
- `src/strange_loops.rs`, `src/dreams.rs`, `src/free_energy.rs`,
  `src/morphogenesis.rs`, `src/collective.rs`, `src/temporal_qualia.rs`,
  `src/multiple_selves.rs`, `src/thermodynamics.rs`,
  `src/emergence.rs`, `src/black_holes.rs`, `src/domain_transfer.rs` —
  one experiment per file.
- `src/experiments/` — additional self-contained experiment programs.
- `benches/exotic_benchmarks.rs` — Criterion suite.

## Build / Bench

```bash
cargo build -p exo-exotic
cargo bench -p exo-exotic
```

## Related

- `../../report/EXOTIC_*` — write-ups for these experiments
- `../../research/` — deeper / standalone prototypes for many of the
  same ideas
