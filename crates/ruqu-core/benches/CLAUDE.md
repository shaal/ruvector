# ruqu-core/benches

- `quantum_sim.rs` — criterion benchmark suite for the simulator, registered as `[[bench]] name = "quantum_sim"` in `../Cargo.toml`. Covers state-vector evolution, stabilizer ops, and (where compiled) tensor-network contractions.

Run with `cargo bench -p ruqu-core`. See `../CLAUDE.md`.
