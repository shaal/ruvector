# verified-applications

Showcases 10 "exotic" applications of `ruvector-verified` — from weapons-content filters to legal forensics — each producing cryptographically verifiable outputs (HNSW proofs, vector signatures, ultra-precision).

## Important files
- `Cargo.toml` — bin `verified-apps`; depends on `ruvector-verified` with `ultra` + `hnsw-proofs` features.
- `src/lib.rs` — shared API.
- `src/main.rs` — CLI that dispatches into the 10 application modules below.
- App modules: `agent_contracts.rs`, `financial_routing.rs`, `legal_forensics.rs`, `medical_diagnostics.rs`, `quantization_proof.rs`, `sensor_swarm.rs`, `simulation_integrity.rs`, `vector_signatures.rs`, `verified_memory.rs`.

## Run
- `cargo run --release --bin verified-apps -- <app>` (consult `main.rs` for the per-app argument shape).

## Tech stack
- `../../crates/ruvector-verified` (features `ultra`, `hnsw-proofs`), `rand`, `anyhow`.

## Related
- General intelligence harnesses in `../benchmarks/` (esp. `acceptance-rvf`, `agi-proof-harness`).
