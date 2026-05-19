# verified-applications/src

Source for the `verified-apps` binary. One Rust module per exotic application.

## Important files
- `lib.rs` / `main.rs` — library + CLI dispatcher.
- `agent_contracts.rs` — verifiable agent contracts.
- `financial_routing.rs` — verifiable financial route selection.
- `legal_forensics.rs` — legal-evidence forensics.
- `medical_diagnostics.rs` — medical-diagnostic recommendations with proofs.
- `quantization_proof.rs` — proof of correct vector quantization.
- `sensor_swarm.rs` — sensor-swarm fusion with provenance.
- `simulation_integrity.rs` — simulation-state integrity proofs.
- `vector_signatures.rs` — signature-over-vector primitives.
- `verified_memory.rs` — verifiable memory store.

## Build
- From parent: `cargo build --release`.
