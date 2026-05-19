# edge (ruvector-edge)

Edge AI swarm communication crate. Combines `ruv-swarm-transport` (WebSocket / shared-memory / GUN), an end-to-end P2P stack (Ed25519/X25519/AES-GCM/HKDF), zero-knowledge proofs via Bulletproofs (the "PLAID" subsystem), and three CLI binaries that demo agent-to-agent coordination on the edge. Also compiles to WASM for in-browser swarms.

## Important files
- `Cargo.toml` - sets its own `[workspace]`. Features: `default = ["websocket","shared-memory","native"]`, `wasm`, `gun`, `full`. Three binaries: `edge-agent`, `edge-coordinator`, `edge-demo`. Two cargo examples: `local_swarm`, `distributed_learning`.
- `Cargo.lock` - committed lockfile (standalone workspace).
- `src/` - library modules: `agent.rs`, `compression.rs` (lz4_flex), `gun.rs`, `intelligence.rs`, `lib.rs`, `memory.rs`, `protocol.rs`, `transport.rs`, `wasm.rs`, plus `bin/`, `p2p/`, `plaid/` subdirs.
- `examples/local_swarm.rs`, `examples/distributed_learning.rs` - cargo examples; entry points for the two demos referenced in `Cargo.toml`.
- `benches/zkproof_bench.rs` - Criterion bench for the ZK-proof pipeline.
- `pkg/` - generated `wasm-pack` artifacts plus demo HTML pages (PLAID local learner, ZK financial proofs, worker pool).
- `scripts/build-wasm.sh` - helper that builds the WASM package into `pkg/`.
- `docs/` - PLAID + ZK-optimization writeups.

## Build / run
- Native: `cargo build -p ruvector-edge --release` then `./target/release/edge-demo`.
- WASM: `bash scripts/build-wasm.sh` (output lands in `pkg/`).
- Bench: `cargo bench -p ruvector-edge`.

## Related
- Used by OSpipe's broader screenpipe story (`../OSpipe/`). Pairs with `../delta-behavior/` for coherence-bounded swarm behaviour.
