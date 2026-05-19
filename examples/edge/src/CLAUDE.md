# edge / src

Rust source tree for `ruvector-edge`.

## Top-level files
- `lib.rs` - crate root, public re-exports.
- `agent.rs` - edge-agent abstraction (identity, intent, messaging).
- `protocol.rs` - wire-level swarm protocol on top of `ruv-swarm-transport`.
- `transport.rs` - transport bridges (WebSocket, shared-memory, GUN).
- `intelligence.rs` - on-device intelligence layer (planning / coordination).
- `memory.rs` - shared memory primitives for swarm coordination.
- `compression.rs` - lz4-based tensor / message compression.
- `gun.rs` - GUN-DB integration (feature `gun`).
- `wasm.rs` - `#[wasm_bindgen]` surface (feature `wasm`).

## Subdirectories
- `bin/` - three CLI binaries (`edge-agent`, `edge-coordinator`, `edge-demo`).
- `p2p/` - P2P transport: crypto, identity, envelope, swarm relay, artifact transfer.
- `plaid/` - the PLAID privacy / local-learning subsystem + Bulletproofs ZK proofs (native and WASM variants).

## Build
- `cargo build -p ruvector-edge --features full`.
