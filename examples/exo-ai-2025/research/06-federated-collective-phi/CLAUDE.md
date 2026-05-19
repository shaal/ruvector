# 06-federated-collective-phi

Standalone research crate: distributed IIT 4.0 consciousness framework
using CRDTs, Byzantine consensus, federated learning, and qualia-level
consensus. Theoretical framework + reference implementation; no benches
or examples.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `federated-collective-phi`. Deps: `serde`, `serde_json`. Dev-dep
  `criterion`.
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`,
  `theoretical_framework.md` — theoretical write-ups.
- `Cargo.lock` — pinned.
- `src/lib.rs` — re-exports.
- `src/distributed_phi.rs` — distributed Phi computation.
- `src/consciousness_crdt.rs` — CRDT for shared consciousness state.
- `src/federation_emergence.rs` — emergence detection across the
  federation.
- `src/qualia_consensus.rs` — qualia-level Byzantine consensus.

## Build

```bash
cd examples/exo-ai-2025/research/06-federated-collective-phi
cargo build --release
cargo test
```

## Related

- `../../crates/exo-federation/` — production federation stack
- `../../../ecosystem-consciousness/` — single-node Phi demo
