# exo-federation

Federated cognitive mesh for the EXO-AI substrate: CRDTs for shared
state, Byzantine-style consensus, post-quantum cryptographic handshake,
and onion-routed transport. Lets multiple substrates coordinate without
a central authority.

## Files

- `Cargo.toml` — depends on `exo-core`, `ruvector-domain-expansion`,
  tokio (full), serde, serde_json.
- `src/lib.rs` — module re-exports.
- `src/crdt.rs` — base CRDT primitives.
- `src/transfer_crdt.rs` — CRDT specialized for cross-domain transfer.
- `src/consensus.rs` — Byzantine-style consensus engine.
- `src/coherent_commit.rs` — coherent multi-replica commit protocol.
- `src/handshake.rs` — post-quantum handshake.
- `src/crypto.rs` — crypto primitives.
- `src/onion.rs` — onion-routed messaging.
- `tests/federation_test.rs` — end-to-end mesh tests.

## Build / Test

```bash
cargo build -p exo-federation
cargo test  -p exo-federation
```

## Related

- `../../research/06-federated-collective-phi/` — research-tier
  prototype of distributed Phi
- `../exo-backend-classical/` — uses this crate for distributed runs
