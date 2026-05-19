# exo-federation/src

Federation engine sources.

## Files

- `lib.rs` — public surface re-exports.
- `crdt.rs` — base CRDT types (registers, sets, counters).
- `transfer_crdt.rs` — CRDT tailored for substrate state transfer.
- `consensus.rs` — Byzantine-style consensus.
- `coherent_commit.rs` — coherent commit across replicas.
- `handshake.rs` — post-quantum key exchange.
- `crypto.rs` — signing / hashing primitives.
- `onion.rs` — layered routing.

## Related

- `../tests/federation_test.rs`
- `../../../docs/SECURITY.md`, `../../../docs/SECURITY_AUDIT_REPORT.md`
