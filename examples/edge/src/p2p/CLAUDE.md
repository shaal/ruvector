# edge / src / p2p

Peer-to-peer transport stack used by the edge swarm. Provides identity, end-to-end crypto, message envelopes, relays, swarm routing, and artifact transfer.

## Important files
- `mod.rs` - module root.
- `identity.rs` - peer identity (Ed25519 keypair, peer-id derivation).
- `crypto.rs` - cryptographic primitives (X25519 ECDH, AES-GCM, HKDF, SHA-256).
- `envelope.rs` - signed/encrypted message envelope format.
- `relay.rs` - relay support for NAT-traversed peers.
- `swarm.rs` - high-level P2P swarm coordination.
- `artifact.rs` - chunked artifact (model / tensor) transfer.
- `advanced.rs` - advanced P2P routines (e.g. multi-hop, sealed delivery).

## Related
- Underlying transport: `../transport.rs`. Higher-level intelligence: `../intelligence.rs`.
