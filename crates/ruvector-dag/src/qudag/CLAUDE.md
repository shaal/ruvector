# ruvector-dag/src/qudag

QuDAG — quantum-resistant distributed pattern learning over a DAG of proposals. Requires `feature = "full"`.

- `mod.rs` — module wiring, re-exports client / consensus / network / proposal / sync; sub-modules `crypto`, `tokens` are public.
- `client.rs` — `QuDagClient` (high-level client into the network).
- `consensus.rs` — `Vote`, `ConsensusResult` and consensus loop.
- `network.rs` — `NetworkConfig`, `NetworkStatus` (transport / peering).
- `proposal.rs` — `PatternProposal`, `ProposalStatus`.
- `sync.rs` — `PatternSync` (peer-to-peer DAG sync).
- `crypto/` — post-quantum / placeholder crypto primitives; see `crypto/CLAUDE.md`.
- `tokens/` — rUv token integration (governance, staking, rewards); see `tokens/CLAUDE.md`.

See `../CLAUDE.md`.
