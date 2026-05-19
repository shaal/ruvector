# docs/architecture/decisions/

Older / parallel ADR namespace for ruvector. Numbering here is independent from `../../adr/` and contains duplicate numbers from different decision tracks (multiple ADR-001s, multiple ADR-003s, etc.). Treat as historical context unless an ADR here is explicitly referenced from current code.

## ADRs (14 total)

- `ADR-001-core-simd-strategy.md`, `ADR-001-simd-first-vector-operations.md` - two takes on SIMD strategy.
- `ADR-002-hyperbolic-embeddings.md` - hyperbolic embedding space.
- `ADR-003-flash-attention.md`, `ADR-003-mcp-protocol.md`, `ADR-008-flash-attention.md` - flash attention and MCP protocol decisions.
- `ADR-004-hnsw-ann.md`, `ADR-004-rvf-format.md` - HNSW choice and the RVF format (superseded by main ADR-029).
- `ADR-005-cross-platform-bindings.md`, `ADR-005-rvf-cognitive-container.md` - bindings and RVF cognitive container.
- `ADR-006-sona-adaptation.md`, `ADR-006-sona-self-optimization.md` - SONA adaptation/self-optimization.
- `ADR-007-differential-privacy.md` - differential privacy.
- `ADR-008-wasm-first-strategy.md` - WASM-first strategy.

## Related

- `../../adr/` - canonical/current ADR series. Prefer reading there first.
- `../` - parent architecture docs.
