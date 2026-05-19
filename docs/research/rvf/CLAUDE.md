# docs/research/rvf/

Research for the **RVF (RuVector Format)** - ruvector's canonical binary format. Contains the spec, wire layout, crypto/signature work, microkernel runtime, profile catalog, and acceptance benchmarks.

## Top-level

- `INDEX.md` - section index.
- `SWARM-GUIDANCE.md` - guidance for swarm-driven implementation.

## Subdirs

- `spec/` - canonical multi-doc spec (segment model, manifests, temperature tiering, progressive indexing, overlays, query optimization, deletion lifecycle, filtered search, concurrency, ops API, WASM bootstrap).
- `wire/` - on-wire binary layout.
- `crypto/` - quantum signatures.
- `microkernel/` - WASM microkernel runtime.
- `profiles/` - domain-specific profiles.
- `benchmarks/` - acceptance benchmarks.

## Related

- `../../adr/` ADR-029..ADR-039 - main RVF ADR run.
- `../../adr/ADR-042-Security-RVF-AIDefence-TEE.md` - TEE/security.
- `../federated-rvf/` - federated extension.
- `../claude-code-rvsource/` - applied RVF use case.
