# ruvLLM / docs / SONA

Design documents for the SONA continual-learning subsystem (Self-Organizing Neural Adapters).

## Important files
- `00-OVERVIEW.md` - SONA mission and high-level architecture.
- `01-LORA-ULTRA.md` - LoRA Ultra adapter design.
- `02-LEARNING-LOOPS.md` - instant / background / coordinator learning loops.
- `03-EWC-PLUS-PLUS.md` - Elastic Weight Consolidation++ for catastrophic-forgetting resistance.
- `04-REASONINGBANK.md` - ReasoningBank pattern store.
- `05-MEMORY-DREAMS.md` - sleep-style memory consolidation.
- `06-COMPONENTS.md` - component decomposition.
- `07-IMPLEMENTATION.md` - implementation notes mapping to `../../src/sona/`.
- `08-BENCHMARKS.md` - SONA benchmark methodology (see `../../benches/sona_bench.rs`).
- `09-API-REFERENCE.md` - public SONA API surface.

## Related
- Code: `../../src/sona/`. Benches: `../../benches/sona_bench.rs`. Tests: `../../tests/sona_integration.rs`.
