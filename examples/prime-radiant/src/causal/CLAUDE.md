# prime-radiant/src/causal

Causal reasoning subsystem: structural causal models, do-calculus, counterfactuals, and causal abstraction.

## Files

- `mod.rs` - Module exports.
- `model.rs` (~38KB) - Structural causal model representation.
- `graph.rs` - Causal DAG primitives.
- `do_calculus.rs` - Pearl's do-calculus.
- `counterfactual.rs` - Counterfactual inference.
- `abstraction.rs` - Causal abstraction between models.
- `coherence.rs` - Causal coherence checks.

## Related

- ADR: `../../docs/adr/ADR-005-causal-abstraction.md`.
- Bench: `../../benches/causal_bench.rs`.
- Sibling causal demos: `examples/rvf/examples/causal_atlas.rs`.
