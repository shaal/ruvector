# ruvix-sched

Coherence-aware scheduler for the RuVix Cognition Kernel (ADR-087 Section 5). Combines three signals for task priority:
(1) deadline pressure (EDF within capability partitions), (2) novelty signal (boost for tasks processing genuinely new
information, measured by vector distance from recent inputs), and (3) structural risk (deprioritize tasks whose pending mutations
would lower the coherence score).

Guarantees: no priority inversion (capability-based access), bounded preemption (only at queue boundaries), partition scheduling
(per-RVF-mount-origin time slices).

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-cap`. Optional `ruvector-coherence` (with `spectral` feature) for coherence
  integration. Dev: criterion.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `benches/scheduler_bench.rs` — scheduling-decision throughput.

## Related

- `../../../ruvector-coherence` — external coherence library consumed via optional feature.
