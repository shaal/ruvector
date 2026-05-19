# ruvllm/src/evaluation

Three-layer evaluation framework:
1. Correctness - does the patch actually work?
2. Diff quality - does it behave like a senior engineer?
3. Systems economics - is it worth running at scale?

Designed for ablation studies (baseline, retrieval-only, adapters-only,
retrieval+adapters, etc.).

## Files
- `mod.rs` - public API + ablation modes + usage docs.
- `harness.rs` - `EvaluationHarness`, `EvalConfig`.
- `real_harness.rs` - production / real-data harness implementation.
- `correctness.rs` - correctness scoring (tests pass / behavior matches).
- `diff_quality.rs` - diff-quality heuristics (style, scope, comments).
- `economics.rs` - cost/performance/throughput scoring.
- `metrics.rs` - shared metric primitives.
- `report.rs` - structured report aggregation + serialization.
