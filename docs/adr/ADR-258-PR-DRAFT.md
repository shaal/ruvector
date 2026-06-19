<!--
DRAFT pull-request description for submission against the project origin
(ruvnet/ruvector). Copy the body below into the PR. Suggested title:

  [ADR-258] Multiresolution Hash Encoding & Neural Index v2 (Phases 1–3, opt-in)

Suggested base: main   Suggested head: claude/ruvector-hash-encoding-upgrade-1gouzf
-->

## What & why

Adapts the Instant-NGP **multiresolution hash encoding** (Müller et al., SIGGRAPH 2022, [arXiv:2201.05989](https://arxiv.org/abs/2201.05989)) into RuVector's GNN-over-HNSW self-learning loop.

The loop today is **bandwidth-bound**: online InfoNCE updates touch full `d_embed` embeddings and accumulate dense gradients through `MmapGradientAccumulator`. MHE replaces that with **O(L) cache-resident table lookups** whose gradients are *sparse* (only `2^d_idx·L` params/sample), giving the GNN an explicit multi-scale signal aligned with the HNSW layer hierarchy at a fixed memory budget.

Full rationale, alternatives, success criteria, risks, and the living status table are in `docs/adr/ADR-258-Multiresolution-Hash-Encoding-and-Neural-Index-Upgrade.md`.

> Everything here is **opt-in behind the `hashenc` feature flag** (default off). Default builds are unchanged.

## What's in this PR

### New crate `ruvector-hashenc` (dependency-light: `thiserror` only, optional `memmap2`; WASM-friendly)
- `HashEncoder` — projection (`LockedRandom` / `PcaInit` / `Learned`) into a low-`d_idx` index space; hashed multiresolution grid (`L` levels, `T` table size, `F` features) with **dense collision-free coarse levels** + spatial hashing for fine levels; d-linear interpolation.
- `FeatureTables` + `GradAccum` — trainable tables, **sparse-scatter backward**, fused AXPY update, file persistence.
- **Learned projection** (Phase 2) — trainable `P` with full analytic gradient (`projection_grad`).
- `sampling` (Phase 2) — `NegativeSampler` (Random / HnswHard mid-rank / Mixed) + `TemperatureSchedule` (cosine anneal).
- `tiered` (Phase 3) — `TieredFeatureStore` (HOT tables / **WARM int8** reconstruction / COLD) wiring quantization into the live path (spirit of #563); **AVX2 + scalar L2 rerank distance** with a differential-equivalence test.

### GNN integration (`ruvector-gnn`, behind `hashenc`)
- `FeatureSource` trait with `FlatEmbedding` (legacy, zero-overhead default) and `HashAugmented` (concat raw + encoded). `RuvectorLayer::forward` signature unchanged; only `input_dim` grows.
- **`ResidualGatBlock`** (Phase 2) — residual skip + learned edge gain over `MultiHeadAttention` + `LayerNorm`.

### Self-learning validation harness (`ruvector-selflearn`)
- Reproducible online workload on a low-dim latent manifold lifted to high-D with multi-frequency relevance (the regime a linear metric can't capture but a multiresolution grid is built for).
- ≥5 seeds, mean ± 95% CI, Cohen's *d*; emits CSV + ASCII curve + `bench_results/selflearn_REPORT.md`.

## Proof / tests (all green, clippy clean)

- **Differentiability:** finite-difference vs analytic gradient checks for **both** the tables and the learned projection (`tests/gradient_check.rs`).
- **End-to-end learning:** training tables + projection reduces loss >50% (`tests/learning.rs`).
- **Correctness:** partition-of-unity, determinism, dense-coarse, save/load; int8 reconstruction-error bound; **SIMD == scalar** distance; sampler/anneal invariants; residual-block behavior.
- **Phase-1 result (5 seeds):** Recall@10 **+47.3%** (d=1.24), Recall@100 **+21.8%** (d=1.00), encoder overhead **+1.83µs/query (+3.1%)**.

```bash
cargo test -p ruvector-hashenc
cargo test -p ruvector-gnn --features hashenc
cargo run -p ruvector-hashenc --bin ruvector-selflearn --release
cargo check -p ruvector-gnn          # default build unaffected
```

## Scope / safety
- Adds `ruvector-hashenc` to workspace members; no behavior change to existing crates.
- `hashenc` and the new crate are opt-in; the default feature set and existing APIs are untouched.

## Follow-ups (tracked in the issue)
- Rerun the harness against the **live** GNN-over-HNSW index (Phase 2 close-out; gates default promotion on S1 ∧ S3).
- WARM tier on PQ / RaBitQ codes; AVX512 / NEON / wasm gather kernels; async query path (overlap prefetch + encode); EWC drift guard in the harness.

---
🤖 Generated with [claude-flow](https://github.com/ruvnet/claude-flow)
