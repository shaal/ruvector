<!--
DRAFT GitHub issue for submission against the project origin (ruvnet/ruvector).
Copy the body below into a new issue. This file is a deliverable artifact, not
the issue itself (Issues are disabled on the working fork, and the automation's
GitHub scope is limited to the fork).
-->

# [ADR-258] Multiresolution Hash Encoding & Neural Index v2 — phased rollout

## Summary

Adopt the Instant-NGP **multiresolution hash encoding** (Müller et al., SIGGRAPH 2022, [arXiv:2201.05989](https://arxiv.org/abs/2201.05989)) as a trainable, multi-scale, persistent feature source for RuVector's GNN-over-HNSW self-learning loop.

- **ADR:** `docs/adr/ADR-258-Multiresolution-Hash-Encoding-and-Neural-Index-Upgrade.md`
- **New crate:** `ruvector-hashenc`
- **GNN integration:** behind the `hashenc` feature flag (default off → backward compatible)

## Motivation

The self-learning loop is **bandwidth-bound**: online InfoNCE updates touch full `d_embed` embeddings and accumulate dense gradients through `MmapGradientAccumulator`. Multiresolution hash encoding replaces this with **O(L) cache-resident table lookups** whose gradients are *sparse* (only `2^d_idx·L` params per sample), giving the GNN an explicit multi-scale signal aligned with the HNSW layer hierarchy at a fixed, bounded memory budget.

Two grounding notes:
- `advanced/neural_hash.rs` already exists but is binary LSH (non-differentiable, single-scale). MHE is **additive** — a different, trainable, continuous, interpolated object — not a replacement.
- Quantization is not yet wired into the live index (#563). The tiered-store design addresses that on the neural path.

## Success criteria (statistical bar: ≥5 seeds, 95% CI, Cohen's d ≥ 0.8)

| # | Metric | Target |
|---|---|---|
| S1 | Recall@10 after self-learning | +25–50% rel. |
| S2 | Recall@100 | +15–35% rel. |
| S3 | Convergence (queries to plateau) | 2–3× fewer |
| S4 | QPS (mixed load) | 1.8–3× |
| S5 | p50 latency | → 25–40µs |
| S6 | Memory / 1M vectors | −25–45% |
| S7 | Self-learning overhead | ≤ +15% |

## Phased plan & status

### Phase 1 — Encoder + integration + harness  ✅ (landed, opt-in)
- [x] `ruvector-hashenc` crate: projection → hashed multiresolution grid → d-linear interpolation; dense collision-free coarse levels; trainable tables + sparse-scatter backward; file persistence
- [x] `FeatureSource` / `FlatEmbedding` / `HashAugmented` in `ruvector-gnn` (flag `hashenc`)
- [x] Differentiability proof: finite-difference vs analytic gradient check
- [x] Self-learning harness with recall@K, 95% CI, Cohen's d, CSV + report
- [x] Criterion benches

### Phase 2 — GNN / self-learning upgrades  ✅ (landed, opt-in)
- [x] **Learned projection** (trainable `P`) + gradient check + end-to-end learning test
- [x] **Hard-negative sampler** (`NegativeSampler`: Random / HnswHard / Mixed)
- [x] **Temperature annealing** (`TemperatureSchedule`, cosine)
- [x] **Residual GAT block** with learned edge gain (`ruvector-gnn::residual`)
- [ ] EWC drift guard wired into the harness *(follow-up)*
- [ ] Rerun harness against the **live** GNN-over-HNSW index *(Phase 2 close-out)*

### Phase 3 — Storage / async / full proof  ◑ (partially landed)
- [x] `TieredFeatureStore` (HOT tables / WARM int8 / COLD) + footprint accounting (wires quantization into the live path, spirit of #563)
- [x] SIMD L2 rerank distance (AVX2 + scalar) with differential equivalence test
- [ ] WARM tier on PQ / RaBitQ codes (vs int8) *(follow-up)*
- [ ] Async query path overlapping prefetch + encode *(design only, §6.5)*
- [ ] AVX512 / NEON / wasm gather kernels *(AVX2 done)*
- [ ] Full benchmark suite + comparison report; promote `hashenc` default if S-criteria pass

## Phase-1 validation (5 seeds, from `bench_results/selflearn_REPORT.md`)

| Metric | Baseline | HashEnc | Δ | Effect size |
|---|---|---|---|---|
| Recall@10 (final) | 0.196 | 0.289 | **+47.3%** | Cohen's d = 1.24 |
| Recall@100 (final) | 0.280 | 0.341 | **+21.8%** | Cohen's d = 1.00 |
| Sessions to surpass baseline | — | 1.0 | reaches + exceeds baseline | — |
| Encode cost added per query | — | +1.83 µs | +3.1% of a ~60µs query | — |

## How to reproduce

```bash
cargo test -p ruvector-hashenc                                    # 13 unit + 2 gradient checks + e2e learning
cargo run -p ruvector-hashenc --bin ruvector-selflearn --release  # -> bench_results/selflearn_REPORT.md
cargo test -p ruvector-gnn --features hashenc                     # feature_source + residual GAT (215+ tests)
cargo check -p ruvector-gnn                                       # default build unaffected
```

## Acceptance for default promotion
Rerun the self-learning harness against the live GNN-over-HNSW path and confirm **S1 ∧ S3** with ≥5 seeds / 95% CI / Cohen's d ≥ 0.8, plus S6 memory accounting with the PQ/RaBitQ WARM tier.

---
🤖 Generated with [claude-flow](https://github.com/ruvnet/claude-flow)
