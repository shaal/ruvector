# ADR-258: Multiresolution Hash Encoding and Neural Index Upgrade (RuVector Neural Index v2)

- **Status:** Proposed
- **Date:** 2026-06-18
- **Deciders:** RuVector Core / GNN / Performance working groups
- **Supersedes / relates to:** ADR-001 (ruvector-core architecture), ADR-003 (SIMD strategy), ADR-006 (memory management), ADR-027 (HNSW parameterized query), ADR-033 (progressive indexing), ADR-046–055 (graph-transformer / graph layers)
- **Primary reference:** T. Müller, A. Evans, C. Schied, A. Keller, *"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"*, SIGGRAPH 2022, arXiv:2201.05989

> **Numbering note:** The task brief requested the filename `ADR-001-…`. `ADR-001` is already allocated to *ruvector-core-architecture* and the ADR series runs through ADR-257; this document is filed as **ADR-258** to avoid collision while preserving the requested descriptive title.

---

## 1. Context and Problem Statement

RuVector is a Rust-native, self-learning vector database whose central thesis is that **the index itself is a neural network**: an HNSW proximity graph carries a Graph Neural Network (GNN) overlay (`ruvector-gnn`) that performs message passing, attention-weighted aggregation, and GRU updates, trained online with InfoNCE contrastive learning using HNSW neighbors as positives. Node embeddings and their gradients are persisted through memory-mapped files (`MmapManager`, `MmapGradientAccumulator`), so every query is a forward pass and the system "gets smarter with usage."

Three structural limitations cap how much the system can learn and how cheaply it can serve:

1. **Node features are flat and single-scale.** Each node carries one embedding vector (`get_embedding(node_id) -> &[f32]`). The GNN's only multi-scale signal is the HNSW layer hierarchy itself (`hierarchical_forward`). There is no compact, trainable, *multi-resolution* feature representation per node, so the model must encode coarse-and-fine structure inside one dense vector — which is both slow to learn (dense gradient over the whole vector on every step) and memory-heavy.

2. **Learning is bandwidth-bound, not compute-bound.** Online updates touch full `d_embed`-wide embeddings and accumulate dense gradients through `MmapGradientAccumulator` (64-node lock granularity). Convergence of the self-learning loop is therefore gated by how much memory traffic each contrastive step incurs, not by arithmetic. This is exactly the regime where Müller et al.'s multiresolution hash encoding wins: it replaces large dense MLP/feature work with **O(L) tiny, cache-resident table lookups** whose gradients are sparse.

3. **Quantization and feature storage are not unified with the learned representation.** `ruvector-core` ships PQ/OPQ/scalar/int4/binary quantizers and `ruvector-rabitq`, but per issue #563 quantization is *not yet applied to the live index*; `ruvector-gnn::compress` tiers embeddings by access frequency independently. There is no single representation that is simultaneously (a) compact, (b) trainable, and (c) tier-able.

**Problem statement:** *How do we give RuVector a compact, trainable, multi-scale node representation that accelerates the self-learning loop, improves recall and convergence, reduces memory per vector, and lowers query latency — without breaking the persistent-differentiable, mmap-backed, WASM/Postgres-portable design?*

---

## 2. Decision

Adopt a **Multiresolution Hash Encoding (MHE)** representation — adapted from Instant-NGP to the high-dimensional vector-retrieval setting — as a first-class, trainable feature source for the GNN, implemented in a new crate **`ruvector-hashenc`** and integrated behind feature flags into `ruvector-gnn` and `ruvector-core`.

Concretely we will:

1. **Build `ruvector-hashenc`** providing trainable multiresolution feature tables with configurable levels `L`, per-level table size `T`, and feature width `F`, plus a fast forward/backward path with SIMD-accelerated d-linear interpolation. Tables persist via the *same* mmap pattern already used for embeddings/gradients, preserving persistent differentiability.

2. **Adapt MHE to high-dim inputs** via a small learned/locked projection `P: ℝ^d → ℝ^{d_idx}` (`d_idx ∈ {2,3,4}`) into an "index space," where a standard multiresolution hash grid with d-linear interpolation is cheap (2^{d_idx} corners per level). The concatenated encoding `enc(x) ∈ ℝ^{L·F}` augments (does not replace) each node's base embedding feeding `RuvectorLayer::forward`.

3. **Upgrade the GNN + self-learning loop**: residual GAT-style attention over MHE-augmented features, hard-negative mining from mid-rank HNSW candidates, temperature-annealed InfoNCE, and temporally-weighted experience replay — all reusing existing `Optimizer`, `LearningRateScheduler`, `ReplayBuffer`, `ElasticWeightConsolidation`.

4. **Unify storage tiering**: MHE tables become the "hot, learnable" tier; full embeddings/PQ/RaBitQ/OPQ become "warm/cold" reconstruction tiers under one `TieredFeatureStore`, finally wiring quantization into the live retrieval path (closes the spirit of #563 for the neural path).

5. **Keep everything backward-compatible** via a `hashenc` feature flag and a `FeatureSource` trait so the legacy flat-embedding path is the default until benchmarks justify promotion.

This is deliberately *additive*: it does not remove the existing `DeepHashEmbedding`/`SimpleLSH` (binary LSH for coarse bucketing) — MHE is a different object (trainable, continuous, multi-scale, interpolated) serving a different role (learned node features), and the two compose.

---

## 3. Why MHE Fits RuVector (Mechanism → Benefit Mapping)

| Instant-NGP property | Mechanism | RuVector benefit |
|---|---|---|
| Multiresolution feature grid (`L` levels, geometric resolution growth `b = exp((ln N_max − ln N_min)/(L−1))`) | Coarse levels capture global structure collision-free; fine levels capture local detail | Directly mirrors HNSW's coarse-to-fine layer hierarchy; gives the GNN an explicit, separable multi-scale signal instead of one entangled vector |
| Hash table per level, size `T` (2¹⁴–2²⁴) | Spatial hash `h(x)=(⊕ᵢ xᵢπᵢ) mod T`, π=(1, 2654435761, 805459861) | Fixed, tiny memory budget independent of N at fine levels; predictable footprint per 1M vectors |
| d-linear interpolation over 2^{d_idx} corners | Smooth, differentiable lookup | Differentiable end-to-end → fits InfoNCE + persistent gradient accumulation already in `training.rs` |
| **Sparse gradients** — only the 2^{d_idx}·L touched entries per sample get gradient | Gradient averaging across colliding entries | Online step touches ≪ `d_embed` parameters → self-learning becomes compute-light and **bandwidth-light**, the current bottleneck |
| Implicit collision handling | Gradients of colliding points average; network disambiguates via concatenated multi-scale context | No explicit collision resolution code; robust under churn/inserts |
| Cache/bandwidth efficiency | Lookups are O(L) small contiguous reads | Aligns with `#[repr(align(64))]` SoA layout and `madvise` prefetch already in `MmapManager` |

**Portability beyond graphics.** Instant-NGP encodes 2D/3D coordinates. Retrieval embeddings are 384–1536-D, so a dense grid (2^d corners) is infeasible. The adaptation is the **learned projection into a low-`d_idx` index space** before hashing — preserving cheap interpolation and the sparse-gradient property while letting the projection learn *which* directions deserve multi-scale resolution. This is the key engineering insight of this ADR.

---

## 4. Alternatives Considered

### A. Status quo — keep growing dense embeddings
- **Pros:** zero new code; simplest.
- **Cons:** dense gradients keep the self-learning loop bandwidth-bound; memory scales linearly with `d_embed`; no real multi-scale. Rejected — does not move any target metric.

### B. Pure LSH / binary hashing expansion (extend `neural_hash.rs`)
- **Pros:** already present; very cheap; great for coarse bucketing.
- **Cons:** binary codes are not smoothly differentiable, lose magnitude information, and don't give multi-scale continuous features for the GNN. Kept as a *complementary* coarse filter, not the learned feature source. Rejected as the primary mechanism.

### C. Learned index / RMI over the embedding space (extend `learned_index.rs`)
- **Pros:** can predict candidate positions, speeding traversal.
- **Cons:** RMI predicts *positions on a sorted key*, not a *trainable feature representation*; brittle under online inserts; 1-D key assumption. Complementary (can consume MHE features as input) but not a substitute. Rejected as primary.

### D. Bigger/deeper GNN (more layers, wider hidden dim)
- **Pros:** more capacity.
- **Cons:** increases per-query FLOPs and gradient traffic — moves latency and convergence in the *wrong* direction; worsens the bandwidth bottleneck. Rejected.

### E. **Multiresolution Hash Encoding with low-`d_idx` projection (CHOSEN)**
- **Pros:** compact, trainable, multi-scale, sparse-gradient, cache-friendly, differentiable, mmap-persistable; composes with B/C; directly attacks the bandwidth bottleneck; bounded memory.
- **Cons:** new crate + integration surface; projection `P` adds a design choice; hash collisions need empirical tuning of `T`/`L`; requires careful SIMD for the gather/scatter. Accepted — risks are bounded and mitigated (§9).

### F. Dense multiresolution grid (no hashing) in index space
- **Pros:** no collisions.
- **Cons:** memory `O(N_max^{d_idx})` explodes even at `d_idx=3`. Rejected — hashing is precisely what makes it tractable.

---

## 5. Decision Drivers / Measurable Success Criteria

The decision is validated only if the self-learning harness (§ proof framework, and `ADR` companion in `crates/ruvector-bench`) demonstrates, with statistical significance (≥5 seeds, 95% CI, Cohen's *d* ≥ 0.8 vs. baseline), the following on SIFT1M / GIST1M / a synthetic agent-memory workload:

| # | Metric | Baseline (current) | Target (v2) | How measured |
|---|---|---|---|---|
| S1 | Recall@10 after self-learning | reference run | **+25–50% relative** | harness recall curve, final plateau |
| S2 | Recall@100 after self-learning | reference run | +15–35% relative | harness recall curve |
| S3 | Self-learning convergence | queries to reach 90% of plateau recall | **2–3× fewer** | convergence detector (§proof) |
| S4 | Query throughput (mixed read/learn) | reference QPS | **1.8–3× QPS** | criterion + harness mixed load |
| S5 | p50 / p99 latency | ~61µs p50 (claimed) | **p50 → 25–40µs** | criterion, warm cache |
| S6 | Memory per 1M vectors | reference | **−25–45%** | RSS + on-disk mmap accounting |
| S7 | Self-learning overhead | added latency on a learning query | **≤ +15%** vs read-only query | harness instrumentation |

All numbers are reported **before vs after** with effect sizes; a result that fails S1+S3 (the core learning thesis) blocks promotion regardless of perf wins.

---

## 6. Architecture (RuVector Neural Index v2)

### 6.1 New crate: `ruvector-hashenc`

```
crates/ruvector-hashenc/
  Cargo.toml
  src/
    lib.rs            // public API, HashEncoder, HashEncConfig, FeatureSource impl
    config.rs         // HashEncConfig (L, T, F, d_idx, N_min, N_max), defaults
    grid.rs           // level resolutions, geometric growth, corner enumeration
    hash.rs           // spatial hash (XOR of coord*prime mod T), per-level
    interp.rs         // d-linear interpolation forward + analytic backward
    projection.rs     // P: R^d -> R^{d_idx} (locked random / learned / PCA-init)
    tables.rs         // FeatureTables: in-memory + mmap-backed, trainable
    backward.rs       // sparse gradient scatter into tables + projection grad
    simd.rs           // AVX2/AVX512/NEON gather + interpolation kernels
    persist.rs        // mmap layout + header, reuse MmapManager conventions
    wasm.rs           // wasm32 path (no mmap; in-memory tables)
```

**Core config and defaults** (mirrors Instant-NGP Table 1, retuned for retrieval):

```rust
// config.rs
#[derive(Clone, Debug)]
pub struct HashEncConfig {
    pub levels: usize,        // L   — default 16
    pub features_per_level: usize, // F — default 2
    pub log2_table_size: u8,  // log2(T) — default 19 (T = 524_288)
    pub index_dims: usize,    // d_idx — default 3 (2^3 = 8 corners)
    pub n_min: u32,           // coarsest resolution — default 16
    pub n_max: u32,           // finest resolution — default 4096 (data-scaled)
    pub projection: ProjectionKind, // LockedRandom | Learned | PcaInit
}

impl Default for HashEncConfig {
    fn default() -> Self {
        Self { levels: 16, features_per_level: 2, log2_table_size: 19,
               index_dims: 3, n_min: 16, n_max: 4096,
               projection: ProjectionKind::PcaInit }
    }
}

impl HashEncConfig {
    /// Geometric per-level growth factor b = exp((ln N_max - ln N_min)/(L-1)).
    pub fn growth(&self) -> f32 {
        ((self.n_max as f32).ln() - (self.n_min as f32).ln())
            / (self.levels.max(2) as f32 - 1.0)
    }
    /// Resolution at level l: floor(N_min * b^l).
    pub fn resolution(&self, level: usize) -> u32 {
        let b = self.growth().exp();
        ((self.n_min as f32) * b.powi(level as i32)).floor() as u32
    }
    /// Output width fed to the GNN: L * F.
    pub fn output_dim(&self) -> usize { self.levels * self.features_per_level }
    pub fn table_size(&self) -> usize { 1usize << self.log2_table_size }
}
```

**Spatial hash** (Instant-NGP eq.; primes for `d_idx ≤ 7`):

```rust
// hash.rs
const PRIMES: [u32; 7] = [1, 2_654_435_761, 805_459_861,
                          3_674_653_429, 2_097_192_037, 1_434_869_437, 2_165_219_737];

#[inline(always)]
pub fn spatial_hash(corner: &[u32], log2_t: u8) -> usize {
    let mut h: u32 = 0;
    for (i, &c) in corner.iter().enumerate() {
        h ^= c.wrapping_mul(PRIMES[i]);
    }
    (h as usize) & ((1usize << log2_t) - 1)   // mod T (T is power of two)
}
```

**Forward encode** (per query/node):

```rust
// lib.rs
pub struct HashEncoder {
    cfg: HashEncConfig,
    projection: Projection,        // R^d -> R^{d_idx}
    tables: FeatureTables,         // L tables, each [T, F], mmap-backed
}

impl HashEncoder {
    /// enc(x): returns L*F features. Records corner/weight cache for backward.
    pub fn encode(&self, x: &[f32], cache: &mut EncodeCache) -> SmallVec<[f32; 64]> {
        let p = self.projection.apply(x);              // d_idx coords in [0,1)
        let mut out = SmallVec::new();
        for l in 0..self.cfg.levels {
            let res = self.cfg.resolution(l) as f32;
            let scaled: ArrayVec<f32, 7> = p.iter().map(|&v| v * res).collect();
            // d-linear interpolation over 2^{d_idx} corners (simd.rs gathers F-wide)
            let feat = interp::dlinear(&self.tables, l, &scaled, &self.cfg, cache);
            out.extend_from_slice(&feat);              // F values
        }
        out                                            // length L*F
    }
}
```

**Backward** (sparse scatter — the cheap part): for each level, only the `2^{d_idx}` touched rows receive `∂L/∂feat · interp_weight`; the projection (if `Learned`) receives a small dense gradient through the chain rule. These scatter into a `MmapGradientAccumulator`-style structure so persistence and the existing `apply(lr, …)` flow are reused unchanged.

### 6.2 Integration with the GNN (`ruvector-gnn`)

A `FeatureSource` trait lets `RuvectorLayer` consume either the legacy flat embedding or the MHE-augmented feature, chosen by feature flag/config:

```rust
// ruvector-gnn/src/feature_source.rs  (new)
pub trait FeatureSource: Send + Sync {
    fn node_features(&self, node_id: u64, raw: &[f32]) -> Cow<'_, [f32]>;
    fn out_dim(&self) -> usize;
}

pub struct FlatEmbedding;                 // legacy: returns raw, dim = d_embed
pub struct HashAugmented {                // new: concat(raw_or_quantized, enc(raw))
    encoder: Arc<HashEncoder>,
    include_raw: bool,                    // concat strategy
}
```

`RuvectorLayer::forward(node_embedding, neighbor_embeddings, edge_weights)` is unchanged in *signature*; `input_dim` becomes `d_embed' = (include_raw ? d_embed : 0) + L·F`. The `w_msg` Linear is sized accordingly at construction. This is the **only** structural change to the layer; attention, GRU, LayerNorm, dropout are untouched.

### 6.3 GNN / self-learning upgrades

- **Residual GAT-style attention.** Add a residual skip around `MultiHeadAttention` (`out = norm(x + attn(x))`) and an additive edge-bias term so attention can up/down-weight HNSW edges by learned affinity (extends existing `MultiHeadAttention` + `edge_weights`).
- **Hard-negative mining.** Today `info_nce_loss` takes random negatives. Add a sampler that draws negatives from **mid-rank HNSW candidates** (ranks `k+1 … ef`) — semantically "near but wrong" — plus a fraction of in-batch negatives. New `NegativeSampler` enum: `Random | HnswHard { band: (usize,usize) } | Mixed`.
- **Temperature annealing.** Wire `LearningRateScheduler`'s pattern into a `temperature` schedule (cosine from 0.2 → 0.05) — sharper distinctions as training matures.
- **Temporally-weighted replay.** `ReplayEntry` already carries `timestamp`; weight replay sampling by recency (exponential decay) so the index tracks workload drift (`detect_distribution_shift` already exists to trigger replay-rate increases).
- **EWC guard.** When `detect_distribution_shift` fires above threshold, `consolidate()` MHE tables + projection to resist catastrophic forgetting of stable structure.

### 6.4 Tiered storage unification (`TieredFeatureStore`)

```rust
// ruvector-gnn/src/tiered_store.rs (new) — composes existing pieces
pub struct TieredFeatureStore {
    hash_tables: FeatureTables,         // HOT  — trainable MHE (mmap)
    raw_or_pq: MmapManager,             // WARM — full f32 or PQ/RaBitQ codes
    cold: Option<FeatureStorage>,       // COLD — block-aligned cold_tier.rs
    policy: TierPolicy,                 // by access_freq (reuse compress.rs levels)
}
```

- **HOT:** MHE tables — small, always resident, trainable.
- **WARM:** reconstruction tier — full `f32`, or PQ/OPQ/**RaBitQ** codes (wires `ruvector-rabitq` / `EnhancedPQ` into the live path, addressing #563 for the neural route). MHE features carry the *learned* signal; the WARM tier supplies *exact-ish* reconstruction for final rerank.
- **COLD:** existing `FeatureStorage` (page-aligned, prefetchable) for >RAM graphs.

### 6.5 Async query path

Wrap the read path in an async executor so HNSW traversal I/O (`madvise` prefetch + mmap page faults) overlaps with MHE table gathers: `prefetch(neighbor_ids)` is issued, MHE `encode` runs on already-resident projection while pages arrive, then the GNN forward consumes both. Exposed as `query_async` returning the existing `QueryResult`.

### 6.6 Portability

- **WASM:** `ruvector-hashenc/src/wasm.rs` keeps tables in-memory (no mmap), same forward/backward. `L·F` small → fits WASM memory budgets; gathers use scalar/`v128` SIMD.
- **Postgres:** MHE tables serialize through the existing snapshot path (`ruvector-snapshot`); the encoder is pure-functional given tables, so the `ruvector-postgres` extension only needs the table blob + config.

---

## 7. Consequences

### Positive
- **Faster, cheaper self-learning** (sparse gradients) — directly targets S3/S7.
- **Multi-scale features** improve recall/convergence (S1/S2) by giving the GNN separable coarse/fine signal aligned with HNSW layers.
- **Bounded, lower memory** (S6): MHE table budget is `L·T·F·4` bytes *total* (config-fixed, ~`16·524288·2·4 ≈ 64 MB` at defaults, *shared across all vectors*) replacing per-vector dense growth; WARM tier can be PQ/RaBitQ-compressed.
- **Unified tiering** finally puts quantization on the live neural path.
- **Backward compatible** via `FeatureSource` + feature flags; legacy path remains default.

### Negative / costs
- New crate + ~3–5 integration points; larger build surface and a new feature-flag matrix to test.
- Hash collisions introduce a tuning dimension (`T`, `L`, `d_idx`); bad settings degrade fine-level fidelity.
- The projection `P` is a new failure mode (a poor projection starves all levels); mitigated by PCA-init + optional learning.
- SIMD gather/scatter is intrinsics-heavy and must be correct across AVX2/AVX512/NEON/wasm — test burden.

### Neutral
- Adds a dependency between `ruvector-gnn` and `ruvector-hashenc` (one-directional, clean).
- Existing `DeepHashEmbedding`/`learned_index` remain; they may later consume MHE features as inputs.

---

## 8. Validation Plan (summary; full framework in companion harness)

1. **Unit / property tests** (`proptest`): hash determinism, interpolation partition-of-unity (`Σ weights = 1`), gradient check (finite-difference vs analytic backward) within `1e-3`.
2. **Criterion microbenchmarks**: `encode` throughput, gather kernels per ISA, end-to-end query p50/p99 (S4/S5).
3. **Self-learning simulation harness** (`crates/ruvector-bench`, new `selflearn` subcommand): inserts a dataset, runs N "sessions" of queries with simulated relevance feedback, logs recall@{10,100}, convergence, latency, RSS per session; emits CSV + plotters PNG + ASCII curve.
4. **Statistical rigor**: ≥5 seeds, report mean ± 95% CI (t-interval), Cohen's *d* baseline-vs-v2; gate promotion on S1∧S3.
5. **Ablations**: `d_idx ∈ {2,3,4}`, `L ∈ {8,12,16}`, `log2_T ∈ {16,19,22}`, projection kind, hard-neg band — to produce a defensible default.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Hash collisions degrade fine-level recall | Med | Med | Tune `T`/`L` via ablation; coarse levels are collision-free and carry global structure; concat with WARM-tier reconstruction for final rerank |
| Poor projection `P` starves encoding | Med | High | PCA-init from a data sample; optional learned `P` with small LR; fall back to `LockedRandom` proven in NGP-style setups |
| SIMD gather/scatter bugs across ISAs | Med | High | Scalar reference path + differential tests; `proptest` gradient check; per-ISA criterion correctness asserts |
| Online learning instability / forgetting | Med | Med | Reuse `EWC`, temperature annealing, gradient clipping (`Loss::MAX_GRAD` already present), temporally-weighted replay |
| Memory regression if WARM kept as full f32 | Low | Med | Default WARM to PQ/RaBitQ once #563 path is on; accounting test in harness gates S6 |
| Scope creep across many crates | Med | Med | Phased rollout (§ roadmap); Phase 1 lands behind a flag with the harness before any default change |
| Build/feature-flag matrix explosion | Med | Low | `hashenc` single gate; CI matrix limited to {default, hashenc, wasm, hashenc+wasm} |

---

## 10. Phased Rollout (see companion roadmap for effort/risk)

- **Phase 1 (high-ROI):** `ruvector-hashenc` crate (config, hash, interp, tables, scalar+AVX2 SIMD, mmap persist), `FeatureSource` integration into `RuvectorLayer`, gradient-check tests, criterion `encode` benches. Behind `hashenc` flag, default off.
- **Phase 2:** GNN/self-learning upgrades (residual GAT, hard negatives, temperature anneal, temporal replay, EWC guard); self-learning harness + statistical reporting; first S1–S3/S7 results.
- **Phase 3:** `TieredFeatureStore` (PQ/RaBitQ on WARM), async query path, AVX512/NEON/wasm kernels, full benchmark suite + comparison report; promote default if S-criteria pass.

---

## 11. Decision Outcome

**Accepted (Proposed → pending Phase-1 harness results).** Promotion to default is contingent on the self-learning harness meeting S1 and S3 with the stated statistical bar; perf criteria (S4–S6) inform default config selection but do not alone justify promotion.
</content>
</invoke>
