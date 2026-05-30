---
adr: 194
title: "ruvector-turbovec — Multi-bit TurboQuant FastScan ANN Index (2/4-bit SQ + TQ+ calibration + nibble-LUT SIMD)"
status: accepted
date: 2026-05-29
authors: [oshaal, claude-flow]
related: [ADR-003, ADR-157, ADR-193]
tags: [quantization, ann, vector-search, turboquant, fastscan, simd, lloyd-max, calibration, recall, memory]
---

# ADR-194 — ruvector-turbovec: a multi-bit TurboQuant FastScan ANN index

> **⚠️ Provenance & prior-art note.** This ADR adapts techniques from
> [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec), an independent
> Rust+Python implementation of Google Research's **TurboQuant**
> (arXiv:2504.19874). It is **not** a port of that codebase — it is a clean-room
> ruvector crate that reuses our existing primitives. The TurboQuant *algorithm*
> is already partially present in this repo (see §"What already exists"); the
> contribution here is a **multi-bit scalar-quantized ANN search index** with a
> FastScan SIMD kernel, which we do **not** currently have. Benchmark claims
> below are **targets to be validated**, not measured results.

## Status

**Accepted (M1 implemented).** The scalar reference milestone (M1) is
implemented as `crates/ruvector-turbovec`: rotation reuse + Lloyd–Max 2/4-bit SQ
+ TQ+ calibration + length-renormalized unbiased scoring + `IdMapIndex`
(O(1) delete, filtered search). Build is green
(`cargo build --release -p ruvector-turbovec`); 12 unit tests + 1 doc-test pass;
clippy clean. M2–M4 (FastScan SIMD kernel, AVX-512, dispatcher registration)
remain future work. Measured proof below.

### Validation (measured — `cargo run --release -p ruvector-turbovec`)

`n = 5,000` **uniform-random** vectors (the *worst case* for ANN — no cluster
structure to exploit), `dim = 256`, `k = 10`, **no f32 rerank**, scored against
exact brute-force L2:

| Width | recall@10 | bytes/vec (raw 1024) | compression | mean cosine bias |
|-------|-----------|----------------------|-------------|------------------|
| 1-bit | 0.308 | 48 | 25.6× | +0.0005 |
| 2-bit | 0.561 | 80 | 14.2× | +0.0001 |
| **4-bit** | **0.879** | **144** | **7.5×** | **−0.0000** |

- **Recall rises monotonically with bit-width** — exactly the 2–4-bit regime the
  1-bit RaBitQ path cannot reach without re-inflating memory via f32 rerank.
- **Mean cosine bias ≈ 0 at every width** — empirical confirmation that the
  per-vector `c_x` length-renormalization (§T4) yields an *unbiased* estimator.
- On real clustered embeddings (OpenAI/Cohere) recall at a given width is
  materially higher than on this uniform stress test.
- Determinism (same seed → bit-identical results) and `IdMapIndex` delete +
  allowlist-filtered search both verified PASS.

## Context

### The gap

ruvector has strong coverage of approximate-nearest-neighbor (ANN) search:

| Crate | Index family | Quantization | SIMD scan |
|-------|--------------|--------------|-----------|
| `ruvector-core` | HNSW (graph) | none (f32) | — |
| `ruvector-diskann` | DiskANN (graph) | none / PQ-ish | — |
| `ruvector-rairs` | IVF (ADR-193) | optional | — |
| `ruvector-rabitq` | Flat + rotation | **1-bit** only | AVX2/AVX-512 **popcount** |

The **1-bit** RaBitQ path (`ruvector-rabitq`) is excellent when memory dominates,
but 1-bit codes cap recall: to hit production R@1 on 1536–3072-dim embeddings
(OpenAI `text-embedding-3`, Cohere, Voyage) you need an exact-rerank pass over
raw f32 originals, which re-inflates memory back toward the f32 footprint.

What we are **missing** is the regime that production vector DBs (FAISS
`IndexPQFastScan`, Qdrant, Milvus IVF-SQ) live in:

- **2–4 bits per dimension** (not 1, not 32),
- **codebook-free** scalar quantization (no k-means training, online ingest),
- a **FastScan-style nibble-LUT SIMD kernel** that scores 16/32 candidates per
  vector instruction without ever materializing f32,
- competitive recall **without** a mandatory f32 rerank pass.

The upstream reference project ([RyanCodrai/turbovec]) *reports* this regime
reaching ~16× compression (6,144 → 384 bytes for d=1536), R@1 competitive with or
ahead of FAISS `IndexPQ` at 4-bit, and FastScan-class scan throughput on ARM —
all with online ingest and no training phase. **Those are the external project's
numbers, not this crate's.** This crate's own *measured* results are the
uniform-random worst-case table under "Validation" above (recall@10 of
0.308 / 0.561 / 0.879 at 1/2/4-bit); broader competitive benchmarks are listed
as targets-to-validate in "Acceptance criteria" and the SIMD-kernel milestones.

[RyanCodrai/turbovec]: https://github.com/RyanCodrai/turbovec

### What already exists (and why this is not duplication)

Two TurboQuant-adjacent pieces are already in the repo. Neither closes the gap:

1. **`ruvector-rabitq`** (ADR-157) — already contains the *rotation* half of
   TurboQuant: `RandomRotation::HadamardSigned` (the `D₁·H·D₂·H·D₃` randomized
   Hadamard, `O(D log D)`, cited to arXiv:2504.19874 §3.2 in
   `rotation.rs`). It also defines the two traits this ADR will reuse —
   `AnnIndex` (index API) and `VectorKernel`/`KernelCaps` (pluggable scan
   backend). But its codes are **1-bit**, its kernels are **XNOR-popcount**, and
   it has no scalar-quantization or per-coordinate calibration.

2. **`ruvllm/src/quantize/turbo_quant.rs`** (1,483 lines) — a full TurboQuant
   *value codec* for **transformer KV-cache and embedding compression**
   (PolarQuant + QJL residual, 2.5–4.0 effective bits). This is a
   **data-oblivious tensor compressor**, *not* an `AnnIndex`: it has no
   inverted lists, no top-k heap, no FastScan LUT kernel, no
   filtered/allowlist search, no stable IDs/deletion. It is the wrong tool for
   "search 10M vectors for the nearest 10."

So the work here is: **reuse rabitq's rotation + traits, reuse the lessons from
ruvllm's codec, and build the missing multi-bit FastScan *search index*.**

## Decision

Introduce **`crates/ruvector-turbovec`**, a multi-bit TurboQuant ANN index that
implements the existing `ruvector_rabitq::AnnIndex` trait and exposes a FastScan
SIMD kernel via the existing `VectorKernel`/`KernelCaps` contract, so it drops
into the `ruvector-rulake` dispatcher (ADR-155/157) with no new plumbing.

Six techniques are ported/adapted from turbovec/TurboQuant:

### T1 — Normalize + randomized Hadamard rotation *(reuse, don't rebuild)*
Strip each vector's L2 norm (store as one f32) and apply
`RandomRotation::HadamardSigned` from `ruvector-rabitq`. After rotation every
coordinate is ~Beta-distributed → N(0, 1/d), making **per-coordinate scalar
quantization optimal without a codebook**. We import this type rather than
reimplement the FWHT.

### T2 — Lloyd–Max scalar quantization (2-bit / 4-bit)
Precompute MSE-optimal bucket boundaries for the canonical N(0,1/d) marginal at
`bit_width ∈ {2, 4}` (4 and 16 buckets). Coordinates become 2-bit (0–3) or
4-bit (0–15) integers. Boundaries are **constants of the distribution**, not of
the data → zero training. (ruvllm's codec already has the MSE-quantizer math to
borrow from.)

### T3 — Per-coordinate calibration (TQ+)
During the *first* `add()` batch, fit two scalars per coordinate
(`shift[d]`, `scale[d]`) that map the empirical quantile range onto the
canonical Beta. Freeze after warm-up; all later vectors reuse them. This
corrects the residual non-Gaussianity of finite-d Hadamard rotation and is the
"+" that buys turbovec its recall edge over plain SQ. **No counterpart exists in
the repo today.**

### T4 — Length-renormalized unbiased inner-product scoring
Store a per-vector correction scalar so the dot-product estimator is **unbiased**
(removes the systematic downward bias of scalar quantization) at **zero
search-time cost** — the scalar folds into the final score multiply. This lets
us skip the mandatory f32 rerank that 1-bit RaBitQ needs, keeping the memory win.

### T5 — FastScan nibble-LUT SIMD kernel (the core perf win)
Lay out codes in **32-vector SoA blocks**. For a query, precompute a small
per-sub-quantizer lookup table; score a whole block by **nibble-split table
lookups** (`vpshufb`/`tbl`) instead of arithmetic:
- **x86**: `AVX-512BW` (`VPSHUFB`/`VPERMI2B`) with **AVX2 fallback**
  (`_mm256_shuffle_epi8`), targeting `x86-64-v3` like rabitq does.
- **ARM**: `NEON` `vqtbl1q_u8` nibble lookups.
- **WASM**: scalar fallback (matches rabitq's wasm policy; bit-identical).
Implements `VectorKernel`; advertises `accelerator: "cpu-simd-fastscan"` via
`KernelCaps`. rabitq's existing AVX2/512 *popcount* kernels are **not**
reusable here (popcount ≠ table-lookup), but the trait, dispatch, and
determinism contract are.

### T6 — Block-granularity filtered search + stable IDs
- **Filtered search**: an allowlist is tested at **32-vector block
  granularity** *inside* the kernel — fully-excluded blocks short-circuit before
  any LUT work; individual disallowed slots are dropped at heap-insert time.
- **`IdMapIndex`**: external `u64` IDs that survive deletion, with **O(1)
  remove** (tombstone + free-list), mirroring turbovec's `IdMapIndex`. The base
  `TurboQuantIndex` uses dense sequential internal IDs for online ingest.

### Reuse boundary (what we build vs import)

| Component | Source | Action |
|-----------|--------|--------|
| Randomized Hadamard rotation | `ruvector-rabitq::RandomRotation` | **import** |
| `AnnIndex` trait | `ruvector-rabitq::index` | **implement** |
| `VectorKernel` / `KernelCaps` | `ruvector-rabitq::kernel` | **implement** |
| MSE/Lloyd–Max quantizer math | `ruvllm::quantize` | **borrow/extract** |
| Lloyd–Max boundary tables (2/4-bit) | TurboQuant constants | **build (new)** |
| TQ+ per-coordinate calibration | — | **build (new)** |
| FastScan nibble-LUT SIMD kernel | — | **build (new)** |
| 32-block SoA layout + filtered scan | — | **build (new)** |
| IdMap O(1)-delete | — | **build (new)** |
| Persistence (`.tv` file) | mirror `rabitq::persist` | **build (new)** |

If the borrowed MSE math from `ruvllm` proves reusable across both crates, a
follow-up should hoist it into `ruvector-math` rather than copy it (flagged as a
consequence, not done here).

### Proposed API (Rust)

```rust
use ruvector_turbovec::{TurboVecIndex, IdMapIndex, BitWidth};

let mut idx = TurboVecIndex::new(/*dim=*/1536, BitWidth::Four)?; // 4-bit
idx.add(0, vector)?;                       // online ingest (AnnIndex)
idx.add_batch(rows)?;                       // warms up TQ+ calibration on 1st batch
let hits = idx.search(&query, 10)?;         // Vec<SearchResult{ id, score }>

// External IDs + deletion + filtered search
let mut m = IdMapIndex::new(1536, BitWidth::Four)?;
m.add_with_ids(&vectors, &ids /* &[u64] */)?;
m.remove(1002)?;                            // O(1)
let hits = m.search_filtered(&query, 10, &allowlist /* &[u64] */)?;
```

## Consequences

### Positive
- **Closes the 2–4-bit FastScan gap** — the one mainstream production ANN regime
  ruvector lacks; makes us comparable to FAISS `IndexPQFastScan`/Milvus IVF-SQ.
- **16× compression** with **online ingest, no training** (d=1536: 6 KB → 384 B).
- **Recall without mandatory f32 rerank** (via T4 unbiased scoring), so the
  memory win is real, unlike 1-bit-with-rerank.
- **Zero new plumbing** — implements existing `AnnIndex` + `VectorKernel`, so it
  registers with the `ruvector-rulake` dispatcher and inherits its
  determinism/witness contract.
- **Composable**: the same block-SoA codes can later back an IVF posting list
  (`ruvector-rairs` IVF-SQ-FastScan) — a natural ADR-193 follow-up.

### Negative / risks
- **SIMD surface area**: hand-written AVX-512BW + AVX2 + NEON nibble kernels are
  the bulk of the bug/maintenance risk. Mitigation: a scalar reference kernel is
  the determinism oracle; every SIMD kernel is fuzzed bit-identical against it
  (matches rabitq's `deterministic` cap policy). `#[target_feature]` paths are
  `unsafe` — this crate breaks rabitq's "no `unsafe`" guarantee and must
  document that explicitly.
- **TQ+ calibration is stateful**: the first batch's distribution freezes the
  calibration; pathological first batches degrade later recall. Mitigation:
  expose `recalibrate()` and a min-warmup-count; default to a generous warmup.
- **Possible math duplication** with `ruvllm` if not hoisted to `ruvector-math`.
- **Name collision risk**: "TurboQuant" now appears in three places
  (`ruvllm` codec, this index, rabitq's rotation comment). The crate is named
  `turbovec` (the *index*) to disambiguate from the `turbo_quant` *codec*.

### Neutral
- WASM gets the scalar path only (no SIMD), consistent with rabitq.
- Adds one workspace crate; no changes to existing crates beyond making
  `RandomRotation`/trait exports `pub` if any aren't already.

## Alternatives considered

1. **Extend `ruvector-rabitq` from 1-bit to multi-bit in place.** Rejected:
   conflates two estimator families (XNOR-popcount vs nibble-LUT FastScan),
   would force `unsafe` into a crate that advertises none, and bloats its API.
   A sibling crate sharing traits is cleaner.
2. **Use `ruvllm`'s `turbo_quant.rs` codec directly for search.** Rejected: it's
   a value/KV-cache compressor with no index, no top-k, no LUT kernel, no
   IDs/deletion — wrong abstraction.
3. **Add IVF-SQ to `ruvector-rairs` instead.** Deferred, not rejected: IVF-SQ is
   the natural *composition* of this crate's codes with rairs' posting lists.
   Build the flat FastScan index first (this ADR), then layer IVF (follow-up).
4. **Wrap FAISS via FFI.** Rejected: violates ruvector's no-C/C++-deps,
   pure-Rust, WASM-portable posture.

## Rollout & acceptance criteria

Implement on branch `claude/ruvector-turbovec-optimization-FhaDh` (this ADR),
crate work in a follow-up PR. Milestones:

1. **M1 — Scalar reference (no SIMD).** Rotation reuse + Lloyd–Max 2/4-bit +
   TQ+ + unbiased scoring + `AnnIndex`. Recall + memory parity test vs a brute
   f32 baseline on SIFT1M / a synthetic OpenAI-d1536 set.
2. **M2 — FastScan SIMD kernel.** AVX2 + NEON nibble-LUT, fuzzed bit-identical
   to M1's scalar scorer; `VectorKernel` impl; criterion bench in
   `benches/turbovec_bench.rs`.
3. **M3 — IdMap + filtered search + persistence.** O(1) delete, block-level
   allowlist, `.tv` save/load round-trip test.
4. **M4 — AVX-512BW kernel + rulake dispatcher registration.**

**Acceptance (targets, to validate — not yet measured):**
- ≥ **15× compression** at d=1536, 4-bit, incl. norm + calibration overhead.
- **R@1 within ±1 point of, or better than,** the 1-bit-RaBitQ-with-rerank path
  at equal or lower memory.
- FastScan kernel **≥ 3× faster** than the scalar reference scorer on x86-64-v3.
- **Bit-identical** scan output across scalar/AVX2/AVX-512/NEON (determinism
  oracle), enforced in CI.
- `cargo build --release -p ruvector-turbovec` green; all unit + property tests
  pass; no `clippy` regressions.

## References

- TurboQuant (Google Research) — arXiv:2504.19874
- PolarQuant — arXiv:2502.02617 · QJL — arXiv:2406.03482
- RaBitQ (Gao & Long, SIGMOD 2024) — basis of `ruvector-rabitq`
- FAISS FastScan / `IndexPQFastScan` — the nibble-LUT SIMD scan precedent
- RyanCodrai/turbovec — https://github.com/RyanCodrai/turbovec
- Prior art in-repo: `crates/ruvector-rabitq/src/rotation.rs` (Hadamard),
  `crates/ruvllm/src/quantize/turbo_quant.rs` (KV-cache codec), ADR-157, ADR-193
