//! Tiered feature storage (ADR-258 §6.4).
//!
//! Composes the three tiers of the neural index v2 storage design:
//! - **HOT**: the trainable multiresolution hash tables (owned by the encoder;
//!   accounted here for footprint reporting).
//! - **WARM**: per-vector int8 scalar quantization of the raw embedding — a 4×
//!   compressed reconstruction tier used for final rerank. This wires
//!   quantization into the live retrieval path (the spirit of issue #563) on
//!   the neural route; production may swap in PQ / RaBitQ codes.
//! - **COLD**: block-aligned on-disk features (handled by `FeatureTables::save`
//!   / the GNN `cold_tier`); represented here only in the footprint accounting.
//!
//! Includes a SIMD-accelerated L2 distance for reranking reconstructed vectors,
//! with a scalar reference and a differential test guaranteeing equivalence.

/// Per-vector int8 scalar-quantized warm tier (4× vs f32).
#[derive(Clone, Debug)]
pub struct WarmInt8 {
    dim: usize,
    n: usize,
    mins: Vec<f32>,
    scales: Vec<f32>,
    codes: Vec<u8>, // row-major [n, dim]
}

impl WarmInt8 {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            n: 0,
            mins: Vec::new(),
            scales: Vec::new(),
            codes: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Quantize and append a vector (min/scale per vector).
    pub fn push(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim);
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &x in v {
            min = min.min(x);
            max = max.max(x);
        }
        let scale = ((max - min) / 255.0).max(1e-12);
        self.mins.push(min);
        self.scales.push(scale);
        for &x in v {
            let q = ((x - min) / scale).round().clamp(0.0, 255.0) as u8;
            self.codes.push(q);
        }
        self.n += 1;
    }

    /// Reconstruct vector `i` into `out` (length `dim`).
    pub fn reconstruct_into(&self, i: usize, out: &mut [f32]) {
        let base = i * self.dim;
        let (min, scale) = (self.mins[i], self.scales[i]);
        for (j, o) in out.iter_mut().enumerate().take(self.dim) {
            *o = min + (self.codes[base + j] as f32) * scale;
        }
    }

    pub fn reconstruct(&self, i: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        self.reconstruct_into(i, &mut out);
        out
    }

    /// Bytes used by the warm tier (codes + per-vector min/scale).
    pub fn byte_size(&self) -> usize {
        self.codes.len() + self.n * 2 * std::mem::size_of::<f32>()
    }
}

/// Footprint accounting across tiers.
#[derive(Clone, Copy, Debug)]
pub struct TierStats {
    pub hot_bytes: usize,
    pub warm_bytes: usize,
    pub raw_f32_bytes: usize,
    /// `raw_f32_bytes / warm_bytes` — WARM-tier compression vs full f32.
    pub warm_compression: f32,
}

/// Tiered store: HOT (hash-table footprint) + WARM (int8 reconstruction).
#[derive(Clone, Debug)]
pub struct TieredFeatureStore {
    warm: WarmInt8,
    hot_table_bytes: usize,
    dim: usize,
}

impl TieredFeatureStore {
    /// `hot_table_bytes` is the shared, fixed footprint of the encoder tables.
    pub fn new(dim: usize, hot_table_bytes: usize) -> Self {
        Self {
            warm: WarmInt8::new(dim),
            hot_table_bytes,
            dim,
        }
    }

    pub fn add(&mut self, raw: &[f32]) {
        self.warm.push(raw);
    }

    pub fn len(&self) -> usize {
        self.warm.len()
    }
    pub fn is_empty(&self) -> bool {
        self.warm.is_empty()
    }

    pub fn reconstruct(&self, i: usize) -> Vec<f32> {
        self.warm.reconstruct(i)
    }

    /// SIMD-accelerated rerank distance between query and reconstructed vector `i`.
    pub fn rerank_distance(&self, i: usize, query: &[f32], scratch: &mut Vec<f32>) -> f32 {
        if scratch.len() != self.dim {
            scratch.resize(self.dim, 0.0);
        }
        self.warm.reconstruct_into(i, scratch);
        l2_distance(query, scratch)
    }

    pub fn stats(&self) -> TierStats {
        let raw = self.warm.len() * self.dim * std::mem::size_of::<f32>();
        let warm = self.warm.byte_size();
        TierStats {
            hot_bytes: self.hot_table_bytes,
            warm_bytes: warm,
            raw_f32_bytes: raw,
            warm_compression: if warm == 0 {
                0.0
            } else {
                raw as f32 / warm as f32
            },
        }
    }
}

// ----------------------------- SIMD L2 distance -----------------------------

/// L2 (Euclidean) distance, dispatching to AVX2 when available, else scalar.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: guarded by runtime feature detection.
            return unsafe { l2_avx2(a, b) };
        }
    }
    l2_scalar(a, b)
}

/// Scalar reference implementation.
pub fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(d, d, sum);
        i += 8;
    }
    // horizontal sum of the 8 lanes
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut acc: f32 = tmp.iter().sum();
    // scalar tail
    while i < n {
        let d = a[i] - b[i];
        acc += d * d;
        i += 1;
    }
    acc.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_reconstruction_error_is_bounded() {
        let dim = 64;
        let mut warm = WarmInt8::new(dim);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        warm.push(&v);
        let r = warm.reconstruct(0);
        let max_err = v
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // error <= scale (one quantization step); range ~2 -> scale ~2/255.
        assert!(max_err < 0.02, "max reconstruction error {max_err} too large");
    }

    #[test]
    fn warm_tier_is_about_4x_smaller() {
        let dim = 128;
        let mut store = TieredFeatureStore::new(dim, 0);
        for k in 0..100 {
            let v: Vec<f32> = (0..dim).map(|i| (i + k) as f32 * 0.01).collect();
            store.add(&v);
        }
        let s = store.stats();
        // 4 bytes/dim f32 vs 1 byte/dim int8 (+ tiny per-vector overhead) ≈ 4×.
        assert!(s.warm_compression > 3.5, "compression {} too low", s.warm_compression);
    }

    #[test]
    fn simd_matches_scalar_distance() {
        for len in [1usize, 7, 8, 9, 31, 64, 257, 768] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.013).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.019).cos()).collect();
            let s = l2_scalar(&a, &b);
            let d = l2_distance(&a, &b);
            assert!((s - d).abs() < 1e-4, "len={len}: simd {d} vs scalar {s}");
        }
    }

    #[test]
    fn rerank_distance_uses_reconstruction() {
        let dim = 32;
        let mut store = TieredFeatureStore::new(dim, 0);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).cos()).collect();
        store.add(&v);
        let mut scratch = Vec::new();
        let d = store.rerank_distance(0, &v, &mut scratch);
        // query == original; distance to its int8 reconstruction is small.
        assert!(d < 0.05, "self-distance {d} too large");
    }
}
