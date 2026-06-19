//! # ruvector-hashenc
//!
//! Multiresolution hash encoding of trainable multi-scale features for
//! RuVector's neural index, adapted from Müller et al., *"Instant Neural
//! Graphics Primitives with a Multiresolution Hash Encoding"* (SIGGRAPH 2022,
//! arXiv:2201.05989). See **ADR-258** for the full design rationale.
//!
//! ## What this provides
//! - [`HashEncoder`]: maps a high-dimensional embedding `x ∈ R^d` to a compact,
//!   trainable, multi-scale feature vector `enc(x) ∈ R^{L·F}` via a low-`d_idx`
//!   projection + a hashed multiresolution grid with d-linear interpolation.
//! - [`FeatureTables`] / [`GradAccum`]: trainable tables and a sparse-scatter
//!   gradient accumulator that preserves RuVector's persistent-differentiable,
//!   mmap-friendly update flow.
//!
//! ## Why it helps (ADR-258 §3)
//! Online updates touch only `2^{d_idx}·L ≪ d_embed` parameters per sample, so
//! the self-learning loop becomes compute- and bandwidth-light. Coarse levels
//! are collision-free and carry global structure; fine levels add detail with a
//! fixed memory budget independent of dataset size.
//!
//! ## Example
//! ```
//! use ruvector_hashenc::{HashEncoder, HashEncConfig};
//! let cfg = HashEncConfig { levels: 8, features_per_level: 2,
//!     log2_table_size: 14, index_dims: 2, n_min: 8, n_max: 256, ..Default::default() };
//! let enc = HashEncoder::new(cfg, 64);
//! let x = vec![0.1f32; 64];
//! let f = enc.encode(&x);
//! assert_eq!(f.len(), enc.output_dim());
//! ```

mod config;
mod hash;
mod interp;
mod projection;
mod rng;
pub mod sampling;
mod tables;
pub mod tiered;

pub use config::{HashEncConfig, ProjectionKind};
pub use interp::EncodeCache;
pub use projection::{ProjGrad, Projection};
pub use rng::SplitMix64;
pub use sampling::{NegativeSampler, TemperatureSchedule};
pub use tables::{FeatureTables, GradAccum};
pub use tiered::{TieredFeatureStore, TierStats};

use std::path::Path;

/// Errors produced by the encoder.
#[derive(Debug, thiserror::Error)]
pub enum HashEncError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// A multiresolution hash encoder: projection + trainable feature tables.
#[derive(Clone, Debug)]
pub struct HashEncoder {
    cfg: HashEncConfig,
    projection: Projection,
    tables: FeatureTables,
}

impl HashEncoder {
    /// Build an encoder for inputs of dimension `input_dim`.
    pub fn new(cfg: HashEncConfig, input_dim: usize) -> Self {
        cfg.validate().expect("invalid HashEncConfig");
        let projection = Projection::new(&cfg, input_dim);
        let tables = FeatureTables::new(&cfg);
        Self {
            cfg,
            projection,
            tables,
        }
    }

    /// Build and, if `cfg.projection == PcaInit`, fit the projection to a sample.
    pub fn new_fitted(cfg: HashEncConfig, input_dim: usize, samples: &[Vec<f32>]) -> Self {
        let mut me = Self::new(cfg, input_dim);
        me.projection.fit(&me.cfg, samples);
        me
    }

    /// Output feature width `L·F`.
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.cfg.output_dim()
    }

    #[inline]
    pub fn config(&self) -> &HashEncConfig {
        &self.cfg
    }

    #[inline]
    pub fn tables(&self) -> &FeatureTables {
        &self.tables
    }

    #[inline]
    pub fn tables_mut(&mut self) -> &mut FeatureTables {
        &mut self.tables
    }

    #[inline]
    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    #[inline]
    pub fn projection_mut(&mut self) -> &mut Projection {
        &mut self.projection
    }

    /// True if the projection is configured to be trained (ADR-258 Phase 2).
    #[inline]
    pub fn projection_is_learned(&self) -> bool {
        matches!(self.cfg.projection, ProjectionKind::Learned)
    }

    /// Forward pass: returns `enc(x)` of length `L·F`.
    pub fn encode(&self, x: &[f32]) -> Vec<f32> {
        let mut cache = self.fresh_cache();
        self.encode_into(x, &mut cache)
    }

    /// Allocate a cache sized for this encoder (reuse across calls to avoid
    /// per-query allocation on the hot path).
    pub fn fresh_cache(&self) -> EncodeCache {
        EncodeCache {
            per_level: (0..self.cfg.levels)
                .map(|_| Vec::with_capacity(self.cfg.corners()))
                .collect(),
        }
    }

    /// Forward pass with an explicit cache for a subsequent [`backward`].
    pub fn encode_into(&self, x: &[f32], cache: &mut EncodeCache) -> Vec<f32> {
        let mut coords = vec![0.0f32; self.cfg.index_dims];
        self.projection.apply(x, &mut coords);
        let mut out = Vec::with_capacity(self.output_dim());
        if cache.per_level.len() != self.cfg.levels {
            *cache = self.fresh_cache();
        }
        for l in 0..self.cfg.levels {
            interp::dlinear(&self.tables, l, &coords, &mut out, &mut cache.per_level[l]);
        }
        out
    }

    /// Backward pass: scatter `grad_out` (length `L·F`) into `grad` using the
    /// corner/weight pairs recorded by the matching `encode_into`. Only the
    /// touched table rows receive gradient — the sparse-update property.
    pub fn backward(&self, cache: &EncodeCache, grad_out: &[f32], grad: &mut GradAccum) {
        let f = self.cfg.features_per_level;
        for (l, corners) in cache.per_level.iter().enumerate() {
            let base = l * f;
            for &(row, weight) in corners {
                for j in 0..f {
                    grad.add(l, row, j, weight * grad_out[base + j]);
                }
            }
        }
    }

    /// Backward pass for the **learned projection** (ADR-258 Phase 2): given
    /// `grad_out` (length `L·F`), accumulate the gradient w.r.t. the projection
    /// rows into `pgrad`. Re-derives coordinates and corners from `x`, so it can
    /// be called independently of the table backward pass.
    pub fn projection_grad(&self, x: &[f32], grad_out: &[f32], pgrad: &mut ProjGrad) {
        let f = self.cfg.features_per_level;
        let mut coords = vec![0.0f32; self.cfg.index_dims];
        self.projection.apply(x, &mut coords);
        let mut coord_grad = vec![0.0f32; self.cfg.index_dims];
        for l in 0..self.cfg.levels {
            let base = l * f;
            interp::dlinear_coord_grad(
                &self.tables,
                l,
                &coords,
                &grad_out[base..base + f],
                &mut coord_grad,
            );
        }
        self.projection
            .accumulate_grad(x, &coords, &coord_grad, pgrad);
    }

    /// Apply an SGD step to the projection rows.
    pub fn apply_projection_grad(&mut self, pgrad: &mut ProjGrad, lr: f32) {
        self.projection.apply_grad(pgrad, lr);
    }

    /// Persist tables to disk.
    pub fn save_tables(&self, path: &Path) -> Result<(), HashEncError> {
        self.tables.save(path).map_err(HashEncError::Io)
    }

    /// Restore tables from disk into a `cfg`-shaped encoder.
    pub fn load_tables(&mut self, path: &Path) -> Result<(), HashEncError> {
        self.tables = FeatureTables::load(&self.cfg, path).map_err(HashEncError::Io)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> HashEncConfig {
        HashEncConfig {
            levels: 6,
            features_per_level: 2,
            log2_table_size: 12,
            index_dims: 3,
            n_min: 4,
            n_max: 64,
            ..Default::default()
        }
    }

    #[test]
    fn output_dim_is_l_times_f() {
        let enc = HashEncoder::new(small_cfg(), 32);
        assert_eq!(enc.output_dim(), 6 * 2);
        let x = vec![0.3f32; 32];
        assert_eq!(enc.encode(&x).len(), 12);
    }

    #[test]
    fn interpolation_weights_sum_to_one() {
        // Partition-of-unity: per level the corner weights must sum to 1.
        let enc = HashEncoder::new(small_cfg(), 16);
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.137).sin()).collect();
        let mut cache = enc.fresh_cache();
        let _ = enc.encode_into(&x, &mut cache);
        for corners in &cache.per_level {
            let sum: f32 = corners.iter().map(|&(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-4, "weights summed to {sum}");
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let enc = HashEncoder::new(small_cfg(), 24);
        let x = vec![0.42f32; 24];
        assert_eq!(enc.encode(&x), enc.encode(&x));
    }

    #[test]
    fn coarse_levels_are_dense_collision_free() {
        let cfg = small_cfg();
        let enc = HashEncoder::new(cfg.clone(), 8);
        // Level 0 (N=4 -> 5^3=125 rows < 4096) must be dense.
        assert!(enc.tables().is_dense(0));
    }

    #[test]
    fn save_load_roundtrip() {
        let enc = HashEncoder::new(small_cfg(), 16);
        let x = vec![0.21f32; 16];
        let before = enc.encode(&x);
        let dir = std::env::temp_dir();
        let path = dir.join("rhe_roundtrip.bin");
        enc.save_tables(&path).unwrap();
        let mut enc2 = HashEncoder::new(small_cfg(), 16);
        enc2.load_tables(&path).unwrap();
        let after = enc2.encode(&x);
        assert_eq!(before, after);
        let _ = std::fs::remove_file(path);
    }
}
