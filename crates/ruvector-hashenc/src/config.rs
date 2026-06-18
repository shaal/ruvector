//! Configuration for the multiresolution hash encoder (ADR-258).
//!
//! Mirrors the hyperparameters from Müller et al., "Instant Neural Graphics
//! Primitives with a Multiresolution Hash Encoding" (SIGGRAPH 2022,
//! arXiv:2201.05989), retuned for high-dimensional vector retrieval.

/// How the input vector is projected into the low-dimensional index space
/// before hashing. High-dimensional embeddings (384–1536-D) cannot be gridded
/// directly (2^d corners), so we project to `index_dims` (2–4) first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionKind {
    /// Fixed Gaussian projection (deterministic from seed). Cheapest, robust.
    LockedRandom,
    /// Initialize projection rows from the top principal components of a sample.
    /// Concentrates multi-scale resolution along the highest-variance directions.
    PcaInit,
}

/// Configuration for [`crate::HashEncoder`].
#[derive(Clone, Debug)]
pub struct HashEncConfig {
    /// Number of resolution levels `L`. Instant-NGP default: 16.
    pub levels: usize,
    /// Feature width per level `F`. Instant-NGP default: 2.
    pub features_per_level: usize,
    /// `log2(T)` — per-level hash table size is `T = 2^log2_table_size`.
    /// Instant-NGP range: 14..=24; default here 19 (T = 524_288).
    pub log2_table_size: u8,
    /// `d_idx` — dimensionality of the index/grid space (2..=7). Default 3.
    pub index_dims: usize,
    /// Coarsest grid resolution `N_min`. Instant-NGP default: 16.
    pub n_min: u32,
    /// Finest grid resolution `N_max` (data-scaled). Default 4096.
    pub n_max: u32,
    /// Projection strategy into index space.
    pub projection: ProjectionKind,
    /// RNG seed for reproducible table/projection initialization.
    pub seed: u64,
}

impl Default for HashEncConfig {
    fn default() -> Self {
        Self {
            levels: 16,
            features_per_level: 2,
            log2_table_size: 19,
            index_dims: 3,
            n_min: 16,
            n_max: 4096,
            projection: ProjectionKind::LockedRandom,
            seed: 0x5217_2358,
        }
    }
}

impl HashEncConfig {
    /// `ln(b)` where `b = exp((ln N_max - ln N_min)/(L-1))` is the per-level
    /// geometric growth factor (Instant-NGP eq. 3).
    #[inline]
    pub fn ln_growth(&self) -> f32 {
        let l = self.levels.max(2) as f32;
        ((self.n_max as f32).ln() - (self.n_min as f32).ln()) / (l - 1.0)
    }

    /// Resolution `N_l = floor(N_min * b^l)` at the given level.
    #[inline]
    pub fn resolution(&self, level: usize) -> u32 {
        let scale = (self.ln_growth() * level as f32).exp();
        let n = (self.n_min as f32 * scale).floor();
        n.max(1.0) as u32
    }

    /// Output feature width fed to the GNN: `L * F`.
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.levels * self.features_per_level
    }

    /// Full (hashed) table size `T = 2^log2_table_size`.
    #[inline]
    pub fn table_size(&self) -> usize {
        1usize << self.log2_table_size
    }

    /// Number of grid corners touched per level: `2^index_dims`.
    #[inline]
    pub fn corners(&self) -> usize {
        1usize << self.index_dims
    }

    /// Effective table rows at a level: dense `(N_l+1)^d_idx` if it fits within
    /// `T` (collision-free coarse levels), otherwise the hashed size `T`.
    pub fn level_rows(&self, level: usize) -> usize {
        let n = self.resolution(level) as u64 + 1;
        let mut dense: u64 = 1;
        for _ in 0..self.index_dims {
            dense = dense.saturating_mul(n);
            if dense >= self.table_size() as u64 {
                return self.table_size();
            }
        }
        dense as usize
    }

    /// True if level `level` is dense (collision-free).
    #[inline]
    pub fn level_is_dense(&self, level: usize) -> bool {
        self.level_rows(level) < self.table_size()
    }

    /// Validate invariants; returns an error string if misconfigured.
    pub fn validate(&self) -> Result<(), String> {
        if self.levels < 1 {
            return Err("levels must be >= 1".into());
        }
        if self.features_per_level < 1 {
            return Err("features_per_level must be >= 1".into());
        }
        if !(2..=7).contains(&self.index_dims) {
            return Err("index_dims must be in 2..=7".into());
        }
        if self.log2_table_size == 0 || self.log2_table_size > 30 {
            return Err("log2_table_size must be in 1..=30".into());
        }
        if self.n_max < self.n_min {
            return Err("n_max must be >= n_min".into());
        }
        Ok(())
    }
}
