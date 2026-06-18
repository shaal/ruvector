//! Projection from the high-dimensional embedding space `R^d` into the
//! low-dimensional index space `R^{d_idx}` (ADR-258 §3, §6.1).
//!
//! This is the key adaptation that makes Instant-NGP's grid encoding tractable
//! for retrieval embeddings: instead of a `2^d`-corner grid we project to
//! `d_idx ∈ {2,3,4}` and grid there. Output coordinates are squashed to `[0,1)`
//! with a logistic so they index a unit grid. Phase 1 projections are *locked*
//! (no gradient); a learned projection is a Phase 2 upgrade.

use crate::config::{HashEncConfig, ProjectionKind};
use crate::rng::SplitMix64;

/// Locked linear projection followed by a per-coordinate logistic squash.
#[derive(Clone, Debug)]
pub struct Projection {
    /// `d_idx` rows, each of length `d` (input dimension).
    rows: Vec<Vec<f32>>,
    /// Per-row scale `1/sqrt(d)` to keep logits well-conditioned.
    scale: f32,
    input_dim: usize,
    index_dims: usize,
}

#[inline]
fn logistic(z: f32) -> f32 {
    // Clamp to keep strictly inside (0,1) for grid safety.
    let s = 1.0 / (1.0 + (-z).exp());
    s.clamp(1e-6, 1.0 - 1e-6)
}

impl Projection {
    /// Construct a locked random Gaussian projection.
    pub fn new(cfg: &HashEncConfig, input_dim: usize) -> Self {
        let mut rng = SplitMix64::new(cfg.seed ^ 0xA5A5_1234_DEAD_BEEF);
        let rows = (0..cfg.index_dims)
            .map(|_| (0..input_dim).map(|_| rng.next_normal()).collect())
            .collect();
        Self {
            rows,
            scale: 1.0 / (input_dim.max(1) as f32).sqrt(),
            input_dim,
            index_dims: cfg.index_dims,
        }
    }

    /// Fit projection rows from sample data when `cfg.projection == PcaInit`.
    /// Uses covariance power-iteration with deflation to extract the top
    /// `d_idx` principal directions. No-op for `LockedRandom`.
    pub fn fit(&mut self, cfg: &HashEncConfig, samples: &[Vec<f32>]) {
        if cfg.projection != ProjectionKind::PcaInit || samples.is_empty() {
            return;
        }
        let d = self.input_dim;
        // Mean.
        let mut mean = vec![0.0f32; d];
        for s in samples {
            for i in 0..d {
                mean[i] += s[i];
            }
        }
        for m in &mut mean {
            *m /= samples.len() as f32;
        }
        // Centered copies.
        let centered: Vec<Vec<f32>> = samples
            .iter()
            .map(|s| (0..d).map(|i| s[i] - mean[i]).collect())
            .collect();

        let mut comps: Vec<Vec<f32>> = Vec::with_capacity(self.index_dims);
        let mut rng = SplitMix64::new(cfg.seed ^ 0x1357_9BDF);
        for _ in 0..self.index_dims {
            // Random init, orthogonalize against found components.
            let mut v: Vec<f32> = (0..d).map(|_| rng.next_normal()).collect();
            for _ in 0..32 {
                // u = C v = (1/n) Σ x_c (x_c · v)
                let mut u = vec![0.0f32; d];
                for x in &centered {
                    let dot: f32 = (0..d).map(|i| x[i] * v[i]).sum();
                    for i in 0..d {
                        u[i] += x[i] * dot;
                    }
                }
                // Deflate: remove projections onto previous comps.
                for c in &comps {
                    let dot: f32 = (0..d).map(|i| u[i] * c[i]).sum();
                    for i in 0..d {
                        u[i] -= dot * c[i];
                    }
                }
                let norm: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-12 {
                    break;
                }
                for i in 0..d {
                    v[i] = u[i] / norm;
                }
            }
            comps.push(v);
        }
        self.rows = comps;
        self.scale = 1.0; // PCA directions are unit-norm; logits already scaled.
    }

    /// Project `x` into `[0,1)^{d_idx}` index coordinates.
    #[inline]
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.index_dims);
        for (j, row) in self.rows.iter().enumerate() {
            let mut z = 0.0f32;
            // Guard against dimension mismatch by iterating the min length.
            let n = row.len().min(x.len());
            for i in 0..n {
                z += row[i] * x[i];
            }
            out[j] = logistic(z * self.scale);
        }
    }

    #[inline]
    pub fn index_dims(&self) -> usize {
        self.index_dims
    }
}
