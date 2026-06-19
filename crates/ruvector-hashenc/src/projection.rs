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

    /// Fit projection rows from sample data when `cfg.projection` is `PcaInit`
    /// or `Learned`. Uses covariance power-iteration with deflation to extract
    /// the top `d_idx` principal directions. No-op for `LockedRandom`.
    pub fn fit(&mut self, cfg: &HashEncConfig, samples: &[Vec<f32>]) {
        if !matches!(cfg.projection, ProjectionKind::PcaInit | ProjectionKind::Learned)
            || samples.is_empty()
        {
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

    #[inline]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Accumulate the projection-row gradient given `coord_grad = dL/d coord`
    /// (length `index_dims`) and the post-logistic `coords` from the forward
    /// pass. Chains through the logistic (`σ' = c(1-c)`) and the linear map.
    pub fn accumulate_grad(&self, x: &[f32], coords: &[f32], coord_grad: &[f32], g: &mut ProjGrad) {
        let n = self.input_dim.min(x.len());
        for j in 0..self.index_dims {
            let c = coords[j];
            let dz = coord_grad[j] * c * (1.0 - c) * self.scale; // dL/dz_j
            let rowg = &mut g.rows[j];
            for i in 0..n {
                rowg[i] += dz * x[i];
            }
        }
    }

    /// SGD step on the projection rows, then zero the accumulator.
    pub fn apply_grad(&mut self, g: &mut ProjGrad, lr: f32) {
        for j in 0..self.index_dims {
            let row = &mut self.rows[j];
            let gr = &mut g.rows[j];
            for i in 0..row.len() {
                row[i] -= lr * gr[i];
                gr[i] = 0.0;
            }
        }
    }

    /// Perturb a single row entry (gradient-check tests / experimentation).
    pub fn perturb(&mut self, j: usize, i: usize, delta: f32) {
        self.rows[j][i] += delta;
    }
}

/// Gradient accumulator for the projection rows (mirrors [`Projection`] shape).
#[derive(Clone, Debug)]
pub struct ProjGrad {
    rows: Vec<Vec<f32>>,
}

impl ProjGrad {
    pub fn new(proj: &Projection) -> Self {
        Self {
            rows: proj.rows.iter().map(|r| vec![0.0f32; r.len()]).collect(),
        }
    }

    pub fn zero(&mut self) {
        for r in &mut self.rows {
            for v in r {
                *v = 0.0;
            }
        }
    }

    /// Accumulated gradient value at projection entry `(j, i)`.
    #[inline]
    pub fn value(&self, j: usize, i: usize) -> f32 {
        self.rows[j][i]
    }

    /// L2 norm of the accumulated gradient (diagnostics).
    pub fn l2_norm(&self) -> f32 {
        self.rows
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
    }
}
