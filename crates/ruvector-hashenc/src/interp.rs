//! d-linear interpolation over grid corners (ADR-258 §6.1).
//!
//! For each level we locate the `2^{d_idx}` corners surrounding the projected
//! point and blend their feature rows by the multilinear weights (a partition
//! of unity: weights sum to 1). The forward pass records `(row, weight)` pairs
//! per corner in an [`EncodeCache`] so the backward pass is a pure sparse
//! scatter — the property that makes online learning bandwidth-light.

// Tight multilinear-interpolation loops read more clearly with explicit indices.
#![allow(clippy::needless_range_loop)]

use crate::hash::row_index;
use crate::tables::FeatureTables;

/// Per-encode cache of touched corners for the backward pass.
#[derive(Clone, Debug, Default)]
pub struct EncodeCache {
    /// `per_level[l]` holds `(row, weight)` for each of the `2^{d_idx}` corners.
    pub per_level: Vec<Vec<(usize, f32)>>,
}

impl EncodeCache {
    pub fn clear(&mut self) {
        for v in &mut self.per_level {
            v.clear();
        }
    }
}

/// Interpolate level `level` at projected coords `coords` (each in `[0,1)`),
/// appending the `F` blended features to `out` and recording corners in `cache`.
pub fn dlinear(
    tables: &FeatureTables,
    level: usize,
    coords: &[f32],
    out: &mut Vec<f32>,
    cache_level: &mut Vec<(usize, f32)>,
) {
    let d = coords.len();
    let f = tables.features_per_level();
    let n = tables.resolution(level);
    let dense = tables.is_dense(level);
    let log2_t = tables.cfg.log2_table_size;

    // Per-dim floor index and fractional offset within the cell.
    let mut floor = [0u32; 7];
    let mut frac = [0f32; 7];
    for i in 0..d {
        let scaled = (coords[i] * n as f32).min(n as f32 - f32::EPSILON).max(0.0);
        let fl = scaled.floor();
        floor[i] = fl as u32;
        frac[i] = scaled - fl;
    }

    let base = out.len();
    out.resize(base + f, 0.0);
    cache_level.clear();

    let corners = 1usize << d;
    let mut corner = [0u32; 7];
    for mask in 0..corners {
        let mut weight = 1.0f32;
        for i in 0..d {
            let bit = (mask >> i) & 1;
            if bit == 1 {
                corner[i] = (floor[i] + 1).min(n);
                weight *= frac[i];
            } else {
                corner[i] = floor[i];
                weight *= 1.0 - frac[i];
            }
        }
        if weight == 0.0 {
            continue;
        }
        let row = row_index(&corner[..d], n, log2_t, dense);
        let feat = tables.row(level, row);
        for j in 0..f {
            out[base + j] += weight * feat[j];
        }
        cache_level.push((row, weight));
    }
}

/// Accumulate `dL/d coord_j` for one level into `coord_grad` (length `d_idx`),
/// given `grad_out_level = dL/d feature` for this level (length `F`). Used by
/// the learned-projection backward pass (ADR-258 Phase 2). Re-derives corners
/// from `coords` so it is independent of the forward cache.
pub fn dlinear_coord_grad(
    tables: &FeatureTables,
    level: usize,
    coords: &[f32],
    grad_out_level: &[f32],
    coord_grad: &mut [f32],
) {
    let d = coords.len();
    let f = tables.features_per_level();
    let n = tables.resolution(level);
    let dense = tables.is_dense(level);
    let log2_t = tables.cfg.log2_table_size;

    let mut floor = [0u32; 7];
    let mut frac = [0f32; 7];
    for i in 0..d {
        let scaled = (coords[i] * n as f32).min(n as f32 - f32::EPSILON).max(0.0);
        let fl = scaled.floor();
        floor[i] = fl as u32;
        frac[i] = scaled - fl;
    }

    let corners = 1usize << d;
    let mut corner = [0u32; 7];
    let mut w = [0f32; 7];
    for mask in 0..corners {
        for i in 0..d {
            let bit = (mask >> i) & 1;
            if bit == 1 {
                corner[i] = (floor[i] + 1).min(n);
                w[i] = frac[i];
            } else {
                corner[i] = floor[i];
                w[i] = 1.0 - frac[i];
            }
        }
        let row = row_index(&corner[..d], n, log2_t, dense);
        let feat = tables.row(level, row);
        // g = <grad_out_level, table_row>  (shared across dims for this corner)
        let mut g = 0.0f32;
        for k in 0..f {
            g += grad_out_level[k] * feat[k];
        }
        // d weight / d frac_j = sign_j * prod_{i != j} w_i ; d frac_j/d coord_j = N
        for j in 0..d {
            let sign = if (mask >> j) & 1 == 1 { 1.0 } else { -1.0 };
            let mut prod_excl = 1.0f32;
            for i in 0..d {
                if i != j {
                    prod_excl *= w[i];
                }
            }
            coord_grad[j] += n as f32 * sign * prod_excl * g;
        }
    }
}
