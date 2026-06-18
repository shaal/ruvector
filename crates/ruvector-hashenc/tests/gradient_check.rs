//! Formal differentiability proof for the multiresolution hash encoder
//! (ADR-258 §8/§9): the analytic sparse-scatter backward pass must match a
//! central finite-difference estimate of the gradient of a scalar loss w.r.t.
//! every trainable table entry, within a tight tolerance.
//!
//! Loss: L(x) = 0.5 * Σ_k (enc(x)_k - target_k)^2
//! dL/d(enc_k) = enc(x)_k - target_k   (this is `grad_out`)
//! The backward pass scatters `grad_out` into table-entry gradients; we verify
//! each one against (L(θ+ε) - L(θ-ε)) / 2ε.

use ruvector_hashenc::{GradAccum, HashEncConfig, HashEncoder};

fn loss(enc_out: &[f32], target: &[f32]) -> f32 {
    0.5 * enc_out
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
}

#[test]
fn analytic_gradient_matches_finite_difference() {
    let cfg = HashEncConfig {
        levels: 4,
        features_per_level: 2,
        log2_table_size: 10,
        index_dims: 2,
        n_min: 2,
        n_max: 16,
        ..Default::default()
    };
    let input_dim = 12;
    let mut enc = HashEncoder::new(cfg.clone(), input_dim);
    let f = cfg.features_per_level;

    // Fixed input and target.
    let x: Vec<f32> = (0..input_dim).map(|i| (i as f32 * 0.31).cos()).collect();
    let target: Vec<f32> = (0..cfg.output_dim()).map(|i| 0.05 * i as f32).collect();

    // --- analytic gradient ---
    let mut cache = enc.fresh_cache();
    let out = enc.encode_into(&x, &mut cache);
    let grad_out: Vec<f32> = out.iter().zip(&target).map(|(a, b)| a - b).collect();
    let mut grad = GradAccum::new(enc.tables());
    enc.backward(&cache, &grad_out, &mut grad);

    // --- finite-difference over every touched (level, row, feat) ---
    let eps = 1e-3f32;
    let mut max_abs_err = 0.0f32;
    let mut checked = 0usize;

    // Collect unique touched rows per level from the cache.
    for (l, corners) in cache.per_level.iter().enumerate() {
        let mut rows: Vec<usize> = corners.iter().map(|&(r, _)| r).collect();
        rows.sort_unstable();
        rows.dedup();
        for &row in &rows {
            for feat in 0..f {
                let analytic = grad.value(l, row, feat);

                // L(θ + ε)
                enc.tables_mut().row_mut(l, row)[feat] += eps;
                let lp = loss(&enc.encode(&x), &target);
                // L(θ - ε)
                enc.tables_mut().row_mut(l, row)[feat] -= 2.0 * eps;
                let lm = loss(&enc.encode(&x), &target);
                // restore
                enc.tables_mut().row_mut(l, row)[feat] += eps;

                let numeric = (lp - lm) / (2.0 * eps);
                let err = (analytic - numeric).abs();
                max_abs_err = max_abs_err.max(err);
                checked += 1;
            }
        }
    }

    assert!(checked > 0, "no table entries were checked");
    assert!(
        max_abs_err < 1e-3,
        "analytic vs finite-difference gradient mismatch: max |err| = {max_abs_err} over {checked} entries"
    );
}
