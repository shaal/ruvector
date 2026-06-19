//! End-to-end online-learning proof (ADR-258 Phase 2): a short contrastive-style
//! training loop on both the feature tables and the learned projection must
//! monotonically reduce a regression loss — demonstrating the encoder is
//! trainable end-to-end, not merely differentiable at a point.

use ruvector_hashenc::{GradAccum, HashEncConfig, HashEncoder, ProjGrad, ProjectionKind};

fn loss(out: &[f32], target: &[f32]) -> f32 {
    0.5 * out.iter().zip(target).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
}

#[test]
fn training_reduces_loss_tables_and_projection() {
    let cfg = HashEncConfig {
        levels: 6,
        features_per_level: 2,
        log2_table_size: 14,
        index_dims: 3,
        n_min: 4,
        n_max: 128,
        projection: ProjectionKind::Learned,
        seed: 42,
    };
    let input_dim = 16;
    let mut enc = HashEncoder::new(cfg.clone(), input_dim);

    // Fixed regression targets for a few inputs.
    let inputs: Vec<Vec<f32>> = (0..8)
        .map(|s| (0..input_dim).map(|i| ((i + s) as f32 * 0.17).sin()).collect())
        .collect();
    let targets: Vec<Vec<f32>> = (0..8)
        .map(|s| (0..cfg.output_dim()).map(|i| 0.2 * ((i * 3 + s) as f32 * 0.11).cos()).collect())
        .collect();

    let total_loss = |enc: &HashEncoder| -> f32 {
        inputs.iter().zip(&targets).map(|(x, t)| loss(&enc.encode(x), t)).sum::<f32>()
    };

    let before = total_loss(&enc);

    let mut tg = GradAccum::new(enc.tables());
    let mut pg = ProjGrad::new(enc.projection());
    let (lr_t, lr_p) = (0.5f32, 0.2f32);

    for _ in 0..200 {
        for (x, t) in inputs.iter().zip(&targets) {
            let mut cache = enc.fresh_cache();
            let out = enc.encode_into(x, &mut cache);
            let grad_out: Vec<f32> = out.iter().zip(t).map(|(a, b)| a - b).collect();
            enc.backward(&cache, &grad_out, &mut tg);
            enc.projection_grad(x, &grad_out, &mut pg);
            enc.apply_projection_grad(&mut pg, lr_p);
            tg.apply(enc.tables_mut(), lr_t);
        }
    }

    let after = total_loss(&enc);
    assert!(
        after < before * 0.5,
        "training did not reduce loss enough: before={before}, after={after}"
    );
}
