//! Criterion microbenchmarks for the hash encoder forward/backward path
//! (ADR-258 §8, feeds success criteria S4/S5/S7).

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ruvector_hashenc::{GradAccum, HashEncConfig, HashEncoder};

fn bench_encode(c: &mut Criterion) {
    let dims = [128usize, 384, 768];
    let mut group = c.benchmark_group("hashenc_encode");
    for &d in &dims {
        let enc = HashEncoder::new(HashEncConfig::default(), d);
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.017).sin()).collect();
        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("encode_d{d}"), |b| {
            let mut cache = enc.fresh_cache();
            b.iter(|| black_box(enc.encode_into(black_box(&x), &mut cache)));
        });
    }
    group.finish();
}

fn bench_forward_backward(c: &mut Criterion) {
    let d = 384usize;
    let enc = HashEncoder::new(HashEncConfig::default(), d);
    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.017).sin()).collect();
    let target = vec![0.0f32; enc.output_dim()];
    let mut grad = GradAccum::new(enc.tables());

    c.bench_function("hashenc_forward_backward_d384", |b| {
        let mut cache = enc.fresh_cache();
        b.iter(|| {
            let out = enc.encode_into(black_box(&x), &mut cache);
            let go: Vec<f32> = out.iter().zip(&target).map(|(a, b)| a - b).collect();
            enc.backward(&cache, &go, &mut grad);
            black_box(&grad);
        });
    });
}

criterion_group!(benches, bench_encode, bench_forward_backward);
criterion_main!(benches);
