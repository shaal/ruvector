//! Paged block-based KV cache micro-benchmarks (ADR-258 Phase 5).
//!
//! Measures the allocator hot paths the design promises to be cheap:
//! - block allocation / free throughput (target: < 100 ns/op)
//! - token append (block fill + seal)
//! - prefix-shared sequence admission (the ADR-011 fast path)
//! - fork + copy-on-write divergence (the agent-branching primitive)
//! - full-context gather (attention reference path)
//!
//! Run with: cargo bench -p ruvllm --no-default-features --features minimal \
//!   --bench paged_kv_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvllm::paged_kv::{BatchScheduler, PagedKvCache, PagedKvConfig, SchedulerConfig};
use std::sync::Arc;

fn config() -> PagedKvConfig {
    PagedKvConfig {
        block_size: 16,
        num_kv_heads: 8,
        head_dim: 128,
        total_blocks: 16_384,
        verify_prefix_tokens: false,
    }
}

/// Build token-major KV payloads of `n` tokens.
fn kv(n: usize, stride: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let tokens: Vec<u32> = (0..n as u32).collect();
    let keys = vec![0.5f32; n * stride];
    let values = vec![0.25f32; n * stride];
    (tokens, keys, values)
}

fn bench_allocate_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_kv/allocate_free_cycle");
    let cfg = config();
    let stride = cfg.token_stride();
    let (tokens, keys, values) = kv(cfg.block_size, stride); // exactly one block
    group.throughput(Throughput::Elements(1));
    group.bench_function("alloc_append_one_block_free", |b| {
        let cache = PagedKvCache::new(cfg.clone());
        let mut seq: u64 = 0;
        b.iter(|| {
            seq += 1;
            cache.allocate_sequence(seq).unwrap();
            cache
                .append(seq, black_box(&tokens), black_box(&keys), black_box(&values))
                .unwrap();
            cache.free_sequence(seq).unwrap();
        });
    });
    group.finish();
}

fn bench_append_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_kv/append");
    let cfg = config();
    let stride = cfg.token_stride();
    for &n in &[64usize, 256, 1024] {
        let (tokens, keys, values) = kv(n, stride);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let cache = PagedKvCache::new(cfg.clone());
            let mut seq: u64 = 0;
            b.iter(|| {
                seq += 1;
                cache.allocate_sequence(seq).unwrap();
                cache.append(seq, &tokens, &keys, &values).unwrap();
                cache.free_sequence(seq).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_prefix_sharing(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_kv/prefix_admission");
    let cfg = config();
    let stride = cfg.token_stride();
    // A 512-token shared prefix (e.g. system prompt + tool schema).
    let prefix_len = 512;
    let (ptokens, pkeys, pvalues) = kv(prefix_len, stride);
    group.throughput(Throughput::Elements(prefix_len as u64));
    group.bench_function("admit_with_512tok_shared_prefix", |b| {
        let cache = PagedKvCache::new(cfg.clone());
        // Seed sequence 0 holding the prefix so the index is warm.
        cache.allocate_sequence(0).unwrap();
        cache.append(0, &ptokens, &pkeys, &pvalues).unwrap();
        let mut seq: u64 = 1;
        b.iter(|| {
            seq += 1;
            // The whole prefix is served from cache: zero KV recompute.
            let shared = cache.allocate_with_prefix(seq, black_box(&ptokens)).unwrap();
            debug_assert_eq!(shared, prefix_len);
            cache.free_sequence(seq).unwrap();
        });
    });
    group.finish();
}

fn bench_fork_cow(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_kv/fork_cow");
    let cfg = config();
    let stride = cfg.token_stride();
    let (tokens, keys, values) = kv(256, stride); // 256-token parent context
    let (etoks, ekeys, evals) = kv(1, stride); // one divergent token
    group.bench_function("fork_then_diverge_one_token", |b| {
        let cache = PagedKvCache::new(cfg.clone());
        cache.allocate_sequence(0).unwrap();
        cache.append(0, &tokens, &keys, &values).unwrap();
        let mut child: u64 = 1;
        b.iter(|| {
            child += 1;
            cache.fork(0, child).unwrap();
            cache.append(child, &etoks, &ekeys, &evals).unwrap();
            cache.free_sequence(child).unwrap();
        });
    });
    group.finish();
}

fn bench_gather(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_kv/gather");
    let cfg = config();
    let stride = cfg.token_stride();
    for &n in &[256usize, 1024] {
        let (tokens, keys, values) = kv(n, stride);
        let cache = PagedKvCache::new(cfg.clone());
        cache.allocate_sequence(1).unwrap();
        cache.append(1, &tokens, &keys, &values).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let (k, v, m) = cache.gather_kv(1).unwrap();
                black_box((k, v, m));
            });
        });
    }
    group.finish();
}

/// End-to-end serving workload: admit a wave of agent requests that all share a
/// long system-prompt prefix, then retire them. This is the high-sharing
/// scenario the ADR targets — admission cost is dominated by block-aligned
/// prefix reuse (zero KV recompute for the shared 512 tokens), and the pool only
/// allocates one suffix block per request rather than a full contiguous context.
fn bench_serving_high_sharing(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_kv/serving_high_sharing");
    let cfg = config();
    let stride = cfg.token_stride();
    let prefix_len = 512; // shared system prompt + tool schema
    let suffix_len = 16; // per-request unique turn
    for &wave in &[16usize, 64] {
        group.throughput(Throughput::Elements(wave as u64));
        group.bench_with_input(BenchmarkId::from_parameter(wave), &wave, |b, _| {
            b.iter_batched(
                || {
                    let cache = Arc::new(PagedKvCache::new(cfg.clone()));
                    let sched = BatchScheduler::new(cache.clone(), SchedulerConfig::default());
                    // Warm the prefix index with one seed sequence.
                    let ptoks: Vec<u32> = (0..prefix_len as u32).collect();
                    let pk = vec![0.5f32; prefix_len * stride];
                    let pv = vec![0.25f32; prefix_len * stride];
                    cache.allocate_sequence(0).unwrap();
                    cache.append(0, &ptoks, &pk, &pv).unwrap();
                    (cache, sched, ptoks)
                },
                |(cache, mut sched, ptoks)| {
                    for r in 1..=wave as u64 {
                        // Same 512-token prefix, unique suffix per request.
                        let mut toks = ptoks.clone();
                        toks.extend((0..suffix_len as u32).map(|i| 1_000_000 + r as u32 * 100 + i));
                        let n = toks.len();
                        let k = vec![0.5f32; n * stride];
                        let v = vec![0.25f32; n * stride];
                        let out = sched.admit(r, &toks, &k, &v).unwrap();
                        black_box(&out);
                    }
                    black_box(cache.stats());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_allocate_free,
    bench_append_scaling,
    bench_prefix_sharing,
    bench_fork_cow,
    bench_gather,
    bench_serving_high_sharing
);
criterion_main!(benches);
