//! ADR-258 paged KV cache — end-to-end engine demo.
//!
//! Drives [`PagedBatchEngine`] through a realistic agent serving wave: many
//! requests that share a long system-prompt prefix and then diverge. It prints
//! the memory, sharing, concurrency, and preemption wins the ADR predicts.
//!
//! Run with:
//! ```bash
//! cargo run -p ruvllm --no-default-features --features minimal,paged-kv \
//!   --example paged_engine_demo
//! ```
//!
//! ## Why a synthetic token generator?
//!
//! The engine is intentionally model-agnostic: token + KV production lives
//! behind the [`TokenGenerator`] trait. A production build supplies a generator
//! backed by the model's decode step (using `PagedKvCacheManager::attention`).
//! `candle-transformers` models, however, manage their KV cache *internally* and
//! do not expose per-layer K/V projections, so feeding real-model KV into the
//! paged pool would require forking their attention. This demo therefore uses a
//! deterministic generator to exercise the full allocate → share → CoW →
//! preempt → free machinery without a model dependency.

#![cfg(feature = "paged-kv")]

use ruvllm::paged_kv::{PagedKvConfig, SchedulerConfig};
use ruvllm::serving::request::RequestId;
use ruvllm::serving::{
    PagedBatchEngine, PagedBatchEngineConfig, PagedRequest, TokenGenerator,
};
use std::time::Instant;

/// Deterministic generator: emits one token's worth of constant KV per step.
struct DemoGen {
    stride: usize,
}

impl TokenGenerator for DemoGen {
    fn next_token(&mut self, _req: RequestId, seq_len: usize) -> (u32, Vec<f32>, Vec<f32>) {
        (seq_len as u32, vec![0.5; self.stride], vec![0.25; self.stride])
    }
}

fn main() {
    // A small but representative config: block_size 16, GQA 8x128.
    let cache = PagedKvConfig {
        block_size: 16,
        num_kv_heads: 8,
        head_dim: 128,
        total_blocks: 4096,
        verify_prefix_tokens: false,
    };
    let stride = cache.token_stride();
    let block_size = cache.block_size;

    let config = PagedBatchEngineConfig {
        cache,
        scheduler: SchedulerConfig {
            max_running: 512,
            watermark_blocks: 16,
            allow_preemption: true,
        },
    };
    let mut engine = PagedBatchEngine::new(config);

    // Workload: 64 agent turns sharing a 512-token system prompt + tool schema,
    // each with a unique 32-token user message, generating 16 tokens.
    let n_requests = 64usize;
    let prefix_len = 512usize;
    let suffix_len = 32usize;
    let max_new = 16usize;
    let shared_prefix: Vec<u32> = (0..prefix_len as u32).collect();

    for i in 0..n_requests {
        let mut tokens = shared_prefix.clone();
        tokens.extend((0..suffix_len as u32).map(|j| 1_000_000 + (i as u32) * 1000 + j));
        let n = tokens.len();
        engine.submit(PagedRequest {
            id: RequestId::new(),
            prompt_tokens: tokens,
            prompt_keys: vec![0.5; n * stride],
            prompt_values: vec![0.25; n * stride],
            max_new_tokens: max_new,
        });
    }

    println!("=== ADR-258 PagedBatchEngine demo ===");
    println!(
        "workload: {n_requests} requests | shared prefix {prefix_len} tok | \
         suffix {suffix_len} tok | gen {max_new} tok | block_size {block_size}"
    );

    let mut gen = DemoGen { stride };
    let mut peak_running = 0usize;
    let mut peak_blocks = 0usize;
    let mut total_preempted = 0usize;
    let mut steps = 0usize;
    let total_tokens_target = n_requests * max_new;

    let t0 = Instant::now();
    // Drive the loop until everything drains.
    loop {
        let out = engine.step(&mut gen).expect("step");
        steps += 1;
        total_preempted += out.preempted.len();
        peak_running = peak_running.max(engine.running_len());
        let st = engine.stats();
        peak_blocks = peak_blocks.max(st.cache.pool.allocated_blocks);
        if engine.pending_len() == 0 && engine.running_len() == 0 {
            break;
        }
        if steps > 100_000 {
            eprintln!("safety stop: too many steps");
            break;
        }
    }
    let elapsed = t0.elapsed();
    let st = engine.stats();

    // Contiguous baseline: each request reserves ceil((prefix+suffix)/bs) blocks,
    // no sharing, for the whole wave simultaneously.
    let per_req_blocks = (prefix_len + suffix_len).div_ceil(block_size);
    let contiguous_peak = per_req_blocks * n_requests;
    // Paged peak blocks observed (shared prefix amortized + unique suffix/decode).
    let savings = 1.0 - (peak_blocks as f64 / contiguous_peak as f64);

    println!("\n--- results ---");
    println!("steps to drain ............ {steps}");
    println!("wall time ................. {:.2?}", elapsed);
    println!(
        "throughput ................ {:.0} tokens/s ({} tokens)",
        total_tokens_target as f64 / elapsed.as_secs_f64().max(1e-9),
        total_tokens_target
    );
    println!("peak concurrent requests .. {peak_running}");
    println!("preemptions ............... {total_preempted}");
    println!("prefix lookups ............ {}", st.cache.prefix_lookups);
    println!(
        "prefix tokens shared ...... {} (~{} requests reused the prompt)",
        st.cache.prefix_hit_tokens,
        st.cache.prefix_hit_tokens / prefix_len as u64
    );
    println!("copy-on-write copies ...... {}", st.cache.cow_copies);
    println!("\n--- memory (blocks) ---");
    println!("paged peak blocks ......... {peak_blocks}");
    println!(
        "contiguous baseline peak .. {contiguous_peak}  ({per_req_blocks} blocks/req x {n_requests})"
    );
    println!(
        "fragmentation/sharing win . {:.1}% fewer blocks at peak",
        savings * 100.0
    );

    // Pool fully reclaimed at the end (no leaks).
    assert_eq!(st.cache.pool.allocated_blocks, 0, "pool should be drained");
    println!("\npool fully reclaimed: {} blocks free", st.cache.pool.free_blocks);
}
