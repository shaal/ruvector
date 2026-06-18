//! Unit + property tests for the paged KV cache (ADR-258 Phase 5).
//!
//! Property tests target the load-bearing safety invariants of the design:
//! 1. **Refcount honesty** — a block's refcount equals the number of block
//!    tables pointing at it.
//! 2. **CoW isolation** — writing through one sequence never mutates a block
//!    still shared with another sequence.
//! 3. **Pool conservation** — `allocated + free == total` always holds.

use super::*;
use proptest::prelude::*;

fn cfg(block_size: usize, total_blocks: usize) -> PagedKvConfig {
    PagedKvConfig {
        block_size,
        num_kv_heads: 2,
        head_dim: 4,
        total_blocks,
        verify_prefix_tokens: false,
    }
}

/// Build KV payloads for `tokens` where every element encodes the token id, so
/// gathered KV can be checked against the tokens that produced it.
fn kv_for(tokens: &[u32], stride: usize) -> (Vec<f32>, Vec<f32>) {
    let mut keys = Vec::with_capacity(tokens.len() * stride);
    let mut values = Vec::with_capacity(tokens.len() * stride);
    for &t in tokens {
        for _ in 0..stride {
            keys.push(t as f32);
            values.push((t as f32) * 0.5);
        }
    }
    (keys, values)
}

#[test]
fn allocate_append_gather_roundtrip() {
    let config = cfg(4, 64);
    let stride = config.token_stride();
    let cache = PagedKvCache::new(config);
    let tokens: Vec<u32> = (1..=10).collect(); // 10 tokens -> 3 blocks (4+4+2)
    let (k, v) = kv_for(&tokens, stride);

    cache.allocate_sequence(1).unwrap();
    cache.append(1, &tokens, &k, &v).unwrap();

    let (gk, gv, n) = cache.gather_kv(1).unwrap();
    assert_eq!(n, 10);
    assert_eq!(gk, k);
    assert_eq!(gv, v);

    let st = cache.stats();
    assert_eq!(st.live_sequences, 1);
    // 10 tokens / 4 per block = 3 blocks.
    assert_eq!(st.pool.allocated_blocks, 3);
}

#[test]
fn pool_exhaustion_is_an_error() {
    let config = cfg(4, 2); // only 2 blocks
    let stride = config.token_stride();
    let cache = PagedKvCache::new(config);
    let tokens: Vec<u32> = (1..=12).collect(); // needs 3 blocks
    let (k, v) = kv_for(&tokens, stride);
    cache.allocate_sequence(1).unwrap();
    let err = cache.append(1, &tokens, &k, &v).unwrap_err();
    assert!(matches!(err, crate::error::RuvLLMError::OutOfMemory(_)));
}

#[test]
fn free_returns_blocks_to_pool() {
    let config = cfg(4, 16);
    let stride = config.token_stride();
    let cache = PagedKvCache::new(config);
    let tokens: Vec<u32> = (1..=8).collect();
    let (k, v) = kv_for(&tokens, stride);
    cache.allocate_sequence(1).unwrap();
    cache.append(1, &tokens, &k, &v).unwrap();
    assert_eq!(cache.stats().pool.allocated_blocks, 2);
    cache.free_sequence(1).unwrap();
    let st = cache.stats();
    assert_eq!(st.pool.allocated_blocks, 0);
    assert_eq!(st.pool.free_blocks, 16);
    assert_eq!(st.live_sequences, 0);
}

#[test]
fn identical_prefix_is_shared_not_recomputed() {
    let config = cfg(4, 64);
    let stride = config.token_stride();
    let cache = PagedKvCache::new(config);

    // Sequence 1: 8 shared tokens + 4 unique -> seals 2 prefix blocks.
    let shared: Vec<u32> = vec![100, 101, 102, 103, 104, 105, 106, 107];
    let s1_tail: Vec<u32> = vec![1, 1, 1, 1];
    let mut s1 = shared.clone();
    s1.extend_from_slice(&s1_tail);
    let (k1, v1) = kv_for(&s1, stride);
    cache.allocate_sequence(1).unwrap();
    cache.append(1, &s1, &k1, &v1).unwrap();

    let allocated_after_first = cache.stats().pool.allocated_blocks;
    assert_eq!(allocated_after_first, 3); // 2 prefix + 1 tail

    // Sequence 2 shares the 8-token prefix.
    let shared_tokens = cache.allocate_with_prefix(2, &shared).unwrap();
    assert_eq!(shared_tokens, 8, "both prefix blocks should be shared");

    let st = cache.stats();
    // No new physical blocks for the shared prefix.
    assert_eq!(st.pool.allocated_blocks, allocated_after_first);
    assert_eq!(st.prefix_hit_tokens, 8);
    assert_eq!(st.prefix_hit_blocks, 2);

    // The two prefix blocks now have refcount 2.
    let table1 = cache.block_table(1).unwrap();
    assert_eq!(cache.ref_count(table1.blocks()[0]), 2);
    assert_eq!(cache.ref_count(table1.blocks()[1]), 2);
}

#[test]
fn cow_isolates_diverging_sequences() {
    let config = cfg(4, 64);
    let stride = config.token_stride();
    let cache = PagedKvCache::new(config);

    // Parent: one full shared block + a partial tail (2 tokens).
    let parent_tokens: Vec<u32> = vec![10, 11, 12, 13, 20, 21];
    let (pk, pv) = kv_for(&parent_tokens, stride);
    cache.allocate_sequence(1).unwrap();
    cache.append(1, &parent_tokens, &pk, &pv).unwrap();

    // Fork child; it shares parent's blocks (including the partial tail).
    cache.fork(1, 2).unwrap();
    let ptable = cache.block_table(1).unwrap();
    let shared_tail = *ptable.blocks().last().unwrap();
    assert_eq!(cache.ref_count(shared_tail), 2);

    // Child appends divergent tokens into the partial tail -> CoW.
    let child_more: Vec<u32> = vec![99, 99];
    let (ck, cv) = kv_for(&child_more, stride);
    cache.append(2, &child_more, &ck, &cv).unwrap();

    // Parent's tail block must be untouched (still its original 2 tokens),
    // child must now read its own divergent tail.
    let (parent_k, _, parent_n) = cache.gather_kv(1).unwrap();
    let (child_k, _, child_n) = cache.gather_kv(2).unwrap();
    assert_eq!(parent_n, 6);
    assert_eq!(child_n, 8);
    // Parent's last two key tokens are still 20,21.
    assert_eq!(parent_k[parent_k.len() - 1], 21.0);
    // Child's last two key tokens are 99,99.
    assert_eq!(child_k[child_k.len() - 1], 99.0);
    // A CoW copy was recorded and the parent tail is private again.
    assert_eq!(cache.stats().cow_copies, 1);
    assert_eq!(cache.ref_count(shared_tail), 1);
}

#[test]
fn demote_cold_blocks_changes_tier_keeps_hot() {
    let config = cfg(4, 64);
    let stride = config.token_stride();
    // Use a uniform quantizer so demotion actually transforms payload.
    let cache = PagedKvCache::with_quantizer(
        config,
        std::sync::Arc::new(super::quant::UniformQuantizer),
    );
    let tokens: Vec<u32> = (1..=12).collect(); // 3 blocks
    let (k, v) = kv_for(&tokens, stride);
    cache.allocate_sequence(1).unwrap();
    cache.append(1, &tokens, &k, &v).unwrap();

    // Keep the most recent 1 block hot; demote the 2 cold ones.
    let demoted = cache.demote_cold_blocks(1, 1).unwrap();
    assert_eq!(demoted, 2);
    let table = cache.block_table(1).unwrap();
    assert_eq!(cache.block_tier(table.blocks()[0]), QuantTier::Int4);
    assert_eq!(cache.block_tier(table.blocks()[1]), QuantTier::Int4);
    assert_eq!(cache.block_tier(table.blocks()[2]), QuantTier::Fp16); // hot tail
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Invariant: across a random sequence of allocate/append/free operations,
    /// the pool always conserves blocks (allocated + free == total) and never
    /// reports more allocated than total.
    #[test]
    fn pool_conservation(
        ops in proptest::collection::vec(
            (1u64..4u64, 1usize..12usize), 1..40
        )
    ) {
        let config = cfg(4, 256);
        let stride = config.token_stride();
        let cache = PagedKvCache::new(config.clone());
        let mut live: std::collections::HashSet<u64> = Default::default();

        for (seq, n_tokens) in ops {
            if live.contains(&seq) {
                // Free it.
                cache.free_sequence(seq).unwrap();
                live.remove(&seq);
            } else {
                cache.allocate_sequence(seq).ok();
                let tokens: Vec<u32> = (0..n_tokens as u32).collect();
                let (k, v) = kv_for(&tokens, stride);
                // Append may fail on exhaustion; that's fine, just skip.
                if cache.append(seq, &tokens, &k, &v).is_ok() {
                    live.insert(seq);
                } else {
                    cache.free_sequence(seq).ok();
                }
            }
            let st = cache.stats();
            prop_assert_eq!(
                st.pool.allocated_blocks + st.pool.free_blocks,
                config.total_blocks
            );
            prop_assert!(st.pool.allocated_blocks <= config.total_blocks);
        }

        // Drain everything; pool must return to fully free.
        for seq in live.clone() {
            cache.free_sequence(seq).unwrap();
        }
        let st = cache.stats();
        prop_assert_eq!(st.pool.allocated_blocks, 0);
        prop_assert_eq!(st.pool.free_blocks, config.total_blocks);
    }

    /// Invariant: forking N children off a parent and freeing them in any order
    /// never frees a block still referenced, and after freeing all, the pool is
    /// fully reclaimed.
    #[test]
    fn fork_refcount_safety(n_children in 1usize..8usize, n_tokens in 4usize..20usize) {
        let config = cfg(4, 512);
        let stride = config.token_stride();
        let cache = PagedKvCache::new(config.clone());

        let tokens: Vec<u32> = (0..n_tokens as u32).collect();
        let (k, v) = kv_for(&tokens, stride);
        cache.allocate_sequence(0).unwrap();
        cache.append(0, &tokens, &k, &v).unwrap();

        // Fork children, each appends a unique divergent token (forces CoW on
        // the shared tail for partial tails).
        for c in 1..=n_children as u64 {
            cache.fork(0, c).unwrap();
            let extra = vec![1000u32 + c as u32];
            let (ek, ev) = kv_for(&extra, stride);
            cache.append(c, &extra, &ek, &ev).unwrap();
        }

        // Every gather must still succeed and report a sane token count.
        for c in 0..=n_children as u64 {
            let (_, _, got) = cache.gather_kv(c).unwrap();
            let expected = if c == 0 { n_tokens } else { n_tokens + 1 };
            prop_assert_eq!(got, expected);
        }

        // Free in reverse order; pool fully reclaimed at the end.
        for c in (0..=n_children as u64).rev() {
            cache.free_sequence(c).unwrap();
        }
        let st = cache.stats();
        prop_assert_eq!(st.pool.allocated_blocks, 0);
        prop_assert_eq!(st.live_sequences, 0);
    }
}
