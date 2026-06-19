//! Paged KV cache serving adapter (ADR-258 Phase 6 integration).
//!
//! [`PagedKvCacheManager`] bridges the request-keyed serving layer onto the
//! block-paged [`PagedKvCache`] + [`BatchScheduler`]. It is the opt-in
//! (`paged-kv` feature) counterpart to the default slot-based
//! [`super::kv_cache_manager::KvCacheManager`]: same lifecycle surface
//! (allocate → extend → free → stats) but with automatic block-aligned prefix
//! sharing, copy-on-write forking, and watermark-based admission/preemption.
//!
//! Requests are identified by [`RequestId`] (a UUID); internally each maps to a
//! dense `SeqId` so the cache and scheduler stay compact and `Copy`-friendly.
//!
//! This module deliberately does **not** replace the engine's existing manager;
//! it is a parallel path an engine build can select, keeping the contiguous
//! baseline available for comparison and rollback.

use super::request::RequestId;
use crate::error::{Result, RuvLLMError};
use crate::paged_kv::{
    AdmitOutcome, AttentionParams, BatchScheduler, CpuPagedAttention, PagedKvCache, PagedKvConfig,
    PagedKvStats, SchedulerConfig, SchedulerStats, SeqId,
};
use crate::paged_kv::BlockAttention;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Request-keyed manager over a paged KV cache and its batch scheduler.
pub struct PagedKvCacheManager {
    cache: Arc<PagedKvCache>,
    scheduler: Mutex<BatchScheduler>,
    /// Bidirectional RequestId (UUID) ↔ dense SeqId mapping. Both directions are
    /// kept in step so the serving layer can translate scheduler preemptions
    /// (reported as `SeqId`) back to `RequestId`.
    seq_map: Mutex<SeqMap>,
    next_seq: AtomicU64,
}

/// Bidirectional request↔seq mapping.
#[derive(Default)]
struct SeqMap {
    fwd: HashMap<RequestId, SeqId>,
    rev: HashMap<SeqId, RequestId>,
}

impl SeqMap {
    fn insert(&mut self, req: RequestId, seq: SeqId) {
        self.fwd.insert(req, seq);
        self.rev.insert(seq, req);
    }
    fn remove_req(&mut self, req: RequestId) -> Option<SeqId> {
        let seq = self.fwd.remove(&req)?;
        self.rev.remove(&seq);
        Some(seq)
    }
    fn remove_seq(&mut self, seq: SeqId) -> Option<RequestId> {
        let req = self.rev.remove(&seq)?;
        self.fwd.remove(&req);
        Some(req)
    }
}

impl PagedKvCacheManager {
    /// Build a manager with the given cache and scheduler configuration.
    pub fn new(cache_config: PagedKvConfig, sched_config: SchedulerConfig) -> Self {
        let cache = Arc::new(PagedKvCache::new(cache_config));
        let scheduler = BatchScheduler::new(cache.clone(), sched_config);
        Self {
            cache,
            scheduler: Mutex::new(scheduler),
            seq_map: Mutex::new(SeqMap::default()),
            next_seq: AtomicU64::new(1),
        }
    }

    /// Shared handle to the underlying cache (for attention kernels / telemetry).
    pub fn cache(&self) -> &Arc<PagedKvCache> {
        &self.cache
    }

    /// Resolve (or assign) the dense `SeqId` for a request.
    fn seq_for(&self, req: RequestId) -> SeqId {
        let mut map = self.seq_map.lock();
        if let Some(&seq) = map.fwd.get(&req) {
            return seq;
        }
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        map.insert(req, seq);
        seq
    }

    fn existing_seq(&self, req: RequestId) -> Result<SeqId> {
        self.seq_map
            .lock()
            .fwd
            .get(&req)
            .copied()
            .ok_or_else(|| RuvLLMError::NotFound(format!("request {req} not allocated")))
    }

    /// Translate a `SeqId` (e.g. a scheduler preemption victim) back to its
    /// `RequestId`, if still mapped.
    pub fn request_for_seq(&self, seq: SeqId) -> Option<RequestId> {
        self.seq_map.lock().rev.get(&seq).copied()
    }

    /// Clear the mapping for a sequence the scheduler already preempted (its
    /// blocks are freed by the preemption path, so this must **not** re-free).
    /// Returns the `RequestId` that owned it.
    pub fn forget_preempted_seq(&self, seq: SeqId) -> Option<RequestId> {
        self.seq_map.lock().remove_seq(seq)
    }

    /// Admit and prefill a request's prompt, sharing the longest cached prefix.
    ///
    /// `keys`/`values` are token-major (`tokens.len() * token_stride` each).
    /// Returns the admission outcome (whether admitted, shared-prefix tokens,
    /// any preempted requests). A queued (not-admitted) result means the engine
    /// should retry on a later step.
    pub fn admit_prefill(
        &self,
        req: RequestId,
        tokens: &[u32],
        keys: &[f32],
        values: &[f32],
    ) -> Result<AdmitOutcome> {
        let seq = self.seq_for(req);
        let outcome = self.scheduler.lock().admit(seq, tokens, keys, values)?;
        Ok(outcome)
    }

    /// Append decode-step tokens to a running request. Checks the block budget
    /// first; on insufficient room returns `OutOfMemory` so the engine can defer
    /// or trigger preemption elsewhere.
    pub fn extend(
        &self,
        req: RequestId,
        tokens: &[u32],
        keys: &[f32],
        values: &[f32],
    ) -> Result<()> {
        let seq = self.existing_seq(req)?;
        {
            let sched = self.scheduler.lock();
            if !sched.can_grow(seq, tokens.len()) {
                return Err(RuvLLMError::OutOfMemory(format!(
                    "no block budget to extend request {req} by {} tokens",
                    tokens.len()
                )));
            }
        }
        self.cache.append(seq, tokens, keys, values)
    }

    /// Logical→physical block table (physical block ids) for a request.
    pub fn block_table(&self, req: RequestId) -> Option<Vec<u32>> {
        let seq = self.seq_map.lock().fwd.get(&req).copied()?;
        self.cache
            .block_table(seq)
            .map(|t| t.blocks().iter().map(|b| b.0).collect())
    }

    /// Run the CPU paged-attention kernel for one decode step of `req`.
    pub fn attention(
        &self,
        req: RequestId,
        query: &[f32],
        num_heads: usize,
    ) -> Result<Vec<f32>> {
        let seq = self.existing_seq(req)?;
        let params = AttentionParams::from_cache(&self.cache, num_heads);
        CpuPagedAttention.forward(&self.cache, seq, query, params)
    }

    /// Free a request, retiring it from the scheduler and releasing its blocks
    /// (shared prefix blocks survive for other requests).
    pub fn free(&self, req: RequestId) -> Result<()> {
        let seq = self.seq_map.lock().remove_req(req);
        if let Some(seq) = seq {
            self.scheduler.lock().finish(seq)?;
        }
        Ok(())
    }

    /// Free physical blocks currently available in the pool.
    pub fn available_blocks(&self) -> usize {
        self.cache.stats().pool.free_blocks
    }

    /// Combined cache + scheduler telemetry.
    pub fn stats(&self) -> PagedKvManagerStats {
        PagedKvManagerStats {
            cache: self.cache.stats(),
            scheduler: self.scheduler.lock().stats(),
            tracked_requests: self.seq_map.lock().fwd.len(),
        }
    }
}

/// Aggregate telemetry for the paged serving manager.
#[derive(Debug, Clone, Copy)]
pub struct PagedKvManagerStats {
    /// Underlying paged cache stats.
    pub cache: PagedKvStats,
    /// Batch scheduler stats.
    pub scheduler: SchedulerStats,
    /// Requests with a live SeqId mapping.
    pub tracked_requests: usize,
}

impl PagedKvManagerStats {
    /// Pool block utilization in `0.0..=1.0`.
    pub fn block_utilization(&self) -> f32 {
        self.cache.pool.utilization()
    }
}

#[cfg(test)]
mod paged_manager_tests {
    use super::*;

    fn cfg(total_blocks: usize) -> PagedKvConfig {
        PagedKvConfig {
            block_size: 4,
            num_kv_heads: 1,
            head_dim: 2,
            total_blocks,
            verify_prefix_tokens: false,
        }
    }

    fn kv(tokens: &[u32], stride: usize) -> (Vec<f32>, Vec<f32>) {
        let n = tokens.len();
        (vec![0.5; n * stride], vec![0.25; n * stride])
    }

    #[test]
    fn prefill_extend_free_lifecycle() {
        let config = cfg(64);
        let stride = config.token_stride();
        let mgr = PagedKvCacheManager::new(config, SchedulerConfig::default());
        let req = RequestId::new();
        let prompt: Vec<u32> = (1000..1008).collect(); // 8 tokens, 2 blocks
        let (k, v) = kv(&prompt, stride);
        let out = mgr.admit_prefill(req, &prompt, &k, &v).unwrap();
        assert!(out.admitted);
        assert_eq!(mgr.block_table(req).unwrap().len(), 2);

        // Decode one token.
        let dec = [2000u32];
        let (dk, dv) = kv(&dec, stride);
        mgr.extend(req, &dec, &dk, &dv).unwrap();
        // 9 tokens -> 3 blocks now.
        assert_eq!(mgr.block_table(req).unwrap().len(), 3);

        mgr.free(req).unwrap();
        assert!(mgr.block_table(req).is_none());
        assert_eq!(mgr.stats().cache.pool.allocated_blocks, 0);
    }

    /// The headline Phase 6 win: N requests sharing a long prefix consume far
    /// fewer blocks than N independent contiguous allocations would.
    #[test]
    fn shared_prefix_amortizes_blocks_across_requests() {
        let config = cfg(256);
        let stride = config.token_stride();
        let mgr = PagedKvCacheManager::new(config.clone(), SchedulerConfig::default());

        // 16-token shared system prompt (4 blocks) + 4-token unique suffix.
        let shared: Vec<u32> = (5000..5016).collect();
        let n_requests = 8;
        let mut reqs = Vec::new();
        for i in 0..n_requests {
            let mut toks = shared.clone();
            toks.extend_from_slice(&[9000 + i, 9001 + i, 9002 + i, 9003 + i]);
            let (k, v) = kv(&toks, stride);
            let req = RequestId::new();
            let out = mgr.admit_prefill(req, &toks, &k, &v).unwrap();
            assert!(out.admitted);
            reqs.push(req);
        }

        let st = mgr.stats();
        // Contiguous baseline would need 8 * 5 = 40 blocks (5 blocks/req).
        // Paged: 4 shared prefix blocks + 8 unique suffix blocks = 12.
        assert_eq!(st.cache.pool.allocated_blocks, 12);
        assert!(st.cache.prefix_hit_tokens >= 16 * (n_requests as u64 - 1));

        // All requests still reconstruct their full 20-token context.
        for &req in &reqs {
            let bt = mgr.block_table(req).unwrap();
            assert_eq!(bt.len(), 5); // 20 tokens / 4 = 5 logical blocks
        }
    }

    #[test]
    fn attention_runs_through_manager() {
        let config = cfg(32);
        let stride = config.token_stride();
        let mgr = PagedKvCacheManager::new(config, SchedulerConfig::default());
        let req = RequestId::new();
        let prompt: Vec<u32> = (0..6).collect();
        let (k, v) = kv(&prompt, stride);
        mgr.admit_prefill(req, &prompt, &k, &v).unwrap();
        let query = vec![0.3f32, -0.1]; // num_heads=1, head_dim=2
        let out = mgr.attention(req, &query, 1).unwrap();
        assert_eq!(out.len(), 2);
    }
}
