//! Continuous-batching scheduler hooks (ADR-258 Phase 4).
//!
//! A block-budget admission controller over a [`PagedKvCache`]. It decides which
//! sequences run this step and preempts under memory pressure, which is the
//! piece that turns paged KV into *higher concurrency*: because blocks are
//! fungible and prefixes are shared, far more sequences fit in the same pool,
//! and the scheduler safely oversubscribes up to a watermark.
//!
//! Preemption here uses the **recompute** policy (vLLM's default): a preempted
//! sequence's blocks are freed and it returns to the waiting queue; the engine
//! re-prefills it later (cheaply, since its prefix is likely still shared in the
//! index). A swap-to-host policy is a future extension behind the same API.
//!
//! The scheduler tracks sequence *ids* and budget only — it never owns KV
//! tensors — so admission is caller-driven: supply prompt KV to [`Self::admit`];
//! on [`AdmitOutcome::queued`] retry on a later step.

use super::{PagedKvCache, SeqId};
use crate::error::Result;
use std::collections::VecDeque;
use std::sync::Arc;

/// Scheduler tuning.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// Hard cap on concurrently running sequences.
    pub max_running: usize,
    /// Blocks kept free as headroom so in-flight sequences can always grow by at
    /// least one block next step (deadlock avoidance).
    pub watermark_blocks: usize,
    /// When `true`, admission under pressure may preempt the most recently
    /// admitted running sequence(s) to make room (recompute policy).
    pub allow_preemption: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_running: 256,
            watermark_blocks: 8,
            allow_preemption: true,
        }
    }
}

/// Outcome of an [`BatchScheduler::admit`] attempt.
#[derive(Debug, Clone, Default)]
pub struct AdmitOutcome {
    /// `true` if the sequence is now running and prefilled.
    pub admitted: bool,
    /// Tokens served from the prefix index (zero-recompute), if admitted.
    pub shared_tokens: usize,
    /// Sequences preempted to make room (now back in the waiting queue).
    pub preempted: Vec<SeqId>,
}

impl AdmitOutcome {
    /// `true` if admission failed and the sequence was (re)queued.
    pub fn queued(&self) -> bool {
        !self.admitted
    }
}

/// Aggregate scheduler telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct SchedulerStats {
    /// Sequences currently running.
    pub running: usize,
    /// Sequences waiting for admission.
    pub waiting: usize,
    /// Total admissions performed.
    pub admitted_total: u64,
    /// Total preemptions performed.
    pub preempted_total: u64,
}

/// Block-budget admission + preemption controller.
pub struct BatchScheduler {
    cache: Arc<PagedKvCache>,
    cfg: SchedulerConfig,
    waiting: VecDeque<SeqId>,
    running: Vec<SeqId>,
    admitted_total: u64,
    preempted_total: u64,
}

impl BatchScheduler {
    /// Create a scheduler over `cache`.
    pub fn new(cache: Arc<PagedKvCache>, cfg: SchedulerConfig) -> Self {
        Self {
            cache,
            cfg,
            waiting: VecDeque::new(),
            running: Vec::new(),
            admitted_total: 0,
            preempted_total: 0,
        }
    }

    /// Blocks a prompt of `num_tokens` needs in the worst case (no sharing).
    #[inline]
    fn blocks_for(&self, num_tokens: usize) -> usize {
        let bs = self.cache.config().block_size;
        num_tokens.div_ceil(bs)
    }

    /// Free blocks currently available in the pool.
    #[inline]
    fn free_blocks(&self) -> usize {
        self.cache.stats().pool.free_blocks
    }

    /// Whether `blocks` can be allocated while preserving the watermark.
    #[inline]
    fn fits(&self, blocks: usize) -> bool {
        self.free_blocks() >= blocks + self.cfg.watermark_blocks
    }

    /// Attempt to admit and prefill `seq_id` with the given prompt KV.
    ///
    /// `keys`/`values` are token-major (`num_tokens * token_stride` each). On
    /// success the sequence is allocated, the longest cached prefix is shared,
    /// and the remaining suffix KV is appended. On failure the id is queued.
    pub fn admit(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        keys: &[f32],
        values: &[f32],
    ) -> Result<AdmitOutcome> {
        let mut outcome = AdmitOutcome::default();
        let need = self.blocks_for(tokens.len());

        // Capacity / budget gates, with optional preemption to make room.
        if self.running.len() >= self.cfg.max_running || !self.fits(need) {
            if self.cfg.allow_preemption {
                outcome.preempted = self.preempt_until(need)?;
            }
            if self.running.len() >= self.cfg.max_running || !self.fits(need) {
                if !self.waiting.contains(&seq_id) {
                    self.waiting.push_back(seq_id);
                }
                return Ok(outcome);
            }
        }

        // Admit: share prefix, then append the non-shared suffix.
        let stride = self.cache.config().token_stride();
        let shared = self.cache.allocate_with_prefix(seq_id, tokens)?;
        if shared < tokens.len() {
            self.cache.append(
                seq_id,
                &tokens[shared..],
                &keys[shared * stride..],
                &values[shared * stride..],
            )?;
        }
        // Remove from waiting if it was a retry, mark running.
        self.waiting.retain(|&s| s != seq_id);
        self.running.push(seq_id);
        self.admitted_total += 1;
        outcome.admitted = true;
        outcome.shared_tokens = shared;
        Ok(outcome)
    }

    /// Reserve budget for a running sequence to append `num_new_tokens` this
    /// step. Returns `true` if it fits (caller may then append); on `false` the
    /// caller should defer the step or the scheduler should preempt elsewhere.
    pub fn can_grow(&self, seq_id: SeqId, num_new_tokens: usize) -> bool {
        let bs = self.cache.config().block_size;
        let tail = self
            .cache
            .block_table(seq_id)
            .map(|t| {
                let n = t.num_tokens();
                if n == 0 {
                    0
                } else {
                    let r = n % bs;
                    if r == 0 {
                        0
                    } else {
                        bs - r
                    }
                }
            })
            .unwrap_or(0);
        let extra = num_new_tokens.saturating_sub(tail);
        let need = extra.div_ceil(bs);
        self.fits(need)
    }

    /// Preempt running sequences (most-recent-first) until `need` blocks plus the
    /// watermark are free, or nothing is left to preempt. Returns preempted ids.
    fn preempt_until(&mut self, need: usize) -> Result<Vec<SeqId>> {
        let mut preempted = Vec::new();
        while !self.fits(need) {
            let Some(victim) = self.running.pop() else {
                break;
            };
            self.cache.free_sequence(victim)?;
            self.waiting.push_front(victim);
            self.preempted_total += 1;
            preempted.push(victim);
        }
        Ok(preempted)
    }

    /// Retire a finished sequence: free its blocks and drop it from running.
    pub fn finish(&mut self, seq_id: SeqId) -> Result<()> {
        self.cache.free_sequence(seq_id)?;
        self.running.retain(|&s| s != seq_id);
        self.waiting.retain(|&s| s != seq_id);
        Ok(())
    }

    /// Next waiting sequence id to retry admission for, if any.
    pub fn next_waiting(&self) -> Option<SeqId> {
        self.waiting.front().copied()
    }

    /// Running sequence ids (for the engine to iterate this step).
    pub fn running(&self) -> &[SeqId] {
        &self.running
    }

    /// Telemetry snapshot.
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            running: self.running.len(),
            waiting: self.waiting.len(),
            admitted_total: self.admitted_total,
            preempted_total: self.preempted_total,
        }
    }
}

#[cfg(test)]
mod scheduler_tests {
    use super::*;
    use crate::paged_kv::PagedKvConfig;

    fn cfg(total_blocks: usize) -> PagedKvConfig {
        PagedKvConfig {
            block_size: 4,
            num_kv_heads: 1,
            head_dim: 2,
            total_blocks,
            verify_prefix_tokens: false,
        }
    }

    fn kv(n: usize, stride: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let tokens: Vec<u32> = (0..n as u32).collect();
        (tokens, vec![0.5; n * stride], vec![0.25; n * stride])
    }

    /// Distinct prompt per sequence so the prefix index does not dedup them
    /// (which would mask budget pressure — itself a feature, tested separately).
    fn kv_unique(seq: u64, n: usize, stride: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let base = (seq as u32) * 1000;
        let tokens: Vec<u32> = (0..n as u32).map(|i| base + i).collect();
        (tokens, vec![0.5; n * stride], vec![0.25; n * stride])
    }

    #[test]
    fn admits_until_budget_then_queues() {
        let config = cfg(8); // 8 blocks total
        let stride = config.token_stride();
        let cache = Arc::new(PagedKvCache::new(config));
        let mut sched = BatchScheduler::new(
            cache.clone(),
            SchedulerConfig {
                max_running: 100,
                watermark_blocks: 2,
                allow_preemption: false,
            },
        );
        // Distinct 8-token (2-block) prompts so none share. Budget: free must be
        // >= need(2) + watermark(2) = 4.
        let (t1, k1, v1) = kv_unique(1, 8, stride);
        let (t2, k2, v2) = kv_unique(2, 8, stride);
        let (t3, k3, v3) = kv_unique(3, 8, stride);
        let (t4, k4, v4) = kv_unique(4, 8, stride);
        assert!(sched.admit(1, &t1, &k1, &v1).unwrap().admitted); // free 8->6
        assert!(sched.admit(2, &t2, &k2, &v2).unwrap().admitted); // free 6->4
        // free=4 >= 2+2 -> admit third; free 4->2.
        assert!(sched.admit(3, &t3, &k3, &v3).unwrap().admitted);
        // free=2, need 2+2=4 -> queued.
        let d = sched.admit(4, &t4, &k4, &v4).unwrap();
        assert!(d.queued());
        assert_eq!(sched.next_waiting(), Some(4));
        assert_eq!(sched.stats().running, 3);
        assert_eq!(sched.stats().waiting, 1);
    }

    #[test]
    fn preemption_frees_room() {
        let config = cfg(8);
        let stride = config.token_stride();
        let cache = Arc::new(PagedKvCache::new(config));
        let mut sched = BatchScheduler::new(
            cache.clone(),
            SchedulerConfig {
                max_running: 100,
                watermark_blocks: 2,
                allow_preemption: true,
            },
        );
        let (t1, k1, v1) = kv_unique(1, 8, stride);
        let (t2, k2, v2) = kv_unique(2, 8, stride);
        let (t3, k3, v3) = kv_unique(3, 8, stride);
        let (t4, k4, v4) = kv_unique(4, 8, stride);
        sched.admit(1, &t1, &k1, &v1).unwrap();
        sched.admit(2, &t2, &k2, &v2).unwrap();
        sched.admit(3, &t3, &k3, &v3).unwrap();
        // Free = 2; admitting #4 forces preemption of the newest running (#3).
        let d = sched.admit(4, &t4, &k4, &v4).unwrap();
        assert!(d.admitted);
        assert_eq!(d.preempted, vec![3]);
        // #3 is now back in the waiting queue (recompute policy).
        assert!(sched.next_waiting() == Some(3));
        assert_eq!(sched.stats().preempted_total, 1);
    }

    #[test]
    fn finish_returns_blocks() {
        let config = cfg(16);
        let stride = config.token_stride();
        let cache = Arc::new(PagedKvCache::new(config));
        let mut sched = BatchScheduler::new(cache.clone(), SchedulerConfig::default());
        let (t, k, v) = kv(8, stride);
        sched.admit(1, &t, &k, &v).unwrap();
        assert_eq!(cache.stats().pool.allocated_blocks, 2);
        sched.finish(1).unwrap();
        assert_eq!(cache.stats().pool.allocated_blocks, 0);
        assert_eq!(sched.stats().running, 0);
    }

    #[test]
    fn shared_prefix_reduces_pressure() {
        let config = cfg(32);
        let stride = config.token_stride();
        let cache = Arc::new(PagedKvCache::new(config));
        let mut sched = BatchScheduler::new(cache.clone(), SchedulerConfig::default());
        // 8-token shared prefix + unique suffix.
        let prefix: Vec<u32> = (100..108).collect();
        let mut s1 = prefix.clone();
        s1.extend_from_slice(&[1, 2, 3, 4]);
        let (_, k1, v1) = {
            let n = s1.len();
            ((), vec![0.5; n * stride], vec![0.25; n * stride])
        };
        sched.admit(1, &s1, &k1, &v1).unwrap();
        let before = cache.stats().pool.allocated_blocks;

        // Second sequence shares the 8-token (2-block) prefix.
        let mut s2 = prefix.clone();
        s2.extend_from_slice(&[9, 9, 9, 9]);
        let n2 = s2.len();
        let out = sched
            .admit(2, &s2, &vec![0.5; n2 * stride], &vec![0.25; n2 * stride])
            .unwrap();
        assert!(out.admitted);
        assert_eq!(out.shared_tokens, 8);
        // Only the unique suffix block is newly allocated (+1), not +3.
        assert_eq!(cache.stats().pool.allocated_blocks, before + 1);
    }
}
