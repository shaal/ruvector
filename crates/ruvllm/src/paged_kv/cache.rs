//! [`PagedKvCache`]: the orchestrator binding pool, block tables, the prefix
//! index, and copy-on-write together into one coherent KV cache (ADR-258).
//!
//! ## Concurrency model
//!
//! Allocation, CoW, table mutation, and prefix registration must move together
//! atomically to keep reference counts honest, so all mutable state lives behind
//! a single `Mutex<Inner>`. The critical sections are O(1)/O(blocks-touched) and
//! never include attention compute (which reads a cloned [`BlockTable`] and
//! gathers payloads separately), so the lock is not on the hot inference path.
//! Read-only telemetry uses lock-free atomics. Finer-grained locking is a future
//! optimization noted in the ADR; correctness-first is deliberate here.

use super::pool::{BlockPool, BlockPoolStats};
use super::prefix::PrefixIndex;
use super::quant::{BlockQuantizer, QuantTier};
use super::table::BlockTable;
use super::{BlockId, PagedKvConfig, SeqId};
use crate::error::{Result, RuvLLMError};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Per-sequence bookkeeping beyond the block table: the running chained hash of
/// the last *sealed* block and the tokens buffered for the not-yet-full tail.
struct SeqState {
    table: BlockTable,
    /// Chained content hash up to and including the last sealed block (`0` if
    /// none sealed yet). Seeds the next block's hash for prefix registration.
    last_chain: u64,
    /// Tokens written into the current unsealed tail block but not yet sealed.
    /// Drained `block_size` at a time as blocks fill.
    pending_tokens: Vec<u32>,
}

struct Inner {
    pool: BlockPool,
    prefix: PrefixIndex,
    seqs: std::collections::HashMap<SeqId, SeqState>,
}

/// Block-paged KV cache with automatic prefix sharing and copy-on-write.
pub struct PagedKvCache {
    config: PagedKvConfig,
    inner: Mutex<Inner>,
    quantizer: Arc<dyn BlockQuantizer>,
    stats: Stats,
}

#[derive(Default)]
struct Stats {
    allocations: AtomicU64,
    cow_copies: AtomicU64,
    prefix_lookups: AtomicU64,
    prefix_hit_blocks: AtomicU64,
    prefix_hit_tokens: AtomicU64,
    tier_demotions: AtomicU64,
}

impl PagedKvCache {
    /// Build a cache with the given config and the default identity quantizer
    /// (paging behavior is independent of the precision codec — see ADR-258).
    pub fn new(config: PagedKvConfig) -> Self {
        Self::with_quantizer(config, Arc::new(super::quant::IdentityQuantizer))
    }

    /// Build a cache with a specific [`BlockQuantizer`] (ADR-004 codec).
    pub fn with_quantizer(config: PagedKvConfig, quantizer: Arc<dyn BlockQuantizer>) -> Self {
        let pool = BlockPool::new(config.clone());
        Self {
            inner: Mutex::new(Inner {
                pool,
                prefix: PrefixIndex::new(),
                seqs: std::collections::HashMap::new(),
            }),
            config,
            quantizer,
            stats: Stats::default(),
        }
    }

    /// Configuration accessor.
    #[inline]
    pub fn config(&self) -> &PagedKvConfig {
        &self.config
    }

    /// Register a new, empty sequence. Errors if the id is already live.
    pub fn allocate_sequence(&self, seq_id: SeqId) -> Result<()> {
        let mut inner = self.inner.lock();
        if inner.seqs.contains_key(&seq_id) {
            return Err(RuvLLMError::InvalidOperation(format!(
                "sequence {seq_id} already allocated"
            )));
        }
        inner.seqs.insert(
            seq_id,
            SeqState {
                table: BlockTable::new(seq_id, self.config.block_size),
                last_chain: 0,
                pending_tokens: Vec::new(),
            },
        );
        Ok(())
    }

    /// Register a new sequence, sharing as long a block-aligned prefix of
    /// `tokens` as the prefix index already holds (ADR-011 path). Returns the
    /// number of tokens served from cache (always a multiple of `block_size`).
    /// The caller then only needs to compute KV for the remaining tokens.
    pub fn allocate_with_prefix(&self, seq_id: SeqId, tokens: &[u32]) -> Result<usize> {
        let mut inner = self.inner.lock();
        if inner.seqs.contains_key(&seq_id) {
            return Err(RuvLLMError::InvalidOperation(format!(
                "sequence {seq_id} already allocated"
            )));
        }
        self.stats.prefix_lookups.fetch_add(1, Ordering::Relaxed);
        let m = inner.prefix.match_prefix(tokens, self.config.block_size);

        let mut table = BlockTable::new(seq_id, self.config.block_size);
        // Share each matched physical block by bumping its refcount.
        for &bid in &m.blocks {
            inner.pool.incref(bid);
        }
        let shared_blocks = m.blocks.clone();
        let shared_tokens = m.num_tokens;
        table.extend_shared_prefix(&shared_blocks, shared_tokens);

        let last_chain = if m.blocks.is_empty() {
            0
        } else {
            // Chained hash up to the last shared block.
            m.block_hashes[m.blocks.len() - 1]
        };

        inner.seqs.insert(
            seq_id,
            SeqState {
                table,
                last_chain,
                pending_tokens: Vec::new(),
            },
        );

        if shared_tokens > 0 {
            self.stats
                .prefix_hit_blocks
                .fetch_add(shared_blocks.len() as u64, Ordering::Relaxed);
            self.stats
                .prefix_hit_tokens
                .fetch_add(shared_tokens as u64, Ordering::Relaxed);
        }
        Ok(shared_tokens)
    }

    /// Append `tokens` and their precomputed KV to `seq_id`.
    ///
    /// `keys` and `values` are laid out token-major and must each contain
    /// `tokens.len() * token_stride` elements. New blocks are allocated on
    /// demand; writing into a shared tail block triggers a single-block
    /// copy-on-write; filled blocks are sealed and registered for sharing.
    pub fn append(&self, seq_id: SeqId, tokens: &[u32], keys: &[f32], values: &[f32]) -> Result<()> {
        let stride = self.config.token_stride();
        if stride == 0 {
            return Err(RuvLLMError::Config("token_stride is zero".into()));
        }
        if keys.len() != values.len() || keys.len() != tokens.len() * stride {
            return Err(RuvLLMError::KvCache(format!(
                "append shape mismatch: {} tokens, {} key elems, stride {stride}",
                tokens.len(),
                keys.len()
            )));
        }

        let block_size = self.config.block_size;
        let mut inner = self.inner.lock();
        // Take the SeqState out to avoid borrowing `inner` mutably twice.
        let mut state = inner
            .seqs
            .remove(&seq_id)
            .ok_or_else(|| RuvLLMError::NotFound(format!("sequence {seq_id}")))?;

        let mut written = 0usize;
        let total = tokens.len();
        while written < total {
            // 1. Ensure a private tail block with room exists.
            let tail = self.ensure_writable_tail(&mut inner, &mut state)?;

            // 2. Write as many tokens as fit.
            let room = block_size - inner.pool.block(tail).num_tokens;
            let take = room.min(total - written);
            let kslice = &keys[written * stride..(written + take) * stride];
            let vslice = &values[written * stride..(written + take) * stride];
            inner.pool.append_tokens(tail, kslice, vslice)?;
            state.table.add_tokens(take);
            state
                .pending_tokens
                .extend_from_slice(&tokens[written..written + take]);
            written += take;

            // 3. Seal the block if it just filled.
            if inner.pool.block(tail).is_full(block_size) {
                self.seal_tail(&mut inner, &mut state, tail);
            }
        }

        inner.seqs.insert(seq_id, state);
        Ok(())
    }

    /// Ensure the sequence's tail block exists, is private (ref_count == 1), and
    /// has at least one free slot — allocating or copying-on-write as needed.
    /// Returns the writable tail block id.
    fn ensure_writable_tail(&self, inner: &mut Inner, state: &mut SeqState) -> Result<BlockId> {
        let block_size = self.config.block_size;
        match state.table.last_block() {
            None => {
                let b = inner.pool.allocate()?;
                self.stats.allocations.fetch_add(1, Ordering::Relaxed);
                state.table.push_block(b);
                Ok(b)
            }
            Some(tail) => {
                let (full, shared) = {
                    let blk = inner.pool.block(tail);
                    (blk.is_full(block_size), blk.is_shared())
                };
                if full {
                    // Tail is sealed/full → start a fresh private block.
                    let b = inner.pool.allocate()?;
                    self.stats.allocations.fetch_add(1, Ordering::Relaxed);
                    state.table.push_block(b);
                    Ok(b)
                } else if shared {
                    // Partially-filled shared block → copy-on-write before write.
                    let cow = inner.pool.copy_on_write(tail)?;
                    self.stats.cow_copies.fetch_add(1, Ordering::Relaxed);
                    let logical_idx = state.table.num_blocks() - 1;
                    state.table.replace_block(logical_idx, cow);
                    Ok(cow)
                } else {
                    Ok(tail)
                }
            }
        }
    }

    /// Seal a just-filled tail block: compute its chained content hash from the
    /// `block_size` pending tokens, stamp the block, and register it for sharing.
    /// If an identical block is already indexed, free our redundant copy and
    /// adopt the shared one (late deduplication).
    fn seal_tail(&self, inner: &mut Inner, state: &mut SeqState, tail: BlockId) {
        let block_size = self.config.block_size;
        if state.pending_tokens.len() < block_size {
            // Tail filled by a shared-prefix block with no pending tokens; its
            // hash is already reflected in `last_chain`. Nothing to seal.
            return;
        }
        let block_tokens: Vec<u32> = state.pending_tokens.drain(..block_size).collect();
        let h = PrefixIndex::chain_hash(state.last_chain, &block_tokens);
        state.last_chain = h;
        inner.pool.block_mut(tail).content_hash = Some(h);

        if let Some(existing) = inner.prefix.get(h) {
            if existing != tail {
                // Another sequence already sealed an identical block. Adopt it:
                // incref the survivor, point our table at it, drop our copy.
                inner.pool.incref(existing);
                let logical_idx = state.table.num_blocks() - 1;
                state.table.replace_block(logical_idx, existing);
                inner.pool.decref(tail);
                return;
            }
        }
        inner.prefix.insert(h, tail);
    }

    /// Fork `child` from `parent`, sharing all of the parent's current blocks via
    /// reference counting. The child diverges lazily: its first write into the
    /// shared (partial) tail block triggers copy-on-write, leaving the parent
    /// untouched. This is the O(1)-per-block branch primitive for speculative /
    /// tree-of-thought agent loops.
    pub fn fork(&self, parent: SeqId, child: SeqId) -> Result<()> {
        let mut inner = self.inner.lock();
        if inner.seqs.contains_key(&child) {
            return Err(RuvLLMError::InvalidOperation(format!(
                "child sequence {child} already exists"
            )));
        }
        let (blocks, num_tokens, last_chain, pending) = {
            let p = inner
                .seqs
                .get(&parent)
                .ok_or_else(|| RuvLLMError::NotFound(format!("parent sequence {parent}")))?;
            (
                p.table.blocks().to_vec(),
                p.table.num_tokens(),
                p.last_chain,
                p.pending_tokens.clone(),
            )
        };
        for &bid in &blocks {
            inner.pool.incref(bid);
        }
        let mut table = BlockTable::new(child, self.config.block_size);
        // Reuse the shared-prefix seeding helper, then carry the (now-shared)
        // partial-tail token count via num_tokens accounting.
        let full_prefix_tokens = num_tokens;
        table.extend_shared_prefix(&blocks, full_prefix_tokens);
        inner.seqs.insert(
            child,
            SeqState {
                table,
                last_chain,
                pending_tokens: pending,
            },
        );
        Ok(())
    }

    /// Free a sequence, dropping one reference to each of its blocks. Blocks
    /// whose refcount hits zero return to the pool; shared prefix blocks survive
    /// for other sequences.
    pub fn free_sequence(&self, seq_id: SeqId) -> Result<()> {
        let mut inner = self.inner.lock();
        let state = inner
            .seqs
            .remove(&seq_id)
            .ok_or_else(|| RuvLLMError::NotFound(format!("sequence {seq_id}")))?;
        for &bid in state.table.blocks() {
            // If this was the last reference to a sealed, indexed block, also
            // drop it from the prefix index so we never hand out a freed id.
            let (hash, freed) = {
                let h = inner.pool.block(bid).content_hash;
                let freed = inner.pool.block(bid).ref_count == 1;
                (h, freed)
            };
            if freed {
                if let Some(h) = hash {
                    if inner.prefix.get(h) == Some(bid) {
                        inner.prefix.remove(h);
                    }
                }
            }
            inner.pool.decref(bid);
        }
        Ok(())
    }

    /// Demote a sequence's cold prefix blocks one precision tier (ADR-004
    /// policy hook). `keep_hot` leaves the most recent `keep_hot` logical blocks
    /// untouched (the FP16 hot tail). Returns the number of blocks demoted.
    pub fn demote_cold_blocks(&self, seq_id: SeqId, keep_hot: usize) -> Result<usize> {
        let mut inner = self.inner.lock();
        let blocks: Vec<BlockId> = {
            let state = inner
                .seqs
                .get(&seq_id)
                .ok_or_else(|| RuvLLMError::NotFound(format!("sequence {seq_id}")))?;
            let n = state.table.num_blocks();
            let cold = n.saturating_sub(keep_hot);
            state.table.blocks()[..cold].to_vec()
        };
        let mut demoted = 0;
        for bid in blocks {
            // Never re-quantize a block shared with another sequence in place;
            // that would corrupt the other view. Shared cold blocks stay put
            // until their sharers release them (CoW on next write).
            if inner.pool.block(bid).is_shared() {
                continue;
            }
            let from = inner.pool.block(bid).tier;
            if let Some(target) = from.demote() {
                // Apply the codec to the block payload, then record the tier.
                {
                    let blk = inner.pool.block_mut(bid);
                    self.quantizer.requantize(&mut blk.keys, from, target);
                    self.quantizer.requantize(&mut blk.values, from, target);
                    blk.tier = target;
                }
                self.stats.tier_demotions.fetch_add(1, Ordering::Relaxed);
                demoted += 1;
            }
        }
        Ok(demoted)
    }

    /// Clone a sequence's block table (cheap: a `Vec<BlockId>`) for an attention
    /// kernel to gather against without holding the cache lock.
    pub fn block_table(&self, seq_id: SeqId) -> Option<BlockTable> {
        let inner = self.inner.lock();
        inner.seqs.get(&seq_id).map(|s| s.table.clone())
    }

    /// Reconstruct the full contiguous (keys, values, num_tokens) for a sequence
    /// by gathering its blocks in logical order. This is the CPU reference path;
    /// production paged-attention kernels gather lazily per block.
    pub fn gather_kv(&self, seq_id: SeqId) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let stride = self.config.token_stride();
        let inner = self.inner.lock();
        let state = inner
            .seqs
            .get(&seq_id)
            .ok_or_else(|| RuvLLMError::NotFound(format!("sequence {seq_id}")))?;
        let mut keys = Vec::with_capacity(state.table.num_tokens() * stride);
        let mut values = Vec::with_capacity(state.table.num_tokens() * stride);
        for &bid in state.table.blocks() {
            let blk = inner.pool.block(bid);
            let used = blk.num_tokens * stride;
            keys.extend_from_slice(&blk.keys[..used]);
            values.extend_from_slice(&blk.values[..used]);
        }
        Ok((keys, values, state.table.num_tokens()))
    }

    /// O(1) telemetry snapshot.
    pub fn stats(&self) -> PagedKvStats {
        let inner = self.inner.lock();
        let pool = inner.pool.stats();
        let live_sequences = inner.seqs.len();
        let indexed_prefix_blocks = inner.prefix.len();
        drop(inner);
        PagedKvStats {
            pool,
            live_sequences,
            indexed_prefix_blocks,
            allocations: self.stats.allocations.load(Ordering::Relaxed),
            cow_copies: self.stats.cow_copies.load(Ordering::Relaxed),
            prefix_lookups: self.stats.prefix_lookups.load(Ordering::Relaxed),
            prefix_hit_blocks: self.stats.prefix_hit_blocks.load(Ordering::Relaxed),
            prefix_hit_tokens: self.stats.prefix_hit_tokens.load(Ordering::Relaxed),
            tier_demotions: self.stats.tier_demotions.load(Ordering::Relaxed),
        }
    }

    /// Test/inspection helper: current reference count of a physical block.
    #[cfg(test)]
    pub(crate) fn ref_count(&self, id: BlockId) -> u32 {
        self.inner.lock().pool.block(id).ref_count
    }

    /// Test/inspection helper: current tier of a physical block.
    #[cfg(test)]
    pub(crate) fn block_tier(&self, id: BlockId) -> QuantTier {
        self.inner.lock().pool.block(id).tier
    }
}

/// Aggregate cache telemetry (ADR-258 observability surface).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PagedKvStats {
    /// Underlying pool occupancy.
    pub pool: BlockPoolStats,
    /// Sequences currently allocated.
    pub live_sequences: usize,
    /// Sealed blocks registered for prefix sharing.
    pub indexed_prefix_blocks: usize,
    /// Total fresh block allocations.
    pub allocations: u64,
    /// Copy-on-write block copies performed.
    pub cow_copies: u64,
    /// Prefix-match lookups attempted.
    pub prefix_lookups: u64,
    /// Blocks served from the prefix index (shared, not recomputed).
    pub prefix_hit_blocks: u64,
    /// Tokens served from the prefix index.
    pub prefix_hit_tokens: u64,
    /// Block tier demotions performed (ADR-004 path).
    pub tier_demotions: u64,
}

impl PagedKvStats {
    /// Prefix cache hit rate over tokens, in `0.0..=1.0`. Requires that callers
    /// track total requested prefix tokens externally; here we expose the
    /// block-sharing ratio as a proxy: shared tokens per lookup.
    pub fn avg_shared_tokens_per_lookup(&self) -> f32 {
        if self.prefix_lookups == 0 {
            0.0
        } else {
            self.prefix_hit_tokens as f32 / self.prefix_lookups as f32
        }
    }
}
