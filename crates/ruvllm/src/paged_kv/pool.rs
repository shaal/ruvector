//! Physical block storage and the shared allocation pool.
//!
//! The [`BlockPool`] owns every physical block and an O(1) free list. Reference
//! counts on each block implement copy-on-write sharing: a block referenced by
//! more than one [`super::BlockTable`] is immutable until copied. The pool is
//! deliberately *not* internally synchronized — [`super::PagedKvCache`] wraps it
//! in a single `Mutex`, which is simpler to reason about than per-block locks
//! and avoids deadlocks between the pool and the prefix index.

use super::quant::QuantTier;
use super::{BlockId, PagedKvConfig};
use crate::error::{Result, RuvLLMError};
use serde::{Deserialize, Serialize};

/// A single fixed-size physical block holding the KV payload for up to
/// `block_size` tokens.
///
/// Keys and values are stored as flat `f32` buffers of `block_size *
/// num_kv_heads * head_dim` elements, laid out token-major:
/// `[tok0 head0 d0..d_{D-1}, tok0 head1 .., tok1 head0 ..]`.
#[derive(Debug)]
pub struct PhysicalBlock {
    /// This block's id (its index in the pool).
    pub id: BlockId,
    /// Key payload, length `block_elems`.
    pub keys: Vec<f32>,
    /// Value payload, length `block_elems`.
    pub values: Vec<f32>,
    /// Tokens currently written into this block (`<= block_size`).
    pub num_tokens: usize,
    /// Number of block tables referencing this block. `0` ⇒ free.
    ///
    /// Kept as a plain field (not atomic) because all access is serialized
    /// through the pool's `Mutex`; this keeps refcount transitions and free-list
    /// updates a single critical section, which is what makes CoW race-free.
    pub ref_count: u32,
    /// Precision tier (ADR-004). Hot blocks are `Fp16`; cooled blocks demote.
    pub tier: QuantTier,
    /// Chained content hash, set when the block is *sealed* (full + immutable
    /// candidate for sharing). `None` while still being appended to.
    pub content_hash: Option<u64>,
}

impl PhysicalBlock {
    fn new(id: BlockId, block_elems: usize) -> Self {
        Self {
            id,
            keys: vec![0.0; block_elems],
            values: vec![0.0; block_elems],
            num_tokens: 0,
            ref_count: 0,
            tier: QuantTier::Fp16,
            content_hash: None,
        }
    }

    /// Reset to the pristine state used when returned to the free list. We zero
    /// only the *used* prefix to avoid touching the whole buffer on every free.
    fn reset(&mut self, token_stride: usize) {
        let used = self.num_tokens * token_stride;
        for v in &mut self.keys[..used] {
            *v = 0.0;
        }
        for v in &mut self.values[..used] {
            *v = 0.0;
        }
        self.num_tokens = 0;
        self.ref_count = 0;
        self.tier = QuantTier::Fp16;
        self.content_hash = None;
    }

    /// `true` once the block holds `block_size` tokens.
    #[inline]
    pub fn is_full(&self, block_size: usize) -> bool {
        self.num_tokens >= block_size
    }

    /// Whether this block is shared (referenced by more than one table) and thus
    /// must be copied before any in-place mutation (copy-on-write).
    #[inline]
    pub fn is_shared(&self) -> bool {
        self.ref_count > 1
    }
}

/// The shared physical block pool: backing store + free list + refcounts.
#[derive(Debug)]
pub struct BlockPool {
    config: PagedKvConfig,
    blocks: Vec<PhysicalBlock>,
    /// Stack of free block ids. A `Vec` used as a stack gives LIFO reuse, which
    /// keeps recently-freed (cache-warm) blocks hot.
    free: Vec<BlockId>,
    /// Count of currently allocated (ref_count > 0) blocks, maintained
    /// incrementally so `stats()` is O(1).
    allocated: usize,
}

impl BlockPool {
    /// Pre-reserve `config.total_blocks` physical blocks. This is the hard
    /// memory budget; allocation never grows the pool.
    pub fn new(config: PagedKvConfig) -> Self {
        let block_elems = config.block_elems();
        let total = config.total_blocks;
        let mut blocks = Vec::with_capacity(total);
        let mut free = Vec::with_capacity(total);
        for i in 0..total {
            blocks.push(PhysicalBlock::new(BlockId(i as u32), block_elems));
            // Push in reverse so block 0 is allocated first (nicer for tests).
            free.push(BlockId((total - 1 - i) as u32));
        }
        Self {
            config,
            blocks,
            free,
            allocated: 0,
        }
    }

    /// Allocate a fresh block with `ref_count = 1`. O(1).
    pub fn allocate(&mut self) -> Result<BlockId> {
        let id = self
            .free
            .pop()
            .ok_or_else(|| RuvLLMError::OutOfMemory("block pool exhausted".into()))?;
        let block = &mut self.blocks[id.index()];
        debug_assert_eq!(block.ref_count, 0, "freed block had non-zero refcount");
        block.ref_count = 1;
        self.allocated += 1;
        Ok(id)
    }

    /// Increment the reference count of a block (prefix sharing / fork). Returns
    /// the new count.
    pub fn incref(&mut self, id: BlockId) -> u32 {
        let block = &mut self.blocks[id.index()];
        debug_assert!(block.ref_count > 0, "incref on a free block {id}");
        block.ref_count += 1;
        block.ref_count
    }

    /// Decrement the reference count; when it reaches zero the block is reset
    /// and returned to the free list. Returns `true` if the block was freed.
    pub fn decref(&mut self, id: BlockId) -> bool {
        let stride = self.config.token_stride();
        let block = &mut self.blocks[id.index()];
        debug_assert!(block.ref_count > 0, "decref on a free block {id}");
        block.ref_count -= 1;
        if block.ref_count == 0 {
            block.reset(stride);
            self.free.push(id);
            self.allocated -= 1;
            true
        } else {
            false
        }
    }

    /// Copy-on-write: produce a private, writable duplicate of `src`, dropping
    /// one reference to the original. The new block starts at `ref_count = 1`
    /// and inherits payload, token count, and tier — but **not** the sealed
    /// content hash (it is about to be mutated, so it is no longer sealed).
    pub fn copy_on_write(&mut self, src: BlockId) -> Result<BlockId> {
        let dst = self.allocate()?;
        // Split borrow: clone source payload first, then write into dst. We copy
        // through owned Vecs to satisfy the borrow checker without unsafe.
        let (keys, values, num_tokens, tier) = {
            let b = &self.blocks[src.index()];
            (b.keys.clone(), b.values.clone(), b.num_tokens, b.tier)
        };
        {
            let d = &mut self.blocks[dst.index()];
            d.keys = keys;
            d.values = values;
            d.num_tokens = num_tokens;
            d.tier = tier;
            d.content_hash = None; // private mutable copy: unsealed
        }
        // Drop the caller's reference to the shared original.
        self.decref(src);
        Ok(dst)
    }

    /// Append `num` tokens of KV into `id`. Caller guarantees the block is
    /// private (ref_count == 1) and has room — both are debug-asserted. Returns
    /// the number of tokens written.
    pub fn append_tokens(
        &mut self,
        id: BlockId,
        keys: &[f32],
        values: &[f32],
    ) -> Result<usize> {
        let stride = self.config.token_stride();
        if keys.len() != values.len() {
            return Err(RuvLLMError::KvCache("key/value length mismatch".into()));
        }
        if stride == 0 || keys.len() % stride != 0 {
            return Err(RuvLLMError::KvCache(
                "KV length not a multiple of token stride".into(),
            ));
        }
        let num = keys.len() / stride;
        let block = &mut self.blocks[id.index()];
        debug_assert_eq!(block.ref_count, 1, "append into shared block {id} (needs CoW)");
        let room = self.config.block_size - block.num_tokens;
        if num > room {
            return Err(RuvLLMError::KvCache(format!(
                "block {id} overflow: appending {num} tokens, room {room}"
            )));
        }
        let start = block.num_tokens * stride;
        let end = start + keys.len();
        block.keys[start..end].copy_from_slice(keys);
        block.values[start..end].copy_from_slice(values);
        block.num_tokens += num;
        // Re-seal invalidated: hash recomputed by the cache when the block fills.
        block.content_hash = None;
        Ok(num)
    }

    /// Shared read access to a block.
    #[inline]
    pub fn block(&self, id: BlockId) -> &PhysicalBlock {
        &self.blocks[id.index()]
    }

    /// Mutable access to a block (for tier demotion). Callers must respect CoW.
    #[inline]
    pub fn block_mut(&mut self, id: BlockId) -> &mut PhysicalBlock {
        &mut self.blocks[id.index()]
    }

    /// Number of free blocks remaining.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// O(1) snapshot of pool occupancy.
    pub fn stats(&self) -> BlockPoolStats {
        BlockPoolStats {
            total_blocks: self.config.total_blocks,
            allocated_blocks: self.allocated,
            free_blocks: self.free.len(),
        }
    }

    /// Config accessor.
    #[inline]
    pub fn config(&self) -> &PagedKvConfig {
        &self.config
    }
}

/// O(1) pool occupancy snapshot.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BlockPoolStats {
    /// Total physical blocks reserved.
    pub total_blocks: usize,
    /// Blocks with ref_count > 0.
    pub allocated_blocks: usize,
    /// Blocks on the free list.
    pub free_blocks: usize,
}

impl BlockPoolStats {
    /// Fraction of the pool currently in use, in `0.0..=1.0`.
    pub fn utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.allocated_blocks as f32 / self.total_blocks as f32
        }
    }
}
