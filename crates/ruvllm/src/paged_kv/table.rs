//! Per-sequence block table: the logical → physical mapping.
//!
//! A [`BlockTable`] is the page table for one sequence. It gives the sequence a
//! contiguous *logical* address space (token positions `0..num_tokens`) over a
//! non-contiguous set of *physical* blocks. The table itself is just an ordered
//! `Vec<BlockId>` plus a token count; all sharing/refcount logic lives in the
//! pool and cache so the table stays a dumb, cheap-to-clone index.

use super::{BlockId, SeqId};
use serde::{Deserialize, Serialize};

/// Ordered logical-block → physical-block map for a single sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockTable {
    /// The sequence this table belongs to.
    pub seq_id: SeqId,
    /// Physical block backing each logical block index, in order.
    /// `physical[i]` backs logical positions `i*block_size .. (i+1)*block_size`.
    physical: Vec<BlockId>,
    /// Total tokens currently stored across all blocks.
    num_tokens: usize,
    /// Tokens per block (cached from config so position math needs no pool).
    block_size: usize,
}

impl BlockTable {
    /// Create an empty table for `seq_id`.
    pub fn new(seq_id: SeqId, block_size: usize) -> Self {
        Self {
            seq_id,
            physical: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    /// Logical block index and intra-block offset for a token position.
    #[inline]
    pub fn locate(&self, pos: usize) -> (usize, usize) {
        (pos / self.block_size, pos % self.block_size)
    }

    /// Physical block backing logical block `logical_idx`, if mapped.
    #[inline]
    pub fn physical(&self, logical_idx: usize) -> Option<BlockId> {
        self.physical.get(logical_idx).copied()
    }

    /// Physical block backing a token position, if that position is mapped.
    #[inline]
    pub fn physical_for_pos(&self, pos: usize) -> Option<BlockId> {
        self.physical(pos / self.block_size)
    }

    /// All physical blocks in logical order (for attention gather / freeing).
    #[inline]
    pub fn blocks(&self) -> &[BlockId] {
        &self.physical
    }

    /// Number of logical blocks currently mapped.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.physical.len()
    }

    /// Total tokens stored.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// The last (tail) physical block, where new tokens are appended.
    #[inline]
    pub fn last_block(&self) -> Option<BlockId> {
        self.physical.last().copied()
    }

    /// Tokens currently sitting in the tail block (`0` means the tail is full or
    /// the table is empty, and a fresh block is needed before appending).
    #[inline]
    pub fn tail_fill(&self) -> usize {
        if self.physical.is_empty() {
            return 0;
        }
        let rem = self.num_tokens % self.block_size;
        // A table whose token count is an exact multiple of block_size has a
        // *full* tail (rem == 0 but blocks present) → report block_size.
        if rem == 0 {
            self.block_size
        } else {
            rem
        }
    }

    /// Append a freshly-allocated physical block to the logical tail.
    #[inline]
    pub fn push_block(&mut self, id: BlockId) {
        self.physical.push(id);
    }

    /// Replace the physical block at `logical_idx` (used by copy-on-write, which
    /// swaps a shared block for a private duplicate). Returns the previous id.
    #[inline]
    pub fn replace_block(&mut self, logical_idx: usize, id: BlockId) -> Option<BlockId> {
        if logical_idx < self.physical.len() {
            let old = self.physical[logical_idx];
            self.physical[logical_idx] = id;
            Some(old)
        } else {
            None
        }
    }

    /// Record that `n` tokens were written (after a successful pool append).
    #[inline]
    pub fn add_tokens(&mut self, n: usize) {
        self.num_tokens += n;
    }

    /// Seed a table from a shared prefix: a run of already-populated physical
    /// blocks (each `incref`'d by the caller) covering `prefix_tokens` tokens.
    pub fn extend_shared_prefix(&mut self, blocks: &[BlockId], prefix_tokens: usize) {
        self.physical.extend_from_slice(blocks);
        self.num_tokens += prefix_tokens;
    }
}
