//! Paged block-based KV cache management (ADR-258).
//!
//! This module implements PagedAttention-style KV cache memory management: KV
//! state is stored in fixed-size **physical blocks** held by a shared
//! [`BlockPool`], and every sequence gets a [`BlockTable`] mapping its
//! contiguous *logical* token positions onto non-contiguous *physical* blocks.
//!
//! It is the unifying substrate the project's two existing KV ADRs sit on top
//! of:
//!
//! * **ADR-011 (Prefix Caching)** — the [`PrefixIndex`] makes prefix sharing
//!   *block-aligned*: full blocks are sealed with a chained content hash and
//!   deduplicated across sequences, so identical system prompts / RAG chunks /
//!   tool schemas occupy a single physical copy. Divergence triggers a cheap
//!   single-block copy-on-write rather than copying the whole context.
//! * **ADR-004 (Tiered Quantization)** — each [`PhysicalBlock`] carries its own
//!   [`QuantTier`], so the hot FP16 tail, the 4-bit warm zone, and the 2-bit
//!   archive all live in the same pool and demote in place, one block at a
//!   time.
//!
//! ## Layout
//!
//! ```text
//!   logical pos ─► (block_idx = pos / block_size, offset = pos % block_size)
//!
//!   Seq A BlockTable:  [ P7 | P3 | P9 | P1 ]   (logical index 0..3)
//!   Seq B BlockTable:  [ P7 | P3 | P5 ]        shares P7,P3 (rc=2), CoW at idx2
//! ```
//!
//! The allocator is pure-Rust and GPU-free; attention kernels (FlashAttention-3
//! / Metal / cudarc) plug in later behind a trait without touching the pool.

mod attention;
mod cache;
mod pool;
mod prefix;
mod quant;
mod scheduler;
mod table;

#[cfg(test)]
mod tests;

pub use attention::{
    paged_attention_forward, AttentionParams, BlockAttention, CpuPagedAttention,
};
pub use cache::{PagedKvCache, PagedKvStats};
pub use pool::{BlockPool, BlockPoolStats, PhysicalBlock};
pub use prefix::{PrefixIndex, PrefixMatch};
pub use quant::{BlockQuantizer, IdentityQuantizer, QuantTier, UniformQuantizer};
pub use scheduler::{AdmitOutcome, BatchScheduler, SchedulerConfig, SchedulerStats};
pub use table::BlockTable;

use serde::{Deserialize, Serialize};

/// Identifier for a physical block in the [`BlockPool`].
///
/// A newtype over `u32` keeps block tables compact (4 bytes/entry) and prevents
/// accidentally mixing a physical block id with a logical block index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockId(pub u32);

impl BlockId {
    /// The raw index into the pool's backing store.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "P{}", self.0)
    }
}

/// Sequence identifier. Sequences are the unit of allocation and sharing.
pub type SeqId = u64;

/// Configuration for a [`PagedKvCache`].
///
/// All sizes are in "elements" of the KV payload, where one token contributes
/// `num_kv_heads * head_dim` elements to the key buffer and the same to the
/// value buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagedKvConfig {
    /// Tokens per physical block (vLLM default: 16).
    pub block_size: usize,
    /// Number of KV heads (GQA-aware; equals `num_heads` for MHA).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Total physical blocks the pool pre-reserves. This is the hard memory
    /// budget: `total_blocks * block_size * num_kv_heads * head_dim` elements
    /// for keys (and the same for values).
    pub total_blocks: usize,
    /// When `true`, a longest-prefix match in the [`PrefixIndex`] is verified
    /// by comparing the full token list before sharing, guarding against the
    /// (astronomically rare) 64-bit hash collision. Costs one token compare per
    /// shared block.
    pub verify_prefix_tokens: bool,
}

impl PagedKvConfig {
    /// Number of f32 elements one block holds for *each* of keys and values.
    #[inline]
    pub fn block_elems(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim
    }

    /// Number of f32 elements one token contributes to *each* of keys/values.
    #[inline]
    pub fn token_stride(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

impl Default for PagedKvConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_kv_heads: 8,
            head_dim: 128,
            total_blocks: 4096,
            verify_prefix_tokens: false,
        }
    }
}
