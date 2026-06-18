//! Block-aligned prefix sharing index — the ADR-011 bridge.
//!
//! ADR-011 chose a Radix Tree + copy-on-write design for prefix caching and
//! explicitly flagged a *block-aligned* variant as future work. This module is
//! that variant: instead of keying a trie on raw token runs, we seal each full
//! block with a **chained content hash**
//!
//! ```text
//!   h_0 = hash(tokens_block0)
//!   h_i = hash(h_{i-1} , tokens_block_i)
//! ```
//!
//! and map `chained_hash → BlockId`. A chained hash uniquely identifies "the KV
//! state after this exact run of blocks", so two sequences that begin with the
//! same tokens produce the same chain and resolve to the *same physical
//! blocks*. Sharing therefore happens automatically and at block granularity —
//! no caller-declared prefixes, and partial-block tails simply aren't shared
//! (they're still being written).
//!
//! The flat `HashMap` here is the minimal form of ADR-011's radix tree: each
//! entry is one tree edge keyed by content hash. Upgrading to an explicit tree
//! (for prefix eviction by subtree) is a drop-in replacement behind the same
//! [`PrefixIndex`] API.

use super::BlockId;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Result of matching a token sequence against the prefix index: the run of
/// already-cached physical blocks that can be shared, plus their chained hashes
/// (so newly-sealed blocks can be registered without recomputing).
#[derive(Debug, Clone, Default)]
pub struct PrefixMatch {
    /// Shared physical blocks, in logical order. Caller `incref`s each.
    pub blocks: Vec<BlockId>,
    /// Number of tokens covered by `blocks` (always a multiple of block_size).
    pub num_tokens: usize,
    /// Chained hash of every *full* block of the input (matched or not), so the
    /// caller can register newly-computed blocks as it fills them.
    pub block_hashes: Vec<u64>,
}

/// Maps chained block-content hashes to the physical block holding that prefix.
#[derive(Debug, Default)]
pub struct PrefixIndex {
    map: HashMap<u64, BlockId>,
}

impl PrefixIndex {
    /// Empty index.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Compute the chained hash of one block given the previous chain value and
    /// the block's tokens. `prev` is `0` for the first block. Public so the
    /// cache can seal a block with the identical function used for matching.
    #[inline]
    pub fn chain_hash(prev: u64, tokens: &[u32]) -> u64 {
        // FxHash-style mixing is plenty for content addressing; we use the std
        // DefaultHasher to avoid adding a dependency. Determinism within a
        // process run is all that's required (the index is in-memory).
        let mut h = std::collections::hash_map::DefaultHasher::new();
        prev.hash(&mut h);
        tokens.hash(&mut h);
        h.finish()
    }

    /// Walk `tokens` block by block, returning the longest run of *consecutive*
    /// full blocks already present in the index.
    ///
    /// Matching stops at the first block not in the index (a prefix must be
    /// contiguous from position 0). Only whole blocks participate; a trailing
    /// partial block is never shared. `block_hashes` covers every full block of
    /// the input so the caller can register the unmatched remainder as it
    /// computes them.
    pub fn match_prefix(&self, tokens: &[u32], block_size: usize) -> PrefixMatch {
        let mut result = PrefixMatch::default();
        if block_size == 0 {
            return result;
        }
        let full_blocks = tokens.len() / block_size;
        let mut prev = 0u64;
        let mut still_matching = true;
        for b in 0..full_blocks {
            let start = b * block_size;
            let block_tokens = &tokens[start..start + block_size];
            let h = Self::chain_hash(prev, block_tokens);
            result.block_hashes.push(h);
            prev = h;
            if still_matching {
                if let Some(&id) = self.map.get(&h) {
                    result.blocks.push(id);
                    result.num_tokens += block_size;
                } else {
                    // First miss ends the shareable prefix; keep computing
                    // hashes for the remaining blocks so the caller can insert.
                    still_matching = false;
                }
            }
        }
        result
    }

    /// Register a sealed full block under its chained hash. If the hash is
    /// already present (an identical prefix was sealed concurrently), the
    /// existing entry wins and `false` is returned so the caller can release its
    /// redundant block.
    pub fn insert(&mut self, chained_hash: u64, block: BlockId) -> bool {
        match self.map.entry(chained_hash) {
            std::collections::hash_map::Entry::Occupied(_) => false,
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(block);
                true
            }
        }
    }

    /// Look up a single chained hash.
    #[inline]
    pub fn get(&self, chained_hash: u64) -> Option<BlockId> {
        self.map.get(&chained_hash).copied()
    }

    /// Remove an entry (called when a sealed block is evicted/freed).
    pub fn remove(&mut self, chained_hash: u64) -> Option<BlockId> {
        self.map.remove(&chained_hash)
    }

    /// Number of indexed (shareable) blocks.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;

    #[test]
    fn identical_prefixes_chain_equally() {
        let a = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let b = [1u32, 2, 3, 4, 9, 9, 9, 9];
        let bs = 4;
        // Same first block ⇒ same first chained hash.
        let ha = PrefixIndex::chain_hash(0, &a[..bs]);
        let hb = PrefixIndex::chain_hash(0, &b[..bs]);
        assert_eq!(ha, hb);
        // Divergent second block ⇒ different chain.
        let ha2 = PrefixIndex::chain_hash(ha, &a[bs..]);
        let hb2 = PrefixIndex::chain_hash(hb, &b[bs..]);
        assert_ne!(ha2, hb2);
    }

    #[test]
    fn match_returns_longest_contiguous_run() {
        let mut idx = PrefixIndex::new();
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let bs = 4;
        let h0 = PrefixIndex::chain_hash(0, &tokens[..bs]);
        idx.insert(h0, BlockId(7));
        // Only the first block is registered.
        let m = idx.match_prefix(&tokens, bs);
        assert_eq!(m.blocks, vec![BlockId(7)]);
        assert_eq!(m.num_tokens, 4);
        assert_eq!(m.block_hashes.len(), 2);
    }

    #[test]
    fn partial_tail_is_not_shared() {
        let mut idx = PrefixIndex::new();
        let tokens = [1u32, 2, 3, 4, 5, 6]; // 1 full block + partial
        let bs = 4;
        let h0 = PrefixIndex::chain_hash(0, &tokens[..bs]);
        idx.insert(h0, BlockId(1));
        let m = idx.match_prefix(&tokens, bs);
        assert_eq!(m.num_tokens, 4); // only the full block
        assert_eq!(m.block_hashes.len(), 1);
    }
}
