//! Block-paged attention kernel (ADR-258 Phase 6 substrate).
//!
//! A CPU reference implementation of single-query attention that reads KV
//! directly from the paged [`PagedKvCache`] blocks — no contiguous gather. It
//! uses a streaming (FlashAttention-style) online softmax so it never
//! materializes the full score vector, which is exactly the access pattern a
//! GPU paged-attention kernel (FlashAttention-3 / Metal / cudarc) implements.
//! Those backends plug in behind [`BlockAttention`] without touching the
//! allocator.
//!
//! Layout contract (matches [`super::pool::PhysicalBlock`]): both the query and
//! each block's KV are token-major, head-major within a token:
//! `query = [head0 d0..d_{D-1}, head1 ..]` (one decode step, `num_heads` heads),
//! `block.keys = [tok0 kvhead0 d.., tok0 kvhead1 .., tok1 ..]`. Grouped-query
//! attention maps query head `h` to KV head `h / (num_heads / num_kv_heads)`.

use super::{PagedKvCache, SeqId};
use crate::error::{Result, RuvLLMError};

/// Shape + scaling parameters for a single decode-step attention call.
#[derive(Debug, Clone, Copy)]
pub struct AttentionParams {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of KV heads (GQA; equals `num_heads` for MHA).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Softmax scale, typically `1 / sqrt(head_dim)`.
    pub scale: f32,
}

impl AttentionParams {
    /// Derive params from a cache config with the standard `1/sqrt(d)` scale.
    pub fn from_cache(cache: &PagedKvCache, num_heads: usize) -> Self {
        let c = cache.config();
        Self {
            num_heads,
            num_kv_heads: c.num_kv_heads,
            head_dim: c.head_dim,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
        }
    }

    #[inline]
    fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// An attention backend over paged KV. CPU reference + future GPU kernels share
/// this contract so the engine can swap implementations transparently.
pub trait BlockAttention {
    /// Compute attention output (`num_heads * head_dim` elements) for one query
    /// step against the cached KV of `seq_id`.
    fn forward(
        &self,
        cache: &PagedKvCache,
        seq_id: SeqId,
        query: &[f32],
        params: AttentionParams,
    ) -> Result<Vec<f32>>;
}

/// CPU reference kernel with streaming online softmax.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuPagedAttention;

impl BlockAttention for CpuPagedAttention {
    fn forward(
        &self,
        cache: &PagedKvCache,
        seq_id: SeqId,
        query: &[f32],
        params: AttentionParams,
    ) -> Result<Vec<f32>> {
        paged_attention_forward(cache, seq_id, query, params)
    }
}

/// Free-function form of the CPU paged-attention kernel.
///
/// Streams over the sequence's blocks once per call, maintaining per-head
/// running max `m`, normalizer `l`, and weighted accumulator `acc` (the flash
/// attention recurrence), so memory is `O(num_heads * head_dim)` regardless of
/// context length.
pub fn paged_attention_forward(
    cache: &PagedKvCache,
    seq_id: SeqId,
    query: &[f32],
    params: AttentionParams,
) -> Result<Vec<f32>> {
    let AttentionParams {
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    } = params;

    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(RuvLLMError::InvalidOperation(format!(
            "num_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}"
        )));
    }
    if query.len() != num_heads * head_dim {
        return Err(RuvLLMError::InvalidOperation(format!(
            "query has {} elems, expected {}",
            query.len(),
            num_heads * head_dim
        )));
    }
    // Kernel layout assumes KV heads/dim match the cache's storage stride.
    let c = cache.config();
    if c.num_kv_heads != num_kv_heads || c.head_dim != head_dim {
        return Err(RuvLLMError::InvalidOperation(
            "attention params do not match cache KV layout".into(),
        ));
    }

    let gqa = params.gqa_ratio();
    let kv_stride = num_kv_heads * head_dim;

    // Flash-style streaming softmax state, per query head.
    let mut m = vec![f32::NEG_INFINITY; num_heads]; // running max score
    let mut l = vec![0.0f32; num_heads]; // running exp-sum
    let mut acc = vec![0.0f32; num_heads * head_dim]; // running weighted value

    cache.for_each_block(seq_id, |keys, values, num_tokens| {
        for t in 0..num_tokens {
            let base = t * kv_stride;
            for h in 0..num_heads {
                let kv_head = h / gqa;
                let k_off = base + kv_head * head_dim;
                let q_off = h * head_dim;
                // score = scale * <q_h, k_t>
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += query[q_off + d] * keys[k_off + d];
                }
                score *= scale;

                // Online softmax update for head h.
                let m_prev = m[h];
                let m_new = m_prev.max(score);
                let correction = (m_prev - m_new).exp();
                let p = (score - m_new).exp();
                l[h] = l[h] * correction + p;
                let a_off = h * head_dim;
                let v_off = base + kv_head * head_dim;
                for d in 0..head_dim {
                    acc[a_off + d] = acc[a_off + d] * correction + p * values[v_off + d];
                }
                m[h] = m_new;
            }
        }
    })?;

    // Normalize.
    let mut out = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let denom = if l[h] > 0.0 { l[h] } else { 1.0 };
        let a_off = h * head_dim;
        for d in 0..head_dim {
            out[a_off + d] = acc[a_off + d] / denom;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use crate::paged_kv::{PagedKvCache, PagedKvConfig};

    fn cfg(num_kv_heads: usize, head_dim: usize) -> PagedKvConfig {
        PagedKvConfig {
            block_size: 4,
            num_kv_heads,
            head_dim,
            total_blocks: 64,
            verify_prefix_tokens: false,
        }
    }

    /// Dense reference: gather KV and compute attention the textbook way.
    fn dense_reference(
        keys: &[f32],
        values: &[f32],
        num_tokens: usize,
        query: &[f32],
        p: AttentionParams,
    ) -> Vec<f32> {
        let gqa = p.num_heads / p.num_kv_heads;
        let kv_stride = p.num_kv_heads * p.head_dim;
        let mut out = vec![0.0f32; p.num_heads * p.head_dim];
        for h in 0..p.num_heads {
            let kv_head = h / gqa;
            let q_off = h * p.head_dim;
            let mut scores = Vec::with_capacity(num_tokens);
            for t in 0..num_tokens {
                let k_off = t * kv_stride + kv_head * p.head_dim;
                let mut s = 0.0f32;
                for d in 0..p.head_dim {
                    s += query[q_off + d] * keys[k_off + d];
                }
                scores.push(s * p.scale);
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for (t, &w) in exp.iter().enumerate() {
                let v_off = t * kv_stride + kv_head * p.head_dim;
                for d in 0..p.head_dim {
                    out[q_off + d] += (w / sum) * values[v_off + d];
                }
            }
        }
        out
    }

    fn fill(cache: &PagedKvCache, seq: u64, n: usize, kv_stride: usize) -> (Vec<f32>, Vec<f32>) {
        let tokens: Vec<u32> = (0..n as u32).collect();
        // Deterministic pseudo-random-ish payload that varies per (token, dim).
        let mut keys = Vec::with_capacity(n * kv_stride);
        let mut values = Vec::with_capacity(n * kv_stride);
        for t in 0..n {
            for j in 0..kv_stride {
                keys.push(((t * 7 + j * 3) % 11) as f32 * 0.1 - 0.5);
                values.push(((t * 5 + j * 2) % 9) as f32 * 0.1 - 0.4);
            }
        }
        cache.allocate_sequence(seq).unwrap();
        cache.append(seq, &tokens, &keys, &values).unwrap();
        (keys, values)
    }

    #[test]
    fn paged_matches_dense_mha() {
        let num_kv_heads = 2;
        let head_dim = 4;
        let num_heads = 2; // MHA
        let config = cfg(num_kv_heads, head_dim);
        let kv_stride = config.token_stride();
        let cache = PagedKvCache::new(config);
        let n = 10; // spans 3 blocks
        let (keys, values) = fill(&cache, 1, n, kv_stride);

        let p = AttentionParams {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };
        let query: Vec<f32> = (0..num_heads * head_dim).map(|i| (i as f32) * 0.05).collect();

        let got = paged_attention_forward(&cache, 1, &query, p).unwrap();
        let want = dense_reference(&keys, &values, n, &query, p);
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "paged {a} vs dense {b}");
        }
    }

    #[test]
    fn paged_matches_dense_gqa() {
        let num_kv_heads = 2;
        let head_dim = 4;
        let num_heads = 6; // GQA ratio 3
        let config = cfg(num_kv_heads, head_dim);
        let kv_stride = config.token_stride();
        let cache = PagedKvCache::new(config);
        let n = 7;
        let (keys, values) = fill(&cache, 1, n, kv_stride);

        let p = AttentionParams {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };
        let query: Vec<f32> = (0..num_heads * head_dim).map(|i| ((i % 5) as f32) * 0.1).collect();

        let got = paged_attention_forward(&cache, 1, &query, p).unwrap();
        let want = dense_reference(&keys, &values, n, &query, p);
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "paged {a} vs dense {b}");
        }
    }

    #[test]
    fn trait_object_dispatch_works() {
        let config = cfg(1, 2);
        let kv_stride = config.token_stride();
        let cache = PagedKvCache::new(config);
        fill(&cache, 1, 5, kv_stride);
        let backend: &dyn BlockAttention = &CpuPagedAttention;
        let p = AttentionParams::from_cache(&cache, 1);
        let query = vec![0.3f32, -0.2];
        let out = backend.forward(&cache, 1, &query, p).unwrap();
        assert_eq!(out.len(), 2);
    }
}
