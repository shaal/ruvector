//! Residual GAT-style attention block (ADR-258 §6.3).
//!
//! Upgrades message passing in two ways over the base `RuvectorLayer`:
//! 1. a **residual skip** that carries the node's own features around the
//!    attention sub-layer (pre-norm transformer style), improving gradient flow
//!    and stability of the online self-learning loop;
//! 2. a **learned edge gain** that lets the block up/down-weight the
//!    HNSW-edge-weighted neighbour aggregation — a lightweight, trainable edge
//!    bias on top of attention.
//!
//! Input, neighbour, and output dimensions are all `embed_dim` so the residual
//! is well-defined. Reuses the existing `MultiHeadAttention` / `LayerNorm`.

use crate::error::Result;
use crate::layer::{LayerNorm, MultiHeadAttention};

/// A residual, edge-biased attention block over a node and its neighbours.
pub struct ResidualGatBlock {
    attention: MultiHeadAttention,
    norm: LayerNorm,
    /// Learned scalar gain on the edge-weighted neighbour aggregation.
    pub edge_gain: f32,
    embed_dim: usize,
}

impl ResidualGatBlock {
    pub fn new(embed_dim: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(embed_dim, heads)?,
            norm: LayerNorm::new(embed_dim, 1e-5),
            edge_gain: 1.0,
            embed_dim,
        })
    }

    #[inline]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// `out = LayerNorm(node + Attention(node, N) + edge_gain · Σ ŵ_e · n_e)`.
    pub fn forward(&self, node: &[f32], neighbors: &[Vec<f32>], edge_weights: &[f32]) -> Vec<f32> {
        let d = node.len();
        let mut out = node.to_vec(); // residual skip

        if !neighbors.is_empty() {
            // Attention sub-layer.
            let attn = self.attention.forward(node, neighbors, neighbors);
            for k in 0..d.min(attn.len()) {
                out[k] += attn[k];
            }
            // Edge-weighted neighbour aggregation (normalized), scaled by gain.
            let wsum: f32 = edge_weights.iter().copied().sum::<f32>().max(1e-9);
            for (nb, &w) in neighbors.iter().zip(edge_weights) {
                let wn = (w / wsum) * self.edge_gain;
                for k in 0..d.min(nb.len()) {
                    out[k] += wn * nb[k];
                }
            }
        }
        self.norm.forward(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_dimension() {
        let blk = ResidualGatBlock::new(8, 2).unwrap();
        let node = vec![0.1f32; 8];
        let neighbors = vec![vec![0.2f32; 8], vec![0.3f32; 8]];
        let out = blk.forward(&node, &neighbors, &[0.6, 0.4]);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn no_neighbors_is_normed_residual() {
        let blk = ResidualGatBlock::new(8, 2).unwrap();
        let node = vec![0.5f32; 8];
        let out = blk.forward(&node, &[], &[]);
        // LayerNorm of a constant vector -> all zeros (centered), finite.
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn edge_gain_changes_output() {
        let mut blk = ResidualGatBlock::new(8, 2).unwrap();
        let node: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).sin()).collect();
        let n1: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).cos()).collect();
        let n2: Vec<f32> = (0..8).map(|i| (i as f32 * 0.2 - 0.5)).collect();
        let neighbors = vec![n1, n2];
        let a = blk.forward(&node, &neighbors, &[0.9, 0.1]);
        blk.edge_gain = 5.0;
        let b = blk.forward(&node, &neighbors, &[0.9, 0.1]);
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-4, "edge_gain should affect the output");
    }
}
