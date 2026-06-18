//! Pluggable node-feature sources for the GNN (ADR-258 §6.2).
//!
//! `RuvectorLayer::forward` consumes a node embedding and its neighbours'
//! embeddings. A [`FeatureSource`] decides what those per-node feature vectors
//! are: either the legacy flat embedding ([`FlatEmbedding`], the default), or a
//! multiresolution-hash-augmented feature ([`HashAugmented`], behind the
//! `hashenc` feature flag). This keeps the integration backward-compatible: the
//! layer's `forward` signature is unchanged; only `input_dim` grows when the
//! augmented source is selected.

use std::borrow::Cow;

/// A source of per-node feature vectors fed into GNN message passing.
pub trait FeatureSource: Send + Sync {
    /// Feature vector for `node_id` given its raw stored embedding `raw`.
    fn node_features<'a>(&self, node_id: u64, raw: &'a [f32]) -> Cow<'a, [f32]>;
    /// Output feature width (the layer's `input_dim`).
    fn out_dim(&self) -> usize;
}

/// Legacy behaviour: features == the raw embedding. Zero overhead, default path.
#[derive(Clone, Debug)]
pub struct FlatEmbedding {
    dim: usize,
}

impl FlatEmbedding {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl FeatureSource for FlatEmbedding {
    #[inline]
    fn node_features<'a>(&self, _node_id: u64, raw: &'a [f32]) -> Cow<'a, [f32]> {
        Cow::Borrowed(raw)
    }
    #[inline]
    fn out_dim(&self) -> usize {
        self.dim
    }
}

#[cfg(feature = "hashenc")]
mod augmented {
    use super::*;
    use ruvector_hashenc::HashEncoder;
    use std::sync::Arc;

    /// `concat(optional raw, multiresolution_hash_encoding(raw))`.
    ///
    /// The encoder's trainable tables are updated through the same persistent
    /// gradient flow as node embeddings (ADR-258 §6.1), so differentiability and
    /// self-improvement are preserved.
    #[derive(Clone)]
    pub struct HashAugmented {
        encoder: Arc<HashEncoder>,
        include_raw: bool,
        raw_dim: usize,
        out_dim: usize,
    }

    impl HashAugmented {
        /// Build from an encoder. If `include_raw`, the raw embedding is
        /// concatenated ahead of the `L*F` encoded features.
        pub fn new(encoder: Arc<HashEncoder>, raw_dim: usize, include_raw: bool) -> Self {
            let out_dim = encoder.output_dim() + if include_raw { raw_dim } else { 0 };
            Self {
                encoder,
                include_raw,
                raw_dim,
                out_dim,
            }
        }

        pub fn encoder(&self) -> &Arc<HashEncoder> {
            &self.encoder
        }
    }

    impl FeatureSource for HashAugmented {
        fn node_features<'a>(&self, _node_id: u64, raw: &'a [f32]) -> Cow<'a, [f32]> {
            let enc = self.encoder.encode(raw);
            if self.include_raw {
                let mut v = Vec::with_capacity(self.raw_dim + enc.len());
                v.extend_from_slice(&raw[..self.raw_dim.min(raw.len())]);
                v.extend_from_slice(&enc);
                Cow::Owned(v)
            } else {
                Cow::Owned(enc)
            }
        }
        #[inline]
        fn out_dim(&self) -> usize {
            self.out_dim
        }
    }
}

#[cfg(feature = "hashenc")]
pub use augmented::HashAugmented;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_is_identity() {
        let fs = FlatEmbedding::new(4);
        let raw = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(&*fs.node_features(7, &raw), &raw);
        assert_eq!(fs.out_dim(), 4);
    }

    #[cfg(feature = "hashenc")]
    #[test]
    fn hash_augmented_concats_and_sizes() {
        use ruvector_hashenc::{HashEncConfig, HashEncoder};
        use std::sync::Arc;
        let cfg = HashEncConfig {
            levels: 6,
            features_per_level: 2,
            log2_table_size: 12,
            index_dims: 3,
            n_min: 4,
            n_max: 64,
            ..Default::default()
        };
        let enc = Arc::new(HashEncoder::new(cfg, 8));
        let fs = HashAugmented::new(enc.clone(), 8, true);
        assert_eq!(fs.out_dim(), 8 + enc.output_dim());
        let raw = vec![0.5f32; 8];
        let feats = fs.node_features(1, &raw);
        assert_eq!(feats.len(), 8 + enc.output_dim());
    }
}
