//! WASM-safe HNSW backend built on `ruvector-hyperbolic-hnsw`.
//!
//! The default HNSW backend (`feature = "hnsw"`) pulls in `mmap-rs`,
//! `rayon`, `num_cpus`, and `cpu-time` through `hnsw_rs`, none of which
//! build for `wasm32-unknown-unknown`. This backend wraps the pure-Rust
//! HNSW implementation from `ruvector-hyperbolic-hnsw`, configured with
//! a Euclidean or Cosine metric (the crate supports both in addition to
//! its native Poincaré mode), so downstream wasm consumers get an ANN
//! index instead of falling back to flat brute force.
//!
//! The hyperbolic crate's `insert`/`search` project vectors onto the
//! Poincaré ball with `curvature = 1.0`. This is benign for `Cosine`
//! (cosine is scale-invariant per vector, so uniform or non-uniform
//! shrinking preserves ranking), and for `Euclidean` it is benign only
//! when all vectors already have `|v| < 1 - 1e-5`. Callers targeting
//! Euclidean with large-norm vectors should normalise upstream.
//!
//! Removal: the underlying HNSW does not support node deletion (same
//! limitation as `hnsw_rs` — see `index/hnsw.rs:339`). `remove` marks
//! the internal id tombstoned and `search` over-fetches by the current
//! tombstone count so the final top-K still contains `k` live hits.

use crate::error::{Result, RuvectorError};
use crate::index::VectorIndex;
use crate::types::{DistanceMetric, HnswConfig, SearchResult, VectorId};

use ruvector_hyperbolic_hnsw::hnsw::{
    DistanceMetric as HhDistanceMetric, HyperbolicHnsw, HyperbolicHnswConfig,
};

use std::collections::HashMap;

/// HNSW index that builds on `wasm32-unknown-unknown`.
pub struct HnswWasmIndex {
    inner: HyperbolicHnsw,
    dimensions: usize,
    /// Internal id (dense `usize`, == index in this Vec) -> user `VectorId`.
    /// `None` slots are tombstones left over from `remove`.
    internal_to_user: Vec<Option<VectorId>>,
    /// Reverse lookup for O(1) `remove`.
    user_to_internal: HashMap<VectorId, usize>,
    live_count: usize,
    tombstone_count: usize,
}

impl HnswWasmIndex {
    /// Build an index. Errors if the requested `DistanceMetric` is not one
    /// the hyperbolic crate exposes for non-Poincaré use.
    pub fn new(
        dimensions: usize,
        metric: DistanceMetric,
        core_cfg: HnswConfig,
    ) -> Result<Self> {
        let hh_metric = match metric {
            DistanceMetric::Euclidean => HhDistanceMetric::Euclidean,
            DistanceMetric::Cosine => HhDistanceMetric::Cosine,
            DistanceMetric::DotProduct | DistanceMetric::Manhattan => {
                return Err(RuvectorError::InvalidParameter(format!(
                    "HnswWasmIndex does not support {metric:?}; \
                     only Euclidean and Cosine are available"
                )));
            }
        };

        let max_conn = core_cfg.m.max(1);
        let level_mult = {
            let m = (max_conn as f32).ln();
            if m > 0.0 { 1.0 / m } else { 1.0 }
        };
        let config = HyperbolicHnswConfig {
            max_connections: max_conn,
            max_connections_0: max_conn * 2,
            ef_construction: core_cfg.ef_construction.max(max_conn),
            ef_search: core_cfg.ef_search.max(1),
            level_mult,
            curvature: 1.0, // ignored for Euclidean/Cosine paths
            metric: hh_metric,
            prune_factor: 10,
            use_tangent_pruning: false, // hyperbolic-only optimisation
        };

        Ok(Self {
            inner: HyperbolicHnsw::new(config),
            dimensions,
            internal_to_user: Vec::new(),
            user_to_internal: HashMap::new(),
            live_count: 0,
            tombstone_count: 0,
        })
    }
}

impl VectorIndex for HnswWasmIndex {
    fn add(&mut self, id: VectorId, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }
        // Upsert semantics: re-inserting the same id tombstones the old
        // entry and adds a new node with the updated vector. FlatIndex
        // does this via `DashMap::insert` (silent replace), and
        // downstream code — the car-learning bridge in particular —
        // calls .insert() with the same id when refreshing observation
        // vectors. Without the tombstone-first step we'd reject with
        // "duplicate id" and break the bridge's archive/observe path.
        // HyperbolicHnsw doesn't support node removal, so the old node
        // lingers in the graph with internal_to_user[old] = None; the
        // search path already filters those out and over-fetches by
        // tombstone_count.
        if let Some(old_internal) = self.user_to_internal.remove(&id) {
            if let Some(slot) = self.internal_to_user.get_mut(old_internal) {
                *slot = None;
            }
            self.tombstone_count += 1;
            self.live_count = self.live_count.saturating_sub(1);
        }
        let internal = self.inner.insert(vector).map_err(|e| {
            RuvectorError::IndexError(format!("hyperbolic insert failed: {e:?}"))
        })?;
        debug_assert_eq!(internal, self.internal_to_user.len());
        self.internal_to_user.push(Some(id.clone()));
        self.user_to_internal.insert(id, internal);
        self.live_count += 1;
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: query.len(),
            });
        }
        if k == 0 || self.live_count == 0 {
            return Ok(Vec::new());
        }
        // Over-fetch so k live hits remain after tombstone filtering.
        let fetch = k.saturating_add(self.tombstone_count).min(self.internal_to_user.len());
        let hits = self.inner.search(query, fetch).map_err(|e| {
            RuvectorError::IndexError(format!("hyperbolic search failed: {e:?}"))
        })?;

        let mut out = Vec::with_capacity(k);
        for hit in hits {
            let Some(user_id) = self
                .internal_to_user
                .get(hit.id)
                .and_then(|slot| slot.as_ref())
            else {
                continue; // tombstone or out-of-range (shouldn't happen)
            };
            out.push(SearchResult {
                id: user_id.clone(),
                score: hit.distance,
                vector: None,
                metadata: None,
            });
            if out.len() == k {
                break;
            }
        }
        Ok(out)
    }

    fn remove(&mut self, id: &VectorId) -> Result<bool> {
        let Some(internal) = self.user_to_internal.remove(id) else {
            return Ok(false);
        };
        if let Some(slot) = self.internal_to_user.get_mut(internal) {
            *slot = None;
        }
        self.live_count = self.live_count.saturating_sub(1);
        self.tombstone_count += 1;
        Ok(true)
    }

    fn len(&self) -> usize {
        self.live_count
    }
}

// Tests live in `crates/ruvector-core/tests/hnsw_wasm_integration.rs`
// rather than an inline `#[cfg(test)]` block, to sidestep unrelated
// pre-existing compile errors in other internal test modules under the
// `--no-default-features --features memory-only` combo this backend
// targets. Integration tests compile against the public API only.

