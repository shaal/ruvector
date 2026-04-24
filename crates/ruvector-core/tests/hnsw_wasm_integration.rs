//! Integration tests for the `hnsw-wasm` backend.
//!
//! See `crates/ruvector-core/src/index/hnsw_wasm.rs` for why these live
//! as an integration test rather than an inline `#[cfg(test)]` module.

#![cfg(feature = "hnsw-wasm")]

use ruvector_core::index::hnsw_wasm::HnswWasmIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

// Compile-time check: VectorIndex trait bound is `Send + Sync`; P2 will
// dispatch through `Box<dyn VectorIndex>`, which requires both. Catch a
// regression here rather than in the vector_db.rs rewrite.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync + ?Sized>() {}
    assert_send_sync::<HnswWasmIndex>();
    assert_send_sync::<dyn VectorIndex>();
};

#[test]
fn cosine_top_k_recovers_planted_match() {
    let mut idx =
        HnswWasmIndex::new(4, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
    idx.add("a".into(), unit(vec![1.0, 0.0, 0.0, 0.0])).unwrap();
    idx.add("b".into(), unit(vec![0.9, 0.1, 0.0, 0.0])).unwrap();
    idx.add("c".into(), unit(vec![0.0, 1.0, 0.0, 0.0])).unwrap();
    idx.add("d".into(), unit(vec![0.0, 0.0, 1.0, 0.0])).unwrap();
    idx.add("e".into(), unit(vec![-1.0, 0.0, 0.0, 0.0])).unwrap();
    assert_eq!(idx.len(), 5);

    let results = idx.search(&unit(vec![1.0, 0.0, 0.0, 0.0]), 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "a", "top-1 should be the exact-match vector");
    assert_eq!(results[1].id, "b");
    assert!(results[0].score <= results[1].score);
}

#[test]
fn euclidean_top_k_recovers_nearest() {
    // Keep |v| < 1 so the hyperbolic crate's Poincaré-ball projection at
    // insert time is a no-op and euclidean distances are preserved exactly.
    let mut idx =
        HnswWasmIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default()).unwrap();
    idx.add("origin".into(), vec![0.0, 0.0]).unwrap();
    idx.add("near".into(), vec![0.01, 0.01]).unwrap();
    idx.add("mid".into(), vec![0.3, 0.3]).unwrap();

    let results = idx.search(&[0.0, 0.0], 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "origin");
    assert_eq!(results[1].id, "near");
}

#[test]
fn unsupported_metrics_error() {
    for m in [DistanceMetric::DotProduct, DistanceMetric::Manhattan] {
        let r = HnswWasmIndex::new(3, m, HnswConfig::default());
        assert!(r.is_err(), "metric {m:?} should be unsupported");
    }
}

#[test]
fn dimension_mismatch_on_add() {
    let mut idx =
        HnswWasmIndex::new(3, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
    assert!(idx.add("a".into(), vec![1.0, 0.0]).is_err());
    assert!(idx.add("b".into(), vec![1.0, 0.0, 0.0, 0.0]).is_err());
}

#[test]
fn duplicate_id_rejected() {
    let mut idx =
        HnswWasmIndex::new(2, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
    idx.add("dup".into(), unit(vec![1.0, 0.0])).unwrap();
    assert!(idx.add("dup".into(), unit(vec![0.0, 1.0])).is_err());
}

#[test]
fn remove_tombstones_and_search_skips() {
    let mut idx =
        HnswWasmIndex::new(2, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
    idx.add("a".into(), unit(vec![1.0, 0.0])).unwrap();
    idx.add("b".into(), unit(vec![0.9, 0.1])).unwrap();
    idx.add("c".into(), unit(vec![0.0, 1.0])).unwrap();
    assert_eq!(idx.len(), 3);

    assert!(idx.remove(&"a".to_string()).unwrap());
    // Idempotent: a second remove on the same id returns false.
    assert!(!idx.remove(&"a".to_string()).unwrap());
    assert_eq!(idx.len(), 2);

    let results = idx.search(&unit(vec![1.0, 0.0]), 2).unwrap();
    let ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
    assert!(
        !ids.contains(&"a".to_string()),
        "tombstoned id must not surface; got {ids:?}"
    );
    assert_eq!(ids[0], "b");
}

#[test]
fn search_over_fetch_still_returns_k_live_when_tombstones_many() {
    // Plant 10 unit vectors, tombstone half, ask for top-3. We should
    // still get 3 live ids back because search over-fetches.
    let mut idx =
        HnswWasmIndex::new(3, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
    for i in 0..10 {
        let v = unit(vec![i as f32, 1.0, 0.5]);
        idx.add(format!("v{i}"), v).unwrap();
    }
    for i in 0..5 {
        assert!(idx.remove(&format!("v{i}")).unwrap());
    }
    let results = idx.search(&unit(vec![9.0, 1.0, 0.5]), 3).unwrap();
    assert_eq!(results.len(), 3, "expected k=3 live hits, got {results:?}");
    for r in &results {
        let n: usize = r.id.trim_start_matches('v').parse().unwrap();
        assert!(n >= 5, "tombstoned id {} surfaced", r.id);
    }
}

#[test]
fn empty_index_returns_empty_results() {
    let idx =
        HnswWasmIndex::new(3, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
    let r = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
    assert!(r.is_empty());
}

#[test]
fn dimension_mismatch_on_search() {
    let mut idx =
        HnswWasmIndex::new(3, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
    idx.add("a".into(), unit(vec![1.0, 0.0, 0.0])).unwrap();
    assert!(idx.search(&[1.0, 0.0], 1).is_err());
}
