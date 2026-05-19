# ruvector-acorn

ACORN: Predicate-Agnostic Filtered HNSW. Pure-Rust implementation of the SIGMOD 2024
paper (Patel et al., arXiv:2403.04871). Solves the collapsed-recall problem of post-filter
HNSW at low selectivity by building a denser graph (gamma * M neighbours) and traversing all
neighbours regardless of predicate match. Provides 2-1000x QPS over post-filter patterns
at low selectivity.

## Layout

- `Cargo.toml` — lib + `[[bin]] acorn-demo` + `[[bench]] acorn_bench`. Deps: rand,
  rand_distr, rayon, thiserror; dev: criterion.
- `src/lib.rs` — module roots and public re-exports.
- `src/index.rs` — `AcornIndex1` (gamma=1, M=16), `AcornIndexGamma` (gamma=2, M=16),
  `FlatFilteredIndex` (brute-force baseline), `FilteredIndex` trait, `recall_at_k`.
- `src/graph.rs` — `AcornGraph`: HNSW graph with denser fan-out.
- `src/search.rs` — predicate-agnostic beam search.
- `src/dist.rs` — L2-squared distance kernels.
- `src/error.rs` — `AcornError`.
- `src/main.rs` — `acorn-demo` binary for ad-hoc benchmarking.
- `benches/acorn_bench.rs` — Criterion benchmark over the three index variants.

## Public API

`AcornGraph`, `FilteredIndex`, `FlatFilteredIndex`, `AcornIndex1`, `AcornIndexGamma`,
`recall_at_k`, `AcornError`.

## Variant guidance

| Struct              | gamma | M  | Use when                       |
|---------------------|-------|----|--------------------------------|
| FlatFilteredIndex   | N/A   | -  | Baseline / high selectivity     |
| AcornIndex1         | 1     | 16 | Moderate selectivity (>= 10%)   |
| AcornIndexGamma     | 2     | 16 | Low selectivity (< 10%)         |

## Related

- `crates/ruvector-acorn-wasm` — WASM bindings of this crate.
