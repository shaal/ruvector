# exo-ai-2025/benches

Workspace-level Criterion benchmarks plus a runner script.

## Files

- `federation_bench.rs` — distributed cognition path benches (CRDT
  merge, consensus, onion routing).
- `hypergraph_bench.rs` — hypergraph topology / persistent-homology
  ops on the exo-hypergraph crate.
- `manifold_bench.rs` — SIREN manifold deformation + retrieval
  (exo-manifold).
- `temporal_bench.rs` — short/long-term temporal memory ops, quantum
  decay (exo-temporal).
- `run_benchmarks.sh` — convenience runner that invokes
  `cargo bench` for each suite and stages HTML reports.

## Run

```bash
bash benches/run_benchmarks.sh
# or
cargo bench --bench hypergraph_bench
```

## Related

- `../docs/BENCHMARK_USAGE.md`, `../docs/PERFORMANCE_BASELINE.md`
- `../crates/` — the crates being benchmarked
