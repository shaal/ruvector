# ruvix/benches/benches

Criterion bench sources for the `ruvix-bench` package. Each is `harness = false`.

## Files

- `syscall_benches.rs` — direct microbenchmarks of the 12 RuVix syscalls.
- `proof_tiers.rs` — measures Tier 0 / Tier 1 / Tier 2 proof verification latency.
- `throughput.rs` — sustained syscall throughput under load.
- `linux_comparison.rs` — side-by-side latency vs equivalent Linux syscalls.
