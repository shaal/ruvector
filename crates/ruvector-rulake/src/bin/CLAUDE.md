# ruvector-rulake/src/bin

Executables.

## Files

- `rulake-demo.rs` → `rulake-demo` binary. Measures direct `RabitqPlusIndex` throughput, ruLake intermediary throughput (cache-hit), the intermediary tax (ratio), prime time vs direct build time, and federation across 2/4 backends. Same dataset / seed / queries for every row so numbers compare directly. Output backs `../../BENCHMARK.md`.
