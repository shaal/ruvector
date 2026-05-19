# profiling

Top-level profiling toolkit for the ruvector monorepo. Not a Rust crate — contains only shell scripts that wire perf / valgrind / flamegraph against the workspace binaries (notably `ruvector-bench`). No `Cargo.toml`.

## Layout

- `scripts/` — runnable bash scripts (install tools, CPU/memory profile, flamegraph, benchmark-all, run_all_analysis).

## Notes

Reports are typically written to `./reports` (created by the scripts). Pair with `../ruvector-bench` for benchmark targets.
