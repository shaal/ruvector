# profiling/scripts

Bash scripts for performance analysis of the ruvector workspace.

## Files

- `install_tools.sh` — apt-install perf, valgrind, etc; cargo-install flamegraph/criterion helpers.
- `cpu_profile.sh` — record perf data for a chosen binary, emit reports under `../reports/`.
- `memory_profile.sh` — valgrind massif / dhat memory profiling.
- `generate_flamegraph.sh` — produce flamegraph SVGs from perf or dtrace data.
- `benchmark_all.sh` — run every benchmark crate end-to-end.
- `run_all_analysis.sh` — top-level driver invoking the above scripts in sequence.

Sets `PROJECT_ROOT` to the workspace root and writes outputs to `crates/profiling/reports/`.
