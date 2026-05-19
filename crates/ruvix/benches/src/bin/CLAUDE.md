# ruvix/benches/src/bin

Binary entry points for the `ruvix-bench` package. Each is registered as a `[[bin]]` target in the parent `Cargo.toml`.

## Files

- `ruvix_vs_linux.rs` — `ruvix-vs-linux` binary: runs the full RuVix-vs-Linux comparison and emits a report.
- `syscall_bench.rs` — `syscall-bench` binary: focused syscall microbench runner.
- `proof_overhead.rs` — `proof-overhead` binary: isolates the per-tier proof verification overhead.
- `memory_bench.rs` — memory subsystem (region/slab/buddy) benchmark binary.
- `throughput_bench.rs` — sustained throughput benchmark binary.

Run with `cargo run -p ruvix-bench --release --bin <name>`.
