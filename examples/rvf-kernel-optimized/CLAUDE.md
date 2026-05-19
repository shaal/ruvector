# rvf-kernel-optimized

`rvf-kernel-optimized` crate: hyper-optimized RVF example showing Linux-kernel embedding with `ruvector-verified` formal proofs (features `ultra`, `hnsw-proofs`) on the RVF runtime/kernel/ebpf/quant stack. Used to benchmark the verified path end-to-end.

## Files

- `Cargo.toml` - Manifest; depends on `ruvector-verified` with `ultra` + `hnsw-proofs`, full `rvf-*` stack; bin `rvf-kernel-opt`, bench `verified_rvf`.
- `src/main.rs` - Demo entry point.
- `src/lib.rs` - Shared helpers.
- `src/kernel_embed.rs` - Linux kernel embedding routine.
- `src/verified_ingest.rs` - Verified ingestion pipeline.
- `benches/verified_rvf.rs` - Criterion benchmark.
- `tests/integration.rs` - Integration tests.

## How to run

```bash
cargo run -p rvf-kernel-optimized --bin rvf-kernel-opt --release
cargo bench -p rvf-kernel-optimized --bench verified_rvf
cargo test -p rvf-kernel-optimized
```

## Tech stack

- Rust 2021 (MSRV 1.77). Internal: `ruvector-verified` (FastTermArena, gated routing, pools, cache), `rvf-types`, `rvf-runtime`, `rvf-kernel`, `rvf-ebpf`, `rvf-quant`.

## Related

- Sibling: `examples/rvf` (broader, less-verified examples).
- Crate: `crates/ruvector-verified`.
