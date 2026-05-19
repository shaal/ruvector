# rvf-cli/src/cmd

One file per CLI subcommand; `mod.rs` exposes them to `main.rs`.

## Files

- `mod.rs` — re-exports each command module.
- `create.rs` — `rvf create` — new empty RVF store.
- `ingest.rs` — `rvf ingest` — load vectors from a JSON file.
- `query.rs` — `rvf query` — nearest-neighbour search.
- `delete.rs` — `rvf delete` — by id or filter.
- `status.rs` — `rvf status` — show store stats.
- `compact.rs` — trigger compaction.
- `derive.rs` — derive a lineage child store.
- `embed_ebpf.rs` — embed compiled eBPF programs into the store.
- `embed_kernel.rs` — embed a Linux bzImage (KERNEL_SEG) for cognitive containers.
- `filter.rs` — apply filter expressions.
- `freeze.rs` — freeze a store (immutable snapshot).
- `inspect.rs` — dump segment-level details.
- `launch.rs` — launch the store as a QEMU microVM (via `rvf-launch`).
- `rebuild_refcounts.rs` — recompute reference counts for CoW maps.
- `serve.rs` — start HTTP/TCP server (uses optional `rvf-server` dep).
- `verify_attestation.rs` — verify TEE attestation in `ATTESTATION_SEG`.
- `verify_witness.rs` — verify WITNESS_SEG audit chain.
