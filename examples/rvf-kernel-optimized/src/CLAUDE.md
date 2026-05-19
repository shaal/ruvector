# rvf-kernel-optimized/src

Sources for the hyper-optimized verified-RVF example.

## Files

- `lib.rs` - Shared types/helpers re-exported to bin and bench.
- `main.rs` - `rvf-kernel-opt` binary entry point (orchestrates kernel embedding demo).
- `kernel_embed.rs` - Embeds Linux kernel artifacts into an RVF store.
- `verified_ingest.rs` - Formally-verified ingestion pipeline using `ruvector-verified` ultra features.

## Related

- Bench: `../benches/verified_rvf.rs`.
- Tests: `../tests/integration.rs`.
