# rvf-kernel-optimized/benches

Criterion benchmark for the verified RVF kernel embedding pipeline.

## Files

- `verified_rvf.rs` - End-to-end benchmark over the verified ingest/lookup path.

## How to run

```bash
cargo bench -p rvf-kernel-optimized --bench verified_rvf
```

## Related

- Implementations: `../src/verified_ingest.rs`, `../src/kernel_embed.rs`.
- Crate: `crates/ruvector-verified`.
