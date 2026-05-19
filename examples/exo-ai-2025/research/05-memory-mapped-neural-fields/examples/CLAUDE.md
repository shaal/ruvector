# 05-memory-mapped-neural-fields/examples

## Files

- `basic_usage.rs` — create + populate + query an mmap-backed neural
  field.
- `petabyte_scale.rs` — scale-test that allocates a sparse logical
  petabyte-scale field and exercises lazy paging + tiered memory.

## Run

```bash
cargo run --release --example basic_usage
cargo run --release --example petabyte_scale
```

## Related

- `../src/`
