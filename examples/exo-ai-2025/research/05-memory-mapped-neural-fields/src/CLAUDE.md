# 05-memory-mapped-neural-fields/src

## Files

- `lib.rs` — re-exports.
- `mmap_neural_field.rs` — neural field stored in an mmap region; lazy
  page-in via `memmap2`.
- `lazy_activation.rs` — defer activation evaluation until needed.
- `tiered_memory.rs` — RAM / SSD / cold storage tiers.
- `prefetch_prediction.rs` — predict next accesses and prefetch.

## Related

- `../examples/` — runnable demos
- `../benches/` — perf measurements
- `../architecture.md`
