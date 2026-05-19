# scipix/src/cache

Result caching layer for scipix using `moka` (async LRU/TinyLFU).

## Files

- `mod.rs` (~13 KB) - Cache trait, key/value types, moka-based implementation, eviction and stats.

## Related

- Benchmark: `../../benches/cache.rs`.
- Used by: `../api/handlers.rs`, `../ocr/engine.rs`.
