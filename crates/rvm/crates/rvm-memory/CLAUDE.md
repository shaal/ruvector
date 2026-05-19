# rvm-memory

Guest physical address space management for the RVM microhypervisor (ADR-136 and ADR-138). Provides a safe abstraction over a four-tier coherence-driven memory model with reconstruction capability.

## Four-tier memory (ADR-136)

| Tier | Name | Description |
|------|------|-------------|
| 0 | Hot | Per-core SRAM / L1-adjacent; always resident during execution |
| 1 | Warm | Shared DRAM; resident if residency rule is met |
| 2 | Dormant | Compressed checkpoint + delta; reconstructed on demand |
| 3 | Cold | Persistent archival; accessed only during recovery |

## Design constraints

`#![no_std] #![forbid(unsafe_code)]`, zero heap allocation, works without the coherence engine (DC-1 static fallback thresholds), all tier transitions explicit.

## Layout

- `Cargo.toml` — `rlib`; dep on `rvm-types`.
- `src/lib.rs` — module wiring and public API.
- `src/tier.rs` — `TierManager` (coherence-driven placement and transitions).
- `src/allocator.rs` — `BuddyAllocator` (power-of-two physical page allocator).
- `src/region.rs` — `RegionManager` (owned region lifecycle and address translation).
- `src/reconstruction.rs` — `ReconstructionPipeline` (dormant-state restoration).

See `../CLAUDE.md`.
