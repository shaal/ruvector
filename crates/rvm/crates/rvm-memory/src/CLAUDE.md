# rvm-memory/src

- `lib.rs` — crate root.
- `tier.rs` — `TierManager`: coherence-driven Hot/Warm/Dormant/Cold placement and explicit transitions.
- `allocator.rs` — `BuddyAllocator`: power-of-two physical page allocator.
- `region.rs` — `RegionManager`: owned region lifecycle, guest physical → host physical translation.
- `reconstruction.rs` — `ReconstructionPipeline`: rebuilds dormant regions from compressed checkpoint + delta.

See `../CLAUDE.md`.
