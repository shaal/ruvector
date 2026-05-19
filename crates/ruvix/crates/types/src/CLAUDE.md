# ruvix-types/src

One module per kernel concept. Pure `no_std`, no external deps.

## Files

- `lib.rs` — crate root + docs; re-exports the modules below.
- `capability.rs` — `Capability`, `CapRights`, `CapHandle`, derivation/grant types.
- `task.rs` — `TaskHandle`, `TaskPriority`, `TaskState`.
- `region.rs` — `RegionHandle`, `RegionPolicy` (Immutable/AppendOnly/Slab), region metadata.
- `queue.rs` — `QueueHandle`, ring descriptor types.
- `timer.rs` — `Timer`, `TimerHandle`, deadline types.
- `proof.rs` — `Proof`, `ProofToken`, `ProofTier` (Reflex/Standard/Deep).
- `proof_cache.rs` / `proof_cache_optimized.rs` — proof-cache entry types and a cache-friendly layout variant.
- `vector.rs` — `VectorKey`, vector-store kernel types.
- `graph.rs` — graph-store kernel types.
- `sensor.rs` — sensor input/event types.
- `rvf.rs` — RuVector Format (RVF) image/manifest types (used by `ruvix-boot`).
- `scheduler.rs` — `SchedulerPartition`, scheduler-facing types.
- `object.rs` — `ObjectType` enum tagging every kernel object.
- `handle.rs` — generic handle plumbing.
- `error.rs` — top-level `Error` / `Result`.
