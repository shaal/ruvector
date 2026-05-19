# edge-net/src/scheduler

Task scheduler that selects compute backends and orders work for a contributor.

## Important files
- `mod.rs` — module entry; scheduling policies.

## Related
- Backends it picks among: `../compute/`. Tasks it consumes: `../tasks/`.
