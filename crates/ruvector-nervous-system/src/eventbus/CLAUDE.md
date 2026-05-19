# ruvector-nervous-system/src/eventbus

Sharded event bus with backpressure used to wire neural subsystems together. See `IMPLEMENTATION.md` (in this directory).

## Files

- `mod.rs` — `EventBus` façade.
- `event.rs` — `Event` and event-payload types.
- `queue.rs` — per-shard MPMC queue.
- `shard.rs` — sharding strategy.
- `backpressure.rs` — backpressure policy when consumers fall behind.
- `IMPLEMENTATION.md` — design notes.
