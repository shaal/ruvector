# agentic-robotics-core/src

Source root for ROS3 Core. Each file is a flat top-level module re-exported from `lib.rs`.

## Files

- `lib.rs` — Crate entry; declares modules and re-exports public API. Provides `init()` to start tracing.
- `middleware.rs` — `Zenoh` middleware abstraction (transport layer over zenoh).
- `serialization.rs` — Message serialization helpers (CDR + rkyv zero-copy).
- `message.rs` — `Message`, `RobotState`, `PointCloud` core message types.
- `publisher.rs` — `Publisher` for topic publication.
- `subscriber.rs` — `Subscriber` for topic subscription.
- `service.rs` — `Service` and `Queryable` (request/response RPC).
- `error.rs` — `Result<T>` and `Error` enum used crate-wide.

## Pointers

- See `benches/message_passing.rs` for performance baselines.
