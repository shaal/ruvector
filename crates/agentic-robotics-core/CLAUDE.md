# agentic-robotics-core

ROS3 Core — a ground-up Rust rewrite of ROS targeting microsecond-scale determinism with hybrid WASM/native deployment via npm. Provides pub/sub messaging, RPC services, and message serialization for next-generation robot operating systems.

## Important files

- `Cargo.toml` — Workspace member. Depends on `zenoh`, `rustdds`, `tokio`, `rkyv`, `cdr`. Criterion benches under `[[bench]]`.
- `src/lib.rs` — Crate root. Declares modules, re-exports `Zenoh`, `Message`, `RobotState`, `PointCloud`, `Publisher`, `Subscriber`, `Service`, `Queryable`. Provides `init()` and `VERSION` constant.

## Source modules (`src/`)

- `middleware.rs` — Zenoh middleware transport.
- `serialization.rs` — CDR/rkyv message encoding.
- `message.rs` — Core `Message`, `RobotState`, `PointCloud` types.
- `publisher.rs` / `subscriber.rs` — Pub/sub primitives.
- `service.rs` — Request/response `Service` and `Queryable` RPC.
- `error.rs` — `Result`/`Error` types.

## Tests / Benches

- `benches/message_passing.rs` — Criterion bench (`harness = false`) measuring pub/sub round-trip latency with `hdrhistogram`.

## Related

- Sibling crates in `crates/ruvector-*` for vector ops. This crate is the "robotics" pillar; cross-references rustdds/zenoh as core middleware.
