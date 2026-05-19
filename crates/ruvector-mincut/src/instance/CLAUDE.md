# ruvector-mincut/src/instance

Instance abstractions: a min-cut "instance" wraps a graph + bounds + witness for a single problem being maintained.

- `mod.rs` — instance façade.
- `bounded.rs` — bounded-cut instance variant (used when cut size is constrained).
- `stub.rs` — minimal stub instance for tests / placeholders.
- `traits.rs` — `Instance` trait + supporting traits.
- `witness.rs` — instance-level witness handling.
