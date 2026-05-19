# ruvix-types

Zero-dep `no_std` kernel interface types for the RuVix Cognition Kernel (ADR-087). Shared between kernel code and RVF component
code. `#![forbid(unsafe_code)]`.

## The six primitives (ADR-087)

| Primitive | Purpose | Analog |
|---|---|---|
| Task | Unit of concurrent execution with a capability set | seL4 TCB |
| Capability | Unforgeable typed token granting access to a resource | seL4 capability |
| Region | Contiguous memory with access policy | seL4 Untyped + frame |
| Queue | Typed ring buffer for inter-task communication | io_uring SQ/CQ |
| Timer | Deadline-driven scheduling primitive | POSIX timer_create |
| Proof | Cryptographic attestation gating state mutation | Novel (ADR-047) |

## Files

- `Cargo.toml` — zero external runtime deps. Dev: criterion, proptest.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `benches/` — type-construction + serialization microbenches.
- `tests/` — `proof_cache_test.rs`, `types_test.rs`.

## Features

- `std`, `alloc` (default depends on context — see Cargo.toml).
