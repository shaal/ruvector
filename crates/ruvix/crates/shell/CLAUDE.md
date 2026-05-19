# ruvix-shell

In-kernel debug shell for the RuVix Cognition Kernel (ADR-087). Enables runtime inspection of kernel state, memory stats, task
info, capability table, queues, vector store, proof subsystem, witness log, performance counters, and syscall tracing.
`no_std` (uses only `alloc`); line-based parsing for serial consoles; trait-based `ShellBackend` for kernel integration.

## Commands

`help`, `info`, `mem`, `tasks`, `caps`, `queues`, `vectors`, `proofs`, `cpu`, `witness`, `perf`, `trace`, `reboot`.

## Files

- `Cargo.toml` — depends only on `ruvix-types` (`alloc` feature).
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.

## Features

- `default = ["alloc"]`.

## Backend integration

Implementer: `ruvix_nucleus::shell_backend` provides a `ShellBackend` impl backed by the live kernel state.
