# rvm-security

Security-policy enforcement for the RVM microhypervisor. Provides the policy decision point combining capability checks, proof verification, and witness logging into a unified security gate. Every hypercall passes through three stages — capability check → proof verification → witness logging — before it proceeds.

`#![no_std] #![forbid(unsafe_code)] #![deny(missing_docs)]`.

## Features

- `default = ["crypto-sha256"]`.
- `std`, `alloc` (forwarded to `rvm-types`, `rvm-witness`).

## Layout

- `Cargo.toml` — `rlib`; deps `rvm-types`, `rvm-witness`, `subtle`; optional `sha2`.
- `src/lib.rs` — crate root and module wiring.
- `src/gate.rs` — unified security gate (single entry point per hypercall).
- `src/validation.rs` — input validation for security-critical parameters.
- `src/attestation.rs` — attestation chain and report generation.
- `src/budget.rs` — DMA and resource budget enforcement.

See `../CLAUDE.md`.
