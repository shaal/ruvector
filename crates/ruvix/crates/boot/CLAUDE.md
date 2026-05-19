# ruvix-boot

RVF (RuVector Format) boot loading for the RuVix Cognition Kernel (ADR-087 Section 9.1). Implements the five-stage boot sequence:
Hardware Init -> RVF Verify (manifest + ML-DSA-65 signature) -> Object Create (root task, regions, queues, witness log) ->
Component Mount + capability distribution -> First Attestation.

Critical security (SEC-001): signature failure PANICs immediately (no fallback boot path); after Stage 3 the root task drops to a
minimum capability set; witness log is append-only and cryptographically linked.

## Files

- `Cargo.toml` — depends on `ruvix-types`/`region`/`queue`/`cap`. Uses `sha2` (no-default-features) for hashing; ML-DSA-65 via
  pqcrypto.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.

## Features

- `std` (default), `alloc`, `metrics`.
