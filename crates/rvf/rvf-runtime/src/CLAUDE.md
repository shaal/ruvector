# rvf-runtime/src

Source for the user-facing RVF runtime.

## Core surface

- `lib.rs` — top-level docs + module decls.
- `store.rs` — `RvfStore` (primary API).
- `options.rs` — `RvfOptions`, `DistanceMetric`, `MetadataEntry`, `MetadataValue`, `QueryOptions`.
- `read_path.rs` / `write_path.rs` — query and ingest flows.
- `filter.rs` — filter expression evaluation (`FilterExpr` / `FilterValue`).
- `membership.rs` — point membership probes.
- `deletion.rs` — tombstone-style deletion.
- `status.rs` — store status / diagnostics.
- `locking.rs` — advisory single-writer lock file.
- `ffi.rs` — C-style FFI bridge.

## Maintenance

- `compaction.rs` / `compress.rs` — background compaction + segment compression.

## RVCOW (ADR-031)

- `cow.rs`, `cow_map.rs`, `cow_compact.rs` — copy-on-write branching, refcount map, compaction.

## AGI Cognitive Container (ADR-036)

- `agi_container.rs` — `AgiContainerBuilder`, `ParsedAgiManifest`.
- `agi_authority.rs` — authority levels.
- `agi_coherence.rs` — coherence thresholds.

## QR Cognitive Seed (ADR-033)

- `qr_encode.rs`, `qr_seed.rs`, `seed_crypto.rs` — encode/decode + HMAC-SHA256.

## Crypto integration

- `witness.rs` — WITNESS_SEG audit chain via `rvf-crypto`.

## Hardening

- `adversarial.rs`, `dos.rs`, `safety_net.rs` — defenses against malformed / hostile inputs.
