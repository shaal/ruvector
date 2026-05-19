# rvf-federation

Federated RVF transfer learning (ADR-057): PII stripping, differential privacy (Gaussian/Laplace, RDP accountant, gradient clipping), federation export/import with version-aware conflict resolution, and federated aggregation (FedAvg, FedProx, Byzantine-tolerant weighted averaging). Adds the `FederatedManifest`, `DiffPrivacyProof`, `RedactionLog`, `AggregateWeights` segment types.

## Layout

- `Cargo.toml` — features `default = ["std"]`, optional `serde`. Deps: `sha3`, `rand`, `rand_distr`, `regex`, `thiserror`; optional `serde`.
- `src/lib.rs` — module decls + public re-exports.
- `src/aggregate.rs` — `FederatedAggregator`, `AggregationStrategy`.
- `src/diff_privacy.rs` — `DiffPrivacyEngine`, `PrivacyAccountant`.
- `src/federation.rs` — `ExportBuilder`, `ImportMerger`.
- `src/pii_strip.rs` — `PiiStripper` (3-stage: detect → redact → attest).
- `src/policy.rs` — `FederationPolicy`.
- `src/types.rs` — wire types for federation segments.
- `src/error.rs` — `FederationError`.
- `benches/federation_bench.rs` — Criterion benches for the pipeline.

## Public API

`FederationError`, `AggregationStrategy`, `FederatedAggregator`, `DiffPrivacyEngine`, `PrivacyAccountant`, `ExportBuilder`, `ImportMerger`, `PiiStripper`, `FederationPolicy`, plus the segment types.

## Related

- `../rvf-runtime`, `../rvf-types` — federation segments plug into the RVF store
- `../rvf-crypto` — used for redaction attestation hashing
