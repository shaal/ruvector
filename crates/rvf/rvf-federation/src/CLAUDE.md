# rvf-federation/src

Source.

## Files

- `lib.rs` — ADR-057 docs, module decls, public re-exports.
- `aggregate.rs` — `FederatedAggregator` + `AggregationStrategy` (FedAvg, FedProx, Byzantine-tolerant).
- `diff_privacy.rs` — `DiffPrivacyEngine` (Gaussian/Laplace noise, gradient clipping) + `PrivacyAccountant` (RDP composition).
- `federation.rs` — `ExportBuilder`, `ImportMerger`; version-aware conflict resolution.
- `pii_strip.rs` — `PiiStripper`: detect → redact → attest (regex-based detection).
- `policy.rs` — `FederationPolicy` (per-jurisdiction config).
- `types.rs` — wire types for `FederatedManifest` / `DiffPrivacyProof` / `RedactionLog` / `AggregateWeights` segments.
- `error.rs` — `FederationError`.
