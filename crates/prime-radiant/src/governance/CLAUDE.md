# prime-radiant/src/governance

Governance layer: policy bundles, witness records, lineage, and storage repository.

## Files

- `mod.rs` — module entry.
- `policy.rs` — policy bundle (thresholds, lane mapping, escalation rules).
- `witness.rs` — witness record format (cryptographically auditable).
- `lineage.rs` — lineage chain of decisions for replay/audit.
- `repository.rs` — persistence interface for witnesses / lineage (backed by `storage/`).

## Related

- Threshold tuning by `sona_tuning/`.
- Storage backends in `storage/` (memory, file, postgres).
