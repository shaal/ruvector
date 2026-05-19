# delta-behavior / src / applications

Internal module that re-exports / glues the feature-gated applications shipped as binaries in `../../applications/`.

## Important files
- `mod.rs` - module root; conditionally includes each application's module behind its feature flag (`self-limiting-reasoning`, `world-model`, `swarm-intelligence`, etc.).

## Related
- The actual application sources: `../../applications/01-*.rs` ... `11-*.rs`.
- Feature definitions: `../../Cargo.toml`.
