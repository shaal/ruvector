# gene-consciousness

Gene regulatory network consciousness explorer. Applies IIT Phi (via `ruvector-consciousness`) to gene regulatory networks to identify emergent regulatory modules, comparing normal vs oncogenic (cancer) network rewiring.

## Important files
- `Cargo.toml` - binary crate `gene-consciousness`. Depends on `ruvector-consciousness` (features `phi`, `emergence`, `collapse`) plus `rand` / `rand_chacha`.
- `src/main.rs` - CLI entry.
- `src/data.rs`, `src/analysis.rs`, `src/report.rs` - data loading / analysis / reporting stages.

## Run
- `cargo run -p gene-consciousness --release`.

## Related
- Same dependency, different domain: `../cmb-consciousness/` (cosmology).
- Boundary-discovery family using mincut/spectral graph tools instead of IIT: `../brain-boundary-discovery/`, `../weather-boundary-discovery/`, `../music-boundary-discovery/`, `../temporal-attractor-discovery/`.
