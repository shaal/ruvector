# ruvector-nervous-system/src/dendrite

Reduced-compartment dendritic models that detect temporal coincidence of synaptic inputs within 10-50 ms windows. Based on the Dendrify framework and DenRAM RRAM circuits.

## Files

- `mod.rs` — façade; exposes `Dendrite`, `DendriticTree`.
- `compartment.rs` — single dendritic compartment.
- `tree.rs` — `DendriticTree` (compartment-tree assembly).
- `coincidence.rs` — coincidence-detection logic with NMDA-like threshold.
- `plateau.rs` — plateau-potential generation when threshold is crossed.
