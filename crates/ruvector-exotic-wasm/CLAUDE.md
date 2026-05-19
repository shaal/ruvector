# ruvector-exotic-wasm

Exotic AI coordination primitives for emergent behavior in distributed agent
systems, exposed as a WASM (and rlib) crate. Includes:

- **Neural Autonomous Organization (NAO)** — decentralized governance via
  stake-weighted quadratic voting, oscillatory sync, and quorum consensus.
- **Morphogenetic Network** — biologically-inspired growth with morphogen-gradient
  differentiation, emergent topology, and synaptic pruning.
- **Time Crystal** — quantum-inspired periodic-state primitive.

## Layout

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Deps: wasm-bindgen, js-sys,
  serde / serde-wasm-bindgen, getrandom, rand, optional console_error_panic_hook.
- `src/lib.rs` — module roots + crate-level docs/examples.
- `src/nao.rs` — `NeuralAutonomousOrg`, member management, proposals, voting.
- `src/morphogenetic.rs` — `MorphogeneticNetwork` growth / pruning.
- `src/time_crystal.rs` — `TimeCrystal` primitive.
- `pkg/` — generated wasm-pack output (`ruvector_exotic_wasm.js`, `.wasm`, `.d.ts`,
  `package.json`).

## Public API (rlib + wasm-bindgen)

`NeuralAutonomousOrg`, `MorphogeneticNetwork`, `TimeCrystal`. JS example in lib.rs
docs.
