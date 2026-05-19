# ruqu-wasm

Browser-compatible quantum circuit simulation — `wasm-bindgen` bindings over `ruqu-core` and `ruqu-algorithms`. Supports up to 25 qubits in WASM (memory limit enforcement, ~512 MB state vector). Exposes a JS-friendly API for building circuits, running simulations, and executing Grover/QAOA.

## Important files

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Depends on `ruqu-core`, `ruqu-algorithms` (path), `wasm-bindgen`, `js-sys`, `serde-wasm-bindgen`, `getrandom` (js feature). Dev: `wasm-bindgen-test`. Disables `wasm-opt` in release profile.
- `src/lib.rs` — Single-file WASM facade. `WasmQuantumCircuit`, `simulate`, `max_qubits`, `estimate_memory` exports. Defines max-qubit constant (~25).

## Public API (JS surface)

- `class WasmQuantumCircuit(n_qubits)` — `.h(i)`, `.cnot(c, t)`, `.measure_all()`, etc.
- `simulate(circuit) -> { probabilities, ... }`.
- `max_qubits()`, `estimate_memory(n_qubits)`.

## Build

```
wasm-pack build crates/ruqu-wasm --target web
```

## Related

- `ruqu-core` (state-vector engine), `ruqu-algorithms` (Grover/QAOA/VQE).
- Sibling exotic crate: `ruqu-exotic`.
- Classical sibling: `ruQu`.
