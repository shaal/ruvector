# ruqu-wasm/src

Single-file source.

## Files

- `lib.rs` — All `#[wasm_bindgen]` exports for browser quantum simulation. `WasmQuantumCircuit` builder, `simulate` runner, `max_qubits` / `estimate_memory` capability helpers. Enforces the 25-qubit memory ceiling.
