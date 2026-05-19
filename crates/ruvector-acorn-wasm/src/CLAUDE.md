# ruvector-acorn-wasm/src

Single-file WASM glue layer.

- `lib.rs` — `#[wasm_bindgen]` exports:
  - `init()` — panic hook entry point (called automatically via `#[wasm_bindgen(start)]`).
  - `SearchResult { id: u32, distance: f32 }` — single nearest-neighbor hit.
  - `AcornIndex` — wraps `AcornIndex1` (gamma=1) or `AcornIndexGamma` (gamma=2);
    static `build(vectors, dim, gamma)` constructor and `search(query, k, predicate)`.

All search distances are approximate L2-squared.
