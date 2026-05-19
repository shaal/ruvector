# ruvector-nervous-system-wasm

Bio-inspired neural system components compiled for the browser. Provides BTSP
(Behavioral Timescale Synaptic Plasticity, one-shot learning), HDC
(Hyperdimensional Computing, 10000-bit binary hypervectors), Winner-Take-All
(<1us decisions), and Global Workspace (4-7 item attention bottleneck).

Target bundle size: <100KB. Release profile is `opt-level = "z"` + LTO +
single codegen unit + panic=abort + strip.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib", "rlib"]`. Self-contained (no
  upstream `ruvector-nervous-system` dep at present). Uses
  `getrandom` with `js` feature for WASM RNG.
- `src/lib.rs` - module wiring + `#[wasm_bindgen]` exports.
- `src/btsp.rs` - BTSP one-shot associative learning (`BTSPLayer`).
- `src/hdc.rs` - Hyperdimensional Computing: `Hypervector`, `HdcMemory`;
  XOR-bind <50ns, Hamming-distance similarity <100ns.
- `src/wta.rs` - Winner-Take-All competition (`WTALayer`, `KWTALayer`).
- `src/workspace.rs` - Global Workspace (`GlobalWorkspace`, `WorkspaceItem`).
- `pkg/` - generated `wasm-pack` output (JS loader, `.wasm`, TypeScript
  declarations). Checked into git for convenience.
- `tests/web.rs` - `wasm-bindgen-test` browser tests.

## Performance targets
- BTSP `one_shot_associate`: immediate (gradient normalization).
- HDC `bind`: <50ns; `similarity`: <100ns (Hamming + SIMD).
- WTA `compete`: <1us; K-WTA `select`: <10us; Workspace `broadcast`: <10us.

## Related
- Sibling WASM crates: `ruvector-consciousness-wasm`, `ruvector-delta-wasm`.
