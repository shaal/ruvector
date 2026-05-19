# ruvector-graph-transformer-node

Node.js bindings for the RuVector Graph Transformer via NAPI-RS. Exposes
proof-gated operations, sublinear attention, physics-informed layers
(Hamiltonian dynamics), biologically-inspired learning (spiking nets, Hebbian
plasticity), verified training with proof receipts, manifold distance,
temporal causal attention, and economic game-theoretic attention.

Embeds a self-contained graph transformer implementation in `src/transformer.rs`
to avoid coupling with the evolving `ruvector-graph-transformer` crate.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib"]`. `napi 2.16` (napi9, async,
  serde-json). Release: `lto`, `strip`. Many lints relaxed.
- `build.rs` - one-liner invoking `napi-build`.
- `index.js` + `index.d.ts` - generated NAPI loader and TypeScript types
  (these ship with the npm package).
- `package.json` - npm metadata; per-platform prebuilts under `npm/`.
- `src/lib.rs` - `#[napi]` bindings, primarily the `GraphTransformer` class.
- `src/transformer.rs` - internal `CoreGraphTransformer` + `Edge` +
  `PipelineStage` (the self-contained engine).

## Public API surface
- `GraphTransformer` (`#[napi]`) - JS class wrapping `CoreGraphTransformer`;
  `version()`, plus pipeline construction and verified-training methods.

## Related
- `../ruvector-graph-transformer` - upstream Rust crate (not directly
  depended on; the binding copies the surface it needs).
- Sibling NAPI binding: `ruvector-gnn-node`.
