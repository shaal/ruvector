# scipix

`ruvector-scipix` crate: Rust OCR engine for scientific documents - extracts LaTeX/MathML from math equations, research papers, and technical diagrams, with ONNX (GPU) acceleration. Ships a CLI, an HTTP API server, MCP integration, a WASM build, and a sample web UI.

## Files

- `Cargo.toml` (~12 KB) - Workspace-member manifest. Many optional features (`image`, `ml`, `wasm`); deps include axum, tower, moka, ort, image, validator, governor, hmac.
- `Makefile` - Common dev/build/run targets.
- `Cargo.lock` is provided at workspace root.
- `.env.example` - Sample environment configuration.
- `BUILD_WASM.md`, `IMPLEMENTATION_SUMMARY.md`, `IMPLEMENTATION_SUMMARY.txt`, `WASM_IMPLEMENTATION_SUMMARY.md`, `CHANGELOG.md` - Project docs.
- `src/` - Library + bins (`server`, `cli`, `benchmark`) split into api/cache/cli/math/ocr/optimize/output/preprocess/wasm.
- `examples/` - Usage demos (simple OCR, batch, streaming, API server, lean agentic, etc.) + `wasm_demo.html`.
- `benches/` - Criterion benchmarks (ocr, api, cache, inference, latex, memory, optimization, preprocessing).
- `tests/` - Unit + integration tests with fixtures.
- `docs/` - 15+ detailed design / spec / roadmap docs.
- `scripts/` - Setup, model download, benchmark scripts.
- `web/` - Browser demo and TypeScript types for the WASM build.
- `assets/fonts/` - Bundled DejaVuSans font for rendering.

## How to run

```bash
cargo run -p ruvector-scipix --bin cli -- --help
cargo run -p ruvector-scipix --bin server
cargo run -p ruvector-scipix --bin benchmark
cargo test -p ruvector-scipix
cd web && ./build.sh && npm run serve
```

## Tech stack

- Rust + axum/tower/hyper, tokio, clap, moka, ort (ONNX), image+imageproc, ndarray+nalgebra, governor, validator, hmac+sha2.

## Related

- Other ONNX inference examples: `examples/onnx-embeddings`, `examples/onnx-embeddings-wasm`.
- WASM siblings: `examples/wasm/ios`, `examples/prime-radiant/wasm`.
