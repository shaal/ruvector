# scipix/src

Library + binaries for `ruvector-scipix`.

## Top-level files

- `lib.rs` - Public library API.
- `config.rs` - Crate configuration.
- `error.rs` - Error type.

## Subdirs

- `bin/` - Binaries: `cli`, `server`, `benchmark`.
- `api/` - HTTP API (axum): routes, handlers, middleware, jobs, requests/responses, state.
- `cli/` - CLI argument parsing and subcommands (`batch`, `config`, `doctor`, `mcp`, `ocr`, `serve`).
- `cache/` - moka-based result cache.
- `math/` - Math parsing/representation: AST, parser, LaTeX/MathML/AsciiMath, symbols.
- `ocr/` - OCR engine: models, decoder, inference, confidence, engine wrapper.
- `optimize/` - SIMD, quantization, parallel batching, memory optimizations.
- `output/` - Formatters for LaTeX, MathML, MMD, HTML, JSON, DOCX, SMILES.
- `preprocess/` - Image preprocessing (deskew, rotation, enhancement, segmentation, transforms, pipeline).
- `wasm/` - WASM bindings (api, canvas, memory, types, worker).

## Related

- Tests: `../tests/`.
- Examples: `../examples/`.
- Web demo: `../web/`.
