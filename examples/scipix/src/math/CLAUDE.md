# scipix/src/math

Math parsing/representation/serialization layer.

## Files

- `mod.rs` - Module surface.
- `ast.rs` - Math AST node types.
- `parser.rs` - Parser into the AST.
- `symbols.rs` (~31 KB) - Large symbol/operator table.
- `latex.rs` - LaTeX serialization.
- `mathml.rs` - MathML serialization.
- `asciimath.rs` - AsciiMath serialization.

## Related

- OCR consumer: `../ocr/decoder.rs`.
- Output formatters: `../output/latex.rs`, `../output/mmd.rs`, `../output/html.rs`.
- Docs: `../../docs/06_LATEX_PIPELINE.md`.
