# scipix/src/output

Output formatters that turn OCR results into target document formats.

## Files

- `mod.rs` - Module surface and `OutputFormat` enum.
- `formatter.rs` - Shared formatter trait/helpers.
- `latex.rs` - LaTeX output.
- `mathml.rs` lives under `../math/`; here we have wrapper format `mmd.rs` (MultiMarkdown).
- `mmd.rs` - MultiMarkdown output.
- `html.rs` - HTML output.
- `json.rs` - JSON output.
- `docx.rs` - DOCX writer.
- `smiles.rs` - SMILES chemistry output.

## Related

- Math AST: `../math/`.
- Docs: `../../docs/06_LATEX_PIPELINE.md`.
