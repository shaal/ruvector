# scipix/src/cli/commands

Implementations of individual scipix CLI subcommands.

## Files

- `mod.rs` - Subcommand enum.
- `ocr.rs` - `scipix ocr <image>` - one-shot OCR.
- `batch.rs` - Batch OCR over a directory.
- `serve.rs` - Start the HTTP API server.
- `config.rs` - Read/write config.
- `doctor.rs` (~30 KB) - Diagnostics: checks models, fonts, GPU, network, perms.
- `mcp.rs` (~28 KB) - Run scipix as an MCP server.

## How to run

```bash
cargo run -p ruvector-scipix --bin cli -- ocr path/to/img.png
cargo run -p ruvector-scipix --bin cli -- batch ./images
cargo run -p ruvector-scipix --bin cli -- doctor
cargo run -p ruvector-scipix --bin cli -- mcp
```

## Related

- Parent CLI: `../mod.rs`.
- API: `../../api/`.
