# scipix/src/bin

Binaries for scipix.

## Files

- `cli.rs` - CLI entrypoint (delegates to `../cli/`).
- `server.rs` - HTTP API server entrypoint.
- `benchmark.rs` (~27 KB) - Standalone benchmarking binary.

## How to run

```bash
cargo run -p ruvector-scipix --bin cli -- --help
cargo run -p ruvector-scipix --bin server
cargo run -p ruvector-scipix --bin benchmark
```

## Related

- API: `../api/`.
- CLI commands: `../cli/commands/`.
