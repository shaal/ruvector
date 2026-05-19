# ruvix-shell/src

## Files

- `lib.rs` — crate root; defines the `ShellBackend` trait and `Shell` driver loop.
- `parser.rs` — line-based command parser suitable for serial consoles.
- `commands/` — one module per built-in command (see `commands/CLAUDE.md`).
