# ruvbot / src / cli

Commander-based command line implementation backing `bin/ruvbot.js`.

## Files
- `index.ts` - Sets up the top-level `commander` program and registers
  each subcommand from `commands/`.
- `commands/` - One file per subcommand (see subdir).
