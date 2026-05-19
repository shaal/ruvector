# ruvbot / bin

Executable entry points published with the `ruvbot` npm package.

## Files
- `ruvbot.js` - Main CLI shebang script wired up by `package.json`'s
  `bin` field. Dispatches `start`, `init`, `doctor`, `config`,
  `memory`, `security`, `plugins`, `agent`, `status` subcommands.
- `cli.js` - Alternate / legacy CLI helper invoked by the main script
  (loads the compiled `dist/cli` implementation).

Run via `npx ruvbot <command>` after install.
