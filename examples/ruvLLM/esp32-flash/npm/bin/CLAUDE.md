# ruvLLM / esp32-flash / npm / bin

Node.js entry points for the `ruvllm-esp32` npm package.

## Important files
- `cli.js` - the `ruvllm-esp32` binary registered in `../package.json`. Exposes flashing / monitoring subcommands and routes to the per-OS scripts under `../scripts/`.
- `postinstall.js` - npm `postinstall` hook; downloads the appropriate prebuilt firmware artifacts on install (driven by the `os` / `cpu` matrix declared in `../package.json`).

## Related
- Manifest: `../package.json`. PowerShell helpers: `../scripts/windows/`. Web flasher: `../web-flasher/`.
