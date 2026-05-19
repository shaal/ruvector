# ruvLLM / esp32-flash / npm

`ruvllm-esp32` npm package - ships a Node.js CLI (`ruvllm-esp32`) that downloads precompiled firmware and flashes ESP32 boards. Also packages a web flasher and per-OS PowerShell helpers.

## Important files
- `package.json` - npm manifest (`bin.ruvllm-esp32 -> bin/cli.js`, `postinstall` runs `bin/postinstall.js`; supports Linux/macOS/Windows on x64/arm64; declares many SEO keywords).
- `bin/` - the CLI entrypoint + postinstall script.
- `scripts/windows/` - PowerShell scripts for Windows users.
- `web-flasher/index.html` - browser-based ESP Web Tools flasher served alongside the CLI.

## Install / run
- `npm install -g ruvllm-esp32` then `ruvllm-esp32 --help`.
- For development: `npm link` from this directory.

## Related
- Native firmware source: `../src/`. Companion top-level scripts: `../scripts/`. Browser flasher (canonical): `../web-flasher/`.
