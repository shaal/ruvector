# ruvLLM / esp32-flash / web-flasher

Canonical (development) copy of the browser-based ESP Web Tools flasher. The same file is mirrored into `../npm/web-flasher/` when the npm package is built.

## Important files
- `index.html` - self-contained HTML page that uses Web Serial + ESP Web Tools to flash the prebuilt `ruvllm-esp32` firmware.

## Related
- npm-distributed copy: `../npm/web-flasher/`. CLI that serves it: `../npm/bin/cli.js`.
