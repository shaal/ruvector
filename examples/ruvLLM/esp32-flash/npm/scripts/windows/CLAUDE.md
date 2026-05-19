# ruvLLM / esp32-flash / npm / scripts / windows

PowerShell helpers used by `../bin/cli.js` on Windows.

## Important files
- `setup.ps1` - one-shot setup: installs ESP-IDF prerequisites, downloads firmware.
- `env.ps1` - sources ESP-IDF environment variables in the current PowerShell session.
- `build.ps1` - builds the firmware from source (requires the ESP-IDF toolchain).
- `flash.ps1` - flashes the prebuilt firmware to the connected board.
- `monitor.ps1` - opens the ESP-IDF serial monitor.

## Related
- Top-level (non-npm) equivalents: `../../../scripts/windows/`.
- CLI: `../../bin/cli.js`.
