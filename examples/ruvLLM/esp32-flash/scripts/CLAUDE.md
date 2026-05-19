# ruvLLM / esp32-flash / scripts

Top-level helper scripts for `ruvllm-esp32-flash` (the development repo - not the npm distribution).

## Important files
- `offline-cache.sh` - prepares an offline cache of toolchain / firmware artifacts so flashing can happen without network access.
- `windows/` - PowerShell equivalents of the main flashing scripts.

## Related
- npm-packaged equivalents: `../npm/scripts/`. Top-level wrappers: `../cluster-flash.sh`, `cluster-flash.ps1`, `cluster-monitor.sh`, `install.sh`, `install.ps1`, `flash-windows.bat`.
