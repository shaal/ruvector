# ruvbot / scripts

Lifecycle scripts for building, running, and installing `ruvbot`.

## Files
- `postinstall.js` - Runs on `npm install ruvbot`. Verifies optional
  native dependencies / downloads platform binaries as needed.
- `install.sh` - Shell helper for end-to-end install (Docker / native
  prerequisites).
- `build-rvf.js` - Packages `ruvbot.rvf` from compiled sources and the
  bundled `kernel/bzImage`.
- `run-rvf.js` - Boots the rvf image (supports `--inspect`); invoked
  by `npm run run:rvf` / `inspect:rvf`.
