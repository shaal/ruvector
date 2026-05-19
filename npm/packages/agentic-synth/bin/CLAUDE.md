# agentic-synth/bin

CLI entry point for the `@ruvector/agentic-synth` package.

## Files

- `cli.js` — Commander-based CLI binary registered as `agentic-synth` in package.json. Imports the built `AgenticSynth` class from `../dist/index.js` and exposes commands for generating time-series, events, and structured data. Supports loading JSON configs and writing output to files.

This file is shipped (referenced in `package.json -> files` and `bin`) and executed via `npx agentic-synth ...`.
