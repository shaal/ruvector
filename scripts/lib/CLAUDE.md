Reusable Node ES-module helpers imported by higher-level scripts in `../`.

Files:
- `module-splitter.mjs` - splits a Claude Code CLI bundle into logical modules. Used by RVF corpus extraction (`../claude-code-rvf-corpus.sh`) and decompile workflows.
- `rvf-builder.mjs` - creates binary RVF (RuVector Format) containers from extracted source modules. Consumed by `../publish-rvf.sh` and `../generate-rvf-manifest.py`.

Both are executables (`#!/usr/bin/env node`) and can also be invoked directly. RVF is the project's binary model/data container format - see crates under `../../crates/rvf/`.
