Heterogeneous collection of build, deploy, training, benchmarking, validation, and operational scripts for the ruvector monorepo. Mixture of bash, Node.js (`.mjs`), and Python.

Subdirectories (each with its own CLAUDE.md):
- `benchmark/` - benchmark runner shell scripts (`run_benchmarks.sh`, `run_llm_benchmarks.sh`).
- `build/` - cross-platform NAPI/binary builds.
- `ci/` - lockfile sync helpers and githook installer.
- `deploy/` - main `deploy.sh` for crates.io + npm publishing, plus docs.
- `lib/` - reusable Node modules used by other scripts (`module-splitter.mjs`, `rvf-builder.mjs`).
- `patches/` - duplicate of top-level `patches/` (likely staging for patch generation).
- `publish/` - per-package publish scripts (crates, CLI, router-wasm).
- `test/` - smoke/integration test runners (`test-wasm.mjs`, docker package tests, graph CLI tests).
- `training/` - LoRA SFT pipeline, quantization calibration, deobfuscator training, dataset builders.
- `validate/` - pre-publish package validation, paper-implementation verification, HNSW build verification.

Notable top-level scripts:
- `analyze-evolution.js`, `analyze-ham10000.js` - dataset / experiment analysis.
- `claude-code-decompile.sh`, `claude-code-rvf-corpus.sh` - Claude Code bundle decompile + RVF corpus extraction.
- `create-brainpedia.py`, `seed-brain*.py`, `seed-specialized.py` - brain memory store seeding.
- `gemini-agents.js`, `deploy-gemini-agents.sh` - Gemini agent orchestration.
- `wet-*.{sh,js,yaml}`, `historical-crawl-import.sh` - Common Crawl WET file processing pipeline.
- `swarm_train_15.sh`, `discover_and_train.sh`, `training_orchestrator.sh`, `train_brain.sh`, `train-lora.py` - orchestration entry points for training swarms.
- `rebuild-all-versions.mjs` - rebuild every published package version.
- `deploy_brain_services.sh`, `deploy_trainer.sh`, `deploy-dragnes.sh`, `deploy-crawl-phase1.sh`, `deploy-wet-job.sh` - service deployments.
- `check_brain_status.sh`, `generate-rvf-manifest.py`, `publish-rvf.sh`, `build-solver.sh`, `run_mincut_bench.sh`, `sync-lockfile.sh`, `upvote_memories.py`, `vote-boost.py`, `sql-audit-v3.sql`, `wet-job.yaml` - misc operational tooling.

These are research / ops scripts - many are one-off or environment-specific. Check the script header before running.
