# agentic-synth/examples

Runnable example programs showing how to use `@ruvector/agentic-synth` for various data generation scenarios. Each `.ts` file ships with its compiled `.js`, `.d.ts`, and source maps (built artifacts present in-tree).

## Top-level examples

- `basic-usage.ts` — minimal Quickstart.
- `benchmark-example.ts` — runs perf measurements.
- `dspy-complete-example.ts`, `dspy-training-example.ts`, `dspy-verify-setup.ts` — DSPy.ts integration end-to-end.
- `integration-examples.ts` — composing agentic-synth with peer packages (ruvector, robotics, midstreamer).
- `test-all-examples.ts` — runs every example in sequence as a smoke test.
- `user-schema.json` — sample schema used by structured generation examples.
- `EXAMPLES.md` — index/overview document.

## Domain subdirectories

`ad-roas/`, `agentic-jujutsu/`, `business-management/`, `cicd/`, `crypto/`, `docs/`, `employee-simulation/`, `security/`, `self-learning/`, `stocks/`, `swarms/` — each contains topic-specific scenarios (CRM, blockchain, threat sim, agent coordination, etc.).

Not included in the published npm tarball.
