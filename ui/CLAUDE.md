# ui/

Frontend / user-interface applications for the `ruvector` monorepo. Currently houses a single SvelteKit application, `ruvocal`, but this directory exists as the canonical location for any web UI / chat UI / dashboard front-ends that ship alongside the Rust vector-DB / agent toolkit core.

## Contents

- `ruvocal/` — SvelteKit-based chat UI (a fork of HuggingChat / `chat-ui`, re-branded as "ruvocal"). Supports tool calling via MCP, multimodal/vision endpoints, voice input, and an intelligent LLM router. See `ruvocal/CLAUDE.md` for details.

## Conventions

- Each sub-app owns its own `package.json`, build tooling, lint config, Dockerfile, and Helm chart — they are independently deployable.
- Apps may consume Rust crates from the workspace through WASM bindings (`ruvocal/src/lib/wasm` loads `rvagent_wasm.js` from `static/wasm/`).

## Related top-level dirs

- `crates/` — Rust crates that produce the WASM artifacts consumed here.
- `docs/`, `examples/` — repo-wide documentation and example workflows.
