# agentic-jujutsu

TypeScript reference examples for the `agentic-jujutsu` ecosystem (a quantum-resistant, AI-aware version control wrapper around the Jujutsu VCS). These are standalone illustrative scripts showing intended API shapes; they do not import a real package and act as documentation/specs.

## Files

- `basic-usage.ts` - Repository init, commits, branches, diffs via a `JjWrapper` interface.
- `learning-workflow.ts` - ReasoningBank-style learning loop integrated with jj commits.
- `multi-agent-coordination.ts` - Multi-agent swarm coordination using jj branches as agent workspaces.
- `quantum-security.ts` - Post-quantum signing/encryption hooks on top of jj operations.

## How to run

These are reference/spec files. To execute them as runnable demos, supply or stub `agentic-jujutsu`:

```bash
# Example with ts-node (after providing a real implementation)
npx ts-node basic-usage.ts
```

Or just read them as design documentation.

## Tech stack

- TypeScript (no `package.json` here; intended to be consumed alongside the `agentic-jujutsu` SDK).
- Jujutsu (jj) VCS as the underlying engine.

## Related

- See the `agentic-jujutsu` skill for orchestration patterns.
- Sibling `agentic-flow` / `claude-flow` based examples in the monorepo (e.g. `examples/exo-ai-2025`, `examples/a2a-swarm`).
