# a2a-swarm/configs

TOML configs loaded by each `rvagent a2a serve` child process spawned by
the parent demo. Each file pins the policy, budget, recursion, and
routing knobs for one node in the three-node A2A swarm.

## Files

- `node-cheap.toml` — low-cost peer (`max_cost_usd = 0.01`, slow
  `max_duration_ms = 10_000`, `allowed_skills = ["echo"]`); represents
  the bulk-compute tier. Binds at `:18001` (set by the orchestrator).
- `node-fast.toml` — fast-but-pricey peer; wins selection when latency
  budget is tight. Binds at `:18002`.
- `node-router.toml` — dispatcher node; carries a
  `[[routing.peers]]` list pointing at the cheap+fast leaves and uses
  `default_selector = "cheapest_under_latency"` to pick a peer per task.
  Binds at `:18003`.

These configs aren't run directly — they're paths passed by
`../src/main.rs` via the `NodeSpec.config` field, then consumed by
`rvagent-cli` config loading.

## Related

- `../src/main.rs` — names, bind addresses, and spawn order
- `../../../crates/rvAgent/rvagent-cli` — config schema and a2a server
