# prime-radiant/src/execution

4-lane execution ladder for gating actions on coherence: Lane 0 (Reflex), Lane 1 (Retrieval), Lane 2 (Heavy), Lane 3 (Human).

## Files

- `mod.rs` — module entry.
- `executor.rs` — top-level `Executor` that drives the ladder.
- `gate.rs` — per-action gate dispatch based on coherence energy thresholds.
- `ladder.rs` — the 4-lane ladder state machine and escalation rules.
- `action.rs` — action enum / trait for things being gated.

## Related

- Consumes signals from `coherence/engine.rs` and `governance/policy.rs`.
