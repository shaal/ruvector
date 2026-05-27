# 06 — SPARC · Completion

Phase **C** of SPARC. Definition of done, rollout, documentation, telemetry, and
contribution path for landing CAJAS in ClickHouse.

---

## 1. Definition of done

- [ ] `query_plan_optimize_join_algorithm` setting added, default `0`.
- [ ] Selector, estimator, cost model, candidate filter, decision object implemented.
- [ ] Runtime guard implemented, reusing the existing on-the-fly conversion path.
- [ ] `EXPLAIN` exposes the decision (A4); golden-plan parity when off (A5).
- [ ] `system.query_log` / ProfileEvents emit chosen algorithm, fallback, switch events.
- [ ] Unit + integration + benchmark suites green across 4 hardware profiles.
- [ ] PRD §6 metrics met: −80% avoidable OOM, −20% median latency, 0 control regression.
- [ ] Docs updated (settings reference, "choosing a join algorithm" guide).
- [ ] Cost-model presets (`conservative`/`auto`/`aggressive`) calibrated and documented.

## 2. Rollout plan

| Stage | Audience | Setting | Promotion gate |
|-------|----------|---------|----------------|
| 0 Dark | internal CI/bench | `=0`, shadow EXPLAIN | suite green |
| 1 Opt-in | adventurous users | `=1` manual | no correctness issues |
| 2 Cloud canary | ClickHouse Cloud canary tier | `=1` by tier | telemetry within SLO |
| 3 Default candidate | new analyzer users | `=1` default proposed | regression gate (G4) on suite |
| 4 Default | all | `=1` default | release sign-off |

Mirrors how join reordering rolled out (experimental setting → default). Each stage is
reversible by flipping one setting.

## 3. Documentation deliverables

- Settings reference entries for all PRD §5.2 settings.
- Update the "Choosing the Right Join Algorithm" guide: explain CAJAS, when it overrides
  `auto`, and how to read the `EXPLAIN` annotation.
- A short "how CAJAS decides" page (cost model intuition + presets).
- Migration note: behavior unchanged unless the setting is enabled.

## 4. Telemetry & post-launch monitoring

- Track `JoinAlgorithmRuntimeSwitch` rate (target < 10% of CAJAS joins). A rising rate
  signals estimator/cost drift → recalibrate.
- Track OOM-failure rate on join queries before/after default flip.
- Collect anonymized (estimate, realized, chosen, oracle-best) tuples from opt-in
  clusters to seed a future learned cost model (PRD §9).

## 5. Contribution path (upstream)

> Note: in this environment, GitHub tooling is scoped to `shaal/ruvector`, so the
> ClickHouse PR itself is out of scope here. This package is the design others (or a
> future session with the right access) can implement and submit.

1. Open a ClickHouse RFC/issue referencing the join-reordering work and stating the
   algorithm-selection gap (this brief).
2. Land behind the default-off setting in a single reviewable PR series:
   estimator+cost → selector+EXPLAIN → runtime guard → telemetry.
3. Provide the benchmark harness and oracle results as the evidence package.
4. Iterate with maintainers on cost-model constants and the analyzer integration seam.

## 6. Exit criteria (initiative complete)

CAJAS is enabled by default for new-analyzer queries, the metrics gate holds across
releases for two minor versions, the runtime-switch rate is stable and low, and the
documentation + telemetry are in place. Future work (auto-statistics, learned cost
model, distributed join placement) is tracked separately per PRD §9.
