# 04 — SPARC · Architecture

Phase **A** of SPARC. Where CAJAS lives in ClickHouse, its components, data flow, and
the integration seams with existing subsystems.

---

## 1. Placement in the query lifecycle

```
SQL ─▶ Parser ─▶ Analyzer (new) ─▶ Logical plan
                     │
                     ├─ cardinality estimation + column statistics
                     ▼
            Join REORDERING pass        (25.9 greedy / 25.12 DPsize)   [EXISTING]
                     ▼
   ┌──────────────────────────────────────────────┐
   │  CAJAS: Join ALGORITHM SELECTION pass   [NEW] │
   │  - per-node decision + fallback annotation     │
   └──────────────────────────────────────────────┘
                     ▼
              Physical plan ─▶ Pipeline builder
                                   │
                                   ├─ Join transforms (hash/grace/merge/direct) [EXISTING]
                                   ▼
                         Execution ◀── Runtime Guard hooks build-phase [NEW]
```

CAJAS runs **after** reordering (order fixes build-side sizes) and **before** pipeline
construction (the pipeline builder reads the annotation to instantiate the right join
transform). v1 keeps the passes sequential; co-optimization is a documented future
option (see Refinement).

## 2. Components

| Component | Responsibility | Touches |
|-----------|----------------|---------|
| `JoinAlgorithmSelector` | Orchestrates the planner pass; iterates join nodes. | Query plan |
| `BuildSideEstimator` | Produces `Estimate` from analyzer stats + schema. | Statistics, analyzer cardinality |
| `MemoryBudgetProvider` | Computes per-node budget from settings + live pressure. | Settings, MemoryTracker |
| `JoinCostModel` | Costs each valid candidate; returns the ranking. | calibration constants |
| `CandidateFilter` | Restricts algorithms by join type/strictness/RHS kind. | Join semantics |
| `JoinAlgorithmDecision` | Immutable value object annotating the node. | Query plan, EXPLAIN |
| `RuntimeJoinGuard` | Build-phase monitor; triggers fallback switch. | Join transform, MemoryTracker, query_log |
| `JoinSelectionExplainer` | Renders decision for `EXPLAIN`. | EXPLAIN |

## 3. Data structures

```
Estimate { rows: u64, bytes: u64, ndv: u64, from_statistics: bool }

JoinAlgorithmDecision {            // attached to physical join node
  primary, fallback: Algorithm
  estimated_build_rows, estimated_build_bytes, memory_budget_bytes: u64
  used_statistics: bool
  reason: String
}

CostModelConfig {                  // preset-driven, hardware-aware
  spill_weight, pm_penalty, budget_fraction, hash_overhead,
  parallel_setup_const, disk_bw_bytes_per_sec
}
```

## 4. Integration seams (reuse, don't rebuild)

1. **Statistics & cardinality** — consume the *same* analyzer estimation the reordering
   pass uses. CAJAS adds no new statistics collection.
2. **Join transforms** — the pipeline builder already maps a chosen algorithm to a
   concrete transform (`HashJoin`, `GraceHashJoin`, `MergeJoin`, `DirectJoin`, …). CAJAS
   only sets *which one*.
3. **On-the-fly conversion** — `auto` already supports a runtime hash→partial_merge
   switch. `RuntimeJoinGuard` generalizes the trigger (predictive, cost-chosen target)
   and reuses the conversion path.
4. **MemoryTracker** — budget and runtime pressure read from the existing per-query and
   server memory trackers.
5. **`system.query_log`** — add columns/profile-events for chosen algorithm, fallback,
   `used_statistics`, and whether a runtime switch fired (for offline tuning).

## 5. Control flow (sequence)

```
Planner            Selector          Estimator/CostModel        PlanNode
  │  select_join_algorithms()
  ├───────────────▶│ for each join node (topo order)
  │                ├── CandidateFilter.valid(...)
  │                ├── BuildSideEstimator.estimate() ─▶ Estimate
  │                ├── MemoryBudgetProvider.budget()  ─▶ budget
  │                ├── JoinCostModel.rank(candidates) ─▶ ranking
  │                ├── build Decision(primary, fallback, reason)
  │                └── node.set_algorithm_decision(Decision) ─▶│
  ▼

Pipeline builder reads Decision ─▶ instantiates transform + attaches RuntimeJoinGuard
```

## 6. Configuration surface

Settings from PRD §5.2. CAJAS is gated by `query_plan_optimize_join_algorithm` and only
active with the new analyzer. `join_algorithm_cost_model` selects a `CostModelConfig`
preset. Existing `join_algorithm` remains an explicit override: if a user pins a single
algorithm, CAJAS respects it (only chooses the runtime fallback when allowed).

## 7. Failure & safety design

- **Estimator failure / missing stats** → conservative path (Invariant I3); never picks
  memory-bound on unknown data.
- **CAJAS internal error** → log + fall back to legacy `auto` selection for that node;
  query never fails *because of* CAJAS.
- **Runtime guard false positive** → switching to a non-memory-bound algorithm is always
  correctness-safe; worst case is a minor latency cost, logged for calibration.

## 8. Observability

- `EXPLAIN actions=1`: per-join algorithm, fallback, est rows/bytes, `used_statistics`,
  reason.
- `system.query_log` / ProfileEvents: `JoinAlgorithmChosen`,
  `JoinAlgorithmRuntimeSwitch`, `JoinBuildBytesEstimateError`.
- A `system.join_algorithm_decisions` debugging view (opt-in) for benchmark analysis.

## 9. Module/file sketch (proposed, illustrative)

```
src/Processors/QueryPlan/Optimizations/
    selectJoinAlgorithm.cpp          # JoinAlgorithmSelector pass
    joinBuildSideEstimator.{h,cpp}
    joinCostModel.{h,cpp}
src/Interpreters/
    JoinAlgorithmDecision.h
    RuntimeJoinGuard.{h,cpp}
```

(Names indicative; final locations follow ClickHouse's planner-optimization layout.)
