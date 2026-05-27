# 07 — Domain-Driven Design: CAJAS Domain Model

Strategic + tactical DDD for the join-algorithm-selection capability. Frames CAJAS as a
domain with bounded contexts, a ubiquitous language, aggregates, and context mappings to
existing ClickHouse subsystems.

---

## 1. Domain & subdomains

**Core domain:** *Join Algorithm Selection* — turning statistics + a memory budget into
the best join algorithm with a safe fallback. This is where the differentiated value is.

| Subdomain | Type | Why |
|-----------|------|-----|
| Join Algorithm Selection | **Core** | The new differentiating decision logic (CAJAS). |
| Cost Modeling & Calibration | **Core** | Quality of selection depends on it. |
| Cardinality & Statistics | **Supporting** | Consumed, not owned by CAJAS. |
| Join Execution (kernels) | **Generic** (to CAJAS) | Mature, reused as-is. |
| Memory Accounting | **Supporting** | Provides budget + live pressure. |
| Query Plan Optimization | **Supporting** | Hosts the pass; owns reordering. |

## 2. Ubiquitous language

| Term | Definition |
|------|------------|
| **Join Node** | A point in the physical plan where two inputs are joined. |
| **Build side** | Input materialized into the join's lookup structure. |
| **Probe side** | Input streamed against the build structure. |
| **Candidate** | A join algorithm valid for a given join's type/strictness/RHS. |
| **Decision** | The chosen primary algorithm + fallback + rationale for a join node. |
| **Estimate** | Predicted build-side rows/bytes/distinct-keys. |
| **Memory budget** | Bytes a memory-bound algorithm may use for this join. |
| **Memory-bound algorithm** | One that fails if its structure exceeds the budget (hash, parallel_hash). |
| **Spilling algorithm** | A non-memory-bound algorithm (grace_hash, merge variants). |
| **Runtime guard** | Execution-time monitor that switches to the fallback on mis-estimate. |
| **Oracle** | The empirically fastest feasible algorithm for a query (calibration target). |
| **Used-statistics flag** | Whether the decision was driven by real stats vs conservative defaults. |

This vocabulary is shared verbatim across PRD, SPARC docs, code identifiers, EXPLAIN
output, and `system.query_log` columns — one language, no translation.

## 3. Bounded contexts

```
┌────────────────────────────┐        ┌──────────────────────────────┐
│  Statistics Context         │        │  Memory Accounting Context     │
│  (analyzer cardinality,     │        │  (MemoryTracker, budgets,      │
│   column statistics)        │        │   live pressure)               │
└─────────────┬──────────────┘        └───────────────┬──────────────┘
              │ Estimate (ACL)                         │ Budget (ACL)
              ▼                                         ▼
        ┌──────────────────────────────────────────────────────┐
        │           Join Algorithm Selection Context  [CORE]     │
        │  Aggregates: JoinAlgorithmDecision (root),             │
        │              BuildSideEstimate, CostRanking            │
        │  Services:   JoinAlgorithmSelector, JoinCostModel,     │
        │              CandidateFilter                           │
        └───────────────┬───────────────────────┬───────────────┘
                        │ Decision               │ Decision + Guard policy
                        ▼                         ▼
        ┌────────────────────────┐    ┌────────────────────────────┐
        │ Query Plan Context      │    │  Join Execution Context     │
        │ (reordering, plan tree, │    │  (HashJoin, GraceHashJoin,  │
        │  EXPLAIN)               │    │   MergeJoin, DirectJoin;    │
        │  — hosts the pass       │    │   RuntimeJoinGuard hooks)   │
        └────────────────────────┘    └────────────────────────────┘
```

## 4. Context mappings

| Upstream → Downstream | Relationship | Notes |
|-----------------------|--------------|-------|
| Statistics → Selection | **Anti-Corruption Layer** | `BuildSideEstimator` translates raw stats into a clean `Estimate`; isolates CAJAS from statistics API churn and absence. |
| Memory Accounting → Selection | **ACL / Conformist** | `MemoryBudgetProvider` reads trackers; CAJAS conforms to existing memory semantics. |
| Selection → Query Plan | **Customer/Supplier** | CAJAS supplies a `Decision`; the plan context is the customer that hosts the pass and renders EXPLAIN. |
| Selection → Join Execution | **Open Host / Published Language** | The `JoinAlgorithmDecision` (primary, fallback, budget) is the published contract the pipeline builder and `RuntimeJoinGuard` consume. |
| Query Plan (reordering) → Selection | **Upstream pass** | Reordering fixes build sides before CAJAS runs; sequential in v1. |

## 5. Aggregates & invariants

### Aggregate root: `JoinAlgorithmDecision`
- **Holds:** primary, fallback, `BuildSideEstimate`, memory budget, `used_statistics`,
  reason.
- **Invariants enforced at construction:**
  - primary is valid for the join (I1);
  - if estimated bytes × safety > budget, primary is a spilling algorithm (I2);
  - fallback is always a spilling algorithm and ≠ primary (I3 support);
  - decision is immutable once attached to the node (I4 determinism).

### Entity: `BuildSideEstimate`
- Value-ish but identified by its join node; carries `from_statistics` provenance so
  downstream and telemetry know how much to trust it.

### Value object: `CostRanking`
- Ordered list of `(Algorithm, cost)`; pure function of `Estimate`, budget, config.

### Domain service: `JoinCostModel`
- Stateless; parameterized by `CostModelConfig` (the calibration value object).

### Policy: `RuntimeAdaptationPolicy`
- Encapsulates the switch rule (ratio / projected-overflow). Lives at the
  Selection↔Execution boundary; the guard enforces it.

## 6. Domain events

| Event | Emitted when | Consumer |
|-------|--------------|----------|
| `JoinAlgorithmSelected` | Decision attached to a node | EXPLAIN, query_log |
| `JoinBuildEstimateExceeded` | Realized build > estimate threshold | Runtime guard, telemetry |
| `JoinAlgorithmRuntimeSwitched` | Guard switches to fallback | query_log, calibration loop |

These feed the Cost Modeling & Calibration subdomain: the gap between
`JoinAlgorithmSelected` (predicted) and realized outcomes is the learning signal that
tunes presets (and, later, a learned model).

## 7. Why DDD here

The decision logic is genuinely a **bounded domain** with its own language and
invariants, distinct from both statistics (upstream) and join kernels (downstream).
Modeling it explicitly keeps CAJAS decoupled via ACLs (so statistics API changes or
missing stats don't corrupt the selector) and gives a single published contract
(`JoinAlgorithmDecision`) that planning, execution, EXPLAIN, and telemetry all share.
