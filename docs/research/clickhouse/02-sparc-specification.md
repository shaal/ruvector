# 02 — SPARC · Specification

Phase **S** of SPARC for CAJAS. Defines precise requirements, inputs/outputs,
invariants, and acceptance criteria. No implementation here.

---

## 1. Mission

Given an already-ordered physical join node and the analyzer's statistics, deterministi-
cally select the join algorithm that minimizes expected execution time subject to a
memory budget, with a safe fallback, and adapt at runtime if the estimate was wrong.

## 2. Inputs

| Input | Source | Notes |
|-------|--------|-------|
| Join node (keys, type, strictness) | Query plan after reordering | INNER/LEFT/RIGHT/FULL, ANY/ALL/ASOF |
| Build-side est. rows | Analyzer cardinality estimation | post-filter, post-reorder |
| Build-side est. avg row width | Column statistics + schema | bytes/row for join key + carried columns |
| Build-side distinct keys | `uniq` (HLL) statistic | for hash-table sizing & skew |
| Key range / selectivity | `tdigest` statistic | for merge-join viability |
| Right-side source kind | Storage metadata | dictionary / Join engine / table |
| Memory budget | `max_bytes_in_join`, `max_memory_usage`, live pressure | per-query, per-node |
| Settings | §PRD 5.2 | safety factors, mode |

When a statistic is missing, CAJAS uses **conservative defaults** (treat build side as
large / non-fitting) so it never picks a memory-bound algorithm on unknown data.

## 3. Outputs

A `JoinAlgorithmDecision` attached to the join node:

```
JoinAlgorithmDecision {
  primary:        Algorithm        // chosen kernel
  fallback:       Algorithm        // non-memory-bound, used by runtime guard
  estimated_build_rows:  u64
  estimated_build_bytes: u64
  memory_budget_bytes:   u64
  used_statistics: bool            // true if real stats drove the choice
  reason:         String           // human-readable, surfaced in EXPLAIN
}
```

## 4. Candidate algorithms & applicability

| Algorithm | Memory-bound | Requires | Best when |
|-----------|:---:|----------|-----------|
| `direct` | no | RHS is dictionary/Join-engine keyed by join key | always preferred when applicable |
| `parallel_hash` | yes | build fits with parallel overhead | build ≪ budget, want max speed |
| `hash` | yes | build fits | build < budget, moderate size |
| `grace_hash` | no | equi-join | build > budget, no useful sort order |
| `full_sorting_merge` | no | equi-join, sortable keys | inputs already/cheaply sorted, large |
| `partial_merge` | no | equi-join | last resort; minimal memory, slow |

Non-equi/ASOF and some strictness modes restrict the candidate set; CAJAS must filter to
**only algorithms valid for that join's type/strictness** before costing.

## 5. Invariants

- **I1 (correctness):** The chosen algorithm must be semantically valid for the join's
  type and strictness. CAJAS never widens the candidate set beyond what the engine
  supports for that join.
- **I2 (safety):** If `estimated_build_bytes × safety_factor > memory_budget`, `primary`
  MUST be a non-memory-bound algorithm.
- **I3 (no-stats safety):** With no statistics, CAJAS behaves no worse than `auto`:
  primary is `hash` only if a cheap upper-bound row count fits the budget; otherwise a
  non-memory-bound algorithm. `fallback` is always non-memory-bound.
- **I4 (determinism):** Same inputs + settings ⇒ same decision.
- **I5 (budget):** Optimizer cost evaluation is O(valid candidates) ≤ 6 per join node.

## 6. Acceptance criteria

- **A1** For a join whose build side provably fits the budget, CAJAS selects `hash` or
  `parallel_hash` and matches `auto`'s latency within ±2%.
- **A2** For a join whose build side provably exceeds the budget, CAJAS selects a
  non-memory-bound algorithm and the query **completes** where `auto` would OOM.
- **A3** When RHS is a dictionary keyed by the join key, CAJAS selects `direct`.
- **A4** `EXPLAIN` exposes `primary`, `fallback`, estimated rows/bytes, `used_statistics`,
  and `reason`.
- **A5** With `query_plan_optimize_join_algorithm=0`, plans and execution are identical
  to the pre-CAJAS build (golden-plan test).
- **A6** Runtime guard: in an injected mis-estimate scenario (realized ≫ estimate),
  execution switches to `fallback` and completes without OOM; `system.query_log`
  records the switch.
- **A7** Optimizer-time per join node < 1ms (measured, NF1).

## 7. Glossary (see DDD doc for the full ubiquitous language)

- **Build side / probe side** — the join input materialized into a hash/sort structure
  vs the streamed side.
- **Memory-bound algorithm** — fails if its in-memory structure exceeds the budget.
- **Runtime guard** — execution-time monitor that triggers the fallback.
