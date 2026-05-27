# 01 — PRD: Cost-Based Adaptive Join Algorithm Selection (CAJAS)

**Product:** ClickHouse query planner
**Feature:** CAJAS — predictive, statistics-driven join algorithm selection with a
runtime adaptive fallback
**Owner:** Query planner team (proposed)
**Status:** Draft for review
**Related:** Join reordering (25.9 greedy, 25.12 DPsize), column statistics

---

## 1. Problem statement

ClickHouse selects its join algorithm reactively: `join_algorithm=auto` runs a hash
join and switches to `partial_merge` only after a memory-limit violation. The analyzer
already computes cardinality estimates (used for join reordering) and supports column
statistics, but that signal is **not** used to choose the join algorithm. The result
is avoidable OOM failures and suboptimal fallbacks (slow `partial_merge` chosen where
`grace_hash` would be faster, or `direct`/`full_sorting_merge` opportunities missed).

## 2. Goals

| ID | Goal |
|----|------|
| G1 | Choose the join algorithm **before execution** from a transparent cost model. |
| G2 | Reduce join-related OOM failures on memory-bound workloads. |
| G3 | Reduce latency on memory-bound joins vs `auto` by picking better spill algorithms. |
| G4 | Never regress queries where `auto`/hash is already optimal. |
| G5 | Make the decision **explainable** (`EXPLAIN` shows chosen algorithm + reason). |
| G6 | Degrade gracefully (and safely) when statistics are missing. |

## 3. Non-goals

- Not changing join **reordering** (already shipped).
- Not writing new join **execution kernels** (existing ones are reused).
- Not building automatic statistics collection (separate roadmap item; CAJAS only
  *consumes* whatever statistics exist).
- Not addressing distributed/`GLOBAL JOIN` shard-placement decisions in v1 (future).

## 4. Users & use cases

- **Analytics engineers** running multi-table joins on large datasets who currently hit
  OOM and must hand-tune `join_algorithm` / `max_bytes_in_join`.
- **Platform/SRE owners** of shared clusters who want fewer OOM-induced query kills and
  more predictable memory behavior.
- **ClickHouse Cloud** operators wanting better default performance without per-query
  tuning.

## 5. Requirements

### 5.1 Functional

- **F1** A planner pass annotates each physical join node with a chosen algorithm.
- **F2** The cost model ranks candidates: `hash`, `parallel_hash`, `grace_hash`,
  `full_sorting_merge`, `partial_merge`, `direct`.
- **F3** `direct` is chosen when the right side is a key-value joinable source and the
  join key matches its key.
- **F4** Selection respects a per-query **memory budget** derived from
  `max_bytes_in_join` / `max_memory_usage` and current server memory pressure.
- **F5** Each annotation carries a **fallback algorithm** for runtime use.
- **F6** A **runtime guard** monitors realized build-side size during the build phase;
  if it exceeds the estimate beyond a configurable ratio, it switches to the fallback
  (a non-memory-bound algorithm) before the hard limit is hit.
- **F7** `EXPLAIN PLAN` / `EXPLAIN actions=1` shows the chosen algorithm, the fallback,
  the estimated build-side rows/bytes, and whether statistics were used.

### 5.2 Configuration

| Setting | Default | Meaning |
|---------|---------|---------|
| `query_plan_optimize_join_algorithm` | `0` (off) | Master switch for CAJAS. |
| `join_algorithm_cost_model` | `'auto'` | `auto` \| `conservative` \| `aggressive`. |
| `join_algorithm_runtime_adapt` | `1` | Enable the runtime fallback guard. |
| `join_algorithm_estimate_safety_factor` | `1.3` | Multiplier on estimated build size. |
| `join_algorithm_runtime_switch_ratio` | `2.0` | Realized/estimated ratio that triggers fallback. |

CAJAS only activates when the new analyzer is enabled. When off, behavior is byte-for-
byte identical to today.

### 5.3 Non-functional

- **NF1** Optimizer-time overhead **< 1ms per join node** (statistics pre-materialized).
- **NF2** No correctness change: same result set/row order semantics as the equivalent
  hand-selected algorithm.
- **NF3** Runtime guard overhead **< 1%** of build-phase time.
- **NF4** Fully behind a default-off setting; safe to ship dark.

## 6. Success metrics

| Metric | Baseline (`auto`) | Target |
|--------|-------------------|--------|
| Avoidable join-OOM failures (benchmark suite) | reference | **−80%** |
| Median latency, memory-bound joins | reference | **−20%** |
| p95 latency, memory-bound joins | reference | **−15%** |
| Regressions on hash-optimal queries | n/a | **0** (within ±2% noise) |
| Optimizer time per join node | n/a | **< 1ms** |
| Mis-selection corrected by runtime guard | n/a | tracked; **< 10%** of CAJAS joins |

Measured on a fixed benchmark suite (see SPARC Refinement §benchmarks): TPC-H/TPC-DS
join-heavy queries plus a curated set of known OOM-prone real-world joins, at multiple
memory-budget settings.

## 7. Rollout

1. Land behind `query_plan_optimize_join_algorithm=0`.
2. Internal benchmark + shadow `EXPLAIN` comparison (no execution change).
3. Opt-in for ClickHouse Cloud canary tier.
4. Flip default to `1` only after the regression gate (G4) passes on the suite.

## 8. Open questions (resolved in SPARC)

- Co-optimize order + algorithm, or run algorithm selection strictly after reordering?
- How to calibrate disk-spill cost across hardware profiles?
- Telemetry: what to log to `system.query_log` for offline model tuning?

## 9. Out-of-scope / future

- Auto-statistics collection feeding CAJAS.
- Learned cost model trained on `query_log` feedback.
- Distributed join placement (`GLOBAL JOIN`) cost-based decisions.
