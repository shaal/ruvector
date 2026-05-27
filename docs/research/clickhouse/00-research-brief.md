# 00 — Research Brief: ClickHouse Join Algorithm Selection

**Date:** 2026-05-27
**Author:** Research + planning pass (Claude Code)
**Status:** Findings to drive PRD/SPARC/DDD in this directory

---

## 1. Context

ClickHouse is a column-oriented OLAP DBMS with a world-class vectorized execution
engine. Historically its **planner/optimizer** lagged the engine: it leaned on the
quality of physical operators and on user-supplied hints rather than a cost-based
optimizer (CBO). That has been changing fast with the "new analyzer"
(`allow_experimental_analyzer`, now the default in recent releases) and a wave of
cost-based planner work.

## 2. What ClickHouse already has (do not rebuild)

### 2.1 Join *reordering* — recently shipped
- **25.9:** Global join reordering chooses build-vs-probe order across >2 tables. The
  search space grows exponentially, so a **greedy** algorithm converges to a
  "good enough" order. Driven by **cardinality estimates** of join columns (factoring
  in WHERE filters). Controlled by `query_plan_optimize_join_order_limit` and
  `allow_statistics_optimize`.
- **25.12:** Adds **DPsize** (dynamic-programming, size-based) as a more expressive but
  more optimizer-time-expensive algorithm for INNER JOIN reordering. Controlled by
  `query_plan_optimize_join_order_algorithm='dpsize,greedy'` (DPsize first, greedy
  fallback). This mirrors classic CBO practice: DP for small joins, greedy beyond a
  table-count threshold.

### 2.2 Column statistics — the cost-model substrate
- `ALTER TABLE ... ADD STATISTICS (col) TYPE <type>` then `MATERIALIZE STATISTICS`.
- Types include **tdigest** (quantiles/range selectivity), **uniq** (HyperLogLog
  distinct-count), and **countmin** (frequency). Approximate aggregate functions
  `uniqHLL12` (~1.6% error), `quantileTDigest` exist independently.
- As of 25.9, statistics are **created manually**; auto-creation for new tables is on
  the roadmap. So any consumer must tolerate **missing statistics**.

### 2.3 Join *execution* algorithms — excellent kernels
- `hash` and `parallel_hash`: fast, memory-bound; OOM if the right-hand side does not
  fit. `parallel_hash` builds several hash tables concurrently (faster, more memory).
- `grace_hash`: non-memory-bound, spills buckets to disk, no sorting required; flexible
  memory-vs-speed control via bucket count.
- `full_sorting_merge` / `partial_merge`: merge joins; `partial_merge` minimizes memory
  at a large speed cost.
- `direct`: O(1)-ish lookups when the right side is a key-value dictionary/joinable
  table (e.g., `Dictionary`, `Join` engine).

### 2.4 The current selection mechanism — the gap
- `join_algorithm=auto`: ClickHouse **starts with hash**, and only switches to
  `partial_merge` **on the fly when the memory limit is violated**. This is *reactive*:
  the decision is made after work has been wasted and after memory pressure is real.
- The user can list multiple algorithms in `join_algorithm` to define a preference
  fallback chain, but the *choice among them* is not driven by the same cost model and
  cardinality statistics that now drive reordering.

## 3. The gap, precisely stated

> ClickHouse now estimates cardinalities to **order** joins, and it has the statistics
> infrastructure to do so, but it still selects the join **algorithm** reactively
> (start hash, spill on OOM) rather than predictively. The cost signal exists; the
> selection decision does not consume it.

Consequences observed/reported by users:
- Memory-bound joins fail with OOM that a cost model could have routed to `grace_hash`
  or `full_sorting_merge` proactively.
- When hash *would* fit, `auto` is fine — but when it would not, the reactive switch to
  `partial_merge` (slow) is often a worse choice than `grace_hash` (faster, also
  non-memory-bound). The reactive path doesn't reason about *which* non-memory-bound
  algorithm is best.
- `direct` join opportunities (dictionary/Join-engine right sides) are missed unless the
  user sets it explicitly.

## 4. Improvement thesis — CAJAS

Add a **cost-based adaptive join algorithm selector** that:

1. Consumes the same analyzer cardinality/statistics already used by reordering.
2. Picks an algorithm per join node under an explicit per-query **memory budget**.
3. Emits a **primary + fallback** annotation visible in `EXPLAIN`.
4. Guards execution: a lightweight **runtime probe** of realized build-side size
   triggers the pre-computed fallback *before* OOM, not after.

This is incremental, composes with the shipped reordering work, and is measurable.

## 5. Why this is tractable

- The hard parts (statistics, cardinality estimation, a cost-aware planner pass, the
  algorithm kernels) **already exist**. CAJAS is mostly a **decision function + plan
  annotation + a runtime guard**, not new execution machinery.
- It can ship behind a setting (`query_plan_optimize_join_algorithm`) defaulting off,
  exactly as reordering did, allowing safe A/B rollout.

## 6. Risks / unknowns to resolve in SPARC

- **Estimate quality without statistics.** Need conservative fallbacks so CAJAS never
  does worse than `auto` when statistics are absent.
- **Cost-model calibration** across hardware (memory bandwidth, disk speed for spills).
- **Optimizer-time budget.** Selection must be O(candidates) per join node and cheap.
- **Interaction with reordering.** Order and algorithm are coupled (a different order
  changes build-side size). Define the pass ordering and whether to co-optimize.

## 7. Sources

- [Guide for query optimization | ClickHouse Docs](https://clickhouse.com/docs/optimize/query-optimization)
- [ClickHouse Release 25.9](https://clickhouse.com/blog/clickhouse-release-25-09)
- [ClickHouse Release 25.12](https://clickhouse.com/blog/clickhouse-release-25-12)
- [Choosing the Right Join Algorithm (Joins Under the Hood, Part 5)](https://clickhouse.com/blog/clickhouse-fully-supports-joins-how-to-choose-the-right-algorithm-part5)
- [Hash Join, Parallel Hash, Grace Hash (Part 2)](https://clickhouse.com/blog/clickhouse-fully-supports-joins-hash-joins-part2)
- [Full Sorting Merge / Partial Merge Join (Part 3)](https://clickhouse.com/blog/clickhouse-fully-supports-joins-full-sort-partial-merge-part3)
- [Direct Join (Part 4)](https://clickhouse.com/blog/clickhouse-fully-supports-joins-direct-join-part4)
- [Manipulating Column Statistics | ClickHouse Docs](https://clickhouse.com/docs/sql-reference/statements/alter/statistics)
- [Best practices: minimize and optimize JOINs](https://github.com/ClickHouse/clickhouse-docs/blob/main/docs/best-practices/minimize_optimize_joins.md)
- [Are ClickHouse JOINs Slow? A 2026 PR-by-PR Analysis (DEV)](https://dev.to/manveer_chawla_64a7283d5a/are-clickhouse-joins-slow-a-2026-pr-by-pr-analysis-21e8)
- [ClickHouse JOINs: Key Limitations (GlassFlow)](https://www.glassflow.dev/blog/clickhouse-limitations-joins)
