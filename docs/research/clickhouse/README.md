# ClickHouse Improvement Initiative — Cost-Based Adaptive Join Algorithm Selection (CAJAS)

> Research + planning package. No upstream code is changed by these documents; they
> define a concrete, measurable improvement to ClickHouse's query planner and the
> SPARC + DDD path to deliver it.

## The one-sentence problem

ClickHouse picks its **join algorithm reactively** — `join_algorithm=auto` starts a
hash join and only spills to `partial_merge` *after* it hits a memory limit — even
though the analyzer now has the cardinality statistics needed to pick the right
algorithm **up front**.

## Why now

ClickHouse shipped join *reordering* recently (greedy in 25.9, DPsize in 25.12) using
cardinality estimates and `ADD STATISTICS`. That infrastructure (statistics +
cost-aware planner pass) is exactly what a cost-based *algorithm selector* needs, but
algorithm selection itself was left on the old reactive path. The data is on the
table; nobody is using it for this decision yet.

## What we propose

A planner component, **CAJAS**, that:

1. Estimates each join's build-side cardinality, row width, and key distribution from
   analyzer statistics (with safe fallbacks when statistics are absent).
2. Runs a transparent cost model over the candidate algorithms
   (`hash`, `parallel_hash`, `grace_hash`, `full_sorting_merge`, `partial_merge`,
   `direct`) under a per-query memory budget.
3. Annotates the query plan with the chosen algorithm **and** a fallback, surfaced in
   `EXPLAIN`.
4. Keeps a **runtime adaptive guard**: if the realized build-side cardinality diverges
   from the estimate beyond a threshold, switch to the pre-computed fallback before
   OOM rather than after.

## Document map

| # | Document | Purpose |
|---|----------|---------|
| 00 | [`00-research-brief.md`](./00-research-brief.md) | Current state, gap analysis, evidence, sources |
| 01 | [`01-prd.md`](./01-prd.md) | Product Requirements Document (goals, scope, metrics) |
| 02 | [`02-sparc-specification.md`](./02-sparc-specification.md) | SPARC **S** — requirements & acceptance criteria |
| 03 | [`03-sparc-pseudocode.md`](./03-sparc-pseudocode.md) | SPARC **P** — algorithms in pseudocode |
| 04 | [`04-sparc-architecture.md`](./04-sparc-architecture.md) | SPARC **A** — component & data design |
| 05 | [`05-sparc-refinement.md`](./05-sparc-refinement.md) | SPARC **R** — TDD, benchmarks, iteration plan |
| 06 | [`06-sparc-completion.md`](./06-sparc-completion.md) | SPARC **C** — rollout, docs, exit criteria |
| 07 | [`07-ddd-domain-model.md`](./07-ddd-domain-model.md) | DDD — bounded contexts, aggregates, ubiquitous language |

## Headline targets (see PRD for full table)

- Eliminate ≥80% of avoidable join-OOM query failures on the benchmark suite.
- ≥20% median latency reduction on memory-bound join queries vs `join_algorithm=auto`.
- Optimizer-time overhead < 1ms per join node (statistics already materialized).
- Zero regression on queries where the current heuristic is already optimal.

## Scope guardrail

This package targets **algorithm selection**, not join *reordering* (already shipped)
nor the join *execution* kernels themselves (already excellent). CAJAS sits between
them: it decides *which* kernel runs for *each already-ordered* join.
