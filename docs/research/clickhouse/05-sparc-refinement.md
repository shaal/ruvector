# 05 — SPARC · Refinement

Phase **R** of SPARC. How CAJAS is built test-first, benchmarked, calibrated, and
iterated. Maps to acceptance criteria A1–A7 in the Specification.

---

## 1. TDD strategy (London-school, mock-first per project conventions)

Unit the decision logic in isolation by mocking the estimator, budget provider, and
statistics so tests are deterministic and fast.

| Test group | Verifies | Maps to |
|------------|----------|---------|
| `CandidateFilter` | Only valid algorithms per join type/strictness/RHS | I1, A3 |
| `BuildSideEstimator` | Correct rows/bytes/ndv from mocked stats; conservative path with no stats | I3 |
| `JoinCostModel` | Monotonic, sane rankings under varied sizes/budgets | A1, A2 |
| `JoinAlgorithmSelector` | End-to-end decision incl. fallback & reason | A1–A4 |
| `RuntimeJoinGuard` | Triggers on injected mis-estimate; never on accurate estimate | A6, NF3 |
| Golden-plan | CAJAS off ⇒ byte-identical plan/exec | A5 |

### Representative unit cases

```
GIVEN build fits budget with margin        EXPECT primary ∈ {hash, parallel_hash}   (A1)
GIVEN build = 5× budget, no sort order      EXPECT primary = grace_hash              (A2)
GIVEN build = 5× budget, pre-sorted inputs  EXPECT primary = full_sorting_merge
GIVEN RHS = dictionary keyed by join key    EXPECT primary = direct                  (A3)
GIVEN no statistics, large part count       EXPECT primary non-memory-bound          (I3)
GIVEN estimate=0 rows                        EXPECT primary=hash, fallback non-MB
GIVEN realized bytes = 3× estimate mid-build EXPECT runtime switch to fallback        (A6)
```

## 2. Benchmark suite

Fixed, versioned suite run on pinned hardware profiles (high-mem, low-mem, fast-disk,
slow-disk) so cost-model constants are calibrated, not guessed.

1. **TPC-H** (SF100, SF1000) — Q3, Q5, Q7, Q8, Q9, Q21 (join-heavy).
2. **TPC-DS** (SF100) — join-heavy subset.
3. **OOM-prone real-world joins** — curated from issue reports: large dimension joins,
   skewed-key joins, joins that currently force `partial_merge`.
4. **Hash-optimal control set** — small-build joins where `auto` is already optimal
   (guards G4 / A1, the no-regression gate).

Each query is run across memory budgets (`max_bytes_in_join` swept) and with CAJOS off
vs on, capturing: latency (median/p95), peak memory, OOM/success, chosen algorithm,
runtime-switch count, optimizer time.

## 3. Calibration loop

```
1. Run benchmark with current CostModelConfig.
2. For each query: compare CAJAS choice vs the empirically fastest feasible algorithm
   (brute-forced offline by running all valid algorithms).
3. Mis-selections → adjust constants (SPILL_WEIGHT, PM_PENALTY, BUDGET_FRACTION, hash
   overhead) and the safety factor.
4. Re-run; lock presets (conservative/auto/aggressive) once mis-selection < target.
```

The "fastest feasible algorithm per query" set becomes the **oracle** the cost model is
scored against — this is the core refinement signal.

## 4. Iteration milestones

| Milestone | Deliverable | Gate |
|-----------|-------------|------|
| R0 | Estimator + cost model + selector behind off-by-default flag | unit suite green |
| R1 | `EXPLAIN` annotation + golden-plan parity (off) | A4, A5 |
| R2 | Benchmark harness + oracle generation | suite runs reproducibly |
| R3 | Cost-model calibration to oracle | mis-selection < 10% |
| R4 | Runtime guard + on-the-fly switch reuse | A6, NF3 |
| R5 | Full metrics gate (PRD §6) | −80% OOM, −20% median, 0 regressions |

## 5. Performance guards

- **NF1 (opt time < 1ms/node):** micro-benchmark the selector; assert in CI on the
  control set. Cost evaluation is ≤ 6 candidates × O(1) — keep it allocation-light.
- **NF3 (runtime overhead < 1%):** the guard reads counters already maintained by the
  build phase; sample progress, don't add per-row work.

## 6. Risk-driven refinements

| Risk | Mitigation refined here |
|------|--------------------------|
| Cost model wrong on novel hardware | Presets + `aggressive`/`conservative` knobs; runtime guard catches build mis-estimates. |
| Statistics stale/missing | Conservative estimator path; guard backstops at runtime. |
| Optimizer time creep on many-join queries | Cap CAJAS to first N joins via reuse of `query_plan_optimize_join_order_limit` semantics; estimate-once for shared build sides. |
| Interaction with reordering | Sequential v1; log cases where a different order would have enabled a memory-bound algorithm, to scope co-optimization later. |

## 7. Exit of Refinement

All A1–A7 pass on the suite across all four hardware profiles; calibration locked into
named presets; runtime-switch rate < 10% of CAJAS joins; no control-set regression
beyond ±2%.
