# 03 — SPARC · Pseudocode

Phase **P** of SPARC. Language-agnostic algorithms for CAJAS. Implementation will be
C++ inside the ClickHouse planner; this captures the logic and edge handling.

---

## 1. Top-level planner pass

```
function select_join_algorithms(plan, settings, server_state):
    if not settings.query_plan_optimize_join_algorithm:
        return plan                      # CAJAS off → identical behavior

    for join_node in plan.physical_joins_in_topological_order():
        decision = decide_algorithm(join_node, settings, server_state)
        join_node.set_algorithm_decision(decision)
    return plan
```

Topological order matters: a downstream join's build-side estimate depends on upstream
joins' output cardinality, which we propagate as we go.

## 2. Per-node decision

```
function decide_algorithm(node, settings, server):
    candidates = valid_candidates_for(node.type, node.strictness)   # Invariant I1

    # Fast path: direct join
    if "direct" in candidates and rhs_is_keyed_source(node.right, node.keys):
        return Decision(primary="direct",
                        fallback=best_non_memory_bound(candidates, est),
                        reason="RHS is keyed source; O(1) lookups")

    est = estimate_build_side(node)            # §3
    budget = memory_budget(node, settings, server)   # §4
    needed = est.bytes * settings.estimate_safety_factor

    # Cost each candidate
    scored = []
    for algo in candidates:
        if is_memory_bound(algo) and needed > budget:
            continue                           # Invariant I2: skip infeasible
        scored.append( (algo, cost(algo, est, budget, server)) )   # §5

    if scored is empty:                        # nothing fits in memory
        primary = cheapest_non_memory_bound(candidates, est, server)
    else:
        primary = argmin_cost(scored)

    fallback = best_non_memory_bound(candidates, est, server)
    if primary == fallback: fallback = second_best_non_memory_bound(...)

    return Decision(primary, fallback, est.rows, est.bytes, budget,
                    used_statistics = est.from_statistics,
                    reason = explain(primary, est, budget))
```

## 3. Build-side estimation (with safe fallbacks — Invariant I3)

```
function estimate_build_side(node):
    if has_statistics(node.right):
        rows  = analyzer.estimate_rows(node.right)          # post-filter cardinality
        width = avg_row_width(node.right, node.carried_columns)  # schema + tdigest
        ndv   = uniq_estimate(node.right, node.keys)         # HLL distinct keys
        bytes = hash_table_overhead(rows, ndv, width)        # buckets + payload
        return Estimate(rows, bytes, ndv, from_statistics=true)
    else:
        # No stats: derive a cheap UPPER bound, bias toward "won't fit"
        rows = upper_bound_rows(node.right)   # part-count × index granule estimate
        width = schema_max_width(node.carried_columns)
        return Estimate(rows, rows*width*HASH_OVERHEAD, ndv=rows,
                        from_statistics=false)
```

`hash_table_overhead` accounts for ClickHouse's open-addressing hash maps: bucket array
sized to next-power-of-two ≥ rows/load_factor, plus per-row key+payload bytes, plus
arena overhead. NDV feeds skew detection (highly skewed keys hurt parallel_hash).

## 4. Memory budget

```
function memory_budget(node, settings, server):
    hard = min(settings.max_bytes_in_join or INF,
               settings.max_memory_usage or INF)
    # Leave headroom for concurrent queries and probe-side buffers
    pressure_factor = clamp(1 - server.memory_utilization, 0.1, 1.0)
    return hard * BUDGET_FRACTION * pressure_factor
```

## 5. Cost model

Cost is **estimated wall-clock**, normalized; lower is better.

```
function cost(algo, est, budget, server):
    r = est.rows ; w = est.bytes
    switch algo:
      case "parallel_hash":
          return build_cost(r,w)/server.threads + probe_cost(est)
                 + skew_penalty(est) + parallel_setup_const
      case "hash":
          return build_cost(r,w) + probe_cost(est)
      case "grace_hash":
          buckets = ceil(w / budget)
          return build_cost(r,w) + probe_cost(est)
                 + spill_cost(w, buckets, server.disk_bw) * SPILL_WEIGHT
      case "full_sorting_merge":
          return sort_cost(r) + merge_cost(est)
                 - already_sorted_discount(node)        # cheap if pre-sorted
      case "partial_merge":
          return sort_cost(r)*PM_PENALTY + merge_cost(est)*PM_PENALTY
      case "direct":
          return lookup_cost(probe_rows(est))           # dominates → tiny
```

Constants (`SPILL_WEIGHT`, `PM_PENALTY`, `BUDGET_FRACTION`, hash overhead) are the
**calibration surface** tuned in the Refinement phase per hardware profile and exposed
via `join_algorithm_cost_model` presets (`conservative`/`auto`/`aggressive`).

## 6. Runtime adaptive guard

Executed during the build phase, not planning time.

```
function on_build_progress(node, rows_seen, bytes_seen, decision, settings):
    if not settings.join_algorithm_runtime_adapt: return CONTINUE
    if not is_memory_bound(decision.primary):     return CONTINUE   # nothing to guard

    ratio = bytes_seen / max(decision.estimated_build_bytes, 1)
    projected = bytes_seen / max(build_progress_fraction(), epsilon)

    if ratio > settings.runtime_switch_ratio
       or projected > decision.memory_budget_bytes:
        log_switch(node, decision.primary, decision.fallback,
                   rows_seen, bytes_seen)         # → system.query_log
        return SWITCH_TO(decision.fallback)       # spill what's built, restart as fallback
    return CONTINUE
```

The switch reuses ClickHouse's existing on-the-fly conversion machinery (the same
mechanism `auto` uses to go hash→partial_merge), but it (a) triggers *predictively*
before the hard limit and (b) targets the **cost-chosen** fallback (often `grace_hash`)
instead of always `partial_merge`.

## 7. Edge cases

- **Cross/comma joins, dictGet rewrites** — outside candidate set; pass through.
- **ASOF / non-equi** — restrict to algorithms that support them; if only memory-bound
  options exist and build won't fit, surface a clear error rather than silent OOM.
- **Multiple joins sharing a build side** — estimate once, reuse.
- **Estimate = 0 rows** — treat as `hash` (trivially fits) but keep a non-memory-bound
  fallback in case the estimate was a false zero.
