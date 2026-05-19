# thermorust/tests

Correctness and invariant tests.

## Files

- `correctness.rs` — covers `anneal_discrete` / `anneal_continuous` / `step_discrete` / `inject_spikes` and the metrics (`binary_entropy` bounds, `magnetisation`, `overlap`). Uses `IsingMotif` with a seeded `StdRng`.
