# 02-quantum-superposition/examples

Three runnable demonstrations of CAFT applied to classic cognitive-bias
problems.

## Files

- `linda_problem.rs` — the conjunction-fallacy ("Linda the bank teller")
  experiment.
- `prisoners_dilemma.rs` — quantum-style decision dynamics in the
  prisoner's dilemma.
- `attention_collapse.rs` — attention-driven measurement collapse demo.

## Run

```bash
cargo run --release --example linda_problem
cargo run --release --example prisoners_dilemma
cargo run --release --example attention_collapse
```

## Related

- `../src/` — the CAFT primitives these examples drive
