# ruvllm/src/reflection

Self-reflection architecture for error recovery and self-correction. Wraps a
base agent so it can revise its own output using one of four strategies:
Retry, IfOrElse (confidence-gated), MultiPerspective, and learned-error
recovery.

## Files
- `mod.rs` - public API + strategy enum + docs.
- `reflective_agent.rs` - `ReflectiveAgent` wrapper that adds reflection
  to any base agent.
- `confidence.rs` - `ConfidenceChecker` implementing the If-or-Else (IoE)
  pattern.
- `perspectives.rs` - `Perspective` and multi-perspective critique
  (correctness, completeness, consistency).
- `error_recovery.rs` - `ErrorPatternLearner` (learns recovery strategies
  from historical errors).
