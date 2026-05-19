# prime-radiant/src/ruvllm_integration

Bridge into the RuvLLM inference stack: lets the coherence engine guard, validate, and witness LLM outputs in real time.

## Files

- `mod.rs` — module entry.
- `config.rs` — integration config (gate mode, witness log path, memory backend).
- `adapter.rs` — `RuvllmAdapter`: connects engine to inference runtime.
- `bridge.rs` — message-level bridge into ruvllm hooks.
- `traits.rs` — integration traits implemented by ruvllm side.
- `gate.rs` — gate decision specifically for LLM tokens / completions.
- `confidence.rs` — confidence scoring derived from coherence energy.
- `coherence_validator.rs` — validator that wraps `CoherenceEngine` for ruvllm.
- `memory_layer.rs` — coherence-aware memory layer for context retrieval.
- `pattern_bridge.rs` — bridge into pattern / token streams.
- `witness.rs` + `witness_log.rs` — LLM-specific witness records + persistent log.
- `error.rs` — module errors.

## Related

- `crates/ruvllm` (workspace) — the inference runtime being integrated.
