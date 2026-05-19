# ruvllm/src/quality

Multi-dimensional quality scoring for LLM generations: schema compliance,
semantic coherence, diversity, temporal realism, uniqueness. Produces both
per-generation scores and longitudinal tracking with recommendations.

## Files
- `mod.rs` - public API + scoring-engine docs.
- `scoring_engine.rs` - `QualityScoringEngine` (score, track,
  recommendations).
- `coherence.rs` - semantic coherence scoring.
- `diversity.rs` - diversity / uniqueness scoring.
- `validators.rs` - schema compliance and constraint validators.
- `metrics.rs` - shared metric primitives.
