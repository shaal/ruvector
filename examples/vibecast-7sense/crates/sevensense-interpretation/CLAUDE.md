# sevensense-interpretation

LLM-powered interpretation for 7sense: natural-language report generation, conservation insights, anomaly explanation, multi-language support.

## Files
- `Cargo.toml` - Depends on `sevensense-core`, `sevensense-analysis`, async + tracing libs.
- `src/lib.rs` - Crate root and architecture overview (reports, insights, prompts, providers).
- `src/templates.rs` - Prompt templates.
- `src/domain/` - Entities and repository traits.
- `src/application/` - Services that drive interpretation use cases.
- `src/infrastructure/` - Claim generator, evidence pack builder, LLM provider adapters.

## Build
```
cargo build -p sevensense-interpretation
```

## Related
- Consumes: `sevensense-analysis`. Consumed by: `sevensense-api`.
