# sevensense-interpretation/src/infrastructure

Infrastructure adapters for the interpretation bounded context.

## Files
- `mod.rs` - Adapter wiring.
- `claim_generator.rs` - Builds factual claims from analysis output using prompt templates.
- `evidence_builder.rs` - Assembles "evidence packs" (cluster excerpts, audio links, metrics) supporting each claim.

## Related
- Application services: `../application/services.rs`.
- Prompt templates: `../templates.rs`.
