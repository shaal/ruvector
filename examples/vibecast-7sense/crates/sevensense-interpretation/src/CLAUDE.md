# sevensense-interpretation/src

Source for the interpretation bounded context.

## Files
- `lib.rs` - Crate root; documents the layout (`reports/`, `insights/`, `prompts/`, `providers/` in logical terms) and re-exports the public API.
- `templates.rs` - Prompt templates used by claim generation and evidence packaging.

## Subdirectories
- `domain/` - Entities and repository traits.
- `application/` - Interpretation use case services.
- `infrastructure/` - Claim generator, evidence builder, LLM provider integration.

## Related
- Parent: `../CLAUDE.md`.
