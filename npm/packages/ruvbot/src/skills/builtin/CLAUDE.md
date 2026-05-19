# ruvbot / src / skills / builtin

Built-in skills shipped with ruvbot.

## Files
- `CodeSkill.ts` - Code generation / execution skill.
- `MemorySkill.ts` - Read/write the learning memory store from chat.
- `SummarizeSkill.ts` - Summarization helper for long contexts.

Each skill registers with `SkillExecutor` via the parent barrel.
