# ruvbot / src / skills

Skill execution layer plus built-in skills.

## Files
- `SkillExecutor.ts` - Runs registered skills against an agent/session,
  enforcing argument schemas and timeouts.
- `index.ts` - Barrel re-exporting the executor and built-in skills.

## Subdirectories
- `builtin/` - Out-of-the-box skills shipped with ruvbot.
