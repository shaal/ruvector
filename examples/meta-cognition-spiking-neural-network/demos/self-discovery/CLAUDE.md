# demos/self-discovery

Cognitive self-discovery demos: an agent explores its own capabilities, reflects, and persists what it learns.

## Files
- `cognitive-explorer.js` - Baseline self-discovery system: explores capabilities, learns from discoveries, stores patterns in memory.
- `enhanced-cognitive-system.js` - Smarter variant that routes work to different attention mechanisms (Multi-Head for relations, Hyperbolic for hierarchy, Flash for long context, MoE for specialist routing).
- `memory.bin` - Persisted baseline memory.
- `enhanced-memory.bin` - Persisted enhanced-system memory.

## Run
```
node cognitive-explorer.js
node enhanced-cognitive-system.js
```

## Related
- Parent: `../CLAUDE.md`.
- Sibling exploration: `../exploration/`.
- Attention mechanisms: `../attention/`.
