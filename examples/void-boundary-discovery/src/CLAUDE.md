# void-boundary-discovery/src

Source for the cosmic-void boundary-discovery binary.

## Files
- `main.rs` - Generates `N_GALAXIES=1000` galaxies in a `BOX_SIZE=100.0` square containing `N_VOIDS=7` voids (radius 12-...), builds a friends-of-friends graph with `LINKING_LENGTH=5.0`, then compares Fiedler value and mincut across each void's boundary, interior, and exterior region using `ruvector-coherence::spectral` and `ruvector-mincut`.

## Related
- Parent: `../CLAUDE.md`.
- Sibling boundary-discovery binaries.
