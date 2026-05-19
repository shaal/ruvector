# spiking-network / src / network

The spiking network itself - topology, event-driven scheduler (using `priority-queue`), and the simulation loop.

## Important files
- `mod.rs` - module root; exposes the `Network` type, connection graph, and the event-driven `simulate` / `step` entry points.

## Related
- Neuron implementations driven by the scheduler: `../neuron/`. Input encoders: `../encoding/`.
