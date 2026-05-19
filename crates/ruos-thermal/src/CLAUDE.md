# ruos-thermal/src

Source for the `ruos-thermal` skeleton.

## Files

- `lib.rs` — Public library: `ThermalSensor`, `ThermalSnapshot`, `CpuTemp`, `CpuPolicy`. Sysfs walker. `#![warn(missing_docs)]`.
- `main.rs` — `ruos-thermal` binary entry. Reads a snapshot and prints zones/policies.
