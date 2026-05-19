# ruos-thermal

Pi 5 thermal supervisor + over/underclock control (ADR-174). Iter 91 ships a pure-read sysfs reader; the supervisor daemon + Unix-socket budget protocol + clock writes land in iter 92-97 per the ADR-174 roadmap.

## Important files

- `Cargo.toml` — `[workspace]` standalone (rejoins parent once API stabilizes). Pure stdlib + tempfile for tests. Provides a `ruos-thermal` binary at `src/main.rs`.
- `Cargo.lock` — Standalone lockfile.
- `src/lib.rs` — Public `ThermalSensor` API. Walks `/sys/class/thermal/thermal_zone*` for `CpuTemp` and `/sys/devices/system/cpu/cpufreq/policy*` for `CpuPolicy`. Pi 5 reports millidegrees.
- `src/main.rs` — CLI binary that prints a thermal snapshot.
- `tests/cli.rs` — Integration test of the CLI behavior.
- `deploy/` — systemd unit + installer (see `deploy/CLAUDE.md`).

## Public API

- `ThermalSensor::system()` — Use real sysfs paths.
- `ThermalSensor::read() -> io::Result<ThermalSnapshot>` — Snapshot with `cpu_temps_celsius: Vec<CpuTemp>` and `cpu_policies: Vec<CpuPolicy>`.
- `CpuTemp { zone, celsius }`, `CpuPolicy { id, cur_hz, max_hz }`.

## Build / Feature notes

- Pure-Rust, zero deps for the skeleton — `tokio`/`serde` join later for the socket protocol.
- ADR-174 deliverable. Get the read path right + tested before adding writer + IPC.

## Related

- `hailort-sys` — Companion Pi 5 / AI HAT+ accelerator binding.
- Future iterations will integrate with `agentic-robotics-core` for telemetry pub/sub.
