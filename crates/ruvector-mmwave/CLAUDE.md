# ruvector-mmwave

Shared parser for Seeed MR60BHA2 (60 GHz) and HLK-LD2410 (24 GHz) mmWave-radar UART streams (ADR-063). One tested state-machine implementation consumed by both the host-side bridge (`ruvector-hailo-cluster::bin::mmwave-bridge`) and the on-device firmware (`examples/esp32-mmwave-sensor`). Faithful Rust port of `RuView/firmware/esp32-csi-node/main/mmwave_sensor.c`.

Internal-only (`publish = false`).

## Features

- `default = []` (no_std, zero-allocation in the hot path).
- `std` — pulls in `Vec`/`String`-using helpers for the host bridge. Core state machine is no_std unconditionally.

## Frame format (Seeed mmWave protocol)

```
[0]    SOF  = 0x01
[1-2]  frame_id        (u16 BE)
[3-4]  data_length     (u16 BE)
[5-6]  frame_type      (u16 BE)
[7]    header_checksum = ~xor(bytes 0..6)
[8..N] payload
[N+1]  data_checksum   = ~xor(payload)
```

Surfaced frame types: `0x0A14` breathing BPM, `0x0A15` heart rate, `0x0A16` distance (cm BE), `0x0F09` presence (0/1). Anything else becomes `Event::Unknown` for observability.

## Layout

- `Cargo.toml` — tiny; `publish = false`; `std` feature is purely additive.
- `Cargo.lock` — committed.
- `src/lib.rs` — single source file. Constants (`MAX_PAYLOAD = 64`), `Event` enum (`Breathing`, `HeartRate`, `Distance`, `Presence`, `Unknown { frame_type, payload_len }`, `ChecksumError`, `Resync`), parser state machine (`State::{Sof, Header, Payload, Trailer}`), and the public parser struct.

## Public API / key types

`Event` enum and a no_std `Parser` that consumes bytes and yields `Event` values; with `feature = "std"` there are convenience `Vec`-based helpers for the host bridge.

## Related

- `crates/ruvector-hailo-cluster` — host-side `mmwave-bridge` binary uses this parser (`features = ["std"]`).
- `examples/esp32-mmwave-sensor` — on-device firmware using the no_std path.
