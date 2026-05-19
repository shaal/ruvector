# scipix/scripts

Helper shell scripts for scipix development.

## Files

- `setup_dev.sh` - Provisions a local dev environment (toolchains, model dirs, deps).
- `download_models.sh` - Downloads required ONNX OCR models into `models/`.
- `run_benchmarks.sh` - Runs the criterion benches in `../benches/` and aggregates results.

## How to run

```bash
bash /home/user/ruvector/examples/scipix/scripts/setup_dev.sh
bash /home/user/ruvector/examples/scipix/scripts/download_models.sh
bash /home/user/ruvector/examples/scipix/scripts/run_benchmarks.sh
```

## Related

- Parent crate: `examples/scipix/`.
- Benches: `../benches/`.
