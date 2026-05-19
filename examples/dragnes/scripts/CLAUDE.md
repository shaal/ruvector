# dragnes / scripts

Operational scripts for the DrAgnes app.

## Important files
- `deploy.sh` - end-to-end deploy script (invoked by `npm run deploy`); builds the Docker image and pushes / deploys per `../cloud-run.yaml`.
- `analyze-ham10000.js` - one-off helper that produced `../docs/HAM10000_stats.json` from the HAM10000 dataset.

## Related
- Container build: `../Dockerfile`. Deploy target: `../cloud-run.yaml`. Generated stats: `../docs/HAM10000_stats.json`.
