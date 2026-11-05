# seekdb x hipVS notebook demo

Open `seekdb_hipvs_batch_demo.ipynb` from this directory and run all cells.
The notebook expects the AMD `seekdb-hipvs-nb-demo` image and a visible
`gfx1100` device.

Layout:

- `seekdb_hipvs_batch_demo.ipynb`: source notebook with no stored benchmark
  output.
- `seekdb_hipvs_batch_demo_executed.ipynb`: successful reference execution on
  AMD Radeon Cloud, including the final scoreboard.
- `lib/seekdb_demo.py`: helper functions for environment checks, observer
  startup, data loading, CPU/GPU batch timing and recall.

The first setup cell imports `pathlib.Path` and creates a stable telemetry UUID.
Do not remove that cell when moving the notebook between JupyterLab deployments.

Reference run: 2026-11-05 on ROCm 7.2.4 with an AMD Radeon `gfx1100` GPU.
The measured batch peak was 8.68x at 4,000 probes; timing values are expected to
vary between runs. See `../RUN_VERIFICATION_20261105.md` for the full record.
