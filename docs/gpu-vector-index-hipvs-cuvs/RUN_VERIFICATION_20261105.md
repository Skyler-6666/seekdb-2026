# AMD gfx1100 end-to-end notebook validation (2026-11-05)

This record captures a fresh execution of the seekdb + hipVS demo on the AMD
Radeon Cloud interactive instance. It complements, but does not replace, the
official build/performance gates in the contest environment.

## Base and purpose

- Base answer commit: `d01525b4c9eecd25079ce4e5289236d9519cbe9d`
- Base branch: `competition/hipvs-latest-20261030`
- Validation date: 2026-11-05
- Runner: AMD Radeon Cloud JupyterLab, Python 3 kernel
- Notebook: `seekdb_hipvs_batch_demo_executed.ipynb`

The notebook is a clean 10-code-cell execution. Cell 2 creates a stable
telemetry instance UUID and imports `Path`; the former missing-`Path`
execution failure is therefore covered by the checked-in source notebook.

## Environment

| Item | Observed value |
| --- | --- |
| GPU | AMD Radeon Graphics |
| Architecture | gfx1100 / RDNA3 |
| VRAM | 49,136 MB total, 26 MB used at environment check |
| ROCm | 7.2.4 |
| seekdb | 5.7.25-OceanBase seekdb-v1.4.0.0 |
| seekdb binary | OK |
| cuVS bridge | OK |
| libcuvs-linked libraries | 3 |
| SEEKDB_HOME | /seekdb_workspace |
| Telemetry | `/workspace/run/obs1/run/telemetry.json` generated |

## Workload

- 10,000 x 128-dimensional base vectors.
- Two equivalent indexes:
  - `t_cuvs`: `WITH (distance=l2, type=hnsw, lib=cuvs)`
  - `t_vsag`: `WITH (distance=l2, type=hnsw, lib=vsag)`
- Batch probe sizes: 100, 500, 1,000, 2,000 and 4,000.
- Top-k: 10.
- GPU index ready time observed through the existing stability probe: about 20 s.
  This is a smoke/readiness measurement, not a CPU-vs-GPU build-speed claim.

## Results

### Single query

| Path | Latency |
| --- | ---: |
| CPU / VSAG | 8.32 ms |
| GPU / cuVS | 12.73 ms |

Single-row GPU search is slower, as expected from dispatch and transfer overhead.

### Batch search

| Probes | CPU ms | GPU ms | Speedup | CPU probes/s | GPU probes/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 318.5 | 1113.6 | 0.29x | 314 | 90 |
| 500 | 1475.3 | 907.9 | 1.62x | 339 | 551 |
| 1,000 | 2953.1 | 987.5 | 2.99x | 339 | 1013 |
| 2,000 | 5860.5 | 1215.6 | 4.82x | 341 | 1645 |
| 4,000 | 10034.5 | 1155.6 | **8.68x** | 399 | **3461** |

### Correctness

- Recall@10, CPU/VSAG: **0.6400**
- Recall@10, GPU/cuVS batch: **0.8760**
- GPU batch results were consistent with GPU single-query results.
- All 10 code cells executed with no Traceback or Error output.

## What this proves

- The notebook source is self-contained with respect to `Path`, helper imports,
  telemetry UUID setup and helper-library discovery.
- The AMD hosted instance exercises a real GPU-capable seekdb binary and bridge.
- The declarative `lib=cuvs` and `CALL dbms_vector.batch_knn(...)` paths execute
  successfully end to end at the documented 10K-vector scale.
- The batch crossover and recall behavior are reproducible, though timings vary
  by run and instance.

## What this does not prove

- It does not by itself establish the official cold-build speedup on the 250K
  contest datasets; use `validation/run_build_compare.sh` for that comparison.
- It does not replace the organizer full-source build, startup, snapshot reload,
  hidden recall, QPS, or VRAM gates.
- The observed 8.68x batch speedup is not a competition score.

## Artifacts

- `notebooks/seekdb_hipvs_batch_demo.ipynb` - source notebook.
- `notebooks/seekdb_hipvs_batch_demo_executed.ipynb` - fresh executed evidence.
- `notebooks/lib/seekdb_demo.py` - SQL/process helper used by the notebook.
- `evidence/summary.csv` - raw batch measurements.
- `evidence/seekdb_hipvs_speedup.png` - generated speedup chart.
- `evidence/seekdb_hipvs_final.png` - rendered final scoreboard.
