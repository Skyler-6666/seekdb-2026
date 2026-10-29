# AMD Radeon Cloud hipVS Validation

## Environment

- Date: 2026-10-30
- Platform: AMD Radeon Cloud China site
- Template: `seekdb-hipvs`
- GPU: AMD Radeon Graphics, `gfx1100` / RDNA3
- GPU memory: 49,136 MB
- CPU: AMD EPYC 9334 32-Core Processor, 128 vCPUs
- Memory: 44 GiB visible to seekdb
- OS: Ubuntu 24.04
- ROCm/HIP: 7.2.4
- seekdb: 1.4.0
- hipVS bridge: `libseekdb_cuvs_bridge.so` present
- `libcuvs` links in `seekdb`: 3

## Validation workload

- 10,000 base vectors per index, 128 dimensions
- 100 query vectors
- HNSW, L2, `m=16`, `ef_construction=200`, `ef_search=64`
- GPU index: `lib=cuvs`
- CPU baseline: `lib=vsag`
- Both indexes loaded with 10,000 rows
- Vector index ready in approximately 20 seconds
- Notebook execution completed successfully on the Radeon GPU instance

## Measured results

| Probe count | CPU ms | GPU ms | Speedup | CPU probes/s | GPU probes/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 322.2 | 1072.1 | 0.30x | 310 | 93 |
| 500 | 1498.6 | 1023.0 | 1.46x | 334 | 489 |
| 1000 | 2959.3 | 972.4 | 3.04x | 338 | 1028 |
| 2000 | 5922.7 | 958.5 | 6.18x | 338 | 2087 |
| 4000 | 10034.0 | 1206.3 | 8.32x | 399 | 3316 |

Additional results:

- Peak GPU speedup: **8.32x at 4,000 probes**
- GPU throughput: **3,316 probes/s**, CPU: **399 probes/s**
- Single-query latency: CPU 8.26 ms, GPU 12.90 ms
- recall@10: GPU `0.8750`, CPU `0.6400`

## Interpretation

The GPU path is beneficial for batched similarity workloads: at 4,000 probes it
is 8.32x faster than the VSAG CPU path and increases recall@10 from 0.6400 to
0.8750. Single-query execution remains faster on the CPU HNSW path, so the
implementation uses the GPU path for bulk workloads rather than forcing it for
latency-sensitive single queries.

Raw measurement data: `tools/benchmark/hipvs_hnsw_amd_radeon_results.csv`.
