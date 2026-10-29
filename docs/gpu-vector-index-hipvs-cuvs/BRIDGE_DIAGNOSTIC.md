# Paired bridge GPU diagnostic

This compares V5 and the candidate bridge on the same installed hipVS and GPU.
It does not build or exercise seekdb SQL/DDL, test the server's CuvsWorker, or
replace official evaluation. It needs Python with NumPy and the real bridge
shared libraries; no fake GPU backend is used.

Build V5's bridge from `634888d9b037153fa143d2c913ba4b25503e1d62` and the
candidate bridge into separate directories, retaining the shared-library name.
The original API functions, including serialize/deserialize, must be present.
Run from a shell with the hipVS and ROCm library directories on LD_LIBRARY_PATH:

```bash
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 python3 \
  unittest/oblib/vector/benchmark_cuvs_bridge.py \
  --baseline /workspace/seekdb-amd-baseline/libseekdb_cuvs_bridge.so \
  --candidate /workspace/seekdb-amd-build/libseekdb_cuvs_bridge.so \
  --base /seekdb_workspace/datasets/base.f32 \
  --queries /seekdb_workspace/datasets/query.f32 \
  --dim 128 --rounds 3 \
  --output /workspace/seekdb-amd-build/bridge-comparison.json
```

The thread limits above match the user's 16-core cgroup quota, not the official
24-core slot. Adjust them to the actual allocation on another host. Input files
must be headerless, little-endian row-major FP32 matrices with the given
dimension. The script prints the inferred row and query counts. The template's
approximately 5 MB dataset is a small demonstration, not the official 250K case.

Before GPU timers start, the script computes exact Top-100 using FP64 arithmetic
over the original FP32 values. It does not use approximate results as truth.
IDs represent zero-based rows in the raw base file. It rejects missing,
duplicate, or out-of-range IDs. At every round, each version builds a fresh
index, serializes it into a fresh temporary file, frees the original handle,
reloads the file, and completes its first query. All GPU calls execute on one
32 MiB-stack worker. The next round alternates baseline/candidate order.

`ready_seconds` covers build through first query, including intermediate free
and reload. The individual `*_call_seconds` fields measure API return times;
asynchronous GPU completion can be charged to a later call. The first search's
host result copy completes GPU work before the ready timer ends. Timed Top-10
and Top-100 workloads use individual bridge calls for all input queries.
CPU recall validation is outside those query timers.

The JSON includes per-round timings, recall, bridge QPS, snapshot sizes, median
ready speedup, and median recall changes. It is updated after each successful
run so partial evidence survives a later failure. Temporary snapshots are
removed. No cross-round graph or index is retained. The process and runtime
remain alive between versions, so context/allocator warmup can still differ
from an official separate-process test.

This omits SQL, table scan, observer persistence streams, external profiling,
server resource gates, and Hidden250K. Snapshot byte size is not VRAM usage.
These diagnostics cannot establish official gates or a score. Retain the raw
JSON and reproduce promising results with the official evaluator.

On the user's instance, GPU 0 has PCI address 0000:e3:00.0. If its sysfs
counter is readable, add:

```bash
--vram-used-file /sys/bus/pci/devices/0000:e3:00.0/mem_info_vram_used
```

The tool samples absolute device VRAM usage every 20 ms from before library
loading until after handle release, reporting initial usage, sampled peak, and
sample count per run. Other GPU users can affect this counter. A sampled peak
is a lower bound because short allocations can fall between samples; it is not
the official profiler's peak and cannot prove a memory gate passed.

CPU metric validation can run without NumPy or a GPU:

```bash
python3 -B -m unittest discover -s unittest/oblib/vector \
  -p test_bridge_benchmark_metrics.py -v
```
