# GPU accelerated HNSW index build

This directory holds the optional GPU build path of the HNSW index of seekdb and
the tooling that measures it.

## What the path does

seekdb builds its HNSW index with VSAG, which runs the classic hnswlib algorithm
on the CPU. The expensive part of that algorithm is the all-neighbours KNN graph:
for every vector the distances to the candidates of the current graph have to be
computed. This path moves exactly that part to an AMD GPU:

1. The rows that are loaded into the tablet are accumulated in memory instead of
   being added to the index one by one.
2. The accumulated vectors are handed to hipVS, which builds the KNN graph with
   CAGRA on the GPU and turns it into an HNSW index.
3. hipVS writes the index in the nmslib/hnswlib v0.7.0 layout. That is the same
   layout VSAG reads, so the bytes are handed to VSAG through `Index::Deserialize`
   after the external labels of the elements have been rewritten to the row vids.

Everything else stays on the CPU, and the whole path is optional: if the plugin
is missing, if no AMD GPU is present or if anything fails, the build falls back
to the plain CPU path.

The GPU path is used for a plain `type=hnsw` index with `distance=l2` or
`distance=ip`. A cosine index is excluded because VSAG normalizes the vectors of
a cosine index itself and those normalized vectors are not part of the
serialized image. An index with extra info columns is excluded as well.

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `OB_GPU_HNSW_BUILD` | `1` | set to `0` to keep the plain CPU build |
| `OB_GPU_HNSW_PLUGIN_PATH` | - | file or directory that holds `libob_gpu_hnsw_plugin.so` |
| `OB_GPU_HNSW_BUILD_ALGO` | `0` | 0 auto, 1 ivf_pq, 2 nn_descent, 3 iterative_cagra |
| `OB_GPU_HNSW_MAX_BYTES` | `4294967296` | memory budget of the buffered vectors |

The plugin is looked up in `OB_GPU_HNSW_PLUGIN_PATH`, next to the observer
binary, in `/usr/local/lib`, in `/usr/lib` and in the working directory.

## Building the plugin

The plugin is compiled with `hipcc` against hipVS and does not belong into the
seekdb build, so that seekdb still builds and runs on machines without an AMD
GPU.

```bash
cd src/oblib/lib/vector/gpu
./build_plugin.sh
```

The script detects hipVS in `/opt/hipvs`, `/usr/local/hipvs`, `/opt/rocm/hipvs`
and `/usr/local/rapids/cuvs`. Set `HIPVS_ROOT` and `DLPACK_INCLUDE_DIR` when the
libraries live somewhere else.

The result is `libob_gpu_hnsw_plugin.so`. Copy it next to the observer binary or
point `OB_GPU_HNSW_PLUGIN_PATH` at it.

## Measuring the speedup

`benchmark_hnsw_build.py` loads a synthetic dataset, builds the index and
compares the approximate top-k results against the exact scan:

```bash
python3 benchmark_hnsw_build.py --rows 200000 --dim 128 --label cpu --output cpu.json
python3 benchmark_hnsw_build.py --rows 200000 --dim 128 --label gpu --output gpu.json
python3 compare_benchmark_results.py cpu.json gpu.json
```

The observer decides which path is taken, the script only measures. The two
runs therefore need an observer that was started in the matching mode, which is
what `run_gpu_vs_cpu_benchmark.sh` does:

```bash
./run_gpu_vs_cpu_benchmark.sh \
    --stop-cmd "obd cluster stop seekdb" \
    --start-cmd "obd cluster start seekdb" \
    --plugin /usr/local/lib/libob_gpu_hnsw_plugin.so \
    --rows 200000 --dim 128
```

The script restarts the observer with `OB_GPU_HNSW_BUILD=0` for the CPU run and
with `OB_GPU_HNSW_BUILD=1` for the GPU run, writes `result_cpu.json`,
`result_gpu.json` and `comparison.json` and prints the comparison table.

## Reading the observer log

The observer logs the time of both build paths with the same shape:

```
build hnsw index with cpu success [count] [dim] [seconds]
build hnsw index with gpu success [count] [dim] [seconds] [max_degree] [graph_degree] [gpu_seconds]
```

A GPU build that falls back to the CPU is logged as
`build hnsw index with gpu failed, falling back to the cpu path`.

## Troubleshooting

| symptom | cause |
| --- | --- |
| `dlopen plugin failed` | the plugin was built against a different ROCm, or `libcuvs_c` is not found |
| `no usable AMD GPU` | the observer does not see the device, check `rocminfo` and the permissions of the device node |
| `gpu hnsw build disabled, the buffered vectors exceed the memory budget` | the dataset is larger than `OB_GPU_HNSW_MAX_BYTES`, raise it or build more shards |
| `not enough rows for the requested graph degree` | the table holds fewer than `2 * m` rows, the GPU path needs a graph |
