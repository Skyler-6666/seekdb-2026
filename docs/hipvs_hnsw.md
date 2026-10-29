# hipVS HNSW Build Backend

## Overview

seekdb normally builds HNSW indexes inside VSAG on the CPU. The optional
`SEEKDB_ENABLE_HIPVS` build adds a hipVS backend for dense `float32` HNSW
indexes:

1. hipVS builds a CAGRA k-NN graph on the AMD Radeon GPU.
2. `cuvsHnswFromCagra` converts that graph to an HNSW index.
3. The default CPU hierarchy keeps the index mutable so seekdb incremental
   writes can use `cuvsHnswExtend`.
4. Any unsupported case falls back to the original VSAG HNSW implementation.

The rest of seekdb (distance semantics, external vector IDs, serialization,
and query result ownership) remains behind the existing VSAG adapter.

## Supported configurations

The accelerated path is selected only when all of these conditions hold:

- the binary was built with `SEEKDB_ENABLE_HIPVS=ON`;
- `SEEKDB_HIPVS` is not `0`, `false`, `off`, or `disabled`;
- index type is `HNSW` (not HNSW-SQ/HNSW-BQ/HGraph);
- dtype is `float32`;
- metric is `L2` or inner product;
- `extra_info_size` is zero, because hipVS filtered search is not exposed by
  this adapter.

Other indexes and configurations continue to use VSAG unchanged.

## Build hipVS

Use a Linux x86_64 AMD ROCm host. hipVS requires ROCm, HIP, CMake, and the
other prerequisites documented by the hipVS project. The C API and the
CAGRA-to-HNSW bridge must both be enabled:

```bash
cd hipVS/cpp
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=NATIVE \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_C_LIBRARY=ON \
  -DBUILD_CAGRA_HNSWLIB=ON \
  -DBUILD_TESTS=OFF
cmake --build build --parallel
cmake --install build
```

Verify that the installation contains the exported `cuvs::c_api` target and
`libcuvs_c.so`:

```bash
find hipVS/cpp/install -name 'cuvs-config.cmake' -o -name 'libcuvs_c.so'
```

## Build seekdb with hipVS

```bash
export CMAKE_PREFIX_PATH="/path/to/hipVS/cpp/install:${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="/path/to/hipVS/cpp/install/lib:${LD_LIBRARY_PATH}"

cd seekdb-2026
./build.sh release --init \
  -DSEEKDB_ENABLE_HIPVS=ON \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
```

Without `SEEKDB_ENABLE_HIPVS=ON`, the code contains no hipVS headers, symbols,
or runtime dependency and the default VSAG build is unchanged.

## Runtime controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `SEEKDB_HIPVS` | enabled when compiled in | Set to `0` to force the VSAG CPU backend. |
| `SEEKDB_HIPVS_GPU_HIERARCHY` | `0` | Set to `1` to build upper HNSW layers on the GPU. This keeps the same search/recall model but makes the index immutable for `Add`. Leave it `0` unless the workload is import-only. |

Logs from the selected path contain `[OBVSAG][HIPVS] selected hipVS HNSW backend`
and `[OBVSAG][HIPVS] built HNSW index on GPU`. If resources cannot be created,
the adapter logs a warning and falls back to VSAG.

## Reproducible build-time check

Create a table without an index, load the workload vectors, and measure index
creation on the same machine and dataset. Run once with `SEEKDB_HIPVS=0` and
once with `SEEKDB_HIPVS=1` (restart seekdb between runs):

```sql
CREATE TABLE hipvs_bench (
  id BIGINT PRIMARY KEY,
  embedding VECTOR(768)
) ORGANIZATION = HEAP;

-- Load N vectors into hipvs_bench here.

SET profiling = 1;
CREATE VECTOR INDEX idx_hipvs_hnsw ON hipvs_bench(embedding)
  WITH (distance=l2, type=hnsw, lib=vsag, m=32,
        ef_construction=200, ef_search=200);
SHOW PROFILES;
```

Compare `Query_time` for the `CREATE VECTOR INDEX` statement. For an
end-to-end run, also execute the same top-k ANN queries after each build and
compare result IDs/recall against a full scan; the backend switch must not
change the SQL contract.

The GPU path is intended for workloads where CAGRA graph construction is large
enough to amortize device setup and host/device transfers. Small indexes may
remain faster with the VSAG CPU path, so set `SEEKDB_HIPVS=0` when testing a
small or latency-sensitive import.
