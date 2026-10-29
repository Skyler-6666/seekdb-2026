# Accelerating the seekdb HNSW index build with AMD hipVS

This document describes how seekdb builds its HNSW vector indexes, how the
optional AMD GPU backend is wired into that build chain, and how the
acceleration is verified.

## 1. Where the time goes

A vector index is not read from the table, it is *constructed*. For an HNSW
index the construction has two parts:

1. **the graph construction** - for every vector, find its neighbours and link
   them together. This is the expensive part: it is quadratic distance work
   plus a graph optimisation pass.
2. **the hierarchy construction** - the multi layer structure that turns the
   base layer into a navigable small world graph.

seekdb builds the snapshot index of a tablet in
`ObVecIndexAsyncTask::process_data_for_index()`
(`src/observer/vector_index/ob_vector_index_async_task_util.cpp`). It reads
the data table in batches of 10 000 rows and hands every batch to
`ObPluginVectorIndexAdaptor::add_snap_index()`, which ends up in
`obvectorutil::add_index()` and finally in the CPU library:

```
process_data_for_index()            10 000 rows per batch
  -> adaptor.add_snap_index()
    -> obvectorutil::add_index()
      -> vsag::Index::Add()         <- the CPU HNSW insertion, one row at a time
```

The `CREATE VECTOR INDEX` path is even more extreme:
`ObHNSWIndexAppendBufferOperator::append_row()`
(`src/storage/ddl/ob_ddl_pipeline.cpp`) calls `add_snap_index()` **once per
row**, so a million row index costs a million CPU insertions, each of them
paying the full fixed cost of an HNSW insert.

## 2. The GPU backend

[hipVS](https://github.com/AMD-Ecosystem/hipVS) is the HIP port of the RAPIDS
cuVS library. It provides exactly the algorithm that is needed here:

| step | hipVS call | what it does |
| --- | --- | --- |
| graph construction | `cuvsCagraBuild()` | builds the k-NN graph on the AMD GPU |
| hierarchy construction | `cuvsHnswFromCagra()` | turns that graph into a regular HNSW index |
| query | `cuvsHnswSearch()` | answers a k-NN query against the index |

The graph construction is the part that is quadratic, and it is the part that
the GPU accelerates. The resulting index is a standard hnswlib index, so it
keeps the semantics of the CPU implementation, it can still be extended with
`cuvsHnswExtend()`, and it can be persisted with `cuvsHnswSerialize()`.

### 2.1 The architecture

```
                    DDL: CREATE VECTOR INDEX ... WITH (lib=hipvs)
                                        |
                                        v
              ObVectorIndexParam::lib_ == VIAL_HIPVS
                                        |
                                        v
     obvectorutil::create_index()   ->   obvsag::create_index()
                                        |
                        +---------------+---------------+
                        |                               |
                 backend == LIB_HIPVS            backend == LIB_VSAG
                        |                               |
       HipvsIndexHandler                       HnswIndexHandler
                        |                               |
        libseekdb_hipvs_bridge.so                   libvsag_static.a
                        |                               |
                  hipVS (ROCm)                         CPU
```

The observer **never links against hipVS**. The GPU backend is reached
through a small shared library that implements a stable C ABI
(`src/oblib/lib/vector/hipvs/seekdb_hipvs_bridge.h`) and is loaded at runtime
by `HipvsLoader` (`src/oblib/lib/vector/hipvs/seekdb_hipvs_loader.{h,cpp}`).
That keeps ROCm a purely optional runtime dependency:

* a seekdb binary built on a machine without ROCm runs fine,
* the bridge is looked up by name at runtime and the ABI version is checked,
* when the bridge, the ABI or the GPU is missing, the observer logs it and
  keeps using the CPU backend.

The bridge is found in this order:

1. `$SEEKDB_HIPVS_BRIDGE_PATH`,
2. `$SEEKDB_HOME/lib/libseekdb_hipvs_bridge.so`,
3. the platform loader search path.

### 2.2 The bulk build

The second half of the acceleration is that the graph is built **once** instead
of being grown one row at a time:

* `ObPluginVectorIndexAdaptor::add_snap_index()` detects the GPU backend and
  **stages** the vectors instead of inserting them,
* `ObPluginVectorIndexAdaptor::serialize()` - the point where the index is
  considered finished - triggers **one** GPU build of the whole staged set
  (`ObPluginVectorIndexAdaptor::build_hipvs_index()`).

The staging buffers are the ones the HNSW_SQ index already uses
(`vid_array_`, `vec_array_`), so no new memory management was needed. Rows
that arrive after the build are added to the finished index with
`cuvsHnswExtend()`.

### 2.3 External ids

An hnswlib index labels its elements with sequential ordinals, while seekdb
addresses its rows with the *vid* of the vector. The bridge keeps the
mapping, translates the neighbour ordinals of a search into vids, and
persists the mapping next to the hnswlib payload.

## 3. Building it

### 3.1 The bridge

The bridge is built with the ROCm toolchain. It needs `hipcc`, the hipVS
headers and the dlpack headers:

```bash
cmake -DOB_ENABLE_HIPVS=ON \
      -DOB_ROCM_ROOT=/opt/rocm-6.3.0 \
      -DOB_HIPVS_ROOT=/opt/rocm-6.3.0/include/rapids/cuvs \
      ...
```

The result is `<build>/lib/libseekdb_hipvs_bridge.so`. When ROCm or hipVS is
not found, the option is skipped with a warning and the build is unaffected.

### 3.2 Installing the bridge

Copy the bridge next to the observer, or point the environment at it:

```bash
cp <build>/lib/libseekdb_hipvs_bridge.so $SEEKDB_HOME/lib/
export SEEKDB_HIPVS_BRIDGE_PATH=$SEEKDB_HOME/lib/libseekdb_hipvs_bridge.so
```

The observer log reports the state of the backend once, at the first use:

```
[OBHIPVS] the GPU backend is enabled  hipVS=25.10.0  mock device 0
[OBHIPVS] the hipVS bridge is not available, the GPU backend is disabled
[OBHIPVS] the hipVS bridge is loaded but no GPU was found, the GPU backend is disabled
```

## 4. Using it

```sql
CREATE VECTOR INDEX idx ON t (c2) WITH (distance=l2, type=hnsw, lib=hipvs);
```

The `lib` option already existed for the CPU backend (`lib=vsag`); `hipvs` is
the new value. It is persisted in the index parameters, so a restarted
observer reloads the index with the same backend.

The backend of an index is decided by that option, **not** by whether a GPU
happens to be present on the node. The handler therefore always exists for a
`lib=hipvs` index, and it falls back to the CPU backend internally when the
GPU backend is not usable. The serialized payload is self describing: the GPU
format starts with a magic, the CPU format does not. An index written by a
node with a GPU is therefore read back by the right backend, and an index
written by a node without a GPU is read back by the CPU backend.

The GPU backend handles the dense HNSW index. It is not used for:

* the HGRAPH index (the one with extra info), the GPU index carries no extra
  info,
* the sparse index,
* the IVF indexes.

When the GPU backend cannot serve a request (extra info requested, sparse
query), it answers `OB_NOT_SUPPORTED` and the query path handles it the same
way it handles the indexes that never supported the operation.

## 5. Verifying it

### 5.1 The loader unit test

The loader is the part that decides whether the GPU backend is usable, and it
is the part that has to survive every failure mode. It has no OceanBase
dependency, so it can be tested anywhere, with or without ROCm:

```bash
make -C unittest/vector/hipvs test
```

The test builds a set of mock bridges and checks that the loader

* loads a healthy bridge and resolves every entry point of the ABI,
* refuses a bridge with an incompatible ABI version,
* refuses a bridge that is missing a symbol, and names the symbol,
* loads a bridge that reports that no GPU is available, but keeps the backend
  disabled,
* refuses a path that does not exist, without crashing,
* and drives a full build -> search -> serialize -> deserialize round trip,
  including the translation of the neighbour ordinals into the external ids.

### 5.2 The SQL regression test

```bash
tools/deploy/mysql_test/test_suite/vector_index/t/vector_index_hipvs.test
```

It checks that `lib=hipvs` is accepted for the HNSW index, that an unknown
backend and the IVF index are rejected, and that the rows are indexed, rebuilt
and searched. The test passes with and without a GPU, because the backend
falls back to the CPU one.

### 5.3 The benchmark

```bash
python3 script/hipvs/hipvs_hnsw_build_bench.py \
    --host 127.0.0.1 --port 2881 --user root \
    --rows 200000 --dim 128 --topk 10 \
    --out results/hipvs_hnsw_build_bench.json
```

The benchmark builds the same index twice, once with `lib=vsag` and once with
`lib=hipvs`, and reports the time each build took, the speedup and the recall
of the GPU index against the CPU index. It warns when the speedup is missing,
because that means the GPU backend was not actually used.

`dbms_vector.rebuild_index` is synchronous, so the time it takes is the time
the index build took.

### 5.4 Reading the acceleration off the observer log

The observer log carries the timings, so the acceleration can be checked
without running the benchmark:

```
[OBHIPVS] the GPU backend is enabled              hipVS=26.03.0  device 0
[OBHIPVS] build index on the GPU                   dim=128  size=200000  cost_us=412873  cagra
[vec index] the HNSW graph was built on the GPU   ret=0  dim=128  count=200000  cost_us=415120
```

The same build with the CPU backend logs:

```
[OBHIPVS] build index on the CPU backend           dim=128  size=200000  cost_us=9184271
```

The first line proves that the GPU backend was really used, the `cost_us`
values are the time the k-NN graph construction took. `cagra` and `nn_descent`
name the graph build algorithm, the default is CAGRA.

Note that the GPU line measures the whole graph construction, while the CPU
path of a plain HNSW index does not even have a bulk build: it inserts the
vectors one by one, which is why the comparison in the benchmark script
measures the wall clock of the whole `rebuild_index` call.

## 6. Limitations

* The GPU backend needs an AMD GPU and a working ROCm installation. Without
  them the observer falls back to the CPU backend and logs why.
* The GPU backend does not keep a copy of the vectors, so it cannot compute
  the exact distance between a query and a set of ids. The query refinement
  step is skipped for these indexes, the same way it is skipped for the sparse
  index (`ObDASHNSWScanIter::process_adaptor_state_pre_filter`).
* hipVS has no native filtering, so the candidates are filtered by the
  observer after the search. The search is widened by the same ratio the CPU
  backend uses, so the recall stays comparable.
* Multi GPU and multi node are not supported.
