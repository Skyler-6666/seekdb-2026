# Second CAGRA optimization candidate

## Evidence and target

The user reported the official result for `556065c62578673d6313f2fdd798cc8753cbb662`:

```text
build=1.005x, query=0.994x, gates=pass
```

No per-dataset times, recall, QPS, memory usage, or logs were supplied. The
first candidate passed gates in that evaluation, but its scratch allocation
savings did not yield a meaningful aggregate performance improvement.

## Changes

Reuse one synchronous 32 MiB-stack worker for cuVS build, search, serialize,
and deserialize. Previously every operation created and joined a pthread,
including every single-row SQL query. The same thread now survives across
jobs, allowing HIP thread-local runtime state to survive. Caller-owned job
data remains alive until completion. No index data is cached across rounds.

Select the frozen library's `ITERATIVE_CAGRA_SEARCH` construction algorithm
instead of `NN_DESCENT`. It grows a graph using batched GPU CAGRA search and
graph optimization, targeting NN-Descent's repeated host graph updates,
sampling, and graph transfers. This is an algorithm change, not a measured
speedup or a proof of equivalent recall.

Graph degree remains 128 and intermediate degree remains 256. Input/query
vectors remain FP32. Search still uses width 4 and at least `max(512, 4*k)`
intermediate candidates. The NN-Descent iteration limit field is unused by
the new algorithm. Snapshot format, persistence, ID mapping, SQL parameters,
and lib=vsag implementation are unchanged.

The persistent worker serializes GPU jobs across indexes. Registry searches
already held a global mutex, but independent builds and raw batch calls can
now queue. Hardware concurrency testing is still needed for that workload.

## Fixed-library source verification

The library was read at the organizer's fixed commit; it was not modified:

- [C API enum](https://github.com/AMD-Ecosystem/hipVS/blob/f96a663d0a4441d6f5344bd3ef552b71ae7b370b/cpp/include/cuvs/neighbors/cagra.h)
- [C API dispatch](https://github.com/AMD-Ecosystem/hipVS/blob/f96a663d0a4441d6f5344bd3ef552b71ae7b370b/cpp/src/neighbors/cagra_c.cpp)
- [Iterative construction](https://github.com/AMD-Ecosystem/hipVS/blob/f96a663d0a4441d6f5344bd3ef552b71ae7b370b/cpp/src/neighbors/detail/cagra/cagra_build.cuh)
- [NN-Descent convergence and host updates](https://github.com/AMD-Ecosystem/hipVS/blob/f96a663d0a4441d6f5344bd3ef552b71ae7b370b/cpp/src/neighbors/detail/nn_descent.cuh)

NN-Descent already checks convergence after a minimum number of iterations.
Reducing its maximum from 100 does not establish proportional build savings
and risks changing graph quality.

## Local verification

With installed MinGW GCC on Windows:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread \
  -DCUVS_WORKER_STANDALONE_TEST -Isrc/oblib \
  unittest/oblib/vector/test_cuvs_worker.cpp -o test_cuvs_worker.exe
./test_cuvs_worker.exe
```

Passed: 4096 repeated calls retain one thread; results are complete before
return; eight concurrent callers finish 8000 serialized jobs; shutdown joins
the worker. Without the standalone macro these tests register with gtest.

One CPU-only diagnostic measured 4096 empty jobs at 1439.467 ms with fresh
pthread create/join and 24.529 ms with the reusable worker. These are Windows
host dispatch measurements, not GPU or SQL QPS measurements.

The bridge fault-injection test passed with the new algorithm selection and
unchanged graph sizes. Its CPU shim tests allocation/error ownership, not
real iterative construction, GPU ABI compatibility, or recall.

## Next evidence required

There is no usable AMD GPU/ROCm server in this session. Full seekdb/hipVS
compilation, GPU execution, recall, build acceleration, QPS, and VRAM remain
unverified for this candidate. Identical graph sizes/search parameters do
not guarantee unchanged quality after changing the construction algorithm.

Run the same official evaluator and require every original gate, especially
baseline-relative recall within 0.005, distinct legal IDs, per-dataset query
ratio at least 0.90, and relative/physical VRAM limits. Keep `556065c62` as
the previously gate-passing control. Intermediate commit `d33813a08` contains
only worker reuse with NN-Descent; it can isolate algorithm regressions.

Full marks require actual official build/query targets and all gates.
Neither CPU diagnostics nor source inspection proves full marks. The next
official result must determine whether to retain the new builder.
