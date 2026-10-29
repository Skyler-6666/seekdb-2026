# Bounded GPU graph construction experiment

Branch: `contest/hipvs-bounded-gpu-graph`. The original revision `4836e6722`
failed the 250K synthetic diagnostic; the revised shared-hash implementation
still needs GPU validation. This is not a submission recommendation. The submission branch retains
NN-Descent. The user relayed permission for custom GPU construction; the full
updated organizer document is still pending.

The built-in IVF-PQ experiment measured 0.360155x ready speed and 24.474601x
sampled peak VRAM on the template 10K data. This implementation instead uses
the fixed library's IVF-PQ C API to select up to 2048 candidates per point,
in batches of at most 512 queries. A HIP kernel computes original FP32 L2
distances by reading the input vectors directly, excludes self, and keeps
256 neighbors. It does not materialize per-query copies of candidate vectors.

Construction then scores two-hop detours to prune to 128 edges, merges distinct
reverse edges, and includes two ring edges for connectivity. Each block builds
an exact shared-memory hash table mapping its 256 neighbors to their ranks.
It reads each neighbor-of-neighbor row once and counts only detours through
an earlier rank. The former global Bloom filter and repeated remote scans are
removed. CPU graph fixtures check exact counts against the original definition.

Reverse edges are collected in 128 rank-ordered launches, following the fixed
optimizer's priority ordering. This prevents lower-priority edges from filling
a destination's capped list before higher-priority edges. Arrival order within
one rank can vary between runs. Pruning/merging remain a custom implementation;
query quality must be measured rather than assumed equivalent to hipVS.
Inputs of at most 128 points use the fixed NN-Descent builder.

The candidate ID/distance buffers total at most 12 MiB. Additional buffers are
the original GPU FP32 dataset, 128-edge output graph, 256-edge intermediate
graph. The intermediate graph is allocated after IVF-PQ training synchronizes,
and the retained graph is allocated after IVF-PQ candidate selection is released.
The intermediate graph is released before allocating reverse edges. This avoids
overlapping graph buffers with training workspace. IVF-PQ training/search workspace is
also present: these buffer bounds do not prove the relative VRAM gate. Measure
actual peak VRAM against the paired baseline on both dataset dimensions.

The fixed `cuvsCagraIndexFromArgs` wraps the GPU graph and original dataset.
The bridge retains both allocations until after index destruction because the
fixed implementation can store non-owning views. Snapshot reload owns its
arrays inside the fixed library. Search still calls `cuvsCagraSearch` with
the V5 parameters and returns its IDs and distances directly. No query-time
reranker or alternative search is added, and no hipVS library file is modified.

Build requires only the existing HIP compiler and installed hipVS C headers
and library. No RAFT/RMM C++ headers or online dependency download is needed:

```bash
bash tools/hipvs_bridge/build.sh /workspace/seekdb-amd-bounded/libseekdb_cuvs_bridge.so
```

Local CPU checks cover the block-sort compare-exchange schedule with ties and
sentinels, collision chains, bounded missing-key lookups, exact detour-count
equivalence at both 128 and 256 neighbors, reverse-edge deduplication, ring preservation, bridge allocation
failures, retained graph lifetime, query cache growth failures, and cleanup.
They do not compile HIP kernels or establish GPU correctness, recall, speed,
or official acceptance. The original revision compiled and passed persistence
smoke testing on the user's GPU; the shared-hash revision must repeat these.
Actual HIP compilation, persistence smoke testing,
paired 10K timing/VRAM, 250K x 128/256 recall, and full SQL evaluation remain
required before promotion.

Run the algorithm check with a C++17 compiler:

```bash
g++ -std=c++17 -Wall -Wextra -Werror \
  unittest/oblib/vector/test_cuvs_graph_algorithms.cpp -o /tmp/test_cuvs_graph
/tmp/test_cuvs_graph
```

Bridge fault-injection commands are in `unittest/oblib/vector/hipvs_bridge_mock/README.md`.

Set `SEEKDB_CUVS_BUILD_TRACE=1` only for diagnosis. It prints accumulated
milliseconds for IVF-PQ training, candidate lookup, FP32 refinement, pruning,
reverse/merge, and index wrapping. It synchronizes after every timed stage,
including each lookup/refinement batch. These trace runs change scheduling
and are not clean performance evidence. Unset the variable for paired speed
measurements and formal evaluation; no algorithm parameters depend on it.
