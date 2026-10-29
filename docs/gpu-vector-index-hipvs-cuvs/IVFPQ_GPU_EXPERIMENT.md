# Isolated IVF-PQ construction experiment

Branch: `contest/hipvs-ivfpq-experiment`. Do not promote or use this as the
official submission until real GPU quality, timing, and resource checks pass.
The submission branch `contest/hipvs-cagra-optimization` remains on NN-Descent.

This experiment changes only CAGRA's initial neighbor graph construction to
IVF-PQ with 8x candidate refinement, up to 32 probes, and FP32 lookup tables
and internal IVF-PQ distance accumulation. It retains graph degrees 128/256
and the V5 CAGRA search parameters (width 4, at least max(512, 4*k) candidates).
The probe count is bounded by the fixed hipVS default number of lists,
max(1, n / 2000), so small datasets do not request more probes than lists.

The original FP32 matrix is copied once into a GPU allocation retained by the
bridge handle. Passing a device dataset selects the fixed library's GPU
refinement path. The host-dataset IVF-PQ path can refine on CPU and is not used.
The fixed library chooses 256*8 neighbor candidates and refines them against
the original FP32 device data before optimizing the 128-edge CAGRA graph.
Quantized IVF-PQ is only a construction candidate selector; final query vectors
and distances use the original FP32 CAGRA index.

The bridge owns the input device allocation until index destruction, since an
aligned CAGRA dataset can be a non-owning view. It releases the index before
this allocation. Allocation or upload failure cleans up without returning a
partial handle. Deserialized indexes continue to own their dataset in cuVS;
they have no extra bridge-owned input allocation. IVF search parameters are
released after construction or on a failed build.

Local C11 fault-injection checks passed, including input allocation/upload
failure cleanup, retained input lifetime, query scratch reuse, failed scratch
growth, batch isolation, distance-copy failure, reload, and final cleanup.
These checks do not verify the real IVF-PQ implementation, ABI, quality, or
speed. The installed library on the user's GPU instance still needs to compile
and run this source.

The reference code was inspected at hipVS commit
`f96a663d0a4441d6f5344bd3ef552b71ae7b370b`, in cagra_c.cpp,
detail/cagra/cagra_build.cuh, and include/cuvs/neighbors/graph_build_types.hpp.
No hipVS library files are changed, and no dependency is added.

Compare against the same V5 library and inputs with the existing
benchmark_cuvs_bridge.py. Use an independent source/output directory and retain
the iterative results. Test 250K vectors at both dimensions 128 and 256. First
check legal IDs and recall changes within 0.005, then ready time and query
throughput. Synthetic inputs do not replace SIFT250K or Hidden250K.

Peak VRAM must be measured separately. The IVF-PQ builder has different large
temporary allocations and retains the FP32 input, so graph size alone does not
prove the 1.25x relative limit. This experiment may fail quality, performance,
or memory gates. No score or speedup is claimed before real measurement.
