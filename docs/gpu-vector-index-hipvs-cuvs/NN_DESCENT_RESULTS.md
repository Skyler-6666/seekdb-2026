# GPU NN-Descent construction results

This candidate retains the known-good server sources and builds the CAGRA graph
with the fixed hipVS NN-Descent C API. The returned neighbor graph is sorted by
exact FP32 L2 distance and reduced to 128 neighbors with the existing GPU prune
and reverse-edge kernels. Query execution still calls the fixed
`cuvsCagraSearch` implementation and returns its results.

## Measurements

The evidence was produced by the user on the Radeon Cloud instance with ROCm
7.2.4. The comparison baseline was the officially executable submission
`556065c62578673d6313f2fdd798cc8753cbb662`. Both dimensions used 250,000
synthetic Gaussian FP32 vectors, 256 queries, exact CPU Top-100 truth, three
alternating rounds, fresh processes, and no build trace.

| Dimension | Median ready: 556 / candidate | Speedup | Sampled VRAM ratio | Worst Recall@10 delta | Worst Recall@100 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 16.893496 / 13.706180 s | 1.232546x | 0.864924 | -0.001953 | -0.001016 |
| 256 | 17.174826 / 13.846913 s | 1.240336x | 1.002987 | -0.004297 | -0.002383 |

The candidate compiled, passed a 10K Graph persistence write/read and negative
shape checks, and kept query bridge QPS above 0.975 of the baseline in every
round. The 256-dimensional Recall@10 worst-round margin to the allowed 0.005
loss was only 0.000703125, so this synthetic result is not sufficient for
submission.

These measurements exclude SQL/DDL, official SIFT250K and Hidden250K, 4096
single-row queries, and the official profiler. The sampled VRAM ratio is a
screen, not evidence that the relative resource gate passes.

## Reproduce

Build the candidate bridge and compare it with V5 using the diagnostic in
[`BRIDGE_DIAGNOSTIC.md`](BRIDGE_DIAGNOSTIC.md). Repeat at 128 and 256
dimensions with build trace disabled:

```bash
python3 prepare_cuvs_gpu_diagnostic.py \
  --revision <candidate-full-commit> \
  --baseline v5 \
  --rounds 3 \
  --dim 128

python3 prepare_cuvs_gpu_diagnostic.py \
  --revision <candidate-full-commit> \
  --baseline v5 \
  --rounds 3 \
  --dim 256
```

Promotion still requires a full seekdb build, SQL/DDL validation with both
`lib=vsag` and `lib=cuvs`, 4096-query official evaluation, profiler resource
checks, and paired official SIFT/Hidden Recall results.
