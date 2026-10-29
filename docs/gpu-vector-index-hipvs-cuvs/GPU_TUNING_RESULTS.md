# GPU tuning evidence

These measurements were supplied by the user from their Radeon Cloud instance
on 2026-10-29. The assistant did not directly control the instance. The reported
software is ROCm 7.2.4, with 16 CPU cores of cgroup quota and 55 GiB memory.
Both bridge versions link to the same installed hipVS library. Its exact source
commit has not been independently verified on the instance.

## Iterative construction withdrawn

Baseline source: `634888d9b037153fa143d2c913ba4b25503e1d62`.
Candidate bridge: `d0eb7ee73483d0190f486bcd8c7deb2562ec01c8`.
Diagnostic tool: `3bac581000cda4365acc987af1c27e9a887608ab`.

The test alternated baseline/candidate for three rounds. Its ready timer covers
bridge build, snapshot serialize, original-handle release, reload, and first
query. It excludes SQL and DDL. Recall uses CPU exact Top-100. These are
development measurements, not official scores or gate results.

| Dataset | Queries | V5 median ready | Iterative median ready | V5 / iterative |
| --- | --- | --- | --- | --- |
| Template demo: 10K x 128 | 100 | 0.824028 s | 1.824523 s | 0.451641x |
| Synthetic Gaussian: 250K x 128 | 256 | 16.863077 s | 41.048169 s | 0.410812x |

For 250K x 128, raw ready seconds were:

```text
V5:        16.863077212125063, 17.111781511455774, 15.981730617582798
Iterative: 41.09958801232278, 40.941389152780175, 41.04816926829517
```

Median Recall@10 was 0.990625 for V5 and 0.995703125 for iterative; median
Recall@100 was 0.966875 and 0.9700390625. Both versions produced valid IDs in
all reported runs. Snapshot size was identical at 256001713 bytes. Query bridge
QPS remained close: approximately 349 versus 352 at Top-10 and 348 versus 350
at Top-100. The observed recall improvement did not offset a 2.43x construction
slowdown. This does not establish any result on SIFT250K or Hidden250K.

V5's build call alone took a median 16.347218 s; snapshot serialization plus
free/reload was about 0.5 s. This evidence directs further work toward graph
construction rather than snapshot I/O or small scratch allocations.

The submission branch restores V5 NN-Descent with graph degree 128,
intermediate degree 256, and iteration cap 100. It retains the reusable
large-stack worker and scratch buffers. The worker still needs full SQL/GPU
validation; the bridge diagnostic cannot measure its observer integration.
No full-score claim is made. New construction experiments must remain separate
from the submission branch until measured on both 128- and 256-dimensional
data, then checked by the official evaluator.

## Built-in IVF-PQ refinement withdrawn

The user supplied a two-round alternating diagnostic on the template 10K x 128
data with 100 queries. Both versions had Recall@10 and Recall@100 of 1.0.

| Version | Ready seconds | Sampled peak VRAM bytes |
| --- | --- | --- |
| V5 round 1 | 1.176538665 | 485339136 |
| IVF-PQ round 1 | 2.764517145 | 13467451392 |
| IVF-PQ round 2 | 2.165080087 | 13470920704 |
| V5 round 2 | 0.598878495 | 550404096 |

The median ready ratio is 0.360155x. The ratio of maximum sampled peaks is
24.474601x, far above the 1.25x resource target. Sampling every 20 ms provides
a lower bound on peaks, not an official gate measurement. The runs shared one
process and later initial VRAM was about 0.5 GB; despite that limitation, this
is sufficient reason not to promote this path or upscale it to 250K.

Source inspection attributes the large allocation to the fixed GPU refinement
implementation's per-query candidate-vector lists. A separate bounded GPU
construction experiment avoids those lists; its performance remains unmeasured.
