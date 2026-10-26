# FTS optimization measurements, 2026 competition

This change reduces repeated tokenizer setup and word aggregation work, and bounds collision chains in TaaT retrieval. It preserves token classification, token frequencies, stopword collation, relevance calculation and index format.

IK classification decodes each character once and passes the cached character information to its processors. Dictionary range wrappers use the parser allocator and a small inline array rather than three independent arenas per document. Parser cleanup clears owned pointers so partial initialization can be cleaned up repeatedly.

TOKENIZE applies built-in defaults directly when no property overrides are supplied, avoiding a JSON construction/serialization/parsing round trip. Explicit overrides retain their original validation. UTF-8 stopword checks use the existing dictionary collation without constructing a temporary charset-conversion allocator. Word counts update existing hash nodes in one lookup, retaining the last spelling of collation-equivalent tokens.

TaaT bucket allocation is deferred until each partition is consumed. Its actual posting count determines the bucket count, bounded by the existing 1,000-document partition target; score accumulation updates existing nodes in place. The standard short-query workload mainly uses DaaT, so this TaaT change is covered directly by a 302-token regression. Dense partitions can allocate up to 1,000 buckets each, with the existing maximum of 20 partitions. Sparse partitions retain small maps.

The unmodified official benchmark was run three times before and three times after, sequentially on the same host. Every run imported all 20,000 documents and built all three indexes (`SKIP_LOAD=0`). Each run used 3,000 tokenization rounds, 200 query rounds, three samples and 30 warmups. Arithmetic means across complete runs are shown below; negative reductions are retained. Build metrics are seconds and tokenization/query metrics are milliseconds.

| Metric | Before mean | After mean | Time reduction |
| --- | ---: | ---: | ---: |
| `build_ik_all_sec` | 12.429000 | 12.596333 | -1.35% |
| `build_ik_content_sec` | 9.512000 | 9.907333 | -4.16% |
| `build_beng_en_sec` | 5.201333 | 5.106667 | 1.82% |
| `tokenize_ik_avg_ms` | 0.246267 | 0.236400 | 4.01% |
| `tokenize_beng_avg_ms` | 0.191700 | 0.176567 | 7.89% |
| `query_cn_avg_ms` | 3.024433 | 3.087500 | -2.09% |
| `query_beng_avg_ms` | 4.473400 | 4.433533 | 0.89% |
| `query_mixed_avg_ms` | 3.287700 | 3.256233 | 0.96% |
| `query_limit_avg_ms` | 3.692567 | 3.648000 | 1.21% |

Category mean reductions are build: -1.23%, tokenize: 5.95%, query: 0.24%. Their equal-weight mean is **1.66%**. Tokenization improved modestly; build and short-query results do not establish a stable improvement. These are local measurements, not a competition score or a claim of full marks. The original CI reference score is hardware dependent and was not used to infer official points.

All six complete runs returned the same query hit counts: Chinese 8,001; English 11,000; mixed 7,332; LIMIT 20. Counts supplement the correctness regressions below.

The baseline binary was built from `5bce9e06d1281347ad61344e4b01fab9158c58a2` and preserved before editing. The measured candidate product commit is `71282989375e22024d485b2c622ab3dde05a725b`. Subsequent commits only add regression coverage and this report. The benchmark script runs from the candidate checkout, so its `git_head` field identifies that checkout even when the saved baseline binary is launched; binary provenance is recorded independently. [Machine-readable results](fts-performance-2026.json) include every report, the exact server settings, binary hashes, and unchanged benchmark asset hashes.

The host exposes 128 logical CPUs and 503 GiB RAM. The server used `memory_budget=8G`, `cpu_count=4`, `datafile_size=4G`, `datafile_maxsize=8G`, `log_disk_size=4G`. The CPU setting does not impose CPU affinity or a cgroup quota. Both versions used the repository's Release toolchain and `./build.sh release --make -j24`.

Release builds succeeded. Fifteen mysqltest cases passed: thirteen shipped FTS cases and the two added regressions. The full 17-case run returned a failure status because these two shipped cases fail identically on the original baseline and candidate under this harness:

- `basic_dml`, line 1598: the script reuses the old `data_table_id` after dropping and recreating a table, producing `WHERE table_id = No such row`.
- `partitioned_simple_query`: expected EXPLAIN parameter `:0` is emitted as `:1` in two lines. The original and candidate have the same mismatch.

The shipped expected files were not changed. Added expected results were recorded on the original binary, then compared with the optimized binary. `parser_hotpath` covers frequencies, Unicode character families, fullwidth text, connectors, IK modes, segment boundaries, empty inputs, collations and index DML. `query_merge_hotpath` covers 1,280 documents, overlapping and missing tokens, scores, top-k, prepared-plan reuse, rollback, sparse matches, and more than 256 distinct tokens selecting TaaT. The extended long-query case and both tokenizer cases passed again on `be767177813b437f128dc4d7a526957c8d6a1c9d`.

For an existing dedicated local instance, reproduce the official workload with its unchanged defaults:

```bash
./build.sh release --init --make -j24
MYSQL='mysql -h127.0.0.1 -P2881 -uroot -N -s --default-character-set=utf8mb4' \
MYSQL_VERBOSE='mysql -h127.0.0.1 -P2881 -uroot --default-character-set=utf8mb4' \
LABEL=before SKIP_LOAD=0 OUTPUT=/tmp/fts-before-1.txt \
  bash tools/benchmark/fts_large_bench.sh
```

Use a fresh report filename for each complete repetition and the same configuration for the candidate. The script recreates its `fts_large_bench` database. To run the new regression cases, initialize the test account with `tools/deploy/init_user.sql`, export `OBMYSQL_MS0=127.0.0.1`, `OBMYSQL_PORT=2881`, `OBMYSQL_PWD=admin`, and invoke the shipped `mysqltest` client with each new `.test` and matching `.result` file in `tools/deploy/mysql_test/test_suite/fts_index/`.

The system disallowed kernel perf events. A separate diagnostic baseline run used a process-local SIGPROF PC sampler; its timings were excluded from the comparison. It identified query decoding, BM25 evaluation and DaaT merge/filter work, rather than a demonstrated short-query TaaT bottleneck. Both original and candidate servers exceeded the harness SIGTERM deadline and were terminated after the SQL checks; normal shutdown and restart were not validated here.
