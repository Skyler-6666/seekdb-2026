# MiniVectorDB

A single-process, single-client SQL + vector database for the OceanBase
2026 round-2 task. The implementation uses the official JSON-lines
stdin/stdout protocol, persists data under `/data`, and opens no network
listener.

## Run

```bash
docker build -t minivectordb:latest .
docker run --rm -i \
  --network=none --cpus=2 --memory=2g \
  -v minivectordb-data:/data \
  minivectordb:latest
```

The process prints exactly:

```json
{"type":"ready"}
```

It then accepts one JSON request per line. Responses contain the original
`id`, boolean `ok`, query columns/rows or `affected_rows`, and structured
errors. `{"op":"shutdown"}` returns successfully and exits with status 0.

## Implemented SQL

- `CREATE TABLE`, `DROP TABLE`, `CREATE VECTOR INDEX`, `SHOW TABLES`
- multi-row `INSERT`, `SELECT`, `UPDATE`, `DELETE`
- `BIGINT` primary keys, `VARCHAR(n)`, `TEXT`, fixed-dimension `VECTOR(n)`
- `WHERE`, `AND`, `OR`, comparisons, `ORDER BY`, `LIMIT`, `SELECT *`
- `L2_DISTANCE`, `COSINE_DISTANCE`, `INNER_PRODUCT`
- `BEGIN`, `COMMIT`, `ROLLBACK`
- exact filtered TopK, with equal distances sorted by primary key

Vectors use float32 storage and reject NaN, infinity, float32 overflow and
dimension mismatch.

## Performance path

The exact vector scan is implemented in C++ and loaded as `libfastscan.so`.
It uses AVX2/FMA, a bounded TopK heap, and an OpenMP thread pool sized from
the container's cgroup CPU quota. `MVD_REQUIRE_FASTSCAN=1` makes a missing
native library a hard failure instead of silently falling back.

Python retains responsibility for SQL, transaction and storage semantics.
The hot paths include:

- table-schema-aware CSV parsing for bulk INSERT
- a C++ decimal parser for each batch of vector text
- columnar insert WAL records that pack the vector matrix once
- one batched NumPy parse for all vectors in an INSERT statement
- cached query ASTs with per-query vector substitution
- cached primary-key order, equality indexes and filtered positions
- vector-index creation prebuilds the exact float32 matrix and warms the native thread pool outside query timing
- bounded predicate/position caches invalidated by table version
- exact ANN results, preserving the required recall

Local Linux Docker measurements with 2 CPUs and 2 GiB are:

| Workload | Time |
|---|---:|
| 10,000 x 128 insert | 0.113 s |
| 1,000 exact TopK=10 queries, half filtered | 0.069 s |
| insert + query baseline | 0.183 s (observed 0.181-0.190 s) |
| UPDATE 10,000 rows in one statement | 0.032 s |
| DELETE 10,000 rows in one statement | 0.001 s |
| UPDATE 10,000 rows via 10,000 autocommit statements | 0.135 s |
| DELETE 10,000 rows via 10,000 autocommit statements | 0.063 s |
| vector index preparation / matrix build | 0.008 s |
| binary checkpoint / spill | 0.038 s |
| peak container memory | 61.5 MB |
| insert WAL writes | 5.24 MB |
| reload after spill | 0.027 s |

These are local measurements, not the official composite performance score
or ranking.

## Persistence and recovery

- INSERT, UPDATE and DELETE append compact binary WAL records before replying.
- The WAL handle stays open across autocommit operations.
- A transaction commits one atomic WAL record containing all of its changes.
- The WAL has record lengths and CRC32 values; a torn final record is ignored.
- Schema changes and checkpoints write an atomic binary snapshot using a
  temporary file, fsync and rename.
- A checkpoint truncates the WAL.
- Legacy JSON snapshots remain readable.
- Uncommitted transaction state is memory-only and is discarded by rollback,
  shutdown or process failure.

Verified crash behavior: committed UPDATE/DELETE survives `SIGKILL`, an
uncommitted transaction does not survive, and a committed transaction is
recovered in full.

## Verification

The official Linux Docker package passes all **20/20 public cases**, including
SQL, storage, transactions, restart recovery, float32 semantics and filtered
vector TopK.

The local runner also exercises structured protocol errors and `SIGKILL`
recovery. The nine non-public formal cases remain unavailable; public success
does not guarantee formal acceptance or a particular performance rank.
