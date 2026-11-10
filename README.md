# MiniVectorDB

A small single-process SQL + vector database for the OceanBase 2026 add-on
round. It is implemented from scratch: a SQL parser, a durable row store,
transactions with rollback, exact L2 top-k retrieval, and the official
JSON-lines stdin/stdout protocol.

## Run

```bash
python3 -m minivectordb.main --data-dir /data
```

The process prints exactly one JSON line:

```json
{"type":"ready"}
```

and then processes one JSON request per line. It does not open a port and
does not print SQL prompts or logs to stdout.

## Official protocol

Request:

```json
{"id":1,"op":"execute","sql":"SELECT id FROM docs ORDER BY id;","search_mode":"exact"}
```

Query response:

```json
{"id":1,"ok":true,"columns":["id"],"rows":[[1]]}
```

Write response:

```json
{"id":2,"ok":true,"affected_rows":1}
```

Failure:

```json
{"id":3,"ok":false,"error":{"code":"duplicate_key","message":"duplicate primary key"}}
```

Shutdown:

```json
{"id":4,"op":"shutdown"}
```

The response is `{"id":4,"ok":true}`; the process flushes it, rolls back any
open transaction and exits with status 0.

The official public package contains `STDIO_PROTOCOL.md`, `mysqltest/` and
`protocol_check/`. The implementation has been run against all **20 public
cases** with a Windows-compatible local runner; all 20 pass.

## SQL subset

```sql
CREATE TABLE docs (
  id BIGINT PRIMARY KEY,
  category VARCHAR(32),
  content VARCHAR(1024),
  embedding VECTOR(4)
);

INSERT INTO docs VALUES
  (1, 'database', '事务与恢复', '[1, 0, 0, 0]'),
  (2, 'database', '向量索引', '[0, 1, 0, 0]');

SELECT id, content,
       L2_DISTANCE(embedding, '[0.9, 0.1, 0, 0]') AS distance
FROM docs
WHERE category = 'database'
ORDER BY distance
LIMIT 2;

BEGIN;
UPDATE docs SET embedding = '[0.8, 0.2, 0, 0]' WHERE id = 1;
DELETE FROM docs WHERE id = 2;
ROLLBACK;
```

Supported:

- `CREATE TABLE`, `DROP TABLE`, `CREATE VECTOR INDEX`
- multi-row `INSERT`, `SELECT`, `UPDATE`, `DELETE`
- `WHERE`, `AND`, `OR`, comparisons
- `ORDER BY`, `LIMIT`, `SELECT *`
- `L2_DISTANCE`, `COSINE_DISTANCE`, `INNER_PRODUCT`
- `BEGIN` / `COMMIT` / `ROLLBACK`
- filtered exact TopK; equal distances sort by primary key

`VARCHAR` lengths count Unicode codepoints and comparisons use codepoint
order. Vectors are validated as finite float32 values and fixed dimensions.

## Persistence and recovery

Committed rows are durable in `/data`. Inserts use an append-only WAL with
`fsync`; schema changes, updates and deletes write an atomic snapshot.
Uncommitted transaction state exists only in memory. On startup the engine
replays the WAL (ignoring a torn final line) and then folds it into a fresh
snapshot. A normal shutdown rolls back any active transaction.

## Docker

```bash
docker build -t minivectordb:latest .
docker run --rm -i \
  -v minivectordb-data:/data \
  --cpus=2 --memory=2g \
  minivectordb:latest
```

The Dockerfile builds from the source in this commit; it does not copy a
local prebuilt binary. NumPy is used only as a vectorized math library for
exact L2 scans.

## Local verification

All 20 public cases (`sql_*`, `storage_*`, `transaction_*`, `vector_*`) pass
with this implementation using the official case files and a local JSONL
runner. The official `protocol_check` on POSIX additionally verifies the
ready/shutdown handshake in the same shape.


## Performance-path verification

After the public-case run, the exact scan was changed to reuse a cached float32
norm vector and compute distances with a matrix-vector product instead of
allocating a full `(rows, dimensions)` difference matrix for every query.

Local verification on the 10K x 128 shape, 1,000 TopK=10 queries (half filtered):

- previous local benchmark: `11.978 s` total
- optimized local benchmark: `9.885 s` total
- insert: `4.102 s`
- query: `5.783 s`

All 20 official public cases were rerun after this optimization and passed
20/20. This is a local benchmark, not an official ranking result.
