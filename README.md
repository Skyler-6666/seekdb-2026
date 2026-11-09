# MiniVectorDB

A small, single-process SQL + vector database for the OceanBase 2026 add-on
round.  It is implemented from scratch: a SQL parser, a row store with
atomic snapshot persistence, transactions with rollback, exact L2 top-k
retrieval, and a MySQL 4.1 protocol server over stdin/stdout.

## Build

```bash
docker build -t minivectordb:latest .
```

## Run

```bash
docker run --rm -i \
  -v minivectordb-data:/data \
  --cpus=2 --memory=2g \
  minivectordb:latest
```

The process speaks the MySQL wire protocol on stdin/stdout.  It does not open
a TCP port.  The data directory defaults to `/data` and can be changed with
`--data-dir`.

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

Supported: `CREATE TABLE`, `DROP TABLE`, `CREATE VECTOR INDEX`, multi-row
`INSERT`, `SELECT` with `WHERE`/`AND`/`OR`, comparisons, `ORDER BY`, `LIMIT`,
`UPDATE`, `DELETE`, `BEGIN`/`COMMIT`/`ROLLBACK`, and `SHOW TABLES`.
`L2_DISTANCE`, `COSINE_DISTANCE`, and `INNER_PRODUCT` are available.

## Persistence

Every committed mutation writes a complete snapshot to
`<data-dir>/minivectordb.json` using a temporary file, `fsync`, and atomic
`rename`.  A restart reloads the last committed snapshot.  Uncommitted
transaction state exists only in memory and is discarded after a crash.

## Local smoke test

```bash
python3 -m minivectordb.main --data-dir /tmp/mvdb --protocol text
```

Then type SQL terminated by `;`.  `MINIVECTORDB_PROTOCOL=mysql` (the default)
selects the MySQL wire protocol used by the evaluator.
