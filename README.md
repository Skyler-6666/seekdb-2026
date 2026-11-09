# MiniVectorDB

A small, single-process SQL + vector database for the OceanBase 2026 add-on
round. It is implemented from scratch: SQL parsing, row storage, CRC-framed WAL
recovery, explicit transactions, exact L2 TopK retrieval, and the MySQL 4.1
wire protocol over stdin/stdout.

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

The default protocol is the MySQL 4.1 packet protocol over stdin/stdout. It does
not open a TCP port. The data directory defaults to `/data`.

For local SQL debugging, the optional text protocol can be selected:

```bash
python3 -m minivectordb.main --data-dir /tmp/mvdb --protocol text
```

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
`INSERT`, `SELECT` with `WHERE`/`AND`, comparisons, `ORDER BY`, `LIMIT`,
`UPDATE`, `DELETE`, `BEGIN`/`COMMIT`/`ROLLBACK`, and `SHOW TABLES`.
Exact L2 TopK is implemented directly; `ann` requests may use the exact path,
which preserves Recall and correctness.

## Persistence and transactions

Each committed DDL/DML statement is appended to `/data/wal.mvd` as a
length-prefixed, CRC32-protected frame and fsynced before success is returned.
An explicit transaction buffers its successful modification statements and
writes one complete frame at COMMIT. ROLLBACK restores the pre-BEGIN in-memory
snapshot and writes no frame. On startup, complete valid frames are replayed;
a torn or CRC-invalid tail is ignored, so an uncommitted or partially written
transaction cannot partially apply.

## Tests

```bash
python3 tests/test_engine.py
python3 tests/test_recovery.py
python3 tests/test_mysql_wire.py
python3 tests/test_mysql_client.py   # optional; needs mysql client
python3 tests/bench_engine.py
```

The Docker image installs NumPy for vectorized exact L2 scans while keeping all
storage and transaction logic in the submitted source. Docker was unavailable
on the local Windows host used for development, so the image itself still needs
a Linux build before final submission.

## Local validation snapshot (2026-11-09)

- `python3 tests/test_engine.py`: `engine test passed`
- `python3 tests/test_recovery.py`: `recovery test passed` (uncommitted rollback, committed reload, torn-tail ignore)
- `python3 tests/test_mysql_wire.py`: `mysql wire test passed`
- MySQL 8.0.46 client through a test-only local TCP relay: exit 0, returned `1  0` and `2  1.414213538`
- `python3 tests/bench_engine.py`: insert/query split ranged approximately `2.5/5.2 s`, with total `7.469-7.738 s` across four runs on 10K x 128 with 1,000 TopK queries.

These are local results, not the official competition run. Docker was not available
on the Windows development host.
