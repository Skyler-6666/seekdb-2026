# File SQL implementation and regression checks

The `file-sql-enhancement-2026` branch inherits the AI function changes and
the cold-start package-loading fix from `ai-function-enhancement-2026`.

## Execution and safety

`read_csv`, `read_json`/`read_ndjson`, `read_parquet` and `glob` are SQL table
functions. Resolution builds native column expressions; a file-table mode in
seekdb's table-function operator supplies rows to native filters, aggregates,
sorts, limits and joins. `DESCRIBE SELECT` resolves the source's column metadata.
`COPY` uses the native SELECT INTO operator and consumes its child query rows.
No external SQL engine executes any query.

The new session/global variable `enable_file_sql` defaults to OFF. Every file
entry point also checks the current global `secure_file_priv`. Empty/NULL roots
are denied. Canonical paths must remain below the configured directory; opened
input identity, size and modification timestamps are checked after inference,
at bounded scan-batch intervals, on rescan and at end of input. Schema inference
and execution honor SQL cancellation. Input handles are read-only.

CSV/JSONL schemas use a full streaming pass; `sample_size` only accepts -1.
Execution reopens the checked file and retains one record. Parquet reads footer
schema and row-group metadata, then buffers bounded pages only for selected
columns. COUNT without file columns avoids data-page decoding. Parquet supports
flat primitive values, PLAIN, dictionary/RLE, delta integer/string encodings,
byte-stream split, uncompressed and Snappy pages, and optional page CRC checks.
Other compression codecs and nested schemas return a SQL error.

File statements disable reuse of cached physical plans through seekdb's native
plan-cache policy: files have no catalog schema version, so each statement must
rebind its current columns, including DESCRIBE after a file changes.

Resource limits include 16 MiB per CSV/JSONL record or Parquet page, 1024 columns,
10000 glob matches, an 8 MiB Parquet footer, bounded metadata nesting/node counts,
262144 values per page and 256 MiB of projected page buffers. Dictionary and
prefix expansion are checked before retaining unbounded decoded values.

COPY writes a same-directory temporary CSV and atomically publishes it with a
no-overwrite hard link only after the child query succeeds. Failed, canceled or
uncommitted writes remove their temporary output. Header, delimiter, quote,
escape and NULL-text options support CSV readback; ordinary SELECT INTO behavior
continues through its existing code path.

## Run the native SQL regression

Use a dedicated local seekdb instance on port 2883. Configure `secure_file_priv`
to an existing test root, for example at first startup:

```sh
seekdb --nodaemon --base-dir /tmp/seekdb-file-sql --port 2883 \
  --variable secure_file_priv=/tmp/file-sql-test-root
python -m pip install pymysql
python -B unittest/sql/engine/basic/run_file_sql_regression.py \
  --prepare-data --data-dir /tmp/file-sql-test-root/public-case
```

`--prepare-data` requires Git and a new directory. It uses the original Git
fixture bytes, avoiding checkout newline conversion. A successful run leaves
the fixture and exported files for inspection; use a fresh directory on rerun.
The SQL runner uses the dedicated `file_sql_regression` database and test tables.

Optional Parquet compatibility fixtures require PyArrow solely for writing test
inputs and recording expected values. All queries still execute in seekdb:

```sh
python -m pip install pyarrow
python -B unittest/sql/engine/basic/generate_file_sql_fixtures.py \
  --data-dir /tmp/file-sql-test-root/parquet-cases
python -B unittest/sql/engine/basic/run_file_sql_regression.py \
  --prepare-data --data-dir /tmp/file-sql-test-root/another-public-case \
  --parquet-fixtures /tmp/file-sql-test-root/parquet-cases
```

## C++ checks

`test_file_scan_utils.cpp` can run under GTest or as a standalone program. From
the repository root, compile its pure file modules with a C++20 compiler and
RapidJSON headers, for example:

```sh
c++ -std=c++20 -DFILE_SQL_STANDALONE_TEST -Isrc -I/path/to/rapidjson/include \
  src/sql/engine/basic/file_scan_utils.cpp \
  src/sql/engine/basic/file_parquet_reader.cpp \
  src/sql/engine/basic/file_copy_writer.cpp \
  unittest/sql/engine/basic/test_file_scan_utils.cpp -o file_scan_test
./file_scan_test
```

The parser GTest cases are in `unittest/sql/parser/test_file_sql_parser.cpp`.

## Local verification

Verified on Windows on 2026-10-28.

- Native Windows RelWithDebInfo build passed with the bundled Clang 18 SDK.
- Published `file_sql_basic.test` SQL results matched the original result file,
  including every returned row and column name. Published tests were unchanged.
- Native expressions, filtering, order/limit, grouping/having, INNER/LEFT and
  cross-format joins, EXPLAIN, COPY table/query/LOB/NULL and no-overwrite passed.
- COPY UNION ordering, custom CSV options, empty-query headers and nested-INTO
  rejection passed. Ordinary identifiers, JSON_TABLE, parallel and nested-loop
  query plans passed. Repeated SQL and DESCRIBE refreshed changed file schemas.
- Twelve Parquet fixtures covered V1/V2 pages, multiple row groups, nullable
  primitives, signed integer boundaries, dates/microseconds, dictionary,
  delta and byte-stream-split encodings with NONE/Snappy compression.
- C++ standalone tests passed for CSV/JSONL inference and streaming, rescan,
  invalid records, schema union, file modification/replacement, paths, glob,
  published Parquet data, projection and COPY roundtrip/abort cleanup.
- Existing outside-root inputs and outside-root COPY output were denied.
  A timed-out COPY left no target or temporary output; SELECT 1 still worked.
- Windows directory-junction escapes were denied for read_csv, glob and COPY;
  the outside input remained unchanged and no outside output was created.
- 100000-row inference and native aggregation passed locally in about 2.6 s.
  A 10 ms schema-inference deadline returned error 4012 promptly and recovered.
- COUNT column pruning passed on a small dictionary file that would expand to
  8 GiB; selecting its data was safely rejected at the buffer limit.
- The final executable also passed the AI SQL regression (13 published result
  sets and row-mapping/error cases) and all seven mock-service self-tests.

The official Linux mysqltest executable, parser GTest target and competition
platform have not been run here. Local native SQL result comparison does not
claim a platform score or a successful platform submission.
