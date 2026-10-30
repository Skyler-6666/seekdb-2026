# Asynchronous AI columns

This implementation extends the native MySQL parser, resolver, schema, DML,
transaction manager and background worker. It builds on AI function commit
`18207bd9ac16d63d03b0e71782974e2983f665ab`. This integration branch also includes
the file-SQL repair at `76b6be1a8006161287d0e8d4b9325e837c4ff683`.
AI-column strict read retains system-variable ID 10754; file SQL uses ID 10755.
Their parser node IDs are distinct as well.

After registering a completion model and endpoint with DBMS_AI_SERVICE:

```sql
CREATE TABLE docs (
  id BIGINT PRIMARY KEY,
  body TEXT,
  summary TEXT AI COLUMN (AI_COMPLETE('text_model', body))
    WITH (sync_mode='async', max_retries=2)
);
INSERT INTO docs(id, body) VALUES (1, 'Source text');
SHOW AI COLUMNS FROM docs;
ALTER TABLE docs RETRY AI COLUMN summary WHERE status='FAILED' AND id=1;
SET SESSION ai_column_strict_read=ON;
SELECT summary FROM docs;
```

AI result columns must be nullable physical non-key columns. AI_EMBED requires
a fixed-dimension VECTOR column; an explicit dimension must match. Expressions
may depend on ordinary columns, including config expressions, but not other AI
columns. SELECTs, user variables and nested model calls are rejected. Business
SQL cannot assign AI results or internal state columns. ALTER ADD and MODIFY
backfill existing rows using the new persisted definition version.

The schema stores the expression, dependency IDs, model, retry options, type
instruction, definition version and three hidden state/request/version columns.
Native generated row triggers invalidate results and persist input snapshots in
`oceanbase.__all_ai_column_task` in the business transaction. Model calls happen
only after tasks become visible to the worker. Rollback also rolls back tasks;
deleting a business row removes its pending tasks.

Four executor threads claim row/change groups, with lease renewal and startup
reconciliation in separate threads. Staged terminal results survive interruption.
Writeback locks the business row before the task rows and checks request IDs,
definition versions and ownership. It never inserts business rows. Terminal
results and READY/FAILED states are published in one transaction, and stale or
deleted-row results cannot restore older data. Native target-type conversion
failures are isolated in rolled-back probes and receive bounded retries without
publishing successful siblings early or invoking their models again. A process
failure after an upstream call but before durable staging may repeat that call.

PENDING and FAILED results read as NULL. READY may also contain SQL NULL, so state
is independent of value. Strict read checks every row of each referenced AI
column, before executing SELECT, UPDATE or DELETE. It uses the statement's
session transaction and checks cached plans again. Unquoted `status` in RETRY is
the AI status; backtick-quoted `status` is an ordinary business column.

Vector invalidation and writeback use native DML/index maintenance. A BEFORE
UPDATE trigger which changes a vector participates in vector-ID generation;
rollback restores both the result and index entries.

Resource limits are 16 AI columns per table, 16 MiB of retained input/config per
claimed group, 1 MiB per model result, and max_retries from 0 to 100. Oversized
input/results fail rather than being truncated and sent to a model.

## Regression commands

Build the normal seekdb release. Install Python `pymysql` for the regression
driver; models are local HTTP mocks and require no real credentials.

```sh
python3 unittest/sql/engine/basic/run_ai_column_native.py \
  --binary build_release/src/observer/seekdb \
  --base-dir /tmp/seekdb-ai-column-regression
```

The driver uses ports 2885 and 19121-19125, compares the five supplied mysqltest
scripts against their supplied result files, and checks mock assertions. It also
covers composite/string and internal primary keys, READY NULL, PL write
protection, own-transaction strict read, existing-row backfill, metadata IDs,
abrupt process termination/recovery, and FLOAT overflow with mixed READY/FAILED
publication and exact retry call counts. Use a dedicated test database directory.
Tests do not change the supplied mysqltest scripts, result files or mock handler.

Typed definition/result validation can also run without a database:

```sh
g++ -std=c++20 -DAI_COLUMN_STANDALONE_TEST -Isrc \
  src/share/ai_service/ai_column_definition.cpp \
  unittest/sql/resolver/test_ai_column_resolver.cpp -o test_ai_column
./test_ai_column
```

Parser and schema GTest cases are in `unittest/sql/parser` and
`unittest/share/schema`. `.github/workflows/ai-column-linux.yml` supplies the
Linux native build and SQL regression entry point. Local Windows native build,
SQL and recovery execution do not replace Linux compilation or judge acceptance.

## Combined submission regression

To check both tasks in the same executable and server instance, use a fresh
file-data directory under secure_file_priv:

```sh
python3 unittest/sql/engine/basic/run_ai_column_native.py \
  --binary build_release/src/observer/seekdb \
  --base-dir /tmp/seekdb-contest-regression \
  --file-data-dir /tmp/seekdb-contest-fixtures/public
```

This checks both system variables default to OFF before running the supplied
file-SQL cases, file-SQL edge cases, five AI-column cases and AI-column recovery
checks. It does not select a different executable between suites.
