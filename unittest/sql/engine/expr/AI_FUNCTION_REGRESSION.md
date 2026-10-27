# AI function regression tests

Development base: `2026-competition`, commit `5bce9e06d1281347ad61344e4b01fab9158c58a2`.

## Run the local SQL checks

Use a dedicated seekdb instance on port 2883. The runner creates and uses the
`ai_contest_regression` database and test AI model/endpoint names.

```sh
python -m pip install pymysql sqlparse
python -B unittest/sql/engine/expr/mock_ai_service.py
```

In a second terminal:

```sh
python -B unittest/sql/engine/expr/run_ai_sql_regression.py --port 2883
python -B -m unittest discover -s unittest/sql/engine/expr -p test_mock_ai_service.py -v
```

The mock is a local deterministic protocol fixture. It never calls a real model.
It deliberately returns OpenAI embeddings in reverse index order and unsorted
rerank results. The original competition SQL and result files are unchanged.
On Windows the SQL runner changes the mock destination address from `0.0.0.0`
to `127.0.0.1`; all SQL operations still run in seekdb.

## Verified on Windows, 2026-10-27

- Full native RelWithDebInfo build succeeded with the bundled Clang 18 SDK.
- The competition enhancement SQL produced all 13 expected result sets.
- 131-row scans passed with NULLs, fixed and varying dimensions, filtering,
  LIMIT and reversed provider indexes. The largest observed request contained
  64 inputs; results matched their source rows across requests.
- Nested/repeated Prompt and a 300,000-character expansion passed. Invalid
  dimensions and invalid placeholder indexes returned SQL errors.
- 43-document string/object rerank calls preserved global indexes, original
  object IDs and stable ties across batches of 20.
- Wrong vector dimensions/elements, duplicate or missing results, malformed
  JSON, empty completion and HTTP failures returned SQL errors. A truncated
  first response retried successfully.
- The existing `ai_prompt.test` executed 101 statements with its 25 expected
  errors; successful Prompt outputs parsed as JSON. This execution checked
  success/error behavior, not a complete mysqltest golden-output comparison.
- All 7 Python mock self-tests passed.

The added C++ unit cases have not been executed: the repository documents its
CMake unit-test targets as Linux-only. Linux compilation, the official
mysqltest runner and platform scoring remain for competition evaluation.
