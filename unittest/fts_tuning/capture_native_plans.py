"""Capture real EXPLAIN output for the four unchanged official MATCH queries.

Run after loading the official benchmark database, on each compiled native
version. This is plan evidence, not a benchmark, profiler or correctness pass.
Connection settings use FTS_AFTER_HOST/PORT/USER/PASSWORD by default.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from native_correctness import connection, jsonable
from official_benchmark import ROOT, verify_contract


def official_queries(source: str) -> list[tuple[str, str]]:
    queries = re.findall(r'run_query_bench "(query_[a-z]+)" \\\n\s*"([^"\n]+)"', source)
    names = [name for name, _ in queries]
    if names != ['query_cn', 'query_beng', 'query_mixed', 'query_limit']:
        raise ValueError('Review plan capture: expected exactly the four frozen official queries')
    if any(not sql.startswith('SELECT COUNT(*) FROM ') or not sql.endswith(';') for _, sql in queries):
        raise ValueError('Unexpected official statement shape')
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection-prefix', choices=('FTS_BEFORE', 'FTS_AFTER'), default='FTS_AFTER')
    parser.add_argument('--revision', required=True, help='Full source revision used to build this running instance')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-fA-F0-9]{40}', args.revision):
        parser.error('--revision requires a full 40-character commit')
    hashes = verify_contract()
    source = (ROOT / 'tools/benchmark/fts_large_bench.sh').read_text(encoding='utf-8')
    queries = official_queries(source)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'declared_revision': args.revision.lower(),
              'started_at': datetime.now(timezone.utc).isoformat(),
              'official_file_sha256': hashes, 'plans': [],
              'native_correctness_verified': False, 'performance_measured': False,
              'limitations': 'Verify the actual running executable and its hash separately. EXPLAIN is not a runtime profile and does not establish a speedup.'}
    handle = None
    try:
        handle, endpoint = connection(args.connection_prefix)
        record['endpoint'] = endpoint
        with handle.cursor() as cursor:
            cursor.execute('SELECT VERSION()')
            record['server_version'] = jsonable(cursor.fetchone()[0])
            cursor.execute('SET ob_query_timeout = 100000000000')
            cursor.execute('SET ob_trx_timeout = 100000000000')
            for name, sql in queries:
                cursor.execute('EXPLAIN ' + sql)
                record['plans'].append({'name': name, 'unchanged_query_sql': sql,
                                        'columns': [column[0] for column in cursor.description],
                                        'rows': jsonable(cursor.fetchall())})
        record['status'] = 'captured'
        return 0
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if handle is not None:
            handle.close()
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        (args.output / 'native-query-plans.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
