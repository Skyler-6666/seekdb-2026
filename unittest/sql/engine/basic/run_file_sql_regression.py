"""Run the published file SQL test through the native seekdb MySQL endpoint.

Requires pymysql. Prepare the published fixtures in --data-dir and configure
secure_file_priv to allow that directory. No external SQL engine is used.
"""
import argparse
import datetime
import json
from decimal import Decimal
from pathlib import Path
import re
import base64
import subprocess

import pymysql


def display(value):
    if value is None:
        return 'NULL'
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, float):
        return format(value, '.15g')
    return str(value)


def published(connection, root, directory):
    suite = root / 'tools/deploy/mysql_test/test_suite/file_sql'
    source = (suite / 't/file_sql_basic.test').read_text(encoding='utf-8')
    expected = (suite / 'r/mysql/file_sql_basic.result').read_text(encoding='utf-8').splitlines()
    actual = []
    sql = []
    errors = (0,)
    result_log = True
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith('--echo '):
            actual.append(stripped[7:])
        elif stripped == '--disable_result_log':
            result_log = False
        elif stripped == '--enable_result_log':
            result_log = True
        elif stripped.startswith('--error '):
            errors = tuple(int(code) for code in stripped[8:].split(','))
        elif not stripped or stripped.startswith(('#', '--', 'let ')):
            continue
        else:
            sql.append(re.sub(r'^eval\s+', '', stripped).replace('$FILE_SQL_DATA_DIR', directory))
            if stripped.endswith(';'):
                statement = '\n'.join(sql)
                with connection.cursor() as cursor:
                    try:
                        cursor.execute(statement)
                        rows = cursor.fetchall()
                    except pymysql.MySQLError as error:
                        if error.args[0] not in errors:
                            raise AssertionError(statement) from error
                    else:
                        assert 0 in errors, f'expected error {errors}: {statement}'
                        if result_log and cursor.description:
                            actual.append('\t'.join(item[0] for item in cursor.description))
                            actual.extend('\t'.join(display(cell) for cell in row) for row in rows)
                sql.clear()
                errors = (0,)
    assert not sql, sql
    if actual != expected:
        import difflib
        raise AssertionError('\n'.join(difflib.unified_diff(expected, actual, fromfile='expected', tofile='native SQL')))
    print('PASS published file_sql_basic: every result row and column name matches the reference', flush=True)


def extra(connection, directory):
    def query(sql):
        with connection.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()
    def rejects(sql, codes=None):
        try:
            query(sql)
        except pymysql.MySQLError as error:
            if codes is not None:
                assert error.args[0] in codes, error
            assert query('SELECT 1') == ((1,),)
        else:
            raise AssertionError('expected SQL error: ' + sql)
    csv = f"read_csv('{directory}/basic.csv')"
    json = f"read_ndjson('{directory}/basic.jsonl')"
    parquet = f"read_parquet('{directory}/simple.parquet')"
    query('SET SESSION enable_file_sql=ON')
    assert query(f'SELECT a.id,b.name FROM {csv} a INNER JOIN {json} b ON a.id=b.id ORDER BY a.id') == ((1,'Alice'),(2,'Bob'),(3,'Chen'))
    assert query(f'SELECT a.id,b.c1 FROM {csv} a LEFT JOIN {parquet} b ON a.id=b.c1 AND b.c1<3 ORDER BY a.id') == ((1,1),(2,2),(3,None))
    assert query(f'SELECT active,count(*),sum(id) FROM {csv} GROUP BY active HAVING count(*)>1') == ((1,2,Decimal(4)),)
    assert query(f'SELECT id*2 AS twice FROM {csv} WHERE score>90 ORDER BY id DESC LIMIT 1') == ((6,),)
    assert query(f'SELECT count(*) FROM {csv} a JOIN {json} b ON a.id=b.id JOIN {parquet} c ON c.c1=a.id') == ((3,),)
    plan = query(f'EXPLAIN SELECT a.id FROM {csv} a JOIN {parquet} b ON a.id=b.c1')
    assert plan and 'JOIN' in str(plan).upper(), plan
    rejects(f"SELECT * FROM read_csv('{directory}/../does-not-exist.csv')")
    rejects(f"SELECT * FROM read_csv('{directory}/basic.csv',unknown_option=1)")
    rejects(f"SELECT * FROM read_csv('{directory}/basic.csv',header=true,header=false)")
    rejects(f"COPY (SELECT id FROM {csv}) TO '{directory}/export.csv' (HEADER)")
    query('SET SESSION ob_query_timeout=100000')
    rejects(f"COPY (SELECT id,SLEEP(0.2) AS pause FROM {csv}) TO '{directory}/failed-copy.csv' (HEADER)", (4012,))
    query('SET SESSION ob_query_timeout=10000000')
    local_directory = Path(directory.replace("''", "'"))
    assert not (local_directory / 'failed-copy.csv').exists()
    assert not list(local_directory.glob('failed-copy.csv.tmp-*'))
    query('DROP TABLE IF EXISTS copy_source')
    query('CREATE TABLE copy_source(id INT, txt LONGTEXT)')
    query("INSERT INTO copy_source VALUES (1,'你好,\"quote\"'),(2,NULL)")
    query(f"COPY copy_source TO '{directory}/table-copy.csv' (header, nullstr 'NULL')")
    assert query(f"SELECT * FROM read_csv('{directory}/table-copy.csv', nullstr='NULL') ORDER BY id") == ((1,'你好,"quote"'),(2,None))
    query('DROP TABLE copy_source')
    query(f"COPY (SELECT 1 AS id UNION ALL SELECT 2 ORDER BY id DESC) TO '{directory}/union-copy.csv' (FORMAT CSV, HEADER TRUE)")
    assert query(f"SELECT * FROM read_csv('{directory}/union-copy.csv') ORDER BY id DESC") == ((2,),(1,))
    import csv as csv_module
    with (local_directory / 'union-copy.csv').open(encoding='utf-8', newline='') as stream:
        assert list(csv_module.reader(stream)) == [['id'],['2'],['1']]
    query(f"COPY (SELECT 'quoted;value' AS txt,'NULL' AS literal_null,NULL AS missing) TO '{directory}/options-copy.csv' (FORMAT CSV, HEADER, DELIMITER ';', QUOTE '\"', ESCAPE '\"', NULL 'NULL')")
    assert query(f"SELECT * FROM read_csv('{directory}/options-copy.csv',delimiter=';',nullstr='NULL')") == (('quoted;value','NULL',None),)
    query(f"COPY (SELECT id FROM {csv} WHERE FALSE) TO '{directory}/empty-copy.csv' (HEADER)")
    assert query(f"SELECT count(*) FROM read_csv('{directory}/empty-copy.csv')") == ((0,),)
    rejects(f"COPY (SELECT 1 INTO OUTFILE '{directory}/forbidden-inner.csv') TO '{directory}/forbidden-outer.csv'")
    assert not (local_directory / 'forbidden-inner.csv').exists()
    assert not (local_directory / 'forbidden-outer.csv').exists()
    query('DROP TABLE IF EXISTS read_csv')
    query('CREATE TABLE read_csv(glob INT, read_parquet VARCHAR(20))')
    query("INSERT INTO read_csv VALUES(7,'ordinary SQL')")
    assert query('SELECT glob,read_parquet FROM read_csv') == ((7,'ordinary SQL'),)
    query('DROP TABLE read_csv')
    assert query("SELECT j.x FROM JSON_TABLE('[1,2]', '$[*]' COLUMNS(x BIGINT PATH '$')) j") == ((1,),(2,))
    assert query('EXPLAIN SELECT 1')
    assert query(f'SELECT /*+ parallel(2) */ sum(c1) FROM {parquet}') == ((Decimal(21),),)
    assert query(f'SELECT /*+ USE_NL(b) LEADING(a b) */ a.id,b.c1 FROM {csv} a JOIN {parquet} b ON a.id=b.c1 WHERE a.id>1 ORDER BY a.id') == ((2,2),(3,3))
    changing = local_directory / 'schema-change.csv'
    changing.write_text('id\n1\n', encoding='utf-8')
    describe = f"DESCRIBE SELECT * FROM read_csv('{directory}/schema-change.csv')"
    scan = f"SELECT * FROM read_csv('{directory}/schema-change.csv')"
    assert [row[0] for row in query(describe)] == ['id']
    assert query(scan) == ((1,),)
    changing.write_text('id,name\n2,updated\n', encoding='utf-8')
    assert [row[0] for row in query(describe)] == ['id','name']
    assert query(scan) == ((2,'updated'),)
    query('SET SESSION enable_file_sql=OFF')
    rejects(f'SELECT count(*) FROM {csv}', (1235,))
    rejects(f"COPY (SELECT 1) TO '{directory}/disabled.csv'", (1235,))
    print('PASS native expressions, WHERE/ORDER/LIMIT, GROUP/HAVING, INNER/LEFT/cross-format JOIN, EXPLAIN, invalid options, COPY table/LOB/NULL and no-overwrite', flush=True)
    print('PASS COPY UNION/order, custom CSV options, empty query, nested-INTO rejection, ordinary SQL identifiers, JSON_TABLE and parallel/nested-loop plans', flush=True)
    print('PASS unchanged SQL rebinds updated file rows and DESCRIBE schema', flush=True)


def parquet_cases(connection, directory):
    cases = json.loads((directory / 'expected.json').read_text(encoding='utf-8'))
    with connection.cursor() as cursor:
        cursor.execute('SET SESSION enable_file_sql=ON')
        for name, expected in cases.items():
            path = (directory / name).resolve().as_posix().replace("'", "''")
            cursor.execute(f"SELECT * FROM read_parquet('{path}')")
            rows = cursor.fetchall()
            assert len(rows) == len(expected), name
            for actual, reference in zip(rows, expected):
                assert [display(value) for value in actual] == [display(value) for value in reference], (name, actual, reference)
            cursor.execute(f"SELECT count(*) FROM read_parquet('{path}')")
            assert cursor.fetchall() == ((len(expected),),), name
        amplifier = (directory / 'dictionary-expansion.parquet').resolve().as_posix().replace("'", "''")
        cursor.execute(f"SELECT count(*) FROM read_parquet('{amplifier}')")
        assert cursor.fetchall() == ((1000,),)
        try:
            cursor.execute(f"SELECT * FROM read_parquet('{amplifier}')")
            cursor.fetchall()
        except pymysql.MySQLError as error:
            assert error.args[0] == 1210, error
        else:
            raise AssertionError('dictionary expansion was not bounded')
        cursor.execute('SELECT 1')
        assert cursor.fetchall() == ((1,),)
    print(f'PASS native SQL against all {len(cases)} Parquet codec/encoding fixtures; every value matches its source row', flush=True)
    print('PASS Parquet COUNT column pruning and safe rejection of an 8 GiB dictionary expansion', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=2883)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--parquet-fixtures', type=Path)
    parser.add_argument('--prepare-data', action='store_true', help='create a new data directory from the published fixture Git blobs')
    parser.add_argument('--git', default='git', help='Git executable used with --prepare-data')
    args = parser.parse_args()
    directory = Path(args.data_dir).resolve().as_posix().replace("'", "''")
    root = Path(__file__).resolve().parents[4]
    if args.prepare_data:
        destination = Path(args.data_dir).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        source = 'tools/deploy/mysql_test/test_suite/file_sql/data/'
        for name in ('basic.csv','basic.jsonl','invalid.txt','simple.parquet.b64'):
            blob = subprocess.check_output([args.git, 'show', 'HEAD:' + source + name], cwd=root)
            if name.endswith('.b64'):
                (destination / 'simple.parquet').write_bytes(base64.b64decode(blob))
            else:
                (destination / name).write_bytes(blob)
    connection = pymysql.connect(host='127.0.0.1', port=args.port, user='root', charset='utf8mb4',autocommit=True,read_timeout=60)
    try:
        with connection.cursor() as cursor:
            cursor.execute('CREATE DATABASE IF NOT EXISTS file_sql_regression')
            cursor.execute('USE file_sql_regression')
        published(connection, root, directory)
        extra(connection, directory)
        if args.parquet_fixtures is not None:
            parquet_cases(connection, args.parquet_fixtures)
    finally:
        connection.close()
