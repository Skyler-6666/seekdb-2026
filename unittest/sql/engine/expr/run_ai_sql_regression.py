"""Run the contest SQL and row-mapping regressions against a local seekdb.

Requires pymysql and sqlparse. Start mock_ai_service.py on port 8000 first.
Uses the dedicated ai_contest_regression database, with no real model calls.
"""
import argparse
import json
import os
from pathlib import Path
import re
import urllib.request

import pymysql
import sqlparse
from mock_ai_service import embedding


def mock(path, body=None):
    request = urllib.request.Request('http://127.0.0.1:8000' + path)
    if body is not None:
        request.data = json.dumps(body).encode()
        request.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def run(connection, source):
    with connection.cursor() as cursor:
        cursor.execute(source)
        return cursor.fetchall()


def json_result(connection, source):
    result = run(connection, source)[0][0]
    return json.loads(result) if isinstance(result, str) else result


def official_test(connection, root):
    source = root / 'tools/deploy/mysql_test/test_suite/ai_function/t/ai_function_enhancement.test'
    lines = []
    for line in source.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped.startswith('--error '):
            lines.append('/* expect:' + stripped.split(' ', 1)[1] + ' */')
        elif stripped and not stripped.startswith(('#', '--')) and stripped != 'connection default;':
            lines.append(line)
    expected = [
        (({'args': [[1, True], {'k': 2}], 'template': '{0}/{0}/{1}'},),),
        (({'args': ['lob'], 'template': '{0}-ok'},),),
        ((1,),), ((1,),), ((768,),), ((768,),),
        ((1, 768), (2, None), (3, 768)), ((4,),), (('a', 2),),
        (('ollama',),), (('Ollama',),), ((1,),), ((1,),),
    ]
    select_index = 0
    for statement in sqlparse.split('\n'.join(lines)):
        allowed = re.match(r'/\* expect:([\d,]+) \*/\s*', statement)
        errors = [int(code) for code in allowed[1].split(',')] if allowed else [0]
        if allowed:
            statement = statement[allowed.end():]
        # Windows clients cannot connect to the wildcard bind address.
        if os.name == 'nt':
            statement = statement.replace('http://0.0.0.0:8000/', 'http://127.0.0.1:8000/')
        try:
            rows = run(connection, statement)
        except pymysql.MySQLError as error:
            if error.args[0] not in errors:
                print('FAIL SQL:', statement[:300], flush=True)
                raise
            continue
        assert 0 in errors, statement
        if statement.lower().startswith('select'):
            if select_index < 2:
                rows = tuple((json.loads(row[0]),) for row in rows)
            assert rows == expected[select_index], (statement, rows, expected[select_index])
            select_index += 1
    assert select_index == len(expected), select_index
    print('PASS: contest SQL, 13 result sets match reference expectations (Windows uses 127.0.0.1 for the mock)', flush=True)


def endpoint(connection, name, kind, path, provider='openai'):
    for procedure, suffix in (('DROP_AI_MODEL', ''), ('DROP_AI_MODEL_ENDPOINT', '_ep')):
        try:
            run(connection, "CALL DBMS_AI_SERVICE.%s('%s')" % (procedure, name + suffix))
        except pymysql.MySQLError as error:
            if error.args[0] not in (11118, 11112):
                raise
    model = json.dumps({'type': kind, 'model_name': 'regression-' + kind})
    config = json.dumps({'ai_model_name': name, 'scope': 'all', 'provider': provider,
                         'url': 'http://127.0.0.1:8000' + path,
                         'access_key': '' if provider.lower() == 'ollama' else 'mock-key'})
    with connection.cursor() as cursor:
        cursor.execute('CALL DBMS_AI_SERVICE.CREATE_AI_MODEL(%s,%s)', (name, model))
        cursor.execute('CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT(%s,%s)', (name + '_ep', config))


def error_case(connection, source):
    try:
        run(connection, source)
    except pymysql.MySQLError:
        assert run(connection, 'SELECT 1') == ((1,),)
    else:
        raise AssertionError('Expected a SQL error: ' + source)


def regressions(connection):
    endpoint(connection, 'qa_embed', 'dense_embedding', '/v1/embeddings')
    endpoint(connection, 'qa_complete', 'completion', '/v1/chat/completions')
    endpoint(connection, 'qa_rank', 'rerank', '/v1/rerank', 'siliconflow')
    run(connection, 'DROP TABLE IF EXISTS ai_rows')
    run(connection, 'CREATE TABLE ai_rows(id INT PRIMARY KEY, content LONGTEXT, dim INT)')
    rows = [(index, None if index % 7 == 0 else '行:%d\\"%s' % (index, 'x' * (index % 13)),
             5 if index % 9 < 4 else 7) for index in range(131)]
    with connection.cursor() as cursor:
        cursor.executemany('INSERT INTO ai_rows VALUES(%s,%s,%s)', rows)
    mock('/__reset', {})
    for predicate, fixed_dim in (('ORDER BY id', 5), ('ORDER BY id', None),
                                 ('WHERE id % 3 = 1 ORDER BY id LIMIT 29', None)):
        dimension_sql = str(fixed_dim) if fixed_dim else 'dim'
        result = run(connection, "SELECT id, ai_embed('qa_embed',content,%s) FROM ai_rows " % dimension_sql + predicate)
        expected_ids = [row[0] for row in rows if predicate.startswith('ORDER') or row[0] % 3 == 1]
        if 'LIMIT' in predicate:
            expected_ids = expected_ids[:29]
        assert [row[0] for row in result] == expected_ids
        for index, vector in result:
            original = rows[index]
            assert (None if vector is None else json.loads(vector)) == (
                None if original[1] is None else embedding(original[1], fixed_dim or original[2])), index
    requests = [record['body']['input'] for record in mock('/__stats')['requests']
                if record['path'] == '/v1/embeddings']
    assert any(len(inputs) > 1 for inputs in requests), 'No cross-row request observed'
    assert all(len(inputs) <= 64 for inputs in requests), 'Provider batch exceeds the bound'
    assert all(isinstance(value, str) for inputs in requests for value in inputs)
    print('PASS: 131 rows, NULL, per-row dimensions, filtering, LIMIT, reversed indexes; %d requests, largest batch %d' %
          (len(requests), max(map(len, requests))), flush=True)

    rendered = json_result(connection, "SELECT JSON_EXTRACT(AI_PROMPT('{0}/{0}',JSON_ARRAY(1,true)),'$.args')")
    assert rendered == [[1, True]]
    result = run(connection, "SELECT ai_complete('qa_complete',ai_prompt('{0}/{0}',ai_prompt('内:{0}','值')))")[0][0]
    assert result == 'mock:内:值/内:值', result
    result = run(connection, "SELECT length(ai_complete('qa_complete',ai_prompt('{0}{0}{0}',repeat('x',100000))))")[0][0]
    assert result == 300005, result
    for dimension in ('0', '-1', '1.5', 'NULL'):
        error_case(connection, "SELECT ai_embed('qa_embed','x',%s)" % dimension)
    for placeholder in ('{1}', '{-1}', '{184467440737095516160}'):
        error_case(connection, "SELECT ai_complete('qa_complete',ai_prompt('%s','x'))" % placeholder)

    documents = ['xx' if index % 3 else 'longer' for index in range(43)]
    with connection.cursor() as cursor:
        cursor.execute("SELECT ai_rerank('qa_rank','q',%s,NULL)", (json.dumps(documents),))
        ranked = json.loads(cursor.fetchone()[0])
        order = sorted(range(len(documents)), key=lambda index: -len(documents[index]))
        assert [item['index'] for item in ranked] == order
        objects = [{'id': 'doc:%d' % index, 'body': text} for index, text in enumerate(documents)]
        cursor.execute("SELECT ai_rerank('qa_rank','q',%s,'body')", (json.dumps(objects),))
        ranked = json.loads(cursor.fetchone()[0])
        assert [item['id'] for item in ranked] == ['doc:%d' % index for index in order]
    print('PASS: long/nested/repeated Prompt, invalid dimensions/indexes, 43-document stable global rerank mapping', flush=True)

    for scenario in ('wrong_dimension', 'non_numeric', 'duplicate_index', 'missing_result', 'invalid_json', 'http_500'):
        endpoint(connection, 'qa_bad', 'dense_embedding', '/v1/embeddings?scenario=' + scenario)
        error_case(connection, "SELECT ai_embed('qa_bad',content,dim) FROM ai_rows WHERE id IN (1,2)")
    endpoint(connection, 'qa_bad_complete', 'completion', '/v1/chat/completions?scenario=empty')
    error_case(connection, "SELECT ai_complete('qa_bad_complete','x')")
    endpoint(connection, 'qa_retry', 'dense_embedding', '/v1/embeddings?scenario=partial_once')
    assert json_result(connection, "SELECT ai_embed('qa_retry','retry',5)") == embedding('retry', 5)
    print('PASS: malformed responses and HTTP failures return SQL errors; partial-transfer retry recovers', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2883)
    args = parser.parse_args()
    connection = pymysql.connect(host='127.0.0.1', port=args.port, user='root',
                                 charset='utf8mb4', autocommit=True, read_timeout=120)
    try:
        run(connection, 'CREATE DATABASE IF NOT EXISTS ai_contest_regression')
        run(connection, 'USE ai_contest_regression')
        official_test(connection, Path(__file__).resolve().parents[4])
        regressions(connection)
    finally:
        connection.close()
