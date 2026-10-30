"""Exercise the first AI management CALL in a fresh native seekdb instance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import socket
import tempfile
import time

import pymysql


def check(args):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    directory = Path(tempfile.mkdtemp(prefix='ai-package-cold-', dir=args.data_parent))
    with (directory / 'console.log').open('wb') as console:
        process = subprocess.Popen([
            str(Path(args.seekdb).resolve()), '--nodaemon', '--base-dir', str(directory),
            '--port', str(args.port), '--parameter', 'memory_budget=2G',
            '--parameter', 'cpu_count=2', '--parameter', 'datafile_size=128M',
            '--parameter', 'datafile_maxsize=1G', '--parameter', 'log_disk_size=2G',
            '--log-level', 'INFO',
        ], stdout=console, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        connection = None
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'seekdb exited; logs: {directory}')
                try:
                    connection = pymysql.connect(host='127.0.0.1', port=args.port,
                        user='root', autocommit=True, connect_timeout=1, read_timeout=20)
                    break
                except pymysql.MySQLError:
                    time.sleep(0.1)
            if connection is None:
                raise RuntimeError(f'seekdb startup timed out; logs: {directory}')
            with connection.cursor() as cursor:
                cursor.execute('SET SESSION ob_query_timeout=%s', (args.timeout_us,))
                # Preserve the DDL and Prompt work preceding the judge's first CALL.
                for query in (
                    'CREATE DATABASE ai_package_cold_start',
                    'USE ai_package_cold_start',
                    "SELECT ai_prompt('{0}/{0}/{1}',json_array(1,true),json_object('k',2))",
                    'CREATE TABLE prompt_rows(t LONGTEXT,a TEXT)',
                    "INSERT INTO prompt_rows VALUES('{0}-ok','lob')",
                    'SELECT ai_prompt(t,a) FROM prompt_rows',
                    'DROP TABLE prompt_rows',
                ):
                    cursor.execute(query)
                    cursor.fetchall()
                start = time.monotonic()
                try:
                    cursor.execute("CALL DBMS_AI_SERVICE.DROP_AI_MODEL('cold_start_model')")
                except pymysql.MySQLError as error:
                    # A missing model is expected; timeout or any other error is a failure.
                    if error.args[0] != 11118:
                        raise
                else:
                    raise AssertionError('dropping a missing model unexpectedly succeeded')
                elapsed = round((time.monotonic() - start) * 1000, 2)
                config = json.dumps({'type': 'completion', 'model_name': 'cold_start_model'})
                cursor.execute('CALL DBMS_AI_SERVICE.CREATE_AI_MODEL(%s,%s)', ('cold_start_model', config))
                cursor.execute("CALL DBMS_AI_SERVICE.DROP_AI_MODEL('cold_start_model')")
                print(f'PASS first CALL {elapsed} ms; create/drop passed; logs: {directory}', flush=True)
        finally:
            if connection is not None:
                connection.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seekdb', required=True)
    parser.add_argument('--data-parent', required=True)
    parser.add_argument('--port', type=int, default=2884)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--timeout-us', type=int, default=3000000)
    args = parser.parse_args()
    for _ in range(args.attempts):
        check(args)
