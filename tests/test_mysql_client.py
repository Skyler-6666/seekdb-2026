#!/usr/bin/env python3
"""Optional end-to-end test with a real MySQL 8 client through a local TCP relay."""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    mysql = shutil.which('mysql') or '/usr/bin/mysql'
    if not Path(mysql).exists():
        print('mysql client not installed; skipped')
        return
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ)
        env['MINIVECTORDB_PROTOCOL'] = 'mysql'
        db = subprocess.Popen(
            [sys.executable, '-m', 'minivectordb.main', '--data-dir', td],
            cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 0))
        port = server.getsockname()[1]
        server.listen(1)

        def relay():
            conn, _ = server.accept()
            def client_to_db():
                try:
                    while True:
                        data = conn.recv(65536)
                        if not data:
                            break
                        os.write(db.stdin.fileno(), data)
                finally:
                    try: db.stdin.close()
                    except Exception: pass
            def db_to_client():
                try:
                    while True:
                        data = os.read(db.stdout.fileno(), 65536)
                        if not data:
                            break
                        conn.sendall(data)
                except Exception:
                    pass
            threading.Thread(target=client_to_db, daemon=True).start()
            threading.Thread(target=db_to_client, daemon=True).start()

        threading.Thread(target=relay, daemon=True).start()
        sql = (
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(32), embedding VECTOR(4));"
            "INSERT INTO docs VALUES (1,'a','[1,0,0,0]'),(2,'a','[0,1,0,0]');"
            "SELECT id,L2_DISTANCE(embedding,'[1,0,0,0]') AS d "
            "FROM docs WHERE category='a' ORDER BY d LIMIT 2;"
        )
        try:
            cp = subprocess.run(
                [mysql, '--protocol=TCP', '-h127.0.0.1', f'-P{port}', '-uroot',
                 '--ssl-mode=DISABLED', '--default-character-set=utf8mb4', '-N', '-e', sql],
                capture_output=True, text=True, timeout=30,
            )
            assert cp.returncode == 0, cp.stderr
            lines = [line for line in cp.stdout.splitlines() if line]
            assert lines and lines[0].split('\t')[0] == '1', cp.stdout
            print('real mysql client test passed')
        finally:
            if db.poll() is None:
                db.kill()
                db.wait(timeout=5)


if __name__ == '__main__':
    main()
