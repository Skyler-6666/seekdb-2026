#!/usr/bin/env python3
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minivectordb.engine import Engine


def main():
    with tempfile.TemporaryDirectory() as td:
        e = Engine(td)
        e.execute('CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2))')
        e.execute("INSERT INTO t VALUES (1,'[0,0]')")
        e.execute('BEGIN')
        e.execute("INSERT INTO t VALUES (2,'[1,1]')")
        e.close()  # process-like shutdown without COMMIT
        e = Engine(td)
        assert [r[0] for r in e.execute('SELECT id FROM t ORDER BY id').rows] == [1]
        e.execute('BEGIN')
        e.execute("INSERT INTO t VALUES (2,'[1,1]')")
        e.execute('COMMIT')
        e.close()
        wal = Path(td) / 'wal.mvd'
        with wal.open('ab') as f:
            f.write(b'MVTX\x00\x00\x00\x00\x00\x00\x00\x10partial')
        e = Engine(td)
        assert [r[0] for r in e.execute('SELECT id FROM t ORDER BY id').rows] == [1, 2]
        e.close()
    print('recovery test passed')


if __name__ == '__main__':
    main()
