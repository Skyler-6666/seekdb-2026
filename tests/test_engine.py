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
        assert e.execute('CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(8), v VECTOR(2))').affected == 0
        assert e.execute("INSERT INTO t VALUES (1,'a','[0,0]'),(2,'a','[1,1]'),(3,'b','[2,2]')").affected == 3
        r = e.execute("SELECT id,L2_DISTANCE(v,'[0,1]') AS d FROM t WHERE category='a' ORDER BY d LIMIT 2")
        assert [row[0] for row in r.rows] == [1, 2], r.rows
        e.execute('BEGIN')
        e.execute("UPDATE t SET category='z' WHERE id=1")
        e.execute('DELETE FROM t WHERE id=2')
        assert e.execute('SELECT id FROM t WHERE category=\'z\'').rows == [[1]]
        e.execute('ROLLBACK')
        assert [row[0] for row in e.execute('SELECT id FROM t ORDER BY id').rows] == [1, 2, 3]
        e.execute('BEGIN')
        e.execute("UPDATE t SET category='c' WHERE id=3")
        e.execute('COMMIT')
        e.close()

        e2 = Engine(td)
        assert e2.execute('SELECT category FROM t WHERE id=3').rows == [['c']]
        try:
            e2.execute("INSERT INTO t VALUES (4,'x','[1]')")
            raise AssertionError('dimension mismatch was accepted')
        except Exception as exc:
            assert getattr(exc, 'code', None) == 1264
        e2.execute('CREATE VECTOR INDEX ix ON t(v)')
        e2.close()
    print('engine test passed')


if __name__ == '__main__':
    main()
