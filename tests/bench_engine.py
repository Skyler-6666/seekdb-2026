#!/usr/bin/env python3
import random, sys, tempfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from minivectordb.engine import Engine
random.seed(20261109)
n=10000; dim=128; qn=1000
def vec(v): return '['+','.join(f'{x:.5f}' for x in v)+']'
with tempfile.TemporaryDirectory() as td:
    e=Engine(td)
    t0=time.perf_counter()
    e.execute(f'CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(16), v VECTOR({dim}))')
    for s in range(0,n,100):
        vals=[]
        for i in range(s,min(s+100,n)):
            v=[random.random() for _ in range(dim)]
            vals.append(f"({i},'c{i%20}','{vec(v)}')")
        e.execute('INSERT INTO t VALUES '+','.join(vals))
    t1=time.perf_counter()
    for j in range(qn):
        q=vec([random.random() for _ in range(dim)])
        where=" WHERE category='c0'" if j%2==0 else ''
        e.execute(f"SELECT id,L2_DISTANCE(v,'{q}') AS d FROM t{where} ORDER BY d LIMIT 10")
    t2=time.perf_counter()
    e.close()
    print('insert_seconds',round(t1-t0,3),'query_seconds',round(t2-t1,3),'total_seconds',round(t2-t0,3))
