#!/usr/bin/env python3
"""Packet-level test for the MySQL 4.1 protocol over stdin/stdout."""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_exact(stream, n):
    data = b''
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


def read_packet(stream):
    h = read_exact(stream, 4)
    n = h[0] | (h[1] << 8) | (h[2] << 16)
    return h[3], read_exact(stream, n)


def write_packet(stream, payload, seq):
    stream.write(struct.pack('<I', len(payload))[:3] + bytes([seq & 255]) + payload)
    stream.flush()


def lenenc(data, pos=0):
    first = data[pos]
    if first < 0xfb:
        return first, pos + 1
    if first == 0xfc:
        return struct.unpack_from('<H', data, pos + 1)[0], pos + 3
    if first == 0xfd:
        return struct.unpack_from('<I', data, pos + 1)[0] & 0xffffff, pos + 4
    if first == 0xfe:
        return struct.unpack_from('<Q', data, pos + 1)[0], pos + 9
    raise ValueError('NULL length')


def lenenc_str(data, pos=0):
    n, pos = lenenc(data, pos)
    return data[pos:pos+n].decode('utf-8'), pos + n


def query(proc, sql):
    write_packet(proc.stdin, b'\x03' + sql.encode('utf-8'), 0)
    seq, first = read_packet(proc.stdout)
    if first[0] == 0xff:
        return {'error': struct.unpack_from('<H', first, 1)[0], 'seq': seq}
    if first[0] == 0x00:
        return {'ok': True, 'seq': seq}
    count, _ = lenenc(first)
    for _ in range(count):
        read_packet(proc.stdout)
    read_packet(proc.stdout)
    rows = []
    while True:
        seq, payload = read_packet(proc.stdout)
        if payload and payload[0] == 0xfe and len(payload) < 9:
            break
        pos = 0; row = []
        for _ in range(count):
            value, pos = lenenc_str(payload, pos)
            row.append(value)
        rows.append(row)
    return {'columns': count, 'rows': rows}


def main():
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ)
        env['MINIVECTORDB_PROTOCOL'] = 'mysql'
        proc = subprocess.Popen(
            [sys.executable, '-m', 'minivectordb.main', '--data-dir', td],
            cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        try:
            seq, handshake = read_packet(proc.stdout)
            assert seq == 0, seq
            caps = 0x00000200 | 0x00008000 | 0x00020000
            auth = struct.pack('<IIB23s', caps, 1 << 24, 33, b'')
            auth += b'root\0' + b'\0' + b'mysql_native_password\0'
            write_packet(proc.stdin, auth, 1)
            seq, ok = read_packet(proc.stdout)
            assert seq == 2 and ok[0] == 0, (seq, ok[:4])
            assert query(proc, 'SET NAMES utf8mb4')['ok']
            init = query(proc, 'SELECT @@version_comment LIMIT 1')
            assert init['rows'][0][0] == 'MiniVectorDB', init
            assert query(proc, 'CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2))')['ok']
            assert query(proc, "INSERT INTO t VALUES (1,'[0,0]'),(2,'[1,1]')")['ok']
            dup = query(proc, "INSERT INTO t VALUES (1,'[9,9]')")
            assert dup.get('error') == 1062, dup
            result = query(proc, "SELECT id, L2_DISTANCE(v,'[1,0]') AS d FROM t ORDER BY d LIMIT 2")
            assert result['columns'] == 2, result
            assert result['rows'][0][0] == '1', result
            write_packet(proc.stdin, b'\x01', 0)
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
            err = proc.stderr.read().decode('utf-8', 'replace')
            assert 'ready' in err, err
    print('mysql wire test passed')


if __name__ == '__main__':
    main()
