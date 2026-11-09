#!/usr/bin/env python3
"""MiniVectorDB: a small single-process SQL vector database.

The implementation intentionally depends only on the Python standard library at
runtime.  NumPy is used when available to accelerate exact vector scans, with a
pure-Python fallback for portability and local testing.
"""
from __future__ import annotations

import argparse
import base64
import itertools
import binascii
import copy
import json
import math
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

try:  # optional, installed by the Docker image
    import numpy as _np
except Exception:  # pragma: no cover - exercised by the fallback tests
    _np = None
if os.environ.get('MVD_DISABLE_NUMPY') == '1':
    _np = None


try:
    from .sql import SQLError as SqlError
except ImportError:  # direct script execution / standalone tests
    class SqlError(Exception):
        def __init__(self, code: int, message: str):
            super().__init__(message)
            self.code = code
            self.message = message


class ColType(str, Enum):
    BIGINT = "BIGINT"
    VARCHAR = "VARCHAR"
    VECTOR = "VECTOR"


@dataclass
class Column:
    name: str
    type: ColType
    length: int = 0


@dataclass
class Row:
    values: list[Any]
    alive: bool = True


@dataclass
class Table:
    name: str
    columns: list[Column]
    primary_key: int
    rows: list[Row] = field(default_factory=list)
    pk_to_row: dict[int, int] = field(default_factory=dict)
    indexes: dict[str, str] = field(default_factory=dict)
    _matrix: Any = None
    _matrix_rows: list[int] = field(default_factory=list)
    _norms: Any = None

    def column_index(self, name: str) -> int:
        key = name.lower()
        for i, col in enumerate(self.columns):
            if col.name.lower() == key:
                return i
        raise SqlError(1054, f"Unknown column '{name}'")

    def vector_index(self) -> int:
        for i, col in enumerate(self.columns):
            if col.type == ColType.VECTOR:
                return i
        raise SqlError(1054, "Table has no VECTOR column")

    def invalidate(self) -> None:
        self._matrix = None
        self._matrix_rows = []
        self._norms = None

    def rebuild_matrix(self) -> None:
        if _np is None:
            return
        vi = self.vector_index()
        alive_count = sum(1 for row in self.rows if row.alive)
        if self._matrix is not None and len(self._matrix_rows) == alive_count:
            return
        alive_ids = [i for i, row in enumerate(self.rows) if row.alive]
        if not alive_ids:
            self._matrix = _np.empty((0, self.columns[vi].length), dtype=_np.float32)
            self._matrix_rows = []
            self._norms = _np.empty((0,), dtype=_np.float32)
            return
        self._matrix = _np.asarray([self.rows[i].values[vi] for i in alive_ids], dtype=_np.float32)
        self._matrix_rows = alive_ids
        self._norms = _np.einsum("ij,ij->i", self._matrix, self._matrix).astype(_np.float32)


def _float32(value: float, column: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SqlError(1264, f"Invalid numeric value for '{column}'")
    d = float(value)
    if not math.isfinite(d) or abs(d) > 3.4028234663852886e38:
        raise SqlError(1264, f"Value out of float32 range for '{column}'")
    f = struct.unpack('<f', struct.pack('<f', d))[0]
    if not math.isfinite(f):
        raise SqlError(1264, f"Value out of float32 range for '{column}'")
    return f


def parse_vector(value: Any) -> list[float]:
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith('[') or not text.endswith(']'):
            raise SqlError(1264, "Invalid VECTOR literal")
        body = text[1:-1].strip()
        if not body:
            return []
        parts = body.split(',')
        try:
            return [_float32(float(p.strip()), "VECTOR") for p in parts]
        except ValueError as exc:
            raise SqlError(1264, "Invalid VECTOR literal") from exc
    if isinstance(value, (list, tuple)):
        return [_float32(v, "VECTOR") for v in value]
    raise SqlError(1264, "Invalid VECTOR value")


@dataclass
class Condition:
    column: str
    op: str
    value: Any


@dataclass
class SelectItem:
    kind: str
    column: str = ""
    alias: str = ""
    vector_column: str = ""
    query: list[float] = field(default_factory=list)


@dataclass
class SelectPlan:
    items: list[SelectItem]
    table: str
    conditions: list[Condition]
    order_column: Optional[str] = None
    order_distance: bool = False
    order_vector_column: str = ""
    order_query: list[float] = field(default_factory=list)
    order_desc: bool = False
    limit: Optional[int] = None


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    affected: int = 0
    is_query: bool = False

_TOKEN_RE = re.compile(
    r'''\s*(?:
    (?P<string>'(?:''|\\.|[^'])*'|"(?:""|\\.|[^"])*")
    |(?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
    |(?P<ident>[A-Za-z_][A-Za-z0-9_$]*)
    |(?P<symbol><=|>=|<>|!=|=|<|>|\[|\]|\(|\)|,|;|\*|-|\+)
    )''',
    re.VERBOSE,
)


def _strip_comments(sql: str) -> str:
    out: list[str] = []
    i = 0
    quote: Optional[str] = None
    while i < len(sql):
        c = sql[i]
        if quote:
            out.append(c)
            if c == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    out.append(sql[i + 1]); i += 2; continue
                quote = None
            elif c == "\\" and i + 1 < len(sql):
                out.append(sql[i + 1]); i += 2; continue
            i += 1
            continue
        if c in ("'", '"'):
            quote = c; out.append(c); i += 1
            continue
        if c == '-' and i + 1 < len(sql) and sql[i + 1] == '-':
            i += 2
            while i < len(sql) and sql[i] != '\n': i += 1
            continue
        if c == '/' and i + 1 < len(sql) and sql[i + 1] == '*':
            i += 2
            while i + 1 < len(sql) and not (sql[i] == '*' and sql[i + 1] == '/'): i += 1
            i = min(len(sql), i + 2)
            continue
        out.append(c); i += 1
    return ''.join(out)


@dataclass
class Token:
    kind: str
    text: str
    value: Any = None


def tokenize(sql: str) -> list[Token]:
    sql = _strip_comments(sql)
    result: list[Token] = []
    pos = 0
    while pos < len(sql):
        m = _TOKEN_RE.match(sql, pos)
        if not m:
            if sql[pos:].strip() == '':
                break
            raise SqlError(1064, f"Unexpected token near: {sql[pos:pos+32]!r}")
        pos = m.end()
        if m.group('string') is not None:
            raw = m.group('string')
            q = raw[0]
            body = raw[1:-1]
            if q == "'":
                body = body.replace("''", "'")
            else:
                body = body.replace('""', '"')
            try:
                body = bytes(body, 'utf-8').decode('unicode_escape') if '\\' in body else body
            except Exception as exc:
                raise SqlError(1064, "Invalid string escape") from exc
            result.append(Token('string', raw, body))
        elif m.group('number') is not None:
            raw = m.group('number')
            result.append(Token('number', raw, float(raw)))
        elif m.group('ident') is not None:
            raw = m.group('ident')
            result.append(Token('ident', raw, raw))
        else:
            raw = m.group('symbol')
            result.append(Token('symbol', raw, raw))
    result.append(Token('eof', ''))
    return result


class Tokens:
    def __init__(self, sql: str):
        self.sql = sql
        self.items = tokenize(sql)
        self.pos = 0
    def peek(self, n: int = 0) -> Token:
        return self.items[min(self.pos + n, len(self.items) - 1)]
    def end(self) -> bool:
        return self.peek().kind == 'eof'
    def take(self) -> Token:
        t = self.peek(); self.pos += 1; return t
    def match_kw(self, word: str) -> bool:
        if self.peek().kind == 'ident' and self.peek().text.upper() == word.upper():
            self.pos += 1; return True
        return False
    def peek_kw(self, word: str) -> bool:
        return self.peek().kind == 'ident' and self.peek().text.upper() == word.upper()
    def expect_kw(self, word: str) -> None:
        if not self.match_kw(word): self.error(f"expected {word}")
    def match_sym(self, sym: str) -> bool:
        if self.peek().kind == 'symbol' and self.peek().text == sym:
            self.pos += 1; return True
        return False
    def expect_sym(self, sym: str) -> None:
        if not self.match_sym(sym): self.error(f"expected '{sym}'")
    def ident(self) -> str:
        if self.peek().kind != 'ident': self.error('expected identifier')
        return self.take().text
    def integer(self) -> int:
        neg = self.match_sym('-')
        if self.peek().kind != 'number': self.error('expected integer')
        d = self.take().value
        if neg: d = -d
        if not math.isfinite(d) or math.floor(d) != d or d < -(2**63) or d > 2**63 - 1:
            self.error('integer out of range')
        return int(d)
    def error(self, message: str):
        raise SqlError(1064, f"SQL syntax error: {message}")


def parse_value(t: Tokens) -> Any:
    if t.peek().kind == 'string':
        return t.take().value
    if t.match_sym('['):
        vals = []
        if not t.match_sym(']'):
            while True:
                neg = t.match_sym('-')
                if t.peek().kind != 'number': t.error('expected vector number')
                v = t.take().value
                if neg: v = -v
                vals.append(_float32(v, 'VECTOR'))
                if t.match_sym(','): continue
                t.expect_sym(']'); break
        return vals
    neg = t.match_sym('-')
    if t.peek().kind != 'number': t.error('expected literal value')
    d = t.take().value
    if neg: d = -d
    if not math.isfinite(d): raise SqlError(1264, 'Value is not finite')
    if d.is_integer() and -(2**63) <= d <= 2**63 - 1:
        return int(d)
    return d


def coerce_value(value: Any, col: Column) -> Any:
    if col.type == ColType.BIGINT:
        if isinstance(value, bool): raise SqlError(1264, f"Invalid BIGINT for '{col.name}'")
        if isinstance(value, int) and -(2**63) <= value <= 2**63 - 1: return value
        if isinstance(value, float) and value.is_integer() and -(2**63) <= value <= 2**63 - 1:
            return int(value)
        raise SqlError(1264, f"BIGINT value out of range for '{col.name}'")
    if col.type == ColType.VARCHAR:
        if not isinstance(value, str): raise SqlError(1366, f"Invalid VARCHAR for '{col.name}'")
        if len(value) > col.length: raise SqlError(1406, f"Data too long for '{col.name}'")
        return value
    vec = parse_vector(value)
    if len(vec) != col.length: raise SqlError(1264, f"Vector dimension mismatch for '{col.name}'")
    return vec

_COMPARE_OPS = {'=', '!=', '<>', '<', '<=', '>', '>='}


def _parse_conditions(t: Tokens) -> list[Condition]:
    out: list[Condition] = []
    while not t.end() and not t.peek_kw('ORDER') and not t.peek_kw('LIMIT'):
        if t.match_kw('AND'):
            continue
        wrapped = t.match_sym('(')
        col = t.ident()
        op = t.take()
        if op.kind != 'symbol' or op.text not in _COMPARE_OPS:
            t.error('invalid comparison operator')
        value = parse_value(t)
        if wrapped: t.expect_sym(')')
        out.append(Condition(col, op.text, value))
        if t.match_kw('AND'):
            continue
        if t.end() or t.peek_kw('ORDER') or t.peek_kw('LIMIT'):
            break
        t.error('only AND is supported in WHERE')
    return out


def parse_select(sql: str) -> SelectPlan:
    t = Tokens(sql)
    t.expect_kw('SELECT')
    items: list[SelectItem] = []
    while not t.peek_kw('FROM'):
        if t.match_sym('*'):
            item = SelectItem('star')
        elif t.peek_kw('L2_DISTANCE'):
            t.take(); t.expect_sym('(')
            vector_column = t.ident(); t.expect_sym(',')
            query = parse_vector(parse_value(t)); t.expect_sym(')')
            item = SelectItem('distance', vector_column=vector_column, query=query)
        else:
            item = SelectItem('column', column=t.ident())
        if t.match_kw('AS'):
            item.alias = t.ident()
        elif t.peek().kind == 'ident' and not any(t.peek_kw(k) for k in ('FROM','WHERE','ORDER','LIMIT')):
            item.alias = t.take().text
        items.append(item)
        if not t.match_sym(','): break
    t.expect_kw('FROM')
    table = t.ident()
    conds: list[Condition] = []
    if t.match_kw('WHERE'): conds = _parse_conditions(t)
    order_column = None; order_distance = False; order_vec_col = ''; order_query: list[float] = []; order_desc = False
    if t.match_kw('ORDER'):
        t.expect_kw('BY')
        if t.peek_kw('L2_DISTANCE'):
            t.take(); t.expect_sym('(')
            order_distance = True
            order_vec_col = t.ident(); t.expect_sym(',')
            order_query = parse_vector(parse_value(t)); t.expect_sym(')')
        else:
            order_column = t.ident()
        if t.match_kw('DESC'): order_desc = True
        else: t.match_kw('ASC')
    limit = None
    if t.match_kw('LIMIT'):
        limit = t.integer()
        if limit < 0: raise SqlError(1064, 'LIMIT must be non-negative')
    if not t.end(): t.error('unexpected token')
    return SelectPlan(items, table, conds, order_column, order_distance, order_vec_col, order_query, order_desc, limit)


def parse_create_table(sql: str) -> Table:
    t = Tokens(sql)
    t.expect_kw('CREATE'); t.expect_kw('TABLE')
    if_not_exists = False
    if t.match_kw('IF'):
        t.expect_kw('NOT'); t.expect_kw('EXISTS'); if_not_exists = True
    name = t.ident()
    t.expect_sym('(')
    cols: list[Column] = []
    pk_inline = -1
    pk_table = -1
    while not t.match_sym(')'):
        if t.peek_kw('PRIMARY'):
            t.take(); t.expect_kw('KEY'); t.expect_sym('(')
            pk_name = t.ident(); t.expect_sym(')')
            pk_table = next((i for i,c in enumerate(cols) if c.name.lower() == pk_name.lower()), -1)
            if pk_table < 0: raise SqlError(1064, 'PRIMARY KEY column must precede table constraint')
        else:
            cname = t.ident()
            ctype = t.take()
            if ctype.kind != 'ident': t.error('expected column type')
            typ = ctype.text.upper()
            length = 0
            if t.match_sym('('):
                length = t.integer(); t.expect_sym(')')
            if typ == 'BIGINT': col = Column(cname, ColType.BIGINT)
            elif typ == 'VARCHAR':
                if length <= 0: raise SqlError(1064, 'VARCHAR length must be positive')
                col = Column(cname, ColType.VARCHAR, length)
            elif typ == 'VECTOR':
                if length <= 0: raise SqlError(1064, 'VECTOR dimension must be positive')
                col = Column(cname, ColType.VECTOR, length)
            else: raise SqlError(1064, f'Unsupported column type {typ}')
            if t.match_kw('PRIMARY'):
                t.expect_kw('KEY'); pk_inline = len(cols)
            cols.append(col)
        if not t.match_sym(','): t.expect_sym(')'); break
    if t.match_kw('WITH'):  # tolerate table options
        while not t.end(): t.take()
    if not t.end(): t.error('unexpected trailing token')
    if not cols: raise SqlError(1064, 'Table must have columns')
    vector_count = sum(c.type == ColType.VECTOR for c in cols)
    if vector_count > 1: raise SqlError(1064, 'At most one VECTOR column is allowed')
    pk = pk_inline if pk_inline >= 0 else pk_table
    if pk < 0:
        bigints = [i for i,c in enumerate(cols) if c.type == ColType.BIGINT and i == 0]
        if bigints: pk = bigints[0]
        else: raise SqlError(1064, 'A BIGINT PRIMARY KEY is required')
    if cols[pk].type != ColType.BIGINT: raise SqlError(1064, 'PRIMARY KEY must be BIGINT')
    table = Table(name=name, columns=cols, primary_key=pk)
    table.if_not_exists = if_not_exists  # compatible annotation
    return table


def parse_insert(sql: str) -> tuple[str, list[str] | None, list[list[Any]]]:
    t = Tokens(sql)
    t.expect_kw('INSERT'); t.expect_kw('INTO')
    table = t.ident()
    columns: list[str] | None = None
    if t.match_sym('('):
        columns = [t.ident()]
        while t.match_sym(','): columns.append(t.ident())
        t.expect_sym(')')
    t.expect_kw('VALUES')
    rows: list[list[Any]] = []
    while True:
        t.expect_sym('(')
        vals = [parse_value(t)]
        while t.match_sym(','): vals.append(parse_value(t))
        t.expect_sym(')')
        rows.append(vals)
        if not t.match_sym(','): break
    if not t.end(): t.error('unexpected token')
    return table, columns, rows


def parse_update(sql: str) -> tuple[str, list[tuple[str, Any]], list[Condition]]:
    t = Tokens(sql)
    t.expect_kw('UPDATE'); table = t.ident(); t.expect_kw('SET')
    sets: list[tuple[str, Any]] = []
    while True:
        col = t.ident(); t.expect_sym('='); val = parse_value(t); sets.append((col, val))
        if not t.match_sym(','): break
    conds: list[Condition] = []
    if t.match_kw('WHERE'): conds = _parse_conditions(t)
    if not t.end(): t.error('unexpected token')
    return table, sets, conds


def parse_delete(sql: str) -> tuple[str, list[Condition]]:
    t = Tokens(sql)
    t.expect_kw('DELETE'); t.expect_kw('FROM'); table = t.ident()
    conds: list[Condition] = []
    if t.match_kw('WHERE'): conds = _parse_conditions(t)
    if not t.end(): t.error('unexpected token')
    return table, conds


def parse_create_index(sql: str) -> tuple[str, str, str]:
    t = Tokens(sql)
    t.expect_kw('CREATE'); t.expect_kw('VECTOR'); t.expect_kw('INDEX')
    name = t.ident(); t.expect_kw('ON'); table = t.ident(); t.expect_sym('(')
    column = t.ident(); t.expect_sym(')')
    return table, name, column

def format_value(value: Any) -> str:
    if isinstance(value, bool): return '1' if value else '0'
    if isinstance(value, int): return str(value)
    if isinstance(value, float):
        if not math.isfinite(value): raise SqlError(1264, 'Non-finite output value')
        return format(value, '.9g')
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(format(float(v), '.9g') for v in value) + ']'
    return str(value)


def compare_values(left: Any, op: str, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        a, b = float(left), float(right)
    elif isinstance(left, str) and isinstance(right, str):
        a, b = left, right
    else:
        raise SqlError(1064, 'Incompatible comparison operands')
    if op == '=': return a == b
    if op in ('!=', '<>'): return a != b
    if op == '<': return a < b
    if op == '<=': return a <= b
    if op == '>': return a > b
    if op == '>=': return a >= b
    raise SqlError(1064, f'Unsupported comparison {op}')


def l2_distance(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a); bv = list(b)
    if len(av) != len(bv): raise SqlError(1264, 'Vector dimension mismatch')
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(av, bv)))


class Database:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wal_path = self.data_dir / 'wal.mvd'
        self.tables: dict[str, Table] = {}
        self.in_txn = False
        self.txn_snapshot: dict[str, Table] | None = None
        self.txn_sql: list[str] = []
        self.replaying = False
        self._load_wal()
        self._wal_file = open(self.wal_path, 'ab', buffering=0)

    def close(self) -> None:
        try:
            self._wal_file.flush(); os.fsync(self._wal_file.fileno())
        except Exception:
            pass
        self._wal_file.close()

    def _table(self, name: str) -> Table:
        key = name.lower()
        table = self.tables.get(key)
        if table is None: raise SqlError(1146, f"Table '{name}' doesn't exist")
        return table

    def _append_frame(self, statements: list[str]) -> None:
        if not statements: return
        payload = bytearray()
        payload += struct.pack('>I', len(statements))
        for stmt in statements:
            raw = stmt.encode('utf-8')
            payload += struct.pack('>I', len(raw)); payload += raw
        crc = binascii.crc32(payload) & 0xffffffff
        frame = struct.pack('>4sQI', b'MVTX', len(payload), crc) + bytes(payload)
        self._wal_file.write(frame)
        self._wal_file.flush()
        os.fsync(self._wal_file.fileno())

    def _load_wal(self) -> None:
        if not self.wal_path.exists(): return
        data = self.wal_path.read_bytes()
        pos = 0
        while pos + 16 <= len(data):
            magic, length, crc = struct.unpack_from('>4sQI', data, pos)
            if magic != b'MVTX' or length > 512 * 1024 * 1024 or pos + 16 + length > len(data):
                break
            payload = data[pos+16:pos+16+length]
            if (binascii.crc32(payload) & 0xffffffff) != crc:
                break
            try:
                count = struct.unpack_from('>I', payload, 0)[0]
                off = 4; statements = []
                for _ in range(count):
                    n = struct.unpack_from('>I', payload, off)[0]; off += 4
                    if off + n > len(payload): raise ValueError('truncated statement')
                    statements.append(payload[off:off+n].decode('utf-8')); off += n
                self.replaying = True
                for stmt in statements:
                    self._execute_sql(stmt, log=False)
                self.replaying = False
            except Exception as exc:
                self.replaying = False
                print(f'[minivectordb] WAL replay stopped: {exc}', file=sys.stderr)
                break
            pos += 16 + length

    def begin(self) -> None:
        if self.in_txn: raise SqlError(1568, 'A transaction is already in progress')
        self.txn_snapshot = copy.deepcopy(self.tables)
        self.txn_sql = []
        self.in_txn = True

    def commit(self) -> None:
        if not self.in_txn: raise SqlError(1568, 'No transaction is active')
        if self.txn_sql: self._append_frame(self.txn_sql)
        self.in_txn = False
        self.txn_snapshot = None
        self.txn_sql = []

    def rollback(self) -> None:
        if not self.in_txn: raise SqlError(1568, 'No transaction is active')
        self.tables = self.txn_snapshot or {}
        self.in_txn = False
        self.txn_snapshot = None
        self.txn_sql = []

    def _record_modification(self, sql: str, replay: bool = False) -> None:
        if replay or self.replaying: return
        if self.in_txn:
            self.txn_sql.append(sql)
        else:
            self._append_frame([sql])

    def execute(self, sql: str) -> QueryResult:
        return self.execute_sql(sql)

    def execute_sql(self, sql: str) -> QueryResult:
        stmt = sql.strip().rstrip(';').strip()
        if not stmt: return QueryResult()
        result, modified = self._execute_sql(stmt, log=False)
        if modified and not self.replaying:
            self._record_modification(stmt)
        return result

    def _execute_sql(self, stmt: str, log: bool = False) -> tuple[QueryResult, bool]:
        first = stmt.split(None, 1)[0].upper() if stmt.split() else ''
        compact = re.sub(r'\s+', ' ', stmt.strip()).upper()
        if first == 'SET' or compact.startswith('USE '):
            return QueryResult(affected=0), False
        if first == 'SELECT' and '@@' in compact:
            if 'VERSION_COMMENT' in compact:
                return QueryResult(['@@version_comment'], [['MiniVectorDB']], is_query=True), False
            if '@@VERSION' in compact:
                return QueryResult(['@@version'], [['5.7.0-MiniVectorDB']], is_query=True), False
            if '@@AUTOCOMMIT' in compact:
                return QueryResult(['@@autocommit'], [[1]], is_query=True), False
            if 'TRANSACTION_ISOLATION' in compact:
                return QueryResult(['@@transaction_isolation'], [['READ-COMMITTED']], is_query=True), False
            if any(x in compact for x in ('CHARACTER_SET', 'COLLATION')):
                return QueryResult(['@@character_set_client'], [['utf8mb4']], is_query=True), False
            if 'LOWER_CASE_TABLE_NAMES' in compact:
                return QueryResult(['@@lower_case_table_names'], [[0]], is_query=True), False
            return QueryResult(['@@value'], [['']], is_query=True), False
        if compact in ('SELECT 1', 'SELECT 1 LIMIT 1'):
            return QueryResult(['1'], [[1]], is_query=True), False
        if compact.startswith('SELECT DATABASE()'):
            return QueryResult(['DATABASE()'], [[None]], is_query=True), False
        if compact.startswith('SELECT CONNECTION_ID()'):
            return QueryResult(['CONNECTION_ID()'], [[1]], is_query=True), False
        if compact.startswith('SELECT VERSION()'):
            return QueryResult(['VERSION()'], [['5.7.0-MiniVectorDB']], is_query=True), False
        if compact.startswith('SHOW WARNINGS'):
            return QueryResult(['Level', 'Code', 'Message'], [], is_query=True), False
        if compact.startswith('SHOW VARIABLES'):
            return QueryResult(['Variable_name', 'Value'], [], is_query=True), False
        if first == 'BEGIN' or first == 'START':
            if first == 'START':
                t = Tokens(stmt); t.expect_kw('START'); t.expect_kw('TRANSACTION')
            self.begin(); return QueryResult(), False
        if first == 'COMMIT':
            self.commit(); return QueryResult(), False
        if first == 'ROLLBACK':
            self.rollback(); return QueryResult(), False
        if first == 'CREATE':
            if re.match(r'(?is)^CREATE\s+TABLE\b', stmt):
                return self._create_table(stmt), True
            if re.match(r'(?is)^CREATE\s+VECTOR\s+INDEX\b', stmt):
                return self._create_index(stmt), True
            raise SqlError(1064, 'Unsupported CREATE statement')
        if first == 'DROP':
            return self._drop_table(stmt), True
        if first == 'INSERT':
            return self._insert(stmt), True
        if first == 'UPDATE':
            return self._update(stmt), True
        if first == 'DELETE':
            return self._delete(stmt), True
        if first == 'SELECT':
            return self._select(stmt), False
        if first == 'SHOW':
            t = Tokens(stmt); t.expect_kw('SHOW'); t.expect_kw('TABLES')
            return QueryResult(['table_name'], [[name] for name in sorted(self.tables)], is_query=True), False
        if first == 'SET':
            return QueryResult(affected=0), False
        raise SqlError(1064, f'Unsupported statement: {first}')

    def _create_table(self, stmt: str) -> QueryResult:
        table = parse_create_table(stmt)
        key = table.name.lower()
        if key in self.tables:
            if getattr(table, 'if_not_exists', False): return QueryResult(affected=0)
            raise SqlError(1050, f"Table '{table.name}' already exists")
        if not hasattr(table, 'if_not_exists'): table.if_not_exists = False
        self.tables[key] = table
        return QueryResult(affected=0)

    def _create_index(self, stmt: str) -> QueryResult:
        table_name, index_name, column = parse_create_index(stmt)
        table = self._table(table_name)
        idx = table.column_index(column)
        if table.columns[idx].type != ColType.VECTOR:
            raise SqlError(1054, f"Column '{column}' is not a VECTOR")
        table.indexes[index_name.lower()] = column
        return QueryResult(affected=0)

    def _drop_table(self, stmt: str) -> QueryResult:
        t = Tokens(stmt); t.expect_kw('DROP'); t.expect_kw('TABLE')
        if_exists = False
        if t.match_kw('IF'):
            t.expect_kw('EXISTS'); if_exists = True
        name = t.ident()
        if not t.end(): t.error('unexpected token')
        key = name.lower()
        if key not in self.tables:
            if if_exists: return QueryResult(affected=0)
            raise SqlError(1051, f"Unknown table '{name}'")
        del self.tables[key]
        return QueryResult(affected=0)

    def _insert(self, stmt: str) -> QueryResult:
        table_name, column_names, raw_rows = parse_insert(stmt)
        table = self._table(table_name)
        if column_names is None:
            order = list(range(len(table.columns)))
        else:
            order = [table.column_index(c) for c in column_names]
            if len(set(order)) != len(order): raise SqlError(1060, 'Duplicate column name')
        prepared: list[list[Any]] = []
        seen_pk: set[int] = set()
        for raw in raw_rows:
            if len(raw) != len(order): raise SqlError(1136, 'Column count does not match value count')
            values: list[Any] = [None] * len(table.columns)
            for idx, val in zip(order, raw): values[idx] = coerce_value(val, table.columns[idx])
            if any(v is None for v in values): raise SqlError(1064, 'All columns require explicit values')
            pk = values[table.primary_key]
            if pk in table.pk_to_row or pk in seen_pk:
                raise SqlError(1062, f"Duplicate entry '{pk}' for PRIMARY KEY")
            seen_pk.add(pk); prepared.append(values)
        for values in prepared:
            pk = values[table.primary_key]
            table.pk_to_row[pk] = len(table.rows)
            table.rows.append(Row(values))
        table.invalidate()
        return QueryResult(affected=len(prepared))

    def _update(self, stmt: str) -> QueryResult:
        table_name, sets, conds = parse_update(stmt)
        table = self._table(table_name)
        set_indices: list[tuple[int, Column, Any]] = []
        seen: set[int] = set()
        for name, value in sets:
            idx = table.column_index(name)
            if idx in seen: raise SqlError(1110, f"Duplicate column '{name}' in SET")
            seen.add(idx)
            if idx == table.primary_key: raise SqlError(1064, 'Primary key updates are not supported')
            set_indices.append((idx, table.columns[idx], coerce_value(value, table.columns[idx])))
        prepared = self._prepare_conditions(table, conds)
        matches: list[int] = []
        for i, row in enumerate(table.rows):
            if row.alive and self._row_matches(table, row, prepared): matches.append(i)
        for i in matches:
            for idx, _, value in set_indices:
                table.rows[i].values[idx] = copy.deepcopy(value)
        if matches: table.invalidate()
        return QueryResult(affected=len(matches))

    def _delete(self, stmt: str) -> QueryResult:
        table_name, conds = parse_delete(stmt)
        table = self._table(table_name)
        prepared = self._prepare_conditions(table, conds)
        matches = [i for i,row in enumerate(table.rows) if row.alive and self._row_matches(table, row, prepared)]
        for i in matches:
            row = table.rows[i]
            row.alive = False
            pk = row.values[table.primary_key]
            table.pk_to_row.pop(pk, None)
        if matches: table.invalidate()
        return QueryResult(affected=len(matches))

    def _prepare_conditions(self, table: Table, conds: list[Condition]) -> list[tuple[int, str, Any]]:
        prepared: list[tuple[int, str, Any]] = []
        for cond in conds:
            idx = table.column_index(cond.column)
            prepared.append((idx, cond.op, coerce_value(cond.value, table.columns[idx])))
        return prepared

    def _row_matches(self, table: Table, row: Row, conds: list[Condition] | list[tuple[int, str, Any]]) -> bool:
        for cond in conds:
            if isinstance(cond, tuple):
                idx, op, rhs = cond
            else:
                idx = table.column_index(cond.column)
                rhs = coerce_value(cond.value, table.columns[idx])
                op = cond.op
            if not compare_values(row.values[idx], op, rhs): return False
        return True

    def _select(self, stmt: str) -> QueryResult:
        plan = parse_select(stmt)
        table = self._table(plan.table)

        prepared_conditions = self._prepare_conditions(table, plan.conditions)
        candidate_indices = [
            i for i, row in enumerate(table.rows)
            if row.alive and self._row_matches(table, row, prepared_conditions)
        ]
        candidates: list[tuple[int, Row]] = [
            (table.rows[i].values[table.primary_key], table.rows[i]) for i in candidate_indices
        ]

        output_columns: list[str] = []
        resolved_items: list[SelectItem] = []
        for item in plan.items:
            if item.kind == 'star':
                for col in table.columns:
                    resolved_items.append(SelectItem('column', column=col.name, alias=col.name))
                    output_columns.append(col.name)
            else:
                resolved_items.append(item)
                if item.alias:
                    output_columns.append(item.alias)
                elif item.kind == 'distance':
                    output_columns.append(f'L2_DISTANCE({item.vector_column})')
                else:
                    output_columns.append(item.column)

        alias_map = {item.alias.lower(): item for item in resolved_items if item.alias}
        distance_cache: dict[tuple[int, str, tuple[float, ...]], float] = {}
        fast_distances: dict[int, float] = {}

        def distance_for(row: Row, vector_column: str, query: list[float]) -> float:
            if id(row) in fast_distances:
                return fast_distances[id(row)]
            key = (id(row), vector_column.lower(), tuple(query))
            if key not in distance_cache:
                vi = table.column_index(vector_column)
                distance_cache[key] = l2_distance(row.values[vi], query)
            return distance_cache[key]

        def order_value(row: Row) -> Any:
            if plan.order_distance:
                return distance_for(row, plan.order_vector_column, plan.order_query)
            if plan.order_column is None:
                return 0
            name = plan.order_column.lower()
            if name in alias_map:
                item = alias_map[name]
                if item.kind == 'distance':
                    return distance_for(row, item.vector_column, item.query)
                if item.kind == 'column':
                    return row.values[table.column_index(item.column)]
            return row.values[table.column_index(plan.order_column)]

        fast_vector_column = ''
        fast_query: list[float] = []
        if plan.order_distance:
            fast_vector_column, fast_query = plan.order_vector_column, plan.order_query
        elif plan.order_column and plan.order_column.lower() in alias_map:
            alias_item = alias_map[plan.order_column.lower()]
            if alias_item.kind == 'distance':
                fast_vector_column, fast_query = alias_item.vector_column, alias_item.query

        used_fast_path = False
        if _np is not None and fast_vector_column and candidates:
            table.rebuild_matrix()
            pos_by_index = {row_index: pos for pos, row_index in enumerate(table._matrix_rows)}
            positions = _np.asarray([pos_by_index[i] for i in candidate_indices], dtype=_np.intp)
            vectors = table._matrix[positions]
            query = _np.asarray(fast_query, dtype=_np.float32)
            d2 = table._norms[positions] - 2.0 * (vectors @ query) + float(query @ query)
            distances = _np.sqrt(_np.maximum(d2, 0.0))
            pks = _np.asarray([table.rows[i].values[table.primary_key] for i in candidate_indices], dtype=_np.int64)
            order = _np.lexsort((pks, -distances if plan.order_desc else distances))
            if plan.limit is not None:
                order = order[:max(0, plan.limit)]
            candidates = [(int(pks[j]), table.rows[candidate_indices[int(j)]]) for j in order]
            for j in order:
                fast_distances[id(table.rows[candidate_indices[int(j)]])] = float(distances[int(j)])
            used_fast_path = True

        if not used_fast_path:
            if plan.order_distance or plan.order_column is not None:
                candidates.sort(key=lambda pair: (order_value(pair[1]), pair[0]))
                if plan.order_desc:
                    candidates.sort(key=lambda pair: (order_value(pair[1]), -pair[0]), reverse=True)
            else:
                candidates.sort(key=lambda pair: pair[0])
            if plan.limit is not None:
                candidates = candidates[:plan.limit]

        out_rows: list[list[Any]] = []
        for _, row in candidates:
            out: list[Any] = []
            for item in resolved_items:
                if item.kind == 'column':
                    out.append(row.values[table.column_index(item.column)])
                elif item.kind == 'distance':
                    out.append(distance_for(row, item.vector_column, item.query))
                else:
                    raise SqlError(1064, 'Invalid projection')
            out_rows.append(out)
        return QueryResult(output_columns, out_rows, is_query=True)

Engine = Database
Result = QueryResult


def jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)): return [float(x) for x in value]
    return value


def emit_text(result: QueryResult, out) -> None:
    if result.is_query:
        for row in result.rows:
            out.write('\t'.join(format_value(v) for v in row) + '\n')
        out.write('\n')
    else:
        out.write(f'OK {result.affected}\n\n')
    out.flush()


def emit_error(err: SqlError, out) -> None:
    out.write(f'ERROR {err.code} {err.message}\n\n')
    out.flush()


def extract_json_sql(obj: dict[str, Any]) -> str:
    for key in ('sql', 'statement', 'query', 'command'):
        value = obj.get(key)
        if isinstance(value, str): return value
    raise SqlError(1064, 'JSON request is missing sql/statement/query/command')


def split_statements_from_buffer(buffer: str) -> tuple[list[str], str]:
    statements: list[str] = []
    start = 0; i = 0; quote: Optional[str] = None
    while i < len(buffer):
        c = buffer[i]
        if quote:
            if c == quote:
                if i + 1 < len(buffer) and buffer[i+1] == quote: i += 2; continue
                quote = None
            elif c == '\\': i += 2; continue
        elif c in ("'", '"'): quote = c
        elif c == ';':
            stmt = buffer[start:i].strip()
            if stmt: statements.append(stmt + ';')
            start = i + 1
        i += 1
    return statements, buffer[start:]


def run_jsonl(db: Database, inp, out) -> int:
    for line in inp:
        if not line.strip(): continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict): raise SqlError(1064, 'JSON request must be an object')
            sql = extract_json_sql(obj)
            if _is_exit_command(sql):
                return 0
            result = db.execute_sql(sql)
            payload = {
                'ok': True,
                'columns': result.columns,
                'rows': [[jsonable(v) for v in row] for row in result.rows],
                'affected_rows': result.affected,
                'is_query': result.is_query,
            }
            if 'id' in obj: payload['id'] = obj['id']
            if 'request_id' in obj: payload['request_id'] = obj['request_id']
            out.write(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n'); out.flush()
        except SqlError as exc:
            payload = {'ok': False, 'error': {'code': exc.code, 'message': exc.message}}
            try:
                if isinstance(obj, dict):
                    if 'id' in obj: payload['id'] = obj['id']
                    if 'request_id' in obj: payload['request_id'] = obj['request_id']
            except Exception:
                pass
            out.write(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n'); out.flush()
            print(f'[minivectordb] SQL error {exc.code}: {exc.message}', file=sys.stderr)
        except Exception as exc:
            out.write(json.dumps({'ok': False, 'error': {'code': 1105, 'message': str(exc)}}, ensure_ascii=False) + '\n'); out.flush()
            print(f'[minivectordb] internal error: {exc}', file=sys.stderr)
    return 0


def _is_exit_command(stmt: str) -> bool:
    return stmt.strip().rstrip(';').strip().upper() in ('EXIT', 'QUIT')


def _is_ping_command(stmt: str) -> bool:
    return stmt.strip().rstrip(';').strip().upper() in ('PING', '__PING__')


def run_text(db: Database, inp, out) -> int:
    buffer = ''
    for line in inp:
        buffer += line
        statements, buffer = split_statements_from_buffer(buffer)
        for stmt in statements:
            stmt_clean = stmt.strip().rstrip(';').strip()
            if not stmt_clean: continue
            if _is_exit_command(stmt_clean): return 0
            if _is_ping_command(stmt_clean): out.write('PONG\n'); out.flush(); continue
            try:
                result = db.execute_sql(stmt)
                emit_text(result, out)
            except SqlError as exc:
                emit_error(exc, out)
                print(f'[minivectordb] SQL error {exc.code}: {exc.message}', file=sys.stderr)
            except Exception as exc:
                emit_error(SqlError(1105, str(exc)), out)
                print(f'[minivectordb] internal error: {exc}', file=sys.stderr)
    trailing = buffer.strip()
    if trailing:
        if trailing.upper() in ('EXIT', 'QUIT'): return 0
        try:
            emit_text(db.execute_sql(trailing), out)
        except SqlError as exc:
            emit_error(exc, out)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='MiniVectorDB stdin/stdout server')
    parser.add_argument('--data-dir', default='/data', help='persistent data directory')
    parser.add_argument('--protocol', choices=('auto', 'text', 'jsonl'), default='auto')
    args = parser.parse_args(argv)
    db = Database(args.data_dir)
    try:
        protocol = args.protocol
        if protocol == 'auto':
            first = sys.stdin.readline()
            if first == '':
                return 0
            protocol = 'jsonl' if first.lstrip().startswith('{') else 'text'
            if protocol == 'jsonl':
                return run_jsonl(db, itertools.chain([first], sys.stdin), sys.stdout)
            return run_text(db, itertools.chain([first], sys.stdin), sys.stdout)
        if protocol == 'jsonl': return run_jsonl(db, sys.stdin, sys.stdout)
        return run_text(db, sys.stdin, sys.stdout)
    finally:
        db.close()


if __name__ == '__main__':
    raise SystemExit(main())
