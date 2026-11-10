"""MiniVectorDB storage and execution engine (no external database engine)."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import tempfile
import uuid
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sql import SQLError, parse_sql

try:  # NumPy is only a math helper; the database itself stays in this package.
    import numpy as np
except Exception:  # pragma: no cover - fallback for minimal images
    np = None


MAX_VECTOR_DIM = 16000


def _json_default(value):
    if isinstance(value, array):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


@dataclass
class Column:
    name: str
    type: str
    length: int | None
    primary: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "length": self.length, "primary": self.primary}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Column":
        return cls(value["name"], value["type"], value.get("length"), bool(value.get("primary")))


@dataclass
class RowStore:
    """Dictionary-like row storage that spills large tables to an append file.

    The small public cases stay in memory.  Once a table crosses the threshold,
    rows are addressed by byte offset instead of keeping Python objects alive.
    """
    path: Path
    table: "Table"
    offsets: dict[int, int] = field(default_factory=dict)
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _handle: Any = field(default=None, repr=False)

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._scan()
        self._handle = self.path.open("ab")

    def _scan(self):
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    record = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if "__delete__" in record:
                    self.offsets.pop(int(record["__delete__"]), None)
                else:
                    try:
                        normalized = _normalize_loaded_row(self.table, record)
                        self.offsets[int(normalized[self.table.primary.name])] = offset
                    except Exception:
                        continue

    def _read(self, pk: int):
        if pk in self._cache:
            self._cache.move_to_end(pk)
            return self._cache[pk]
        offset = self.offsets[pk]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            row = json.loads(handle.readline().decode("utf-8"))
        normalized = _normalize_loaded_row(self.table, row)
        self._cache[pk] = normalized
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return normalized

    def __getitem__(self, pk):
        return self._read(int(pk))

    def __setitem__(self, pk, row):
        pk = int(pk)
        data = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n").encode("utf-8")
        self._handle.seek(0, os.SEEK_END)
        offset = self._handle.tell()
        self._handle.write(data)
        self._handle.flush()
        self.offsets[pk] = offset
        self._cache.pop(pk, None)

    def __delitem__(self, pk):
        pk = int(pk)
        self.offsets.pop(pk, None)
        self._cache.pop(pk, None)
        self._handle.seek(0, os.SEEK_END)
        self._handle.write((json.dumps({"__delete__": pk}, separators=(",", ":")) + "\n").encode("utf-8"))
        self._handle.flush()

    def __iter__(self):
        return iter(self.offsets)

    def __len__(self):
        return len(self.offsets)

    def __contains__(self, pk):
        return int(pk) in self.offsets

    def keys(self):
        return self.offsets.keys()

    def items(self):
        for pk in list(self.offsets):
            yield pk, self[pk]

    def values(self):
        for pk in list(self.offsets):
            yield self[pk]

    def get(self, pk, default=None):
        try:
            return self[pk]
        except KeyError:
            return default

    def pop(self, pk, *args):
        if pk in self.offsets:
            row = self[pk]
            del self[pk]
            return row
        if args:
            return args[0]
        raise KeyError(pk)

    def __deepcopy__(self, memo):
        new_path = self.path.with_name(f"rows-{uuid.uuid4().hex}.jsonl")
        shutil.copy2(self.path, new_path)
        clone = object.__new__(type(self))
        clone.path = new_path
        clone.table = self.table
        clone.offsets = dict(self.offsets)
        clone._cache = OrderedDict()
        clone._handle = new_path.open("a", encoding="utf-8")
        return clone

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass
class Table:
    name: str
    columns: list[Column]
    rows: dict[int, dict[str, Any]] = field(default_factory=dict)
    vector_indexes: dict[str, dict[str, Any]] = field(default_factory=dict)
    version: int = 0
    _cache: dict[str, tuple[int, Any, list[int]]] = field(default_factory=dict, repr=False)

    @property
    def primary(self) -> Column:
        for column in self.columns:
            if column.primary:
                return column
        raise SQLError(1105, "table has no primary key")

    def column(self, name: str) -> Column:
        for column in self.columns:
            if column.name.lower() == name.lower():
                return column
        raise SQLError(1054, f"unknown column {name}")

    def touch(self) -> None:
        self.version += 1
        self._cache.clear()

    def vector_column(self) -> Column | None:
        for column in self.columns:
            if column.type == "VECTOR":
                return column
        return None

    def sorted_rows(self) -> list[dict[str, Any]]:
        return [self.rows[pk] for pk in sorted(self.rows)]

    def vector_matrix(self, column: Column) -> tuple[Any, list[int]]:
        cached = self._cache.get(column.name)
        if cached and cached[0] == self.version:
            return cached[1], cached[2]
        pks = sorted(self.rows)
        # Building a full float32 matrix roughly doubles the vector memory.
        # For very large tables, callers use the row-wise path instead.
        if len(pks) > 200_000:
            return None, pks
        values = [self.rows[pk][column.name] for pk in pks]
        if np is not None:
            matrix = np.asarray(values, dtype=np.float32)
        else:  # pragma: no cover
            matrix = values
        self._cache[column.name] = (self.version, matrix, pks)
        return matrix, pks

    def to_json(self) -> dict[str, Any]:
        if isinstance(self.rows, RowStore):
            return {
                "name": self.name,
                "columns": [column.to_json() for column in self.columns],
                "rows": [],
                "row_file": self.rows.path.name,
                "vector_indexes": self.vector_indexes,
            }
        return {
            "name": self.name,
            "columns": [column.to_json() for column in self.columns],
            "rows": [self.rows[pk] for pk in sorted(self.rows)],
            "vector_indexes": self.vector_indexes,
        }

    def maybe_spill(self, data_dir: str | os.PathLike[str], threshold: int = 100_000) -> None:
        if isinstance(self.rows, RowStore) or len(self.rows) < threshold:
            return
        path = Path(data_dir) / f"rows-{self.name}-{uuid.uuid4().hex}.jsonl"
        store = RowStore(path, self)
        for pk in sorted(self.rows):
            store[pk] = self.rows[pk]
        self.rows = store

    @classmethod
    def from_json(cls, value: dict[str, Any], data_dir: str | os.PathLike[str] | None = None) -> "Table":
        table = cls(
            name=value["name"],
            columns=[Column.from_json(item) for item in value["columns"]],
        )
        table.vector_indexes = value.get("vector_indexes", {})
        pk_name = table.primary.name
        row_file = value.get("row_file")
        if row_file and data_dir is not None:
            table.rows = RowStore(Path(data_dir) / row_file, table)
        else:
            for row in value.get("rows", []):
                normalized = _normalize_loaded_row(table, row)
                table.rows[int(normalized[pk_name])] = normalized
        return table


@dataclass
class Result:
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    affected: int = 0
    is_query: bool = False


def _normalize_loaded_row(table: Table, row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for column in table.columns:
        value = row.get(column.name)
        if value is None:
            raise SQLError(1105, "NULL is not supported")
        if column.type in {"BIGINT", "INT", "INTEGER"}:
            result[column.name] = int(value)
        elif column.type == "VARCHAR":
            result[column.name] = str(value)
        elif column.type == "VECTOR":
            result[column.name] = array("f", (_as_float32(item) for item in value))
        else:
            result[column.name] = value
    return result


def _as_float32(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SQLError(1690, "invalid vector element") from exc
    if not math.isfinite(number):
        raise SQLError(1690, "vector element must be finite")
    if np is not None:
        converted = np.float32(number)
        if not np.isfinite(converted):
            raise SQLError(1690, "vector element out of float32 range")
        return float(converted)
    return number


def _normalize_value(table: Table, column: Column, value: Any) -> Any:
    if value is None:
        raise SQLError(1048, f"column {column.name} cannot be NULL")
    if column.type in {"BIGINT", "INT", "INTEGER"}:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                raise SQLError(1366, f"invalid integer for {column.name}")
        if value < -(2**63) or value > 2**63 - 1:
            raise SQLError(1264, f"integer out of range for {column.name}")
        return int(value)
    if column.type == "VARCHAR":
        text = value if isinstance(value, str) else str(value)
        if column.length is not None and len(text) > column.length:
            raise SQLError(1406, f"value too long for {column.name}")
        return text
    if column.type == "VECTOR":
        if not isinstance(value, (list, tuple, array)):
            raise SQLError(1690, f"invalid vector for {column.name}")
        if column.length is None or len(value) != column.length:
            raise SQLError(1366, f"vector dimension mismatch for {column.name}")
        return array("f", (_as_float32(item) for item in value))
    return value


def _compare(left: Any, right: Any, op: str) -> bool:
    if op in {"=", "=="}:
        return left == right
    if op in {"!=", "<>"}:
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise SQLError(1064, f"unsupported comparison {op}")


def _vector(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, array)):
        return [float(item) for item in value]
    raise SQLError(1366, "expected vector")


def _l2_distance(left: Any, right: Any) -> float:
    a = _vector(left)
    b = _vector(right)
    if len(a) != len(b):
        raise SQLError(1366, "vector dimension mismatch")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _cosine_distance(left: Any, right: Any) -> float:
    a = _vector(left)
    b = _vector(right)
    if len(a) != len(b):
        raise SQLError(1366, "vector dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def _inner_product(left: Any, right: Any) -> float:
    a = _vector(left)
    b = _vector(right)
    if len(a) != len(b):
        raise SQLError(1366, "vector dimension mismatch")
    return sum(x * y for x, y in zip(a, b))


def eval_expr(expr: Any, row: dict[str, Any], table: Table | None = None, aliases: dict[str, Any] | None = None):
    kind = expr[0]
    if kind == "literal":
        return expr[1]
    if kind == "vector":
        return expr[1]
    if kind == "column":
        name = expr[1]
        if aliases and name.lower() in aliases:
            return aliases[name.lower()]
        if row is None or name not in row:
            if table is not None and any(c.name.lower() == name.lower() for c in table.columns):
                return row[next(c.name for c in table.columns if c.name.lower() == name.lower())]
            raise SQLError(1054, f"unknown column {name}")
        return row[name]
    if kind == "star":
        raise SQLError(1054, "star is not an expression")
    if kind == "and":
        return bool(eval_expr(expr[1], row, table, aliases)) and bool(eval_expr(expr[2], row, table, aliases))
    if kind == "or":
        return bool(eval_expr(expr[1], row, table, aliases)) or bool(eval_expr(expr[2], row, table, aliases))
    if kind == "compare":
        op, left_expr, right_expr = expr[1], expr[2], expr[3]
        left = eval_expr(left_expr, row, table, aliases)
        right = eval_expr(right_expr, row, table, aliases)
        if isinstance(left, list) or isinstance(right, list):
            raise SQLError(1235, "vector comparison is not supported")
        return _compare(left, right, op)
    if kind == "arith":
        op = expr[1]
        left = eval_expr(expr[2], row, table, aliases)
        right = eval_expr(expr[3], row, table, aliases)
        if isinstance(left, list) or isinstance(right, list):
            a = _vector(left)
            b = _vector(right)
            if len(a) != len(b):
                raise SQLError(1366, "vector dimension mismatch")
            if op == "+":
                return [x + y for x, y in zip(a, b)]
            if op == "-":
                return [x - y for x, y in zip(a, b)]
            if op == "*":
                return [x * y for x, y in zip(a, b)]
            raise SQLError(1235, "unsupported vector operator")
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        raise SQLError(1235, "unsupported operator")
    if kind == "func":
        name = expr[1].upper()
        args = [eval_expr(arg, row, table, aliases) for arg in expr[2]]
        if name == "L2_DISTANCE":
            if len(args) != 2:
                raise SQLError(1582, "L2_DISTANCE expects two arguments")
            return _l2_distance(args[0], args[1])
        if name == "COSINE_DISTANCE":
            if len(args) != 2:
                raise SQLError(1582, "COSINE_DISTANCE expects two arguments")
            return _cosine_distance(args[0], args[1])
        if name == "INNER_PRODUCT":
            if len(args) != 2:
                raise SQLError(1582, "INNER_PRODUCT expects two arguments")
            return _inner_product(args[0], args[1])
        raise SQLError(1305, f"unsupported function {name}")
    raise SQLError(1235, "unsupported expression")


class Engine:
    def __init__(self, data_dir: str | os.PathLike[str]):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "minivectordb.json"
        self.wal_path = self.data_dir / "minivectordb.wal"
        self.tables: dict[str, Table] = {}
        self.in_transaction = False
        self._backup: dict[str, Table] | None = None
        self._last_inserts: tuple[str, list[dict[str, Any]]] | None = None
        self._load()

    # -- persistence ---------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                self.tables = {
                    name: Table.from_json(value, self.data_dir) for name, value in data.get("tables", {}).items()
                }
            except Exception as exc:
                raise RuntimeError(f"cannot recover data file: {exc}") from exc
        if self.wal_path.exists():
            # A batch insert is one complete line.  A torn final line is ignored.
            with self.wal_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    table = self.tables.get(record.get("table"))
                    if table is None:
                        continue
                    for row in record.get("rows", []):
                        try:
                            normalized = _normalize_loaded_row(table, row)
                            pk = int(normalized[table.primary.name])
                        except Exception:
                            continue
                        if pk not in table.rows:
                            table.rows[pk] = normalized
            # Fold the WAL into a fresh snapshot after recovery.
            self._snapshot()
            try:
                self.wal_path.unlink()
            except FileNotFoundError:
                pass

    def _snapshot(self) -> None:
        payload = {"format": 1, "tables": {name: table.to_json() for name, table in self.tables.items()}}
        fd, tmp_path = tempfile.mkstemp(prefix=".minivectordb-", dir=self.data_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), default=_json_default)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
            directory_flags = getattr(os, "O_DIRECTORY", 0)
            if directory_flags:
                try:
                    directory_fd = os.open(self.data_dir, directory_flags)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    pass
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _save(self) -> None:
        self._snapshot()
        try:
            self.wal_path.unlink()
        except FileNotFoundError:
            pass

    def _append_insert_wal(self, table_name: str, rows: list[dict[str, Any]]) -> None:
        line = json.dumps({"table": table_name, "rows": rows}, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        with self.wal_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    # -- transaction ---------------------------------------------------
    def _begin(self) -> None:
        if self.in_transaction:
            raise SQLError(1568, "transaction already in progress")
        self._backup = copy.deepcopy(self.tables)
        self.in_transaction = True

    def _commit(self) -> None:
        if not self.in_transaction:
            # A COMMIT outside a transaction is accepted as a no-op.
            return
        self._save()
        self.in_transaction = False
        self._backup = None
        self._last_inserts = None

    def _rollback(self) -> None:
        if not self.in_transaction:
            return
        self.tables = self._backup or {}
        self.in_transaction = False
        self._backup = None
        self._last_inserts = None

    def rollback_if_active(self) -> None:
        if self.in_transaction:
            self._rollback()

    def close(self) -> None:
        self.rollback_if_active()
        for table in self.tables.values():
            if isinstance(table.rows, RowStore):
                table.rows.close()

    def _finish_write(self) -> None:
        if not self.in_transaction:
            if self._last_inserts is not None:
                self._append_insert_wal(*self._last_inserts)
                self._last_inserts = None
            else:
                self._save()

    # -- public --------------------------------------------------------
    def execute(self, sql: str) -> Result:
        statement = parse_sql(sql)
        kind = statement["type"]
        self._last_inserts = None
        if kind == "begin":
            self._begin()
            return Result()
        if kind == "commit":
            self._commit()
            return Result()
        if kind == "rollback":
            self._rollback()
            return Result()
        if kind == "noop":
            return Result()
        handler = getattr(self, f"_execute_{kind}", None)
        if handler is None:
            raise SQLError(1235, "unsupported statement")
        result = handler(statement)
        if kind in {"create_table", "drop_table", "insert", "update", "delete", "create_vector_index"}:
            self._finish_write()
        return result

    # -- DDL -----------------------------------------------------------
    def _execute_create_table(self, statement: dict[str, Any]) -> Result:
        name = statement["name"]
        if name in self.tables:
            raise SQLError(1050, f"table {name} already exists")
        columns = [Column(**column) for column in statement["columns"]]
        table = Table(name=name, columns=columns)
        # Force primary lookup validation before accepting the schema.
        table.primary
        self.tables[name] = table
        return Result()

    def _execute_drop_table(self, statement: dict[str, Any]) -> Result:
        name = statement["name"]
        if name not in self.tables:
            raise SQLError(1051, f"unknown table {name}")
        table = self.tables.pop(name)
        if isinstance(table.rows, RowStore):
            try:
                table.rows.path.unlink()
            except FileNotFoundError:
                pass
        return Result()

    def _execute_create_vector_index(self, statement: dict[str, Any]) -> Result:
        table = self._table(statement["table"])
        column = table.column(statement["column"])
        if column.type != "VECTOR":
            raise SQLError(1235, "vector index requires a VECTOR column")
        table.vector_indexes[statement["name"]] = {
            "column": column.name,
            "options": statement.get("options", {}),
        }
        table.touch()
        return Result()

    def _execute_show_tables(self, statement: dict[str, Any]) -> Result:
        return Result(columns=["Tables"], rows=[[name] for name in sorted(self.tables)], is_query=True)

    # -- DML -----------------------------------------------------------
    def _execute_insert(self, statement: dict[str, Any]) -> Result:
        table = self._table(statement["table"])
        columns = (
            [table.column(name) for name in statement["columns"]]
            if statement["columns"]
            else table.columns
        )
        count = 0
        new_rows: list[dict[str, Any]] = []
        seen = set(table.rows)
        for row_expr in statement["rows"]:
            if len(row_expr) != len(columns):
                raise SQLError(1136, "column count does not match value count")
            row: dict[str, Any] = {}
            for column, expr in zip(columns, row_expr):
                row[column.name] = _normalize_value(table, column, eval_expr(expr, {}, table))
            for column in table.columns:
                if column.name not in row:
                    raise SQLError(1364, f"field {column.name} has no default")
            pk = int(row[table.primary.name])
            if pk in seen:
                raise SQLError(1062, f"duplicate primary key {pk}")
            seen.add(pk)
            new_rows.append(row)
        for row in new_rows:
            table.rows[int(row[table.primary.name])] = row
            count += 1
        table.maybe_spill(self.data_dir)
        table.touch()
        self._last_inserts = (table.name, new_rows)
        return Result(affected=count)

    def _execute_update(self, statement: dict[str, Any]) -> Result:
        table = self._table(statement["table"])
        assignments: list[tuple[Column, Any]] = []
        for name, expr in statement["assignments"]:
            column = table.column(name)
            if column.primary:
                raise SQLError(1235, "primary key updates are not supported")
            assignments.append((column, expr))
        targets = self._filter(table, statement.get("where"))
        updates: list[tuple[int, dict[str, Any]]] = []
        for pk in targets:
            row = dict(table.rows[pk])
            for column, expr in assignments:
                row[column.name] = _normalize_value(table, column, eval_expr(expr, table.rows[pk], table))
            updates.append((pk, row))
        for pk, row in updates:
            table.rows[pk] = row
        if updates:
            table.touch()
        return Result(affected=len(updates))

    def _execute_delete(self, statement: dict[str, Any]) -> Result:
        table = self._table(statement["table"])
        targets = self._filter(table, statement.get("where"))
        for pk in targets:
            del table.rows[pk]
        if targets:
            table.touch()
        return Result(affected=len(targets))

    # -- SELECT --------------------------------------------------------
    def _execute_select(self, statement: dict[str, Any]) -> Result:
        table = self._table(statement["table"]) if statement.get("table") else None
        projections = (
            self._expand_projections(table, statement["projections"])
            if table is not None
            else statement["projections"]
        )
        if table is None:
            row: dict[str, Any] = {}
            values = [eval_expr(expr, row, None) for expr, alias in projections]
            return Result(columns=self._projection_columns(projections), rows=[values], is_query=True)
        filtered = self._filter(table, statement.get("where"))
        if projections and all(
            expr[0] == "func" and expr[1].upper() == "COUNT" for expr, alias in projections
        ):
            values: list[Any] = []
            for expr, alias in projections:
                args = expr[2]
                if len(args) != 1:
                    raise SQLError(1582, "COUNT expects one argument")
                if args[0][0] == "star":
                    values.append(len(filtered))
                else:
                    values.append(
                        sum(
                            1
                            for pk in filtered
                            if eval_expr(args[0], table.rows[pk], table) is not None
                        )
                    )
            names = [alias or "count(*)" for expr, alias in projections]
            return Result(columns=names, rows=[values], is_query=True)
        aliases: dict[str, Any] = {}
        for expr, alias in projections:
            if alias:
                aliases[alias.lower()] = expr
        output_columns = self._projection_columns(projections)
        order = self._resolve_order(statement.get("order", []), aliases)
        limit = statement.get("limit")

        vector_plan = self._vector_order_plan(table, order, filtered)
        if vector_plan is not None:
            ordered, distances, distance_expr = vector_plan
            selected = ordered if limit is None else ordered[:limit]
            rows = []
            for pk in selected:
                row = table.rows[pk]
                values = []
                for expr, alias in projections:
                    if expr == distance_expr:
                        values.append(distances[pk])
                    else:
                        values.append(eval_expr(expr, row, table, aliases))
                rows.append(values)
            return Result(columns=output_columns, rows=rows, is_query=True)

        if limit is not None and not order:
            filtered = filtered[: max(0, int(limit))]
        decorated = []
        for pk in filtered:
            row = table.rows[pk]
            decorated.append((pk, row))
        if order:
            decorated.sort(key=lambda item: item[0])
            for expr, descending in reversed(order):
                decorated.sort(
                    key=lambda item: eval_expr(expr, item[1], table, aliases),
                    reverse=descending,
                )
        selected = decorated if limit is None else decorated[:limit]
        rows = []
        for pk, row in selected:
            values = []
            local_aliases = dict(aliases)
            for expr, alias in projections:
                value = eval_expr(expr, row, table, local_aliases)
                values.append(value)
                if alias:
                    local_aliases[alias.lower()] = value
            rows.append(values)
        return Result(columns=output_columns, rows=rows, is_query=True)

    def _expand_projections(
        self, table: Table, projections: list[tuple[Any, str | None]]
    ) -> list[tuple[Any, str | None]]:
        expanded: list[tuple[Any, str | None]] = []
        for expr, alias in projections:
            if expr[0] == "star":
                expanded.extend((("column", column.name), None) for column in table.columns)
            else:
                expanded.append((expr, alias))
        return expanded

    def _projection_columns(self, projections: list[tuple[Any, str | None]]) -> list[str]:
        names: list[str] = []
        for expr, alias in projections:
            if alias:
                names.append(alias)
            elif expr[0] == "column":
                names.append(expr[1])
            elif expr[0] == "func":
                names.append(expr[1].lower())
            else:
                names.append("expr")
        return names

    def _resolve_order(self, order: list[tuple[Any, bool]], aliases: dict[str, Any]):
        resolved = []
        for expr, descending in order:
            if expr[0] == "column" and expr[1].lower() in aliases:
                resolved.append((aliases[expr[1].lower()], descending))
            else:
                resolved.append((expr, descending))
        return resolved

    def _table(self, name: str) -> Table:
        if name not in self.tables:
            raise SQLError(1146, f"unknown table {name}")
        return self.tables[name]

    def _filter(self, table: Table, where: Any) -> list[int]:
        if where is None:
            return sorted(table.rows)
        result = []
        for pk in sorted(table.rows):
            if eval_expr(where, table.rows[pk], table):
                result.append(pk)
        return result

    # -- vector top-k ---------------------------------------------------
    def _is_distance_expr(self, expr: Any, target: Any) -> bool:
        return isinstance(target, tuple) and expr == target

    def _extract_distance(self, table: Table, expr: Any) -> tuple[Column, list[float]] | None:
        if not isinstance(expr, tuple) or expr[0] != "func" or expr[1].upper() != "L2_DISTANCE":
            return None
        if len(expr[2]) != 2:
            return None
        left, right = expr[2]
        column = None
        vector = None
        if left[0] == "column":
            try:
                candidate = table.column(left[1])
            except SQLError:
                candidate = None
            if candidate and candidate.type == "VECTOR":
                column = candidate
                if right[0] in {"vector", "literal"} and isinstance(right[1], (list, tuple)):
                    vector = [float(x) for x in right[1]]
        if column is None and right[0] == "column":
            try:
                candidate = table.column(right[1])
            except SQLError:
                candidate = None
            if candidate and candidate.type == "VECTOR":
                column = candidate
                if left[0] in {"vector", "literal"} and isinstance(left[1], (list, tuple)):
                    vector = [float(x) for x in left[1]]
        if column is None or vector is None or len(vector) != (column.length or -1):
            return None
        return column, vector

    def _vector_order_plan(self, table, order, filtered):
        if not order or len(order) != 1:
            return None
        expr, descending = order[0]
        if descending:
            return None
        distance = self._extract_distance(table, expr)
        if distance is None:
            return None
        column, query = distance
        matrix, pks = table.vector_matrix(column)
        allowed = set(filtered)
        if np is not None and isinstance(matrix, np.ndarray):
            query_vec = np.asarray(query, dtype=np.float32)
            values = np.linalg.norm(matrix - query_vec, axis=1)
            distances = {pk: float(value) for pk, value in zip(pks, values)}
        else:
            # Large tables (and the no-NumPy fallback) avoid a full matrix.
            # Compute only the rows that can survive the WHERE clause.
            distances = {
                pk: _l2_distance(table.rows[pk][column.name], query) for pk in allowed
            }
        ordered = sorted(allowed, key=lambda pk: (distances[pk], pk))
        return ordered, distances, expr
