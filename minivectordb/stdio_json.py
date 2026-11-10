"""Official MiniVectorDB JSON-lines stdio protocol.

The judge sends one JSON object per line and expects one JSON response per
line.  Nothing else may be written to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .engine import Engine, Result
from .sql import SQLError

_ERROR_CODES = {
    1050: "table_exists",
    1051: "unknown_table",
    1054: "unknown_column",
    1062: "duplicate_key",
    1064: "syntax_error",
    1146: "unknown_table",
    1235: "unsupported_sql",
    1264: "value_out_of_range",
    1366: "dimension_mismatch",
    1406: "value_too_long",
    1690: "invalid_vector",
}


def _error_code(exc: SQLError) -> str:
    return _ERROR_CODES.get(exc.code, "sql_error")


def _statement_word(sql: str) -> str:
    stripped = sql.lstrip()
    if not stripped:
        return ""
    return stripped.split(None, 1)[0].upper()


def _result_message(ident: Any, result: Result, sql: str) -> dict[str, Any]:
    if result.is_query:
        return {"id": ident, "ok": True, "columns": list(result.columns), "rows": result.rows}
    word = _statement_word(sql)
    if word in {"INSERT", "UPDATE", "DELETE"}:
        return {"id": ident, "ok": True, "affected_rows": int(result.affected)}
    return {"id": ident, "ok": True}


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


class JSONLServer:
    def __init__(self, engine: Engine, stdin=None, stdout=None):
        self.engine = engine
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout

    def emit(self, message: dict[str, Any]) -> None:
        self.stdout.write(
            json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        )
        self.stdout.flush()

    def serve(self) -> int:
        self.emit({"type": "ready"})
        for raw_line in self.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except Exception as exc:
                self.emit({"id": None, "ok": False, "error": {"code": "invalid_json", "message": str(exc)}})
                continue
            if not isinstance(request, dict):
                self.emit({"id": None, "ok": False, "error": {"code": "invalid_request", "message": "object required"}})
                continue
            ident = request.get("id")
            op = request.get("op")
            if op == "shutdown":
                self.engine.rollback_if_active()
                self.emit({"id": ident, "ok": True})
                return 0
            if op != "execute":
                self.emit(
                    {
                        "id": ident,
                        "ok": False,
                        "error": {"code": "unsupported_op", "message": "unsupported operation"},
                    }
                )
                continue
            sql = request.get("sql")
            if not isinstance(sql, str):
                self.emit(
                    {
                        "id": ident,
                        "ok": False,
                        "error": {"code": "invalid_request", "message": "sql must be a string"},
                    }
                )
                continue
            try:
                result = self.engine.execute(sql)
                self.emit(_result_message(ident, result, sql))
            except SQLError as exc:
                self.emit(
                    {
                        "id": ident,
                        "ok": False,
                        "error": {"code": _error_code(exc), "message": exc.message},
                    }
                )
            except Exception as exc:  # never kill the protocol loop on a SQL bug
                self.emit(
                    {
                        "id": ident,
                        "ok": False,
                        "error": {"code": "internal_error", "message": str(exc)},
                    }
                )
        self.engine.rollback_if_active()
        return 0
