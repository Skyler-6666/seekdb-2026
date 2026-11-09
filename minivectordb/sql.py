"""Small SQL tokenizer/parser used by MiniVectorDB.

The grammar is intentionally limited to the task SQL subset.  The parser
returns plain tuples/dicts so the storage layer stays independent from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class SQLError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int


_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<line_comment>--[^\n]*)
  | (?P<block_comment>/\*.*?\*/)
  | (?P<string>'(?:[^'\\]|\\.|'')*'|"(?:[^"\\]|\\.|"")*")
  | (?P<number>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<identifier>[A-Za-z_][A-Za-z0-9_$]*)
  | (?P<operator><=|>=|<>|!=|==|=|<|>)
  | (?P<punct>[(),.;*\[\]])
    """,
    re.VERBOSE | re.DOTALL,
)


def tokenize(sql: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(sql):
        match = _TOKEN_RE.match(sql, pos)
        if not match:
            raise SQLError(1064, f"unexpected character at {pos}")
        kind = match.lastgroup or ""
        value = match.group(0)
        if kind not in {"space", "line_comment", "block_comment"}:
            tokens.append(Token(kind, value, pos))
        pos = match.end()
    tokens.append(Token("eof", "", len(sql)))
    return tokens


def _unquote(token: Token) -> str:
    value = token.value
    quote = value[0]
    body = value[1:-1]
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            mapping = {"n": "\n", "r": "\r", "t": "\t", "0": "\0"}
            out.append(mapping.get(nxt, nxt))
            i += 2
        elif ch == quote and i + 1 < len(body) and body[i + 1] == quote:
            out.append(quote)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _parse_vector_text(text: str) -> list[float]:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise SQLError(1366, "invalid vector literal")
    body = stripped[1:-1].strip()
    if not body:
        return []
    values: list[float] = []
    for part in body.split(","):
        part = part.strip()
        try:
            number = float(part)
        except ValueError as exc:
            raise SQLError(1366, "invalid vector element") from exc
        values.append(number)
    return values


class Parser:
    def __init__(self, sql: str):
        self.sql = sql
        self.tokens = tokenize(sql)
        self.index = 0

    # -- token helpers -------------------------------------------------
    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def next(self) -> Token:
        token = self.peek()
        self.index += 1
        return token

    def at_keyword(self, *words: str) -> bool:
        token = self.peek()
        return token.kind == "identifier" and token.value.upper() in {w.upper() for w in words}

    def accept_keyword(self, *words: str) -> bool:
        if self.at_keyword(*words):
            self.next()
            return True
        return False

    def expect_keyword(self, word: str) -> None:
        if not self.accept_keyword(word):
            raise SQLError(1064, f"expected {word}")

    def at(self, value: str) -> bool:
        return self.peek().value == value

    def accept(self, value: str) -> bool:
        if self.at(value):
            self.next()
            return True
        return False

    def expect(self, value: str) -> None:
        if not self.accept(value):
            raise SQLError(1064, f"expected {value}")

    def identifier(self) -> str:
        token = self.peek()
        if token.kind != "identifier":
            raise SQLError(1064, "expected identifier")
        self.next()
        return token.value

    def end(self) -> None:
        self.accept(";")
        if self.peek().kind != "eof":
            raise SQLError(1064, "unexpected token after statement")

    # -- entry ---------------------------------------------------------
    def parse(self) -> dict[str, Any]:
        token = self.peek()
        if token.kind != "identifier":
            raise SQLError(1064, "expected statement")
        word = token.value.upper()
        if word == "CREATE":
            return self.parse_create()
        if word == "INSERT":
            return self.parse_insert()
        if word == "SELECT":
            return self.parse_select()
        if word == "UPDATE":
            return self.parse_update()
        if word == "DELETE":
            return self.parse_delete()
        if word in {"BEGIN", "START"}:
            self.next()
            if word == "START":
                self.expect_keyword("TRANSACTION")
            self.end()
            return {"type": "begin"}
        if word == "COMMIT":
            self.next()
            self.end()
            return {"type": "commit"}
        if word == "ROLLBACK":
            self.next()
            self.end()
            return {"type": "rollback"}
        if word == "DROP":
            return self.parse_drop()
        if word == "SHOW":
            return self.parse_show()
        if word == "USE":
            self.next()
            self.identifier()
            self.end()
            return {"type": "noop"}
        if word == "SET":
            # Session settings are irrelevant to the single-connection engine.
            while self.peek().kind != "eof" and not self.at(";"):
                self.next()
            self.end()
            return {"type": "noop"}
        raise SQLError(1235, f"unsupported statement: {word}")

    # -- DDL -----------------------------------------------------------
    def parse_create(self) -> dict[str, Any]:
        self.expect_keyword("CREATE")
        if self.accept_keyword("TABLE"):
            if self.accept_keyword("IF"):
                self.expect_keyword("NOT")
                self.expect_keyword("EXISTS")
            name = self.identifier()
            self.expect("(")
            columns: list[dict[str, Any]] = []
            primary: str | None = None
            while True:
                if self.at_keyword("PRIMARY"):
                    self.next()
                    self.expect_keyword("KEY")
                    self.expect("(")
                    primary = self.identifier()
                    self.expect(")")
                else:
                    col_name = self.identifier()
                    type_token = self.peek()
                    if type_token.kind != "identifier":
                        raise SQLError(1064, "expected column type")
                    type_name = self.next().value.upper()
                    length = None
                    if self.accept("("):
                        num = self.next()
                        if num.kind != "number" or "." in num.value:
                            raise SQLError(1064, "invalid type length")
                        length = int(num.value)
                        self.expect(")")
                    if type_name not in {"BIGINT", "INT", "INTEGER", "VARCHAR", "VECTOR", "TEXT"}:
                        raise SQLError(1235, f"unsupported type {type_name}")
                    if type_name in {"VARCHAR", "VECTOR"} and (length is None or length <= 0):
                        raise SQLError(1064, f"{type_name} requires a positive length")
                    is_primary = False
                    while not self.at(",") and not self.at(")"):
                        if self.accept_keyword("PRIMARY"):
                            self.expect_keyword("KEY")
                            is_primary = True
                        elif self.accept_keyword("NOT"):
                            self.expect_keyword("NULL")
                        elif self.accept_keyword("NULL"):
                            pass
                        else:
                            raise SQLError(1064, "unsupported column attribute")
                    columns.append(
                        {
                            "name": col_name,
                            "type": type_name,
                            "length": length,
                            "primary": is_primary,
                        }
                    )
                    if is_primary:
                        primary = col_name
                if self.accept(","):
                    continue
                self.expect(")")
                break
            if primary is None:
                raise SQLError(1075, "table must have exactly one primary key")
            pk_count = sum(1 for c in columns if c["name"] == primary)
            if pk_count != 1:
                raise SQLError(1075, "invalid primary key")
            for c in columns:
                c["primary"] = c["name"] == primary
                if c["primary"] and c["type"] not in {"BIGINT", "INT", "INTEGER"}:
                    raise SQLError(1075, "primary key must be BIGINT")
            if sum(1 for c in columns if c["type"] == "VECTOR") > 1:
                raise SQLError(1235, "at most one VECTOR column per table")
            self.end()
            return {"type": "create_table", "name": name, "columns": columns, "primary": primary}
        if self.accept_keyword("VECTOR"):
            self.expect_keyword("INDEX")
            index_name = self.identifier()
            self.expect_keyword("ON")
            table = self.identifier()
            self.expect("(")
            column = self.identifier()
            self.expect(")")
            options: dict[str, str] = {}
            if self.accept_keyword("WITH"):
                self.expect("(")
                while True:
                    key = self.identifier().lower()
                    self.expect("=")
                    value = self.next()
                    if value.kind not in {"identifier", "number", "string"}:
                        raise SQLError(1064, "invalid index option")
                    options[key] = value.value
                    if not self.accept(","):
                        break
                self.expect(")")
            self.end()
            return {
                "type": "create_vector_index",
                "name": index_name,
                "table": table,
                "column": column,
                "options": options,
            }
        raise SQLError(1235, "unsupported CREATE statement")

    def parse_drop(self) -> dict[str, Any]:
        self.expect_keyword("DROP")
        self.expect_keyword("TABLE")
        if self.accept_keyword("IF"):
            self.expect_keyword("EXISTS")
        name = self.identifier()
        self.end()
        return {"type": "drop_table", "name": name}

    def parse_show(self) -> dict[str, Any]:
        self.expect_keyword("SHOW")
        if self.accept_keyword("TABLES"):
            self.end()
            return {"type": "show_tables"}
        raise SQLError(1235, "unsupported SHOW")

    # -- DML -----------------------------------------------------------
    def parse_insert(self) -> dict[str, Any]:
        self.expect_keyword("INSERT")
        self.expect_keyword("INTO")
        table = self.identifier()
        columns: list[str] | None = None
        if self.accept("("):
            columns = []
            while True:
                columns.append(self.identifier())
                if not self.accept(","):
                    break
            self.expect(")")
        self.expect_keyword("VALUES")
        rows: list[list[Any]] = []
        while True:
            self.expect("(")
            values: list[Any] = []
            while True:
                values.append(self.parse_expr())
                if not self.accept(","):
                    break
            self.expect(")")
            rows.append(values)
            if not self.accept(","):
                break
        self.end()
        return {"type": "insert", "table": table, "columns": columns, "rows": rows}

    def parse_select(self) -> dict[str, Any]:
        self.expect_keyword("SELECT")
        projections: list[tuple[Any, str | None]] = []
        if self.accept("*"):
            projections.append((("star",), None))
        else:
            while True:
                expr = self.parse_expr()
                alias = None
                if self.accept_keyword("AS"):
                    alias = self.identifier()
                elif self.peek().kind == "identifier" and not self.at_keyword(
                    "FROM", "WHERE", "ORDER", "LIMIT", "AND", "OR"
                ):
                    alias = self.identifier()
                projections.append((expr, alias))
                if not self.accept(","):
                    break
        table = None
        if self.accept_keyword("FROM"):
            table = self.identifier()
        where = None
        if self.accept_keyword("WHERE"):
            where = self.parse_expr()
        order: list[tuple[Any, bool]] = []
        if self.accept_keyword("ORDER"):
            self.expect_keyword("BY")
            while True:
                expr = self.parse_expr()
                descending = False
                if self.accept_keyword("DESC"):
                    descending = True
                elif self.accept_keyword("ASC"):
                    descending = False
                order.append((expr, descending))
                if not self.accept(","):
                    break
        limit = None
        if self.accept_keyword("LIMIT"):
            token = self.next()
            if token.kind != "number" or "." in token.value:
                raise SQLError(1064, "LIMIT must be an integer")
            limit = int(token.value)
            if limit < 0:
                raise SQLError(1064, "LIMIT must be non-negative")
        self.end()
        return {
            "type": "select",
            "table": table,
            "projections": projections,
            "where": where,
            "order": order,
            "limit": limit,
        }

    def parse_update(self) -> dict[str, Any]:
        self.expect_keyword("UPDATE")
        table = self.identifier()
        self.expect_keyword("SET")
        assignments: list[tuple[str, Any]] = []
        while True:
            column = self.identifier()
            self.expect("=")
            assignments.append((column, self.parse_expr()))
            if not self.accept(","):
                break
        where = None
        if self.accept_keyword("WHERE"):
            where = self.parse_expr()
        self.end()
        return {"type": "update", "table": table, "assignments": assignments, "where": where}

    def parse_delete(self) -> dict[str, Any]:
        self.expect_keyword("DELETE")
        self.expect_keyword("FROM")
        table = self.identifier()
        where = None
        if self.accept_keyword("WHERE"):
            where = self.parse_expr()
        self.end()
        return {"type": "delete", "table": table, "where": where}

    # -- expressions ---------------------------------------------------
    def parse_expr(self) -> Any:
        return self.parse_or()

    def parse_or(self) -> Any:
        left = self.parse_and()
        while self.accept_keyword("OR"):
            left = ("or", left, self.parse_and())
        return left

    def parse_and(self) -> Any:
        left = self.parse_comparison()
        while self.accept_keyword("AND"):
            left = ("and", left, self.parse_comparison())
        return left

    def parse_comparison(self) -> Any:
        left = self.parse_additive()
        token = self.peek()
        if token.value in {"=", "==", "<", "<=", ">", ">=", "!=", "<>"}:
            self.next()
            return ("compare", token.value, left, self.parse_additive())
        return left

    def parse_additive(self) -> Any:
        left = self.parse_multiplicative()
        while self.peek().value in {"+", "-"}:
            op = self.next().value
            left = ("arith", op, left, self.parse_multiplicative())
        return left

    def parse_multiplicative(self) -> Any:
        left = self.parse_primary()
        while self.peek().value == "*":
            self.next()
            left = ("arith", "*", left, self.parse_primary())
        return left

    def parse_primary(self) -> Any:
        token = self.peek()
        if token.kind == "number":
            self.next()
            text = token.value
            if "." in text or "e" in text.lower():
                return ("literal", float(text))
            return ("literal", int(text))
        if token.kind == "string":
            self.next()
            text = _unquote(token)
            if text.strip().startswith("[") and text.strip().endswith("]"):
                return ("vector", _parse_vector_text(text))
            return ("literal", text)
        if self.accept("["):
            values: list[float] = []
            if not self.at("]"):
                while True:
                    item = self.next()
                    if item.kind != "number":
                        raise SQLError(1366, "invalid vector element")
                    values.append(float(item.value))
                    if not self.accept(","):
                        break
            self.expect("]")
            return ("vector", values)
        if self.accept("*"):
            return ("star",)
        if self.accept("("):
            expr = self.parse_expr()
            self.expect(")")
            return expr
        if token.kind == "identifier":
            name = self.next().value
            if self.accept("("):
                args = []
                if not self.at(")"):
                    while True:
                        args.append(self.parse_expr())
                        if not self.accept(","):
                            break
                self.expect(")")
                return ("func", name.upper(), args)
            return ("column", name)
        raise SQLError(1064, f"unexpected token {token.value!r}")


def parse_sql(sql: str) -> dict[str, Any]:
    return Parser(sql).parse()


def split_statements(sql: str) -> list[str]:
    """Split a multi-statement text while respecting quotes and comments."""
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(sql):
        ch = sql[i]
        if quote:
            current.append(ch)
            if ch == "\\" and i + 1 < len(sql):
                current.append(sql[i + 1])
                i += 2
                continue
            if ch == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    current.append(sql[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            current.append(ch)
        elif ch == ";":
            text = "".join(current).strip()
            if text:
                statements.append(text)
            current.clear()
        else:
            current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements
