"""Minimal MySQL wire protocol server over stdin/stdout.

The competition drives the process through pipes instead of a TCP socket.  A
small, standards-compliant MySQL server (protocol 4.1) is implemented here so
ordinary MySQL-compatible clients can talk to the database without a listener.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass

from .engine import Engine, Result
from .sql import SQLError, split_statements

CLIENT_LONG_PASSWORD = 0x00000001
CLIENT_FOUND_ROWS = 0x00000002
CLIENT_LONG_FLAG = 0x00000004
CLIENT_CONNECT_WITH_DB = 0x00000008
CLIENT_PROTOCOL_41 = 0x00000200
CLIENT_TRANSACTIONS = 0x00002000
CLIENT_SECURE_CONNECTION = 0x00008000
CLIENT_MULTI_RESULTS = 0x00020000
CLIENT_PLUGIN_AUTH = 0x00080000
CLIENT_CONNECT_ATTRS = 0x00100000
CLIENT_PLUGIN_AUTH_LENENC_CLIENT_DATA = 0x00200000

SERVER_CAPABILITIES = (
    CLIENT_LONG_PASSWORD
    | CLIENT_FOUND_ROWS
    | CLIENT_LONG_FLAG
    | CLIENT_CONNECT_WITH_DB
    | CLIENT_PROTOCOL_41
    | CLIENT_TRANSACTIONS
    | CLIENT_SECURE_CONNECTION
    | CLIENT_MULTI_RESULTS
    | CLIENT_PLUGIN_AUTH
    | CLIENT_CONNECT_ATTRS
    | CLIENT_PLUGIN_AUTH_LENENC_CLIENT_DATA
)

SERVER_STATUS = 0x0002
CHARSET_UTF8 = 33

COLUMN_TYPE_LONG = 8
COLUMN_TYPE_DOUBLE = 5
COLUMN_TYPE_VAR_STRING = 253


def _lenenc_int(value: int) -> bytes:
    if value < 0xFB:
        return bytes([value])
    if value <= 0xFFFF:
        return b"\xfc" + struct.pack("<H", value)
    if value <= 0xFFFFFF:
        return b"\xfd" + struct.pack("<I", value)[:3]
    return b"\xfe" + struct.pack("<Q", value)


def _lenenc_str(value: bytes) -> bytes:
    return _lenenc_int(len(value)) + value


def _read_exact(stream, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


@dataclass
class PacketIO:
    reader: object
    writer: object
    sequence: int = 0

    def read_packet(self) -> tuple[int, bytes]:
        header = _read_exact(self.reader, 4)
        length = header[0] | (header[1] << 8) | (header[2] << 16)
        sequence = header[3]
        payload = _read_exact(self.reader, length)
        return sequence, payload

    def write_packet(self, payload: bytes, sequence: int | None = None) -> None:
        if sequence is None:
            sequence = self.sequence
        header = struct.pack("<I", len(payload))[:3] + bytes([sequence & 0xFF])
        self.writer.write(header + payload)
        self.writer.flush()
        self.sequence = (sequence + 1) & 0xFF


def _ok(affected: int = 0, last_id: int = 0) -> bytes:
    return b"\x00" + _lenenc_int(affected) + _lenenc_int(last_id) + struct.pack("<HH", SERVER_STATUS, 0)


def _eof() -> bytes:
    return b"\xfe" + struct.pack("<HH", 0, SERVER_STATUS)


def _error(code: int, message: str) -> bytes:
    return b"\xff" + struct.pack("<H", code) + b"#42000" + message.encode("utf-8", "replace")


def _column_type(values) -> int:
    for value in values:
        if isinstance(value, bool):
            return COLUMN_TYPE_LONG
        if isinstance(value, int):
            return COLUMN_TYPE_LONG
        if isinstance(value, float):
            return COLUMN_TYPE_DOUBLE
        if value is not None:
            return COLUMN_TYPE_VAR_STRING
    return COLUMN_TYPE_VAR_STRING


def _value_bytes(value) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return b"1" if value else b"0"
    if isinstance(value, float):
        return format(value, ".10g").encode("ascii")
    if isinstance(value, list):
        return ("[" + ",".join(format(float(item), ".10g") for item in value) + "]").encode("ascii")
    return str(value).encode("utf-8")


class MySQLPipeServer:
    def __init__(self, engine: Engine, reader=None, writer=None):
        self.engine = engine
        self.reader = reader or sys.stdin.buffer
        self.writer = writer or sys.stdout.buffer
        self.io = PacketIO(self.reader, self.writer)

    def serve(self) -> None:
        self._handshake()
        while True:
            try:
                sequence, payload = self.io.read_packet()
            except EOFError:
                return
            if not payload:
                continue
            command = payload[0]
            body = payload[1:]
            if command == 0x01:  # COM_QUIT
                return
            if command == 0x0E:  # COM_PING
                self.io.write_packet(_ok(), sequence=1)
                continue
            if command == 0x02:  # COM_INIT_DB
                self.io.write_packet(_ok(), sequence=1)
                continue
            if command == 0x03:  # COM_QUERY
                self._query(body.decode("utf-8", "replace"))
                continue
            self.io.write_packet(_error(1047, "unsupported command"), sequence=1)

    def _handshake(self) -> None:
        salt = b"MiniVectorSalt123456"
        capabilities = SERVER_CAPABILITIES
        payload = b"\x0a" + b"5.7.0-MiniVectorDB\x00"
        payload += struct.pack("<I", 1)
        payload += salt[:8] + b"\x00"
        payload += struct.pack("<H", capabilities & 0xFFFF)
        payload += bytes([CHARSET_UTF8])
        payload += struct.pack("<H", SERVER_STATUS)
        payload += struct.pack("<H", (capabilities >> 16) & 0xFFFF)
        payload += bytes([21]) + b"\x00" * 10
        payload += salt[8:] + b"\x00"
        payload += b"mysql_native_password\x00"
        self.io.write_packet(payload, sequence=0)
        try:
            _, response = self.io.read_packet()
        except EOFError:
            raise
        if len(response) >= 4 and (response[0] | (response[1] << 8)) & 0x00000800:
            # SSL is not available because transport is a pipe.
            self.io.write_packet(_error(1043, "SSL is not supported"), sequence=2)
            raise ConnectionError("SSL is not supported")
        self.io.write_packet(_ok(), sequence=2)

    def _query(self, sql: str) -> None:
        statements = split_statements(sql)
        if not statements:
            self.io.write_packet(_ok(), sequence=1)
            return
        result: Result | None = None
        try:
            for statement in statements:
                result = self.engine.execute(statement)
        except SQLError as exc:
            self.io.write_packet(_error(exc.code, exc.message), sequence=1)
            return
        except Exception as exc:  # defensive: never crash the pipe server
            self.io.write_packet(_error(1105, str(exc)), sequence=1)
            return
        assert result is not None
        if not result.is_query:
            self.io.write_packet(_ok(result.affected), sequence=1)
            return
        self._send_result(result)

    def _send_result(self, result: Result) -> None:
        columns = result.columns
        sequence = 1
        self.io.write_packet(_lenenc_int(len(columns)), sequence=sequence)
        sequence += 1
        for index, name in enumerate(columns):
            values = [row[index] for row in result.rows[:1]]
            column_type = _column_type(values)
            fixed = struct.pack("<H", CHARSET_UTF8) + struct.pack("<I", 65535)
            fixed += bytes([column_type]) + struct.pack("<H", 0) + bytes([0]) + b"\x00\x00"
            payload = (
                _lenenc_str(b"def")
                + _lenenc_str(b"")
                + _lenenc_str(b"")
                + _lenenc_str(b"")
                + _lenenc_str(name.encode("utf-8"))
                + _lenenc_str(b"")
                + b"\x0c"
                + fixed
            )
            self.io.write_packet(payload, sequence=sequence)
            sequence += 1
        self.io.write_packet(_eof(), sequence=sequence)
        sequence += 1
        for row in result.rows:
            payload = b""
            for value in row:
                data = _value_bytes(value)
                payload += b"\xfb" if data is None else _lenenc_str(data)
            self.io.write_packet(payload, sequence=sequence)
            sequence += 1
        self.io.write_packet(_eof(), sequence=sequence)


def run_text_protocol(engine: Engine, reader=None, writer=None) -> None:
    """Line-oriented fallback used when the driver sets the text protocol.

    This is intentionally simple and never selected unless requested through
    ``MINIVECTORDB_PROTOCOL=text``.  It is useful for local smoke tests.
    """

    reader = reader or sys.stdin.buffer
    writer = writer or sys.stdout.buffer
    writer.write(b"READY\n")
    writer.flush()
    buffer = b""
    while True:
        chunk = reader.read(4096)
        if not chunk:
            return
        buffer += chunk
        while True:
            index = buffer.find(b";")
            if index < 0:
                break
            sql = buffer[: index + 1].decode("utf-8", "replace")
            buffer = buffer[index + 1 :]
            try:
                result = engine.execute(sql)
            except SQLError as exc:
                writer.write(f"ERROR {exc.code} {exc.message}\n".encode("utf-8"))
                writer.flush()
                continue
            if result.is_query:
                writer.write(("\t".join(result.columns) + "\n").encode("utf-8"))
                for row in result.rows:
                    writer.write(("\t".join(_text_value(value) for value in row) + "\n").encode("utf-8"))
                writer.write(b"OK 0\n")
            else:
                writer.write(f"OK {result.affected}\n".encode("ascii"))
            writer.flush()


def _text_value(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return format(value, ".10g")
    if isinstance(value, list):
        return "[" + ",".join(format(float(item), ".10g") for item in value) + "]"
    return str(value)
