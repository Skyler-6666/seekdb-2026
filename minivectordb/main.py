"""MiniVectorDB command line entry point."""

from __future__ import annotations

import argparse
import os
import sys

from .engine import Engine
from .mysql_wire import MySQLPipeServer, run_text_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MiniVectorDB")
    parser.add_argument("--data-dir", default=os.environ.get("MINIVECTORDB_DATA_DIR", "/data"))
    parser.add_argument(
        "--protocol",
        choices=("mysql", "text"),
        default=os.environ.get("MINIVECTORDB_PROTOCOL", "mysql"),
    )
    args = parser.parse_args(argv)
    try:
        engine = Engine(args.data_dir)
    except Exception as exc:
        sys.stderr.write(f"minivectordb: cannot open data directory: {exc}\n")
        return 2
    sys.stderr.write("minivectordb: ready\n")
    sys.stderr.flush()
    if args.protocol == "text":
        run_text_protocol(engine)
    else:
        MySQLPipeServer(engine).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
