"""MiniVectorDB command line entry point."""

from __future__ import annotations

import argparse
import os
import sys

from .engine import Engine
from .stdio_json import JSONLServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MiniVectorDB")
    parser.add_argument("--data-dir", default=os.environ.get("VECTORDB_DATA_DIR", "/data"))
    parser.add_argument(
        "--protocol",
        choices=("jsonl",),
        default=os.environ.get("MINIVECTORDB_PROTOCOL", "jsonl"),
        help="official stdin/stdout protocol",
    )
    args = parser.parse_args(argv)
    try:
        engine = Engine(args.data_dir)
    except Exception as exc:
        sys.stderr.write(f"minivectordb: cannot open data directory: {exc}\n")
        return 2
    sys.stderr.write("minivectordb: ready\n")
    sys.stderr.flush()
    return JSONLServer(engine).serve()


if __name__ == "__main__":
    raise SystemExit(main())
