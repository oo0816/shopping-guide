"""Import all local product JSON, SKUs and knowledge passages without API calls."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.catalog import DEFAULT_DB, DEFAULT_SOURCE, import_catalog


def main():
    parser = argparse.ArgumentParser(description="导入商品、SKU 和 RAG 片段；离线运行。")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    try:
        result = import_catalog(args.source)
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        parser.exit(2, "导入失败：" + str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("商品库：" + str(DEFAULT_DB))
    print("商品资料按可信处理；没有调用 LLM 或 Embedding。")


if __name__ == "__main__":
    main()
