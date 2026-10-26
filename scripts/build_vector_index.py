"""Build/rebuild the local vector index; API requests require --online."""

import argparse
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embedding import EmbeddingClient, EmbeddingError, load_config
from app.corpus import index_path, load_dataset
from app.vector_retrieval import build_index


def main():
    parser = argparse.ArgumentParser(description="生成本地向量索引；默认只使用已有缓存。")
    parser.add_argument("--online", action="store_true", help="允许为缺少缓存的文本请求 Embedding API")
    parser.add_argument("--refresh", action="store_true", help="重新请求所有片段，必须同时指定 --online")
    parser.add_argument("--dataset", choices=("original", "products", "all"), default="original",
                        help="original 保留原有电脑索引；products/all 为商品库建立独立索引")
    args = parser.parse_args()
    if args.refresh and not args.online:
        parser.error("--refresh 需要 --online，会产生新的 API 用量")
    try:
        client = EmbeddingClient(load_config())
        chunks, _ = load_dataset(args.dataset)
        output = index_path(args.dataset)
        index = build_index(chunks, client, path=output, online=args.online, refresh=args.refresh)
    except EmbeddingError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, sqlite3.Error, OverflowError):
        print("索引构建失败，请检查本机配置、片段及缓存。", file=sys.stderr)
        return 2
    print(f"向量索引已生成：{len(index['vectors'])} 个片段，{client.config.dimensions} 维。")
    print(f"复用缓存：{client.stats['cache_hits']}；新向量：{client.stats['embedded_texts']}；API 请求：{client.stats['requests']}。")
    print("存储位置：" + str(output) + "；没有调用 LLM。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
