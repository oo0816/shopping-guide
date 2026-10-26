"""Evaluate keyword or vector retrieval on the labeled development set."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from pathlib import Path

from app.retrieval import B, K1, ROOT, KeywordRetriever, load_chunks
from app.embedding import EmbeddingClient, load_config
from app.vector_retrieval import VectorRetriever


def evaluate(retriever, cases, top_k):
    known = {row["chunk_id"] for row in retriever.chunks}
    rows, supported, negatives = [], [], []
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("评测集为空或用例 ID 重复。")
    for case in cases:
        gold = set(case["expected_chunk_ids"])
        label = case["answerability"]
        if label not in ("supported", "not_supported") or bool(gold) != (label == "supported"):
            raise ValueError("评测标签与证据标注不一致。")
        if gold - known:
            raise ValueError("评测用例引用了不存在的片段。")
        result = retriever.search(case["query"], top_k)
        ids = [hit["chunk_id"] for hit in result["hits"]]
        row = {"id": case["id"], "query": case["query"], "answerability": label,
               "expected_chunk_ids": sorted(gold), "retrieved_chunk_ids": ids,
               "retrieval_status": result["status"]}
        if gold:
            row["recall_at_1"] = len(gold.intersection(ids[:1])) / len(gold)
            row["recall_at_k"] = len(gold.intersection(ids)) / len(gold)
            supported.append(row)
        else:
            row["no_hits"] = not ids
            negatives.append(row)
        rows.append(row)
    return {
        "supported_count": len(supported), "not_supported_count": len(negatives),
        "recall_at_1": sum(row["recall_at_1"] for row in supported) / len(supported) if supported else None,
        "recall_at_k": sum(row["recall_at_k"] for row in supported) / len(supported) if supported else None,
        "unsupported_no_hit_rate": sum(row["no_hits"] for row in negatives) / len(negatives) if negatives else None,
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="检索开发集评测；默认关键词离线，向量请求需要 --online。")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--method", choices=("keyword", "vector"), default="keyword")
    parser.add_argument("--online", action="store_true", help="允许补充缺少的查询向量，产生 API 用量")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k 必须在 1～20 之间")
    corpus = ROOT / "data/processed/chunks.jsonl"
    dataset = ROOT / "evaluations/retrieval_cases.jsonl"
    try:
        cases = [json.loads(line) for line in dataset.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        chunks = load_chunks(corpus)
        client = EmbeddingClient(load_config()) if args.method == "vector" else None
        retriever = (VectorRetriever(chunks, client, online=args.online) if client else KeywordRetriever(chunks))
        metrics = evaluate(retriever, cases, args.top_k)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, OverflowError) as exc:
        parser.exit(2, "评测失败：" + str(exc) + "\n")
    report = {
        "evaluation_kind": args.method + "_retrieval_development_set",
        "method": args.method,
        "embedding_profile": client.config.profile if client else None,
        "embedding_stats": dict(client.stats) if client else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k, "bm25_parameters": {"k1": K1, "b": B},
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "limitations": [
            "三个型号、八个开发用例；不是独立最终测试集。",
            "资料不足问题的无命中率只描述检索行为，不代表回答拒答准确率。",
            "纯向量最近邻可能为无答案问题返回候选，必须另外核验回答依据。",
            "未测试生成回答或真实选购效果。",
        ], **metrics,
    }
    output = ROOT / "runs" / ("retrieval_" + args.method + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"事实问题：{metrics['supported_count']} 条；资料不足问题：{metrics['not_supported_count']} 条。")
    for label, name in (("Recall@1", "recall_at_1"), (f"Recall@{args.top_k}", "recall_at_k"),
                        ("资料不足问题无命中率", "unsupported_no_hit_rate")):
        value = metrics[name]
        print(f"{label}：" + (f"{value:.1%}" if value is not None else "不适用"))
    print("报告：" + str(output))
    print("本次仅评测小规模开发集的检索，不证明问答质量。")


if __name__ == "__main__":
    main()
