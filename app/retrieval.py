"""Offline BM25 baseline: exact Lenovo MTM filtering plus lexical ranking."""

from collections import Counter
import json
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data/processed/chunks.jsonl"
MTM = re.compile(r"(?<![a-z0-9])\d{2}[a-z0-9]{8}(?![a-z0-9])")
TOKENS = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*|[\u4e00-\u9fff]+")
STOPWORDS = {"什么", "多少", "可以", "这个", "这款", "那个", "请问", "一下", "是否"}
# General domain vocabulary, not per-question or per-model answer rules.
ALIASES = (
    (r"(?<![a-z])(?:cpu|processor)(?![a-z])|处理器|芯片", "processor"),
    (r"(?<![a-z])ram(?![a-z])|内存", "memory"),
    (r"(?<![a-z])ssd(?![a-z])|固态硬盘|硬盘|存储", "storage"),
    (r"(?<![a-z])withdrawn(?![a-z])|在售|停售|停产|撤销", "availability"),
)
K1, B = 1.2, 0.75


def normalize(text):
    return unicodedata.normalize("NFKC", text).lower()


def identifiers(text):
    return set(MTM.findall(normalize(text)))


def tokenize(text):
    text = MTM.sub(" ", normalize(text))
    for pattern, canonical in ALIASES:
        text = re.sub(pattern, " " + canonical + " ", text)
    terms = []
    for value in TOKENS.findall(text):
        if "\u4e00" <= value[0] <= "\u9fff":
            terms.extend(value[i:i + 2] for i in range(len(value) - 1)
                         if value[i:i + 2] not in STOPWORDS)
        else:
            terms.append(value)
    return terms


def load_chunks(path=DEFAULT_CORPUS):
    if not Path(path).is_file():
        raise ValueError("未找到资料片段，请先运行 scripts/prepare_corpus.py。")
    rows = [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    required = ("chunk_id", "product_id", "section", "text")
    if not rows or any(
        not isinstance(row, dict) or any(
            not isinstance(row.get(name), str) or not row[name].strip()
            for name in required
        ) for row in rows
    ):
        raise ValueError("资料片段缺少必填字段或内容为空。")
    if any(not isinstance(row.get("mtm"), str) or not (
        row.get("source_url") or row.get("local_path")) for row in rows):
        raise ValueError("资料片段缺少型号字段或本地资料位置。")
    if len({row["chunk_id"] for row in rows}) != len(rows):
        raise ValueError("资料片段 ID 重复。")
    return rows


class KeywordRetriever:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("资料片段为空。")
        self.chunks = chunks
        self.models = {normalize(row["mtm"]) for row in chunks}
        # The first line is the repeated model header, not topical evidence.
        bodies = [row["text"].partition("\n")[2] or row["text"] for row in chunks]
        self.frequencies = [Counter(tokenize(body)) for body in bodies]
        self.lengths = [sum(counts.values()) for counts in self.frequencies]
        self.average_length = sum(self.lengths) / len(chunks) or 1.0
        self.document_frequency = Counter(
            term for counts in self.frequencies for term in counts
        )

    def search(self, query, top_k=3, product_ids=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("问题不能为空。")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k 必须是 1～20 的整数。")
        allowed = set(product_ids) if product_ids is not None else None
        requested = identifiers(query)
        result = {
            "query": query, "method": "keyword_bm25",
            "model_filter": sorted(value.upper() for value in requested),
            "status": "no_lexical_match", "hits": [],
        }
        if requested - self.models:
            result["status"] = "unknown_model"
            return result
        if allowed is not None:
            result["product_filter"] = sorted(allowed)
            if not allowed or allowed - {row["product_id"] for row in self.chunks}:
                result["status"] = "unknown_product"
                return result
        terms = set(tokenize(query))
        hits = []
        size = len(self.chunks)
        for row, counts, length in zip(self.chunks, self.frequencies, self.lengths):
            if allowed is not None and row["product_id"] not in allowed:
                continue
            if requested and normalize(row["mtm"]) not in requested:
                continue
            matched = sorted(terms.intersection(counts))
            if not matched:
                continue
            score = 0.0
            for term in matched:
                tf = counts[term]
                df = self.document_frequency[term]
                idf = math.log(1 + (size - df + 0.5) / (df + 0.5))
                score += idf * tf * (K1 + 1) / (
                    tf + K1 * (1 - B + B * length / self.average_length)
                )
            hits.append({**row, "score": round(score, 6), "matched_terms": matched})
        hits.sort(key=lambda item: (-item["score"], item["chunk_id"]))
        result["hits"] = hits[:top_k]
        if result["hits"]:
            result["status"] = "matched"
        return result
