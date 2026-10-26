"""Exact cosine retrieval for the tiny local corpus, with strict index identity."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from app.embedding import EmbeddingError, validate_vector
from app.retrieval import MTM, ROOT, identifiers, normalize

DEFAULT_INDEX = ROOT / "data/processed/vector_index.json"


def corpus_digest(chunks):
    payload = json.dumps(chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unit(vector):
    norm = math.hypot(*vector)
    return [number / norm for number in vector]


def build_index(chunks, client, path=DEFAULT_INDEX, online=False, refresh=False):
    if not chunks or len({row["chunk_id"] for row in chunks}) != len(chunks):
        raise EmbeddingError("片段为空或 ID 重复。")
    vectors = client.embed([row["text"] for row in chunks], online=online, refresh=refresh)
    payload = {
        "profile": client.config.profile, "corpus_sha256": corpus_digest(chunks),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vectors": {row["chunk_id"]: vector for row, vector in zip(chunks, vectors)},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return payload


class VectorRetriever:
    def __init__(self, chunks, client, index_path=DEFAULT_INDEX, online=False):
        self.chunks, self.client, self.online = chunks, client, online
        path = Path(index_path)
        if not path.is_file():
            raise EmbeddingError("请先运行 scripts/build_vector_index.py --online 建立向量索引。")
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(index, dict):
                raise EmbeddingError("向量索引格式无效。")
            if index.get("profile") != client.config.profile:
                raise EmbeddingError("Embedding 模型、地址、维度或版本改变，请重建索引。")
            if index.get("corpus_sha256") != corpus_digest(chunks):
                raise EmbeddingError("资料片段已改变，请重新运行索引构建脚本。")
            stored = index.get("vectors")
            if not isinstance(stored, dict) or set(stored) != {row["chunk_id"] for row in chunks}:
                raise EmbeddingError("向量索引与片段 ID 不一致。")
            self.vectors = {
                key: unit(validate_vector(value, client.config.dimensions))
                for key, value in stored.items()
            }
        except (json.JSONDecodeError, UnicodeError, TypeError):
            raise EmbeddingError("向量索引损坏，请重建索引。") from None
        self.models = {normalize(row["mtm"]) for row in chunks}

    def search(self, query, top_k=3, product_ids=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("问题不能为空。")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k 必须是 1～20 的整数。")
        allowed = set(product_ids) if product_ids is not None else None
        requested = identifiers(query)
        result = {
            "query": query, "method": "vector_cosine",
            "model_filter": sorted(value.upper() for value in requested),
            "status": "no_candidates", "hits": [],
        }
        if requested - self.models:
            result["status"] = "unknown_model"
            return result
        if allowed is not None:
            result["product_filter"] = sorted(allowed)
            if not allowed or allowed - {row["product_id"] for row in self.chunks}:
                result["status"] = "unknown_product"
                return result
        semantic_query = " ".join(MTM.sub(" ", normalize(query)).split())
        if not semantic_query:
            result["status"] = "needs_topic"
            return result
        vector = unit(self.client.embed([semantic_query], online=self.online)[0])
        hits = []
        for chunk in self.chunks:
            if allowed is not None and chunk["product_id"] not in allowed:
                continue
            if requested and normalize(chunk["mtm"]) not in requested:
                continue
            similarity = math.fsum(x * y for x, y in zip(vector, self.vectors[chunk["chunk_id"]]))
            hits.append({**chunk, "score": round(max(-1.0, min(1.0, similarity)), 6)})
        hits.sort(key=lambda item: (-item["score"], item["chunk_id"]))
        result["hits"] = hits[:top_k]
        if result["hits"]:
            result["status"] = "candidates"
        result["embedding_stats"] = dict(self.client.stats)
        return result
