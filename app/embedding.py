"""Embedding API client with a local SQLite cache; no SDK dependency."""

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.retrieval import ROOT


class EmbeddingError(ValueError):
    """Safe-to-display configuration, cache or provider error."""


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str
    model: str
    dimensions: int
    api_key: str = field(default="", repr=False, compare=False)
    revision: str = "1"

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or "{" in self.base_url or "}" in self.base_url):
            raise EmbeddingError("Embedding 地址必须是完整的 HTTPS Base URL。")
        if not self.model.strip() or type(self.dimensions) is not int or self.dimensions <= 0:
            raise EmbeddingError("Embedding 模型或维度配置无效。")

    @property
    def profile(self):
        return {"base_url": self.base_url.rstrip("/"), "model": self.model,
                "dimensions": self.dimensions, "revision": self.revision, "format_version": 1}


def load_config():
    values = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[name.strip()] = value
    def get(name, default=""):
        return os.environ.get(name) or values.get(name) or default
    try:
        dimensions = int(get("EMBEDDING_DIMENSIONS"))
    except ValueError:
        raise EmbeddingError("请填写 EMBEDDING_DIMENSIONS 正整数。") from None
    return EmbeddingConfig(
        base_url=get("EMBEDDING_BASE_URL").strip().rstrip("/"),
        model=get("EMBEDDING_MODEL").strip(), dimensions=dimensions,
        api_key=get("EMBEDDING_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", ""),
        revision=get("EMBEDDING_REVISION", "1"),
    )


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_vector(vector, dimensions):
    if (not isinstance(vector, list) or len(vector) != dimensions
            or not all(type(x) in (int, float) and math.isfinite(x) for x in vector)):
        raise EmbeddingError("向量维度或数值无效，请检查配置和缓存。")
    norm = math.hypot(*vector)
    if norm <= 0 or not math.isfinite(norm):
        raise EmbeddingError("向量范数无效。")
    return vector


def parse_vectors(data, count, dimensions):
    if not isinstance(data, dict) or not isinstance(data.get("data"), list) or len(data["data"]) != count:
        raise EmbeddingError("Embedding 响应数量或格式不正确。")
    vectors = {}
    for row in data["data"]:
        if not isinstance(row, dict):
            raise EmbeddingError("Embedding 响应条目无效。")
        index = row.get("index")
        if type(index) is not int or not 0 <= index < count or index in vectors:
            raise EmbeddingError("Embedding 响应索引重复或无效。")
        vectors[index] = validate_vector(row.get("embedding"), dimensions)
    return [vectors[i] for i in range(count)]


class EmbeddingClient:
    def __init__(self, config, cache_path=None):
        self.config = config
        self.cache_path = Path(cache_path or ROOT / "data/processed/embedding_cache.sqlite")
        self.stats = {"cache_hits": 0, "embedded_texts": 0, "requests": 0}

    def _key(self, text):
        payload = json.dumps([self.config.profile, text], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _connect(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.cache_path, timeout=10)
        connection.execute("CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, vector_json TEXT NOT NULL)")
        return connection

    def _fetch(self, texts):
        if not self.config.api_key.strip():
            raise EmbeddingError("请在本机 .env 填写 EMBEDDING_API_KEY。")
        body = json.dumps({"model": self.config.model, "input": texts,
                           "dimensions": self.config.dimensions, "encoding_format": "float"},
                          ensure_ascii=False).encode("utf-8")
        request = Request(self.config.base_url.rstrip("/") + "/embeddings",
                          data=body, method="POST", headers={
                              "Content-Type": "application/json",
                              "Authorization": "Bearer " + self.config.api_key})
        self.stats["requests"] += 1
        try:
            with build_opener(NoRedirect()).open(request, timeout=30) as response:
                raw = response.read(4_000_001)
            if len(raw) > 4_000_000:
                raise EmbeddingError("Embedding 响应过大。")
            result = parse_vectors(json.loads(raw), len(texts), self.config.dimensions)
        except HTTPError as exc:
            raise EmbeddingError(f"Embedding HTTP {exc.code}；请检查 Key、地域、模型、余额或限流状态。") from None
        except (URLError, TimeoutError, OSError):
            raise EmbeddingError("Embedding 网络请求失败，请检查网络与接入地址。") from None
        except (json.JSONDecodeError, UnicodeError):
            raise EmbeddingError("Embedding 响应不是有效 JSON。") from None
        self.stats["embedded_texts"] += len(texts)
        return result

    def embed(self, texts, online=False, refresh=False):
        if not isinstance(texts, list) or not texts or any(
                not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("向量输入必须是非空文本列表。")
        keys = [self._key(text) for text in texts]
        vectors, pending = {}, {}
        connection = self._connect()
        try:
            for key, text in zip(keys, texts):
                if key in vectors or key in pending:
                    continue
                row = None if refresh else connection.execute(
                    "SELECT vector_json FROM embeddings WHERE cache_key = ?", (key,)
                ).fetchone()
                if row:
                    try:
                        vectors[key] = validate_vector(json.loads(row[0]), self.config.dimensions)
                    except (ValueError, TypeError, OverflowError):
                        raise EmbeddingError("向量缓存损坏，请备份后重建对应缓存。") from None
                else:
                    pending[key] = text
        finally:
            connection.close()
        self.stats["cache_hits"] += sum(key in vectors for key in keys)
        if pending and not online:
            raise EmbeddingError("缺少向量缓存；添加 --online 才会请求 API 并产生用量。")
        todo = list(pending.items())
        for start in range(0, len(todo), 10):
            batch = todo[start:start + 10]
            embedded = self._fetch([text for _, text in batch])
            connection = self._connect()
            try:
                with connection:
                    for (key, _), vector in zip(batch, embedded):
                        connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                                           (key, json.dumps(vector)))
                        vectors[key] = vector
            finally:
                connection.close()
        return [vectors[key] for key in keys]
