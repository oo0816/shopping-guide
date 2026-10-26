"""Check local configuration, or make one small embedding request with --online."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
TEXTS = [
    "这款笔记本轻便，适合每天背去上课。",
    "这款笔记本侧重游戏性能，机身较重。",
    "我需要一台方便携带去教室的电脑。",
]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_config():
    values = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[name.strip()] = value
    def get(name):
        return os.environ.get(name) or values.get(name, "")
    key = get("EMBEDDING_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    base = get("EMBEDDING_BASE_URL").rstrip("/")
    model = get("EMBEDDING_MODEL")
    missing = [name for name, value in (
        ("EMBEDDING_API_KEY", key), ("EMBEDDING_BASE_URL", base),
        ("EMBEDDING_MODEL", model), ("EMBEDDING_DIMENSIONS", get("EMBEDDING_DIMENSIONS")),
    ) if not value.strip()]
    if missing:
        raise ValueError("请在项目 .env 中填写：" + ", ".join(missing))
    url = urlsplit(base)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or "{" in base or "}" in base):
        raise ValueError("EMBEDDING_BASE_URL 必须是控制台对应地域的完整 HTTPS Base URL。")
    try:
        dimensions = int(get("EMBEDDING_DIMENSIONS"))
    except ValueError:
        raise ValueError("EMBEDDING_DIMENSIONS 必须是正整数。") from None
    if dimensions <= 0:
        raise ValueError("EMBEDDING_DIMENSIONS 必须是正整数。")
    return key, base, model, dimensions


def vectors_from_response(response, dimensions):
    rows = response.get("data")
    if not isinstance(rows, list) or len(rows) != len(TEXTS):
        raise ValueError("接口未返回预期数量的向量。")
    vectors = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("向量响应格式异常。")
        index, vector = row.get("index"), row.get("embedding")
        if type(index) is not int or index not in range(len(TEXTS)) or index in vectors:
            raise ValueError("向量响应的 index 缺失、重复或无效。")
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise ValueError("实际向量维度与配置不一致。")
        if not all(type(x) in (int, float) and math.isfinite(x) for x in vector):
            raise ValueError("向量包含无效数值。")
        if sum(x * x for x in vector) <= 0:
            raise ValueError("接口返回了全零向量。")
        vectors[index] = vector
    return [vectors[i] for i in range(len(TEXTS))]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(
        sum(x * x for x in a) * sum(y * y for y in b)
    )


def main():
    parser = argparse.ArgumentParser(description="默认只检查配置，--online 才请求模型。")
    parser.add_argument("--online", action="store_true", help="发送一次三条短文本的真实请求")
    args = parser.parse_args()
    key, base, model, dimensions = read_config()
    print("本机配置已填写；未显示密钥。")
    if not args.online:
        print("尚未验证 Key 是否有效。加 --online 进行一次实际调用。")
        return
    payload = json.dumps({
        "model": model, "input": TEXTS, "dimensions": dimensions,
        "encoding_format": "float",
    }, ensure_ascii=False).encode("utf-8")
    request = Request(base + "/embeddings", data=payload, method="POST", headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    })
    started = time.perf_counter()
    with build_opener(NoRedirect()).open(request, timeout=30) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("接口响应过大，已停止处理。")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("接口响应格式异常。")
    vectors = vectors_from_response(decoded, dimensions)
    print(f"Embedding 调用成功：{len(vectors)} 条文本，{dimensions} 维。")
    print(f"耗时：{time.perf_counter() - started:.2f} 秒。")
    print(f"查询与便携描述的相似度：{cosine(vectors[2], vectors[0]):.4f}")
    print(f"查询与游戏描述的相似度：{cosine(vectors[2], vectors[1]):.4f}")
    print("这次仅验证连通性和响应格式；三个例子不能证明实际检索质量。")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as exc:
        hint = {
            400: "检查模型名称、维度和请求参数。",
            401: "检查 Key 是否正确以及地域是否对应。",
            403: "检查模型调用权限和服务开通情况。",
            404: "检查 Base URL 和模型是否可用。",
            429: "检查额度、余额或限流状态。",
        }.get(exc.code, "检查服务状态和对应地域的接入地址。")
        print(f"接口返回 HTTP {exc.code}。{hint}", file=sys.stderr)
        sys.exit(1)
    except (URLError, TimeoutError, OSError):
        print("网络请求或本地文件读取失败，请检查网络、接入地址和文件权限。", file=sys.stderr)
        sys.exit(1)
    except (ValueError, TypeError, OverflowError):
        print("配置或响应校验失败，请检查必填项、接入地址、模型和维度。", file=sys.stderr)
        # Do not echo arbitrary file values or provider response content.
        sys.exit(2)
