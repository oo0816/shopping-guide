"""Check LLM configuration; use --online for one short chat request."""

import argparse
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from check_embedding import NoRedirect, ROOT


class ConfigurationError(ValueError):
    pass


def read_llm_config():
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
    def get(name):
        return os.environ.get(name) or values.get(name, "")
    key = get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    base = get("LLM_BASE_URL").strip().rstrip("/")
    model = get("LLM_MODEL").strip()
    missing = [name for name, value in (
        ("LLM_API_KEY", key), ("LLM_BASE_URL", base), ("LLM_MODEL", model),
    ) if not value.strip()]
    if missing:
        raise ConfigurationError("请在项目 .env 中填写：" + ", ".join(missing))
    url = urlsplit(base)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or "{" in base or "}" in base):
        raise ConfigurationError("LLM_BASE_URL 必须是服务商提供的完整 HTTPS Base URL。")
    return key, base, model


def validate_response(decoded):
    if not isinstance(decoded, dict):
        raise ValueError("Invalid response")
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Missing choices")
    choice = choices[0]
    message = choice.get("message")
    if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
        raise ValueError("Incomplete reply")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty reply")


def main():
    parser = argparse.ArgumentParser(description="默认只检查 LLM 配置，--online 才请求模型。")
    parser.add_argument("--online", action="store_true", help="发送一次短对话请求")
    args = parser.parse_args()
    key, base, model = read_llm_config()
    print("LLM 配置已填写；未显示密钥。")
    if not args.online:
        print("尚未验证 Key 是否有效。加 --online 进行一次实际调用。")
        return
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "请只回复 OK。"}],
        "stream": False,
        "max_tokens": 64,
    }
    # Disable DeepSeek thinking for this small connectivity check.
    if urlsplit(base).hostname == "api.deepseek.com":
        payload["thinking"] = {"type": "disabled"}
    request = Request(base + "/chat/completions", method="POST",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.perf_counter()
    with build_opener(NoRedirect()).open(request, timeout=30) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("Response too large")
    validate_response(json.loads(raw))
    print("LLM 调用成功，已收到完整的非空回答。")
    print(f"耗时：{time.perf_counter() - started:.2f} 秒。")
    print("这次仅验证对话连通性，不代表 RAG、结构化输出或工具调用已验证。")


if __name__ == "__main__":
    try:
        main()
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except HTTPError as exc:
        hint = {
            400: "检查模型名称和参数。",
            401: "检查是否填写了正确服务商的 API Key。",
            402: "检查账号余额。",
            403: "检查访问权限。",
            404: "检查 Base URL 和模型名称。",
            429: "检查限流状态，稍后重试。",
        }.get(exc.code, "检查服务状态。")
        print(f"LLM 接口返回 HTTP {exc.code}。{hint}", file=sys.stderr)
        sys.exit(1)
    except (URLError, TimeoutError, OSError):
        print("网络请求或文件读取失败，请检查网络、接入地址和文件权限。", file=sys.stderr)
        sys.exit(1)
    except (ValueError, TypeError, OverflowError):
        print("LLM 配置无效或响应不完整；未输出服务端原始内容。", file=sys.stderr)
        sys.exit(2)
