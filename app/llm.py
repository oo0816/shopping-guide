"""Bounded JSON chat client for grounded answers; no SDK dependency."""

from dataclasses import dataclass, field
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from app.embedding import NoRedirect
from app.retrieval import ROOT


class LLMError(ValueError):
    """Safe-to-display model configuration or transport error."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str = field(default="", repr=False, compare=False)
    revision: str = "1"
    max_tokens: int = 1200

    def __post_init__(self):
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 8192:
            raise LLMError("LLM 输出长度配置无效。")
        url = urlsplit(self.base_url)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or "{" in self.base_url or "}" in self.base_url
                or not self.model.strip()):
            raise LLMError("LLM 地址或模型配置无效。")

    @property
    def profile(self):
        return {"base_url": self.base_url.rstrip("/"), "model": self.model,
                "revision": self.revision, "max_tokens": self.max_tokens, "temperature": 0}


def load_llm_config():
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
    return LLMConfig(
        base_url=get("LLM_BASE_URL").strip().rstrip("/"),
        model=get("LLM_MODEL").strip(),
        api_key=get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", ""),
        revision=get("LLM_REVISION", "1"),
    )


def response_json(decoded):
    if not isinstance(decoded, dict):
        raise LLMError("LLM 响应格式无效。")
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMError("LLM 未返回回答。")
    choice = choices[0]
    message = choice.get("message")
    if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
        raise LLMError("LLM 回答未正常完成，未展示截断内容。")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMError("LLM 返回了空回答。")
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        raise LLMError("LLM 未返回有效 JSON。") from None
    if not isinstance(data, dict):
        raise LLMError("LLM JSON 必须是对象。")
    return data


class LLMClient:
    def __init__(self, config):
        self.config = config
        self.stats = {"requests": 0}

    def chat_json(self, messages, online=False):
        if not online:
            raise LLMError("没有回答缓存；添加 --online 才允许调用 LLM。")
        if not self.config.api_key.strip():
            raise LLMError("请在本机 .env 填写 LLM_API_KEY。")
        body = {
            "model": self.config.model, "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0, "max_tokens": self.config.max_tokens, "stream": False,
        }
        if urlsplit(self.config.base_url).hostname == "api.deepseek.com":
            body["thinking"] = {"type": "disabled"}
        request = Request(self.config.base_url.rstrip("/") + "/chat/completions",
                          data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                          headers={"Authorization": "Bearer " + self.config.api_key,
                                   "Content-Type": "application/json"})
        self.stats["requests"] += 1
        try:
            with build_opener(NoRedirect()).open(request, timeout=45) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise LLMError("LLM 响应过大。")
            decoded = json.loads(raw)
            result = response_json(decoded)
        except HTTPError as exc:
            raise LLMError(f"LLM HTTP {exc.code}；请检查 Key、模型、余额或服务状态。") from None
        except (URLError, TimeoutError, OSError):
            raise LLMError("LLM 网络请求失败。") from None
        except (json.JSONDecodeError, UnicodeError):
            raise LLMError("LLM 接口返回了无效 JSON。") from None
        usage = decoded.get("usage", {})
        if isinstance(usage, dict):
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if type(usage.get(name)) is int and usage[name] >= 0:
                    self.stats[name] = usage[name]
        return result
