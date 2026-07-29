"""第11章完整示例：安全调用 Dify Chatflow 与 Workflow API。

运行方式：
    python examples/11_dify_client.py chat "请解释 Workflow"
    python examples/11_dify_client.py workflow '{"question":"解释 Workflow"}'

应用 API Key 只能放在可信后端。不要把本文件改造成直接暴露 Key 的前端代码。
"""

from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from dotenv import load_dotenv


MAX_QUERY_CHARS = 4000
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0


class DifyClientError(RuntimeError):
    """对外暴露的脱敏 Dify 客户端错误。"""


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise DifyClientError(f"缺少环境变量 {name}")
    return value


def validate_base_url(base_url: str) -> str:
    """只接受完整 URL，远程主机必须使用 HTTPS。"""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("DIFY_BASE_URL 必须是完整的 http(s) 地址")
    if parsed.username or parsed.password:
        raise ValueError("DIFY_BASE_URL 不能包含用户名或密码")
    if parsed.query or parsed.fragment:
        raise ValueError("DIFY_BASE_URL 不能包含 query 或 fragment")
    if parsed.scheme != "https" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("远程 Dify 服务必须使用 HTTPS")

    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def validate_user_id(user: str) -> str:
    """user 用于终端用户隔离，不能为空或无限长。"""
    user = user.strip()
    if not user or len(user) > 128:
        raise ValueError("user 长度必须在 1 到 128 之间")
    return user


@dataclass(frozen=True)
class DifyResponse:
    """保留业务结果和排障所需的非敏感元数据。"""

    text: str
    request_id: str
    task_id: str | None
    raw: dict[str, Any]


class DifyClient:
    """一个只支持 blocking 模式的最小后端客户端。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = validate_base_url(base_url)
        if not api_key.strip():
            raise ValueError("api_key 不能为空")
        if timeout <= 0 or timeout > 120:
            raise ValueError("timeout 必须在 0 到 120 秒之间")
        self._api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "DifyClient":
        """从环境变量创建客户端。"""
        return cls(
            base_url=require_env("DIFY_BASE_URL"),
            api_key=require_env("DIFY_API_KEY"),
            timeout=float(os.getenv("DIFY_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)),
        )

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> DifyResponse:
        """发送 JSON 请求，并统一处理大小、编码和 HTTP 错误。"""
        request_id = uuid4().hex
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}{endpoint}",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Request-ID": request_id,
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            # 不回显 Authorization，也不把服务端完整错误直接交给终端用户。
            detail = exc.read(4096).decode("utf-8", errors="replace")
            safe_detail = self._safe_error_detail(detail)
            if exc.code in {401, 403}:
                raise DifyClientError(
                    f"Dify 鉴权失败（HTTP {exc.code}，request_id={request_id}）"
                ) from exc
            if exc.code == 404:
                raise DifyClientError(
                    f"Dify 端点不存在（HTTP 404，检查应用类型和 BASE_URL，"
                    f"request_id={request_id}）"
                ) from exc
            if exc.code == 429:
                raise DifyClientError(
                    f"Dify 请求过于频繁（HTTP 429，request_id={request_id}）"
                ) from exc
            raise DifyClientError(
                f"Dify 请求失败（HTTP {exc.code}，{safe_detail}，"
                f"request_id={request_id}）"
            ) from exc
        except (URLError, socket.timeout, TimeoutError) as exc:
            raise DifyClientError(
                f"Dify 网络请求失败或超时（request_id={request_id}）"
            ) from exc

        if len(raw) > MAX_RESPONSE_BYTES:
            raise DifyClientError(
                f"响应超过 2 MiB，已拒绝继续读取（request_id={request_id}）"
            )

        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DifyClientError(
                f"Dify 返回的不是合法 UTF-8 JSON（request_id={request_id}）"
            ) from exc
        if not isinstance(result, dict):
            raise DifyClientError(
                f"Dify 返回 JSON 顶层不是对象（request_id={request_id}）"
            )

        return DifyResponse(
            text="",
            request_id=request_id,
            task_id=result.get("task_id"),
            raw=result,
        )

    @staticmethod
    def _safe_error_detail(detail: str) -> str:
        """只抽取简短错误码，避免把服务端敏感响应写入日志。"""
        try:
            parsed = json.loads(detail)
            code = parsed.get("code") if isinstance(parsed, dict) else None
            return f"code={str(code)[:100]}" if code else "无公开错误码"
        except json.JSONDecodeError:
            return "无公开错误码"

    def chat(
        self,
        query: str,
        user: str,
        inputs: dict[str, Any] | None = None,
        conversation_id: str = "",
    ) -> DifyResponse:
        """调用 Chatflow/聊天应用的 /v1/chat-messages。"""
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query 不能超过 {MAX_QUERY_CHARS} 字符")

        response = self._post(
            "/v1/chat-messages",
            {
                "inputs": inputs or {},
                "query": query,
                "response_mode": "blocking",
                "conversation_id": conversation_id,
                "user": validate_user_id(user),
            },
        )
        answer = response.raw.get("answer")
        if not isinstance(answer, str):
            raise DifyClientError(
                f"Dify 未返回字符串 answer（request_id={response.request_id}）"
            )
        return DifyResponse(
            text=answer,
            request_id=response.request_id,
            task_id=response.task_id,
            raw=response.raw,
        )

    def run_workflow(
        self,
        inputs: dict[str, Any],
        user: str,
    ) -> DifyResponse:
        """调用 Workflow 应用的 /v1/workflows/run。"""
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError("workflow inputs 必须是非空 JSON 对象")
        response = self._post(
            "/v1/workflows/run",
            {
                "inputs": inputs,
                "response_mode": "blocking",
                "user": validate_user_id(user),
            },
        )

        data = response.raw.get("data")
        if not isinstance(data, dict):
            raise DifyClientError(
                f"Workflow 未返回 data 对象（request_id={response.request_id}）"
            )
        status = data.get("status")
        if status not in {"succeeded", "success"}:
            raise DifyClientError(
                f"Workflow 未成功：status={status!r}"
                f"（request_id={response.request_id}）"
            )
        outputs = data.get("outputs", {})
        text = json.dumps(outputs, ensure_ascii=False, indent=2)
        return DifyResponse(
            text=text,
            request_id=response.request_id,
            task_id=response.task_id,
            raw=response.raw,
        )


def parse_workflow_inputs(raw: str) -> dict[str, Any]:
    """从命令行解析 Workflow 输入。"""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Workflow 输入必须是合法 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Workflow 输入 JSON 顶层必须是对象")
    return value


def main() -> None:
    load_dotenv()
    client = DifyClient.from_env()
    user = os.getenv("DIFY_USER_ID", "tutorial-local-user")
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if mode == "chat":
        query = (
            " ".join(sys.argv[2:])
            if len(sys.argv) > 2
            else "请用三句话解释 Workflow 和 Agent 的区别。"
        )
        response = client.chat(query=query, user=user)
    elif mode == "workflow":
        raw_inputs = (
            sys.argv[2]
            if len(sys.argv) > 2
            else '{"question": "请解释 Dify Workflow"}'
        )
        response = client.run_workflow(
            inputs=parse_workflow_inputs(raw_inputs),
            user=user,
        )
    else:
        raise SystemExit(
            "用法：python examples/11_dify_client.py "
            "[chat 问题 | workflow '{\"question\":\"...\"}']"
        )

    print(response.text)
    print(f"\nrequest_id={response.request_id}")
    if response.task_id:
        print(f"task_id={response.task_id}")


if __name__ == "__main__":
    main()
