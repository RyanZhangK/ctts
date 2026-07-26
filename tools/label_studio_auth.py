#!/usr/bin/env python3
"""Label Studio 认证模块 —— 自动获取 Token + HTTP 客户端基类。

Token 获取优先级：SQLite DB → LABEL_STUDIO_TOKEN 环境变量

示例:
  # 作为模块使用
  from label_studio_auth import get_client
  client = get_client()
  projects = client.get("/api/projects/")

  # 独立运行，验证 token 状态
  python label_studio_auth.py
"""

import os
import sqlite3
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_DB_PATH = os.path.expanduser("~/.local/share/label-studio/label_studio.sqlite3")


# ---------------------------------------------------------------------------
# Token 获取
# ---------------------------------------------------------------------------

def get_token_from_db(db_path: str = DEFAULT_DB_PATH) -> str | None:
    """从 Label Studio SQLite 数据库中读取 API token。"""
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute("SELECT key FROM authtoken_token ORDER BY created DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
    except (sqlite3.Error, sqlite3.OperationalError) as e:
        print(f"⚠ 读取 token 数据库失败: {e}")
    return None


def get_token_from_env() -> str | None:
    """从环境变量 LABEL_STUDIO_TOKEN 读取 token。"""
    return os.environ.get("LABEL_STUDIO_TOKEN")


def get_token(db_path: str = DEFAULT_DB_PATH) -> str:
    """获取 API token，按优先级降级：DB → 环境变量。

    Raises:
        RuntimeError: 所有来源均未找到 token
    """
    token = get_token_from_db(db_path)
    if token:
        return token

    token = get_token_from_env()
    if token:
        return token

    raise RuntimeError(
        "未找到 Label Studio API token。\n"
        f"  - 数据库: {db_path}\n"
        "  - 环境变量: LABEL_STUDIO_TOKEN 未设置\n"
        "请先在 Label Studio Web UI 中登录，或设置 LABEL_STUDIO_TOKEN 环境变量。"
    )


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------

class _LabelStudioClient:
    """Label Studio API 客户端，封装 HTTP 请求与认证。"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token or get_token()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json",
        })

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"

    def _unwrap(self, data: dict | list) -> dict | list:
        """自动解包分页响应 → 数据列表。

        Label Studio 不同端点返回不同分页格式:
        - /api/projects/ → {"count", "results", ...}
        - /api/tasks/    → {"total", "tasks", ...}
        """
        if not isinstance(data, dict):
            return data
        if "results" in data:
            return data["results"]
        if "tasks" in data:
            return data["tasks"]
        return data

    def get(self, path: str, params: dict | None = None) -> dict | list:
        """发送 GET 请求。"""
        r = self.session.get(self._url(path), params=params)
        r.raise_for_status()
        return self._unwrap(r.json())

    def post(self, path: str, data: dict | None = None, files: dict | None = None) -> dict:
        """发送 POST 请求。"""
        if files:
            r = self.session.post(self._url(path), data=data, files=files)
        else:
            r = self.session.post(self._url(path), json=data)
        r.raise_for_status()
        return r.json()

    def patch(self, path: str, data: dict) -> dict:
        """发送 PATCH 请求。"""
        r = self.session.patch(self._url(path), json=data)
        r.raise_for_status()
        return r.json()

    def delete(self, path: str) -> dict:
        """发送 DELETE 请求。"""
        r = self.session.delete(self._url(path))
        r.raise_for_status()
        return r.json() if r.text else {}


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

_client: _LabelStudioClient | None = None


def get_client(base_url: str = DEFAULT_BASE_URL, token: str | None = None) -> _LabelStudioClient:
    """获取 Label Studio API 客户端（单例）。

    首次调用时自动从 DB/环境变量获取 token，后续调用复用。
    若后续调用传入不同的 base_url 或 token，会打印警告并使用已缓存的客户端。
    """
    global _client
    if _client is None:
        _client = _LabelStudioClient(base_url=base_url, token=token)
    else:
        if base_url != DEFAULT_BASE_URL and base_url != _client.base_url:
            import warnings
            warnings.warn(
                f"get_client(base_url='{base_url}') 被忽略，"
                f"已缓存客户端指向 {_client.base_url}"
            )
    return _client


# ---------------------------------------------------------------------------
# 独立运行
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Label Studio 认证 —— 获取并验证 API Token",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=f"Label Studio 地址（默认: {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB_PATH,
        help="SQLite 数据库路径",
    )
    args = parser.parse_args()

    print(f"Label Studio: {args.base_url}")

    # 尝试获取 token
    token_db = get_token_from_db(args.db)
    token_env = get_token_from_env()

    print(f"Token (DB):   {'✅ ' + token_db[:8] + '...' if token_db else '❌ 未找到'}")
    print(f"Token (ENV):  {'✅ ' + token_env[:8] + '...' if token_env else '❌ 未设置'}")

    try:
        token = get_token(args.db)
        print(f"使用 Token:   {token[:8]}...")

        # 验证 token 是否有效
        client = get_client(args.base_url, token)
        user_info = client.get("/api/current-user/whoami")
        print(f"用户:         {user_info.get('email', '?')}  (id={user_info.get('id', '?')})")
        print("\n✅ Token 有效，认证成功！")
    except RuntimeError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Token 验证失败: {e}")


if __name__ == "__main__":
    main()
