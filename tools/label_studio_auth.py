#!/usr/bin/env python3
"""Label Studio 认证模块 —— Token 解析 + 官方 SDK 客户端（支持本机 / 远程实例）。

连接配置优先级:
  base_url: CLI/--base-url → LABEL_STUDIO_URL / LABEL_STUDIO_BASE_URL → http://127.0.0.1:8080
  token:    CLI/--token → LABEL_STUDIO_TOKEN → (仅本机) 本地 SQLite

远程实例无法读取本机 Label Studio DB，必须提供 Token（CLI 或环境变量）。

所有 API 调用均通过官方 ``label-studio-sdk`` 完成（需安装 ``label-studio-sdk``）。

示例:
  # 本机(自动从本地 DB 读 token)
  python label_studio_auth.py

  # 远程
  python label_studio_auth.py --base-url http://192.168.1.10:8080 --token <TOKEN>
  # 或
  set LABEL_STUDIO_URL=http://192.168.1.10:8080
  set LABEL_STUDIO_TOKEN=<TOKEN>
  python label_studio_auth.py
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from label_studio_sdk import LabelStudio


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 30.0

# 兼容 Unix / Windows 本机 Label Studio 安装路径
_DB_CANDIDATES = [
    Path.home() / ".local" / "share" / "label-studio" / "label_studio.sqlite3",
    Path(os.environ.get("LOCALAPPDATA", "")) / "label-studio" / "label_studio.sqlite3",
    Path.home() / "AppData" / "Local" / "label-studio" / "label_studio.sqlite3",
]


def default_db_path() -> str:
    """返回第一个存在的本机 DB 路径；都不存在则返回 Unix 默认路径（供报错提示）。"""
    for p in _DB_CANDIDATES:
        if p and str(p) != "." and p.is_file():
            return str(p)
    return str(_DB_CANDIDATES[0])


DEFAULT_DB_PATH = default_db_path()


# ---------------------------------------------------------------------------
# URL / Token 解析
# ---------------------------------------------------------------------------

def resolve_base_url(base_url: str | None = None) -> str:
    """解析 Label Studio 地址。"""
    if base_url:
        return base_url.rstrip("/")
    for key in ("LABEL_STUDIO_URL", "LABEL_STUDIO_BASE_URL"):
        val = os.environ.get(key)
        if val:
            return val.rstrip("/")
    return DEFAULT_BASE_URL


def is_local_url(base_url: str) -> bool:
    """判断是否为本机 Label Studio（才允许从本地 SQLite 读 token）。"""
    host = (urlparse(base_url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1") or host.endswith(".localhost")


def get_token_from_db(db_path: str | None = None) -> str | None:
    """从本机 Label Studio SQLite 读取 API token。"""
    paths: list[str] = []
    if db_path:
        paths.append(db_path)
    else:
        paths.extend(str(p) for p in _DB_CANDIDATES if p and str(p) != ".")

    seen: set[str] = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    "SELECT key FROM authtoken_token ORDER BY created DESC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            if row:
                return row[0]
        except (sqlite3.Error, sqlite3.OperationalError) as e:
            print(f"⚠ 读取 token 数据库失败 ({path}): {e}")
    return None


def get_token_from_env() -> str | None:
    """从环境变量 LABEL_STUDIO_TOKEN 读取 token。"""
    return os.environ.get("LABEL_STUDIO_TOKEN") or None


def get_token(
    db_path: str | None = None,
    token: str | None = None,
    base_url: str | None = None,
    allow_db: bool | None = None,
) -> str:
    """获取 API token。

    优先级:
      1. 显式传入 token
      2. 环境变量 LABEL_STUDIO_TOKEN
      3. 本机 SQLite（仅当目标为本地实例，或 allow_db=True）

    Raises:
        RuntimeError: 所有来源均未找到 token
    """
    if token:
        return token

    env_token = get_token_from_env()
    if env_token:
        return env_token

    url = resolve_base_url(base_url)
    if allow_db is None:
        allow_db = is_local_url(url)

    if allow_db:
        db_token = get_token_from_db(db_path)
        if db_token:
            return db_token

    db_hint = db_path or DEFAULT_DB_PATH
    if is_local_url(url):
        raise RuntimeError(
            "未找到 Label Studio API token。\n"
            f"  - 目标: {url}（本机）\n"
            f"  - 数据库: {db_hint}\n"
            "  - 环境变量: LABEL_STUDIO_TOKEN 未设置\n"
            "请先在 Label Studio Web UI 登录，或设置 LABEL_STUDIO_TOKEN。"
        )
    raise RuntimeError(
        "未找到 Label Studio API token。\n"
        f"  - 目标: {url}（远程，不会读取本机 DB）\n"
        "  - 请通过 --token 或环境变量 LABEL_STUDIO_TOKEN 提供 Token。\n"
        "  - Token 可在远程 Label Studio: Account & Settings → Access Token 复制。"
    )


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    """为 CLI 添加通用连接参数。"""
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Label Studio 地址（默认: 环境变量 LABEL_STUDIO_URL / "
            f"LABEL_STUDIO_BASE_URL，否则 {DEFAULT_BASE_URL}）"
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help="API Token（默认: 环境变量 LABEL_STUDIO_TOKEN；本机还可从 SQLite 读取）",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"本机 SQLite 路径（仅本地实例；默认自动探测，如 {DEFAULT_DB_PATH}）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP 超时秒数（默认: {DEFAULT_TIMEOUT}）",
    )


def client_from_args(args) -> LabelStudioClient:
    """从 argparse Namespace 构建客户端。"""
    return get_client(
        base_url=getattr(args, "base_url", None),
        token=getattr(args, "token", None),
        db_path=getattr(args, "db", None),
        timeout=getattr(args, "timeout", DEFAULT_TIMEOUT),
    )


# ---------------------------------------------------------------------------
# SDK 客户端
# ---------------------------------------------------------------------------

def to_dict(obj) -> Any:
    """把 SDK 返回的模型对象转换为普通 dict / list。"""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    legacy = getattr(obj, "dict", None)
    if callable(legacy):
        return legacy()
    return obj


class LabelStudioClient:
    """官方 label-studio-sdk 的轻量封装。

    ``projects`` / ``tasks`` / ``users`` 等命名空间通过 ``__getattr__``
    直接透传给 SDK，因此可以照常使用::

        client.projects.list()
        client.tasks.list(project=1)
        client.users.whoami()

    仅对 SDK 尚未封装的接口（如项目内文件上传）补充便捷方法。
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = resolve_base_url(base_url)
        self.token = token or get_token(base_url=self.base_url)
        self.timeout = timeout
        self.sdk = LabelStudio(
            base_url=self.base_url,
            api_key=self.token,
            timeout=timeout,
        )

    def __getattr__(self, name: str):
        # 未定义的属性/方法交给 SDK（projects、tasks、users、import_storage ...）
        if name == "sdk":
            raise AttributeError(name)
        return getattr(self.sdk, name)

    def _headers(self) -> dict[str, str]:
        """复用 SDK 的认证头（自动处理旧版 Token / JWT）。"""
        return self.sdk._client_wrapper.get_headers()

    def whoami(self) -> dict:
        """当前登录用户信息。"""
        return to_dict(self.sdk.users.whoami())

    def upload_file(self, project_id: int, file_path: str) -> dict:
        """上传单个文件并创建为项目任务。

        官方 SDK 2.x 的 ``projects.import_tasks`` 只接受 JSON 任务数据，
        未封装 multipart 文件上传，因此这里复用 SDK 的认证头直接调用端点。
        """
        mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        url = f"{self.base_url}/api/projects/{project_id}/import"
        with open(file_path, "rb") as fh:
            resp = httpx.post(
                url,
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh, mime)},
                timeout=self.timeout,
            )
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {}


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

_clients: dict[tuple[str, str, float], LabelStudioClient] = {}


def get_client(
    base_url: str | None = None,
    token: str | None = None,
    db_path: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    force_new: bool = False,
) -> LabelStudioClient:
    """获取 Label Studio SDK 客户端。

    按 (base_url, token, timeout) 缓存；参数变化时返回对应客户端，不再错误复用本机单例。
    """
    url = resolve_base_url(base_url)
    tok = get_token(db_path=db_path, token=token, base_url=url)
    key = (url, tok, float(timeout))
    if force_new or key not in _clients:
        _clients[key] = LabelStudioClient(base_url=url, token=tok, timeout=timeout)
    return _clients[key]


def reset_clients() -> None:
    """清空客户端缓存（测试或切换账号时用）。"""
    _clients.clear()


# ---------------------------------------------------------------------------
# 独立运行
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label Studio 认证 —— 验证本机或远程实例的 API Token（SDK）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "环境变量:\n"
            "  LABEL_STUDIO_URL / LABEL_STUDIO_BASE_URL  实例地址\n"
            "  LABEL_STUDIO_TOKEN                       API Token（远程必需）\n"
        ),
    )
    add_connection_args(parser)
    args = parser.parse_args()

    url = resolve_base_url(args.base_url)
    local = is_local_url(url)
    print(f"Label Studio: {url}  ({'本机' if local else '远程'})")

    token_env = get_token_from_env()
    print(f"Token (ENV):  {'✅ ' + token_env[:8] + '...' if token_env else '❌ 未设置'}")
    if local:
        token_db = get_token_from_db(args.db)
        print(f"Token (DB):   {'✅ ' + token_db[:8] + '...' if token_db else '❌ 未找到'}")
    else:
        print("Token (DB):   ⏭ 远程实例跳过本机数据库")

    try:
        token = get_token(db_path=args.db, token=args.token, base_url=url)
        print(f"使用 Token:   {token[:8]}...")

        client = get_client(
            base_url=url, token=token, db_path=args.db, timeout=args.timeout, force_new=True,
        )
        user_info = client.whoami()
        print(f"用户:         {user_info.get('email', '?')}  (id={user_info.get('id', '?')})")
        print("\n✅ Token 有效，认证成功！")
    except RuntimeError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Token 验证失败: {e}")


if __name__ == "__main__":
    main()
