"""训练高级参数（JSON / KEY=VALUE）解析与校验。

GUI 的「高级设置」和 CLI 的 `--json` / `--set` 共用这一层，
保证所有参数覆盖都经过同样的校验，并统一保护程序托管的结构性参数。
"""

from __future__ import annotations

import json
from typing import Any, Iterable

# 这些参数由调用方（项目、数据集、输出目录）托管，不允许通过高级参数覆盖，
# 否则会绕过 GUI/CLI 的路径与产物管理。
MANAGED_KEYS = frozenset({"data", "project", "name", "exist_ok"})


class OverrideError(ValueError):
    """高级参数字符串或取值非法。"""


def _check_value(key: str, value: Any) -> None:
    if isinstance(value, dict):
        raise OverrideError(f"参数 {key!r} 不支持对象值，请用标量或列表")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                raise OverrideError(f"参数 {key!r} 的列表只能包含标量")


def normalize(overrides: Any) -> tuple[dict[str, Any], list[str]]:
    """校验并返回 (可覆盖参数, 被忽略的托管参数名)。

    Args:
        overrides: 必须是 dict；key 为非空字符串，value 为 JSON 标量或标量列表。

    Raises:
        OverrideError: 类型或取值非法。
    """
    if overrides is None:
        return {}, []
    if not isinstance(overrides, dict):
        raise OverrideError("高级参数必须是 JSON 对象（{...}）")

    clean: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in overrides.items():
        if not isinstance(key, str) or not key.strip():
            raise OverrideError("参数名必须是非空字符串")
        key = key.strip()
        if key in MANAGED_KEYS:
            ignored.append(key)
            continue
        _check_value(key, value)
        clean[key] = value
    return clean, ignored


def parse_json_text(text: str) -> tuple[dict[str, Any], list[str]]:
    """解析 GUI 文本框中的 JSON 对象。空文本视为无覆盖。"""
    stripped = (text or "").strip()
    if not stripped:
        return {}, []
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise OverrideError(
            f"JSON 解析失败：{exc.msg}（第 {exc.lineno} 行第 {exc.colno} 列）"
        ) from exc
    return normalize(obj)


def parse_set(items: Iterable[str] | None) -> tuple[dict[str, Any], list[str]]:
    """解析 CLI 的 `--set key=value`，value 先按 JSON 解析，失败则当字符串。"""
    overrides: dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise OverrideError(f"--set 需要 KEY=VALUE：{raw!r}")
        key, value_text = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise OverrideError(f"--set 缺少参数名：{raw!r}")
        try:
            value = json.loads(value_text)
        except json.JSONDecodeError:
            value = value_text
        overrides[key] = value
    return normalize(overrides)


def merged(base: dict[str, Any], overrides: dict[str, Any],
           ignored: list[str] | None = None) -> dict[str, Any]:
    """base 为默认参数，overrides 覆盖同名项。"""
    args = dict(base)
    args.update(overrides)
    return args
