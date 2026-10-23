"""预标注复核：按批次查看 predictions，并批量转成 annotations。

设计约定：
- 只处理 model_version 以 `ctts-prelabel` 开头的预测（即本工具写入的）。
- 同一 task 有多版预测时取 id 最大（最新）的一条。
- 默认跳过已标注任务；认可时按任务重新校验，避免并发重复标注。
- 支持只认可任务中的部分预测框（indices）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable
from urllib.parse import urlparse

import httpx

from label_studio_auth import get_client

TAG_PREFIX = "ctts-prelabel"


def _emit(on_event, stage: str, status: str, message: str,
          progress: float | None = None, visible: bool = True) -> None:
    if on_event is None:
        return
    from prelabel import StageEvent

    on_event(StageEvent(stage=stage, status=status, message=message,
                        progress=progress, visible=visible))


@dataclass
class ReviewTask:
    task_id: int
    prediction_id: int
    version: str
    result: list[dict]
    is_labeled: bool = False
    annotations: int = 0
    image: str = ""

    @property
    def boxes(self) -> int:
        return len(self.result)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prediction_id": self.prediction_id,
            "version": self.version,
            "boxes": self.boxes,
            "is_labeled": self.is_labeled,
            "annotations": self.annotations,
            "image": self.image,
        }

    def selected_result(self, indices: Iterable[int] | None = None) -> list[dict]:
        """取出选中框（默认全部），并去掉预测专用字段 score。"""
        if indices is None:
            picked = list(self.result)
        else:
            picked = [self.result[i] for i in sorted(set(indices))
                      if 0 <= i < len(self.result)]
        cleaned: list[dict] = []
        for item in picked:
            row = dict(item)
            row.pop("score", None)
            cleaned.append(row)
        return cleaned


def _latest_predictions(predictions: Iterable, version: str = "") -> dict[int, object]:
    """按 task 取最新一条 ctts-prelabel 预测。"""
    by_task: dict[int, object] = {}
    for pred in predictions:
        model_version = str(getattr(pred, "model_version", "") or "")
        if not model_version.startswith(TAG_PREFIX):
            continue
        if version and model_version != version:
            continue
        task_id = int(getattr(pred, "task", 0) or 0)
        if not task_id:
            continue
        prev = by_task.get(task_id)
        if prev is None or int(getattr(pred, "id", 0) or 0) > int(getattr(prev, "id", 0) or 0):
            by_task[task_id] = pred
    return by_task


def scan_review(project_id: int, *, version: str = "",
                only_unlabeled: bool = True, limit: int = 0,
                client=None, on_event=None, log: Callable[[str], None] | None = None,
                should_stop: Callable[[], bool] | None = None) -> list[ReviewTask]:
    """扫描项目的预标注结果，返回按 task id 排序的复核项。"""
    c = client or get_client()
    _emit(on_event, "review", "start", "加载预测…")

    predictions = c.predictions.list(project=project_id)
    by_task = _latest_predictions(predictions, version)

    # task_only 足够拿 is_labeled / annotations 计数，且体积小。
    tasks: dict[int, object] = {}
    for task in c.tasks.list(project=project_id, fields="task_only", page_size=100):
        if should_stop and should_stop():
            raise RuntimeError("已取消")
        tasks[int(task.id)] = task

    items: list[ReviewTask] = []
    for task_id, pred in sorted(by_task.items()):
        task = tasks.get(task_id)
        if task is None:
            continue
        labeled = bool(getattr(task, "is_labeled", False))
        ann = int(getattr(task, "total_annotations", 0) or 0)
        if only_unlabeled and (labeled or ann):
            continue
        items.append(ReviewTask(
            task_id=task_id,
            prediction_id=int(getattr(pred, "id", 0) or 0),
            version=str(getattr(pred, "model_version", "") or ""),
            result=list(getattr(pred, "result", []) or []),
            is_labeled=labeled,
            annotations=ann,
            image=str(getattr(task, "image", "") or ""),
        ))
        if limit and len(items) >= limit:
            break
        if len(items) % 100 == 0:
            _emit(on_event, "review", "progress",
                  f"整理 {len(items)} 条…", None, visible=False)

    _emit(on_event, "review", "done",
          f"共 {len(items)} 条待复核", 1.0)
    if log:
        log(f"预测任务 {len(by_task)} 条，待复核 {len(items)} 条")
    return items


def accept_items(project_id: int, items: Iterable[ReviewTask], *,
                 selections: dict[int, Iterable[int]] | None = None,
                 client=None, on_event=None,
                 log: Callable[[str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None) -> dict:
    """把选中的预测框写成 annotation（每个 task 一条）。

    返回统计里的 ``skipped_ids`` 只包含「已有标注而跳过」的任务：
    它们在待复核列表里也不该再出现，调用方可以直接从列表移除。
    用户主动不选任何框的任务不算在内。
    """
    c = client or get_client()
    items = list(items)
    stats = {"accepted": 0, "skipped": 0, "failed": 0, "boxes": 0,
             "accepted_ids": [], "skipped_ids": [], "errors": []}
    started = time.time()
    _emit(on_event, "accept", "start", f"认可 {len(items)} 条…")

    for i, item in enumerate(items, 1):
        if should_stop and should_stop():
            raise RuntimeError("已取消")
        if item.is_labeled or item.annotations:
            stats["skipped"] += 1
            stats["skipped_ids"].append(item.task_id)
            continue
        result = item.selected_result(
            (selections or {}).get(item.task_id))
        if not result:
            stats["skipped"] += 1
            continue
        try:
            # 复核时重新拉一次，避免列表加载后被别处标注过。
            task = c.tasks.get(item.task_id)
            if bool(getattr(task, "is_labeled", False)) or int(
                    getattr(task, "total_annotations", 0) or 0):
                stats["skipped"] += 1
                stats["skipped_ids"].append(item.task_id)
                continue
            c.annotations.create(
                item.task_id,
                result=result,
                project=project_id,
                was_cancelled=False,
                last_action="prediction",
                parent_prediction=item.prediction_id,
            )
            stats["accepted"] += 1
            stats["boxes"] += len(result)
            stats["accepted_ids"].append(item.task_id)
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append({"task_id": item.task_id, "error": str(e)})
            if log:
                log(f"⚠ 任务 {item.task_id} 认可失败: {e}")
        if i % 20 == 0 or i == len(items):
            _emit(on_event, "accept", "progress",
                  f"已处理 {i}/{len(items)}", i / len(items))

    _emit(on_event, "accept", "done",
          f"认可 {stats['accepted']} 条 / 跳过 {stats['skipped']} / 失败 {stats['failed']}",
          1.0)
    stats["elapsed"] = round(time.time() - started, 1)
    return stats


def task_image_url(client, task_id: int) -> str:
    """读取任务的图片地址；相对路径按 Label Studio base_url 补全。"""
    task = client.tasks.get(task_id)
    data = getattr(task, "data", None) or {}
    image = str(data.get("image") or "").strip()
    if not image:
        raise ValueError(f"任务 {task_id} 没有图片地址")
    base = str(getattr(client, "base_url", "") or "").rstrip("/")
    if image.startswith("/") and base:
        image = base + image
    return image


def fetch_image_bytes(client, url: str, *, timeout: float = 20.0) -> bytes:
    """下载任务图片；仅对 Label Studio 同源地址附带 API Token。"""
    if not url:
        raise ValueError("图片地址为空")
    base = str(getattr(client, "base_url", "") or "")
    headers = {}
    try:
        same_origin = urlparse(url).netloc == urlparse(base).netloc
    except ValueError:
        same_origin = True
    if same_origin and hasattr(client, "auth_headers"):
        headers = client.auth_headers()
    resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.content
