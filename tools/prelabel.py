#!/usr/bin/env python3
"""小模型预标注流水线 —— 评估 → 建集 → 训练 → 导出 ONNX → 预标注。

设计文档: docs/prelabel_design.md（v2）

CLI 示例:
  python prelabel.py assess -p 10
  python prelabel.py build  -p 10 --out runs/prelabel/demo/dataset
  python prelabel.py train  --data runs/prelabel/demo/dataset/data.yaml
  python prelabel.py predict -p 10 -m runs/prelabel/20261020-1810_v1/model/best.onnx
  python prelabel.py run    -p 10

  # 远程 Label Studio（连接参数跟在子命令后）
  python prelabel.py assess -p 10 --base-url http://192.168.1.10:8080 --token <TOKEN>
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlparse

import httpx
import numpy as np

from label_studio_auth import (
    add_connection_args,
    client_from_args,
    get_client,
    to_dict,
)


# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------

DEFAULT_MIN_LABELED = 50
DEFAULT_MIN_INSTANCES = 100
DEFAULT_MIN_PER_CLASS = 30
MARGINAL_FACTOR = 0.5
INVALID_WARN_RATIO = 0.05
INVALID_FAIL_RATIO = 0.15

RUN_ROOT = Path("runs/prelabel")
CACHE_DIR = Path.home() / ".cache" / "ctts-prelabel" / "images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CONTENT_TYPE_EXTS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
    "image/tiff": ".tif",
}

TAG_PREFIX = "ctts-prelabel"
BATCH_WRITE = 50
PROGRESS_STEP = 20

# 小目标信息项：面积占整图比例 < 0.5%
SMALL_AREA_RATIO = 0.005


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LabelConfig:
    labels: list[str]
    from_name: str = "label"
    to_name: str = "image"
    image_key: str = "image"


@dataclass
class Policy:
    min_labeled: int = DEFAULT_MIN_LABELED
    min_instances: int = DEFAULT_MIN_INSTANCES
    min_per_class: int = DEFAULT_MIN_PER_CLASS
    invalid_warn: float = INVALID_WARN_RATIO
    invalid_fail: float = INVALID_FAIL_RATIO

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Check:
    item: str
    require: str
    actual: object
    level: str  # pass | warn | fail


@dataclass
class Assessment:
    project: int
    verdict: str  # ready | marginal | insufficient
    labeled_tasks: int
    instances: int
    per_class: dict[str, int]
    invalid_boxes: int
    negative_tasks: int
    small_ratio: float
    checks: list[Check]
    suggestion: str | None = None
    policy: dict = field(default_factory=dict)
    # 信息项
    total_tasks: int = 0
    annotated_tasks: int = 0
    annotators: int = 0
    boxes_per_task: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetInfo:
    root: Path
    classes: list[str]
    train: int
    val: int
    skipped: int
    failures: list[dict]
    labeled_tasks: int = 0
    instances: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["root"] = str(self.root)
        return d


@dataclass
class TrainResult:
    run_dir: Path
    best_pt: Path
    metrics: dict
    onnx: Path | None = None
    manifest: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        d["best_pt"] = str(self.best_pt)
        d["onnx"] = str(self.onnx) if self.onnx else None
        return d


@dataclass
class PrelabelStats:
    processed: int
    skipped: int
    failed: int
    boxes: int
    empty: int
    elapsed_sec: float
    failures: list[dict] = field(default_factory=list)
    written: int = 0
    tag: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StageEvent:
    stage: str     # assess | dataset | train | export | prelabel
    status: str    # start | progress | done | error
    message: str = ""
    fraction: float | None = None
    visible: bool = True


class PipelineCancelled(RuntimeError):
    """用户请求取消流水线。"""


def _check_stop(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise PipelineCancelled("已取消")


def _emit(on_event, stage: str, status: str, message: str = "",
          fraction: float | None = None, visible: bool = True) -> None:
    if on_event is None:
        return
    on_event(StageEvent(stage=stage, status=status, message=message,
                        fraction=fraction, visible=visible))


# ---------------------------------------------------------------------------
# label_config 解析
# ---------------------------------------------------------------------------

def parse_label_config(label_config: str) -> LabelConfig:
    """解析 Label Studio 的 XML 配置，提取检测标签与字段名。

    Raises:
        ValueError: XML 无效，或缺少 RectangleLabels / Label。
    """
    if not label_config or not label_config.strip():
        raise ValueError("项目没有 label_config，请使用检测模板创建项目")
    try:
        root = ET.fromstring(label_config)
    except ET.ParseError as e:
        raise ValueError(f"label_config XML 解析失败: {e}") from e

    rect_tags = list(root.iter("RectangleLabels"))
    if not rect_tags:
        raise ValueError(
            "label_config 缺少 RectangleLabels（不是目标检测模板）。\n"
            "可用: python ctts.py label project create \"项目名\" -t detection"
        )

    tag = rect_tags[0]
    from_name = tag.get("name") or "label"
    to_name = tag.get("toName") or "image"

    labels: list[str] = []
    for label in tag.findall("Label"):
        value = (label.get("value") or "").strip()
        if value and value not in labels:
            labels.append(value)
    if not labels:
        raise ValueError("RectangleLabels 下没有 Label，无法确定类别")

    image_key = "image"
    for image in root.iter("Image"):
        value = (image.get("value") or "").strip()
        if value:
            image_key = value[1:] if value.startswith("$") else value
            break
        name = image.get("name")
        if name:
            image_key = name
            break

    return LabelConfig(
        labels=labels,
        from_name=from_name,
        to_name=to_name,
        image_key=image_key,
    )


# ---------------------------------------------------------------------------
# annotation 处理
# ---------------------------------------------------------------------------

@dataclass
class ParsedAnnotation:
    lines: list[str] = field(default_factory=list)
    invalid: int = 0
    small: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)

    @property
    def instances(self) -> int:
        return len(self.lines)


def latest_annotation(task: dict) -> dict | None:
    """取任务最新一条未取消的 annotation（按 updated_at / created_at）。"""
    anns = [to_dict(a) for a in (task.get("annotations") or [])]
    anns = [a for a in anns if not a.get("was_cancelled")]
    if not anns:
        return None
    return max(
        anns,
        key=lambda a: str(a.get("updated_at") or a.get("created_at") or ""),
    )


def _rect_results(annotation: dict) -> list[dict]:
    out = []
    for item in annotation.get("result") or []:
        item = to_dict(item)
        if item.get("type") == "rectanglelabels":
            out.append(item)
    return out


def parse_annotation(annotation: dict, cfg: LabelConfig) -> ParsedAnnotation:
    """把一条 LS annotation 转成 YOLO 标签行（百分比透传）。"""
    parsed = ParsedAnnotation(per_class={label: 0 for label in cfg.labels})
    for item in _rect_results(annotation):
        value = item.get("value") or {}
        labels = value.get("rectanglelabels") or []
        if not labels:
            parsed.invalid += 1
            continue
        label = str(labels[0])
        if label not in cfg.labels:
            parsed.unknown.append(label)
            parsed.invalid += 1
            continue

        try:
            x = float(value["x"])
            y = float(value["y"])
            w = float(value["width"])
            h = float(value["height"])
        except (KeyError, TypeError, ValueError):
            parsed.invalid += 1
            continue

        if not all(math.isfinite(v) for v in (x, y, w, h)):
            parsed.invalid += 1
            continue
        # 越界视为无效（LS 正常数据不会出现；容差 1e-6 防浮点尾差）
        if (w <= 0 or h <= 0 or x < -1e-6 or y < -1e-6
                or x + w > 100 + 1e-6 or y + h > 100 + 1e-6):
            parsed.invalid += 1
            continue

        cls_id = cfg.labels.index(label)
        cx = min(max((x + w / 2) / 100.0, 0.0), 1.0)
        cy = min(max((y + h / 2) / 100.0, 0.0), 1.0)
        nw = min(max(w / 100.0, 0.0), 1.0)
        nh = min(max(h / 100.0, 0.0), 1.0)
        if nw <= 0 or nh <= 0:
            parsed.invalid += 1
            continue

        parsed.lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        parsed.per_class[label] += 1
        if (w / 100.0) * (h / 100.0) < SMALL_AREA_RATIO:
            parsed.small += 1
    return parsed


# ---------------------------------------------------------------------------
# 数据就绪评估
# ---------------------------------------------------------------------------

def _check_level(actual: float, minimum: float) -> str:
    if actual >= minimum:
        return "pass"
    if actual >= minimum * MARGINAL_FACTOR:
        return "warn"
    return "fail"


def _build_suggestion(a: Assessment, policy: Policy) -> str | None:
    reasons: list[str] = []
    if a.verdict == "insufficient":
        if a.labeled_tasks < policy.min_labeled:
            reasons.append(f"还差 {policy.min_labeled - a.labeled_tasks} 张已标注图")
        if a.instances < policy.min_instances:
            reasons.append(f"还差 {policy.min_instances - a.instances} 个框")
        for label in a.per_class:
            need = policy.min_per_class
            have = a.per_class[label]
            if have < need:
                reasons.append(f"`{label}` 还差 {need - have} 个框")
        if reasons:
            return "；".join(reasons) + "。建议优先标遮挡、密集、远距离帧。"
        return "数据量不足，请先补充标注。"

    if a.verdict == "marginal":
        gaps = []
        if a.labeled_tasks < policy.min_labeled:
            gaps.append(f"再标 {policy.min_labeled - a.labeled_tasks} 张")
        if a.instances < policy.min_instances:
            gaps.append(f"再补 {policy.min_instances - a.instances} 个框")
        for label, have in a.per_class.items():
            if have < policy.min_per_class:
                gaps.append(f"`{label}` 再补 {policy.min_per_class - have} 个框")
        base = "数据偏少，训练可能不稳定"
        if gaps:
            base += "；建议" + "、".join(gaps)
        base += "。可确认继续，或先补标。"
        reasons.append(base)

    total = a.instances + a.invalid_boxes
    invalid_ratio = a.invalid_boxes / total if total else 0.0
    if invalid_ratio > policy.invalid_warn:
        reasons.append(f"无效框占比 {invalid_ratio * 100:.1f}%，建议先修正标注。")

    return " ".join(reasons) if reasons else None


def assess_tasks(
    tasks: Iterable[dict],
    cfg: LabelConfig,
    *,
    project_id: int = 0,
    total_tasks: int = 0,
    policy: Policy | None = None,
    log: Callable[[str], None] | None = None,
) -> Assessment:
    """离线评估（不访问 LS），供 assess_project 与单测复用。"""
    policy = policy or Policy()
    per_class: dict[str, int] = {label: 0 for label in cfg.labels}
    labeled = 0
    instances = 0
    invalid = 0
    negative = 0
    small = 0
    annotated = 0
    annotators: set = set()
    boxes_per_task: list[int] = []

    for i, task in enumerate(tasks, 1):
        annotation = latest_annotation(to_dict(task))
        if annotation is None:
            continue
        annotated += 1
        completed_by = annotation.get("completed_by")
        if completed_by is not None:
            annotators.add(str(completed_by))

        rects = _rect_results(annotation)
        if not rects:
            negative += 1
            continue

        parsed = parse_annotation(annotation, cfg)
        invalid += parsed.invalid
        small += parsed.small
        if not parsed.lines:
            continue
        labeled += 1
        instances += parsed.instances
        boxes_per_task.append(parsed.instances)
        for label, n in parsed.per_class.items():
            if label in per_class:
                per_class[label] += n

        if log and i % 200 == 0:
            log(f"  已扫描 {i} 条已标注任务…")

    checks: list[Check] = []
    checks.append(Check("已标注任务", f">={policy.min_labeled}", labeled,
                        _check_level(labeled, policy.min_labeled)))
    checks.append(Check("实例总数", f">={policy.min_instances}", instances,
                        _check_level(instances, policy.min_instances)))
    for label in cfg.labels:
        checks.append(Check(
            f"{label} 实例", f">={policy.min_per_class}", per_class[label],
            _check_level(per_class[label], policy.min_per_class)))

    total = instances + invalid
    invalid_ratio = invalid / total if total else 0.0
    if invalid_ratio <= policy.invalid_warn:
        invalid_level = "pass"
    elif invalid_ratio <= policy.invalid_fail:
        invalid_level = "warn"
    else:
        invalid_level = "fail"
    checks.append(Check("无效框占比", f"<={policy.invalid_warn * 100:.0f}%",
                        f"{invalid_ratio * 100:.1f}%", invalid_level))
    checks.append(Check("类别覆盖", "每类 >0",
                        f"{sum(1 for n in per_class.values() if n > 0)}/{len(cfg.labels)}",
                        "pass" if all(n > 0 for n in per_class.values()) else "fail"))

    hard_ok = (
        labeled >= policy.min_labeled * MARGINAL_FACTOR
        and instances >= policy.min_instances * MARGINAL_FACTOR
        and all(n >= policy.min_per_class * MARGINAL_FACTOR
                for n in per_class.values() if n > 0)
    )
    ready = (
        labeled >= policy.min_labeled
        and instances >= policy.min_instances
        and all(n > 0 for n in per_class.values())
        and all(n >= policy.min_per_class for n in per_class.values() if n > 0)
    )
    if not hard_ok:
        verdict = "insufficient"
    elif ready:
        verdict = "ready"
    else:
        verdict = "marginal"

    a = Assessment(
        project=project_id,
        verdict=verdict,
        labeled_tasks=labeled,
        instances=instances,
        per_class=per_class,
        invalid_boxes=invalid,
        negative_tasks=negative,
        small_ratio=(small / instances) if instances else 0.0,
        checks=checks,
        policy=policy.to_dict(),
        total_tasks=total_tasks,
        annotated_tasks=annotated,
        annotators=len(annotators),
        boxes_per_task=float(statistics.median(boxes_per_task)) if boxes_per_task else 0.0,
    )
    a.suggestion = _build_suggestion(a, policy)
    return a


def assess_project(
    project_id: int,
    *,
    client=None,
    policy: Policy | None = None,
    log: Callable[[str], None] | None = None,
) -> Assessment:
    """通过 Label Studio API 评估数据是否足够训练 nano 模型。"""
    c = client or get_client()
    if log:
        log(f"评估项目 {project_id} 的已标注数据…")

    project = to_dict(c.projects.get(project_id))
    cfg = parse_label_config(project.get("label_config") or "")

    tasks = [
        to_dict(t)
        for t in c.tasks.list(
            project=project_id,
            only_annotated=True,
            fields="all",
            page_size=200,
        )
    ]
    return assess_tasks(
        tasks,
        cfg,
        project_id=project_id,
        total_tasks=project.get("task_number") or len(tasks),
        policy=policy,
        log=log,
    )


# ---------------------------------------------------------------------------
# 图片获取（本地映射 / 共享缓存 / HTTP 下载）
# ---------------------------------------------------------------------------

def resolve_image_url(raw: str, base_url: str) -> str | None:
    """把任务字段解析为可下载 URL；本机路径返回 None。"""
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith(("http://", "https://")):
        return raw
    if raw.startswith("/"):
        return base_url.rstrip("/") + raw
    return None


def _image_ext(url: str, content_type: str = "") -> str:
    ext = CONTENT_TYPE_EXTS.get((content_type or "").split(";")[0].strip().lower())
    if ext:
        return ext
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    return ".jpg"


def _cache_file(url: str) -> tuple[Path, str]:
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, key


def _local_image(name: str, images_dir: str | None) -> Path | None:
    if not images_dir:
        return None
    base = Path(images_dir)
    if not name:
        return None
    candidate = base / name
    if candidate.is_file():
        return candidate
    # 允许 URL 编码过的中文文件名
    decoded = unquote(name)
    candidate = base / decoded
    return candidate if candidate.is_file() else None


def _ensure_image(
    raw: str,
    *,
    base_url: str,
    images_dir: str | None = None,
    client=None,
    timeout: float = 60.0,
) -> Path:
    """返回本地图片路径：--images-dir 映射 → 缓存 → HTTP 下载。"""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("任务没有 image 字段")

    direct = Path(raw)
    if direct.is_file():
        return direct

    url = resolve_image_url(raw, base_url)
    name = Path(urlparse(raw).path).name if url else raw
    local = _local_image(name, images_dir)
    if local:
        return local

    if not url:
        path = Path(raw)
        if path.is_file():
            return path
        raise ValueError(f"图片字段不是 URL，且本地不存在: {raw}")

    cache_dir, key = _cache_file(url)
    for cached in cache_dir.glob(f"{key}.*"):
        if cached.is_file() and cached.suffix != ".tmp":
            return cached

    headers: dict[str, str] = {}
    if urlparse(url).netloc.lower() == urlparse(base_url).netloc.lower():
        getter = getattr(client, "auth_headers", None) or getattr(client, "_headers", None)
        if callable(getter):
            headers = getter()
    # S3 预签名 URL 不能带 Authorization，上面的同源判断已保证

    part = cache_dir / f"{key}.tmp"
    with httpx.stream("GET", url, headers=headers, timeout=timeout,
                      follow_redirects=True) as resp:
        resp.raise_for_status()
        ext = _image_ext(url, resp.headers.get("content-type", ""))
        dest = cache_dir / f"{key}{ext}"
        with open(part, "wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 16):
                fh.write(chunk)
    os.replace(part, dest)
    return dest


def _task_image_raw(task: dict, cfg: LabelConfig) -> str:
    data = task.get("data") or {}
    raw = data.get(cfg.image_key)
    if raw:
        return str(raw)
    for value in data.values():
        if isinstance(value, str) and value:
            return value
    return ""


def _prefetch_images(
    tasks: list[dict],
    cfg: LabelConfig,
    *,
    client,
    base_url: str,
    images_dir: str | None,
    on_event=None,
    log: Callable[[str], None] | None = None,
) -> dict[int, Path | Exception]:
    results: dict[int, Path | Exception] = {}
    if not tasks:
        return results
    done = 0

    def work(task: dict):
        raw = _task_image_raw(task, cfg)
        return _ensure_image(raw, base_url=base_url, images_dir=images_dir, client=client)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(work, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                results[int(task["id"])] = fut.result()
            except Exception as e:  # 单张失败不中断，由调用方按比例判定
                results[int(task["id"])] = e
            done += 1
            if on_event and (done % PROGRESS_STEP == 0 or done == len(tasks)):
                _emit(on_event, "dataset", "progress",
                      f"图片准备 {done}/{len(tasks)}", done / len(tasks),
                      visible=False)
    return results


# ---------------------------------------------------------------------------
# 微数据集构建
# ---------------------------------------------------------------------------

def build_micro_dataset(
    project_id: int,
    *,
    out_dir: str,
    client=None,
    split_ratio: float = 0.2,
    seed: int = 42,
    images_dir: str | None = None,
    on_event=None,
    should_stop: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> DatasetInfo:
    """从已标注任务生成 YOLO 微数据集并划分 train/val。"""
    c = client or get_client()
    base_url = getattr(c, "base_url", "http://127.0.0.1:8080")
    _emit(on_event, "dataset", "start", "构建微数据集…")

    project = to_dict(c.projects.get(project_id))
    cfg = parse_label_config(project.get("label_config") or "")
    tasks = [
        to_dict(t)
        for t in c.tasks.list(project=project_id, only_annotated=True,
                              fields="all", page_size=200)
    ]
    if not tasks:
        raise RuntimeError("没有已标注任务，无法构建数据集")

    root = Path(out_dir)
    if root.name == "dataset":
        run_root = root.parent
    else:
        run_root = root
    images_out = root / "images"
    labels_out = root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    if log:
        log(f"已标注任务 {len(tasks)} 条，准备图片…")
    fetched = _prefetch_images(tasks, cfg, client=c, base_url=base_url,
                               images_dir=images_dir, on_event=on_event, log=log)

    failures: list[dict] = []
    skipped = 0
    labeled_tasks = 0
    instances = 0
    written = 0

    for i, task in enumerate(tasks, 1):
        _check_stop(should_stop)
        task_id = int(task["id"])
        annotation = latest_annotation(task)
        if annotation is None:
            skipped += 1
            continue
        parsed = parse_annotation(annotation, cfg)
        if parsed.unknown:
            if log:
                log(f"⚠ 任务 {task_id} 出现未配置类别 {sorted(set(parsed.unknown))}，已跳过这些框")

        image = fetched.get(task_id)
        if isinstance(image, Exception) or image is None:
            failures.append({"task_id": task_id,
                             "error": str(image) if image else "图片缺失"})
            continue

        ext = image.suffix if image.suffix.lower() in IMAGE_EXTS else ".jpg"
        dest_img = images_out / f"{task_id}{ext}"
        try:
            shutil.copy2(image, dest_img)
        except OSError as e:
            failures.append({"task_id": task_id, "error": f"复制图片失败: {e}"})
            continue

        (labels_out / f"{task_id}.txt").write_text(
            "\n".join(parsed.lines) + ("\n" if parsed.lines else ""))
        written += 1
        if parsed.lines:
            labeled_tasks += 1
            instances += parsed.instances
        if on_event and (i % PROGRESS_STEP == 0 or i == len(tasks)):
            _emit(on_event, "dataset", "progress",
                  f"写入 {i}/{len(tasks)}", i / len(tasks), visible=False)

    failure_ratio = len(failures) / len(tasks) if tasks else 0.0
    if failure_ratio > 0.10:
        raise RuntimeError(
            f"图片获取失败率 {failure_ratio * 100:.1f}% > 10%，已中止建集。\n"
            f"失败示例: {failures[:3]}\n"
            "检查网络，或用 --images-dir 指定本地图片目录。"
        )
    if written < 2:
        raise RuntimeError(f"可用图片只有 {written} 张，无法训练")

    (root / "classes.txt").write_text("\n".join(cfg.labels) + "\n")

    # 复用现有划分逻辑，产物为 train/ val/ data.yaml
    from split_dataset import split_dataset

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        split_dataset(root, val_ratio=split_ratio, seed=seed)
    if log:
        log(buf.getvalue().strip())

    train = len(list((root / "train" / "images").glob("*")))
    val = len(list((root / "val" / "images").glob("*")))
    if train < 10 or val < 2:
        raise RuntimeError(
            f"划分后 train {train} 张 / val {val} 张，低于最低要求（10/2）。"
            "请补充标注后重试。"
        )

    info = DatasetInfo(
        root=root,
        classes=cfg.labels,
        train=train,
        val=val,
        skipped=skipped,
        failures=failures,
        labeled_tasks=labeled_tasks,
        instances=instances,
    )
    _emit(on_event, "dataset", "done",
          f"微数据集就绪：train {train} / val {val}", 1.0)
    return info


# ---------------------------------------------------------------------------
# 训练与 ONNX 导出
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> tuple[str, bool]:
    """返回 (ultralytics device, 是否 GPU)。"""
    if device in (None, "", "auto"):
        try:
            import torch
            return ("0", True) if torch.cuda.is_available() else ("cpu", False)
        except Exception:
            return "cpu", False
    gpu = device not in ("cpu",)
    return device, gpu


def _resolve_base_weights(base_weights: str) -> str:
    """本地权重不存在时退回文件名，交给 ultralytics 自动下载。"""
    path = Path(base_weights)
    if path.is_file():
        return str(path)
    if path.name and path.name != base_weights:
        return path.name
    return base_weights


def train_micro(
    dataset: DatasetInfo,
    *,
    base_weights: str,
    epochs: int = 150,
    imgsz: int = 480,
    freeze: int = 8,
    device: str = "auto",
    batch: int | None = None,
    run_dir: str | None = None,
    on_event=None,
    should_stop: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> TrainResult:
    """在微数据集上微调 nano 模型。"""
    from ultralytics import YOLO

    data_yaml = dataset.root / "data.yaml"
    if not data_yaml.is_file():
        raise RuntimeError(f"缺少 data.yaml: {data_yaml}")

    run_root = Path(run_dir) if run_dir else (
        RUN_ROOT / f"{datetime.now():%Y%m%d-%H%M%S}")
    run_root.mkdir(parents=True, exist_ok=True)

    dev, gpu = _resolve_device(device)
    batch_size = batch or (16 if gpu else 8)
    workers = 8 if gpu else 2
    amp = gpu
    _emit(on_event, "train", "start",
          f"训练 {Path(base_weights).name}（{'GPU' if gpu else 'CPU'}，batch {batch_size}）")

    load_weights = _resolve_base_weights(base_weights)
    model = YOLO(load_weights)
    state = {"attempt": 0}

    def on_epoch(trainer):
        _check_stop(should_stop)
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            metrics = getattr(trainer, "metrics", {}) or {}
            m50 = metrics.get("metrics/mAP50(B)")
            msg = f"epoch {epoch}/{epochs}"
            if m50 is not None:
                msg += f" · mAP50 {float(m50):.2f}"
            _emit(on_event, "train", "progress", msg, epoch / epochs)

    model.add_callback("on_train_epoch_end", on_epoch)

    def do_train():
        return model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            freeze=freeze,
            batch=batch_size,
            device=dev,
            workers=workers,
            amp=amp,
            cos_lr=True,
            patience=20,
            verbose=False,
            project=str(run_root / "train"),
            name="exp",
            exist_ok=True,
            seed=0,
        )

    try:
        do_train()
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or state["attempt"] >= 1 or batch_size <= 4:
            raise
        state["attempt"] += 1
        batch_size = max(4, batch_size // 2)
        if log:
            log(f"⚠ 显存不足，batch 降为 {batch_size} 重试一次…")
        _emit(on_event, "train", "progress",
              f"显存不足，batch 降为 {batch_size} 重试", 0.0)
        model = YOLO(load_weights)
        model.add_callback("on_train_epoch_end", on_epoch)
        do_train()

    trainer = getattr(model, "trainer", None)
    best = Path(getattr(trainer, "best", "") or (run_root / "train" / "exp" / "weights" / "best.pt"))
    if not best.is_file():
        best = run_root / "train" / "exp" / "weights" / "last.pt"
    if not best.is_file():
        raise RuntimeError(f"训练完成但找不到权重文件: {best}")

    metrics: dict[str, float] = {}
    try:
        val_result = model.val(data=str(data_yaml), split="val", verbose=False)
        raw = getattr(val_result, "results_dict", {}) or {}
        mapping = {
            "mAP50": "metrics/mAP50(B)",
            "mAP50_95": "metrics/mAP50-95(B)",
            "precision": "metrics/precision(B)",
            "recall": "metrics/recall(B)",
        }
        for name, key in mapping.items():
            if key in raw:
                metrics[name] = float(raw[key])
    except Exception as e:
        if log:
            log(f"⚠ 验证集评估失败（不影响权重）: {e}")

    result = TrainResult(run_dir=run_root, best_pt=best, metrics=metrics)
    _emit(on_event, "train", "done",
          f"训练完成：mAP50 {metrics.get('mAP50', 0):.2f}", 1.0)
    return result


def export_onnx(result: TrainResult, *, imgsz: int = 480,
                on_event=None, log: Callable[[str], None] | None = None) -> Path:
    """导出 best.pt → model/best.onnx，并做一次可加载性自检。"""
    from ultralytics import YOLO

    if not Path(result.best_pt).is_file():
        raise RuntimeError(f"权重不存在: {result.best_pt}")
    _emit(on_event, "export", "start", "导出 ONNX…")

    model = YOLO(str(result.best_pt))
    exported = Path(model.export(format="onnx", imgsz=imgsz,
                                 dynamic=False, simplify=True))
    if not exported.is_file():
        raise RuntimeError(f"ONNX 导出失败: {exported}")

    dest_dir = Path(result.run_dir) / "model"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "best.onnx"
    shutil.copy2(exported, dest)

    try:
        check = YOLO(str(dest))
        if not getattr(check, "names", None):
            raise RuntimeError("ONNX 加载后没有类别信息")
        val_images = list((Path(result.run_dir) / "dataset" / "val" / "images").glob("*"))
        if val_images:
            check.predict(source=str(val_images[0]), imgsz=imgsz, verbose=False)
    except Exception as e:
        raise RuntimeError(f"ONNX 自检失败: {e}") from e

    result.onnx = dest
    _emit(on_event, "export", "done", "ONNX 已导出", 1.0)
    return dest


# ---------------------------------------------------------------------------
# 版本与清单
# ---------------------------------------------------------------------------

def next_version(run_root: str | Path | None = None) -> str:
    """扫描已有 manifest 的版本号并自增。"""
    root = Path(run_root) if run_root else RUN_ROOT
    versions: list[int] = []
    for manifest in root.glob("*/model/manifest.json"):
        try:
            data = json.loads(manifest.read_text())
            version = str(data.get("version", ""))
            if version.startswith("v") and version[1:].isdigit():
                versions.append(int(version[1:]))
        except (OSError, ValueError, TypeError):
            continue
    return f"v{max(versions, default=0) + 1}"


def make_tag(version: str, imgsz: int) -> str:
    return f"{TAG_PREFIX}:{version}@{imgsz}"


def tag_from_manifest(model_path: str | Path, imgsz: int) -> str | None:
    """从模型附近的 manifest 读取 tag；没有则返回 None。"""
    path = Path(model_path)
    candidates = [
        path.parent / "manifest.json",
        path.parent.parent / "manifest.json",
        path.parent.parent.parent / "model" / "manifest.json",
    ]
    for manifest in candidates:
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        tag = data.get("tag")
        if tag:
            return str(tag)
        version = data.get("version")
        if version:
            return make_tag(str(version), imgsz)
    return None


def _write_manifest(result: TrainResult, *, version: str, run_id: str,
                    base: str, imgsz: int, dataset: DatasetInfo,
                    ls_project: int, marginal: bool) -> dict:
    tag = make_tag(version, imgsz)
    manifest = {
        "version": version,
        "run_id": run_id,
        "base": base,
        "weights": str(Path(result.best_pt).relative_to(result.run_dir)),
        "onnx": "model/best.onnx",
        "imgsz": imgsz,
        "classes": dataset.classes,
        "dataset": {
            "labeled_tasks": dataset.labeled_tasks,
            "instances": dataset.instances,
            "train": dataset.train,
            "val": dataset.val,
        },
        "metrics": result.metrics,
        "marginal": marginal,
        "tag": tag,
        "ls_project": ls_project,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    model_dir = Path(result.run_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# 预标注
# ---------------------------------------------------------------------------

def resolve_class_mapping(
    model_names: dict | list,
    labels: list[str],
    class_map: dict | None = None,
) -> dict[int, str]:
    """模型类别 → LS 标签名映射（精确 → 忽略大小写 → 显式覆盖）。"""
    if isinstance(model_names, dict):
        items = sorted(model_names.items())
    else:
        items = list(enumerate(model_names))
    lowered = {label.lower(): label for label in labels}
    mapping: dict[int, str] = {}
    for idx, name in items:
        name = str(name)
        target = None
        if class_map:
            target = class_map.get(name) or class_map.get(str(idx))
        if not target:
            if name in labels:
                target = name
            elif name.lower() in lowered:
                target = lowered[name.lower()]
        if target in labels:
            mapping[int(idx)] = target
    return mapping


def _to_numpy(value):
    """兼容 torch.Tensor 与 numpy 数组。"""
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _prediction_items(result, cfg: LabelConfig, mapping: dict[int, str],
                      conf: float) -> tuple[list[dict], list[float]]:
    """YOLO result → LS rectanglelabels 结果项（百分比坐标）。"""
    height, width = result.orig_shape
    if width <= 0 or height <= 0:
        return [], []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], []

    xyxy = _to_numpy(boxes.xyxy)
    if len(xyxy) == 0:
        return [], []
    confs = _to_numpy(boxes.conf)
    clss = _to_numpy(boxes.cls).astype(int)

    items: list[dict] = []
    scores: list[float] = []
    for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
        if float(score) < conf:
            continue
        label = mapping.get(int(cls_id))
        if label is None:
            continue
        x1 = min(max(float(x1), 0.0), width)
        x2 = min(max(float(x2), 0.0), width)
        y1 = min(max(float(y1), 0.0), height)
        y2 = min(max(float(y2), 0.0), height)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        w_pct = (x2 - x1) / width * 100.0
        h_pct = (y2 - y1) / height * 100.0
        x_pct = x1 / width * 100.0
        y_pct = y1 / height * 100.0
        items.append({
            "from_name": cfg.from_name,
            "to_name": cfg.to_name,
            "type": "rectanglelabels",
            "original_width": int(width),
            "original_height": int(height),
            "image_rotation": 0,
            "value": {
                "x": round(x_pct, 4),
                "y": round(y_pct, 4),
                "width": round(w_pct, 4),
                "height": round(h_pct, 4),
                "rotation": 0,
                "rectanglelabels": [label],
            },
            "score": round(float(score), 4),
        })
        scores.append(float(score))
    return items, scores


def _write_predictions(client, project_id: int, tag: str,
                       batch: list[tuple[int, list[dict], float]],
                       log: Callable[[str], None] | None = None) -> tuple[int, list[dict]]:
    """批量写预测；批量失败时逐条降级。返回 (成功数, 失败列表)。"""
    from label_studio_sdk.types.prediction_request import PredictionRequest

    try:
        requests = [
            PredictionRequest(task=task_id, result=results,
                              score=score, model_version=tag,
                              project=project_id)
            for task_id, results, score in batch
        ]
        client.projects.import_predictions(project_id, request=requests)
        return len(batch), []
    except Exception as e:
        if log:
            log(f"⚠ 批量写入失败（{e}），降级逐条写入…")

    ok = 0
    failures: list[dict] = []
    for task_id, results, score in batch:
        try:
            client.predictions.create(task=task_id, result=results,
                                      score=score, model_version=tag,
                                      project=project_id)
            ok += 1
        except Exception as e:
            failures.append({"task_id": task_id, "error": str(e)})
    return ok, failures


def prelabel_project(
    project_id: int,
    *,
    model: str,
    tag: str | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 480,
    overwrite: bool = False,
    limit: int = 0,
    device: str = "auto",
    images_dir: str | None = None,
    client=None,
    on_event=None,
    should_stop: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> PrelabelStats:
    """用 ONNX/pt 模型给未标注任务写 LS predictions。"""
    from ultralytics import YOLO

    c = client or get_client()
    base_url = getattr(c, "base_url", "http://127.0.0.1:8080")
    started = time.time()

    project = to_dict(c.projects.get(project_id))
    cfg = parse_label_config(project.get("label_config") or "")

    _emit(on_event, "prelabel", "start", "加载模型并筛选任务…")
    yolo = YOLO(str(model))
    names = getattr(yolo, "names", {}) or {}
    mapping = resolve_class_mapping(names, cfg.labels)
    if not mapping:
        model_labels = sorted(str(v) for v in (
            names.values() if isinstance(names, dict) else names))
        raise ValueError(
            "模型类别与 label_config 没有交集，已中止。\n"
            f"  模型: {model_labels}\n  项目: {cfg.labels}\n"
            "请检查训练数据类别或项目模板。"
        )

    tag = tag or tag_from_manifest(model, imgsz) or make_tag("adhoc", imgsz)
    if log:
        log(f"版本标签: {tag}")

    # 已有同 tag 预测：默认跳过；overwrite 时删除
    old_pred_ids: dict[int, list[int]] = {}
    if overwrite:
        for pred in c.predictions.list(project=project_id):
            pred = to_dict(pred)
            if pred.get("model_version") == tag:
                old_pred_ids.setdefault(int(pred.get("task")), []).append(int(pred["id"]))

    tasks = [to_dict(t) for t in c.tasks.list(project=project_id, fields="all",
                                              page_size=200)]
    targets: list[dict] = []
    skipped = 0
    for task in tasks:
        if task.get("is_labeled"):
            continue
        same_tag = any(
            to_dict(p).get("model_version") == tag
            for p in (task.get("predictions") or [])
        )
        if same_tag and not overwrite:
            skipped += 1
            continue
        targets.append(task)
        if limit and len(targets) >= limit:
            break

    total = len(targets)
    if log:
        log(f"待预标注 {total} 张（跳过 {skipped} 张）")
    _emit(on_event, "prelabel", "start", f"预标注 {total} 张", 0.0)

    dev, _ = _resolve_device(device)
    stats = PrelabelStats(processed=0, skipped=skipped, failed=0, boxes=0,
                          empty=0, elapsed_sec=0.0, tag=tag)
    batch: list[tuple[int, list[dict], float]] = []
    written = 0

    def flush():
        nonlocal written, batch
        if not batch:
            return
        ok, fails = _write_predictions(c, project_id, tag, batch, log=log)
        written += ok
        stats.failed += len(fails)
        stats.failures.extend(fails)
        batch = []

    for i, task in enumerate(targets, 1):
        _check_stop(should_stop)
        task_id = int(task["id"])
        try:
            raw = _task_image_raw(task, cfg)
            image = _ensure_image(raw, base_url=base_url,
                                  images_dir=images_dir, client=c)
            results = yolo.predict(source=str(image), imgsz=imgsz, conf=conf,
                                   iou=iou, max_det=300, device=dev,
                                   verbose=False)
            if not results:
                stats.empty += 1
                stats.processed += 1
                continue
            items, scores = _prediction_items(results[0], cfg, mapping, conf)
            stats.processed += 1
            if not items:
                stats.empty += 1
            else:
                stats.boxes += len(items)
                if overwrite:
                    for pred_id in old_pred_ids.get(task_id, []):
                        try:
                            c.predictions.delete(pred_id)
                        except Exception as e:
                            if log:
                                log(f"⚠ 删除旧预测 {pred_id} 失败: {e}")
                batch.append((task_id, items,
                              sum(scores) / len(scores) if scores else 0.0))
                if len(batch) >= BATCH_WRITE:
                    flush()
        except Exception as e:
            stats.failed += 1
            stats.failures.append({"task_id": task_id, "error": str(e)})

        if on_event and (i % PROGRESS_STEP == 0 or i == total):
            _emit(on_event, "prelabel", "progress",
                  f"已处理 {i}/{total}", i / total if total else None)

    flush()
    stats.written = written
    stats.elapsed_sec = round(time.time() - started, 2)
    _emit(on_event, "prelabel", "done",
          f"预标注完成：写入 {written} 张 / 失败 {stats.failed}", 1.0)
    return stats


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def run_pipeline(
    project_id: int,
    *,
    base_weights: str = "weights/yolo26n.pt",
    epochs: int = 150,
    imgsz: int = 480,
    conf: float = 0.25,
    iou: float = 0.7,
    freeze: int = 8,
    device: str = "auto",
    reuse_model: str | None = None,
    skip_train: bool = False,
    images_dir: str | None = None,
    on_event=None,
    should_stop: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """评估 → 建集 → 训练 → 导出 → 预标注。"""
    report: dict[str, Any] = {
        "project": project_id,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    if skip_train or reuse_model:
        model = reuse_model or str(Path("runs") / "prelabel" / "model" / "best.onnx")
        if not Path(model).is_file():
            raise RuntimeError(f"复用模型不存在: {model}")
        if log:
            log(f"跳过训练，直接预标注：{model}")
        stats = prelabel_project(project_id, model=model, imgsz=imgsz,
                                 conf=conf, iou=iou, device=device,
                                 images_dir=images_dir,
                                 on_event=on_event, should_stop=should_stop,
                                 log=log)
        report["prelabel"] = stats.to_dict()
        return report

    current_stage = "assess"
    current_run_dir: Path | None = None
    try:
        _emit(on_event, "assess", "start", "评估数据…")
        assessment = assess_project(project_id, log=log)
        report["assessment"] = assessment.to_dict()
        _emit(on_event, "assess", "done",
              f"评估完成：{assessment.verdict}（{assessment.labeled_tasks} 张 / "
              f"{assessment.instances} 框）", 1.0)
        if assessment.verdict == "insufficient":
            raise RuntimeError(assessment.suggestion or "数据不足，请补充标注")

        version = next_version()
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}"
        run_dir = RUN_ROOT / f"{run_id}_{version}"
        current_run_dir = run_dir
        current_stage = "dataset"
        dataset = build_micro_dataset(
            project_id, out_dir=str(run_dir / "dataset"), images_dir=images_dir,
            on_event=on_event, should_stop=should_stop, log=log)
        report["dataset"] = dataset.to_dict()

        current_stage = "train"
        train_result = train_micro(
            dataset, base_weights=base_weights, epochs=epochs, imgsz=imgsz,
            freeze=freeze, device=device, run_dir=str(run_dir),
            on_event=on_event, should_stop=should_stop,
            log=log)
        report["train"] = train_result.to_dict()

        current_stage = "export"
        export_onnx(train_result, imgsz=imgsz, on_event=on_event, log=log)
        report["train"]["onnx"] = str(train_result.onnx)

        manifest = _write_manifest(
            train_result, version=version, run_id=run_id, base=base_weights,
            imgsz=imgsz, dataset=dataset, ls_project=project_id,
            marginal=assessment.verdict == "marginal")
        report["manifest"] = manifest

        current_stage = "prelabel"
        stats = prelabel_project(
            project_id, model=str(train_result.onnx), tag=manifest["tag"],
            conf=conf, iou=iou, imgsz=imgsz, device=device,
            images_dir=images_dir,
            on_event=on_event, should_stop=should_stop, log=log)
        report["prelabel"] = stats.to_dict()
    except Exception as e:
        report["error"] = str(e)
        _emit(on_event, current_stage, "error", f"失败：{e}")
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if current_run_dir is not None:
            try:
                (Path(current_run_dir) / "report.json").write_text(
                    json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
            except OSError:
                pass

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_assessment(a: Assessment) -> None:
    print(f"\n项目 {a.project} · 已标注 {a.annotated_tasks} 条 / 总计 {a.total_tasks or '?'} 张")
    print(f"判定: {a.verdict}")
    print(f"摘要: {a.labeled_tasks} 张 · {a.instances} 个框 · "
          f"无效 {a.invalid_boxes} · 负样本 {a.negative_tasks} · "
          f"小目标 {a.small_ratio * 100:.0f}%")
    print(f"{'检查项':<14}{'门槛':<12}{'实测':<12}结论")
    print("-" * 48)
    for c in a.checks:
        print(f"{c.item:<14}{c.require:<12}{str(c.actual):<12}{c.level}")
    if a.suggestion:
        print(f"\n建议: {a.suggestion}")


def main():
    conn = argparse.ArgumentParser(add_help=False)
    add_connection_args(conn)

    parser = argparse.ArgumentParser(
        description="小模型预标注流水线（评估 → 建集 → 训练 → ONNX → 预标注）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_assess = sub.add_parser("assess", help="评估数据是否足够训练",
                              parents=[conn])
    p_assess.add_argument("-p", "--project", type=int, required=True, help="项目 ID")
    p_assess.add_argument("--json", action="store_true", help="输出 JSON")

    p_build = sub.add_parser("build", help="构建微数据集", parents=[conn])
    p_build.add_argument("-p", "--project", type=int, required=True, help="项目 ID")
    p_build.add_argument("--out", required=True, help="输出目录（dataset）")
    p_build.add_argument("--images-dir", default=None, help="本地图片映射目录")

    p_train = sub.add_parser("train", help="训练微数据集", parents=[conn])
    p_train.add_argument("--data", required=True, help="数据集目录或 data.yaml")
    p_train.add_argument("--base", default="weights/yolo26n.pt", help="基座权重")
    p_train.add_argument("--epochs", type=int, default=150)
    p_train.add_argument("--imgsz", type=int, default=480)
    p_train.add_argument("--device", default="auto")

    p_predict = sub.add_parser("predict", help="预标注项目未标注任务", parents=[conn])
    p_predict.add_argument("-p", "--project", type=int, required=True)
    p_predict.add_argument("-m", "--model", required=True, help="ONNX / pt 权重")
    p_predict.add_argument("--tag", default=None, help="版本标签（默认取 manifest）")
    p_predict.add_argument("--conf", type=float, default=0.25)
    p_predict.add_argument("--iou", type=float, default=0.7)
    p_predict.add_argument("--imgsz", type=int, default=480)
    p_predict.add_argument("--overwrite", action="store_true", help="覆盖同 tag 预测")
    p_predict.add_argument("--limit", type=int, default=0)
    p_predict.add_argument("--images-dir", default=None)

    p_run = sub.add_parser("run", help="评估→建集→训练→导出→预标注", parents=[conn])
    p_run.add_argument("-p", "--project", type=int, required=True)
    p_run.add_argument("--base", default="weights/yolo26n.pt")
    p_run.add_argument("--epochs", type=int, default=150)
    p_run.add_argument("--imgsz", type=int, default=480)
    p_run.add_argument("--conf", type=float, default=0.25)
    p_run.add_argument("--iou", type=float, default=0.7)
    p_run.add_argument("--images-dir", default=None)
    p_run.add_argument("--skip-train", action="store_true")
    p_run.add_argument("--model", default=None, help="复用模型（配合 --skip-train）")

    args = parser.parse_args()
    client = client_from_args(args)

    def log(msg: str) -> None:
        print(msg)

    def on_event(event: StageEvent) -> None:
        if event.status in ("done", "error"):
            print(f"[{event.stage}] {event.message}")

    try:
        if args.command == "assess":
            a = assess_project(args.project, client=client, log=log)
            if args.json:
                print(json.dumps(_jsonable(a.to_dict()), ensure_ascii=False, indent=2))
            else:
                _print_assessment(a)
            if a.verdict == "insufficient":
                sys.exit(2)

        elif args.command == "build":
            info = build_micro_dataset(args.project, out_dir=args.out,
                                       images_dir=args.images_dir,
                                       client=client, log=log, on_event=on_event)
            print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))

        elif args.command == "train":
            data_path = Path(args.data)
            root = data_path.parent if data_path.name == "data.yaml" else data_path
            classes_file = root / "classes.txt"
            classes = classes_file.read_text().split() if classes_file.is_file() else []
            info = DatasetInfo(root=root, classes=classes, train=0, val=0,
                               skipped=0, failures=[])
            result = train_micro(info, base_weights=args.base, epochs=args.epochs,
                                 imgsz=args.imgsz, device=args.device,
                                 run_dir=str(root.parent), log=log, on_event=on_event)
            onnx = export_onnx(result, imgsz=args.imgsz, log=log, on_event=on_event)
            print(f"best.pt: {result.best_pt}\nonnx: {onnx}")

        elif args.command == "predict":
            stats = prelabel_project(args.project, model=args.model, tag=args.tag,
                                     conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                     overwrite=args.overwrite, limit=args.limit,
                                     images_dir=args.images_dir, client=client,
                                     log=log, on_event=on_event)
            print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))

        elif args.command == "run":
            run_pipeline(args.project, base_weights=args.base, epochs=args.epochs,
                         imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                         images_dir=args.images_dir, skip_train=args.skip_train,
                         reuse_model=args.model, log=log, on_event=on_event)

    except KeyboardInterrupt:
        print("\n⚠ 已中断（已完成阶段保留）")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
