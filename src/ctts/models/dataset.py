"""属性检测数据集（方案 A：灯条颜色与图案两套标签）。

**标签约定**

- `labels/`        —— 标准 YOLO 5 字段（`cls cx cy w h`），供 ultralytics 直接使用，保持兼容；
- `labels_attr/`   —— 与图片同名的属性文件，每行 `cls cx cy w h <color> <pattern>`，
  颜色/图案写**类别名**（如 `blue` / `three`），未知写 `-`。

属性放在独立目录而不是往 YOLO 行里追加字段，原因：ultralytics 会拒绝多余字段，
而属性训练用的是本仓自定义 loader。两套文件同源生成（标注脚本或转换脚本），
不改变原有标注产物的可用性。

**图像级与实例级**

`AttrDataset(level="image")` 假设每张图对应一个主目标（裁剪后的 patch 集 / ROI 集），
取面积最大框的属性作为图像属性。一张图内有多个不同属性目标时，会在 `conflicts`
里累计次数并在训练前打印 —— 这提示该数据集应当改用实例级标签或先做裁剪。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "COLORS",
    "PATTERNS",
    "AttrBox",
    "parse_attr_line",
    "load_attr_file",
    "resolve_image_attr",
    "AttrDataset",
]

# 与 sp_vision_25/tasks/auto_aim/armor.hpp 的枚举保持一致
COLORS = ["red", "blue", "extinguish", "purple"]
PATTERNS = ["one", "two", "three", "four", "five", "sentry", "outpost", "base", "not_armor"]

_COLOR_INDEX = {name: i for i, name in enumerate(COLORS)}
_PATTERN_INDEX = {name: i for i, name in enumerate(PATTERNS)}


@dataclass
class AttrBox:
    """一个检测框及其属性。属性未知时为 -1。"""

    cls: int
    xywh: tuple[float, float, float, float]
    color: int = -1
    pattern: int = -1

    @property
    def area(self) -> float:
        return self.xywh[2] * self.xywh[3]


def parse_attr_line(line: str) -> AttrBox | None:
    """解析一行 ``cls cx cy w h color pattern``；空行/注释返回 None。"""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"属性标签至少需要 5 个字段，收到 {len(parts)} 个：{line!r}")
    cls = int(parts[0])
    xywh = tuple(float(v) for v in parts[1:5])  # type: ignore[assignment]
    color = _COLOR_INDEX.get(parts[5], -1) if len(parts) > 5 else -1
    pattern = _PATTERN_INDEX.get(parts[6], -1) if len(parts) > 6 else -1
    if len(parts) > 5 and parts[5] not in {"-", ""} and color < 0:
        raise ValueError(f"未知颜色名 {parts[5]!r}，应在 {COLORS} 中")
    if len(parts) > 6 and parts[6] not in {"-", ""} and pattern < 0:
        raise ValueError(f"未知图案名 {parts[6]!r}，应在 {PATTERNS} 中")
    return AttrBox(cls, xywh, color, pattern)


def load_attr_file(path: str | Path) -> list[AttrBox]:
    """读取属性标签文件。文件不存在时返回空列表（调用方可据此判定该图未标注）。"""
    p = Path(path)
    if not p.is_file():
        return []
    boxes: list[AttrBox] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        box = parse_attr_line(line)
        if box is not None:
            boxes.append(box)
    return boxes


def resolve_image_attr(boxes: list[AttrBox]) -> tuple[int, int]:
    """图像级属性 = 面积最大框的属性。无有效属性时返回 (-1, -1)。"""
    if not boxes:
        return -1, -1
    main = max(boxes, key=lambda b: b.area)
    return main.color, main.pattern


def _read_yaml(p: Path) -> dict:
    """读取数据集配置。

    优先用 pyyaml；未安装时回退到内置极简解析（支持 ``key: value``、行尾注释、
    以及 ultralytics 生成的 data.yaml 常见形态）。这样数据集 loader 不额外依赖 pyyaml。
    """
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        pass
    cfg: dict = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or ":" not in line or line.startswith((" ", "-")):
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("\"'")
        cfg[key.strip()] = value or None
    return cfg


class AttrDataset(Dataset):
    """属性数据集。

    Args:
        root: 数据集根目录（其下含 ``data.yaml``）。
        split: ``train`` 或 ``val``（对应 data.yaml 的同名字段）。
        image_loader: ``(Path) -> np.ndarray``（HWC、uint8、BGR）。默认 opencv；
            测试可注入自定义实现，避免依赖 cv2。
        level: ``image``（每图一个主属性）或 ``instance``（返回每框属性）。
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_loader=None,
        level: str = "image",
    ):
        self.root = Path(root)
        if level not in {"image", "instance"}:
            raise ValueError("level 只能是 'image' 或 'instance'")
        self.level = level

        cfg = _read_yaml(self.root / "data.yaml")
        rel = cfg.get(split)
        if rel is None:
            raise KeyError(f"data.yaml 缺少 '{split}' 字段")
        # data.yaml 里 train/val 指向图片目录；标签目录与之同级
        self.images_dir = (self.root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        self.labels_dir = Path(str(self.images_dir).replace("images", "labels"))
        self.attr_dir = Path(str(self.images_dir).replace("images", "labels_attr"))

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.samples = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in exts
        ) if self.images_dir.is_dir() else []
        if not self.samples:
            raise FileNotFoundError(f"{self.images_dir} 下没有图片（或目录不存在）")

        self._loader = image_loader or _default_loader
        self.conflicts: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.samples[idx]
        boxes = load_attr_file(self.attr_dir / f"{img_path.stem}.txt")

        if self.level == "image":
            color, pattern = resolve_image_attr(boxes)
            if len(boxes) > 2:
                colors = {b.color for b in boxes if b.color >= 0}
                patterns = {b.pattern for b in boxes if b.pattern >= 0}
                if len(colors) > 1 or len(patterns) > 1:
                    self.conflicts[img_path.name] = self.conflicts.get(img_path.name, 0) + 1
            box_t = torch.zeros((0, 4), dtype=torch.float32)
        else:
            color, pattern = -1, -1
            box_t = (
                torch.tensor([b.xywh for b in boxes], dtype=torch.float32)
                if boxes
                else torch.zeros((0, 4), dtype=torch.float32)
            )

        bgr = self._loader(img_path)
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"{img_path} 不是 HWC 三通道图像：shape={bgr.shape}")
        img = torch.from_numpy(np.ascontiguousarray(bgr.transpose(2, 0, 1))).float().div_(255.0)

        return {
            "bgr": img,                       # (3,H,W) float[0,1]
            "color": torch.tensor(color, dtype=torch.long),
            "pattern": torch.tensor(pattern, dtype=torch.long),
            "boxes": box_t,                   # (N,4) 归一化 xywh（instance 级才非空）
            "path": str(img_path),
        }


def _default_loader(path: Path) -> np.ndarray:
    import cv2  # 延迟导入：只有真正读图时才需要 opencv

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return img
