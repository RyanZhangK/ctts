"""AttrDataset / 属性标签解析的单元测试（无需 opencv、无需数据集）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ctts.models.dataset import (
    COLORS,
    PATTERNS,
    parse_attr_line,
    resolve_image_attr,
    AttrDataset,
)


def _fake_loader(path: Path) -> np.ndarray:
    """固定尺寸 BGR 图，避免依赖 opencv。"""
    return np.full((32, 48, 3), 128, dtype=np.uint8)


def test_parse_seven_fields():
    box = parse_attr_line("0 0.5 0.5 0.2 0.1 blue three")
    assert box is not None
    assert box.cls == 0
    assert box.xywh == (0.5, 0.5, 0.2, 0.1)
    assert box.color == COLORS.index("blue")
    assert box.pattern == PATTERNS.index("three")


def test_parse_five_fields_keeps_attr_unknown():
    box = parse_attr_line("0 0.1 0.2 0.3 0.4")
    assert box is not None
    assert (box.color, box.pattern) == (-1, -1)


def test_parse_unknown_color_raises():
    with pytest.raises(ValueError, match="未知颜色名"):
        parse_attr_line("0 0.5 0.5 0.1 0.1 green one")


def test_parse_dash_means_unknown():
    box = parse_attr_line("0 0.5 0.5 0.1 0.1 - -")
    assert box is not None
    assert (box.color, box.pattern) == (-1, -1)


def test_resolve_takes_largest_box():
    small = parse_attr_line("0 0.5 0.5 0.01 0.01 red one")
    large = parse_attr_line("0 0.5 0.5 0.20 0.10 blue sentry")
    assert resolve_image_attr([small, large]) == (COLORS.index("blue"), PATTERNS.index("sentry"))


def test_resolve_empty():
    assert resolve_image_attr([]) == (-1, -1)


def _make_dataset(tmp_path: Path, attr_lines: str = "0 0.5 0.5 0.4 0.3 blue three\n") -> AttrDataset:
    root = tmp_path / "ds"
    (root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (root / "train" / "labels_attr").mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "# test\npath: .\ntrain: train/images\nval: train/images\nnc: 1\nnames:\n  0: stb\n",
        encoding="utf-8",
    )
    (root / "train" / "images" / "a.jpg").write_bytes(b"")
    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
    (root / "train" / "labels_attr" / "a.txt").write_text(attr_lines, encoding="utf-8")
    return AttrDataset(root, "train", image_loader=_fake_loader)


def test_dataset_image_level(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    assert len(ds) == 1
    item = ds[0]
    assert item["bgr"].shape == (3, 32, 48)
    assert item["bgr"].dtype == torch.float32
    assert float(item["bgr"].max()) <= 1.0
    assert int(item["color"]) == COLORS.index("blue")
    assert int(item["pattern"]) == PATTERNS.index("three")


def test_dataset_missing_attr_file_gives_unknown(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    (ds.root / "train" / "images" / "b.jpg").write_bytes(b"")
    (ds.root / "train" / "labels" / "b.txt").write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
    ds = AttrDataset(ds.root, "train", image_loader=_fake_loader)  # 重新扫描样本列表
    item = ds[1]
    assert (int(item["color"]), int(item["pattern"])) == (-1, -1)


def test_dataset_instance_level_returns_boxes(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    ds.level = "instance"
    item = ds[0]
    assert item["boxes"].shape == (1, 4)
    assert (int(item["color"]), int(item["pattern"])) == (-1, -1)


def test_dataset_missing_split_raises(tmp_path: Path):
    ds_root = tmp_path / "ds"
    ds_root.mkdir()
    (ds_root / "data.yaml").write_text("path: .\ntrain: train/images\n", encoding="utf-8")
    with pytest.raises(KeyError, match="val"):
        AttrDataset(ds_root, "val", image_loader=_fake_loader)


def test_dataset_no_images_raises(tmp_path: Path):
    ds_root = tmp_path / "ds"
    (ds_root / "train" / "images").mkdir(parents=True)
    (ds_root / "data.yaml").write_text("train: train/images\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        AttrDataset(ds_root, "train", image_loader=_fake_loader)
