#!/usr/bin/env python3
"""将数据集按比例拆分为 train / val 子集，并自动生成 data.yaml。

支持从 classes.txt 或 Label Studio 的 notes.json 中读取类别名称。
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def load_classes(dataset_dir: Path) -> list[str]:
    """从 dataset_dir 中加载类别名，优先读取 classes.txt，其次 notes.json。"""
    # 1) classes.txt（每行一个类名）
    classes_txt = dataset_dir / "classes.txt"
    if classes_txt.exists():
        lines = [ln.strip() for ln in classes_txt.read_text().splitlines() if ln.strip()]
        if lines:
            return lines

    # 2) Label Studio 导出的 notes.json（categories 数组）
    notes_json = dataset_dir / "notes.json"
    if notes_json.exists():
        data = json.loads(notes_json.read_text())
        cats = data.get("categories", [])
        if cats:
            return [c["name"] for c in sorted(cats, key=lambda x: x["id"])]

    # 3) 兜底：简单数字编号
    raise FileNotFoundError(
        f"在 {dataset_dir} 中未找到 classes.txt 或 notes.json，无法确定类别。\n"
        "请在该目录下手动创建 classes.txt（每行一个类名）。"
    )


def split_dataset(
    dataset_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    dry_run: bool = False,
):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images 目录不存在: {images_dir}")

    # 收集所有图片
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    images = []
    for ext in exts:
        images.extend(sorted(images_dir.glob(ext)))
    images = sorted(set(images))  # 去重排序

    if not images:
        raise FileNotFoundError(f"在 {images_dir} 中未找到任何图片 ({', '.join(exts)})")

    print(f"找到 {len(images)} 张图片")

    # 打乱并划分
    random.seed(seed)
    random.shuffle(images)

    n_val = max(1, int(len(images) * val_ratio))
    val_images = set(images[:n_val])
    train_images = set(images[n_val:])

    print(f"train: {len(train_images)}   val: {len(val_images)}   (val_ratio={val_ratio}, seed={seed})")

    if dry_run:
        print("[dry-run] 仅预览，不实际复制文件")
        return

    # 创建目录结构
    for split_name in ("train", "val"):
        (dataset_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    # 复制文件
    for split_name, subset in [("train", train_images), ("val", val_images)]:
        for img_path in subset:
            dest_img = dataset_dir / split_name / "images" / img_path.name
            shutil.copy2(img_path, dest_img)

            # 同名标签文件（.txt）
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                dest_label = dataset_dir / split_name / "labels" / label_path.name
                shutil.copy2(label_path, dest_label)
        print(f"  {split_name}: {len(subset)} 张图片已复制")

    # 生成 data.yaml
    classes = load_classes(dataset_dir)
    yaml_path = dataset_dir / "data.yaml"
    names_block = "\n".join(f'  {i}: "{name}"' for i, name in enumerate(classes))
    yaml_content = (
        f"# 由 split_dataset.py 自动生成\n"
        f"path: {dataset_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(classes)}\n"
        f"names:\n"
        f"{names_block}\n"
    )
    yaml_path.write_text(yaml_content)
    print(f"\ndata.yaml 已生成: {yaml_path}")
    print(f"类别数: {len(classes)} → {classes}")


def main():
    parser = argparse.ArgumentParser(
        description="将数据集按比例拆分为 train/val 子集并生成 data.yaml"
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="dataset",
        help="数据集根目录（包含 images/ 和 labels/ 子目录），默认: dataset",
    )
    parser.add_argument(
        "-v", "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例 (0~1)，默认: 0.2",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="随机种子，默认: 42",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="仅预览划分结果，不实际复制文件",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    split_dataset(
        dataset_dir=dataset_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
