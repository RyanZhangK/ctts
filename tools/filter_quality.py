#!/usr/bin/env python3
"""
快速剔除低质量图片：拖影（运动模糊）、丢焦（失焦模糊）、失真。

核心原理
--------
1. 拉普拉斯方差 (Laplacian Variance)
   - 对图像求拉普拉斯算子（二阶导数），然后计算方差。
   - 清晰图像边缘锐利 → 拉普拉斯响应强 → 方差大。
   - 模糊图像边缘平滑 → 拉普拉斯响应弱 → 方差小。
   - 对运动模糊和失焦模糊都有效。

2. Sobel 梯度幅值
   - 补充指标，衡量一阶梯度的强度。

3. 灰度直方图熵
   - 检测过曝、欠曝、对比度异常。

阈值参考（经验值，需要根据你的数据集微调）:
  - laplacian_var > 150  : 清晰
  - laplacian_var 50-150 : 轻微模糊（取决于你的容忍度）
  - laplacian_var < 50   : 严重模糊
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 核心质量指标
# ---------------------------------------------------------------------------

def laplacian_variance(img: np.ndarray) -> float:
    """拉普拉斯方差——最经典的无参考模糊检测指标。

    值越大越清晰。典型清晰图片 > 150，严重模糊图片 < 50。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def sobel_sharpness(img: np.ndarray) -> float:
    """Sobel 梯度幅值均值——辅助判断边缘锐度。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(mag))


def histogram_entropy(img: np.ndarray) -> float:
    """灰度直方图的熵——衡量对比度/信息丰富度。

    过低 → 可能过曝或欠曝；过高 → 可能是噪声。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # 忽略零值避免 log(0)
    return float(-np.sum(hist * np.log2(hist)))


def blocking_artifact_score(img: np.ndarray) -> float:
    """检测 JPEG/视频编码块效应。

    计算 8x8 块边界处水平方向像素差的平均值，与块内部对比。
    返回值越大表示块效应越明显（压缩痕迹越重）。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape

    # 水平方向：块边界 (列 7, 15, 23, ...) vs 内部
    boundary_diffs = []
    inner_diffs = []
    for col in range(1, w):
        diff = np.mean(np.abs(gray[:, col] - gray[:, col - 1]))
        if col % 8 == 0:
            boundary_diffs.append(diff)
        else:
            inner_diffs.append(diff)

    if not boundary_diffs or not inner_diffs:
        return 0.0

    boundary_mean = np.mean(boundary_diffs)
    inner_mean = np.mean(inner_diffs)
    if inner_mean == 0:
        return 0.0
    return float(boundary_mean / inner_mean)


def brightness_stats(img: np.ndarray) -> dict:
    """亮度统计：均值、标准差，辅助判断过曝/欠曝。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "brightness_mean": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
    }


# ---------------------------------------------------------------------------
# 综合评估
# ---------------------------------------------------------------------------

def assess_image(image_path: str) -> dict:
    """对单张图片计算所有质量指标。"""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Cannot read: {image_path}"}

    h, w = img.shape[:2]

    metrics = {
        "file": image_path,
        "size": f"{w}x{h}",
        "laplacian_var": laplacian_variance(img),
        "sobel_sharpness": sobel_sharpness(img),
        "hist_entropy": histogram_entropy(img),
        "blocking_score": blocking_artifact_score(img),
        **brightness_stats(img),
    }
    return metrics


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def collect_images(paths: list[str], exts: tuple) -> list[str]:
    """收集所有图片路径。"""
    images = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for ext in exts:
                images.extend(str(x) for x in sorted(p.glob(f"*{ext}")))
        elif p.is_file():
            images.append(str(p))
        else:
            print(f"警告: 路径不存在，跳过: {p}")
    return sorted(set(images))


def main():
    parser = argparse.ArgumentParser(
        description="快速剔除低质量图片（拖影、丢焦、失真）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览模式：只输出 JSON 报告，不移动文件
  python tools/filter_quality.py dataset/images/ -o report.json

  # 实际剔除（移动到 _rejected/ 目录）
  python tools/filter_quality.py dataset/images/ --move --move-dir dataset/_rejected/

  # 自定义阈值
  python tools/filter_quality.py dataset/images/ --min-laplacian 100 --move

  # 结合抽帧使用（管道模式：从 stdin 读取路径列表）
  python tools/filter_quality.py --stdin --move
        """,
    )
    parser.add_argument(
        "paths", nargs="*",
        help="图片文件或目录路径（支持多个）",
    )
    parser.add_argument(
        "--stdin", action="store_true",
        help="从标准输入读取图片路径（每行一个），用于管道模式",
    )
    parser.add_argument(
        "-o", "--output",
        help="将评估结果输出到 JSON 文件",
    )
    parser.add_argument(
        "-e", "--ext",
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="搜索目录时的图片扩展名，逗号分隔（默认: .jpg,.jpeg,.png,.bmp,.tif,.tiff）",
    )

    # 阈值
    thresh = parser.add_argument_group("阈值设置")
    thresh.add_argument(
        "--min-laplacian", type=float, default=50.0,
        help="拉普拉斯方差下限（默认: 50）。低于此值判定为模糊。"
             "建议: 清晰 >150, 轻微模糊 50-150, 严重模糊 <50",
    )
    thresh.add_argument(
        "--min-hist-entropy", type=float, default=3.0,
        help="直方图熵下限（默认: 3.0）。低于此值可能过曝/欠曝。"
             "典型范围 3-7，自然图像一般 > 4",
    )
    thresh.add_argument(
        "--max-blocking", type=float, default=1.5,
        help="块效应分数上限（默认: 1.5）。高于此值有较明显的压缩痕迹。"
             "无压缩 ≈ 1.0，轻度压缩 1.0-1.3，重度压缩 > 1.5",
    )

    # 操作模式
    action = parser.add_argument_group("操作模式")
    action.add_argument(
        "--move", action="store_true",
        help="实际移动不合格图片到 --move-dir 目录",
    )
    action.add_argument(
        "--move-dir", default="_rejected",
        help="不合格图片移动目标目录（默认: _rejected）",
    )
    action.add_argument(
        "--copy-to", default=None,
        help="将合格图片复制到指定目录（不移动原文件）",
    )

    args = parser.parse_args()

    # ---- 收集图片路径 ----
    if args.stdin:
        paths = [line.strip() for line in sys.stdin if line.strip()]
    elif args.paths:
        exts = tuple(e.strip() for e in args.ext.split(","))
        paths = collect_images(args.paths, exts)
    else:
        parser.error("请指定路径，或使用 --stdin 从管道读取")

    if not paths:
        print("未找到任何图片")
        return

    print(f"找到 {len(paths)} 张图片，开始评估...")

    # ---- 逐张评估 ----
    results = []
    rejected = []

    for i, p in enumerate(paths):
        metrics = assess_image(p)
        results.append(metrics)

        if "error" in metrics:
            print(f"  [{i+1}/{len(paths)}] {p}  → 读取失败")
            rejected.append(p)
            continue

        # 判定逻辑
        reasons = []
        lv = metrics["laplacian_var"]
        he = metrics["hist_entropy"]
        bs = metrics["blocking_score"]

        if lv < args.min_laplacian:
            reasons.append(f"模糊 (lap_var={lv:.1f} < {args.min_laplacian})")
        if he < args.min_hist_entropy:
            reasons.append(f"对比度异常 (entropy={he:.2f} < {args.min_hist_entropy})")
        if bs > args.max_blocking:
            reasons.append(f"压缩失真 (blocking={bs:.3f} > {args.max_blocking})")

        if reasons:
            rejected.append(p)
            print(f"  [{i+1}/{len(paths)}] ❌ {p}  → {', '.join(reasons)}")
        else:
            print(f"  [{i+1}/{len(paths)}] ✅ {p}  → lap={lv:.1f}, ent={he:.2f}, blk={bs:.3f}")

    # ---- 汇总 ----
    n_total = len(paths)
    n_rejected = len(rejected)
    n_pass = n_total - n_rejected

    print(f"\n{'='*60}")
    print(f"评估完成: 总计 {n_total}  合格 {n_pass}  不合格 {n_rejected}")
    print(f"阈值: laplacian_var >= {args.min_laplacian}, "
          f"hist_entropy >= {args.min_hist_entropy}, "
          f"blocking <= {args.max_blocking}")

    # ---- 输出报告 ----
    if args.output:
        report = {
            "thresholds": {
                "min_laplacian_var": args.min_laplacian,
                "min_hist_entropy": args.min_hist_entropy,
                "max_blocking_score": args.max_blocking,
            },
            "summary": {
                "total": n_total,
                "pass": n_pass,
                "rejected": n_rejected,
            },
            "rejected": rejected,
            "details": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {args.output}")

    # ---- 执行文件操作 ----
    if args.move and rejected:
        move_dir = Path(args.move_dir)
        move_dir.mkdir(parents=True, exist_ok=True)
        for src in rejected:
            dst = move_dir / Path(src).name
            # 处理重名
            if dst.exists():
                stem = Path(src).stem
                suffix = Path(src).suffix
                dst = move_dir / f"{stem}_dup{suffix}"
            os.rename(src, dst)
        print(f"{len(rejected)} 张图片已移动到: {move_dir.resolve()}")

    if args.copy_to and n_pass > 0:
        copy_dir = Path(args.copy_to)
        copy_dir.mkdir(parents=True, exist_ok=True)
        pass_set = set(rejected)
        copied = 0
        for src in paths:
            if src not in pass_set:
                import shutil
                shutil.copy2(src, copy_dir / Path(src).name)
                copied += 1
        print(f"{copied} 张合格图片已复制到: {copy_dir.resolve()}")

    if not args.move and not args.copy_to and not args.output:
        print("\n提示: 这是预览模式。加上 --move 执行剔除，或 -o report.json 保存报告。")


if __name__ == "__main__":
    main()
