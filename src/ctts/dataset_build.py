#!/usr/bin/env python3
"""从视频批量抽帧，内置质量过滤（模糊 / 压缩失真 / 过曝欠曝）。

支持一次输入多个视频，每个视频的帧输出到独立子目录。

质量过滤指标：
  - 拉普拉斯方差 (Laplacian Variance)：检测模糊
  - 块效应分数 (Blocking Score)：检测压缩失真
  - 直方图熵 (Histogram Entropy)：检测过曝/欠曝
  - 亮度 (Brightness)：检测极端明暗

示例:
  # 单视频 + 无过滤
  python dataset_build.py video.mp4 output/ -i 0.5

  # 多视频 + 质量过滤
  python dataset_build.py a.mp4 b.mp4 output/ -i 1.0 --min-laplacian 50

  # 完整过滤参数
  python dataset_build.py *.mp4 output/ -i 0.5 --min-laplacian 100 --min-entropy 3.5
"""

import argparse
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 帧质量指标
# ---------------------------------------------------------------------------

def frame_laplacian_var(frame: np.ndarray) -> float:
    """拉普拉斯方差——无参考模糊检测。

    清晰 > 150  |  轻微模糊 50-150  |  严重模糊 < 50
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_blocking_score(frame: np.ndarray) -> float:
    """块效应分数——检测视频编码压缩痕迹。

    无压缩 ≈ 1.0  |  轻度压缩 1.0-1.3  |  重度压缩 > 1.5
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    _h, w = gray.shape

    boundary_diffs = []
    inner_diffs = []
    for col in range(1, w):
        diff = float(np.mean(np.abs(gray[:, col] - gray[:, col - 1])))
        if col % 8 == 0:
            boundary_diffs.append(diff)
        else:
            inner_diffs.append(diff)

    if not boundary_diffs or not inner_diffs:
        return 0.0
    inner_mean = float(np.mean(inner_diffs))
    if inner_mean == 0:
        return 0.0
    return float(np.mean(boundary_diffs)) / inner_mean


def frame_histogram_entropy(frame: np.ndarray) -> float:
    """灰度直方图熵——检测过曝/欠曝/对比度异常。

    自然图像一般 > 4，过低表示可能过曝或欠曝。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def frame_brightness(frame: np.ndarray) -> float:
    """亮度均值——辅助判断过曝(>200)或欠曝(<30)。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


# ---------------------------------------------------------------------------
# 综合质量判定
# ---------------------------------------------------------------------------

def is_frame_quality_ok(
    frame: np.ndarray,
    min_laplacian: float = 50.0,
    max_blocking: float = 1.5,
    min_entropy: float = 3.0,
    min_brightness: float = 10.0,
    max_brightness: float = 245.0,
) -> tuple[bool, dict]:
    """综合质量检查。

    Returns:
        (is_ok, metrics_dict)
    """
    lap = frame_laplacian_var(frame)
    blk = frame_blocking_score(frame)
    ent = frame_histogram_entropy(frame)
    brt = frame_brightness(frame)

    ok = (
        lap >= min_laplacian
        and blk <= max_blocking
        and ent >= min_entropy
        and brt >= min_brightness
        and brt <= max_brightness
    )
    return ok, {
        "laplacian_var": lap,
        "blocking_score": blk,
        "hist_entropy": ent,
        "brightness": brt,
    }


# ---------------------------------------------------------------------------
# 抽帧主逻辑
# ---------------------------------------------------------------------------

def collect_videos(paths: list[str]) -> list[str]:
    """展开目录和 glob，收集所有视频文件。"""
    exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm")
    result = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in exts:
                result.extend(str(x) for x in sorted(path.glob(f"*{ext}")))
                result.extend(str(x) for x in sorted(path.glob(f"*{ext.upper()}")))
        elif path.is_file():
            result.append(str(path))
        else:
            print(f"⚠ 跳过（不存在）: {p}")
    return sorted(set(result))


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_sec: float = 1.0,
    min_laplacian: float | None = None,
    max_blocking: float | None = None,
    min_entropy: float | None = None,
    min_brightness: float | None = None,
    max_brightness: float | None = None,
):
    """从单个视频按时间间隔抽取帧，可选质量过滤。

    Returns:
        (saved_count, skipped_count, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * interval_sec))

    # 多视频时每个视频独立子目录
    video_name = Path(video_path).stem
    video_out = os.path.join(output_dir, video_name)
    os.makedirs(video_out, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    filter_enabled = min_laplacian is not None

    saved = 0
    skipped = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if filter_enabled:
                ok, m = is_frame_quality_ok(
                    frame,
                    min_laplacian=min_laplacian or 50,
                    max_blocking=max_blocking or 1.5,
                    min_entropy=min_entropy or 3.0,
                    min_brightness=min_brightness or 10,
                    max_brightness=max_brightness or 245,
                )
                if not ok:
                    skipped += 1
                    reasons = []
                    if m["laplacian_var"] < (min_laplacian or 50):
                        reasons.append(f"模糊({m['laplacian_var']:.0f})")
                    if m["blocking_score"] > (max_blocking or 1.5):
                        reasons.append(f"压缩({m['blocking_score']:.2f})")
                    if m["hist_entropy"] < (min_entropy or 3.0):
                        reasons.append(f"对比度({m['hist_entropy']:.2f})")
                    if m["brightness"] < (min_brightness or 10):
                        reasons.append(f"欠曝({m['brightness']:.0f})")
                    if m["brightness"] > (max_brightness or 245):
                        reasons.append(f"过曝({m['brightness']:.0f})")
                    print(f"  ⏭  #{frame_idx:06d}  {', '.join(reasons)}")
                    frame_idx += 1
                    continue

            filename = os.path.join(video_out, video_hash + f"_frame_{frame_idx:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        frame_idx += 1

    cap.release()

    filter_info = ""
    if filter_enabled:
        filter_info = (f", 过滤: 跳过 {skipped} 帧 "
                       f"(laplacian>={min_laplacian}, blocking<={max_blocking}, "
                       f"entropy>={min_entropy}, brightness={min_brightness}-{max_brightness})")
    print(f"  {video_name}: {saved}/{total_frames} 保存{filter_info}")

    return saved, skipped, total_frames


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="视频批量抽帧，内置质量过滤（模糊/压缩/过曝欠曝检测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dataset_build.py a.mp4 b.mp4 output/
  python dataset_build.py *.mp4 output/ -i 0.5 --min-laplacian 100
  python dataset_build.py videos/ output/ --min-laplacian 50 --min-entropy 3.5
        """,
    )
    parser.add_argument(
        "videos", nargs="+",
        help="视频文件路径（支持多个，支持目录）",
    )
    parser.add_argument("output", help="输出目录")
    parser.add_argument(
        "-i", "--interval", type=float, default=1.0,
        help="抽帧时间间隔（秒），默认: 1.0",
    )

    quality = parser.add_argument_group("质量过滤（不指定则不过滤）")
    quality.add_argument("--min-laplacian", type=float, default=None,
        help="拉普拉斯方差下限（建议 50-150），不指定则不过滤")
    quality.add_argument("--max-blocking", type=float, default=1.5,
        help="块效应分数上限（默认: 1.5）")
    quality.add_argument("--min-entropy", type=float, default=3.0,
        help="直方图熵下限（默认: 3.0）")
    quality.add_argument("--min-brightness", type=float, default=10,
        help="亮度下限，低于此值判为欠曝（默认: 10）")
    quality.add_argument("--max-brightness", type=float, default=245,
        help="亮度上限，高于此值判为过曝（默认: 245）")

    args = parser.parse_args()

    # 收集所有视频
    videos = collect_videos(args.videos)
    if not videos:
        print("❌ 未找到任何视频文件")
        return

    print(f"找到 {len(videos)} 个视频")

    total_saved = 0
    total_skipped = 0
    for i, v in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {v}")
        saved, skipped, _ = extract_frames(
            v, args.output, args.interval,
            min_laplacian=args.min_laplacian,
            max_blocking=args.max_blocking,
            min_entropy=args.min_entropy,
            min_brightness=args.min_brightness,
            max_brightness=args.max_brightness,
        )
        total_saved += saved
        total_skipped += skipped

    print(f"\n{'='*50}")
    print(f"全部完成: {len(videos)} 个视频, 保存 {total_saved} 帧")
    if args.min_laplacian is not None:
        print(f"质量过滤: 跳过 {total_skipped} 帧")


if __name__ == "__main__":
    main()
