#!/usr/bin/env python3
"""Extract frames from video at a fixed time interval, with optional quality filter.

Quality filter 会跳过拖影（运动模糊）、丢焦（失焦模糊）和严重压缩失真的帧。
核心指标：拉普拉斯方差 (Laplacian Variance)——值越大越清晰。
"""

import argparse
import hashlib
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 帧质量指标
# ---------------------------------------------------------------------------

def frame_laplacian_var(frame: np.ndarray) -> float:
    """拉普拉斯方差——最经典的无参考模糊检测指标。

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


def is_frame_quality_ok(
    frame: np.ndarray,
    min_laplacian: float = 50.0,
    max_blocking: float = 1.5,
) -> tuple[bool, dict]:
    """检查单帧质量是否合格。

    Returns:
        (is_ok, metrics_dict)
    """
    lap = frame_laplacian_var(frame)
    blk = frame_blocking_score(frame)

    ok = (lap >= min_laplacian) and (blk <= max_blocking)
    return ok, {"laplacian_var": lap, "blocking_score": blk}


# ---------------------------------------------------------------------------
# 抽帧主逻辑
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    output_dir: str,
    interval_sec: float = 1.0,
    min_laplacian: float | None = None,
    max_blocking: float | None = None,
):
    """从视频按时间间隔抽取帧，可选质量过滤。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * interval_sec))

    os.makedirs(output_dir, exist_ok=True)

    # 基于视频路径生成短 hash，防止多个视频的帧互相覆盖
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
            # 质量过滤
            if filter_enabled:
                ok, metrics = is_frame_quality_ok(frame, min_laplacian, max_blocking or 1.5)
                if not ok:
                    skipped += 1
                    print(f"  ⏭  frame {frame_idx:06d} 跳过 "
                          f"(lap={metrics['laplacian_var']:.1f}, blk={metrics['blocking_score']:.3f})")
                    frame_idx += 1
                    continue

            filename = os.path.join(output_dir, video_hash + f"_frame_{frame_idx:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        frame_idx += 1

    cap.release()

    # 汇总
    attempted = saved + skipped
    filter_info = ""
    if filter_enabled:
        filter_info = (f", 质量过滤: 跳过 {skipped} 帧 "
                       f"(laplacian >= {min_laplacian}, blocking <= {max_blocking})")
    print(f"Done: {saved} frames saved from {total_frames} total, "
          f"{attempted} sampled (interval={interval_sec}s, fps={fps:.1f}){filter_info}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video at fixed time intervals, with optional quality filter",
    )
    parser.add_argument("video", help="Input video file path")
    parser.add_argument("output", help="Output directory for frames")
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Time interval between frames in seconds (default: 1.0)",
    )

    # 质量过滤参数
    quality = parser.add_argument_group("质量过滤（可选）")
    quality.add_argument(
        "--min-laplacian",
        type=float,
        default=None,
        help="拉普拉斯方差下限，低于此值判定为模糊。建议值: 50-150。"
             "不指定则不启用质量过滤",
    )
    quality.add_argument(
        "--max-blocking",
        type=float,
        default=1.5,
        help="块效应分数上限，高于此值判定为压缩失真 (default: 1.5)",
    )

    args = parser.parse_args()

    extract_frames(
        args.video,
        args.output,
        args.interval,
        min_laplacian=args.min_laplacian,
        max_blocking=args.max_blocking,
    )


if __name__ == "__main__":
    main()
