"""YUV 4:2:0 数据通路工具（纯 numpy，不依赖 torch）。

设计依据：``foray_docs/dual_stream_disentanglement.md`` §3/§4

双流假设：装甲板的「灯条颜色」是低频信息，「图案形状」是高频信息。
把视频流拆成两路分别喂给两个主干，可以避免单一主干在训练中被低频颜色
吃掉高频图案的梯度：

* ``Y``   亮度（保留高分辨率，图案轮廓）      → ``(B, 1, H, W)``
* ``CbCr`` 色度（半分辨率，灯条颜色）        → ``(B, 2, H/2, W/2)``

本模块只做像素域变换，训练/推理共用同一套实现，保证零域差。
数值约定：输入 ``BGR`` 为 uint8 或 [0, 1] 浮点；内部按 BT.709 **full range**
计算，输出 ``Y / Cb / Cr`` 均在 [0, 1]。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bgr_to_ycbcr",
    "ycbcr_to_bgr",
    "chroma_downsample",
    "chroma_upsample",
    "split_420",
    "merge_420",
    "swap_rb",
]

# BT.709 full-range 系数
_KR = 0.2126
_KB = 0.0722
_KG = 1.0 - _KR - _KB


def _as_float01(bgr: np.ndarray) -> np.ndarray:
    """uint8 / float 统一到 [0, 1] float32，形状 (..., 3)，通道顺序 BGR。"""
    arr = np.asarray(bgr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32, copy=False)


def bgr_to_ycbcr(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BGR → (Y, CbCr)。

    Args:
        bgr: ``(H, W, 3)`` BGR，uint8 或 [0,1] 浮点。

    Returns:
        ``y``: ``(H, W)`` 浮点 [0,1]；
        ``cbcr``: ``(H, W, 2)`` 浮点，通道顺序 ``(Cb, Cr)``，均居中于 0.5。
    """
    rgb = _as_float01(bgr)[..., ::-1]  # BGR → RGB
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = _KR * r + _KG * g + _KB * b
    cb = 0.5 * (b - y) / (1.0 - _KB) + 0.5
    cr = 0.5 * (r - y) / (1.0 - _KR) + 0.5
    return np.clip(y, 0.0, 1.0), np.stack([np.clip(cb, 0.0, 1.0), np.clip(cr, 0.0, 1.0)], axis=-1)


def ycbcr_to_bgr(y: np.ndarray, cbcr: np.ndarray) -> np.ndarray:
    """(Y, CbCr) → BGR，``bgr_to_ycbcr`` 的逆变换（用于可视化与一致性校验）。"""
    y = np.asarray(y, dtype=np.float32)
    cb = np.asarray(cbcr, dtype=np.float32)[..., 0] - 0.5
    cr = np.asarray(cbcr, dtype=np.float32)[..., 1] - 0.5
    r = y + 2.0 * (1.0 - _KR) * cr
    b = y + 2.0 * (1.0 - _KB) * cb
    g = (y - _KR * r - _KB * b) / _KG
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)[..., ::-1]  # RGB → BGR


def chroma_downsample(cbcr: np.ndarray, factor: int = 2) -> np.ndarray:
    """色度下采样：``factor x factor`` 块平均（等价于 4:2:0 的色度子采样）。

    Args:
        cbcr: ``(H, W, 2)`` 或 ``(H, W)``。
        factor: 下采样倍数，需整除 H/W。
    """
    if factor <= 1:
        return np.asarray(cbcr)
    h, w = cbcr.shape[:2]
    if h % factor or w % factor:
        raise ValueError(f"色度尺寸 ({h},{w}) 不能被下采样倍数 {factor} 整除")
    if cbcr.ndim == 2:
        return cbcr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return cbcr.reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3))


def chroma_upsample(cbcr: np.ndarray, factor: int = 2) -> np.ndarray:
    """色度上采样（最近邻，与 TensorRT 预处理保持同一种插值）。"""
    if factor <= 1:
        return np.asarray(cbcr)
    return np.repeat(np.repeat(cbcr, factor, axis=0), factor, axis=1)


def split_420(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BGR → ``(Y, CbCr@420)``，训练与推理共用的一条通路。

    Returns:
        ``y``: ``(H, W)``；``cbcr``: ``(H/2, W/2, 2)``。
    """
    y, cbcr = bgr_to_ycbcr(bgr)
    return y, chroma_downsample(cbcr, 2)


def merge_420(y: np.ndarray, cbcr420: np.ndarray) -> np.ndarray:
    """``(Y, CbCr@420)`` → BGR，用于检查 4:2:0 往返损失（不改变模型输入）。"""
    return ycbcr_to_bgr(y, chroma_upsample(cbcr420, 2))


def swap_rb(bgr: np.ndarray) -> np.ndarray:
    """R↔B 通道交换 —— 反事实颜色增强（见调研报告 §4.3）。

    蓝灯条 ↔ 红灯条可互换，而图案形状不变，用来迫使主干不把「颜色」当作
    图案的捷径。注意：不能用 HSV 全局 hue 旋转（蓝 +120° 会变绿），
    也不能作用于过曝的白色灯条（其 R/G/B 已饱和，交换无效）。
    """
    arr = np.asarray(bgr)
    out = arr.copy()
    out[..., 0] = arr[..., 2]
    out[..., 2] = arr[..., 0]
    return out
