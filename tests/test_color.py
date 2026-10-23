"""YUV 4:2:0 数据通路测试（纯 numpy，无需 GPU / torch）。"""

from __future__ import annotations

import numpy as np
import pytest

from ctts.models import color


def solid_bgr(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[...] = bgr
    return img


def test_split_420_shapes_and_range():
    img = solid_bgr(64, 96, (255, 0, 0))  # 纯蓝（BGR）
    y, cbcr = color.split_420(img)
    assert y.shape == (64, 96)
    assert cbcr.shape == (32, 48, 2)
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0
    assert 0.0 <= float(cbcr.min()) and float(cbcr.max()) <= 1.0


def test_blue_and_red_land_on_opposite_chroma_sides():
    """蓝色 Cb 高、红色 Cr 高 —— 灯条颜色正好落在色度支路。"""
    y_b, cbcr_b = color.bgr_to_ycbcr(solid_bgr(8, 8, (255, 0, 0)))
    y_r, cbcr_r = color.bgr_to_ycbcr(solid_bgr(8, 8, (0, 0, 255)))
    cb_b, cr_b = cbcr_b[..., 0].mean(), cbcr_b[..., 1].mean()
    cb_r, cr_r = cbcr_r[..., 0].mean(), cbcr_r[..., 1].mean()
    assert cb_b > 0.5 > cb_r
    assert cr_r > 0.5 > cr_b
    # 颜色信息主要落在 Cb/Cr 上：纯蓝与纯红的 Y 都不高（BT.709 下蓝更暗）
    assert y_b.mean() < 0.2 and y_r.mean() < 0.3


def test_chroma_downsample_is_block_mean():
    cbcr = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    out = color.chroma_downsample(cbcr, 2)
    assert out.shape == (2, 2, 1)
    assert out[0, 0, 0] == pytest.approx(cbcr[:2, :2, 0].mean())


def test_chroma_downsample_rejects_indivisible():
    with pytest.raises(ValueError):
        color.chroma_downsample(np.zeros((5, 6, 2), dtype=np.float32), 2)


def test_420_roundtrip_error_is_small_on_smooth_image():
    """平坦色块经 4:2:0 往返后误差应很小（图案高频损失另说）。"""
    img = solid_bgr(32, 32, (40, 180, 220))
    y, cbcr = color.split_420(img)
    restored = color.merge_420(y, cbcr)
    err = np.abs(restored - img.astype(np.float32) / 255.0).mean()
    assert err < 0.01


def test_swap_rb_is_involution_and_preserves_shape():
    img = solid_bgr(8, 8, (10, 20, 30))
    swapped = color.swap_rb(img)
    assert swapped[0, 0, 0] == 30 and swapped[0, 0, 2] == 10
    np.testing.assert_array_equal(color.swap_rb(swapped), img)


def test_swap_rb_keeps_luma_for_gray():
    """灰色（R≈B）下 R↔B 交换亮度不变 —— 反事实增强不改变图案区域的中性像素。

    注意：彩色灯条（纯红/纯蓝）交换后亮度**会变**（BT.709 中 R 与 B 权重相差约 3 倍），
    这是该增强的已知性质，需在实验记录中说明，不要假设「亮度恒定」。
    """
    img = solid_bgr(16, 16, (120, 120, 120))
    y0, _ = color.bgr_to_ycbcr(img)
    y1, _ = color.bgr_to_ycbcr(color.swap_rb(img))
    assert abs(float(y0.mean()) - float(y1.mean())) < 0.005
