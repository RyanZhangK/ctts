"""双流特征解耦骨架测试。

本机/CI 未安装 torch 时整文件跳过（``pytest.importorskip``），
保证 ``foray_ml`` 的 CI 在没有 GPU 依赖的环境下仍能验证 numpy 侧通路。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ctts.models.dualstream import (  # noqa: E402
    DualStreamConfig,
    DualStreamNet,
    GateFuse,
    split_yuv,
)


def make_input(batch: int = 2, size: int = 128) -> "torch.Tensor":
    g = torch.Generator().manual_seed(0)
    return torch.rand(batch, 3, size, size, generator=g)


def test_split_yuv_shapes():
    x = make_input()
    y, cbcr = split_yuv(x)
    assert y.shape == (2, 1, 128, 128)
    assert cbcr.shape == (2, 2, 64, 64)
    assert y.min() >= 0 and y.max() <= 1
    assert cbcr.min() >= 0 and cbcr.max() <= 1


def test_split_yuv_rejects_odd_size():
    with pytest.raises(ValueError):
        split_yuv(torch.rand(1, 3, 65, 64))


def test_pure_blue_excites_cb_not_cr():
    x = torch.zeros(1, 3, 8, 8)
    x[:, 0] = 1.0  # B 通道
    _, cbcr = split_yuv(x)
    assert cbcr[:, 0].mean() > 0.5 > cbcr[:, 1].mean()


def test_gate_fuse_starts_near_identity():
    """门控偏置为负 → 初始几乎不放过色度特征（抑制色度支路泄题）。"""
    y = torch.randn(1, 8, 16, 16)
    c = torch.ones(1, 4, 8, 8) * 3.0
    fuse = GateFuse(8, 4, gate_bias_init=-4.0)
    out = fuse(y, c)
    assert out.shape == y.shape
    assert (out - y).abs().mean() < 0.2


def test_dual_stream_forward_shapes():
    cfg = DualStreamConfig(y_size=128, c_size=64, y_width=16, c_width=8)
    net = DualStreamNet(cfg).eval()
    with torch.no_grad():
        out = net(make_input())
    assert out["color_logits"].shape == (2, cfg.num_colors)
    assert out["pattern_logits"].shape == (2, cfg.num_patterns)
    assert len(out["feats"]) == 4
    assert out["feats"][0].shape[-2:] == (32, 32)  # stem stride2 后再 stride2


def test_dual_stream_parameter_budget():
    cfg = DualStreamConfig(y_width=32, c_width=16)
    n = sum(p.numel() for p in DualStreamNet(cfg).parameters())
    assert n < 5_000_000, f"参数量 {n:,} 超出实验预算（色度支路可能过宽）"


def test_dual_stream_backward_runs():
    cfg = DualStreamConfig(y_size=64, c_size=32, y_width=8, c_width=4)
    net = DualStreamNet(cfg)
    out = net(make_input(size=64))
    loss = out["color_logits"].sum() + out["pattern_logits"].sum()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "反向传播未产生任何梯度"
