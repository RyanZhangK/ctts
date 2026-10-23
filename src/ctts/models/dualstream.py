"""双流特征解耦实验骨架（PyTorch）。

设计依据：``foray_docs/dual_stream_disentanglement.md`` 的 L2 方案。

结构（自下而上）：

1. **输入拆两路**（与训练、TensorRT 推理共用同一像素通路 ``ctts.models.color``）::

       BGR (B,3,640,640) ──split──► Y    (B,1,640,640)   高分辨率、无颜色
                                     └► CbCr (B,2,320,320)  半分辨率、有颜色

2. **非对称双主干**：Y 主干全宽（图案需要细节），色度主干半宽（颜色只需低频）。
3. **逐级门控融合**（GMU 风格）：``out = y + g * up(c)``，门控偏置初始化为负值，
   保证训练初期色度支路几乎不参与 —— 抑制「色度支路泄题学图案」。
4. **属性解耦头**：颜色 4 类 + 图案 9 类，各自 sigmoid 输出（多标签，互不竞争）。

本模块只管特征通路与属性头；检测框回归仍沿用现有 YOLO 流程（阶段 L0）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "split_yuv",
    "DualStreamConfig",
    "ConvBN",
    "TinyStage",
    "TinyBackbone",
    "GateFuse",
    "AttrHeads",
    "DualStreamNet",
]

# BT.709 full-range 系数（与 ctts.models.color 保持一致）
_KR = 0.2126
_KB = 0.0722
_KG = 1.0 - _KR - _KB


def split_yuv(bgr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按 4:2:0 把 BGR 张量拆成亮度 / 色度两路。

    Args:
        bgr: ``(B, 3, H, W)``，通道顺序 BGR，浮点 [0, 1] 或 [0, 255]。

    Returns:
        ``y``: ``(B, 1, H, W)``；
        ``cbcr``: ``(B, 2, H/2, W/2)``，通道顺序 ``(Cb, Cr)``。

    Note:
        若输入是 [0,255]，调用方需先自行归一化；本函数不做量纲猜测。
    """
    if bgr.ndim != 4 or bgr.shape[1] != 3:
        raise ValueError(f"期望 (B,3,H,W) 的 BGR 张量，收到 {tuple(bgr.shape)}")
    if bgr.shape[-2] % 2 or bgr.shape[-1] % 2:
        raise ValueError("色度 2x2 下采样要求 H/W 为偶数")

    b, g, r = bgr[:, 0:1], bgr[:, 1:2], bgr[:, 2:3]
    y = _KR * r + _KG * g + _KB * b
    cb = 0.5 * (b - y) / (1.0 - _KB) + 0.5
    cr = 0.5 * (r - y) / (1.0 - _KR) + 0.5
    cbcr = F.avg_pool2d(torch.cat([cb, cr], dim=1), kernel_size=2, stride=2)
    return y.clamp(0.0, 1.0), cbcr.clamp(0.0, 1.0)


@dataclass
class DualStreamConfig:
    """实验配置。改结构只动这里，便于消融。"""

    num_colors: int = 4
    num_patterns: int = 9
    y_size: int = 640        # 亮度支路输入边长（与部署 imgsz 对齐）
    c_size: int = 320        # 色度支路输入边长（y_size 的一半）
    y_width: int = 32        # 亮度主干基础宽度
    c_width: int = 16        # 色度主干基础宽度（故意更窄）
    y_depths: tuple[int, ...] = (1, 1, 2, 1)
    c_depths: tuple[int, ...] = (1, 1, 2, 1)
    gate_bias_init: float = -4.0  # sigmoid(-4)≈0.018：初始几乎关闭色度通路
    head_hidden: int = 128
    fusion: str = "gated"    # "gated" | "add" | "concat"（消融用）


class ConvBN(nn.Sequential):
    """Conv2d + BN + SiLU。"""

    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int | None = None):
        p = k // 2 if p is None else p
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class TinyStage(nn.Module):
    """一个下采样阶段：首个卷积 stride=2，其后为同分辨率残差式堆叠。"""

    def __init__(self, cin: int, cout: int, depth: int = 1):
        super().__init__()
        layers: list[nn.Module] = [ConvBN(cin, cout, 3, 2)]
        for _ in range(max(depth - 1, 0)):
            layers.append(ConvBN(cout, cout, 3, 1))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyBackbone(nn.Module):
    """4 个阶段的小主干，返回相对输入 stride=2,4,8,16 的四级特征。"""

    def __init__(self, cin: int, width: int, depths: tuple[int, ...] = (1, 1, 2, 1)):
        super().__init__()
        channels = [width, width * 2, width * 4, width * 8]
        self.stages = nn.ModuleList()
        prev = cin
        for ch, depth in zip(channels, depths):
            self.stages.append(TinyStage(prev, ch, depth))
            prev = ch
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class GateFuse(nn.Module):
    """GMU 风格门控融合：``out = y + g * c_up``，``g`` 逐元素。

    门控权重初始化为 0、偏置为负 —— 训练开始时输出几乎等于亮度特征，
    色度信息只能通过梯度「争取」有限的通道权重，避免色度支路偷学图案。
    """

    def __init__(self, c_y: int, c_c: int, gate_bias_init: float = -4.0):
        super().__init__()
        # 通道投影：色度主干比亮度主干窄，先对齐通道再门控
        self.proj = nn.Conv2d(c_c, c_y, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(c_y + c_c, c_y, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias_init)

    def forward(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.shape[-2:] != y.shape[-2:]:
            c = F.interpolate(c, size=y.shape[-2:], mode="nearest")
        gate = torch.sigmoid(self.gate(torch.cat([y, c], dim=1)))
        return y + gate * self.proj(c)


class AttrHeads(nn.Module):
    """多尺度属性头：四级融合特征各自全局池化后拼接 → MLP → 颜色/图案两套 logits。

    用多级特征而不是只看最深层，原因有二：
    1. 图案是高频细节，浅层特征比最深层更敏感；
    2. 每级融合层都能获得梯度，否则中间级门控会一直保持初始（几乎关闭）状态。
    """

    def __init__(self, chans: list[int], num_colors: int, num_patterns: int, hidden: int = 128):
        super().__init__()
        total = sum(chans)
        self.trunk = nn.Sequential(nn.Linear(total, hidden), nn.SiLU(inplace=True))
        self.color = nn.Linear(hidden, num_colors)
        self.pattern = nn.Linear(hidden, num_patterns)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.cat([f.mean(dim=(-2, -1)) for f in feats], dim=1)
        hidden = self.trunk(pooled)
        return self.color(hidden), self.pattern(hidden)


class DualStreamNet(nn.Module):
    """双流特征解耦网络（无检测框头，只验证特征通路与属性分类）。"""

    def __init__(self, cfg: DualStreamConfig | None = None):
        super().__init__()
        self.cfg = cfg or DualStreamConfig()
        c = self.cfg
        if c.c_size * 2 != c.y_size:
            raise ValueError("当前实现要求 c_size * 2 == y_size")

        self.y_stem = ConvBN(1, c.y_width, 3, 2)
        self.c_stem = ConvBN(2, c.c_width, 3, 2)
        self.y_backbone = TinyBackbone(c.y_width, c.y_width, c.y_depths)
        self.c_backbone = TinyBackbone(c.c_width, c.c_width, c.c_depths)

        n = len(self.y_backbone.out_channels)
        if len(self.c_backbone.out_channels) != n:
            raise ValueError("两条主干的阶段数必须一致")

        self.fuses = nn.ModuleList(
            GateFuse(cy, cc, c.gate_bias_init)
            for cy, cc in zip(self.y_backbone.out_channels, self.c_backbone.out_channels)
        )
        self.attr_heads = AttrHeads(
            self.y_backbone.out_channels, c.num_colors, c.num_patterns, c.head_hidden
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """``x``: ``(B,3,H,W)`` BGR [0,1]。"""
        y, cbcr = split_yuv(x)
        fy = self.y_backbone(self.y_stem(y))
        fc = self.c_backbone(self.c_stem(cbcr))
        fused = [self.fuses[i](a, b) for i, (a, b) in enumerate(zip(fy, fc))]
        color_logits, pattern_logits = self.attr_heads(fused)
        return {
            "color_logits": color_logits,
            "pattern_logits": pattern_logits,
            "feats": fused,
        }
