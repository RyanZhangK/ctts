#!/usr/bin/env python3
"""双流特征解耦实验入口。

两种用法：

1. **烟雾测试**（无需数据集、无需 GPU，验证结构 + 反向传播能否跑通）::

       python3 experiments/dualstream/train_dualstream.py --dry-run

2. **真实训练**（尚未实现，接口已定）::

       python3 experiments/dualstream/train_dualstream.py --data /path/to/*.yaml --epochs 150

   真实训练需要先确定「颜色 + 图案」属性标签格式（见 README 的待定项），
   当前 --data 分支只做参数校验与计划打印，不会开始训练。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 环境未装 torch 时的友好提示
    print("❌ 未安装 PyTorch。安装：pip install torch --index-url https://download.pytorch.org/whl/cpu", file=sys.stderr)
    raise SystemExit(1)

from ctts.models.dualstream import DualStreamConfig, DualStreamNet  # noqa: E402


def describe(cfg: DualStreamConfig) -> None:
    print("== 双流配置 ==")
    print(f"  亮度支路输入 : {cfg.y_size}×{cfg.y_size} ×1ch（全分辨率、无颜色）")
    print(f"  色度支路输入 : {cfg.c_size}×{cfg.c_size} ×2ch（半分辨率、CbCr）")
    print(f"  主干宽度     : Y={cfg.y_width}  C={cfg.c_width}")
    print(f"  属性头       : 颜色 {cfg.num_colors} 类 + 图案 {cfg.num_patterns} 类（sigmoid 多标签）")
    print(f"  门控初始化   : bias={cfg.gate_bias_init}（sigmoid≈{torch.sigmoid(torch.tensor(cfg.gate_bias_init)):.3f}）")


def dry_run(cfg: DualStreamConfig, device: str) -> int:
    net = DualStreamNet(cfg).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    describe(cfg)
    print(f"== 参数量 : {n_params:,} ==")

    # 用缩小尺寸跑前向，兼顾速度；结构对输入尺寸无硬约束（cfg 只用于尺寸校验）
    h = w = cfg.c_size * 2
    x = torch.rand(2, 3, h, w, device=device)
    out = net(x)
    print("== 前向 ==")
    for i, f in enumerate(out["feats"]):
        print(f"  feat[{i}]: {tuple(f.shape)}")
    print(f"  color_logits  : {tuple(out['color_logits'].shape)}")
    print(f"  pattern_logits: {tuple(out['pattern_logits'].shape)}")

    # 反事实一致性提示：R↔B 交换后 Y 几乎不变、色度反转
    from ctts.models.color import swap_rb

    import numpy as np

    img = (x[0].detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8)  # RGB→BGR
    swapped = swap_rb(img)
    y0 = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    y1 = 0.2126 * swapped[..., 2] + 0.7152 * swapped[..., 1] + 0.0722 * swapped[..., 0]
    print(f"== 反事实检查 == R↔B 交换后亮度平均绝对变化 {abs(float(y0.mean()) - float(y1.mean())):.2f}/255")

    loss = out["color_logits"].sum() + out["pattern_logits"].sum()
    loss.backward()
    grad_norms = {name: float(p.grad.norm()) for name, p in net.named_parameters() if p.grad is not None}
    nonzero = sum(1 for v in grad_norms.values() if v > 0)
    print(f"== 反向 ==")
    print(f"  有梯度的参数张量 : {nonzero}/{len(list(net.parameters()))}")
    print(f"  门控层梯度范数   : {grad_norms.get('fuses.0.gate.weight', 0.0):.6f}（初始应接近 0）")
    if nonzero == 0:
        print("❌ 没有任何参数获得梯度", file=sys.stderr)
        return 1
    print("✅ 骨架烟雾测试通过")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="双流特征解耦实验（骨架）")
    p.add_argument("--dry-run", action="store_true", help="只做结构/梯度烟雾测试")
    p.add_argument("--data", help="数据集 YAML（真实训练，尚未实现）")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--y-size", type=int, default=640)
    p.add_argument("--c-size", type=int, default=320)
    args = p.parse_args()

    cfg = DualStreamConfig(y_size=args.y_size, c_size=args.c_size)
    if args.dry_run or not args.data:
        return dry_run(cfg, args.device)

    print("真实训练尚未实现。需要先确定属性标签格式，待办：")
    print("  1. dataset: 读取 颜色(4) / 图案(9) 两套标签，产出 (bgr, color_t, pattern_t)")
    print("  2. loss    : BCEWithLogits，颜色与图案分别加权（可用 LibMTL/PCGrad 自动平衡）")
    print("  3. 消融    : E0 基线 / E1 单流4ch / E2 双流gated / E3 双流add / E4 concat / E5 色度剪枝")
    print("  4. 导出    : ONNX 双输入（y, cbcr）→ TensorRT 多 binding，与本仓 model_contract.yaml 对齐")
    print(f"（收到 --data {args.data}，--epochs {args.epochs}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
