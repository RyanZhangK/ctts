#!/usr/bin/env python3
"""双流特征解耦实验入口。

两种用法：

1. **烟雾测试**（无需数据集、无需 GPU，验证结构 + 反向传播能否跑通）::

       python3 experiments/dualstream/train_dualstream.py --dry-run

2. **属性分类训练**（数据集需含 `labels_attr/`，见 model_contract.yaml）::

       python3 experiments/dualstream/train_dualstream.py --data /path/to/dataset --epochs 50

   数据集布局（与 ultralytics data.yaml 兼容）::

       dataset/
       ├── data.yaml            # train/val 指向图片目录
       ├── train/images/*.jpg
       ├── train/labels/*.txt   # 标准 YOLO 5 字段（ultralytics 可直接用）
       └── train/labels_attr/*.txt   # `<cls> <cx> <cy> <w> <h> <color> <pattern>`

   消融组（E0–E5）通过 `--variant` 切换，见 experiments/dualstream/README.md。
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
    import torch.nn.functional as F
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
    print("== 反向 ==")
    print(f"  有梯度的参数张量 : {nonzero}/{len(list(net.parameters()))}")
    print(f"  门控层梯度范数   : {grad_norms.get('fuses.0.gate.weight', 0.0):.6f}（初始应接近 0）")
    if nonzero == 0:
        print("❌ 没有任何参数获得梯度", file=sys.stderr)
        return 1
    print("✅ 骨架烟雾测试通过")
    return 0


def _resolve_root(data: str) -> Path:
    """`--data` 既可传数据集目录，也可传其 data.yaml。"""
    p = Path(data).expanduser().resolve()
    return p if p.is_dir() else p.parent


def train(
    cfg: DualStreamConfig,
    data: str,
    epochs: int,
    device: str,
    batch_size: int = 8,
    lr: float = 1e-3,
    out_dir: str = "runs/dualstream",
    num_workers: int = 0,
    image_loader=None,
) -> int:
    """属性分类训练（图像级：每图取面积最大框的属性）。

    `image_loader` 仅供测试注入（避免依赖 opencv）；生产路径为 None 时用 cv2。
    """
    from torch.utils.data import DataLoader

    from ctts.models.dataset import AttrDataset

    root = _resolve_root(data)
    ds_train = AttrDataset(root, "train", image_loader=image_loader)
    ds_val = AttrDataset(root, "val", image_loader=image_loader)
    print(f"数据集: {root}  train={len(ds_train)}  val={len(ds_val)}")
    if ds_train.conflicts:
        print(f"⚠️  {len(ds_train.conflicts)} 张训练图存在多属性目标，图像级训练会丢弃次要目标", file=sys.stderr)

    loader = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    train_dl, val_dl = loader(ds_train, True), loader(ds_val, False)

    net = DualStreamNet(cfg).to(device)
    describe(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def run_batch(batch: dict) -> tuple[torch.Tensor, dict]:
        x = batch["bgr"].to(device).float()
        x = F.interpolate(x, size=(cfg.y_size, cfg.y_size), mode="bilinear", align_corners=False)
        out = net(x)
        total = torch.zeros((), device=device)
        stats: dict = {}
        tasks = 0
        for logits, target, num_cls, key in (
            (out["color_logits"], batch["color"].to(device), cfg.num_colors, "color"),
            (out["pattern_logits"], batch["pattern"].to(device), cfg.num_patterns, "pattern"),
        ):
            valid = target >= 0
            if not bool(valid.any()):
                continue
            onehot = F.one_hot(target[valid], num_cls).float()
            loss = bce(logits[valid], onehot).mean()
            total = total + loss
            stats[f"{key}_loss"] = float(loss.detach())
            stats[f"{key}_acc"] = float((logits[valid].argmax(1) == target[valid]).float().mean())
            tasks += 1
        if tasks == 0:
            raise RuntimeError("整个 batch 都没有有效属性标签，检查 labels_attr/ 是否生成")
        return total / tasks, stats

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        net.train()
        agg: dict = {}
        for batch in train_dl:
            loss, stats = run_batch(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in stats.items():
                agg.setdefault(k, []).append(v)
        train_line = " ".join(f"{k}={sum(v) / len(v):.4f}" for k, v in agg.items())

        net.eval()
        vagg: dict = {}
        with torch.no_grad():
            for batch in val_dl:
                _, stats = run_batch(batch)
                for k, v in stats.items():
                    vagg.setdefault(k, []).append(v)
        val_line = " ".join(f"val_{k}={sum(v) / len(v):.4f}" for k, v in vagg.items())
        val_loss = sum(vagg.get("color_loss", [0.0])) / max(len(vagg.get("color_loss", [1])), 1)
        print(f"[{epoch:3d}/{epochs}] {train_line}  {val_line}")

        state = {"cfg": cfg.__dict__, "state_dict": net.state_dict(), "epoch": epoch}
        torch.save(state, out_path / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, out_path / "best.pt")
    print(f"✅ 训练结束，权重保存在 {out_path}/（best.pt / last.pt）")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="双流特征解耦实验")
    p.add_argument("--dry-run", action="store_true", help="只做结构/梯度烟雾测试")
    p.add_argument("--data", help="数据集目录或其 data.yaml（含 labels_attr/）")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="runs/dualstream", help="权重输出目录")
    p.add_argument("--variant", default="E2-gated", choices=["E0", "E1", "E2-gated", "E3-add", "E4-concat", "E5"],
                   help="消融组（目前 E2 为默认实现，其余见 README 待实现）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--y-size", type=int, default=640)
    p.add_argument("--c-size", type=int, default=320)
    args = p.parse_args()

    cfg = DualStreamConfig(y_size=args.y_size, c_size=args.c_size)
    if args.dry_run or not args.data:
        return dry_run(cfg, args.device)
    return train(
        cfg, args.data, args.epochs, args.device,
        batch_size=args.batch_size, lr=args.lr, out_dir=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
