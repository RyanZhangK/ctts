"""YOLO 微调入口。

所有训练参数集中在 DEFAULT_TRAIN_ARGS。GUI「训练」页的「高级设置」和 CLI
`--json/--set` 可以覆盖其中任意一项（data/project/name/exist_ok 由程序托管）。
"""

from __future__ import annotations

import argparse
import json
import os as _os
from pathlib import Path

from ultralytics.models.yolo import YOLO

from train_args import (
    MANAGED_KEYS,
    OverrideError,
    merged,
    normalize,
    parse_json_text,
    parse_set,
)

_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
# SOURCE_WEIGHT = _os.path.join(_PROJECT_ROOT, "weights", "yolo26s.pt")
SOURCE_WEIGHT = "/home/ryanz/eDisk/nuedc/26/ctts/runs/detect/26钢珠/weights/best.pt"
# DATASET_DIR = _os.path.join(_PROJECT_ROOT, "dataset", "LabelData", "data.yaml")
DATASET_DIR = "/home/ryanz/eDisk/nuedc/26/ctts/dataset/sLabelData/data.yaml"
FREEZE_LAYERS = 8

# 一份完整的可调参数清单：新增/调试参数时只改这里。
# 值即默认值；GUI「载入默认参数」与 CLI --print-defaults 都读取它。
DEFAULT_TRAIN_ARGS: dict = {
    "data": DATASET_DIR,
    "epochs": 150,
    "imgsz": 320,
    "batch": 8,
    "amp": True,
    # --- 学习率 ---
    "lr0": 5e-4,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 5e-4,
    "warmup_epochs": 5,
    "cos_lr": True,
    # --- 增强 ---
    "degrees": 15,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 5,
    "perspective": 0.0005,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "close_mosaic": 0,        # 全程 mosaic
    # --- 正则化 ---
    "dropout": 0.2,
    "patience": 15,
    # --- 其他 ---
    "cache": True,
    "name": "26钢珠_small",
    "exist_ok": True,
    "device": 0,
    "workers": 4,
    "save": True,
    "save_period": 10,
    "val": True,
}


def freeze_layers(model, n: int):
    freeze_names = [f"model.{i}." for i in range(n)]
    for name, param in model.model.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze_names):
            param.requires_grad = False
    frozen = sum(
        1 for _, p in model.model.named_parameters() if not p.requires_grad
    )
    print(f"[freeze] 已冻结前 {n} 层，共 {frozen} 参数量")


def build_train_args(overrides: dict | None = None) -> tuple[dict, list[str]]:
    """默认参数 + 高级覆盖；返回 (训练参数, 被忽略的托管参数)。"""
    clean, ignored = normalize(overrides)
    return merged(DEFAULT_TRAIN_ARGS, clean), ignored


def run_training(
    weights: str,
    *,
    data: str | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    device: str | int | None = None,
    freeze: int | None = None,
    overrides: dict | None = None,
    log=print,
):
    """按「默认 → GUI 显式字段 → 高级覆盖」的优先级训练。

    高级 JSON 优先级最高，便于临时调试任意参数；结构性的
    data/project/name/exist_ok 会从高级参数中忽略。
    """
    args, ignored = build_train_args(overrides)
    args["data"] = data or args["data"]
    if epochs is not None:
        args["epochs"] = epochs
    if batch is not None:
        args["batch"] = batch
    if imgsz is not None:
        args["imgsz"] = imgsz
    if device is not None:
        args["device"] = device
    if ignored:
        log(f"⚠ 高级参数中的托管项已忽略: {', '.join(ignored)}")

    log(f"[args] {json.dumps(args, ensure_ascii=False, default=str)}")
    model = YOLO(weights)
    if freeze is not None and freeze > 0:
        freeze_layers(model, freeze)
    return model.train(**args)


def _parse_cli(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="YOLO 微调；可覆盖 DEFAULT_TRAIN_ARGS 中的任意参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python yolo_train.py --set lr0=0.001 --set cos_lr=false\n"
               "  python yolo_train.py --json '{\"epochs\": 50, \"mosaic\": 0.0}'\n"
               "  python yolo_train.py --print-defaults",
    )
    parser.add_argument("--weights", default=SOURCE_WEIGHT, help="预训练权重")
    parser.add_argument("--data", default=None, help="data.yaml（默认用 DATASET_DIR）")
    parser.add_argument("--device", default=None, help="如 0 / cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--freeze-layers", type=int, default=None,
                        help="冻结前 N 层（按层名冻结，区别于 ultralytics 的 freeze 参数）")
    parser.add_argument("--json", default=None, help="JSON 对象，覆盖任意训练参数")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="单参数覆盖，可重复；值按 JSON 解析，失败按字符串")
    parser.add_argument("--print-defaults", action="store_true",
                        help="打印完整默认参数 JSON 后退出")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印最终参数，不启动训练")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_cli(argv)

    if args.print_defaults:
        clean, _ = build_train_args()
        print(json.dumps(clean, ensure_ascii=False, indent=2, default=str))
        return

    overrides: dict = {}
    try:
        if args.json:
            json_over, ignored = parse_json_text(args.json)
            overrides.update(json_over)
            if ignored:
                print(f"⚠ 托管项已忽略: {', '.join(ignored)}")
        set_over, ignored = parse_set(args.set)
        overrides.update(set_over)
        if ignored:
            print(f"⚠ 托管项已忽略: {', '.join(ignored)}")
    except OverrideError as exc:
        print(f"❌ {exc}")
        raise SystemExit(2)

    if args.dry_run:
        final, ignored = build_train_args(overrides)
        final["data"] = args.data or final["data"]
        if args.epochs is not None:
            final["epochs"] = args.epochs
        print(json.dumps(final, ensure_ascii=False, indent=2, default=str))
        return

    run_training(
        args.weights, data=args.data, epochs=args.epochs, batch=args.batch,
        imgsz=args.imgsz, device=args.device, freeze=args.freeze_layers,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
