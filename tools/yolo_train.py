from pathlib import Path
from ultralytics.models.yolo import YOLO

import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
# SOURCE_WEIGHT = _os.path.join(_PROJECT_ROOT, "weights", "yolo26s.pt")
SOURCE_WEIGHT = "/home/ryanz/eDisk/nuedc/26/ctts/runs/detect/26钢珠/weights/best.pt"
# DATASET_DIR = _os.path.join(_PROJECT_ROOT, "dataset", "LabelData", "data.yaml")
DATASET_DIR = "/home/ryanz/eDisk/nuedc/26/ctts/dataset/sLabelData/data.yaml"
FREEZE_LAYERS = 8


def freeze_layers(model, n: int):
    freeze_names = [f"model.{i}." for i in range(n)]
    for name, param in model.model.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze_names):
            param.requires_grad = False
    frozen = sum(
        1 for _, p in model.model.named_parameters() if not p.requires_grad
    )
    print(f"[freeze] 已冻结前 {n} 层，共 {frozen} 参数量")


def main():
    print("！！！微调参数！！！记得改回来")
    model = YOLO(SOURCE_WEIGHT)
    freeze_layers(model, FREEZE_LAYERS)

    model.train(
    data=DATASET_DIR,
    epochs=150,
    imgsz=320,
    batch=8,
    amp=True,
    # --- 学习率 ---
    lr0=5e-4,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=5,
    cos_lr=True,
    # --- 增强 ---
    degrees=15,
    translate=0.1,
    scale=0.5,
    shear=5,
    perspective=0.0005,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    close_mosaic=0,        # 全程 mosaic
    # --- 正则化 ---
    dropout=0.2,
    patience=15,
    # --- 其他 ---
    cache=True,
    name="26钢珠_small",
    exist_ok=True,
    device=0,
    workers=4,
    save=True,
    save_period=10,
    val=True,
)
#！！！微调参数！！！记得改回来

if __name__ == "__main__":
    main()
