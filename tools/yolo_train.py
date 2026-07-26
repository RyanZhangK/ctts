from pathlib import Path
from ultralytics.models.yolo import YOLO

SOURCE_WEIGHT = "/home/ryanz/eDisk/nuedc/26/vision/weights/yolo26n.pt"
DATASET_DIR = "/home/ryanz/eDisk/nuedc/26/vision/dataset/LabelData/data.yaml"
FREEZE_LAYERS = 6


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
    model = YOLO(SOURCE_WEIGHT)
    freeze_layers(model, FREEZE_LAYERS)

    model.train(
        data=DATASET_DIR,
        epochs=40,
        imgsz=640,
        batch=4,
        amp=True,
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=15,            
        patience=10,             
        label_smoothing=0.1,      
        dropout=0.1,              
        cache=True,
        name="21F_yolo26m_finetune",
        exist_ok=True,
        device=0,
        workers=4,               
        save=True,
        save_period=15,
        val=True,
    )

if __name__ == "__main__":
    main()
