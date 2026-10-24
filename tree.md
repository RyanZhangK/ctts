# 目录结构说明

## 目录结构

```
.
├── README.md                 本仓定位、工作流、契约、上下游
├── AGENTS.md                 AI 协作规范
├── plan.md                   开发计划与进度
├── decision.md               关键工程决策日志
├── tree.md                   本文件
├── CODEOWNERS                Review 自动分派
├── .foray-layer              所属层与依赖边界（layer: T 离线工具层）
├── .gitignore
├── LICENSE                   GPL-3.0-only
├── pyproject.toml            打包/依赖/脚本入口（console script: ctts）
├── .github/workflows/ci.yml  CI（ruff + pytest）
├── src/ctts/                 Python 包源码
│   ├── __main__.py           统一入口（CLI + GUI）
│   ├── dataset_build.py      视频抽帧 + 质量过滤
│   ├── split_dataset.py      train/val 划分
│   ├── yolo_train.py         YOLO 微调训练
│   ├── train_args.py         参数规范化 / 覆盖规则
│   ├── prelabel.py           小模型预标注流水线
│   ├── review.py             复核页（批次认可）
│   ├── label_studio_auth.py  LS 认证与 API 客户端
│   ├── label_studio_project.py
│   ├── label_studio_data.py
│   ├── label_studio_main.py  LS CLI 入口
│   ├── detect.py             实时推理（PySide6 GUI）
│   └── models/               模型与数据通路（color: numpy 4:2:0；dualstream: torch 双流骨架）
├── experiments/dualstream/   双流实验（训练脚本、模型契约、消融表）
│   ├── train_dualstream.py   --dry-run 烟雾测试 / --data 属性训练
│   └── model_contract.yaml   对下游推理仓的接口约定
├── tests/                    单元测试（prelabel / review / color / dualstream / dataset / 训练冒烟）
├── scripts/                  环境初始化与 GUI 启动脚本（.sh/.cmd/.ps1）
├── docs/                     设计文档与截图
└── dataset/                  标注定义（*.yaml / classes.txt），不含图片
```

## 记录约束

MUST NOT 记录以下内容：

- `.venv/`、`runs/`、`__pycache__/`、`weight/` 等依赖与产物目录
- 权重与导出文件（`*.pt` / `*.onnx` / `*.engine`）
- 数据集图片与标签（`dataset/*/images`、`labels`、`train`、`val`）
- 临时文件、缓存文件、日志文件
