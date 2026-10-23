# foray_ml

> **T（离线工具层）** · owner 角色：视觉

视觉 CNN 训练全流程工具集 **CTTS**（CNN Train Tools Set）：数据准备 → 标注管理 → 小模型预标注 → 训练 → 推理验证。本仓是 Foray 战队的**模型产线**：数据集规范、模型训练、离线评估、ONNX/engine 导出与模型契约都在这里；实时推理（ROS 2 / TensorRT）在 `foray_vision`，不在本仓。

> **命名说明**：组织仓名为 `foray_ml`（职责名）；Python 包与命令行工具仍叫 `ctts`（保留既有品牌与文档习惯）。
> 本仓由个人仓 `RyanZhangK/ctts` 迁入，完整保留原有提交历史。

## 有什么用？

目标检测最贵的一步是标注。CTTS 把「抽帧 → 质量过滤 → Label Studio 标注 → 小模型预标注 → 导出 → 训练 → 推理」串成一条可重复的流程：

- **少标一点也能开始** — 默认只需 ≥50 张已标注图、≥100 个框，即可微调 YOLO nano 模型。
- **预标注省人工** — nano 模型先给未标注任务画框，人工从「画框」变成「改框」。
- **数据不离开 Label Studio** — 数据就绪度评估、微数据集构建、预测写回都走 LS API。
- **单机 / 内网都能用** — 本机 LS 零配置读取 Token；远程实例用 `--base-url` + `--token`。
- **CLI 与 GUI 等价** — 批量任务走 CLI，日常操作走六页签 GUI，二者产物一致。

## 工作流

```text
视频 / 图片
   │  data build / data filter
   ▼
RawData ──► Label Studio 人工标注一小部分
   │                          │
   │                          ▼
   │                prelabel assess（数据够不够？）
   │                          │ 够
   │                          ▼
   │                建微数据集 → 训练 yolo26n → 导出 ONNX
   │                          │
   │                          ▼
   │                写回 predictions（仅未标注任务）
   │                          ▼
   │                人工复核并提交
   ▼
导出 YOLO → data split → train → detect
```

## 快速开始

```bash
# 1. 环境初始化（优先 uv，其次 venv；默认 venv 在 ~/.cache/ctts-venv）
python3 scripts/setup_env.py
# 或：./scripts/setup_env.sh

# 2. 安装为本仓可编辑包（开发用）
pip install -e ".[dev]"

# 3. 使用
ctts --gui                        # 图形界面
ctts data build video.mp4 out/ -i 0.5
ctts data filter images/ --move
ctts data split dataset/ -v 0.2
ctts label project list
ctts prelabel assess -p 10
ctts train
ctts detect --cam 0
```

未安装时可用模块方式运行（需 `PYTHONPATH=src`）：`python3 -m ctts --help`。

## 目录结构

```text
foray_ml/
├── src/ctts/               Python 包（训练/标注/预标注/推理工具）
│   ├── __main__.py         ★ 统一入口（CLI + GUI），console script: ctts
│   ├── dataset_build.py    视频抽帧 + 质量过滤
│   ├── split_dataset.py    train/val 划分
│   ├── yolo_train.py       YOLO 微调训练（DEFAULT_TRAIN_ARGS 在这里）
│   ├── train_args.py       参数规范化 / 覆盖规则
│   ├── prelabel.py         小模型预标注流水线
│   ├── review.py           复核页（批次认可）
│   ├── label_studio_*.py   Label Studio 认证 / 项目 / 任务 API
│   └── detect.py           实时推理（PySide6 GUI）
├── tests/                  纯逻辑单元测试（无 GPU / 无网络）
├── scripts/                环境初始化与启动脚本（跨平台）
├── docs/                   设计文档
├── dataset/                标注定义（data.yaml / classes.txt），不含图片
└── weight/                 本地权重（gitignored）
```

## 依赖

| 类型 | 依赖 |
|---|---|
| Python | ≥ 3.11 |
| 运行时 | opencv · numpy · httpx · label-studio-sdk · PySide6 · ultralytics |
| 外部服务 | Label Studio（本机或内网实例） |

可选环境变量：`CTTS_WEIGHTS`（默认权重路径）、`CTTS_DATASET`（默认 data.yaml）、`CTTS_VENV`（venv 位置）。

## 产出与契约

本仓对下游（`foray_vision`）的交付物：**权重 / ONNX / engine + 模型契约**。导出时须同时给出一份契约，至少包含：输入 layout（如 Y/U/V 或 BGR 的尺寸与归一化）、预处理（letterbox 参数）、类别映射（含颜色/图案分头时的索引区间）、opset 与 TensorRT 版本。契约变更按破坏性接口处理，在 `decision.md` 留痕。

## 上下游

| 方向 | 对象 |
|---|---|
| 上游 | 无实时仓依赖；数据集来自 `DataSet_*` / Label Studio |
| 下游 | `foray_vision`（按契约做 C++/TensorRT 推理）、`foray_auto_aim`（间接） |

## 纪律

- **权重、数据集图片、训练产物不进 git**（见 `.gitignore`）；只提交标注定义与脚本
- 不在本仓实现 ROS 2 / 实时推理逻辑
- 魔法路径必须来自环境变量或 CLI 参数，禁止硬编码个人机器路径
- 新增/变更训练参数只改 `DEFAULT_TRAIN_ARGS` 一处
