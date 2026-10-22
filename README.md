# CTTS — CNN Train Tools Set

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](#环境配置)

视觉 CNN 训练全流程工具集：数据准备 → 标注管理 → 小模型预标注 → 训练 → 推理验证。

## 有什么用？

目标检测最贵的一步是标注。CTTS 把「抽帧 → 质量过滤 → Label Studio 标注 → 小模型预标注 → 导出 → 训练 → 推理」串成一条可重复的流程，核心是：

- **少标一点也能开始** — 默认只需 ≥50 张已标注图、≥100 个框，即可微调 YOLO nano 模型。
- **预标注省人工** — nano 模型先给未标注任务画框，人工从「画框」变成「改框」。
- **数据不离开 Label Studio** — 数据就绪度评估、微数据集构建、预测写回都走 LS API，不用手工搬文件。
- **单机 / 内网都能用** — 本机 LS 零配置读取 Token；远程实例用 `--base-url` + `--token`。
- **CLI 与 GUI 等价** — 批量任务走 CLI，日常操作走六页签 GUI，二者产物一致。

实际能省多少标注取决于目标尺度、场景复杂度和类别数。建议先跑 `prelabel assess` 看数据是否就绪，再决定是否进入训练。

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
   │                          │
   │                          ▼
   │                人工复核并提交
   ▼
导出 YOLO → data split → train → detect
```

## 项目结构

```text
ctts/
├── LICENSE                 # GPL-3.0
├── README.md
├── tools/                  # 工具脚本
│   ├── ctts.py            # ★ 统一入口（CLI + GUI）
│   ├── dataset_build.py   # 视频抽帧 + 质量过滤
│   ├── filter_quality.py  # 图片质量筛选
│   ├── split_dataset.py   # train/val 数据集划分
│   ├── yolo_train.py      # YOLO 微调训练
│   ├── detect.py          # 实时推理 (PySide6 GUI)
│   ├── label_studio_auth.py     # LS 认证与 API 客户端
│   ├── label_studio_project.py  # LS 项目管理
│   ├── label_studio_data.py     # LS 任务 CRUD
│   ├── label_studio_main.py     # LS CLI 入口
│   ├── prelabel.py        # ★ 小模型预标注流水线
│   ├── test_prelabel.py   # 预标注离线单测（无需 LS/GPU）
│   ├── pyproject.toml     # 依赖声明
│   └── .gitignore
├── docs/                  # 设计文档
│   └── prelabel_design.md # 预标注功能设计
├── runs/prelabel/         # 预标注产物（不纳入版本控制）
├── dataset/               # 数据集（不纳入版本控制）
│   ├── RawData/           # 原始图片
│   └── LabelData/         # 标注后数据 (images/ labels/ data.yaml)
└── weights/               # 模型权重（不纳入版本控制）
    ├── yolo26n.pt
    ├── yolo26s.pt
    └── yolo26m.pt
```

## 快速开始

### 获取代码

```bash
git clone git@github.com:RyanZhangK/ctts.git
cd ctts
```

### 环境配置

跨平台一键初始化（Linux / macOS / Windows）：

```bash
cd ctts/tools/

python setup_env.py          # 创建 venv 并安装依赖（优先 uv，否则回退到 pip）
python setup_env.py --run    # 初始化后直接启动 GUI
python setup_env.py --check  # 只检查环境
```

也可使用平台入口：

```bash
./setup_env.sh      # Linux / macOS
setup_env.cmd       # Windows（双击或命令行）
```

venv 默认位于 `~/.cache/ctts-venv`（可通过 `--venv` 或 `CTTS_VENV` 修改）。
初始化后手动激活：

```bash
source ~/.cache/ctts-venv/bin/activate    # Linux / macOS
# 或 PowerShell: & "$HOME\.cache\ctts-venv\Scripts\Activate.ps1"
```

### CLI 使用

```bash
source ~/.cache/ctts-venv/bin/activate

# 数据准备
python ctts.py data build video.mp4 output/ -i 0.5     # 视频抽帧
python ctts.py data filter images/ --move              # 质量过滤
python ctts.py data split dataset/ -v 0.2              # 划分训练/验证集

# 标注管理（需先启动 Label Studio）
python ctts.py label project list                      # 列出项目
python ctts.py label project create "新项目" --labels stb
python ctts.py label data stats -p 10                  # 标注进度
python ctts.py label data list -p 10                   # 查看任务

# 小模型预标注（评估 → 建微数据集 → 训练 nano → 导出 ONNX → 预标注）
python ctts.py prelabel assess -p 10                   # 只评估数据是否足够
python ctts.py prelabel run -p 10                      # 全流程
python ctts.py prelabel build -p 10 --out /tmp/ds      # 只构建微数据集
python ctts.py prelabel train --data /tmp/ds           # 只训练并导出 ONNX
python ctts.py prelabel predict -p 10 -m runs/prelabel/<run>/model/best.onnx

# 训练
python ctts.py train                                   # 使用 yolo_train.py 配置

# 推理
python ctts.py detect --cam 0                          # 摄像头推理
python ctts.py detect --video test.mp4                 # 视频推理
```

### GUI 使用

```bash
source ~/.cache/ctts-venv/bin/activate
python ctts.py --gui
```

GUI 包含六个 Tab：

- **📦 数据准备** — 视频抽帧、质量过滤、数据集划分
- **🏷️ 标注管理** — Label Studio 项目与任务管理；上传逐张显示进度，项目列表展示已标注数/进度并每 15s 自动刷新
- **🎯 预标注** — 评估数据 → 训练小模型 → API 预标注 → 复核
- **🗂️ 数据集划分** — 已标注数据划分为 train/val
- **🚀 训练** — YOLO 训练参数配置、高级参数调试与启动
- **🔍 推理** — 实时检测窗口启动

## 小模型预标注

人工只标一小部分，用 nano 模型先标剩下的，再人工复核，可显著减少重复劳动。

流程（一个 Tab 内完成，也可用 `ctts.py prelabel`）：

1. **评估数据就绪度** — 通过 Label Studio API 统计已标注任务、实例数、类别覆盖。
   默认门槛：≥50 张、≥100 框、每类 ≥30 框。达到一半（25/50/15）判为「勉强可训练」，
   会弹窗确认；不足则明确提示还差多少。
2. **构建微数据集** — 只下载/匹配已标注图片，百分比坐标直接转 YOLO 归一化坐标，
   无效框丢弃并计数，复用 `split_dataset` 划分 train/val。
3. **训练并导出** — 以 `weights/yolo26n.pt` 为基座微调，导出 `best.onnx` 并做加载自检。
4. **API 预标注** — 只处理未标注任务，分批写入 Label Studio predictions；
   同一版本标签默认跳过，勾选「覆盖」才会重写。
5. **人工复核** — 在 Label Studio 中确认/修正预测并提交；工具不会自动接受预测。

产物固定落在 `runs/prelabel/<run_id>/`（dataset、train、model、report.json、run.log），
版本标签形如 `ctts-prelabel:v1@480`，可在「仅预标注」时复用历史模型。

> 复核抽样 CSV 与待复核数量统计在「④ 复核」分组；预测框不会被自动确认为标注。

## Label Studio 集成

CTTS 通过 Label Studio API 管理标注项目。支持：

- **零配置认证** — 自动从本机 Label Studio SQLite 数据库读取 Token
  （本机场景足够方便；共享机器上请改用 `--token` 或环境变量 `LABEL_STUDIO_TOKEN`）。
- **远程实例** — `--base-url http://host:8080 --token <TOKEN>`，适用于内网/容器部署。
- **项目管理** — 创建/删除项目，按类别生成目标检测 label config。
- **本地文件存储** — 直接指向图片目录，无需上传。
- **任务管理** — 浏览、删除、统计标注进度；GUI 上传逐张显示进度。
- **状态自动刷新** — GUI 停留在「标注管理」时每 15s 静默刷新项目计数，
  上传/创建/删除后立即刷新，无需手动点「刷新」。

### Label Studio 环境要求

- Label Studio 运行在 `http://127.0.0.1:8080`
- 首次使用前需在 Web UI 登录以生成 API Token
- 本地图片导入需在 Label Studio 启动目录创建 `mydata/` 目录

```bash
# 创建本地文件目录
mkdir -p ~/mydata
ln -sf /path/to/your/images ~/mydata/

# 然后在 CTTS 中添加存储
python ctts.py label project storage-add 10 /home/ryanz/mydata/images
python ctts.py label project storage-sync 1
```

## 典型工作流

```bash
# 1. 视频抽帧
python ctts.py data build video.mp4 dataset/RawData/ -i 0.5

# 2. 质量过滤
python ctts.py data filter dataset/RawData/ --move

# 3. 导入 Label Studio 标注（通过 GUI 或 CLI storage-add）
python ctts.py label project storage-add 10 /home/ryanz/mydata/RawData
python ctts.py label project storage-sync 1

# 4. 在 Label Studio Web UI 完成一小部分人工标注

# 5. 小模型预标注（数据达到 50 张 / 100 框后）
python ctts.py prelabel run -p 10
# → 回到 Label Studio 复核预测框并提交

# 6. 导出标注（Label Studio Web UI → Export → YOLO）

# 7. 划分数据集
python ctts.py data split dataset/LabelData/ -v 0.2

# 8. 训练
python ctts.py train

# 9. 推理验证
python ctts.py detect --video test.mp4 -w weights/best.pt
```

## 依赖

| 包 | 用途 |
|----|------|
| opencv-python | 视频/图像处理 |
| numpy | 数值计算 |
| httpx | 网络请求 |
| ultralytics | YOLO 训练与推理 |
| PySide6 | 图形界面 |
| label-studio-sdk | Label Studio API |

运行 `python setup_env.py` 一键安装全部依赖。

## 自定义训练参数

训练默认值集中在 `tools/yolo_train.py` 的 `DEFAULT_TRAIN_ARGS`；预标注微调的默认值由
`tools/prelabel.py` 的 `default_train_args()` 生成。两处都支持「基础字段 + 高级 JSON」双层覆盖：

- **GUI**：⑤ 训练 和 ③ 预标注 的「高级设置」文本框可粘贴 JSON 对象，点「载入默认参数」
  会填入当前全部可调项；留空即使用默认值。
- **CLI**：`yolo_train.py` 支持 `--json`、`--set KEY=VALUE`、`--print-defaults`、`--dry-run`；
  `prelabel.py train/run` 支持 `--train-json`、`--train-set`。

优先级为 **默认值 → GUI 基础字段 / CLI 基础参数 → 高级 JSON**，高级项最后生效。
`data`、`project`、`name`、`exist_ok` 属于程序托管的结构性参数，写在高级 JSON 中会被忽略。

```bash
# 列出全部训练参数
python yolo_train.py --print-defaults

# 只打印最终参数，不启动训练（调试用）
python yolo_train.py --dry-run --set lr0=0.001 --set cos_lr=false \
  --json '{"mosaic": 0.0, "mixup": 0.1}'

# 预标注微调时覆盖任意 ultralytics 训练参数
python prelabel.py train --data runs/prelabel/demo/dataset \
  --train-set lr0=0.001 --train-json '{"copy_paste": 0.1, "dropout": 0.1}'
```

GUI 训练页的基础字段（weights/data/epochs/batch/imgsz/device）仍按原方式生效；
高级 JSON 可覆盖它们中的任意项，也可传入文件里未列出的 ultralytics 参数，便于逐个调试。

## 贡献

欢迎 Issue 和 Pull Request。提交前建议：

```bash
cd tools
python -m unittest test_prelabel -q
```

单测不依赖 Label Studio 或 GPU；涉及 GUI 的改动请至少跑一次 `python ctts.py --gui` 手工验证。

## 开源协议

本项目采用 **GNU General Public License v3.0 only（GPL-3.0）** 发布，
完整条款见 [LICENSE](LICENSE)。

- 可以自由使用、修改、分发，包括商业用途；
- 分发修改后的版本时，需同样以 GPL-3.0 开源并提供对应源代码；
- 软件按「原样」提供，不附带任何担保。

Copyright (C) 2026 RyanZ
