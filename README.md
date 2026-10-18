# CTTS — CNN Train Tools Set

视觉 CNN 训练全流程工具集：数据准备 → 标注管理 → YOLO训练 → 推理验证。

## 有什么用？


## 项目结构

```
ctts/
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
│   ├── pyproject.toml     # uv 依赖声明
│   └── .gitignore
├── dataset/               # 数据集（不纳入版本控制）
│   ├── RawData/           # 原始图片
│   └── LabelData/         # 标注后数据 (images/ labels/ data.yaml)
└── weights/               # 模型权重（不纳入版本控制）
    ├── yolo26n.pt
    ├── yolo26s.pt
    └── yolo26m.pt
```

## 快速开始

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
python ctts.py label project create "新项目"           # 创建项目
python ctts.py label data stats -p 10                  # 标注进度
python ctts.py label data list -p 10                   # 查看任务

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

GUI 包含四个 Tab：
- **📦 数据准备** — 视频抽帧、质量过滤、数据集划分
- **🏷️ 标注管理** — Label Studio 项目与任务管理
- **🚀 训练** — YOLO 训练参数配置与启动
- **🔍 推理** — 实时检测窗口启动

## Label Studio 集成

CTTS 通过 Label Studio API 管理标注项目。支持：

- **零配置认证** — 自动从 Label Studio SQLite 数据库读取 Token  (其实不是很安全的说，鉴于LS的常见场景下都是本机运行所以选择性无视了)
- **项目管理** — 创建/删除项目，预设目标检测和分类模板
- **本地文件存储** — 直接指向图片目录，无需上传
- **任务管理** — 浏览、删除、统计标注进度

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

# 4. 在 Label Studio Web UI 完成人工标注

# 5. 导出标注（Label Studio Web UI → Export → YOLO）

# 6. 划分数据集
python ctts.py data split dataset/LabelData/ -v 0.2

# 7. 训练
python ctts.py train

# 8. 推理验证
python ctts.py detect --video test.mp4 -w weights/best.pt
```

## 依赖

| 包 | 用途 |
|----|------|
| opencv-python | 视频/图像处理 |
| numpy | 数值计算 |
| ultralytics | YOLO 训练与推理 |
| PySide6 | 图形界面 |
| requests | Label Studio API |

通过 `uv sync` 一键安装全部依赖。

## 自定义训练参数

编辑 `tools/yolo_train.py` 修改训练配置，或在 GUI 训练 Tab 中调整参数后启动。
