# CTTS 小模型预标注（Prelabel）功能设计 · v2

| 项 | 内容 |
|---|---|
| 状态 | 设计稿 v2 · 2026-10-20（替代 v1） |
| 目标仓库 | `ctts` |
| 核心变化 | 预标注页签升级为「评估 → 建集 → 训练 → 预标注 → 复核」的引导式流水线；新增数据就绪评估与自动化训练；推理统一走 ONNX |
| 新增模块 | `tools/prelabel.py` |
| 改动模块 | `tools/ctts.py`、`tools/label_studio_auth.py`、`tools/label_studio_data.py`、`tools/label_studio_project.py`、`README.md` |
| 依赖变化 | 无（`ultralytics` / `onnx` 导出能力由 ultralytics 自带，`httpx` / `opencv` / `numpy` 已在 `pyproject.toml`） |

---

## 1. 目标与流程变化

### 1.1 最终目标（用户确认的流程）

1. 先人工标注**一小部分**数据（冷启动）；
2. 进入「预标注」页签；
3. 工具通过 Label Studio API 判断这些标注**是否足以训练 nano 模型**；
4. 足够 → 自动构建**微数据集**并训练 `yolo26n`，再导出 ONNX；
5. 调用 LS API 对**未标注任务**批量预标注；
6. 人工在 Label Studio 中复核、修正、提交；
7. 修正后的标注回流，支撑下一轮。

### 1.2 与 v1 的差异

| 维度 | v1 | v2 |
|---|---|---|
| 页签定位 | 参数面板：选权重、跑推理 | **流水线驾驶舱**：评估、训练、推理、复核一条线 |
| 模型来源 | 用户手动选权重 | 默认由本页签**自动训练产出**；也允许复用已有 ONNX |
| 数据判断 | 无 | **就绪评估**（任务数、实例数、类别覆盖），带结论与建议 |
| 训练 | 手动去「训练」页签 | 数据足够后**自动建集 + 训练 + 校验 + 导出 ONNX** |
| 进度呈现 | 原始日志流全量滚动 | **三级披露**：阶段状态 → 关键指标 → 可展开明细/日志文件 |
| 复核 | 一句提示 | 待复核计数、一键打开 LS、导出审计抽样清单 |

### 1.3 非目标

- 不自动把 prediction 转成 annotation（人工提交仍是唯一数据闸门）。
- 不做标注质量自动打分、不做多标注员投票合并（取最新一条 annotation）。
- 不做分布式训练 / 超参搜索；训练参数走固定小配置 + 少量可调项。
- 不接管 Label Studio 的账号、项目、存储管理。

---

## 2. 端到端流程

### 2.1 阶段状态机

```
 ┌──────────┐   不足(红)   ┌───────────────────────┐
 │ 1 评估    │────────────▶│ 停止，给出补标建议       │
 │ assess   │             └───────────────────────┘
 └────┬─────┘
      │ 可训练(绿) / 勉强(橙，需确认)
      ▼
 ┌──────────┐   失败      ┌───────────────────────┐
 │ 2 建集    │────────────▶│ 停止，保留 run 目录      │
 │ dataset  │             └───────────────────────┘
 └────┬─────┘
      ▼
 ┌──────────┐   OOM/异常   ┌───────────────────────┐
 │ 3 训练    │────────────▶│ 降 batch 重试一次 → 仍失败停止 │
 │ train    │             └───────────────────────┘
 └────┬─────┘
      ▼ 校验通过
 ┌──────────┐
 │ 4 导出    │  best.pt → best.onnx + manifest.json
 │ export   │
 └────┬─────┘
      ▼
 ┌──────────┐   全部失败   ┌───────────────────────┐
 │ 5 预标注  │────────────▶│ 报告失败原因，不写入      │
 │ prelabel │             └───────────────────────┘
 └────┬─────┘
      ▼ 写入成功
 ┌──────────┐
 │ 6 人工复核 │  在 Label Studio 中确认/修正/提交
 └────┬─────┘
      ▼ 提交后的标注回流
   回到 1（版本 +1）
```

### 2.2 阶段输入 / 输出 / 闸门

| 阶段 | 输入 | 输出 | 闸门 |
|---|---|---|---|
| 1 评估 | 项目 ID、`label_config`、已标注任务 | 就绪结论 + 明细指标 | 不足则终止；勉强需用户确认 |
| 2 建集 | 已标注任务及其图片 | `dataset/{images,labels,train,val,data.yaml}` | 成败任务数、图片下载失败率 ≤10% |
| 3 训练 | 微数据集、`yolo26n.pt` | `train/exp/weights/best.pt`、指标 | train ≥10 张、val ≥2 张，否则中止 |
| 4 导出 | best.pt | `model/best.onnx` + `manifest.json` | ONNX 可加载并输出正确形状 |
| 5 预标注 | ONNX、未标注任务 | LS predictions（含 `model_version`） | 类别与 `label_config` 有交集 |
| 6 复核 | 预标注结果 | 人工提交的 annotations | 人工完成，工具不代劳 |

### 2.3 用户操作（最短路径）

1. 在「② 标注管理」里标 30～100 张并 Submit；
2. 切到「③ 预标注」，项目 ID 自动带入，点 **「评估并预标注」**；
3. 若提示数据不足：按建议补标后重试；
4. 流水线跑完后查看摘要弹窗，点 **「打开 Label Studio」** 复核；
5. 提交后回到本页签重跑（自动 `v2`），继续预标注剩余图片。

---

## 3. 数据就绪评估

### 3.1 数据来源（只信 LS API）

```python
project = client.projects.get(project_id)      # 计数器 + label_config
tasks   = client.tasks.list(
    project=project_id,
    only_annotated=True,                       # 只要已标注任务，减少翻页
    fields="all",                              # 需要 annotations 明细
    page_size=200,
)
```

- 项目级计数器（`num_tasks_with_annotations`、`total_annotations_number`）只用于**快速预览**，不用于判定，因为不能区分类别与实例。
- 判定以遍历 annotation 明细为准；只遍历已标注任务，10k 任务量级也可接受。
- token 必须是 admin/owner，annotator token 会被裁剪任务可见性。

### 3.2 采集指标

对每个已标注任务，取**最新一条 annotation**（`max(updated_at or created_at)`；多标注员暂不合并），遍历 `result` 中 `type == "rectanglelabels"` 的项：

| 指标 | 计算 | 用途 |
|---|---|---|
| `labeled_tasks` | 有非空 annotation 的任务数 | 主门槛 |
| `instances` | 有效矩形总数 | 主门槛 |
| `per_class` | 按 `value.rectanglelabels[0]` 计数 | 类别覆盖 |
| `invalid_boxes` | 坐标为 0/负/超 100，或宽高 ≤0 | 数据质量告警 |
| `negative_tasks` | annotation 存在但矩形为空 | 负样本计数（信息项） |
| `small_ratio` | 面积 `w%*h% / 10000 < 0.5%` 的占比 | stb 小目标信息项，不设门槛 |
| `annotators` | 出现的 `completed_by` 去重数 | 信息项 |
| `boxes_per_task` | 中位数 | 信息项 |

### 3.3 门槛策略（默认值可调）

| 检查 | 可训练（绿） | 勉强（橙） | 说明 |
|---|---|---|---|
| 已标注任务 | ≥ 50 | ≥ 25 | 少于 25 张时 val 不足 5 张，指标不可信 |
| 总实例数 | ≥ 100 | ≥ 50 | nano 微调至少需要百级目标框 |
| 每类实例数 | ≥ 30 | ≥ 15 | 单类少于 15 个框基本学不会 |
| 类别覆盖 | `label_config` 中每类 ≥1 | 同左 | 某类为 0 = 橙：可训练但该类学不到，需明确告知 |
| 无效框占比 | ≤ 5% | ≤ 15% | 超阈值提示先修标注，不硬阻断 |

> 数字是策略默认值，按「能出可用预标注」的经验定，不做性能承诺；UI「高级」里可改，改动记录进 manifest。

### 3.4 判定逻辑

```python
def verdict(a: Assessment) -> str:
    hard_ok = (
        a.labeled_tasks >= a.min_labeled * 0.5
        and a.instances >= a.min_instances * 0.5
        and all(n >= a.min_per_class * 0.5 for n in a.per_class.values() if n > 0)
    )
    if not hard_ok:
        return "insufficient"
    if (a.labeled_tasks >= a.min_labeled
            and a.instances >= a.min_instances
            and all(n >= a.min_per_class for n in a.per_class.values() if n > 0)):
        return "ready"
    return "marginal"
```

- `insufficient`：按钮置灰训练阶段，摘要给出**补什么、补多少**：
  「还差 18 张已标注图 / 还差 42 个 `stb` 框；建议优先标遮挡、密集、远距离帧。」
- `marginal`：弹确认框「数据偏少，训练可能不稳定，是否继续？」；继续则正常跑，manifest 标记 `marginal: true`。
- `ready`：直接进入建集。

### 3.5 输出示例

```json
{
  "project": 10,
  "verdict": "ready",
  "labeled_tasks": 62,
  "instances": 138,
  "per_class": {"stb": 138},
  "invalid_boxes": 2,
  "negative_tasks": 3,
  "small_ratio": 0.64,
  "checks": [
    {"item": "已标注任务", "require": ">=50", "actual": 62, "level": "pass"},
    {"item": "实例总数", "require": ">=100", "actual": 138, "level": "pass"},
    {"item": "stb 实例", "require": ">=30", "actual": 138, "level": "pass"},
    {"item": "无效框占比", "require": "<=5%", "actual": "1.4%", "level": "warn"}
  ],
  "suggestion": null
}
```

---

## 4. 微数据集构建

### 4.1 产物目录

```
runs/prelabel/20261020-1810_v1/
├── dataset/
│   ├── images/          # 已标注任务的图片（task id 命名）
│   ├── labels/          # 同名 .txt
│   ├── classes.txt      # 顺序 = label_config 中 Label 顺序
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   └── data.yaml
├── train/exp/weights/{best.pt,last.pt}
├── model/{best.onnx, manifest.json}
├── report.json
├── run.log
└── review_sample.csv
```

### 4.2 标注 → YOLO：直接透传百分比

LS 的矩形坐标本身就是**相对整图的百分比**，YOLO 标签也是相对整图的归一化值，因此**不需要解码图片尺寸**，避免 EXIF/缩放带来的偏移：

```python
cx = (value["x"] + value["width"] / 2) / 100
cy = (value["y"] + value["height"] / 2) / 100
w  = value["width"] / 100
h  = value["height"] / 100
# clamp 到 [0,1]，w/h <= 0 或置信度无效的框丢弃
```

- class id = `label_config.labels.index(label)`，顺序与 `classes.txt`、`data.yaml` 严格一致。
- 无效框（坐标越界、宽高为 0）丢弃并计数；占比 >15% 时告警。
- 空 annotation 的任务写**空 txt**，作为负样本保留（减少误检）。
- 命名用 `task_id`，天然去重；扩展名按响应 `Content-Type` 推断。

### 4.3 图片获取

- 优先级：`--images-dir` 本地同名文件 → 共享缓存 `~/.cache/ctts-prelabel/images/<sha1(url)[:16]>.<ext>` → HTTP 下载（只对 LS 同源 URL 带鉴权头）。
- 下载失败计入 `failures`；失败率 >10% 中止并提示「检查网络 / 用 --images-dir 映射」。
- 构建与预标注共用同一缓存，避免二次下载。

### 4.4 划分

复用现有 `split_dataset(dataset_dir, val_ratio=0.2, seed=42)`：

- 输入 `<run>/dataset`，输出 `train/`、`val/`、`data.yaml`；
- 划分前校验：`images` 与 `labels` 一一对应，孤儿文件告警；
- 划分后：`train/images ≥ 10` 且 `val/images ≥ 2`，否则中止并提示补标；
- 默认不设固定 holdout（主动学习全量回流）。若需要客观对比，可用 `--holdout 0.1` 锁定 10% 永不入训（M4 可选）。

---

## 5. 自动化训练与 ONNX 导出

### 5.1 配置

| 参数 | 默认 | 理由 |
|---|---|---|
| 基座 | `weights/yolo26n.pt`（缺失时 ultralytics 自动下载 `yolo26n.pt`） | 与现有「训练」「推理」页签一致 |
| `imgsz` | 480 | 小目标召回与 CPU/GPU 速度折中 |
| `epochs` | 150 | 微数据集足够收敛，配合早停不会空转 |
| `freeze` | 8 | 沿用 `yolo_train.py` 的低层冻结策略 |
| `batch` | GPU 16 / CPU 8；OOM 自动减半重试一次 | 适配标注机与训练机 |
| `workers` | GPU 8 / CPU 2 | 避免小机器卡死 |
| `cos_lr` / `patience` | True / 20 | 微调常规设置 |
| `amp` | GPU 开，CPU 关 | CPU 上 AMP 无收益且易报错 |
| `device` | 自动探测 CUDA，否则 CPU | 不静默变慢：UI 明确显示当前设备 |

### 5.2 训练回调与进度

```python
model = YOLO(base_weights)
model.add_callback("on_train_epoch_end", on_epoch)   # 取 trainer.epoch / trainer.metrics
model.train(data=data_yaml, epochs=150, imgsz=480, freeze=8,
            batch=batch, device=device, workers=workers, amp=amp,
            cos_lr=True, patience=20, verbose=False,
            project=str(run_dir / "train"), name="exp", exist_ok=True)
```

- `on_epoch` 只在 **每 10 个 epoch**（以及第 1 个、最后 1 个）发一次事件，避免日志刷屏；
- 训练 stdout 全量重定向到 `run.log`，不进入 UI 日志；
- 训练结束后显式 `model.val(data=..., split="val")`，记录 `mAP50 / mAP50-95 / precision / recall`。

### 5.3 ONNX 导出

```python
onnx_path = YOLO(best_pt).export(format="onnx", imgsz=480,
                                 dynamic=False, simplify=True)
```

- 导出后立即用 `YOLO(onnx_path)` 对一张验证图跑一次，确认输出非空、类别数匹配；
- ONNX 作为预标注默认推理后端（CPU 友好、部署简单）；`.pt` 仍可作为高级选项。

### 5.4 模型清单与版本

```json
{
  "version": "v1",
  "run_id": "20261020-1810",
  "base": "yolo26n.pt",
  "weights": "train/exp/weights/best.pt",
  "onnx": "model/best.onnx",
  "imgsz": 480,
  "classes": ["stb"],
  "dataset": {"labeled_tasks": 62, "instances": 138, "train": 50, "val": 12},
  "metrics": {"mAP50": 0.87, "mAP50_95": 0.55, "precision": 0.82, "recall": 0.79},
  "marginal": false,
  "tag": "ctts-prelabel:v1@480",
  "ls_project": 10,
  "created_at": "2026-10-20T18:10:00+08:00"
}
```

- 版本号 = 扫描 `runs/prelabel/*/model/manifest.json` 后自增（`v1 → v2`）；
- tag 固定 `ctts-prelabel:<version>@<imgsz>`，**不带日期**，保证可跳过/可覆盖；
- 若本次跳过训练（复用旧模型），tag 从该模型 manifest 读取，不重新生成。

### 5.5 失败处理

| 情况 | 行为 |
|---|---|
| CUDA OOM | `batch //= 2` 重试一次；再失败则中止并保留目录 |
| 数据 yaml 缺类 / 类别数不符 | 建集阶段已校验，不进入训练 |
| 训练中断（用户取消 / 进程退出） | 保留 `last.pt` 与 `run.log`；UI 提示可重跑 |
| 导出失败 | 训练结果保留，提示用 `best.pt` 手动导出；不进入预标注 |
| val mAP50 < 0.3 | 不阻断，但摘要标橙并提示「建议补标后再预标注」 |

---

## 6. 预标注

### 6.1 模型选择

默认「自动：最新 ONNX」= 本次训练产物；下拉可选历史 `runs/prelabel/*/model/best.onnx`，或手选文件。模型清单缺失时退回 `weights/*.onnx`，再退回 `weights/*.pt`。

### 6.2 目标任务与跳过规则

- 目标：`is_labeled == false` 的任务（默认）；
- 跳过：已有**同 tag** prediction 的任务；
- `--overwrite`：先删同 tag 旧预测再写（换模型刷新时用）；
- 其他 tag 的 prediction 允许并存，Data Manager 可按 `model_version` 过滤；
- 空检测不写 prediction，计入 `empty_tasks`。

### 6.3 写入

```python
client.projects.import_predictions(
    project_id,
    request=[PredictionRequest(task=t.id, result=results,
                               score=mean_conf, model_version=tag)
             for t in batch],
)
```

- 每批 50 条；接口报错则降级逐条 `client.predictions.create(...)`，失败任务进 `failures`，不中断整轮；
- 写前校验：模型类名与 `label_config.labels` 至少一个交集，否则中止并打印两侧集合；
- 坐标转换沿用 v1：百分比、`rotation=0`、带 `original_width/height`、clamp、去掉 <2px 的框；
- 统计：处理/跳过/失败/生成框数/空图/吞吐，写入 `report.json`。

### 6.4 版本幂等

| 场景 | 行为 |
|---|---|
| 第一次跑 v1 | 写 `ctts-prelabel:v1@480` |
| 重跑 v1 | 全部跳过（同 tag 已存在） |
| 训练出 v2 再跑 | 新 tag 写入；v1 预测保留，可对比 |
| 想刷新 v1 | `--overwrite`：删同 tag 再写 |

---

## 7. 人工复核

### 7.1 复核面板（页签内 L2 信息）

- **待复核** = 有 `tag` prediction 且无 annotation 的任务数；
- **已提交** = 复核窗口内新增 annotation 数；
- 按钮：`打开 Label Studio`（`{base_url}/projects/{pid}`）、`导出抽样清单`（默认 10% 随机 + 固定种子 CSV）。

### 7.2 复核规则（写进 UI 提示，不写进任务指令）

1. 逐条确认预测框，补漏框、删误检、改类别；
2. 不要「全选接受」——未修改直接提交的任务会被抽样审计；
3. Submit 之后数据才进入训练集；
4. 预标注只提供草稿，不改变 `is_labeled`。

### 7.3 闭环

```
复核提交 → 项目计数增加 → 重新点「评估并预标注」
        → 若就绪：自动 v2 训练 → 对剩余未标注任务预标注
```

- 每轮新增确认量建议 30～100 张；
- 若 v2 的 val 指标没有提升，保留 v1 继续用，避免版本退化；
- 抽样审计结果用于调整标注规范，不自动改数据。

---

## 8. UI 设计

### 8.1 页签结构（③ 预标注）

```
┌ ① 数据就绪 ───────────────────────────────────────────────┐
│ 项目 ID [10]  [检查数据]        已标注 62 / 总共 300 张      │
│ 判定: ● 可训练                                   [查看明细 ▸] │
│ 摘要: 62 张 · 138 个框 · 1/1 类 · 小目标 64% · 无效框 2      │
│ ┌ 明细表（折叠）：检查项 | 门槛 | 实测 | 结论 ─────────────┐ │
│ └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
┌ ② 微数据集与模型 ─────────────────────────────────────────┐
│ 基座 [yolo26n.pt] 尺寸 [480] 轮数 [150] 冻结 [8] 设备 [自动] │
│ 上次产物: runs/prelabel/20261020-1810_v1 · mAP50 0.87        │
│ 数据: train 50 / val 12 · 类别: stb          [打开目录]      │
└──────────────────────────────────────────────────────────────┘
┌ ③ 预标注 ─────────────────────────────────────────────────┐
│ 模型 [自动：最新 ONNX ▼]  置信度 [0.25]  IOU [0.70]          │
│ 目标: 未标注任务   [x] 跳过同版本预测   [ ] 覆盖同版本        │
│ 版本标签: ctts-prelabel:v1@480                                │
│ 预计: 待处理 238 张 / 跳过 0                                  │
└──────────────────────────────────────────────────────────────┘
┌ ④ 复核 ───────────────────────────────────────────────────┐
│ 待复核 238 张 · 已提交 0 张      [打开 Label Studio] [导出抽样] │
└──────────────────────────────────────────────────────────────┘

  ○ 评估   ○ 建集   ○ 训练   ○ 导出   ○ 预标注
  [ 评估并预标注 ]   [ 仅评估 ]   [ 高级 ▾ ]
```

### 8.2 三级信息披露（「适度披露」的落地规则）

| 层级 | 内容 | 何时出现 |
|---|---|---|
| L1 阶段状态 | 5 个阶段芯片 `○ 待执行 / ◐ 进行中 / ● 完成 / ✕ 失败`，当前阶段一行状态 | 始终 |
| L2 关键指标 | 评估判定与 5 个数字；训练完成后的 mAP50、训练/验证张数；预标注处理/跳过/失败/框数 | 阶段完成时 |
| L3 明细 | 评估明细表、逐 epoch 指标、完整训练日志、失败任务清单 | 点「查看明细」或打开 `run.log` / `report.json` |

硬规则：

- UI 日志区**不逐行流式打印**训练/推理原始输出；只发阶段开始、每 10 epoch、阶段完成、错误四类事件；
- 任何失败必须给出**一句可执行的下一步**（如「还差 18 张 → 去 ② 补标」），不贴 traceback（traceback 进 `run.log`）；
- 进度条：评估/导出不确定态；建集/训练/预标注按任务数或 epoch 确定态；
- 弹窗只用于**决策点**（勉强是否继续、错误、完成摘要），不用于过程播报。

### 8.3 统一 UI 风格

新增页签必须与现有 ①～⑤ 页签同一套观感，不做独立主题：

| 维度 | 规范 |
|---|---|
| 布局 | `QVBoxLayout` + 若干 `QGroupBox`；组内 `QFormLayout`；末尾 `addStretch()` |
| 按钮 | 主操作 `font-size: 14px; padding: 8px`（沿用现有「开始抽帧/训练」）；普通按钮默认；危险操作用红色 |
| 状态色 | 定义常量 `OK/WARN/ERR/MUTED`（绿/橙/红/灰），② 与 ③ 共用同一组，不再各写各的 |
| 表格 | `QTableWidget` + 只读 + 整行选择 + 最后一列 Stretch（与项目/任务表一致） |
| 字体 | 沿用 `app.setFont` 的平台字体；不单独设 family |
| 日志/进度 | 复用底部共享日志与顶部共享进度条，不新建第二套 |
| 文案 | 中文、短句、动词开头；指标用「62 张 · 138 个框」，不用长句 |
| 窗口 | 保持 960×700，新增内容通过折叠区控制高度，不撑大窗口 |

实现上在 `ctts.py` 增加小工具：

```python
PRIMARY_BTN_QSS = "font-size: 14px; padding: 8px;"
MUTED_QSS = "color: gray;"
STATUS_COLORS = {"ok": "green", "warn": "orange", "err": "red", "idle": "gray"}

def _set_status(label: QLabel, text: str, level: str = "idle") -> None: ...
```

② 标注管理里现有的 `setStyleSheet("color: green;")` 等调用顺手替换为 `_set_status`，保证两页签颜色语义一致。

### 8.4 交互细节

- 未连接 LS 时：整个 ③ 只显示一行「请先在 ② 标注管理连接」，主按钮禁用；
- 项目 ID 与 ② 的「任务管理」项目联动；在 ③ 改 ID 后，评估结果自动失效（判定回到「未评估」）；
- 「评估并预标注」运行期间：输入控件锁定、按钮变为「运行中…」，可点「取消」置位停止标志（阶段之间检查，训练靠回调抛出）；
- 训练完成但预标注失败：阶段芯片停在「预标注 ✕」，可只重跑预标注（模型已存在）；
- 复用模型时跳过建集/训练/导出，芯片直接置灰；
- 「高级」折叠区放：门槛覆盖、`--holdout`、`--limit`、设备、日志级别；默认收起。

### 8.5 事件 → UI 映射

| 事件 | 芯片 | 进度条 | 日志区一行 |
|---|---|---|---|
| `assess.start` | 1 ◐ | 不确定 | `评估数据…` |
| `assess.done` | 1 ●/✕ | 隐藏 | `评估完成：可训练（62 张 / 138 框）` |
| `dataset.progress` | 2 ◐ | i/n | 不打印 |
| `dataset.done` | 2 ● | 隐藏 | `微数据集就绪：train 50 / val 12` |
| `train.epoch` | 3 ◐ | epoch/n | 每 10 轮一行 `epoch 40/150 · mAP50 0.71` |
| `train.done` | 3 ● | 100% | `训练完成：mAP50 0.87` |
| `export.done` | 4 ● | 隐藏 | `ONNX 已导出` |
| `prelabel.progress` | 5 ◐ | i/n | 不打印 |
| `prelabel.done` | 5 ● | 隐藏 | `预标注完成：写入 238 张 / 失败 0` |
| `stage.error` | 当前 ✕ | 隐藏 | `失败：<一句原因>` + 详情按钮 |

---

## 9. 技术设计

### 9.1 模块划分

| 文件 | 改动 | 职责 |
|---|---|---|
| `tools/prelabel.py` | **新增**（约 600 行） | 评估、建集、训练、导出、预标注、事件、CLI |
| `tools/ctts.py` | 新增 Tab（约 200 行）+ CLI 转发 | ③ 预标注页面、状态芯片、复用连接/日志/进度 |
| `tools/label_studio_auth.py` | +10 行 | 公开 `auth_headers()` |
| `tools/label_studio_data.py` | +40 行 | `iter_annotated_tasks()`、`annotation_stats()` |
| `tools/label_studio_project.py` | +30 行 | `detection_label_config(labels)`，建项目时带真实类别 |
| `README.md` | +30 行 | 新流程、命令、目录说明 |
| `tools/test_prelabel.py` | 新增 | 纯逻辑单测（无 LS / 无 GPU） |

### 9.2 核心数据结构

```python
@dataclass
class Check:
    item: str            # "已标注任务"
    require: str         # ">=50"
    actual: object       # 62
    level: str           # pass | warn | fail

@dataclass
class Assessment:
    project: int
    verdict: str                     # ready | marginal | insufficient
    labeled_tasks: int
    instances: int
    per_class: dict[str, int]
    invalid_boxes: int
    negative_tasks: int
    small_ratio: float
    checks: list[Check]
    suggestion: str | None = None
    policy: dict = field(default_factory=dict)

@dataclass
class DatasetInfo:
    root: Path
    classes: list[str]
    train: int
    val: int
    skipped: int
    failures: list[str]

@dataclass
class TrainResult:
    run_dir: Path
    best_pt: Path
    onnx: Path
    metrics: dict
    manifest: dict

@dataclass
class PrelabelStats:
    processed: int
    skipped: int
    failed: int
    boxes: int
    empty: int
    elapsed_sec: float
    failures: list[dict]

@dataclass
class StageEvent:
    stage: str       # assess | dataset | train | export | prelabel
    status: str      # start | progress | done | error
    message: str
    fraction: float | None = None
    visible: bool = True
```

### 9.3 函数签名

```python
# tools/prelabel.py
def parse_label_config(label_config: str) -> LabelConfig: ...
def assess_project(project_id: int, *, client=None,
                   min_labeled: int = 50, min_instances: int = 100,
                   min_per_class: int = 30,
                   log: Callable[[str], None] | None = None) -> Assessment: ...

def build_micro_dataset(project_id: int, *, out_dir: str, client=None,
                        split_ratio: float = 0.2, seed: int = 42,
                        images_dir: str | None = None,
                        on_event: Callable[[StageEvent], None] | None = None) -> DatasetInfo: ...

def train_micro(dataset: DatasetInfo, *, base_weights: str,
                epochs: int = 150, imgsz: int = 480, freeze: int = 8,
                device: str = "auto", batch: int | None = None,
                run_dir: str | None = None,
                on_event: Callable[[StageEvent], None] | None = None) -> TrainResult: ...

def export_onnx(result: TrainResult, *, imgsz: int = 480) -> Path: ...
def prelabel_project(project_id: int, *, model: str, tag: str | None = None,
                     conf: float = 0.25, iou: float = 0.7, imgsz: int = 480,
                     overwrite: bool = False, limit: int = 0,
                     client=None,
                     on_event: Callable[[StageEvent], None] | None = None) -> PrelabelStats: ...
def run_pipeline(project_id: int, *, base_weights: str = "weights/yolo26n.pt",
                 epochs: int = 150, imgsz: int = 480,
                 conf: float = 0.25, iou: float = 0.7,
                 reuse_model: str | None = None,
                 skip_train: bool = False,
                 on_event: Callable[[StageEvent], None] | None = None) -> dict: ...
```

CLI 子命令（`ctts.py prelabel ...` 转发）：

```bash
python ctts.py prelabel assess -p 10            # 只评估，打印检查表
python ctts.py prelabel run -p 10               # 评估→建集→训练→导出→预标注
python ctts.py prelabel run -p 10 --skip-train --model runs/prelabel/.../best.onnx
python ctts.py prelabel predict -p 10 --model runs/prelabel/.../best.onnx
python ctts.py prelabel review -p 10 --tag ctts-prelabel:v1@480 --sample 0.1
```

### 9.4 SDK 调用清单（2.1.2 已核实）

| 用途 | 调用 |
|---|---|
| 项目计数器 / label_config | `client.projects.get(pid)` |
| 已标注任务明细 | `client.tasks.list(project=pid, only_annotated=True, fields="all", page_size=200)` |
| 待预标注任务 | `client.tasks.list(project=pid, fields="all", page_size=200)`，本地过滤 |
| 批量写预测 | `client.projects.import_predictions(pid, request=[PredictionRequest(...)])` |
| 单条降级 | `client.predictions.create(task=..., result=..., score=..., model_version=...)` |
| 删除同 tag 预测 | `client.predictions.list(project=pid)` + `client.predictions.delete(id)` |
| 鉴权头 | `client._headers()` → 包装为 `auth_headers()` |
| 任务数/已标注数 | `project.task_number` / `project.num_tasks_with_annotations` |

### 9.5 其他关键点

- **图片鉴权**：只对 `url.startswith(base_url)` 带 `Authorization`；S3 预签名 URL 不能带，否则 400。
- **EXIF**：建集用百分比透传，无需解码；预标注需要实际尺寸，用 `cv2.imread` 后的 shape，并记录 EXIF 旋转告警。
- **类别预检**：建集写入 classes.txt 前、预标注写库前，各校验一次类别集合。
- **并发**：图片下载用 8 线程预取；推理与写库串行，避免 LS 压力。
- **缓存**：图片按 URL sha1 命名；`--clear-cache` 清理。
- **报告**：`report.json` 汇总评估、建集、训练、导出、预标注五段，供事后审计。
- **CPU 回退**：GPU 不可用时训练与预标注均自动 CPU，并在 L1 状态行显示「设备: CPU」。

---

## 10. 边界与错误处理

| 情况 | 处理 |
|---|---|
| 未连接 LS / token 无效 | ③ 顶部提示，按钮禁用，不进入流水线 |
| 项目不是检测模板 | 报错「缺少 RectangleLabels」，给出建项目命令 |
| 无已标注任务 | `insufficient`，建议「先标 20 张再来」 |
| 某配置类别 0 实例 | 橙：告警该类别学不到；训练仍可继续 |
| `label_config` 与数据类别不一致 | 预检中止，打印两侧集合 |
| 图片 404 / 401 / 损坏 | 计入 failures；>10% 中止建集 |
| 远程 LS 返回 `127.0.0.1` 图片 URL | 提示用 `--images-dir` 或修 `LABEL_STUDIO_HOST` |
| CPU 训练过慢 | L1 显示设备与预计；可改小 `epochs`/`imgsz`（高级） |
| OOM | batch 减半重试一次 |
| 训练取消 | 保留 `last.pt`、`run.log`，不写 manifest |
| ONNX 加载失败 | 不进入预标注；提示用 `best.pt` 手动导出 |
| 同 tag 预测已存在 | 默认跳过；`--overwrite` 先删后写 |
| 写入 400（类别/字段） | 降级逐条；仍失败则报告具体 task 与错误 |
| 大批量（>5k 任务） | 分页、`--limit`、每批 50；`--overwrite` 成本高，建议换 tag |
| 中断 / Ctrl-C | 已完成阶段保留；`report.json` 标记 `interrupted: true` |

---

## 11. 验收与测试

### 11.1 功能验收

1. **评估准确**：构造 40 张已标注、每张 2 框的项目 → `marginal`；补到 60 张 → `ready`；删到 10 张 → `insufficient`。
2. **不足可解释**：摘要明确给出差多少张/多少个框，且点击后能跳到 ② 补标。
3. **一键闭环**：60 张已标注 + 100 张未标注 → 一次点击后：建集、训练、导出 ONNX、写入 100 条 prediction，弹窗数字与 LS 一致。
4. **跳过幂等**：立即重跑，全部任务被跳过，prediction 总数不变。
5. **版本递增**：再提交 20 张后重跑 → 新 run 为 `v2`，tag 变化，v1 预测保留。
6. **复用模型**：`--skip-train --model ...` 只跑预标注，不建集不训练。
7. **坐标正确**：1920×1080 已知框，LS 渲染偏差 <0.1%。
8. **异常隔离**：删掉 1 张图，建集只多 1 个 failure，其余照常。
9. **UI 一致性**：组框、表单、按钮、表格、日志位置与 ② 页签一致；训练过程日志不刷屏。
10. **取消**：训练中取消，进程在 10 秒内停下并保留产物。

### 11.2 测试计划

- 单元测试（`tools/test_prelabel.py`，标准库 `unittest`，无网络无 GPU）：
  - 百分比 → YOLO 归一化（含 clamp、退化框）；
  - 最新 annotation 选择；
  - 就绪判定（ready/marginal/insufficient 边界值）；
  - 版本号自增与 tag 生成；
  - 事件节流（epoch 每 10 次、任务每 20 次）。
- 集成测试（本机 LS + 20 张图 + 已有权重）：建集 20 张、训练 2 epoch、导出 ONNX、预标注 5 张。
- 离线回归：固定图片与 annotation JSON，建集产物逐行比对。
- 性能基线：记录 CPU 下训练 1 epoch 与 ONNX 推理 100 张的耗时，写入 README。

---

## 12. 实施计划

| 里程碑 | 内容 | 预计 |
|---|---|---|
| **M1 评估与建集** | `prelabel.py` 骨架、`parse_label_config`、`assess_project`、`build_micro_dataset`、单测；`ctts.py prelabel` 转发 | 1.5 天 |
| **M2 训练与导出** | `train_micro`、回调节流、校验、ONNX 导出、manifest、版本自增、`run.log` | 1.5 天 |
| **M3 预标注** | 任务筛选、ONNX 推理、批量/降级写入、跳过/覆盖、`report.json` | 1 天 |
| **M4 流水线与 GUI** | `run_pipeline`、事件模型、③ 页签、三级披露、复核面板、统一状态色 | 1.5 天 |
| **M5 打磨（可选）** | 审计抽样导出、holdout、图片预取并发、README 文档 | 0.5 天 |

合计约 **6 人日**。M1+M3 先通（可绕过训练用已有权重验证预标注），M2 与 M4 可并行。

---

## 13. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 训练数据太少导致模型无用 | 预标注全是漏检/误检 | 门槛 + 橙/红判定 + 补标建议；mAP50 <0.3 时标橙 |
| 预标注污染训练集 | 模型学习自己的错误 | 只写 prediction；人工 Submit 才入集；抽样审计 |
| 训练/推理耗时不可控 | 标注机卡住 | 默认 480/nano；分阶段取消；L1 显示设备与进度 |
| ONNX 与 PT 结果不一致 | 预标注框偏移/漏检 | 导出后自检一次；保留 `.pt` 选项对比 |
| 微数据集类别顺序错位 | 训练全错 | 统一以 `label_config` 顺序生成 classes.txt/data.yaml，并做断言 |
| 主动学习无固定测试集 | 指标虚高 | 文档说明；可选 `--holdout` 锁定评估集 |
| 大项目翻页慢 | 评估/预标注耗时长 | `only_annotated`、分页 200、`--limit`、图片缓存 |
| LS 版本差异 | `import_predictions` 不可用 | 逐条 `predictions.create` 降级 |

---

## 14. 后续扩展

1. **ML Backend 实时预标注**：把 `prelabel_project` 的推理与转换抽成 FastAPI `/predict`，打开任务即出框。
2. **多标注员合并**：同一任务多条 annotation 做多数投票或 IoU 聚类，替代「取最新」。
3. **难例队列**：把空图、低置信、密集框任务导出为下一轮标注清单。
4. **INT8 量化 ONNX**：标注机 CPU 吞吐再提一档（需校准集）。
5. **固定评估集**：默认锁定 10% 作为跨轮次可比指标。
6. **训练可观测性**：把每轮 run 的 mAP、修正率、耗时汇总成项目级趋势表。

---

## 附录 A：评估输出（可直接落盘）

```json
{
  "project": 10,
  "verdict": "marginal",
  "labeled_tasks": 32,
  "instances": 74,
  "per_class": {"stb": 74},
  "invalid_boxes": 1,
  "negative_tasks": 2,
  "small_ratio": 0.58,
  "checks": [
    {"item": "已标注任务", "require": ">=50", "actual": 32, "level": "warn"},
    {"item": "实例总数", "require": ">=100", "actual": 74, "level": "warn"},
    {"item": "stb 实例", "require": ">=30", "actual": 74, "level": "pass"}
  ],
  "suggestion": "建议再标 18 张（优先遮挡/密集/远距离帧），或确认继续"
}
```

## 附录 B：`manifest.json`

```json
{
  "version": "v1",
  "run_id": "20261020-1810",
  "base": "weights/yolo26n.pt",
  "imgsz": 480,
  "classes": ["stb"],
  "dataset": {"labeled_tasks": 62, "instances": 138, "train": 50, "val": 12},
  "metrics": {"mAP50": 0.87, "mAP50_95": 0.55},
  "marginal": false,
  "tag": "ctts-prelabel:v1@480",
  "ls_project": 10
}
```

## 附录 C：默认参数速查

| 参数 | 默认 | 位置 |
|---|---|---|
| 就绪：已标注任务 | 50 | 评估 |
| 就绪：实例数 | 100 | 评估 |
| 就绪：每类实例 | 30 | 评估 |
| 微数据集划分 | 0.2 / seed 42 | 建集 |
| 基座 | `weights/yolo26n.pt` | 训练 |
| epochs / imgsz / freeze | 150 / 480 / 8 | 训练 |
| batch / workers | GPU 16/8，CPU 8/2 | 训练 |
| ONNX | dynamic=False, simplify=True | 导出 |
| conf / iou / max_det | 0.25 / 0.7 / 300 | 预标注 |
| 每批写入 | 50 | 预标注 |
| 版本标签 | `ctts-prelabel:<version>@<imgsz>` | 预标注 |
| 抽样审计 | 10% | 复核 |
| 日志节流 | 每 10 epoch / 每 20 任务 | 全局 |

---

## 落地检查清单

- [x] `tools/prelabel.py`：评估、建集、训练、导出、预标注、流水线
- [x] `tools/test_prelabel.py`（21 项离线单测）：就绪判定、坐标、版本、节流
- [x] `ctts.py`：`③ 预标注` Tab + `prelabel` 子命令转发
- [x] `label_studio_auth.py`：`auth_headers()`
- [x] 数据遍历复用 `label_studio_data.list_tasks`（评估逻辑集中在 `prelabel.py`，未新增包装）
- [x] `label_studio_project.py`：从 classes.txt 生成检测模板
- [x] 统一状态色常量，并同步 ② 页签
- [x] README：新流程、命令、目录结构
- [x] `tools/review.py` + `⑦ 复核` Tab：按批加载、逐任务勾选、框级子集、批量认可为 annotation
- [x] `tools/test_review.py`（4 项离线单测）：扫描过滤/最新版本、部分框、跳过已标注、失败收集
- [ ] 本机 E2E：30 张标注 → 训练 2 epoch → 预标注 → 复核（当前环境无 LS 服务/数据，未执行）
