# 关键工程决策日志

> 每条决策记录「为什么」，供后来者判断是否仍成立。破坏性接口变更必须留痕。

## D-001 仓名 `foray_ml`，包名与 CLI 保留 `ctts`

- 组织仓按**职责/生命周期**命名（`interfaces`/`vision`/`auto_aim`…），`ctts`（CNN Train Tools Set）是工具品牌，装不下未来新增的模型定义、数据集规范、导出契约与离线评估。
- 保留 `ctts` 作为 Python 包名与 console script：既有文档、肌肉记忆与外部引用零改动。
- 2026-10-23 由 owner（RyanZhangK）拍板。

## D-002 许可证 GPL-3.0-only

- 沿用原个人仓 CTTS 的 GPL-3.0-only。
- 组织其他仓当前**没有** LICENSE 文件；本仓单独声明，不改变其他仓。对外分发时注意 GPL 传染性。
- 2026-10-23 由 owner 拍板。

## D-003 迁移方式：保留原始提交历史

- 迁移前核查：34 个跟踪文件、无 >1 MB 文件、权重/数据集图片已在 `.gitignore` 中排除 → 无需 `git filter-repo`。
- 直接 `git push` 到组织仓，21 个提交原样保留；个人仓不删除，作为保底。

## D-004 `src/` 布局，`pyproject.toml` 置于仓根

- `tools/` 平铺 + 同目录 import 依赖 `sys.path` hack，不利于打包与测试。
- 改为 `src/ctts/` + `tests/` + `scripts/`，`pyproject.toml` 提供 `ctts` console script；`pythonpath = ["src"]` 供 pytest 使用。

## D-005 Python 下限从 3.13 下调到 3.11

- 原 `requires-python = ">=3.13"` 过新：队伍机器 Ubuntu 22.04 自带 3.10、Orin JetPack 为 3.10，torch 对 3.13 支持刚跟上。
- 3.11 兼顾新语法与装得上；CI 矩阵 3.11/3.12。

## D-006 `.foray-layer` 声明为 `T`（离线工具层）

- 本仓是训练/数据/导出工具，不属于 X2/L1–L5 实时链路，也不被任何实时仓代码依赖（`foray_vision` 消费的是导出产物）。
- 若误声明为 `L3`，CI 层级校验会把训练仓当作感知层，造成依赖方向误判。

## D-007 移除硬编码个人机器路径

- 迁移前 `yolo_train.py` 含 `/home/ryanz/...` 绝对路径（个人历史工程残留）。
- 改为仓根相对默认值 + `CTTS_WEIGHTS` / `CTTS_DATASET` 环境变量覆盖；CLI/GUI 参数优先级最高。

## D-008 运行时路径锚点

- `_PROJECT_ROOT` 由 `__file__` 上溯三级（`src/ctts/x.py` → 仓根），用于 `scripts/`、`dataset/`、`weight/` 等。
- GUI 启动 `detect` 子进程改用 `python -m ctts.detect`，并以仓根为 cwd、注入 `PYTHONPATH=src`。

## D-009 双流实验骨架放在本仓（不新建仓）

- **决策**：双流特征解耦的模型骨架与实验脚本放 `src/ctts/models/` + `experiments/dualstream/`，不新建 `foray_dualstream` 仓。
- **理由**：按 `foray_docs/repository_structure.md` §4「拆仓是因为痛了才拆」，双流是一个实验分支而非独立生命周期；仓级切分应在验证成立后进行（届时若需要独立产线再拆）。
- **门控初始化**：`GateFuse` 的门控权重置 0、偏置 -4（σ≈0.018），使训练初期输出≈亮度特征。目的是抑制「色度支路偷学图案」——这是双流方案最大的已知风险（见调研报告 §7）。
- **属性头**：颜色 4 类与图案 9 类**各自 sigmoid**（BCE），不用 softmax——YOLO 原生 head 的 softmax 强制互斥，是「两类特征竞争」的根源之一。
- **数据通路单一实现**：`models/color.py` 用 numpy 实现 BT.709 全范围 YUV，训练预处理与推理侧（`foray_vision`）都引用同一组系数，避免训练/部署色度域偏移。

## D-010 属性标签采用方案 A（`labels_attr/` 并行文件）

- **决策**（2026-10-24 拍板）：颜色 4 类与图案 9 类作为**属性**独立标注，不展开成 4×9 组合类。
  - `labels/` 保持标准 YOLO 5 字段，`ultralytics` 可直接使用，不破坏现有标注产物；
  - `labels_attr/` 按图同名存放 `<cls> <cx> <cy> <w> <h> <color> <pattern>`，颜色/图案写类别名，未知写 `-`。
- **理由**：组合类展开会让模型在 36 个互斥类上学 softmax，正是「颜色捷径压制图案」的根源；
  属性独立监督（sigmoid + BCE）让颜色与图案各自有独立梯度通路。
- **类别名来源**：与 `sp_vision_25/tasks/auto_aim/armor.hpp` 一致（COLORS 4 类、ARMOR_NAMES 9 类）。
- **图像级视图**：训练脚本取面积最大框的属性作为整图属性；多属性目标会告警，
  提示改用实例级标签或先裁剪。实例级数据通路（`AttrDataset(level="instance")`）已预留。
- **契约**：`experiments/dualstream/model_contract.yaml` 固化输入 layout、类别映射、阈值与导出参数；
  改契约属破坏性接口变更，须在此留痕并通知 `foray_vision`。
