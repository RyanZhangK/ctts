# 双流特征解耦实验

**要解决的问题**：CNN 在「兵种 × 队伍」组合识别时，灯条颜色特征与装甲板图案特征不能同时学好。

**假设**：颜色是低频信号、图案是高频信号；用两路不同分辨率 + 不同容量输入把它们在**数据通路层面**分开，再用门控融合，能缓解「颜色捷径压制图案」。

完整文献与方案矩阵见 [`foray_docs/dual_stream_disentanglement.md`](../../foray_docs/dual_stream_disentanglement.md)。

## 当前状态

骨架、数据通路与属性训练已就绪（标签格式采用**方案 A**，见下节）：

- `src/ctts/models/color.py` —— YUV 4:2:0 通路（numpy，训练/推理共享同一组系数）
- `src/ctts/models/dualstream.py` —— 非对称双主干 + 门控融合 + 属性双头（torch）
- `src/ctts/models/dataset.py` —— 属性数据集（`labels_attr/` 解析、图像级/实例级两种视图）
- `model_contract.yaml` —— 对下游推理仓的接口约定（输入 layout / 类别映射 / 导出）
- `tests/` —— 无 GPU 单测（torch 缺失时相关用例自动跳过）

## 运行

```bash
# 结构 / 梯度烟雾测试（无需数据与 GPU）
python3 experiments/dualstream/train_dualstream.py --dry-run

# 属性分类训练（数据集需含 labels_attr/）
python3 experiments/dualstream/train_dualstream.py --data /path/to/dataset --epochs 50

# 单测
PYTHONPATH=src python3 -m pytest tests/ -v
```

## 标签格式（方案 A）

颜色与图案是**属性**，不展开成 4×9 组合类（展开正是「颜色捷径压制图案」的根源）。

- `labels/` 仍是标准 YOLO 5 字段，`ultralytics` 可直接用，**不破坏现有标注流程**；
- `labels_attr/` 按图片同名存放属性：每行 `<cls> <cx> <cy> <w> <h> <color> <pattern>`，
  颜色/图案写类别名（`red`/`blue`/`extinguish`/`purple`、`one`…`not_armor`），未知写 `-`。

类别名与 `sp_vision_25/tasks/auto_aim/armor.hpp` 的枚举一致：颜色 4 类、图案 9 类。

训练采用**图像级**视图（取面积最大框的属性），适用于裁剪 patch / ROI 数据集；
一张图内出现多个不同属性目标时训练脚本会告警，此时应改用实例级标签或先裁剪。

## 实验矩阵（待跑）

| 编号 | 变体 | 目的 |
|---|---|---|
| E0 | 现有单流 YOLO + 拆分属性标签 | 基线：颜色/图案分开监督是否已足够 |
| E1 | 单流 4 通道（Y + CbCr 下采样，同分辨率） | 只加信息不加结构 |
| E2 | 双流 + 门控融合（本骨架默认） | 主假设 |
| E3 | 双流 + 朴素相加（无门控） | 门控是否必要 |
| E4 | 双流 + concat 融合 | 融合方式对照 |
| E5 | E2 去掉色度支路 | 色度支路是否真的有用 |

**判据**（三条同时看）：

1. 灯条颜色准确率不显著下降（允许 ≤1 个点）；
2. 图案准确率相对 E0 有可复现提升；
3. 色度支路对图案区域的响应不显著高于随机（检验「色度支路泄题」）。

## 已定 / 待定

已定：

- **属性标签格式 = 方案 A**（`labels_attr/` 并行文件，见上节）；
- **反事实增强**：R↔B 通道交换（不是 HSV 旋转），`color.swap_rb` 已实现；
- **导出契约**：`model_contract.yaml` 已给出输入 layout、类别映射、ONNX/TensorRT 约定。

待定：

- **域一致性**：训练与推理都必须走 `color.py` 的同一套 4:2:0 通路（`foray_vision` 侧待接）；
- **属性标注工具**：`labels_attr/` 目前需由标注脚本生成，尚未接入 Label Studio 导出；
- **类别不平衡**：extinguish / not_armor 样本可能偏少，需统计后再决定是否加权。
