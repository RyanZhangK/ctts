# 双流特征解耦实验

**要解决的问题**：CNN 在「兵种 × 队伍」组合识别时，灯条颜色特征与装甲板图案特征不能同时学好。

**假设**：颜色是低频信号、图案是高频信号；用两路不同分辨率 + 不同容量输入把它们在**数据通路层面**分开，再用门控融合，能缓解「颜色捷径压制图案」。

完整文献与方案矩阵见 [`foray_docs/dual_stream_disentanglement.md`](../../foray_docs/dual_stream_disentanglement.md)。

## 当前状态

骨架已就绪，**尚未接入数据集**：

- `src/ctts/models/color.py` —— YUV 4:2:0 通路（numpy，训练/推理共享同一组系数）
- `src/ctts/models/dualstream.py` —— 非对称双主干 + 门控融合 + 属性双头（torch）
- `tests/test_color.py`、`tests/test_dualstream.py` —— 无 GPU 单测（后者在无 torch 环境自动跳过）

## 运行

```bash
# 结构 / 梯度烟雾测试（无需数据与 GPU）
python3 experiments/dualstream/train_dualstream.py --dry-run

# 单测
PYTHONPATH=src python3 -m pytest tests/test_color.py tests/test_dualstream.py -v
```

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

## 待定项

- **属性标签格式**：`data.yaml` 里如何表达「颜色 4 类 + 图案 9 类」两套标签（同图多标签 vs 组合类展开），需与数据准备脚本一起定；
- **域一致性**：训练时若用 4:2:0 预处理，推理（`foray_vision`）必须一致，否则线上分布偏移；
- **反事实增强**：R↔B 通道交换（不是 HSV 旋转）用于生成「颜色对调」样本，已实现 `color.swap_rb`；
- **导出契约**：ONNX 双输入（`y`, `cbcr`）→ TensorRT 多 binding，需写入 `model_contract.yaml`。
