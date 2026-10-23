# 开发计划

## 任务列表

- [x] 从个人仓 `RyanZhangK/ctts` 迁入组织仓，保留历史（21 提交）
- [x] 目录重构：`tools/` 平铺 → `src/ctts/` 包 + `tests/` + `scripts/`；pyproject 归位仓根
- [x] 修复包内导入（相对导入）、入口（`ctts` console script）、路径锚点（仓根相对）
- [x] 移除硬编码个人机器路径（`CTTS_WEIGHTS` / `CTTS_DATASET`）
- [x] 组织规范五件套 + CODEOWNERS + `.foray-layer` + CI
- [ ] CI 首次绿灯验证（ruff/pytest，Python 3.11/3.12）
- [ ] 补齐单元测试：`dataset_build` / `split_dataset` / `label_studio_*` / `yolo_train`
- [ ] 双流特征解耦模型实验（见 `foray_docs/dual_stream_disentanglement.md`），结论回写该文档
  - [x] 骨架：`models/color.py`（YUV 4:2:0）+ `models/dualstream.py`（双主干/门控/属性双头）+ 单测
  - [ ] 属性标签格式定稿（颜色 4 类 + 图案 9 类）
  - [ ] E0–E5 消融实验与判据记录
- [ ] 模型契约 `model_contract.yaml` 定稿（输入 layout / 类别映射 / opset）
- [ ] 在 `foray_ws` 的 `.repos` 中 pin 本仓（`ml/` 目录，不进 colcon 构建）

## 当前 Agent State

DUALSTREAM_SKELETON

## Before Snapshot

commit hash:   见 `git log -1`
branch:        main
modified files: 迁移提交见 `git log`
risk level:    L2（结构迁移，已验证语法与导入解析；运行时未在 GPU 环境回归）

## 模糊点与待确认项

- 双流实验是否成立尚未验证；若失败，替代方案为两阶段 ROI（`foray_vision` + 分类器）
- `detect.py` 依赖 PySide6，CI 不安装该依赖，GUI 路径无自动化测试
- `foray_ws` 接入方式（`ml/` 目录）需与架构组确认后再改 `.repos`

## Acceptance Criteria

- [x] `python -m compileall` 通过；包内相对导入全部可解析
- [ ] CI 绿灯（ruff + pytest）
- [ ] 迁移后 `ctts --help` / `ctts data build` 行为与迁移前一致（GPU 环境回归）
