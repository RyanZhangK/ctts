# AGENTS.md — AI 协作规范

本仓遵循组织 [贡献指南](https://github.com/ZQU-Foray/.github/blob/main/CONTRIBUTING.md)，
以及 [`foray_docs`](https://github.com/ZQU-Foray/foray_docs) 的算法结构与仓库结构约定。

## 需求来源

**`README.md` 是本仓唯一的需求来源。** 不得实现 `README.md` 中不存在的需求。

## 必读

| 文档 | 内容 |
|---|---|
| `README.md` | 本仓定位、工作流、契约、上下游 |
| `tree.md` | 目录结构与各目录职责 |
| `plan.md` | 当前计划与进度 |
| `decision.md` | 关键工程决策（含破坏性接口变更留痕） |
| `foray_docs/algorithm_structure.md` | 全局算法结构 |

## 必须

- **Conventional Commits**——scope 见 `foray_docs/repository_structure.md` §6.6
- 按逻辑单元**小步提交**，不得一次性提交全部代码
- 维护 `plan.md` 最新状态，任意时刻可中断恢复
- 每完成一个逻辑单元，同步更新 `tree.md` 与 `decision.md`
- 公共函数（被 CLI/GUI/其他模块调用）必须有 docstring
- **禁止硬编码个人机器路径**：一律走环境变量（`CTTS_*`）或 CLI 参数，默认值基于仓根
- 4 空格缩进 · LF · UTF-8 · 文件末尾一个空行

## 禁止

- 修改与任务无关的文件
- 删除用户提供的源材料（数据集定义、设计文档）
- 破坏性 Git 操作（`reset --hard` / `push --force` / 非必要 `rebase`）
- 提交权重（`*.pt` / `*.onnx` / `*.engine`）、数据集图片、`runs/` 产物
- 引入无必要依赖；训练/推理所属环节不得越界到 `foray_vision`

## Git Workflow

```
main          ← 稳定版本，经过测试与 Review
  └─ dev      ← 开发分支，所有更改在此分支调试
       └─ feat/xxx · fix/xxx · docs/xxx
```

- **所有 PR 的目标分支是 `dev`，不是 `main`**
- 功能分支从 `dev` 切出，用 `git rebase dev` 同步上游
