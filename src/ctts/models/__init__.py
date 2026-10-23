"""模型与数据通路。

``color`` 为纯 numpy 实现（训练/推理共用的 YUV 4:2:0 通路）；
``dualstream`` 依赖 PyTorch，按需导入，避免未装 torch 的环境在此处失败。
"""

from . import color

__all__ = ["color"]
